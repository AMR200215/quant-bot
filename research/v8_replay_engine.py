"""
research/v8_replay_engine.py — V8-FILTER-DERIVATION Phase 2 (FD14):
reusable replay interface, REFACTORED (not rewritten) out of
research/analysis/replay_exits.py's tick-resolution simulator
(_replay_one / _find_price_at_lag / _effective_trail_pct / _Position).

    replay_strategy(rows, entry_ts, entry_spec, exit_spec, execution_model) -> ReplayResult | None

Preserves every semantic of the original _replay_one exactly: hard stop,
tiered trailing stop (armed at the highest tier whose activates_at the
peak has crossed), TP ladder (partial exits, SOL-weighted average exit
price across all partial fills), profit-lock stall detector, time stop
with a gain floor, and execution-lag fill simulation (nearest tick at or
after decision_ts + lag; falls back to the path's last price past EOF).

research/analysis/replay_exits.py now calls into this module instead of
defining its own simulator inline -- its CLI output is unchanged
(verified in research/tests/test_v8_replay_engine_refactor.py by a
live before/after comparison run on the same path corpus, receipted in
docs/RECEIPTS.md's Phase 2 section).

entry_ts is a REQUIRED, explicit parameter -- this interface never
assumes rows[0] is the entry tick (P2-7 fixes real entry-time alignment
on top of this; this module just refuses to bake the old, wrong
assumption into its own signature). Passing entry_ts=rows[0]["ts_ms"]
reproduces the pre-P2-7 behavior exactly, which is what
replay_exits.py's CLI does today (unchanged until P2-7 lands).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class ReplayResult:
    exit_price: float
    exit_reason: str
    pnl_pct: float
    hold_time_s: float
    partial_exits: int
    entry_price: float
    entry_ts_ms: int


class ExecutionModel(Protocol):
    """Duck-typed execution-cost interface. FD20's real cost model
    (P2-9) implements this same fill_price(rows, decision_ts_ms) method
    with slippage/curve-impact/fee modeling; the replay loop below never
    needs to know which implementation it's holding."""

    def fill_price(self, rows: list[dict], decision_ts_ms: int) -> float:
        ...


class FixedLagExecutionModel:
    """Default/legacy execution model -- exact behavior of the original
    _find_price_at_lag: nearest tick at or after decision_ts + lag_ms,
    or the path's last price if the lag runs past the end of the file."""

    def __init__(self, exec_lag_ms: int = 500):
        self.exec_lag_ms = exec_lag_ms

    def fill_price(self, rows: list[dict], decision_ts_ms: int) -> float:
        target = decision_ts_ms + self.exec_lag_ms
        after = [r for r in rows if r["ts_ms"] >= target]
        if after:
            return after[0]["price_usd"]
        return rows[-1]["price_usd"]


@dataclass
class _Position:
    entry_price: float
    entry_ts_ms: int
    remaining: float = 1.0
    peak_price: float = 0.0
    peak_ts_ms: int = 0
    trail_stop: float = 0.0
    profit_locked: bool = False
    tp_idx: int = 0
    exits: list = field(default_factory=list)


def _effective_trail_pct(exit_spec: dict, gain: float) -> Optional[float]:
    active = None
    for tier in exit_spec["trail_tiers"]:
        if gain >= tier["activates_at"]:
            active = -abs(tier["trail_pct"])
    return active


def replay_strategy(
    rows: list[dict],
    entry_ts: int,
    entry_spec: dict,
    exit_spec: dict,
    execution_model: ExecutionModel,
) -> Optional[ReplayResult]:
    """
    rows: full canonical path rows (already loaded/typed via
        research.path_schema.load_path_file + the caller's own typing
        pass), sorted by ts_ms ascending.
    entry_ts: the entry decision timestamp. Caller resolves this
        (P2-7's entry-alignment rules decide it in real usage); this
        function only requires ts_ms >= entry_ts to be in-position.
    entry_spec: reserved for entry-side parameters an exit rule might
        condition on (e.g. candidate_id) -- not consulted by the exit
        logic below today, kept for interface symmetry with FD14's
        eventual conditional-exit candidates.
    exit_spec: same shape replay_exits.py's spec dicts already use --
        hard_stop, trail_tiers, tp_levels, time_stop_min,
        time_stop_min_gain, profit_lock_min_gain, profit_lock_max_gain,
        profit_lock_stall_sec.
    execution_model: supplies fill_price(rows, decision_ts_ms).

    Returns None if fewer than 2 in-position ticks exist (path too
    short to simulate) or entry_price <= 0.
    """
    tick_rows = [r for r in rows if r["ts_ms"] >= entry_ts]
    if len(tick_rows) < 2:
        return None

    entry_price = tick_rows[0]["price_usd"]
    entry_ts_ms = tick_rows[0]["ts_ms"]
    if entry_price <= 0:
        return None

    pos = _Position(
        entry_price=entry_price,
        entry_ts_ms=entry_ts_ms,
        peak_price=entry_price,
        peak_ts_ms=entry_ts_ms,
    )

    hard_stop_price = entry_price * (1 + exit_spec["hard_stop"])
    time_stop_ms = exit_spec["time_stop_min"] * 60 * 1000
    time_stop_floor = exit_spec.get("time_stop_min_gain", 0.30)
    pl_min = exit_spec.get("profit_lock_min_gain", 0.40)
    pl_max = exit_spec.get("profit_lock_max_gain", 1.00)
    pl_stall_ms = exit_spec.get("profit_lock_stall_sec", 60) * 1000
    tp_levels = exit_spec.get("tp_levels", [])

    def _exit(price: float, reason: str, fraction: Optional[float] = None) -> ReplayResult:
        frac = fraction if fraction is not None else pos.remaining
        pos.exits.append((price, frac, reason))
        pos.remaining -= frac

        weighted = sum(p * f for p, f, _ in pos.exits)
        total_f = sum(f for _, f, _ in pos.exits)
        avg_exit = weighted / total_f if total_f > 0 else price

        pnl_pct = (avg_exit / entry_price - 1) * 100
        hold_time_s = (tick_rows[-1]["ts_ms"] - entry_ts_ms) / 1000
        return ReplayResult(
            exit_price=avg_exit,
            exit_reason=reason,
            pnl_pct=round(pnl_pct, 2),
            hold_time_s=round(hold_time_s, 1),
            partial_exits=len(pos.exits),
            entry_price=entry_price,
            entry_ts_ms=entry_ts_ms,
        )

    for tick in tick_rows[1:]:
        price = tick["price_usd"]
        now_ms = tick["ts_ms"]
        if price <= 0:
            continue

        gain = price / entry_price - 1

        if price > pos.peak_price:
            pos.peak_price = price
            pos.peak_ts_ms = now_ms

        peak_gain = pos.peak_price / entry_price - 1

        # TP ladder (partial exits)
        for tp_idx in range(pos.tp_idx, len(tp_levels)):
            tp_gain, tp_fraction = tp_levels[tp_idx]
            if gain >= tp_gain and pos.remaining > 0:
                fill = execution_model.fill_price(tick_rows, now_ms)
                frac = min(tp_fraction, pos.remaining)
                pos.exits.append((fill, frac, f"tp_{tp_idx}"))
                pos.remaining -= frac
                pos.tp_idx = tp_idx + 1
                if pos.remaining <= 0.01:
                    return _exit(fill, f"tp_{tp_idx}_final")

        # Hard stop
        if price <= hard_stop_price:
            fill = execution_model.fill_price(tick_rows, now_ms)
            return _exit(fill, "hard_stop")

        # Trailing stop
        trail_pct = _effective_trail_pct(exit_spec, peak_gain)
        if trail_pct is not None:
            trail_price = pos.peak_price * (1 + trail_pct)
            if price <= trail_price:
                fill = execution_model.fill_price(tick_rows, now_ms)
                return _exit(fill, "trail_stop")
            pos.trail_stop = max(pos.trail_stop, trail_price)

        # Profit lock (stall detector)
        if (not pos.profit_locked
                and pl_min <= peak_gain <= pl_max
                and (now_ms - pos.peak_ts_ms) >= pl_stall_ms):
            fill = execution_model.fill_price(tick_rows, now_ms)
            return _exit(fill, "profit_lock")

        # Time stop
        elapsed_ms = now_ms - entry_ts_ms
        if elapsed_ms >= time_stop_ms and gain < time_stop_floor:
            fill = execution_model.fill_price(tick_rows, now_ms)
            return _exit(fill, "time_stop")

    last_price = tick_rows[-1]["price_usd"]
    return _exit(last_price, "path_end")
