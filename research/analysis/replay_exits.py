"""
Exit-rule replay harness — tick-resolution simulation on path-file CSVs.

Simulates two named exit-rule specs on each path file and compares outcomes.

Spec A (default: "v7") mirrors the live social_alert settings in memecoin/config.py:
  hard_stop=-35%, trail_tiers=[{+30%/−25%}, {+100%/−25%}, {+300%/−15%}],
  profit_lock at +40–100% if peak stalled 60s, time_stop=90min if gain<30%.

Spec B (default: "alt1") is an alternative to test:
  Same hard stop. Tighter trail at +30% (−20% vs −25%).
  Shorter time_stop=45min. Profit_lock stall=90s (vs 60s).

Execution lag (--exec-lag-ms, default 500): after a trigger tick, fill is
simulated at the price lag_ms later (nearest tick). Mimics real sell latency.

TP ladder: optional, configurable per spec. Each TP partially exits the
position; final exit price is the SOL-weighted average across all exits.

Output: per-spec table (n, win-rate, median PnL, p25/75/90, exit-reason mix)
       and a side-by-side comparison summary.

Run:
    python -m research.analysis.replay_exits
    python -m research.analysis.replay_exits --exec-lag-ms 800 --live-only
    python -m research.analysis.replay_exits --spec-b-json '{"hard_stop":-0.30,"trail_tiers":[...]}'
"""

import argparse
import csv
import gzip
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, quantiles

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Default exit specs ─────────────────────────────────────────────────────────

_V7_SPEC = {
    "name":             "v7_social_alert",
    "hard_stop":        -0.35,
    "trail_tiers":      [
        {"activates_at": 0.30, "trail_pct": 0.25},
        {"activates_at": 1.00, "trail_pct": 0.25},
        {"activates_at": 3.00, "trail_pct": 0.15},
    ],
    "tp_levels":        [],          # v7 has no TP ladder for social_alert
    "time_stop_min":    90,
    "time_stop_min_gain": 0.30,      # don't fire time stop if gain > 30%
    "profit_lock_min_gain":  0.40,
    "profit_lock_max_gain":  1.00,
    "profit_lock_stall_sec": 60,
}

_ALT1_SPEC = {
    "name":             "alt1_tighter_trail",
    "hard_stop":        -0.35,       # same hard stop
    "trail_tiers":      [
        {"activates_at": 0.30, "trail_pct": 0.20},  # tighter: −20% vs −25%
        {"activates_at": 1.00, "trail_pct": 0.20},
        {"activates_at": 3.00, "trail_pct": 0.15},
    ],
    "tp_levels":        [],
    "time_stop_min":    45,          # shorter time stop
    "time_stop_min_gain": 0.30,
    "profit_lock_min_gain":  0.40,
    "profit_lock_max_gain":  1.00,
    "profit_lock_stall_sec": 90,     # slower profit_lock (give more room)
}


# ── Path file loader (shared with path_stats) ─────────────────────────────────

def _load_path(p: Path) -> list[dict]:
    rows = []
    try:
        opener = gzip.open(p, "rt", encoding="utf-8") if p.suffix == ".gz" \
                 else open(p, "r", encoding="utf-8", newline="")
        with opener as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rows.append({
                        "ts_ms":      int(row["ts_ms"]),
                        "price_usd":  float(row["price_usd"]),
                        "side":       row.get("side", ""),
                        "sol_amount": float(row.get("sol_amount") or 0),
                        "vsol":       float(row.get("vsol") or 0),
                    })
                except (ValueError, KeyError):
                    pass
    except Exception as e:
        log.debug("Failed to load %s: %s", p.name, e)
        return []
    rows.sort(key=lambda r: r["ts_ms"])
    return rows


def _discover_paths(research_paths_dir: Path, live_only: bool) -> list[Path]:
    paths = []
    if not research_paths_dir.exists():
        return paths
    for p in research_paths_dir.rglob("*.csv"):
        if live_only and "backfill" in str(p):
            continue
        paths.append(p)
    for p in research_paths_dir.rglob("*.csv.gz"):
        if live_only and "backfill" in str(p):
            continue
        paths.append(p)
    return paths


# ── Replay engine ──────────────────────────────────────────────────────────────

@dataclass
class _Position:
    entry_price:   float
    entry_ts_ms:   int
    remaining:     float = 1.0          # fraction of position still open
    peak_price:    float = 0.0
    peak_ts_ms:    int   = 0
    trail_stop:    float = 0.0          # absolute price level (0 = not armed)
    profit_locked: bool  = False
    tp_idx:        int   = 0            # which TP level we're up to
    exits:         list  = field(default_factory=list)
    # each exit: (price, fraction_of_entry, reason)


def _find_price_at_lag(rows: list[dict], from_ts_ms: int, lag_ms: int) -> float:
    """Return price of nearest tick at or after from_ts_ms + lag_ms."""
    target = from_ts_ms + lag_ms
    after  = [r for r in rows if r["ts_ms"] >= target]
    if after:
        return after[0]["price_usd"]
    # Past end of file — use last price
    return rows[-1]["price_usd"]


def _effective_trail_pct(spec: dict, gain: float) -> float | None:
    """Return the currently active trailing stop width (as a negative fraction)."""
    active = None
    for tier in spec["trail_tiers"]:
        if gain >= tier["activates_at"]:
            active = -abs(tier["trail_pct"])
    return active


def _replay_one(rows: list[dict], spec: dict, exec_lag_ms: int) -> dict | None:
    """
    Simulate spec on one path. Returns:
    {exit_price, exit_reason, pnl_pct, hold_time_s, partial_exits}
    Returns None if path is too short.
    """
    if len(rows) < 2:
        return None

    entry_price  = rows[0]["price_usd"]
    entry_ts_ms  = rows[0]["ts_ms"]
    if entry_price <= 0:
        return None

    pos = _Position(
        entry_price = entry_price,
        entry_ts_ms = entry_ts_ms,
        peak_price  = entry_price,
        peak_ts_ms  = entry_ts_ms,
    )

    # Helpers
    hard_stop_price  = entry_price * (1 + spec["hard_stop"])
    time_stop_ms     = spec["time_stop_min"] * 60 * 1000
    time_stop_floor  = spec.get("time_stop_min_gain", 0.30)
    pl_min           = spec.get("profit_lock_min_gain", 0.40)
    pl_max           = spec.get("profit_lock_max_gain", 1.00)
    pl_stall_ms      = spec.get("profit_lock_stall_sec", 60) * 1000
    tp_levels        = spec.get("tp_levels", [])

    def _exit(price: float, reason: str, fraction: float = None) -> dict:
        """Build result dict from a full or final-partial exit."""
        frac = fraction if fraction is not None else pos.remaining
        pos.exits.append((price, frac, reason))
        pos.remaining -= frac

        # Compute weighted average exit price across all partial exits
        weighted = sum(p * f for p, f, _ in pos.exits)
        total_f  = sum(f for _, f, _ in pos.exits)
        avg_exit = weighted / total_f if total_f > 0 else price

        pnl_pct     = (avg_exit / entry_price - 1) * 100
        hold_time_s = (rows[-1]["ts_ms"] - entry_ts_ms) / 1000
        return {
            "exit_price":    avg_exit,
            "exit_reason":   reason,
            "pnl_pct":       round(pnl_pct, 2),
            "hold_time_s":   round(hold_time_s, 1),
            "partial_exits": len(pos.exits),
        }

    for tick in rows[1:]:
        price  = tick["price_usd"]
        now_ms = tick["ts_ms"]
        if price <= 0:
            continue

        gain = price / entry_price - 1

        # Update peak
        if price > pos.peak_price:
            pos.peak_price = price
            pos.peak_ts_ms = now_ms

        peak_gain = pos.peak_price / entry_price - 1

        # ── TP ladder (partial exits) ─────────────────────────────────────────
        for tp_idx in range(pos.tp_idx, len(tp_levels)):
            tp_gain, tp_fraction = tp_levels[tp_idx]
            if gain >= tp_gain and pos.remaining > 0:
                fill = _find_price_at_lag(rows, now_ms, exec_lag_ms)
                frac = min(tp_fraction, pos.remaining)
                pos.exits.append((fill, frac, f"tp_{tp_idx}"))
                pos.remaining -= frac
                pos.tp_idx = tp_idx + 1
                if pos.remaining <= 0.01:
                    return _exit(fill, f"tp_{tp_idx}_final")

        # ── Hard stop ─────────────────────────────────────────────────────────
        if price <= hard_stop_price:
            fill = _find_price_at_lag(rows, now_ms, exec_lag_ms)
            return _exit(fill, "hard_stop")

        # ── Trailing stop (armed when peak_gain crosses tier) ─────────────────
        trail_pct = _effective_trail_pct(spec, peak_gain)
        if trail_pct is not None:
            trail_price = pos.peak_price * (1 + trail_pct)
            if price <= trail_price:
                fill = _find_price_at_lag(rows, now_ms, exec_lag_ms)
                return _exit(fill, "trail_stop")
            # Keep highest possible trail level
            pos.trail_stop = max(pos.trail_stop, trail_price)

        # ── Profit lock (stall detector) ──────────────────────────────────────
        if (not pos.profit_locked
                and pl_min <= peak_gain <= pl_max
                and (now_ms - pos.peak_ts_ms) >= pl_stall_ms):
            fill = _find_price_at_lag(rows, now_ms, exec_lag_ms)
            return _exit(fill, "profit_lock")

        # ── Time stop ─────────────────────────────────────────────────────────
        elapsed_ms = now_ms - entry_ts_ms
        if elapsed_ms >= time_stop_ms and gain < time_stop_floor:
            fill = _find_price_at_lag(rows, now_ms, exec_lag_ms)
            return _exit(fill, "time_stop")

    # Path ended without exit trigger → exit at last price
    last_price = rows[-1]["price_usd"]
    fill = last_price  # no lag available at end of file
    return _exit(fill, "path_end")


# ── Statistics ─────────────────────────────────────────────────────────────────

def _summarise(results: list[dict], spec_name: str) -> dict:
    pnls         = [r["pnl_pct"] for r in results]
    wins         = [p for p in pnls if p > 0]
    reasons      = Counter(r["exit_reason"] for r in results)
    n            = len(pnls)
    win_rate     = len(wins) / n * 100 if n > 0 else 0
    qs           = quantiles(pnls, n=100) if len(pnls) >= 10 else [0] * 100

    return {
        "spec":     spec_name,
        "n":        n,
        "win_rate": win_rate,
        "mean_pnl": mean(pnls) if pnls else 0,
        "med_pnl":  median(pnls) if pnls else 0,
        "p25":      qs[24],
        "p75":      qs[74],
        "p90":      qs[89],
        "reasons":  dict(reasons),
    }


def _print_summary(s: dict):
    print(f"\n  Spec: {s['spec']}")
    print(f"  n={s['n']}  win_rate={s['win_rate']:.1f}%  "
          f"mean_pnl={s['mean_pnl']:+.1f}%  median={s['med_pnl']:+.1f}%  "
          f"p25={s['p25']:+.1f}%  p75={s['p75']:+.1f}%  p90={s['p90']:+.1f}%")
    top_reasons = sorted(s["reasons"].items(), key=lambda x: -x[1])
    reason_str  = "  ".join(f"{r}:{cnt}" for r, cnt in top_reasons[:6])
    print(f"  Exit reasons: {reason_str}")


def _compare(sa: dict, sb: dict):
    print(f"\n  {'Metric':<22}  {'Spec A':>12}  {'Spec B':>12}  {'Delta (B−A)':>14}")
    print(f"  {'-'*64}")
    for label, ka, kb in [
        ("win_rate (%)",   "win_rate", "win_rate"),
        ("mean_pnl (%)",   "mean_pnl", "mean_pnl"),
        ("median_pnl (%)", "med_pnl",  "med_pnl"),
        ("p25 (%)",        "p25",      "p25"),
        ("p75 (%)",        "p75",      "p75"),
        ("p90 (%)",        "p90",      "p90"),
    ]:
        va, vb = sa[ka], sb[kb]
        delta  = vb - va
        sign   = "+" if delta >= 0 else ""
        print(f"  {label:<22}  {va:>12.1f}  {vb:>12.1f}  {sign}{delta:>13.1f}")
    print(f"\n  Winner: {'Spec B' if sb['med_pnl'] > sa['med_pnl'] else 'Spec A'} "
          f"(by median PnL)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exit-rule tick-resolution replay")
    parser.add_argument("--exec-lag-ms",  type=int, default=500,
                        help="simulated execution lag in ms (default: 500)")
    parser.add_argument("--live-only",    action="store_true",
                        help="exclude backfill paths")
    parser.add_argument("--spec-b-json",  type=str, default=None,
                        help="JSON string overriding Spec B fields")
    parser.add_argument("--max-paths",    type=int, default=0,
                        help="cap number of paths (0=all, for quick checks)")
    args = parser.parse_args()

    from research.config import RESEARCH_PATHS_DIR

    spec_a = dict(_V7_SPEC)
    spec_b = dict(_ALT1_SPEC)

    if args.spec_b_json:
        try:
            overrides = json.loads(args.spec_b_json)
            spec_b.update(overrides)
            log.info("Spec B overridden: %s", overrides)
        except json.JSONDecodeError as e:
            log.error("Invalid --spec-b-json: %s", e)
            sys.exit(1)

    path_files = _discover_paths(RESEARCH_PATHS_DIR, live_only=args.live_only)
    if not path_files:
        print("No path files found. Run PeakTracker or backfill_paths.py first.")
        sys.exit(0)

    if args.max_paths and len(path_files) > args.max_paths:
        path_files = path_files[:args.max_paths]
        log.info("Capped to %d paths", args.max_paths)

    log.info("Replaying %d paths  exec_lag=%dms", len(path_files), args.exec_lag_ms)

    results_a: list[dict] = []
    results_b: list[dict] = []
    skipped = 0

    for i, p in enumerate(path_files, 1):
        rows = _load_path(p)
        if not rows:
            skipped += 1
            continue
        res_a = _replay_one(rows, spec_a, args.exec_lag_ms)
        res_b = _replay_one(rows, spec_b, args.exec_lag_ms)
        if res_a:
            results_a.append(res_a)
        if res_b:
            results_b.append(res_b)

        if i % 50 == 0:
            log.info("  %d/%d paths processed", i, len(path_files))

    log.info("Done. Spec A: %d results, Spec B: %d results, skipped: %d",
             len(results_a), len(results_b), skipped)

    if not results_a:
        print("No results — all paths too short or empty.")
        sys.exit(0)

    sa = _summarise(results_a, spec_a["name"])
    sb = _summarise(results_b, spec_b["name"])

    print(f"\n{'=' * 72}")
    print(f"  EXIT REPLAY — {len(results_a)} paths  exec_lag={args.exec_lag_ms}ms")
    print(f"{'=' * 72}")

    print(f"\n── Spec A ──")
    _print_summary(sa)
    print(f"\n── Spec B ──")
    _print_summary(sb)

    print(f"\n{'─' * 72}")
    print(f"  Side-by-side comparison")
    _compare(sa, sb)

    print(f"\n{'=' * 72}")
    print(f"  END REPLAY")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
