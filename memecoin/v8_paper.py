"""
v8_paper.py — V8 paper-trading twin (N6, post-measurement batch, 2026-07-30).

Runs the SAME signals the v7 paper/live pipeline sees through a separate
V8 candidate rule, journaling to logs/memecoin_v8_journal.csv. Fully
independent from v7: separate Position dict, separate journal, separate
monitor thread, its own price polling. A bug here can never touch v7 paper
or live trading — every public entry point is wrapped so it fails silent.

V8 candidate rule (entry gate)
-------------------------------
  progress_at_signal < 0.70   AND   no-dex (token has no dex_id yet, i.e.
  still pre-graduation / not indexed by a DEX aggregator).

  progress_at_signal is computed live from PumpPortal's cached vsol for the
  token (pumpportal_monitor.get_screening_state) — zero new Helius calls,
  reuses data the scanner already collects for the research snapshot.

  SCOPE LIMITATION: the "smart-money-v1" half of the spec's gate
  ("progress<70 + smart-money-v1/no-dex gate") is NOT applied live here.
  Live smart-money classification requires research/smart_wallets.py's
  fetch_first_buyers(), which is a dedicated Helius call — adding it to the
  live signal path would increase Helius RPC volume under SOCIAL_ALERT_ONLY,
  which CLAUDE.md explicitly rules out. smart_money_hit is joined in later
  for REPORTING only, from research_tokens (offline pipeline), not as an
  entry gate. This is a real, deliberate scope cut, not an oversight.

Exit config
-----------
  V8_EXIT_CONFIG below is a PLACEHOLDER — the spec calls for "exit config =
  the replay winner", but replay_exits.py has no results yet (blocked on the
  PC2 backfill per docs/V8_INPUTS.md N4d). Until a real winner is chosen,
  this mirrors the current v7 social_alert production config so the twin is
  at least a like-for-like comparison. Swap V8_EXIT_CONFIG once replay_exits
  produces a winner — nothing else in this file needs to change.
"""

import json
import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path

from memecoin.config import GRAD_SOL_UI as _GRAD_SOL   # PROGRESS-FIX PF9: canonical, was a locally hardcoded 115.0 manually kept "in sync" with path_stats.py's own copy

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

V8_CONFIG_TAG      = "v8_candidate_2026-07-30"
V8_PROGRESS_MAX    = 0.70          # gate: progress_at_signal must be below this
_MONITOR_INTERVAL_S = 5.0

# PLACEHOLDER — see module docstring. Mirrors v7 social_alert prod config
# (memecoin/config.py SIGNAL_SETTINGS["social_alert"]) at the time this was
# written. Not re-read live from config.py so a v7 config change doesn't
# silently change the v8 comparison mid-flight.
V8_EXIT_CONFIG = {
    "hard_stop_pct":  -0.35,
    "time_stop_minutes": 90,
    "trail_tiers": [
        {"activates_at": 0.30, "trail_pct": 0.25},
        {"activates_at": 1.00, "trail_pct": 0.25},
        {"activates_at": 3.00, "trail_pct": 0.15},
    ],
}

V8_JOURNAL_FIELDS = [
    "id", "signal_id", "chain", "token_address", "token_symbol",
    "signal_type", "strength",
    "signal_price", "signal_time", "entry_price", "entry_time", "size_usd",
    "exit_price", "exit_time", "exit_reason", "pnl_usd", "pnl_pct",
    "peak_price", "hard_stop_pct",
    "progress_at_signal", "dex_id", "entry_source",
    "notes", "config_tag",
]


def _paths():
    """Resolve output paths lazily so this module never needs memecoin.config
    imported at module scope (keeps failure modes local to call sites)."""
    from memecoin.config import LOGS_DIR, DATA_DIR
    return (
        LOGS_DIR / "memecoin_v8_journal.csv",
        DATA_DIR / "memecoin_v8_positions.json",
    )


# ---------------------------------------------------------------------------
# Entry gate
# ---------------------------------------------------------------------------

# PROGRESS-FIX PF6: how long passes_v8_gate will wait for an in-flight
# capture before failing closed. By the time this runs (after screen_token's
# ~1-2s), capture usually already has a head start from _t0 — this is just a
# short final grace period, not the primary wait mechanism (that's PF3's
# async capture, started at alert time). Paper-only book, not live capital.
_GATE_CAPTURE_WAIT_S = 0.5


def compute_progress_at_signal(chain: str, token_address: str, event_id: str = "") -> float | None:
    """
    PROGRESS-FIX PF6: no longer an independent measurement — reads the SAME
    canonical ProgressCapture result (memecoin/progress_capture.py) that
    research/tracker.py consumes for this event_id, so Research and V8
    cannot disagree about the same signal (the old version called
    get_screening_state() live, on its own timing, independently of what
    research had already captured/stored).

    Returns None (fail-closed) on any failure — including the old race
    where a freshly-created ScreeningState with vsol=0 falsely looked like
    "no progress data" via a different code path; now that's just one of
    several explicit progress_status failure reasons upstream.
    """
    if chain != "solana":
        return None
    if not event_id:
        # No event_id means this signal didn't originate from
        # capture_progress_async's alert-time hookup (e.g. a non-Telegram
        # signal path) — nothing to look up.
        return None
    try:
        from memecoin.progress_capture import wait_for_capture
        cap = wait_for_capture(event_id, timeout_s=_GATE_CAPTURE_WAIT_S)
        if cap is None or cap.progress_status != "ok" or cap.progress_at_signal is None:
            return None
        return cap.progress_at_signal
    except Exception as e:
        log.debug("v8_paper: progress_at_signal lookup failed for %s (event %s): %s",
                  token_address[:8] if token_address else "?", event_id[:12], e)
        return None


def passes_v8_gate(signal) -> tuple[bool, str, float | None]:
    """Returns (passed, reason, progress_at_signal)."""
    progress = compute_progress_at_signal(
        signal.chain, signal.token_address, getattr(signal, "event_id", ""),
    )
    if progress is None:
        return False, "progress_unknown", None
    if progress >= V8_PROGRESS_MAX:
        return False, f"progress_{progress:.2f}_over_{V8_PROGRESS_MAX:.2f}", progress
    dex_id = (getattr(signal, "dex_id", "") or "").strip()
    if dex_id:
        return False, f"has_dex_id:{dex_id}", progress
    return True, "ok", progress


# ---------------------------------------------------------------------------
# Position (plain dict — deliberately not memecoin.portfolio.Position;
# this book must never share state/serialization with the live engine)
# ---------------------------------------------------------------------------

def _new_position(signal, progress: float) -> dict:
    sig_price_pp = getattr(signal, "_price_pp", 0) or 0
    baseline = sig_price_pp if sig_price_pp > 0 else (signal.price_usd or 0)
    now = time.time()
    import uuid
    return {
        "id": "V8" + str(uuid.uuid4())[:6],
        "signal_id": getattr(signal, "id", ""),
        "chain": signal.chain,
        "token_address": signal.token_address,
        "token_symbol": signal.token_symbol,
        "signal_type": signal.signal_type,
        "strength": signal.strength,
        "signal_price": baseline,
        "signal_time": now,
        "entry_price": baseline,          # curve-baseline entry (N6 requirement)
        "entry_time": now,
        "size_usd": 1.0,                  # paper-only; size is irrelevant to pct outcomes
        "current_price": baseline,
        "peak_price": baseline,
        "status": "open",
        "exit_price": 0.0,
        "exit_time": 0.0,
        "exit_reason": "",
        "progress_at_signal": progress,
        "dex_id": getattr(signal, "dex_id", "") or "",
        "entry_source": "pp_tick" if sig_price_pp > 0 else "dex_stale",
        "notes": "",
    }


def _pnl(pos: dict) -> tuple[float, float]:
    if pos["entry_price"] <= 0:
        return 0.0, 0.0
    ref = pos["exit_price"] if pos["status"] == "closed" else pos["current_price"]
    pnl_pct = (ref - pos["entry_price"]) / pos["entry_price"]
    pnl_usd = pnl_pct * pos["size_usd"]
    return pnl_usd, pnl_pct


def _check_exit(pos: dict, cfg: dict) -> str:
    """Returns exit_reason string, or "" if the position should stay open."""
    _, pnl_pct = _pnl(pos)
    peak_gain = (pos["peak_price"] - pos["entry_price"]) / pos["entry_price"] if pos["entry_price"] > 0 else 0.0

    if pnl_pct <= cfg["hard_stop_pct"]:
        return "hard_stop"

    # ATH-anchored trail tiers: pick the highest tier whose activates_at <= peak_gain
    active_tier = None
    for tier in cfg["trail_tiers"]:
        if peak_gain >= tier["activates_at"]:
            if active_tier is None or tier["activates_at"] > active_tier["activates_at"]:
                active_tier = tier
    if active_tier is not None and pos["peak_price"] > 0:
        trail_stop_price = pos["peak_price"] * (1 - active_tier["trail_pct"])
        if pos["current_price"] <= trail_stop_price:
            return "trailing_stop"

    # Time stop only while peak_gain < 30% (never interrupts a runner mid-leg)
    age_min = (time.time() - pos["entry_time"]) / 60.0
    if age_min >= cfg["time_stop_minutes"] and peak_gain < 0.30:
        return "time_stop"

    return ""


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------

def _append_journal(pos: dict) -> None:
    import csv
    journal_file, _ = _paths()
    journal_file.parent.mkdir(parents=True, exist_ok=True)
    pnl_usd, pnl_pct = _pnl(pos)
    row = {
        "id": pos["id"], "signal_id": pos["signal_id"], "chain": pos["chain"],
        "token_address": pos["token_address"], "token_symbol": pos["token_symbol"],
        "signal_type": pos["signal_type"], "strength": pos["strength"],
        "signal_price": pos["signal_price"],
        "signal_time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(pos["signal_time"])),
        "entry_price": pos["entry_price"],
        "entry_time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(pos["entry_time"])),
        "size_usd": pos["size_usd"],
        "exit_price": pos["exit_price"],
        "exit_time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(pos["exit_time"])) if pos["exit_time"] else "",
        "exit_reason": pos["exit_reason"],
        "pnl_usd": round(pnl_usd, 4),
        "pnl_pct": round(pnl_pct * 100, 2),
        "peak_price": pos["peak_price"],
        "hard_stop_pct": V8_EXIT_CONFIG["hard_stop_pct"],
        "progress_at_signal": pos["progress_at_signal"],
        "dex_id": pos["dex_id"],
        "entry_source": pos["entry_source"],
        "notes": pos["notes"],
        "config_tag": V8_CONFIG_TAG,
    }
    write_header = not journal_file.exists() or journal_file.stat().st_size == 0
    with open(journal_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=V8_JOURNAL_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Book
# ---------------------------------------------------------------------------

class V8PaperBook:
    """Independent paper-trading book. No live execution, no shared state
    with memecoin.portfolio.Portfolio. Safe to import/instantiate even if
    the rest of v8_paper is broken elsewhere (every public method self-guards)."""

    def __init__(self):
        self._positions: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        try:
            _, pos_file = _paths()
            if pos_file.exists():
                data = json.loads(pos_file.read_text())
                self._positions = {d["id"]: d for d in data}
        except Exception as e:
            log.warning("v8_paper: position load failed (starting empty): %s", e)
            self._positions = {}

    def _save(self) -> None:
        try:
            _, pos_file = _paths()
            pos_file.parent.mkdir(parents=True, exist_ok=True)
            data = list(self._positions.values())
            tmp = pos_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2))
            tmp.replace(pos_file)
        except Exception as e:
            log.warning("v8_paper: position save failed: %s", e)

    def open_positions(self) -> list[dict]:
        with self._lock:
            return [p for p in self._positions.values() if p["status"] == "open"]

    def maybe_open(self, signal) -> None:
        """Called once per incoming signal (same funnel as v7's open_position).
        Never raises — any failure here must not affect v7/live."""
        try:
            with self._lock:
                already_open = any(
                    p["token_address"] == signal.token_address and p["status"] == "open"
                    for p in self._positions.values()
                )
            if already_open:
                return
            passed, reason, progress = passes_v8_gate(signal)
            if not passed:
                log.debug("v8_paper: gate reject %s — %s", signal.token_symbol, reason)
                return
            pos = _new_position(signal, progress)
            with self._lock:
                self._positions[pos["id"]] = pos
                self._save()
            log.info("v8_paper OPEN %s  progress=%.2f  entry=$%.10f  id=%s",
                     signal.token_symbol, progress, pos["entry_price"], pos["id"])
        except Exception as e:
            log.warning("v8_paper: maybe_open failed (non-fatal): %s", e)

    def _close(self, pos_id: str, price: float, reason: str) -> None:
        with self._lock:
            pos = self._positions.get(pos_id)
            if pos is None or pos["status"] != "open":
                return
            pos["exit_price"] = price
            pos["exit_time"] = time.time()
            pos["exit_reason"] = reason
            pos["status"] = "closed"
            self._save()
        try:
            _append_journal(pos)
        except Exception as e:
            log.warning("v8_paper: journal write failed for %s (position still closed): %s",
                       pos_id, e)
        log.info("v8_paper CLOSE %s  reason=%s  price=$%.10f  id=%s",
                 pos["token_symbol"], reason, price, pos_id)

    def update_price(self, token_address: str, price: float) -> None:
        """Called by the monitor loop with a fresh price for one token.
        Updates all open v8 positions on that token and closes any that
        trigger an exit under V8_EXIT_CONFIG."""
        if price <= 0:
            return
        try:
            to_close = []
            with self._lock:
                for pos in self._positions.values():
                    if pos["status"] != "open" or pos["token_address"] != token_address:
                        continue
                    pos["current_price"] = price
                    if price > pos["peak_price"]:
                        pos["peak_price"] = price
                    reason = _check_exit(pos, V8_EXIT_CONFIG)
                    if reason:
                        to_close.append((pos["id"], price, reason))
                self._save()
            for pos_id, px, reason in to_close:
                self._close(pos_id, px, reason)
        except Exception as e:
            log.warning("v8_paper: update_price failed for %s (non-fatal): %s",
                       token_address[:8] if token_address else "?", e)

    def _monitor_loop(self) -> None:
        from memecoin.pumpportal_monitor import monitor as _pp_monitor
        while True:
            try:
                time.sleep(_MONITOR_INTERVAL_S)
                open_pos = self.open_positions()
                if not open_pos:
                    continue
                prices = _pp_monitor.get_prices()
                for pos in open_pos:
                    price = prices.get(pos["token_address"], 0)
                    if price > 0:
                        self.update_price(pos["token_address"], price)
            except Exception as e:
                log.warning("v8_paper: monitor loop error (continuing): %s", e)

    def start(self, daemon: bool = True) -> None:
        threading.Thread(target=self._monitor_loop, daemon=daemon,
                         name="v8-paper-monitor").start()
        log.info("v8_paper: monitor thread started (interval=%.0fs, gate=progress<%.0f%%+no-dex)",
                 _MONITOR_INTERVAL_S, V8_PROGRESS_MAX * 100)


# Module-level singleton, mirroring memecoin.portfolio's `portfolio` convention.
book = V8PaperBook()
