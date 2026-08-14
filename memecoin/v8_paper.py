"""
v8_paper.py — V8 paper-trading twin (N6, post-measurement batch,
2026-07-30; re-architected V8-REWIRE, 2026-08-14).

V8-REWIRE: forks directly off the raw Telegram alert stream
(memecoin.alert_event.TelegramAlertEvent), evaluated in
maybe_open_from_alert() from memecoin.scanner._on_telegram_signal()
BEFORE V7's screen_token() ever runs -- not "the same signals the v7
pipeline sees" (that was the pre-rewire architecture: V8 only ever saw
whatever survived V7's filters + V7's own dedup, making "V8 vs v7" a
comparison against a self-selected subset, not the real population). See
docs/RECEIPTS.md's V8-REWIRE section for the full investigation that led
here.

Journaling to logs/memecoin_v8_journal.csv. Fully independent from v7:
separate Position dict, separate journal, separate monitor thread, its
own price polling, its own dedup (v8_transport_duplicate, distinct from
V7's _is_duplicate), its own entry-price provenance (PumpPortal screening
state, resolved independently -- never signal._price_pp, which is a V7
Signal attribute this module must never touch). A bug here can never
touch v7 paper or live trading — every public entry point is wrapped so
it fails silent, and maybe_open_from_alert() dispatches onto its own
thread so a slow price wait can never add latency to V7's live-buy
critical path.

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

# V8-REWIRE VR12/VR13: era tag written to every journal row. Anything
# opened before this module's V8-REWIRE code actually ran in production
# is PRE_REWIRE_V7_CONDITIONAL (V8 only ever saw V7-screened survivors);
# anything after is V8_TELEGRAM_INDEPENDENT_V1 (V8 forks off the raw
# alert). Any forward comparison / freeze-gate math MUST filter to the
# new era only -- mixing eras silently understates V8's real opportunity
# set with pre-rewire rows.
#
# The cutover timestamp is self-bootstrapping rather than a hand-set
# constant: the first time this process ever runs the new code, it
# stamps logs/watchdog/v8_rewire_deploy_ts.txt with the real wall-clock
# time and every subsequent process (this one and future restarts) reads
# that same stamp back. A hardcoded constant would require a human to
# guess or manually update it in a follow-up commit after watching the
# real deploy happen -- fragile and easy to get subtly wrong (see this
# session's own V8_INPUTS.md / v8_funnel.jsonl data-loss incidents,
# VR22 -- another case where a manual step silently drifted from
# reality). Fails closed: if the stamp can't be read or written for any
# reason, _current_era() returns PRE_REWIRE (conservative -- never
# over-counts the new era).
V8_ERA_PRE_REWIRE = "PRE_REWIRE_V7_CONDITIONAL"
V8_ERA_INDEPENDENT = "V8_TELEGRAM_INDEPENDENT_V1"
_DEPLOY_TS_LOCK = threading.Lock()


def _deploy_stamp_path() -> "Path":
    from memecoin.config import LOGS_DIR
    return LOGS_DIR / "watchdog" / "v8_rewire_deploy_ts.txt"


def _independent_validation_start() -> float:
    """Returns the frozen V8-REWIRE cutover epoch ts, creating it on first
    call if it doesn't exist yet. Cached in-process after the first read."""
    cached = getattr(_independent_validation_start, "_cached", None)
    if cached is not None:
        return cached
    try:
        stamp_path = _deploy_stamp_path()
        with _DEPLOY_TS_LOCK:
            if stamp_path.exists():
                ts = float(stamp_path.read_text().strip())
            else:
                ts = time.time()
                stamp_path.parent.mkdir(parents=True, exist_ok=True)
                stamp_path.write_text(str(ts))
        _independent_validation_start._cached = ts
        return ts
    except Exception as e:
        log.warning("v8_paper: could not read/create V8-REWIRE deploy stamp "
                    "(era tagging fails closed to PRE_REWIRE): %s", e)
        return float("inf")   # nothing is ever >= this -> always PRE_REWIRE

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
    "notes", "config_tag", "era",
]


def _current_era() -> str:
    return V8_ERA_INDEPENDENT if time.time() >= _independent_validation_start() else V8_ERA_PRE_REWIRE


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


def _get_capture_for_gate(chain: str, token_address: str, event_id: str = ""):
    """
    PROGRESS-FIX PF6: reads the SAME canonical ProgressCapture result
    (memecoin/progress_capture.py) that research/tracker.py consumes for
    this event_id, so Research and V8 cannot disagree about the same
    signal.

    V8-TWIN-FIX VF2: returns the full ProgressCapture (not just a float)
    so the gate can check venue_state_at_signal, not just progress —
    dex_id was never a reliable graduation signal (see RECEIPTS.md
    V8-TWIN-FIX section; DexScreener indexes pump.fun bonding-curve
    tokens as dex_id="pumpfun" long before graduation, so the old
    `if dex_id: reject` check rejected every single indexed candidate
    regardless of actual on-curve state).

    Returns None (fail-closed) on any failure.
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
        return cap
    except Exception as e:
        log.debug("v8_paper: capture lookup failed for %s (event %s): %s",
                  token_address[:8] if token_address else "?", event_id[:12], e)
        return None


def compute_progress_at_signal(chain: str, token_address: str, event_id: str = "") -> float | None:
    """Convenience wrapper kept for callers that only need the progress
    value (see _get_capture_for_gate for the full-capture version used by
    the gate itself, which also needs venue_state)."""
    cap = _get_capture_for_gate(chain, token_address, event_id)
    return cap.progress_at_signal if cap is not None else None


def passes_v8_gate(signal) -> tuple[bool, str, float | None]:
    """Returns (passed, reason, progress_at_signal).

    V8-TWIN-FIX VF2: gate is progress_at_signal < V8_PROGRESS_MAX AND
    venue_state_at_signal == CURVE_ACTIVE. Never dex_id — see
    _get_capture_for_gate's docstring for why that was wrong. UNKNOWN
    venue state fails closed (rejected), same as unknown progress.

    V8-REWIRE VR3/VR4: pure with respect to strategy identity -- only
    ever reads .chain/.token_address/.event_id off `signal`, so it works
    identically whether called with a V7 Signal (legacy tests) or a
    memecoin.alert_event.TelegramAlertEvent (the real V8-REWIRE call
    path). This is deliberately duck-typed rather than given two
    signatures: the point is that the candidate rule itself never changed
    and never needed to know which object shape it was fed.
    """
    cap = _get_capture_for_gate(
        signal.chain, signal.token_address, getattr(signal, "event_id", ""),
    )
    if cap is None:
        return False, "progress_unknown", None
    progress = cap.progress_at_signal
    if progress >= V8_PROGRESS_MAX:
        return False, f"progress_{progress:.2f}_over_{V8_PROGRESS_MAX:.2f}", progress
    venue = cap.venue_state_at_signal
    if venue != "CURVE_ACTIVE":
        return False, f"venue_state:{venue}", progress
    return True, "ok", progress


# ---------------------------------------------------------------------------
# V8-native dedup (V8-REWIRE VR5/VR6) -- deliberately NOT memecoin.scanner's
# _is_duplicate(): that function reads/mutates V7 strategy state (_seen
# cooldown dict, portfolio.open_positions(), _traded_today blacklist) and
# gating V8 on it is exactly the "V8 conditional on V7" bug this rewire
# fixes. V8's own dedup only ever needs to answer one narrow question:
# "did I already evaluate this exact alert event?" -- a transport-level
# concern (guards against a literal double-invocation of
# maybe_open_from_alert for the same event_id), not a strategy decision.
# Same-token-different-event_id re-alerts are handled separately by the
# already_open check inside maybe_open_from_alert, which is V8's own
# position book, not shared with V7's.
# ---------------------------------------------------------------------------

_SEEN_EVENT_IDS_MAX = 5000
_seen_event_ids: "dict[str, float]" = {}
_seen_lock = threading.Lock()


def _is_transport_duplicate(event_id: str) -> bool:
    if not event_id:
        return False
    with _seen_lock:
        if event_id in _seen_event_ids:
            return True
        _seen_event_ids[event_id] = time.time()
        if len(_seen_event_ids) > _SEEN_EVENT_IDS_MAX:
            # Drop the oldest ~10% -- bounded memory, no external dependency.
            oldest = sorted(_seen_event_ids.items(), key=lambda kv: kv[1])
            for k, _ in oldest[: _SEEN_EVENT_IDS_MAX // 10]:
                _seen_event_ids.pop(k, None)
        return False


# ---------------------------------------------------------------------------
# Entry price provenance (V8-REWIRE VR8) -- V8 forks before screen_token()
# runs, so it can never read signal._price_pp / signal.price_usd (those are
# populated by V7's own DexScreener/PP-wait code, well after the fork
# point). V8 gets its own independent PumpPortal read instead -- the same
# subscription memecoin.scanner._on_telegram_signal() already fires
# unconditionally (before any V7 branching) via
# pumpportal_monitor.subscribe_screening(), so ticks are already in flight
# by the time this runs on V8's own thread.
# ---------------------------------------------------------------------------

_PRICE_WAIT_S = 2.0
_PRICE_POLL_INTERVAL_S = 0.2


def _resolve_entry_price(chain: str, token_address: str) -> tuple[float, str]:
    """Returns (price, entry_source). price==0.0 means unpriced within
    budget -- caller must treat that as an explicit terminal state
    (v8_pass_unpriced), not silently substitute a stale/foreign price."""
    if chain != "solana":
        return 0.0, "unsupported_chain"
    try:
        from memecoin.pumpportal_monitor import monitor as _pp_monitor
    except Exception:
        return 0.0, "pp_import_failed"
    deadline = time.time() + _PRICE_WAIT_S
    while time.time() < deadline:
        try:
            price = _pp_monitor.get_prices().get(token_address, 0.0)
        except Exception:
            return 0.0, "pp_read_failed"
        if price > 0:
            return price, "pp_tick_v8_fork"
        time.sleep(_PRICE_POLL_INTERVAL_S)
    return 0.0, "pp_unpriced"


# ---------------------------------------------------------------------------
# Position (plain dict — deliberately not memecoin.portfolio.Position;
# this book must never share state/serialization with the live engine)
# ---------------------------------------------------------------------------

def _new_position_from_alert(event, progress: float, entry_price: float, entry_source: str) -> dict:
    """V8-REWIRE VR8/VR9: builds a V8 position from a TelegramAlertEvent +
    an independently-resolved entry price -- never from a V7 Signal.
    entry_price==0.0 must never reach here (caller emits v8_pass_unpriced
    and returns before constructing a position for that case)."""
    now = time.time()
    import uuid
    return {
        "id": "V8" + str(uuid.uuid4())[:6],
        "signal_id": event.event_id,
        "chain": event.chain,
        "token_address": event.token_address,
        "token_symbol": event.token_symbol,
        "signal_type": "social_alert",
        "strength": "v8_fork",   # V8-REWIRE: not a V7 strength classification -- V8 has no such concept
        "signal_price": entry_price,
        "signal_time": now,
        "entry_price": entry_price,       # curve-baseline entry (N6 requirement)
        "entry_time": now,
        "size_usd": 1.0,                  # paper-only; size is irrelevant to pct outcomes
        "current_price": entry_price,
        "peak_price": entry_price,
        "status": "open",
        "exit_price": 0.0,
        "exit_time": 0.0,
        "exit_reason": "",
        "progress_at_signal": progress,
        "dex_id": "",   # V8-REWIRE: never populated pre-fork -- V8 forks before V7's dex_id is even resolved
        "entry_source": entry_source,
        "era": _current_era(),
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
        "era": pos.get("era", V8_ERA_PRE_REWIRE),   # old on-disk positions predate era tagging
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

    def maybe_open_from_alert(self, event) -> None:
        """V8-REWIRE VR1/VR7: entry point called from
        memecoin.scanner._on_telegram_signal() for EVERY raw Telegram
        alert, before screen_token() runs -- `event` is a
        memecoin.alert_event.TelegramAlertEvent, never a V7 Signal.

        Dispatches onto its own daemon thread immediately: the entry-price
        wait (up to _PRICE_WAIT_S) must never add latency to the caller,
        which is on V7's synchronous, latency-budgeted signal path. Never
        raises back to the caller — the thread target self-guards.
        """
        threading.Thread(target=self._evaluate_alert, args=(event,),
                         daemon=True, name="v8-fork-eval").start()

    def _evaluate_alert(self, event) -> None:
        """Runs on its own thread — see maybe_open_from_alert(). Never
        raises — any failure here must not affect v7/live, and there is
        no caller left to propagate to by the time this runs anyway.

        V8-REWIRE VR1: v8_fork_entered is emitted as the literal first
        statement, before transport-dedup/already_open/gate or any other
        return -- this is the one checkpoint that proves V8 was handed
        THIS raw alert at all, independent of what it decides. Paired
        with telegram_received in watchdog/checks/v8_funnel.py's
        _TRANSITIONS: every telegram_received event must reach this stage,
        which is the structural, monitored proof V8 sees the raw stream.
        """
        try:
            from memecoin import v8_telemetry as _v8_tel
            _v8_tel.emit("v8_fork_entered", event_id=event.event_id,
                         mint=event.token_address)
        except Exception:
            _v8_tel = None
        try:
            if _is_transport_duplicate(event.event_id):
                if _v8_tel is not None:
                    try:
                        _v8_tel.emit("v8_transport_duplicate", event_id=event.event_id,
                                     mint=event.token_address, result="rejected",
                                     reason="transport_duplicate")
                    except Exception:
                        pass
                return
            with self._lock:
                already_open = any(
                    p["token_address"] == event.token_address and p["status"] == "open"
                    for p in self._positions.values()
                )
            if already_open:
                if _v8_tel is not None:
                    try:
                        _v8_tel.emit("v8_gate_rejected", event_id=event.event_id,
                                     mint=event.token_address, result="rejected",
                                     reason="already_open")
                    except Exception:
                        pass
                return
            passed, reason, progress = passes_v8_gate(event)
            cap = _get_capture_for_gate(event.chain, event.token_address, event.event_id)
            venue = cap.venue_state_at_signal if cap is not None else "UNKNOWN"
            psrc  = cap.progress_source if cap is not None else ""
            if not passed:
                log.info("v8_paper: gate reject %s (event=%s) — %s",
                         event.token_symbol or event.token_address[:8], event.event_id[:12], reason)
                if _v8_tel is not None:
                    try:
                        _v8_tel.emit("v8_gate_rejected", event_id=event.event_id,
                                     mint=event.token_address, progress=progress,
                                     progress_source=psrc, venue_state=venue,
                                     result="rejected", reason=reason)
                    except Exception:
                        pass
                return

            entry_price, entry_source = _resolve_entry_price(event.chain, event.token_address)
            if entry_price <= 0:
                # V8-REWIRE VR8: gate passed but no independent price arrived
                # within budget -- an explicit terminal state, not a silent
                # drop. Distinguishable from v8_gate_rejected in the funnel.
                log.info("v8_paper: pass but unpriced %s (event=%s) — %s",
                         event.token_symbol or event.token_address[:8], event.event_id[:12], entry_source)
                if _v8_tel is not None:
                    try:
                        _v8_tel.emit("v8_pass_unpriced", event_id=event.event_id,
                                     mint=event.token_address, progress=progress,
                                     progress_source=psrc, venue_state=venue,
                                     result="unpriced", reason=entry_source)
                    except Exception:
                        pass
                return

            pos = _new_position_from_alert(event, progress, entry_price, entry_source)
            with self._lock:
                self._positions[pos["id"]] = pos
                self._save()
            log.info("v8_paper OPEN %s  progress=%.2f  entry=$%.10f  id=%s",
                     event.token_symbol or event.token_address[:8], progress, pos["entry_price"], pos["id"])
            if _v8_tel is not None:
                try:
                    _v8_tel.emit("v8_opened", event_id=event.event_id,
                                 mint=event.token_address, progress=progress,
                                 progress_source=psrc, venue_state=venue,
                                 result="opened", reason=pos["id"])
                except Exception:
                    pass
        except Exception as e:
            log.warning("v8_paper: _evaluate_alert failed (non-fatal): %s", e)

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
