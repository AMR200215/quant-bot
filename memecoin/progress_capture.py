"""
memecoin/progress_capture.py — PROGRESS-FIX PF2/PF3/PF4/PF5/PF6/PF7.

Canonical, source-provenanced progress_at_signal measurement, captured at
alert time off the trading critical path, shared between Research and
V8-paper (one ProgressCapture result per event_id, never two independent
measurements for the same signal).

Root cause this replaces (PROGRESS-FIX PF1, found 2026-08-06/07):
  subscribe_screening(address) -> ScreeningState created with latest_vsol=0
  -> get_screening_state(address) read back essentially immediately
  -> zero snapshot written before any real PumpPortal trade message could
     possibly have arrived.
That race is why progress_at_signal sat at ~0% populated from 2026-08-03
onward even after the underlying PumpPortal auth/subscribe bugs were fixed.

Source order (first success wins):
  A. pp_warm          — an ALREADY-warm PumpPortal ScreeningState (a real
                         observation, not a freshly-created default; see
                         ScreeningState.latest_vsol_ts in pumpportal_monitor.py)
  B. curve_account     — direct on-chain bonding-curve account read via
                         research.curve_oracle's PDA/layout code, batched
                         (50-200ms micro-batch window) via getMultipleAccounts.
                         Independent of SOL/USD price and of PumpPortal
                         entirely — this is the primary source, since it
                         gives genuine alert-time state rather than
                         whatever PumpPortal happens to have delivered.
  C. pp_post_alert     — bounded, event-driven (not polling) wait for the
                         first real PumpPortal vsol tick after B fails.
  D. failure           — vsol_at_signal/progress_at_signal stay NULL, with
                         an explicit progress_status reason. NEVER 0 — 0 is
                         a genuine physical value (freshly-launched curve),
                         not a missing-data sentinel.

Nothing in this module blocks the Telegram/scanner signal path. Capture
runs on background threads; capture_progress_async() returns immediately.

PF4: source B/C never depend on the scanner's `_screening` dict lifetime —
B talks to the chain directly, C uses its own waiter registry (not
get_screening_state after the fact), so an immediate scanner rejection/
eviction of the screening slot cannot destroy an in-flight capture.
"""

import dataclasses
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

from memecoin.config import GRAD_SOL_UI


def _helius_api_key() -> str:
    """Matches the rest of memecoin/ (executor.py, mint_classifier.py, etc.)
    — no HELIUS_API_KEY in memecoin.config, direct os.getenv at each site."""
    return os.getenv("HELIUS_API_KEY", "")

log = logging.getLogger(__name__)

VALID_SOURCES = frozenset({
    "curve_account", "pp_warm", "pp_post_alert", "pc_path_nearest_tick", "unknown",
})

# V8-TWIN-FIX VF2: canonical venue-state vocabulary. CURVE_ACTIVE and
# GRADUATED are genuine, positive determinations; DEX_ACTIVE is reserved
# for a confirmed post-graduation external-venue read (not currently
# produced by any capture source -- no source here queries PumpSwap/
# Jupiter pool state at alert time); UNKNOWN is the fail-closed default
# for every case where venue could not be positively confirmed. Gates on
# this must never treat UNKNOWN as passing.
VALID_VENUE_STATES = frozenset({"CURVE_ACTIVE", "GRADUATED", "DEX_ACTIVE", "UNKNOWN"})

# research.curve_oracle's richer venue_state vocabulary, normalized down
# to the four canonical states above. CURVE_MISSING is deliberately NOT
# mapped to GRADUATED -- a missing curve account is ambiguous (could be
# graduated-and-closed, could be a non-pump.fun token, could be a
# derivation edge case) and must fail closed to UNKNOWN, not be treated
# as a confirmed graduation.
_CURVE_ORACLE_VENUE_MAP = {
    "CURVE_ACTIVE": "CURVE_ACTIVE",
    "GRADUATED":    "GRADUATED",
    "CURVE_MISSING": "UNKNOWN",
    "PARSE_ERROR":  "UNKNOWN",
    "RPC_ERROR":    "UNKNOWN",
}


def _normalize_venue_state(raw: Optional[str]) -> str:
    return _CURVE_ORACLE_VENUE_MAP.get(raw or "", "UNKNOWN")


# ── PF2: canonical result ────────────────────────────────────────────────


@dataclasses.dataclass
class ProgressCapture:
    event_id:                 str
    token_address:             str
    alert_ts:                  float
    vsol_at_signal:             Optional[float]
    progress_at_signal:         Optional[float]
    progress_source:            str              # one of VALID_SOURCES
    progress_observed_at:       Optional[float]
    progress_capture_lag_ms:    Optional[float]
    progress_status:            str              # "ok" | explicit failure reason
    venue_state_at_signal:       str = "UNKNOWN"  # one of VALID_VENUE_STATES (V8-TWIN-FIX VF2)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ProgressCapture":
        return cls(**{f.name: d.get(f.name, "UNKNOWN" if f.name == "venue_state_at_signal" else None)
                       for f in dataclasses.fields(cls)})

    @classmethod
    def failure(cls, event_id: str, token_address: str, alert_ts: float,
                reason: str) -> "ProgressCapture":
        return cls(
            event_id=event_id, token_address=token_address, alert_ts=alert_ts,
            vsol_at_signal=None, progress_at_signal=None,
            progress_source="unknown", progress_observed_at=None,
            progress_capture_lag_ms=None, progress_status=reason,
            venue_state_at_signal="UNKNOWN",
        )

    @classmethod
    def success(cls, event_id: str, token_address: str, alert_ts: float,
                vsol_ui: float, source: str,
                venue_state: str = "UNKNOWN") -> "ProgressCapture":
        assert source in VALID_SOURCES, f"invalid progress_source: {source}"
        assert venue_state in VALID_VENUE_STATES, f"invalid venue_state: {venue_state}"
        now = time.time()
        return cls(
            event_id=event_id, token_address=token_address, alert_ts=alert_ts,
            vsol_at_signal=vsol_ui,
            progress_at_signal=round(vsol_ui / GRAD_SOL_UI, 4),
            progress_source=source,
            progress_observed_at=now,
            progress_capture_lag_ms=round((now - alert_ts) * 1000, 1),
            progress_status="ok",
            venue_state_at_signal=venue_state,
        )


# ── PF5: event-keyed durable store + in-process cache ───────────────────
# The durable file is what research/tracker.py (a SEPARATE process/service)
# reads. The in-process cache is what memecoin/v8_paper.py (SAME process as
# this module) reads — no file I/O needed for that side, but both paths
# always carry the exact same ProgressCapture object (PF6).

SNAPSHOT_PATH = Path(__file__).parent.parent / "research" / "data" / "progress_snapshots.jsonl"

_cache_lock: threading.Lock = threading.Lock()
_cache: dict[str, ProgressCapture] = {}
_cache_order: list[str] = []          # insertion order, for bounded FIFO eviction
_cache_waiters: dict[str, threading.Event] = {}
_CACHE_MAX = 5000


def _store_result(cap: ProgressCapture) -> None:
    with _cache_lock:
        if cap.event_id not in _cache:
            _cache_order.append(cap.event_id)
        _cache[cap.event_id] = cap
        while len(_cache_order) > _CACHE_MAX:
            oldest = _cache_order.pop(0)
            _cache.pop(oldest, None)
        waiter = _cache_waiters.get(cap.event_id)
    if waiter is not None:
        waiter.set()
    _append_snapshot(cap)
    log.debug("progress_capture: %s (%s) source=%s status=%s lag_ms=%s",
              cap.token_address[:8], cap.event_id[:12], cap.progress_source,
              cap.progress_status, cap.progress_capture_lag_ms)


SNAPSHOT_RETENTION_S = 900   # 15 min — this file is a cross-process handoff
                              # mailbox (research reads it within ~1.5s of an
                              # alert, PF10's historical recovery reads path
                              # ticks instead), not a permanent log. Without
                              # pruning it grows unbounded and every read
                              # becomes an O(n) scan over the whole history.
_prune_lock = threading.Lock()


def _append_snapshot(cap: ProgressCapture) -> None:
    """Best-effort append-with-pruning to the durable event-keyed store.
    Never raises. Mirrors memecoin/scanner.py's pp_snapshots.jsonl pattern
    (read, filter to retention window, append, rewrite)."""
    try:
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _prune_lock:
            now = time.time()
            kept = []
            if SNAPSHOT_PATH.exists():
                with open(SNAPSHOT_PATH) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            e = json.loads(line)
                            if now - e.get("alert_ts", 0) < SNAPSHOT_RETENTION_S:
                                kept.append(line)
                        except Exception:
                            pass
            kept.append(json.dumps(cap.to_dict()))
            with open(SNAPSHOT_PATH, "w") as f:
                f.write("\n".join(kept) + "\n")
    except Exception as e:
        log.debug("progress_capture: snapshot write failed for %s: %s", cap.event_id, e)


def get_capture(event_id: str) -> Optional[ProgressCapture]:
    """Non-blocking read of a (possibly not-yet-complete) capture."""
    with _cache_lock:
        return _cache.get(event_id)


def wait_for_capture(event_id: str, timeout_s: float) -> Optional[ProgressCapture]:
    """
    Bounded wait for an in-flight capture to complete. Returns None if still
    not ready after timeout_s — callers (V8's gate, research ingestion) must
    treat that as "unknown", never as progress=0.
    """
    with _cache_lock:
        existing = _cache.get(event_id)
        if existing is not None:
            return existing
        ev = _cache_waiters.get(event_id)
        if ev is None:
            ev = threading.Event()
            _cache_waiters[event_id] = ev
    ev.wait(timeout=timeout_s)
    with _cache_lock:
        _cache_waiters.pop(event_id, None)
        return _cache.get(event_id)


# ── Source A: warm PumpPortal state ──────────────────────────────────────

PP_WARM_FRESHNESS_S = 5.0   # latest_vsol_ts must be within this many seconds of "now"


def _try_pp_warm(token_address: str) -> Optional[float]:
    """Returns vsol_ui if a genuinely warm (real, timestamped) PP observation
    exists, else None. A freshly-created ScreeningState (latest_vsol_ts==0,
    the exact PF1 race) never counts — this is the fix for that bug."""
    try:
        from memecoin.pumpportal_monitor import monitor as _pp_monitor
    except Exception:
        return None
    state = _pp_monitor.get_screening_state(token_address)
    if state is None:
        return None
    if state.latest_vsol <= 0 or state.latest_vsol_ts <= 0:
        return None
    age = time.time() - state.latest_vsol_ts
    if age > PP_WARM_FRESHNESS_S:
        return None
    return state.latest_vsol


# ── Source C: bounded, event-driven PP post-alert fallback ──────────────
# PF4: keyed by mint in a registry OWNED by this module, not by reading
# scanner._screening after the fact — survives immediate screening eviction.

PP_POST_ALERT_TIMEOUT_S = 2.0

_waiting_lock = threading.Lock()
_waiting: dict[str, list[dict]] = {}   # mint -> [{"event_id","alert_ts","event","vsol"}]
_callback_registered = False
_callback_reg_lock = threading.Lock()


def _ensure_callback_registered() -> None:
    global _callback_registered
    if _callback_registered:
        return
    with _callback_reg_lock:
        if _callback_registered:
            return
        try:
            from memecoin.pumpportal_monitor import monitor as _pp_monitor
            _pp_monitor.add_vsol_update_callback(_on_vsol_update)
            _callback_registered = True
        except Exception as e:
            log.debug("progress_capture: vsol_update callback registration failed: %s", e)


def _on_vsol_update(mint: str, vsol_ui: float) -> None:
    """Called from the WS recv thread (via pumpportal_monitor) — must not block."""
    with _waiting_lock:
        waiters = _waiting.get(mint)
        if not waiters:
            return
        for w in waiters:
            if w["vsol"] is None:
                w["vsol"] = vsol_ui
        events = [w["event"] for w in waiters]
    for ev in events:
        ev.set()


def _fallback_pp_post_alert(event_id: str, token_address: str, alert_ts: float,
                             prior_failure_reason: str) -> None:
    """Runs on its own background thread — bounded wait, never blocks the caller."""
    _ensure_callback_registered()

    def _worker():
        w = {"event_id": event_id, "alert_ts": alert_ts,
             "event": threading.Event(), "vsol": None}
        with _waiting_lock:
            _waiting.setdefault(token_address, []).append(w)
        try:
            got = w["event"].wait(timeout=PP_POST_ALERT_TIMEOUT_S)
            if got and w["vsol"] is not None and w["vsol"] > 0:
                # V8-TWIN-FIX VF2: a real vSolInBondingCurve tick from a live
                # PumpPortal trade message can only occur while the token is
                # actively trading on the bonding curve -- PumpPortal's
                # subscribeTokenTrade fires on real swap events, and a swap
                # event carrying this field asserts on-curve state at that
                # moment. Not a guess: it's the same field this whole
                # measurement is built on.
                cap = ProgressCapture.success(
                    event_id, token_address, alert_ts, w["vsol"], "pp_post_alert",
                    venue_state="CURVE_ACTIVE",
                )
            else:
                reason = "pp_timeout" if not got else prior_failure_reason
                cap = ProgressCapture.failure(event_id, token_address, alert_ts, reason)
            _store_result(cap)
        finally:
            with _waiting_lock:
                lst = _waiting.get(token_address)
                if lst:
                    lst[:] = [x for x in lst if x is not w]
                    if not lst:
                        _waiting.pop(token_address, None)

    t = threading.Thread(
        target=_worker, daemon=True,
        name=f"progress-pp-fallback-{token_address[:8]}",
    )
    t.start()


# ── Source B: curve-account micro-batcher ────────────────────────────────

BATCH_WINDOW_S = 0.15   # 150ms, within the spec's 50-200ms range

_batch_lock = threading.Lock()
_batch_pending: dict[str, list] = {}   # mint -> [(event_id, alert_ts), ...]
_batch_timer: Optional[threading.Timer] = None


def _queue_curve_capture(event_id: str, token_address: str, alert_ts: float) -> None:
    global _batch_timer
    with _batch_lock:
        _batch_pending.setdefault(token_address, []).append((event_id, alert_ts))
        if _batch_timer is None:
            _batch_timer = threading.Timer(BATCH_WINDOW_S, _flush_curve_batch)
            _batch_timer.daemon = True
            _batch_timer.start()


def _flush_curve_batch() -> None:
    global _batch_timer, _batch_pending
    with _batch_lock:
        pending = _batch_pending
        _batch_pending = {}
        _batch_timer = None

    if not pending:
        return

    helius_key = _helius_api_key()
    if not helius_key:
        for mint, waiters in pending.items():
            for event_id, alert_ts in waiters:
                _fallback_pp_post_alert(event_id, mint, alert_ts, "curve_rpc_error")
        return

    try:
        from research.curve_oracle import get_curve_state_batch
        results = get_curve_state_batch(list(pending.keys()), helius_key)
    except Exception as e:
        log.debug("progress_capture: curve state batch failed: %s", e)
        results = {}

    for mint, waiters in pending.items():
        r = results.get(mint)
        ok = (
            r is not None
            and r.get("failure_reason") is None
            and r.get("vsol_ui") is not None
        )
        for event_id, alert_ts in waiters:
            if ok:
                # V8-TWIN-FIX VF2: curve_oracle already determined venue
                # state as part of this same read -- reuse it directly
                # rather than adding a second independent measurement.
                cap = ProgressCapture.success(
                    event_id, mint, alert_ts, r["vsol_ui"], "curve_account",
                    venue_state=_normalize_venue_state(r.get("venue_state")),
                )
                _store_result(cap)
            else:
                reason = (r or {}).get("failure_reason") or "curve_rpc_error"
                if reason == "curve_account_missing":
                    # Non-existent bonding-curve account: either not a
                    # pump.fun-native token, or already graduated with no
                    # historical curve state left — try the PP fallback
                    # rather than declaring failure immediately, since the
                    # token may still be legitimately screenable.
                    _fallback_pp_post_alert(event_id, mint, alert_ts,
                                            "graduated_no_historical_vsol")
                else:
                    _fallback_pp_post_alert(event_id, mint, alert_ts, reason)


# ── Public entrypoint (PF3) ───────────────────────────────────────────────

def capture_progress_async(event_id: str, token_address: str, alert_ts: float,
                            chain: str = "solana") -> None:
    """
    Fire off progress capture for one signal event. Returns immediately —
    never blocks the caller (the Telegram/scanner signal path).

    event_id must be the same deterministic event_id used elsewhere in the
    signal's lifecycle (RF5 convention) — PF5 requires event-keyed identity,
    not mint+time-window matching.
    """
    if not token_address or not event_id:
        return

    if chain != "solana":
        _store_result(ProgressCapture.failure(event_id, token_address, alert_ts, "non_pumpfun"))
        return

    # Source A: already-warm PP state — synchronous, just a dict lookup, no I/O.
    vsol = _try_pp_warm(token_address)
    if vsol is not None:
        # V8-TWIN-FIX VF2: same on-curve reasoning as the pp_post_alert
        # source above -- a real, fresh (PP_WARM_FRESHNESS_S-bounded)
        # latest_vsol observation only exists because a live bonding-curve
        # trade tick updated it.
        _store_result(ProgressCapture.success(event_id, token_address, alert_ts, vsol, "pp_warm",
                                               venue_state="CURVE_ACTIVE"))
        return

    # Source B (async, micro-batched) -> falls through to C on failure.
    _queue_curve_capture(event_id, token_address, alert_ts)
