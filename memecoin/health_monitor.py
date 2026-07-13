"""
memecoin/health_monitor.py — zero-flow alarms.

Call bump_* functions on every matching event.  A background thread checks
every 5 minutes and fires Telegram alerts when thresholds are breached.

Alarms:
  (a) no TG message received in 2 h            → TG feed silent
  (b) preflight_no_price rate > 30%/hr         → PP monitor quota / connectivity
  (c) creator-resolution fail rate > 20%/hr    → Helius quota suspect
  (d) zero live-eligible signals in 6 h        → whole funnel stalled
  (e) F3 live-drought: >=3 social_alert paper opens in 3h AND 0 live buy
      attempts in 3h → "live pipeline silent: last block reasons: <top 3>"
      Detects Bug-1-class silent deaths within 3h instead of 3 days.

Usage (from scanner.py / portfolio.py):
    from memecoin.health_monitor import (
        bump_tg_message, bump_live_eligible,
        bump_preflight_no_price, bump_creator_fail,
        bump_social_alert_paper, bump_live_attempt, bump_gate_block,
    )
"""

import logging
import threading
import time
from collections import deque, Counter

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state (thread-safe via _lock)
# ---------------------------------------------------------------------------
_lock = threading.Lock()

# Timestamps of recent events (kept for sliding-window rate calcs)
_TG_WINDOW_SEC      = 2 * 3600    # 2h TG silence alarm
_RATE_WINDOW_SEC    = 3600        # 1h window for rate alarms
_LIVE_WINDOW_SEC    = 6 * 3600    # 6h zero-live-eligible alarm

_tg_message_times:        deque = deque()   # timestamps of TG messages received
_preflight_no_price_times: deque = deque()  # timestamps of preflight-no-price events
_creator_fail_times:       deque = deque()  # timestamps of creator-resolution fails
_live_eligible_times:      deque = deque()  # timestamps of signals reaching live gate

# Per-hour denominators for rate alarms (total events in the rate window)
_preflight_total_times: deque = deque()   # all preflight attempts (for % calc)
_creator_total_times:   deque = deque()   # all creator-resolution attempts

# F3 live-drought alarm state
_DROUGHT_PAPER_WINDOW = 3 * 3600    # 3h window for social paper count
_DROUGHT_MIN_PAPERS   = 3           # fire if >= 3 paper opens with 0 live attempts
_social_paper_times:   deque = deque()           # timestamps of social_alert paper opens
_live_attempt_times:   deque = deque()           # timestamps of live buy path entered
_gate_block_log:       deque = deque(maxlen=50)  # (timestamp, reason_str) recent gate blocks


# ---------------------------------------------------------------------------
# Rate-threshold configuration
# ---------------------------------------------------------------------------
PREFLIGHT_NO_PRICE_ALARM_PCT = 0.30   # > 30% of preflight attempts have no PP price
CREATOR_FAIL_ALARM_PCT       = 0.20   # > 20% of creator lookups failed
CHECK_INTERVAL_SEC           = 300    # check every 5 minutes


# ---------------------------------------------------------------------------
# Public bump functions — called from scanner.py / portfolio.py
# ---------------------------------------------------------------------------

def bump_tg_message():
    """Call once per TG message received (before any filtering)."""
    with _lock:
        _tg_message_times.append(time.time())


def bump_live_eligible():
    """Call once per signal that passes all screens and reaches the live gate."""
    with _lock:
        _live_eligible_times.append(time.time())


def bump_preflight_no_price():
    """Call when PumpPortal pre-flight returns no price (trade blocked)."""
    with _lock:
        now = time.time()
        _preflight_no_price_times.append(now)
        _preflight_total_times.append(now)


def bump_preflight_attempt():
    """Call on every preflight attempt (hit or miss) to track the denominator."""
    with _lock:
        _preflight_total_times.append(time.time())


def bump_creator_fail():
    """Call when creator-wallet resolution fails."""
    with _lock:
        now = time.time()
        _creator_fail_times.append(now)
        _creator_total_times.append(now)


def bump_creator_attempt():
    """Call on every creator lookup attempt to track the denominator."""
    with _lock:
        _creator_total_times.append(time.time())


# F3 — live-drought alarm bumps
def bump_social_alert_paper():
    """Call when a social_alert paper position is opened."""
    with _lock:
        _social_paper_times.append(time.time())


def bump_live_attempt():
    """Call when the live buy path is entered (just before program gate check)."""
    with _lock:
        _live_attempt_times.append(time.time())


def bump_gate_block(reason: str):
    """Call at every gate block with a short reason string."""
    with _lock:
        _gate_block_log.append((time.time(), reason))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _trim(q: deque, window_sec: float):
    """Remove entries older than window_sec from the front of the deque."""
    cutoff = time.time() - window_sec
    while q and q[0] < cutoff:
        q.popleft()


def _rate(events: deque, total: deque, window_sec: float) -> float | None:
    """Return events/total within the window, or None if no total events."""
    _trim(events, window_sec)
    _trim(total, window_sec)
    if not total:
        return None
    return len(events) / len(total)


def _send_alert(msg: str):
    try:
        from app.alerts import _send
        _send(msg)
    except Exception as e:
        log.warning("health_monitor: alert send failed: %s", e)


# ---------------------------------------------------------------------------
# Alarm state — prevent alert spam (fire at most once per 30 min per alarm)
# ---------------------------------------------------------------------------
_ALARM_COOLDOWN = 1800   # 30 min
_last_alarm: dict[str, float] = {}


def _should_fire(alarm_key: str) -> bool:
    now = time.time()
    last = _last_alarm.get(alarm_key, 0)
    if now - last >= _ALARM_COOLDOWN:
        _last_alarm[alarm_key] = now
        return True
    return False


# ---------------------------------------------------------------------------
# Checker — runs in background thread
# ---------------------------------------------------------------------------

def _check():
    """Run all zero-flow checks and fire alerts if thresholds breached."""
    now = time.time()

    with _lock:
        # (a) TG silence
        _trim(_tg_message_times, _TG_WINDOW_SEC)
        tg_count = len(_tg_message_times)

        # (b) preflight no-price rate
        pf_rate = _rate(_preflight_no_price_times, _preflight_total_times, _RATE_WINDOW_SEC)

        # (c) creator-fail rate
        cr_rate = _rate(_creator_fail_times, _creator_total_times, _RATE_WINDOW_SEC)

        # (d) live-eligible silence
        _trim(_live_eligible_times, _LIVE_WINDOW_SEC)
        live_count = len(_live_eligible_times)

        # (e) F3 live-drought: papers flowing, live buy path never entered
        _trim(_social_paper_times, _DROUGHT_PAPER_WINDOW)
        _trim(_live_attempt_times, _DROUGHT_PAPER_WINDOW)
        drought_papers = len(_social_paper_times)
        drought_attempts = len(_live_attempt_times)
        # Snapshot recent block reasons within the window
        cutoff_drought = now - _DROUGHT_PAPER_WINDOW
        recent_reasons = [r for t, r in _gate_block_log if t >= cutoff_drought]

    # Alarm (a)
    if tg_count == 0 and _should_fire("tg_silence"):
        log.warning("HEALTH ALARM: no TG message in 2h — feed may be disconnected")
        _send_alert(
            "🚨 HEALTH: No Telegram message received in 2h. "
            "TG monitor may be disconnected."
        )

    # Alarm (b)
    if pf_rate is not None and pf_rate > PREFLIGHT_NO_PRICE_ALARM_PCT:
        if _should_fire("pf_no_price"):
            log.warning(
                "HEALTH ALARM: preflight_no_price rate %.0f%% > %.0f%% threshold",
                pf_rate * 100, PREFLIGHT_NO_PRICE_ALARM_PCT * 100,
            )
            _send_alert(
                f"🚨 HEALTH: PP pre-flight no-price rate {pf_rate*100:.0f}% "
                f"(>{PREFLIGHT_NO_PRICE_ALARM_PCT*100:.0f}%). "
                f"Check PumpPortal WS / Helius quota."
            )

    # Alarm (c)
    if cr_rate is not None and cr_rate > CREATOR_FAIL_ALARM_PCT:
        if _should_fire("creator_fail"):
            log.warning(
                "HEALTH ALARM: creator-resolution fail rate %.0f%% > %.0f%% threshold",
                cr_rate * 100, CREATOR_FAIL_ALARM_PCT * 100,
            )
            _send_alert(
                f"🚨 HEALTH: Creator-resolution fail rate {cr_rate*100:.0f}% "
                f"(>{CREATOR_FAIL_ALARM_PCT*100:.0f}%). "
                f"Helius quota may be exhausted."
            )

    # Alarm (d)
    if live_count == 0 and _should_fire("live_eligible_silence"):
        log.warning("HEALTH ALARM: zero live-eligible signals in 6h — funnel stalled")
        _send_alert(
            "🚨 HEALTH: Zero live-eligible signals in 6h. "
            "Screening funnel may be stalled."
        )

    # Alarm (e) — F3 live-drought
    if drought_papers >= _DROUGHT_MIN_PAPERS and drought_attempts == 0:
        if _should_fire("live_drought"):
            top_reasons = [r for r, _ in Counter(recent_reasons).most_common(3)] if recent_reasons else ["none_logged"]
            top_str = " | ".join(top_reasons)
            log.warning(
                "HEALTH ALARM: live pipeline silent — %d paper entries, 0 live attempts in 3h. "
                "top block reasons: %s",
                drought_papers, top_str,
            )
            _send_alert(
                f"🚨 LIVE DROUGHT: {drought_papers} social_alert paper trades, 0 live buy attempts in 3h.\n"
                f"Last block reasons: {top_str}\n"
                f"Check entry gate / kill switch / preflight."
            )


def _monitor_loop():
    """Background thread: check alarms every CHECK_INTERVAL_SEC."""
    log.info("health_monitor: background alarm thread started")
    while True:
        try:
            time.sleep(CHECK_INTERVAL_SEC)
            _check()
        except Exception as e:
            log.warning("health_monitor: check error: %s", e)


# ---------------------------------------------------------------------------
# Start background thread (called once from scanner.py start())
# ---------------------------------------------------------------------------
_started = False
_start_lock = threading.Lock()


def force_test_live_drought(n_papers: int = 3):
    """
    Inject N fake social_alert paper opens with 0 live attempts, then run _check().
    Used for forced-test artifact verification (F3).
    """
    with _lock:
        now = time.time()
        for _ in range(n_papers):
            _social_paper_times.append(now - 60)   # 1 min ago
        # No live attempts injected — drought condition
    _check()


def start():
    """Start the background alarm monitor thread (idempotent)."""
    global _started
    with _start_lock:
        if _started:
            return
        t = threading.Thread(target=_monitor_loop, daemon=True, name="health_monitor")
        t.start()
        _started = True
        log.info(
            "health_monitor: started  intervals=%ds  "
            "alarms: tg=2h pf=30%% cr=20%% live=6h drought=3h/K3",
            CHECK_INTERVAL_SEC,
        )
