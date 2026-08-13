"""watchdog/checks/pumpportal_feed.py — W6A: PumpPortal real-time tick feed
liveness.

V8's paper book (pp_warm/pp_post_alert progress sources) and, once
LIVE_TRADING is re-enabled, live position monitoring both depend on real
PumpPortal ticks -- this is the exact feed that was silently dead for 24h
earlier in this project's history (~$36 of real losses before it was
root-caused). K2 already added an in-process tick deadman
(pumpportal_monitor.py's _check_tick_deadman) that alerts when an open,
subscribed position gets 0 ticks for 5min -- but that check dies with the
process it's running in. This is the same class of gap Layer 1 exists to
close: an independent, externally-scheduled check that reads durable
evidence (journalctl) rather than trusting the in-process check's own
continued existence.

Important, confirmed against real production logs before writing this:
PumpPortal reconnects roughly every 45-60s by design (log line: "PumpPortal
using pre-warmed rotation WS (gap <100ms)" -- a deliberate rotation
strategy, not a failure loop). This check must never flag reconnect
*frequency* as anomalous -- only actual error/deadman evidence.
"""

from __future__ import annotations

import time
from typing import Optional

from watchdog.checks import CheckResult, STATUS_CRITICAL, STATUS_OK, STATUS_UNKNOWN, STATUS_WARN, cap_severity

_CONNECTED_MARKER = "pumpportal websocket connected"
_ERROR_MARKER = "pumpportal ws error"
_DEADMAN_LIVE_MARKER = "pumpportal tick deadman"  # WARNING-level, fires only when LIVE_TRADING=true
_DEADMAN_SUPPRESSED_MARKER = "suppressed, live_trading=false"


def _extract_ts_and_marker(lines: list[str], marker: str) -> Optional[str]:
    """Returns the last matching raw line (used for its journalctl-prefixed
    timestamp) or None. journalctl lines already carry their own
    timestamp prefix -- we don't need to re-parse it, just need
    presence/absence and ordering, which `journalctl`'s own chronological
    output order gives us for free."""
    matches = [l for l in lines if marker in l.lower()]
    return matches[-1] if matches else None


def check_pumpportal_feed(now_ts: Optional[float] = None,
                           journal_lines: Optional[list[str]] = None,
                           journal_fetch_failed: bool = False,
                           severity_ceiling: str = "CRITICAL",
                           check_id: str = "feed.pumpportal") -> list[CheckResult]:
    now_ts = now_ts if now_ts is not None else time.time()

    if journal_lines is None:
        return [CheckResult(
            check_id=check_id, status=STATUS_UNKNOWN,
            reason="journalctl evidence unavailable this run — cannot assess PumpPortal feed state",
            severity=severity_ceiling,
        )]

    low_lines = [l.lower() for l in journal_lines]

    live_deadman_hits = [l for l in low_lines if _DEADMAN_LIVE_MARKER in l
                          and _DEADMAN_SUPPRESSED_MARKER not in l]
    if live_deadman_hits:
        return [CheckResult(
            check_id=check_id, status=cap_severity(STATUS_CRITICAL, severity_ceiling),
            reason="PRIMARY_FEED_DEGRADED — PumpPortal tick deadman fired: an open, subscribed "
                   "position has had 0 real-time ticks for >5min while LIVE_TRADING=true, "
                   "likely running on stale/fallback pricing without anyone noticing",
            severity=severity_ceiling,
            evidence={"deadman_lines": live_deadman_hits[-3:]},
        )]

    last_connect = _extract_ts_and_marker(journal_lines, _CONNECTED_MARKER)
    last_error = _extract_ts_and_marker(journal_lines, _ERROR_MARKER)

    # journalctl preserves chronological order, so "last in the list" for
    # each marker tells us which happened more recently only by comparing
    # their positions in the merged, ordered line list.
    if last_connect is not None:
        connect_idx = len(journal_lines) - 1 - journal_lines[::-1].index(last_connect)
    else:
        connect_idx = -1
    if last_error is not None:
        error_idx = len(journal_lines) - 1 - journal_lines[::-1].index(last_error)
    else:
        error_idx = -1

    if connect_idx == -1 and error_idx == -1:
        suppressed = any(_DEADMAN_LIVE_MARKER in l and _DEADMAN_SUPPRESSED_MARKER in l for l in low_lines)
        status = STATUS_UNKNOWN
        reason = "no PumpPortal connect/error/deadman evidence found in lookback window"
        if suppressed:
            reason += " (weekly suppressed-deadman note present — LIVE_TRADING=false, not itself a failure signal)"
        return [CheckResult(check_id=check_id, status=status, reason=reason, severity=severity_ceiling)]

    if error_idx > connect_idx:
        return [CheckResult(
            check_id=check_id, status=cap_severity(STATUS_WARN, severity_ceiling),
            reason="most recent PumpPortal WS event was an error/retry, with no subsequent "
                   "successful connect logged yet in this window — possibly degraded",
            severity=severity_ceiling,
            evidence={"last_error_line": last_error, "last_connect_line": last_connect},
        )]

    return [CheckResult(
        check_id=check_id, status=STATUS_OK,
        reason="most recent PumpPortal WS event was a successful connect "
               "(reconnects every ~45-60s are expected rotation behavior, not evaluated here)",
        evidence={"last_connect_line": last_connect},
    )]
