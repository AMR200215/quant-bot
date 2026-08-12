"""watchdog/checks/telegram_feed.py — W6B: Telegram monitor liveness.

telegram_monitor.py is the sole signal source in the current
SOCIAL_ALERT_ONLY deployment mode -- if it silently died, nothing else in
the pipeline would ever fire, and a dead connection can look identical to
a quiet channel from the outside if you only watch for absence.

Two independent, already-existing evidence sources are combined rather
than adding new instrumentation:
1. telegram_monitor.py already self-reports unambiguous failure states via
   log lines (TELEGRAM_AUTH_REQUIRED, "HEALTH: tg-monitor thread is dead")
   -- these are read from journalctl, same pattern as cron_static.py.
2. logs/v8_funnel.jsonl's `telegram_received` stage (built for
   V8-TWIN-FIX) gives independent evidence of actual message flow,
   without asking telegram_monitor whether telegram_monitor is healthy.

Per the design spec's explicit requirement: a quiet channel is NOT the
same as a dead connection. Silence alone (no auth/thread-dead signal, but
no recent telegram_received either) is reported as WARN with that
ambiguity stated, never as CRITICAL and never silently as OK.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Optional

from watchdog.checks import CheckResult, STATUS_CRITICAL, STATUS_OK, STATUS_UNKNOWN, STATUS_WARN, cap_severity
from watchdog.checks.v8_funnel import DEFAULT_FUNNEL_PATH, parse_funnel_events

_AUTH_MARKER = "telegram_auth_required"
_THREAD_DEAD_MARKER = "health: tg-monitor thread is dead"
_STALE_MARKER = "connected but no message for >2h"


def fetch_journal_quantbot_lines(since_iso: str) -> Optional[list[str]]:
    try:
        out = subprocess.run(
            ["journalctl", "-u", "quantbot", "--since", since_iso, "--no-pager"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        return out.stdout.splitlines()
    except Exception:
        return None


def _most_recent_stage_ts(events: list[dict], stage: str) -> Optional[float]:
    ts_values = [e.get("ts") for e in events if e.get("stage") == stage and e.get("ts") is not None]
    return max(ts_values) if ts_values else None


def check_telegram_feed(now_ts: Optional[float] = None,
                         journal_lines: Optional[list[str]] = None,
                         journal_fetch_failed: bool = False,
                         funnel_events: Optional[list[dict]] = None,
                         funnel_path: Optional[Path] = None,
                         stale_threshold_s: float = 2 * 3600,
                         severity_ceiling: str = "CRITICAL",
                         check_id: str = "feed.telegram") -> list[CheckResult]:
    now_ts = now_ts if now_ts is not None else time.time()

    if journal_lines is not None:
        low_lines = [l.lower() for l in journal_lines]
        if any(_AUTH_MARKER in l for l in low_lines):
            return [CheckResult(
                check_id=check_id, status=cap_severity(STATUS_CRITICAL, severity_ceiling),
                reason="TELEGRAM_AUTH_REQUIRED logged recently — Telethon session expired/missing, "
                       "run: python -m research.tg_auth",
                severity=severity_ceiling,
            )]
        if any(_THREAD_DEAD_MARKER in l for l in low_lines):
            return [CheckResult(
                check_id=check_id, status=cap_severity(STATUS_CRITICAL, severity_ceiling),
                reason="tg-monitor thread reported dead recently — quantbot restart required",
                severity=severity_ceiling,
            )]
        app_reported_stale = any(_STALE_MARKER in l for l in low_lines)
    else:
        app_reported_stale = False

    if funnel_events is None:
        path = Path(funnel_path) if funnel_path else DEFAULT_FUNNEL_PATH
        if not path.exists():
            return [CheckResult(
                check_id=check_id, status=STATUS_UNKNOWN,
                reason=f"no auth/thread-dead signal, but funnel telemetry file missing "
                       f"({path}) so message-flow cannot be independently verified either",
                severity=severity_ceiling,
            )]
        try:
            funnel_events = parse_funnel_events(path)
        except Exception as e:
            return [CheckResult(
                check_id=check_id, status=STATUS_UNKNOWN,
                reason=f"no auth/thread-dead signal, but failed to read funnel telemetry: {e!r}",
                severity=severity_ceiling,
            )]

    last_received = _most_recent_stage_ts(funnel_events, "telegram_received")
    if last_received is None:
        return [CheckResult(
            check_id=check_id, status=STATUS_UNKNOWN,
            reason="no auth/thread-dead signal, but no telegram_received event found in "
                   "readable funnel history — cannot independently confirm message flow",
            severity=severity_ceiling,
        )]

    age = now_ts - last_received
    if age <= stale_threshold_s:
        status = STATUS_OK
        if journal_lines is None and journal_fetch_failed:
            status = STATUS_UNKNOWN  # incomplete evidence -- don't claim OK on funnel alone
        return [CheckResult(
            check_id=check_id, status=status,
            reason=f"last telegram_received event {age:.0f}s ago (threshold {stale_threshold_s:.0f}s)"
                   + ("; journal cross-check unavailable this run" if status == STATUS_UNKNOWN else ""),
            evidence={"last_received_ts": last_received, "age_s": age},
        )]

    reason = (f"no telegram_received event in {age:.0f}s (threshold {stale_threshold_s:.0f}s) "
              f"and no explicit AUTH_REQUIRED/THREAD_DEAD signal — ambiguous: could be a "
              f"legitimately quiet channel or a silently dead connection")
    if app_reported_stale:
        reason += "; app's own in-process watchdog independently logged CONNECTED_BUT_STALE, corroborating"
    return [CheckResult(
        check_id=check_id, status=cap_severity(STATUS_WARN, severity_ceiling),
        reason=reason, severity=severity_ceiling,
        evidence={"last_received_ts": last_received, "age_s": age, "app_reported_stale": app_reported_stale},
    )]
