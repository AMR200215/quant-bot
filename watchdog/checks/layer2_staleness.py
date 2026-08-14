"""watchdog/checks/layer2_staleness.py — W17: Layer 2's own existence is
not enough; if it stops running, that must be visible too.

Reads logs/watchdog/layer2_heartbeat.json, written by
watchdog/layer2/findings_store.py on the GitHub Actions runner and
persisted back to the repo via a normal git commit -- this Layer 1 check
sees it locally once the VPS's own periodic `git pull` (existing cron,
9am/6pm) has picked it up. That pull cadence is the real visibility
latency here (documented, not engineered around, in the Phase 5 receipt).

Reciprocal supervision: Layer 1 watches Layer 2 via this check; nothing
currently watches whether Layer 1 itself has gone stale from OUTSIDE the
VPS except Layer 2's own audit (which reads Layer 1's watchdog_runs table
as evidence, per the evidence_dump.py collector) -- each one is the
other's external observer, per W17's explicit design.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from watchdog.checks import CheckResult, STATUS_CRITICAL, STATUS_OK, STATUS_UNKNOWN, cap_severity

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_HEARTBEAT_PATH = REPO_ROOT / "logs" / "watchdog" / "layer2_heartbeat.json"
DEFAULT_STALE_THRESHOLD_S = 30 * 3600  # 30h: daily cadence + generous grace for pull-cadence lag


def check_layer2_staleness(now_ts: Optional[float] = None,
                            heartbeat_path: Optional[Path] = None,
                            stale_threshold_s: float = DEFAULT_STALE_THRESHOLD_S,
                            severity_ceiling: str = "WARN",
                            check_id: str = "layer2.staleness") -> list[CheckResult]:
    now_ts = now_ts if now_ts is not None else time.time()
    path = Path(heartbeat_path) if heartbeat_path else DEFAULT_HEARTBEAT_PATH

    if not path.exists():
        return [CheckResult(
            check_id=check_id, status=STATUS_UNKNOWN,
            reason="no Layer 2 heartbeat file found — either Layer 2 has never run "
                   "successfully yet, or this VPS hasn't pulled a commit containing one",
            severity=severity_ceiling,
        )]

    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return [CheckResult(check_id=check_id, status=STATUS_UNKNOWN,
                             reason=f"failed to parse heartbeat file: {e!r}", severity=severity_ceiling)]

    completed_at = data.get("last_audit_completed_at")
    if completed_at is None:
        return [CheckResult(check_id=check_id, status=STATUS_UNKNOWN,
                             reason="heartbeat file present but missing last_audit_completed_at",
                             severity=severity_ceiling)]

    age = now_ts - completed_at
    evidence = {"last_audit_id": data.get("last_audit_id"), "age_s": age,
                "findings_count": data.get("findings_count"),
                "critical_count": data.get("critical_count")}

    if age > stale_threshold_s:
        return [CheckResult(
            check_id=check_id, status=cap_severity(STATUS_CRITICAL, severity_ceiling),
            reason=f"Layer 2's last successful audit was {age/3600:.1f}h ago "
                   f"(threshold {stale_threshold_s/3600:.0f}h) — the external audit agent "
                   f"itself appears to have stopped running",
            severity=severity_ceiling, evidence=evidence,
        )]

    return [CheckResult(
        check_id=check_id, status=STATUS_OK,
        reason=f"last Layer 2 audit {age/3600:.1f}h ago ({data.get('last_audit_id')}, "
               f"{data.get('findings_count', 0)} finding(s))",
        evidence=evidence,
    )]
