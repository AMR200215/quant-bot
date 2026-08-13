"""watchdog/checks/research_pipeline.py — W6D: research pipeline
upstream-flowing/downstream-stalled detection.

Two independent stages checked, both against durable, already-existing
on-disk evidence -- zero new instrumentation:

1. Queue consumption lag: research/data/signal_queue.jsonl (written by
   memecoin/scanner.py) vs research/data/.queue_offset (persisted by
   research/tg_listener.py's FileQueueListener after each processed
   line). A large, sustained gap between file size and offset with the
   queue still growing is exactly "upstream flowing, downstream
   stalled" -- the same shape as the historical pp_vsol-never-reached-
   Supabase bug, just at an earlier pipeline stage.

2. Spool growth: research/spool/failed_inserts.jsonl (written by
   research/spool/writer.py whenever a Supabase insert fails) is direct,
   durable evidence of active data loss if it's growing. A historical
   failure mode is already visible in this file (a progress_capture_lag_ms
   float rejected by an integer-typed Supabase column) -- confirmed dormant
   for 5+ days as of this check's authoring, which is exactly why growth
   detection (not raw historical count) is the right signal: the file's
   total size says nothing about whether the problem is still happening.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from watchdog.checks import CheckResult, STATUS_CRITICAL, STATUS_OK, STATUS_UNKNOWN, STATUS_WARN, cap_severity

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_QUEUE_PATH = REPO_ROOT / "research" / "data" / "signal_queue.jsonl"
DEFAULT_OFFSET_PATH = REPO_ROOT / "research" / "data" / ".queue_offset"
DEFAULT_FAILED_INSERTS_PATH = REPO_ROOT / "research" / "spool" / "failed_inserts.jsonl"

_MIN_SAMPLES_FOR_SPOOL_ALERT = 5   # per spec: never alert on n=1-2
_STALL_BYTES_THRESHOLD = 50_000    # ~10-20 messages behind, given real message sizes seen (~2-5KB)


def check_queue_lag(now_ts: Optional[float] = None,
                     queue_path: Optional[Path] = None, offset_path: Optional[Path] = None,
                     severity_ceiling: str = "CRITICAL",
                     check_id: str = "pipeline.research_queue_lag") -> list[CheckResult]:
    now_ts = now_ts if now_ts is not None else time.time()
    qpath = Path(queue_path) if queue_path else DEFAULT_QUEUE_PATH
    opath = Path(offset_path) if offset_path else DEFAULT_OFFSET_PATH

    if not qpath.exists():
        return [CheckResult(check_id=check_id, status=STATUS_UNKNOWN,
                             reason=f"signal queue file does not exist: {qpath}", severity=severity_ceiling)]
    if not opath.exists():
        return [CheckResult(check_id=check_id, status=STATUS_UNKNOWN,
                             reason=f"queue offset file does not exist: {opath} — consumer may never "
                                    f"have started, or this is a fresh install",
                             severity=severity_ceiling)]

    try:
        queue_size = qpath.stat().st_size
        offset_val = int(opath.read_text().strip())
    except Exception as e:
        return [CheckResult(check_id=check_id, status=STATUS_UNKNOWN,
                             reason=f"failed to read queue/offset state: {e!r}", severity=severity_ceiling)]

    gap = queue_size - offset_val
    evidence = {"queue_size": queue_size, "offset": offset_val, "gap_bytes": gap}

    if gap < 0:
        # Offset ahead of file size -- file was truncated/rotated since the
        # offset was last persisted. Not evidence of a stall; flag as
        # UNKNOWN rather than guessing.
        return [CheckResult(check_id=check_id, status=STATUS_UNKNOWN,
                             reason=f"offset ({offset_val}) exceeds queue file size ({queue_size}) — "
                                    f"file likely rotated/truncated since offset was last persisted",
                             evidence=evidence, severity=severity_ceiling)]

    if gap <= _STALL_BYTES_THRESHOLD:
        return [CheckResult(check_id=check_id, status=STATUS_OK,
                             reason=f"consumer caught up (gap={gap} bytes)", evidence=evidence)]

    return [CheckResult(
        check_id=check_id, status=cap_severity(STATUS_WARN, severity_ceiling),
        reason=f"consumer {gap} bytes behind the queue's write position — upstream is writing, "
               f"downstream (research.tg_listener.FileQueueListener) has not consumed it. "
               f"Note: this alone doesn't distinguish 'consumer slow' from 'consumer dead' — "
               f"cross-check the research.main process directly if this persists.",
        severity=severity_ceiling, evidence=evidence,
    )]


def _parse_iso_ts(s: str) -> Optional[float]:
    try:
        import datetime
        return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def check_spool_growth(now_ts: Optional[float] = None,
                        failed_inserts_path: Optional[Path] = None,
                        lookback_seconds: float = 2 * 3600,
                        severity_ceiling: str = "WARN",
                        check_id: str = "pipeline.research_spool") -> list[CheckResult]:
    now_ts = now_ts if now_ts is not None else time.time()
    path = Path(failed_inserts_path) if failed_inserts_path else DEFAULT_FAILED_INSERTS_PATH

    if not path.exists():
        return [CheckResult(check_id=check_id, status=STATUS_OK,
                             reason="no failed_inserts.jsonl on disk — nothing has ever failed to "
                                    "insert (or spool has never been touched)")]

    try:
        lines = path.read_text().splitlines()
    except Exception as e:
        return [CheckResult(check_id=check_id, status=STATUS_UNKNOWN,
                             reason=f"failed to read spool file: {e!r}", severity=severity_ceiling)]

    recent = []
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        ts = _parse_iso_ts(obj.get("ts", ""))
        if ts is not None and now_ts - ts <= lookback_seconds:
            recent.append(obj)

    if len(recent) < _MIN_SAMPLES_FOR_SPOOL_ALERT:
        return [CheckResult(
            check_id=check_id, status=STATUS_OK,
            reason=f"{len(recent)} failed insert(s) in the last {lookback_seconds/3600:.1f}h — "
                   f"below the {_MIN_SAMPLES_FOR_SPOOL_ALERT}-sample floor for an alert "
                   f"(total historical lines on disk: {len(lines)}, not itself evidence of an active problem)",
            evidence={"recent_count": len(recent), "total_lines": len(lines)},
        )]

    examples = [r.get("error", "")[:200] for r in recent[:3]]
    return [CheckResult(
        check_id=check_id, status=cap_severity(STATUS_WARN, severity_ceiling),
        reason=f"{len(recent)} Supabase insert failures in the last {lookback_seconds/3600:.1f}h — "
               f"active data loss to the research DB",
        severity=severity_ceiling,
        evidence={"recent_count": len(recent), "examples": examples},
    )]
