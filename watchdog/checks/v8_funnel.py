"""watchdog/checks/v8_funnel.py — W7: stuck-stage / silent-disappearance
detection for the V8 candidate funnel.

Reuses logs/v8_funnel.jsonl, the 9-stage JSONL telemetry built during
V8-TWIN-FIX (memecoin/v8_telemetry.py) to root-cause the exact same class
of bug this check now guards against automatically: V8's gate rejected
100% of candidates for days because dex_id was misread as a graduation
signal, and nothing noticed until a manual audit.

Deliberately does NOT implement a reject-rate/conversion-percentage
anomaly detector. Real production data (docs/RECEIPTS.md's V8-TWIN-FIX
section) shows a genuine, legitimate ~1% open rate (1 open / 107 gate
entries over ~27h) -- V8's whole thesis is a narrow entry window. A
static "too many rejects" threshold would false-positive against known-
good behavior on day one. The safe, strong invariant instead is:
*every* candidate that enters a stage must reach exactly one terminal
disposition within grace -- that's true regardless of what the real
accept/reject ratio happens to be.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from watchdog.checks import CheckResult, STATUS_CRITICAL, STATUS_OK, STATUS_UNKNOWN, cap_severity

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FUNNEL_PATH = REPO_ROOT / "logs" / "v8_funnel.jsonl"

# (entry_stage, {terminal_stages}) pairs to check for completeness.
_TRANSITIONS = [
    ("add_signal_entered", {"dedup_rejected", "v8_gate_entered"}),
    ("v8_gate_entered", {"v8_gate_rejected", "v8_opened"}),
]

_MAX_LINES_READ = 20_000  # bound cost against unbounded file growth
_EXAMPLE_LIMIT = 5


def _read_recent_lines(path: Path, max_lines: int) -> list[str]:
    """Tail-like bounded read -- avoids loading an unboundedly large file
    into memory on every 5-min fast-profile run."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        block = 1 << 20  # 1 MiB chunks
        data = b""
        pos = size
        lines: list[bytes] = []
        while pos > 0 and len(lines) <= max_lines:
            read_size = min(block, pos)
            pos -= read_size
            f.seek(pos)
            data = f.read(read_size) + data
            lines = data.split(b"\n")
        text_lines = [l.decode("utf-8", errors="replace") for l in lines if l.strip()]
        return text_lines[-max_lines:]


def parse_funnel_events(path: Path, max_lines: int = _MAX_LINES_READ) -> list[dict]:
    lines = _read_recent_lines(path, max_lines)
    events = []
    for line in lines:
        try:
            events.append(json.loads(line))
        except Exception:
            continue  # malformed line -- best-effort telemetry, skip
    return events


def find_missing_terminal_dispositions(events: list[dict], now_ts: float,
                                        grace_seconds: float) -> dict[str, list[dict]]:
    """Returns {entry_stage: [event, ...]} for events that entered a stage
    more than grace_seconds ago with no matching terminal-stage event for
    the same event_id."""
    by_event: dict[str, dict[str, dict]] = {}
    for e in events:
        eid = e.get("event_id")
        stage = e.get("stage")
        if not eid or not stage:
            continue
        by_event.setdefault(eid, {})[stage] = e

    missing: dict[str, list[dict]] = {}
    for entry_stage, terminal_stages in _TRANSITIONS:
        offenders = []
        for eid, stages_seen in by_event.items():
            entry_event = stages_seen.get(entry_stage)
            if not entry_event:
                continue
            if now_ts - entry_event.get("ts", now_ts) < grace_seconds:
                continue  # too recent to expect resolution yet
            if not (terminal_stages & stages_seen.keys()):
                offenders.append(entry_event)
        if offenders:
            missing[entry_stage] = offenders
    return missing


def check_v8_funnel(now_ts: Optional[float] = None,
                     funnel_path: Optional[Path] = None,
                     grace_seconds: float = 120.0, severity_ceiling: str = "CRITICAL",
                     check_id: str = "funnel.v8") -> list[CheckResult]:
    import time
    now_ts = now_ts if now_ts is not None else time.time()
    path = Path(funnel_path) if funnel_path else DEFAULT_FUNNEL_PATH

    if not path.exists():
        return [CheckResult(
            check_id=check_id, status=STATUS_UNKNOWN,
            reason=f"funnel telemetry file does not exist: {path}",
            severity=severity_ceiling,
        )]

    try:
        events = parse_funnel_events(path)
    except Exception as e:
        return [CheckResult(
            check_id=check_id, status=STATUS_UNKNOWN,
            reason=f"failed to read/parse funnel telemetry: {e!r}",
            severity=severity_ceiling,
        )]

    if not events:
        return [CheckResult(
            check_id=check_id, status=STATUS_UNKNOWN,
            reason="funnel telemetry file exists but contains no parseable events",
            severity=severity_ceiling,
        )]

    missing = find_missing_terminal_dispositions(events, now_ts, grace_seconds)
    if not missing:
        return [CheckResult(
            check_id=check_id, status=STATUS_OK,
            reason=f"all {len(events)} recent funnel events (within read window) that are "
                   f"past grace have a terminal disposition",
        )]

    total_missing = sum(len(v) for v in missing.values())
    examples = []
    for stage, offenders in missing.items():
        for e in offenders[:_EXAMPLE_LIMIT]:
            examples.append({"stage": stage, "event_id": e.get("event_id"),
                              "mint": e.get("mint"), "ts": e.get("ts")})

    status = cap_severity(STATUS_CRITICAL, severity_ceiling)
    return [CheckResult(
        check_id=check_id, status=status,
        reason=(f"{total_missing} candidate(s) entered a funnel stage more than "
                f"{grace_seconds:.0f}s ago with no terminal disposition recorded — "
                f"possible silent-disappearance regression (this is the exact V8-TWIN-FIX "
                f"failure class: a candidate enters the pipeline and is never accounted for "
                f"again). Stages affected: {sorted(missing.keys())}"),
        severity=severity_ceiling,
        evidence={"missing_by_stage": {k: len(v) for k, v in missing.items()}, "examples": examples},
    )]
