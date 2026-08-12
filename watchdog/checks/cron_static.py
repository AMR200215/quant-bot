"""watchdog/checks/cron_static.py — W5A: is the /etc/cron.d definition
itself syntactically valid?

This is one of two independent questions (the other is W5B: did it
actually run). A file can be valid syntax and still never fire (disabled
service, wrong permissions) — and it can be invalid syntax while an
artifact still looks "fresh" from a past manual run (the actual K5
incident). Neither check alone is sufficient; see cron_execution.py for
the other half.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional

import croniter

from watchdog.checks import CheckResult, STATUS_CRITICAL, STATUS_OK, STATUS_UNKNOWN, STATUS_WARN, cap_severity

_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\s*=")
_JOURNAL_ERROR_PATTERNS = (
    "syntax error", "bad minute", "bad hour", "bad day", "bad month",
    "bad usercommand", "error (", "will be ignored", "ignored crontab",
    "bad file mode",
)


def _classify_line(line: str, prev_line_had_trailing_backslash: bool) -> tuple[str, Optional[str]]:
    """Returns (kind, malformed_reason). kind in
    {blank, comment, env, cron_entry, malformed}."""
    stripped = line.rstrip("\n").rstrip("\r")
    if not stripped.strip():
        return "blank", None
    if stripped.lstrip().startswith("#"):
        return "comment", None
    if _ENV_ASSIGN_RE.match(stripped.strip()):
        return "env", None

    if stripped.lstrip().startswith("@"):
        fields = stripped.split(None, 2)  # @nickname user command
        if len(fields) < 3:
            return "malformed", "@nickname line missing user or command field"
        return "cron_entry", None

    fields = stripped.split(None, 6)  # min hour dom month dow user command...
    if len(fields) < 7:
        reason = (f"line has {len(fields)} whitespace-separated field(s), "
                   f"need >=7 (5 schedule fields + user + command)")
        if prev_line_had_trailing_backslash:
            reason += (" — previous line ended with a trailing backslash; "
                        "/etc/cron.d does NOT support shell-style line "
                        "continuation, so this is almost certainly an "
                        "orphaned continuation line being misread as its "
                        "own (invalid) cron entry")
        return "malformed", reason

    schedule_expr = " ".join(fields[:5])
    try:
        croniter.croniter(schedule_expr)
    except Exception as e:
        return "malformed", f"croniter rejected schedule {schedule_expr!r}: {e}"
    return "cron_entry", None


def parse_cron_d_file(path: Path) -> tuple[bool, list[str], bool]:
    """Returns (has_any_cron_entry, malformed_reasons, file_readable)."""
    try:
        lines = path.read_text().splitlines(keepends=True)
    except Exception:
        return False, [f"could not read {path}"], False

    malformed: list[str] = []
    has_entry = False
    prev_trailing_backslash = False
    for i, line in enumerate(lines, start=1):
        kind, reason = _classify_line(line, prev_trailing_backslash)
        if kind == "cron_entry":
            has_entry = True
        elif kind == "malformed":
            malformed.append(f"line {i}: {reason}")
        prev_trailing_backslash = line.rstrip("\n").rstrip("\r").endswith("\\")
    return has_entry, malformed, True


def fetch_journal_cron_errors(since_iso: str) -> Optional[list[str]]:
    """Live journalctl fetch. Returns None (UNKNOWN evidence, not a pass)
    if journalctl is unavailable — this only happens on non-Linux dev
    machines or restricted environments; callers must not treat None as OK."""
    try:
        out = subprocess.run(
            ["journalctl", "-u", "cron", "--since", since_iso, "--no-pager"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None
        return out.stdout.splitlines()
    except Exception:
        return None


def check_cron_static(jobs: list[dict], cron_dir: str = "/etc/cron.d",
                       journal_lines: Optional[list[str]] = None,
                       journal_fetch_failed: bool = False) -> list[CheckResult]:
    """jobs: list of registry entries from checks.yaml's `jobs:` key.

    journal_lines: pre-fetched journalctl lines to cross-check against, or
    None. journal_fetch_failed distinguishes "we tried and it failed"
    (-> per-job evidence note, not a hard failure) from "caller didn't
    fetch" (tests pass their own fixed list instead).
    """
    cron_dir_path = Path(cron_dir)
    results: list[CheckResult] = []
    registered_files = {j["cron_file"] for j in jobs}

    for job in jobs:
        check_id = f"cron_static.{job['id']}"
        ceiling = job.get("severity", "CRITICAL")
        file_path = cron_dir_path / job["cron_file"]
        if not file_path.exists():
            results.append(CheckResult(
                check_id=check_id, status=cap_severity(STATUS_CRITICAL, ceiling),
                reason=f"managed cron file missing: {file_path}",
                subject=job["id"], severity=ceiling,
                evidence={"path": str(file_path)},
            ))
            continue

        has_entry, malformed, readable = parse_cron_d_file(file_path)
        journal_hits = []
        if journal_lines is not None:
            fname = job["cron_file"]
            for line in journal_lines:
                low = line.lower()
                if fname in line and any(p in low for p in _JOURNAL_ERROR_PATTERNS):
                    journal_hits.append(line)

        if not readable:
            results.append(CheckResult(
                check_id=check_id, status=cap_severity(STATUS_CRITICAL, ceiling),
                reason=f"cron file exists but is not readable: {file_path}",
                subject=job["id"], severity=ceiling,
            ))
        elif malformed:
            results.append(CheckResult(
                check_id=check_id, status=cap_severity(STATUS_CRITICAL, ceiling),
                reason=f"malformed cron.d syntax in {file_path}: " + "; ".join(malformed),
                subject=job["id"], severity=ceiling,
                evidence={"malformed_lines": malformed, "journal_hits": journal_hits},
            ))
        elif not has_entry:
            results.append(CheckResult(
                check_id=check_id, status=cap_severity(STATUS_CRITICAL, ceiling),
                reason=f"{file_path} contains no valid cron entry (comments/env only)",
                subject=job["id"], severity=ceiling,
            ))
        elif journal_hits:
            # Syntax parses fine by our reading, but cron's own daemon
            # logged a parser error against this exact file recently.
            # Trust the daemon over our parser here -- flag it.
            results.append(CheckResult(
                check_id=check_id, status=cap_severity(STATUS_CRITICAL, ceiling),
                reason=f"cron daemon logged a parser error against {file_path} recently",
                subject=job["id"], severity=ceiling,
                evidence={"journal_hits": journal_hits},
            ))
        else:
            status = STATUS_OK
            reason = "parses cleanly, no recent journal parser errors"
            if journal_lines is None and journal_fetch_failed:
                status = STATUS_UNKNOWN
                reason = "parses cleanly, but journal cross-check unavailable this run"
            results.append(CheckResult(
                check_id=check_id, status=status, reason=reason, subject=job["id"],
            ))

    # W5C: unmanaged jobs
    if cron_dir_path.is_dir():
        for f in sorted(cron_dir_path.glob("quantbot*")):
            if f.name not in registered_files:
                results.append(CheckResult(
                    check_id="cron_static.unmanaged",
                    status=STATUS_WARN,
                    reason=f"UNMANAGED_SCHEDULE: {f} exists but is not in watchdog/checks.yaml "
                           f"— it has no execution-liveness monitoring",
                    subject=f.name, severity="WARN",
                    evidence={"path": str(f)},
                ))

    return results
