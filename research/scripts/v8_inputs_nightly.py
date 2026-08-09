"""
research/scripts/v8_inputs_nightly.py — K5: nightly refresh of N4(a/b),
appended to docs/V8_INPUTS.md with a date + clean-n header.

Deliberately does NOT reimplement report.py's era-split / cohort-verdict
logic — that would risk a second, divergent copy of the same computation.
Instead runs report.py and path_stats.py as subprocesses (same code every
other analysis in this repo uses) and appends their relevant sections
verbatim, dated, so V8_INPUTS.md accumulates a real history rather than
being overwritten each run.

Freeze gate (K5): no V8 freeze until clean-n >= 2500 priced outcomes AND
every path_stats cell clears n>=100. This script computes and prints that
verdict every night — it does not gate anything itself (no automated
action should follow from a freeze decision), it just makes the answer
impossible to miss.

Run:
    python -m research.scripts.v8_inputs_nightly
"""

import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_V8_INPUTS_PATH = Path(__file__).parent.parent.parent / "docs" / "V8_INPUTS.md"
_CLEAN_N_TARGET = 2500
_PATH_STATS_MIN_N = 100


def _run(module: str, extra_args: list[str]) -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-m", module] + extra_args,
            capture_output=True, text=True, timeout=300,
        )
        return result.stdout + (("\n" + result.stderr) if result.returncode != 0 else "")
    except Exception as e:
        return f"[v8_inputs_nightly] {module} failed to run: {e}"


def _extract_section(text: str, start_marker: str, end_marker: str | None) -> str:
    """Slice out one report.py section by its numbered header line."""
    start = text.find(start_marker)
    if start == -1:
        return f"(section not found: {start_marker!r})"
    if end_marker:
        end = text.find(end_marker, start + len(start_marker))
        if end == -1:
            end = len(text)
    else:
        end = len(text)
    return text[start:end].rstrip()


def _clean_n(report_text: str) -> int | None:
    m = re.search(r"\[RC1\] era split:\s*(\d+)\s*clean", report_text)
    return int(m.group(1)) if m else None


def main():
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"v8_inputs_nightly: running report.py + path_stats.py for {date_str}...")

    report_out = _run("research.analysis.report", [])
    path_out = _run("research.analysis.path_stats", ["--min-n", str(_PATH_STATS_MIN_N)])

    clean_n = _clean_n(report_out)
    section7 = _extract_section(report_out, "7. PROGRESS_AT_SIGNAL BUCKETS",
                                 "8. READINESS VERDICTS")
    section8 = _extract_section(report_out, "8. READINESS VERDICTS",
                                 "9. [RF4] REALERT")

    path_insufficient = path_out.count("INSUFFICIENT")
    clean_ready = clean_n is not None and clean_n >= _CLEAN_N_TARGET
    freeze_ready = clean_ready and path_insufficient == 0

    verdict = (
        f"FREEZE GATE (K5): clean_n={clean_n} (target {_CLEAN_N_TARGET}, "
        f"{'MET' if clean_ready else 'NOT MET'})  |  "
        f"path_stats INSUFFICIENT cells={path_insufficient} "
        f"({'ALL CLEAR' if path_insufficient == 0 else 'still blocking'})  |  "
        f"{'>>> READY TO FREEZE <<<' if freeze_ready else 'NOT READY'}"
    )
    print(verdict)

    entry = f"""
---

## [K5 nightly] {date_str}

{verdict}

{section7}

{section8}
"""

    with open(_V8_INPUTS_PATH, "a") as f:
        f.write(entry)
    print(f"v8_inputs_nightly: appended to {_V8_INPUTS_PATH}")


if __name__ == "__main__":
    main()
