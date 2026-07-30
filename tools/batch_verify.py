"""
tools/batch_verify.py — Machine-readable batch verification.

Reads one or all batches/*.yaml manifests and checks:
  greps   — file patterns that must (or must not) match
  tests   — pytest files that must exist AND pass
  receipt — RECEIPTS.md section heading + optional commit-hash freshness

Prints a FINAL TABLE per batch:
  item | greps | tests | receipt | VERDICT

Exits 0 if all items are GREEN or PARTIAL.
Exits 1 if any item is FAIL.

Also maintains batches/.red_since.json so health_monitor can detect >48h stalls.

Usage:
  python tools/batch_verify.py                          # all batches/
  python tools/batch_verify.py batches/rc_closure.yaml  # one manifest
  python tools/batch_verify.py --all                    # explicit all
  python tools/batch_verify.py --no-tests               # skip pytest (CI grep-only)
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    print("pyyaml not installed — run: pip install pyyaml")
    sys.exit(1)

_REPO_ROOT    = Path(__file__).parent.parent
_BATCHES_DIR  = _REPO_ROOT / "batches"
_RED_SINCE    = _BATCHES_DIR / ".red_since.json"
_RECEIPTS     = _REPO_ROOT / "docs" / "RECEIPTS.md"

# ── Status constants ───────────────────────────────────────────────────────────
OK      = "OK"
FAIL    = "FAIL"
PARTIAL = "PARTIAL"
NA      = "n/a"

GREEN   = "GREEN"
RED     = "FAIL"
YELLOW  = "PARTIAL"


# ── Grep check ─────────────────────────────────────────────────────────────────

def _check_greps(greps: list) -> tuple[str, list[str]]:
    """Return (OK|FAIL, [reason, ...])."""
    if not greps:
        return NA, []
    failures = []
    for g in greps:
        path    = _REPO_ROOT / g["file"]
        pattern = g["pattern"]
        must_not = g.get("must_not", False)
        if not path.exists():
            failures.append(f"file not found: {g['file']}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        found = bool(re.search(pattern, text))
        if must_not and found:
            failures.append(f"must_not pattern found in {g['file']}: {pattern!r}")
        elif not must_not and not found:
            failures.append(f"pattern not found in {g['file']}: {pattern!r}")
    return (FAIL if failures else OK), failures


# ── Test check ─────────────────────────────────────────────────────────────────

def _check_tests(tests: list, run_tests: bool) -> tuple[str, list[str]]:
    """Return (OK|FAIL|n/a, [reason, ...])."""
    if not tests:
        return NA, []
    failures = []
    for t in tests:
        path = _REPO_ROOT / t
        if not path.exists():
            failures.append(f"test file not found: {t}")
    if failures:
        return FAIL, failures
    if not run_tests:
        return OK, ["(not executed — --no-tests)"]
    result = subprocess.run(
        [sys.executable, "-m", "pytest"] + [str(_REPO_ROOT / t) for t in tests]
        + ["-q", "--tb=no", "--no-header"],
        capture_output=True, text=True, cwd=_REPO_ROOT
    )
    if result.returncode != 0:
        # Extract failure summary
        lines = (result.stdout + result.stderr).splitlines()
        summary = [l for l in lines if "FAILED" in l or "ERROR" in l or "error" in l.lower()][:3]
        failures.extend(summary or [result.stdout.strip()[-200:]])
        return FAIL, failures
    return OK, []


# ── Receipt check ──────────────────────────────────────────────────────────────

def _check_receipt(section: str, commit: str, complete: bool) -> tuple[str, list[str]]:
    """
    Return (OK|PARTIAL|FAIL, [reason]).

    FAIL    — section heading not found in RECEIPTS.md
    PARTIAL — heading found, but commit hash not in that section (or complete=False)
    OK      — heading found AND commit hash present in section
    """
    if not _RECEIPTS.exists():
        return FAIL, ["docs/RECEIPTS.md not found"]
    text = _RECEIPTS.read_text(encoding="utf-8")

    # Find section heading
    lines = text.splitlines()
    section_start = None
    for i, line in enumerate(lines):
        if line.startswith("##") and section in line:
            section_start = i
            break
    if section_start is None:
        return FAIL, [f"section not found in RECEIPTS.md: {section!r}"]

    # Extract section body (until next ##)
    body_lines = []
    for line in lines[section_start + 1:]:
        if line.startswith("##"):
            break
        body_lines.append(line)
    body = "\n".join(body_lines)

    if not complete:
        return PARTIAL, [f"receipt_complete=false — section exists, pending full data"]

    if commit and commit not in body and commit not in lines[section_start]:
        return PARTIAL, [f"commit {commit!r} not found in section body"]

    return OK, []


# ── Red-since tracker ──────────────────────────────────────────────────────────

def _load_red_since() -> dict:
    if _RED_SINCE.exists():
        try:
            return json.loads(_RED_SINCE.read_text())
        except Exception:
            pass
    return {}


def _save_red_since(data: dict):
    _RED_SINCE.write_text(json.dumps(data, indent=2))


def _update_red_since(batch_id: str, has_fail: bool):
    data = _load_red_since()
    if has_fail:
        if batch_id not in data:
            data[batch_id] = time.time()
    else:
        data.pop(batch_id, None)
    _save_red_since(data)


# ── Verify one manifest ────────────────────────────────────────────────────────

def verify_batch(manifest_path: Path, run_tests: bool = True) -> dict:
    """
    Parse manifest, check all items, return result dict.
    result keys: batch_id, commit, items (list of item dicts), has_fail, has_partial
    """
    raw = yaml.safe_load(manifest_path.read_text())
    batch_id   = raw.get("batch_id", manifest_path.stem)
    commit     = raw.get("commit", "")
    items_raw  = raw.get("items", [])

    items = []
    has_fail    = False
    has_partial = False

    for item_raw in items_raw:
        item_id   = item_raw["id"]
        greps     = item_raw.get("greps", [])
        tests     = item_raw.get("tests", [])
        sect      = item_raw.get("receipts_section", "")
        complete  = item_raw.get("receipt_complete", True)

        grep_status,    grep_reasons    = _check_greps(greps)
        test_status,    test_reasons    = _check_tests(tests, run_tests)
        receipt_status, receipt_reasons = _check_receipt(sect, commit, complete)

        # Determine verdict
        statuses = [s for s in [grep_status, test_status, receipt_status] if s != NA]
        if FAIL in statuses:
            verdict = RED
            has_fail = True
        elif PARTIAL in statuses:
            verdict = YELLOW
            has_partial = True
        else:
            verdict = GREEN

        items.append({
            "id":             item_id,
            "grep_status":    grep_status,
            "grep_reasons":   grep_reasons,
            "test_status":    test_status,
            "test_reasons":   test_reasons,
            "receipt_status": receipt_status,
            "receipt_reasons": receipt_reasons,
            "verdict":        verdict,
        })

    _update_red_since(batch_id, has_fail)

    return {
        "batch_id":   batch_id,
        "commit":     commit,
        "items":      items,
        "has_fail":   has_fail,
        "has_partial": has_partial,
    }


# ── Print table ────────────────────────────────────────────────────────────────

def _verdict_sym(v):
    return {"GREEN": "✓ GREEN", "FAIL": "✗ FAIL ", "PARTIAL": "~ PART "}.get(v, v)


def print_result(result: dict, verbose: bool = False):
    bid = result["batch_id"]
    print(f"\nBATCH: {bid}  (commit: {result['commit'] or 'unknown'})")
    print(f"  {'item':<8}  {'greps':<7}  {'tests':<7}  {'receipt':<8}  VERDICT")
    print(f"  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*8}  {'-------'}")
    for item in result["items"]:
        gs = item["grep_status"]
        ts = item["test_status"]
        rs = item["receipt_status"]
        vd = _verdict_sym(item["verdict"])
        print(f"  {item['id']:<8}  {gs:<7}  {ts:<7}  {rs:<8}  {vd}")
        if verbose:
            for r in item["grep_reasons"] + item["test_reasons"] + item["receipt_reasons"]:
                print(f"           ↳ {r}")

    green   = sum(1 for i in result["items"] if i["verdict"] == GREEN)
    partial = sum(1 for i in result["items"] if i["verdict"] == YELLOW)
    fail    = sum(1 for i in result["items"] if i["verdict"] == RED)
    print(f"\n  SUMMARY: {green} GREEN  {partial} PARTIAL  {fail} FAIL")


# ── Public helper for health_monitor ──────────────────────────────────────────

def get_stale_batches(threshold_hours: float = 48.0) -> list[str]:
    """
    Return list of batch_ids that have been continuously red for > threshold_hours.
    Called by health_monitor — no output, no side effects beyond reading .red_since.
    """
    data     = _load_red_since()
    cutoff   = time.time() - threshold_hours * 3600
    return [bid for bid, since in data.items() if since < cutoff]


def find_all_manifests() -> list[Path]:
    return sorted(_BATCHES_DIR.glob("*.yaml"))


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Verify batch manifests")
    parser.add_argument("manifests", nargs="*",
                        help="Path(s) to manifest YAML (default: all batches/*.yaml)")
    parser.add_argument("--all",      action="store_true", help="Verify all manifests")
    parser.add_argument("--no-tests", action="store_true", help="Skip pytest execution")
    parser.add_argument("--verbose",  action="store_true", help="Show failure reasons")
    args = parser.parse_args()

    paths = []
    if args.manifests:
        paths = [Path(p) for p in args.manifests]
    else:
        paths = find_all_manifests()

    if not paths:
        print("No manifests found in batches/")
        sys.exit(0)

    any_fail = False
    for p in paths:
        if not p.exists():
            print(f"ERROR: manifest not found: {p}")
            any_fail = True
            continue
        result = verify_batch(p, run_tests=not args.no_tests)
        print_result(result, verbose=args.verbose)
        if result["has_fail"]:
            any_fail = True

    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
