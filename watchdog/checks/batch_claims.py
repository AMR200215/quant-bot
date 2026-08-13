"""watchdog/checks/batch_claims.py — W9: claim-vs-artifact verification.

`tools/batch_verify.py` already does the hard part (per-item GREEN/PARTIAL/
FAIL verdicts against RECEIPTS.md sections, greps, and tests), including
the exact nuance W9 cares about: `_check_receipt()` already distinguishes
"section missing" (FAIL) from "section exists but receipt_complete=false
or commit hash absent" (PARTIAL) from "fully backed" (OK). `verify_batch()`
already returns this as a structured dict — no `--json` flag needed, no
new parallel claims registry needed. `batches/*.yaml` already *is* the
claims registry the design spec asks for.

What this check adds: the thing `batch_verify`'s own CLI hides.
`main()`'s exit code is `1 if any_fail else 0` — PARTIAL items exit 0,
identically to a fully-GREEN batch. A CI script or a human glancing at
"exit code 0" would see no difference between a batch that's 7/7 complete
and one that's 3/7 GREEN + 4/7 PARTIAL (this project's own
`v8_readiness.yaml`, confirmed live, right now, while writing this). This
check makes that distinction a real, continuously-monitored WARN instead
of something that only shows up if someone happens to run `--verbose` and
read the table.

Runs on the slow profile (batch_verify with tests disabled is cheap
— ~0.1s across all current manifests — but this is claim-verification,
not a hot-path liveness signal, so hourly is the right cadence)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from watchdog.checks import CheckResult, STATUS_CRITICAL, STATUS_OK, STATUS_UNKNOWN, STATUS_WARN, cap_severity

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _import_batch_verify():
    """Local import: keeps watchdog importable even if tools/batch_verify.py
    has an import-time error, which then surfaces as an UNKNOWN finding
    (see check_batch_verify) rather than crashing the whole watchdog run."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import tools.batch_verify as bv
    return bv


def check_batch_verify(severity_ceiling: str = "WARN",
                        check_id_prefix: str = "claims.batch") -> list[CheckResult]:
    try:
        bv = _import_batch_verify()
    except Exception as e:
        return [CheckResult(
            check_id=f"{check_id_prefix}._import", status=STATUS_UNKNOWN,
            reason=f"failed to import tools.batch_verify: {e!r}", severity=severity_ceiling,
        )]

    try:
        manifests = bv.find_all_manifests()
    except Exception as e:
        return [CheckResult(
            check_id=f"{check_id_prefix}._discover", status=STATUS_UNKNOWN,
            reason=f"failed to discover batch manifests: {e!r}", severity=severity_ceiling,
        )]

    if not manifests:
        return [CheckResult(check_id=f"{check_id_prefix}._discover", status=STATUS_OK,
                             reason="no batch manifests found in batches/")]

    results: list[CheckResult] = []
    for path in manifests:
        check_id = f"{check_id_prefix}.{path.stem}"
        try:
            result = bv.verify_batch(path, run_tests=False)
        except Exception as e:
            results.append(CheckResult(
                check_id=check_id, status=STATUS_UNKNOWN,
                reason=f"verify_batch raised for {path.name}: {e!r}", severity=severity_ceiling,
            ))
            continue

        green = sum(1 for i in result["items"] if i["verdict"] == "GREEN")
        partial = sum(1 for i in result["items"] if i["verdict"] == "YELLOW" or i["verdict"] == "PARTIAL")
        fail = sum(1 for i in result["items"] if i["verdict"] == "RED" or i["verdict"] == "FAIL")
        total = len(result["items"])

        evidence = {"batch_id": result["batch_id"], "commit": result["commit"],
                    "green": green, "partial": partial, "fail": fail, "total": total,
                    "partial_item_ids": [i["id"] for i in result["items"]
                                          if i["verdict"] in ("YELLOW", "PARTIAL")],
                    "fail_item_ids": [i["id"] for i in result["items"]
                                       if i["verdict"] in ("RED", "FAIL")]}

        if result["has_fail"]:
            status = cap_severity(STATUS_CRITICAL, severity_ceiling)
            reason = (f"batch {result['batch_id']!r}: {fail}/{total} item(s) FAIL "
                       f"({evidence['fail_item_ids']}) — claimed but not backed by evidence")
        elif result["has_partial"]:
            status = cap_severity(STATUS_WARN, severity_ceiling)
            reason = (f"batch {result['batch_id']!r}: {partial}/{total} item(s) PARTIAL "
                       f"({evidence['partial_item_ids']}) — batch_verify's own exit code would "
                       f"be 0 here (PARTIAL doesn't fail the CLI), so this is exactly the gap "
                       f"between 'exit code says fine' and 'actually complete'")
        else:
            status = STATUS_OK
            reason = f"batch {result['batch_id']!r}: {green}/{total} item(s) GREEN"

        results.append(CheckResult(check_id=check_id, status=status, reason=reason,
                                    severity=severity_ceiling, evidence=evidence))

    return results
