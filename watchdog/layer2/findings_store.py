"""watchdog/layer2/findings_store.py — W16 (findings output) + W17
(Layer 2 staleness is itself an incident).

Layer 2 runs on an ephemeral GitHub Actions runner, not the VPS -- there
is no long-lived local disk to write to. This module writes into the
checked-out repo working tree; the calling GitHub Actions workflow (not
this module) is responsible for `git add && git commit && git push`
using its own built-in, repo-scoped token, so no new credential is
needed for persistence.

`docs/WATCHDOG_FINDINGS.md`-style human summaries, if ever added, would
be a generated index over this data -- this directory
(logs/watchdog/audits/<audit_id>/) is the authoritative store; automated
runs must never rewrite historical audit directories, only add new ones.

The heartbeat file (logs/watchdog/layer2_heartbeat.json) is what closes
the reciprocal-supervision loop: Layer 1 (running continuously on the
VPS) reads this file locally after its own `git pull` picks up whatever
Layer 2 last committed, and flags staleness if it hasn't moved in too
long. Known limitation, stated plainly rather than engineered around:
the VPS's existing git pull cadence (9am/6pm cron, twice daily) is the
actual visibility latency for Layer 1 noticing a Layer 2 problem -- up to
~12h, not real-time. Fine for a v1; revisit if that latency turns out to
matter in practice.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from watchdog.layer2.audit_prompt import Finding
from watchdog.layer2.evidence_bundle import EvidenceBundle

AUDITS_SUBDIR = "logs/watchdog/audits"
HEARTBEAT_PATH = "logs/watchdog/layer2_heartbeat.json"


def write_audit_artifacts(repo_root: Path, bundle: EvidenceBundle, ground_truth: str,
                           findings: list[Finding], raw_comparison_response: str,
                           status: str = "ok") -> Path:
    """Writes evidence.json, findings.json, findings.md, ground_truth.txt
    into logs/watchdog/audits/<audit_id>/. Never overwrites a prior
    audit_id's directory -- each audit gets its own, append-only."""
    audit_dir = repo_root / AUDITS_SUBDIR / bundle.audit_id
    audit_dir.mkdir(parents=True, exist_ok=True)

    (audit_dir / "evidence.json").write_text(json.dumps(bundle.to_dict(), indent=2, default=str))
    (audit_dir / "ground_truth.txt").write_text(ground_truth)
    (audit_dir / "raw_comparison_response.txt").write_text(raw_comparison_response)

    findings_data = [
        {"finding_id": f.finding_id, "severity": f.severity, "component": f.component,
         "claim": f.claim, "observed_ground_truth": f.observed_ground_truth,
         "expected_ground_truth": f.expected_ground_truth, "evidence_ids": f.evidence_ids,
         "operational_impact": f.operational_impact, "suggested_next_step": f.suggested_next_step,
         "confidence": f.confidence, "audit_id": f.audit_id, "timestamp": f.timestamp}
        for f in findings
    ]
    (audit_dir / "findings.json").write_text(json.dumps(findings_data, indent=2))
    (audit_dir / "findings.md").write_text(_render_findings_md(bundle, findings, status))

    return audit_dir


def _render_findings_md(bundle: EvidenceBundle, findings: list[Finding], status: str) -> str:
    lines = [f"# Layer 2 Audit: {bundle.audit_id}",
             f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(bundle.generated_at))}",
             f"Evidence SHA-256: `{bundle.raw_bundle_sha256}`",
             f"Status: {status}", ""]
    if not findings:
        lines.append("No findings — claims matched ground truth, or no claims content was present.")
        return "\n".join(lines)
    by_severity = {"CRITICAL": [], "WARN": [], "INFO": []}
    for f in findings:
        by_severity.setdefault(f.severity, []).append(f)
    for sev in ("CRITICAL", "WARN", "INFO"):
        for f in by_severity.get(sev, []):
            lines += [
                f"## [{f.severity}] {f.finding_id}: {f.component}",
                f"- **Claim**: {f.claim or '(none)'}",
                f"- **Observed**: {f.observed_ground_truth}",
                f"- **Expected**: {f.expected_ground_truth}",
                f"- **Evidence**: {', '.join(f.evidence_ids)}",
                f"- **Impact**: {f.operational_impact}",
                f"- **Next step**: {f.suggested_next_step}",
                f"- **Confidence**: {f.confidence}",
                "",
            ]
    return "\n".join(lines)


def write_heartbeat(repo_root: Path, bundle: EvidenceBundle, findings: list[Finding],
                     status: str = "ok", now_ts: Optional[float] = None) -> Path:
    now_ts = now_ts if now_ts is not None else time.time()
    path = repo_root / HEARTBEAT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "last_audit_id": bundle.audit_id,
        "last_audit_completed_at": now_ts,
        "status": status,
        "findings_count": len(findings),
        "critical_count": sum(1 for f in findings if f.severity == "CRITICAL"),
        "warn_count": sum(1 for f in findings if f.severity == "WARN"),
        "evidence_sha256": bundle.raw_bundle_sha256,
    }
    path.write_text(json.dumps(data, indent=2))
    return path


def read_heartbeat(repo_root: Path) -> Optional[dict]:
    path = repo_root / HEARTBEAT_PATH
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None
