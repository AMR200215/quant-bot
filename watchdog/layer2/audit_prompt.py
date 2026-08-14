"""watchdog/layer2/audit_prompt.py — W15 (strict ordering) + W16 (findings).

Two separate model calls, not one prompt with "please read evidence
first" instructions -- structural enforcement, not hopeful phrasing:

  Call 1 (ground truth pass): the model sees ONLY the evidence bundle
  with claims fields stripped (EvidenceBundle.to_prompt_text(hide_claims
  =True)) and is asked to independently describe current state, citing
  EV-IDs for every fact. It cannot see RECEIPTS.md's prose at this point
  -- the claims literally are not in its context yet.

  Call 2 (comparison pass): the model receives its OWN ground-truth
  summary from Call 1, plus the claims text (RECEIPTS.md prose,
  explicitly labeled as assertions, not evidence), and is asked to
  compare and produce findings. By this point it has already committed
  to a ground-truth description before ever seeing what anything claims
  about itself -- the anchoring risk W15 describes ("reads 'deployed ✓'
  and just finds evidence that agrees") has nothing to anchor to yet in
  Call 1, and in Call 2 it's comparing against its own prior analysis,
  not forming one from scratch.

`call_model: Callable[[str], str]` is injected (prompt in, raw text out)
rather than this module depending on the anthropic SDK directly -- keeps
this fully unit-testable with a fake model, and keeps the actual API
wiring (model choice, retries, token accounting) in one place
(run_audit.py) instead of scattered through the prompt logic.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from watchdog.layer2.evidence_bundle import EvidenceBundle

VALID_SEVERITIES = ("INFO", "WARN", "CRITICAL")

GROUND_TRUTH_SYSTEM_PROMPT = """You are auditing a production trading-bot infrastructure from RAW EVIDENCE ONLY.

You have NOT been shown any documentation, README, commit message, or status claim of any kind. Everything below is primary evidence: process state, file contents, timestamps, database rows, command output. Some of it may look incomplete or ambiguous -- that is normal and expected; do not assume the best case.

Your job in this pass is ONLY to describe what is verifiably true right now, based strictly on the evidence below. Do not speculate about intent, history, or whether something is "probably fine." For each fact you state, cite the EV-ID(s) it comes from.

Produce a structured summary covering, where the evidence supports it:
- deployed git SHA and whether the working tree is clean
- which systemd services/timers are active vs inactive vs failed
- cron file validity and recent parser errors, if any
- the watchdog's own most recent run status and any active incidents
- research pipeline queue lag
- anything else the evidence clearly shows, positive or negative

If evidence for something is missing or ambiguous, say so explicitly -- do not fill the gap with an assumption. Cite EV-IDs for every claim you make."""

COMPARISON_SYSTEM_PROMPT = """You previously derived the following ground-truth summary from raw evidence alone, before seeing any documentation or status claims:

---
{ground_truth}
---

You are now shown claims -- assertions from project documentation (e.g. RECEIPTS.md) about what is supposedly true. These are ASSERTIONS, not evidence. A claim saying something is "COMPLETE" or "LIVE_VERIFIED" is not proof it currently is -- only your own ground-truth summary above, grounded in evidence, is proof.

{claims}

Compare the claims against your ground-truth summary. For each place they agree, disagree, or where a claim asserts something your ground truth cannot confirm, produce a finding.

Output STRICT JSON ONLY (no prose outside the JSON), an array of finding objects, each with exactly these fields:
  finding_id: string, e.g. "F1", "F2"
  severity: one of "INFO", "WARN", "CRITICAL"
  component: short string naming what this is about
  claim: the specific claim text being checked, or "" if this finding isn't about a specific claim
  observed_ground_truth: what your evidence-grounded summary actually shows
  expected_ground_truth: what the claim implies should be true
  evidence_ids: array of EV-ID strings this finding is grounded in -- REQUIRED, must be non-empty
  operational_impact: one sentence on what this means in practice
  suggested_next_step: a concrete diagnostic command or action, not a fix
  confidence: one of "high", "medium", "low"

A finding with an empty evidence_ids array is invalid and will be discarded. If everything checked out with no discrepancies, output an empty JSON array: []"""


@dataclass
class Finding:
    finding_id: str
    severity: str
    component: str
    claim: str
    observed_ground_truth: str
    expected_ground_truth: str
    evidence_ids: list[str]
    operational_impact: str
    suggested_next_step: str
    confidence: str
    audit_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def is_valid(self) -> bool:
        return (self.severity in VALID_SEVERITIES
                and bool(self.evidence_ids)
                and bool(self.component))


def run_ground_truth_pass(call_model: Callable[[str], str], bundle: EvidenceBundle) -> str:
    prompt = GROUND_TRUTH_SYSTEM_PROMPT + "\n\n" + bundle.to_prompt_text(hide_claims=True)
    return call_model(prompt)


def run_comparison_pass(call_model: Callable[[str], str], ground_truth: str,
                         bundle: EvidenceBundle) -> list[Finding]:
    claims_text = bundle.to_claims_text()
    prompt = COMPARISON_SYSTEM_PROMPT.format(ground_truth=ground_truth, claims=claims_text)
    raw_response = call_model(prompt)
    return parse_findings(raw_response, audit_id=bundle.audit_id)


def parse_findings(raw_response: str, audit_id: str = "") -> list[Finding]:
    """Never raises -- a parse failure is not a crash, it's zero findings
    plus this being visible to whoever inspects the raw response (the
    orchestrator persists raw_response regardless, see run_audit.py)."""
    text = raw_response.strip()
    # Models sometimes wrap JSON in a ```json fence despite instructions --
    # strip it defensively rather than failing the whole audit over it.
    match = re.search(r"```(?:json)?\s*(\[.*\])\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        raw_items = json.loads(text)
    except Exception:
        return []

    if not isinstance(raw_items, list):
        return []

    findings = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        try:
            f = Finding(
                finding_id=str(item.get("finding_id", "")),
                severity=str(item.get("severity", "")).upper(),
                component=str(item.get("component", "")),
                claim=str(item.get("claim", "")),
                observed_ground_truth=str(item.get("observed_ground_truth", "")),
                expected_ground_truth=str(item.get("expected_ground_truth", "")),
                evidence_ids=list(item.get("evidence_ids", []) or []),
                operational_impact=str(item.get("operational_impact", "")),
                suggested_next_step=str(item.get("suggested_next_step", "")),
                confidence=str(item.get("confidence", "")).lower(),
                audit_id=audit_id,
            )
        except Exception:
            continue
        if f.is_valid():
            findings.append(f)
    return findings
