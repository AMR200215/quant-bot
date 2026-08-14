"""watchdog/layer2/evidence_bundle.py — wraps the raw JSON dump from
deploy/layer2/evidence_dump.py into a structured, hashed, evidence-ID'd
bundle (W14).

Every Layer 2 finding must cite one or more evidence IDs (EV001, EV002,
...) from this bundle — the audit model is not allowed to assert a fact
that can't be traced back to one of these. IDs are assigned per top-level
section of the raw dump (host/git/systemd/cron/watchdog/research_pipeline/
receipts_tail) — coarse enough to stay simple, fine enough that a finding
citing "EV004 (watchdog)" or "EV003 (systemd)" is unambiguous about which
part of the evidence it's grounded in.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

# Defense in depth only -- evidence_dump.py is designed to never read
# secrets in the first place. This exists in case a future section
# collector accidentally captures something it shouldn't (e.g. a stray
# env var in a systemctl show dump).
_SECRET_PATTERNS = [
    re.compile(r'(TELEGRAM_BOT_TOKEN["\']?\s*[:=]\s*["\']?)[\w:-]+', re.IGNORECASE),
    re.compile(r'(SOLANA_PRIVATE_KEY["\']?\s*[:=]\s*["\']?)[\w-]+', re.IGNORECASE),
    re.compile(r'(HELIUS_API_KEY["\']?\s*[:=]\s*["\']?)[\w-]+', re.IGNORECASE),
    re.compile(r'(PUMPPORTAL_API_KEY["\']?\s*[:=]\s*["\']?)[\w-]+', re.IGNORECASE),
    re.compile(r'(ANTHROPIC_API_KEY["\']?\s*[:=]\s*["\']?)[\w-]+', re.IGNORECASE),
]

_SECTION_ORDER = ["host", "git", "systemd", "cron", "watchdog", "research_pipeline", "receipts_tail"]

# (section, field) pairs that are CLAIMS, not evidence -- W15's ordering
# requires these be invisible during the ground-truth pass and only
# introduced afterward, explicitly labeled as claims. receipts_tail's
# `tail` is RECEIPTS.md prose (exactly the "deployed ✓" / "LIVE_VERIFIED"
# kind of text W15 warns against anchoring on); its `mtime`/`length` stay
# visible as evidence -- staleness is a fact, the prose asserting
# completeness is a claim.
_CLAIMS_FIELDS = {("receipts_tail", "tail")}


def _redact(text: str) -> str:
    for pat in _SECRET_PATTERNS:
        text = pat.sub(r"\1[REDACTED]", text)
    return text


@dataclass
class EvidenceItem:
    ev_id: str
    section: str
    content: Any
    collected_at: float


@dataclass
class EvidenceBundle:
    audit_id: str
    generated_at: float
    raw_bundle_sha256: str
    items: list[EvidenceItem] = field(default_factory=list)

    def get(self, ev_id: str) -> EvidenceItem | None:
        return next((i for i in self.items if i.ev_id == ev_id), None)

    def to_prompt_text(self, hide_claims: bool = True) -> str:
        """Renders the bundle as text for the audit prompt -- each item
        prefixed with its EV-ID so the model can cite it directly.

        hide_claims=True (the default, used for the ground-truth pass):
        strips any field listed in _CLAIMS_FIELDS -- e.g. RECEIPTS.md's
        prose content is invisible here, only its mtime/length (facts
        about the artifact) remain. hide_claims=False is used to build
        the claims-comparison pass's context, where that same content is
        reintroduced explicitly labeled as a claim, not evidence."""
        lines = [f"# Evidence Bundle {self.audit_id}",
                 f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self.generated_at))}",
                 f"SHA-256 of raw evidence dump: {self.raw_bundle_sha256}", ""]
        for item in self.items:
            lines.append(f"## {item.ev_id} ({item.section})")
            content = item.content
            if hide_claims and isinstance(content, dict):
                content = {k: v for k, v in content.items()
                           if (item.section, k) not in _CLAIMS_FIELDS}
            if isinstance(content, (dict, list)):
                content_str = json.dumps(content, indent=2, default=str)
            else:
                content_str = str(content)
            lines.append(_redact(content_str))
            lines.append("")
        return "\n".join(lines)

    def to_claims_text(self) -> str:
        """The claims-designated content only, clearly labeled -- fed to
        the model in the SECOND pass, after it has already committed to a
        ground-truth summary derived from evidence alone."""
        lines = ["# Claims (RECEIPTS.md and similar -- these are ASSERTIONS, not evidence)", ""]
        found_any = False
        for item in self.items:
            if not isinstance(item.content, dict):
                continue
            for k, v in item.content.items():
                if (item.section, k) in _CLAIMS_FIELDS:
                    found_any = True
                    lines.append(f"## Claim from {item.ev_id} ({item.section}.{k})")
                    lines.append(_redact(str(v)))
                    lines.append("")
        if not found_any:
            lines.append("(no claims content present in this evidence bundle)")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "audit_id": self.audit_id,
            "generated_at": self.generated_at,
            "raw_bundle_sha256": self.raw_bundle_sha256,
            "items": [{"ev_id": i.ev_id, "section": i.section, "content": i.content,
                       "collected_at": i.collected_at} for i in self.items],
        }


def build_evidence_bundle(raw_dump: dict, audit_id: str | None = None) -> EvidenceBundle:
    """raw_dump is the parsed JSON output of deploy/layer2/evidence_dump.py.
    Deterministic EV-ID assignment (section order fixed above) so the same
    raw dump always produces the same IDs -- makes findings reproducible
    across a re-run against the same evidence."""
    audit_id = audit_id or time.strftime("audit-%Y%m%dT%H%M%SZ", time.gmtime())
    raw_bytes = json.dumps(raw_dump, sort_keys=True, default=str).encode()
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    generated_at = raw_dump.get("collected_at", time.time())

    items = []
    n = 0
    sections = list(raw_dump.keys())
    ordered = [s for s in _SECTION_ORDER if s in sections] + [s for s in sections if s not in _SECTION_ORDER]
    for section in ordered:
        if section == "collected_at":
            continue
        n += 1
        items.append(EvidenceItem(
            ev_id=f"EV{n:03d}", section=section, content=raw_dump[section],
            collected_at=generated_at,
        ))

    return EvidenceBundle(audit_id=audit_id, generated_at=generated_at,
                           raw_bundle_sha256=raw_sha, items=items)
