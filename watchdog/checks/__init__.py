"""Check implementations. Each check is a plain function returning a list
of CheckResult — no model calls anywhere in this package (Layer 1 is
deterministic by design)."""

from dataclasses import dataclass, field
from typing import Any

STATUS_OK = "OK"
STATUS_WARN = "WARN"
STATUS_CRITICAL = "CRITICAL"
STATUS_UNKNOWN = "UNKNOWN"
STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"

VALID_STATUSES = frozenset({
    STATUS_OK, STATUS_WARN, STATUS_CRITICAL, STATUS_UNKNOWN, STATUS_NOT_APPLICABLE,
})


@dataclass
class CheckResult:
    check_id: str
    status: str
    reason: str = ""
    subject: str = ""
    severity: str = "WARN"          # severity to use IF this fires as an incident
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.status in VALID_STATUSES, f"invalid status: {self.status}"


_RANK = {STATUS_OK: 0, STATUS_NOT_APPLICABLE: 0, STATUS_UNKNOWN: 1, STATUS_WARN: 2, STATUS_CRITICAL: 3}


def cap_severity(computed_status: str, ceiling: str) -> str:
    """Caps a computed status at a job's configured severity ceiling.
    OK/NOT_APPLICABLE/UNKNOWN pass through unchanged -- only WARN/CRITICAL
    findings can be capped down (e.g. a job registered at severity=WARN
    never escalates past WARN even if the underlying finding is as
    deterministic as a malformed cron file)."""
    if computed_status not in (STATUS_WARN, STATUS_CRITICAL):
        return computed_status
    return computed_status if _RANK[computed_status] <= _RANK.get(ceiling, 3) else ceiling
