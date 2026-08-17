"""
research/v8_ipw.py — V8-FILTER-DERIVATION Phase 2 (P2-8): inverse
probability weighting (IPW) DESIGN for the probabilistically-sampled
naturalistic path-collection data.

Per the explicit P2-8 instruction: design IPW support now, but do NOT
apply weighting anywhere yet. This module provides the weight function
and read-only diagnostics; nothing in the live pipeline (research/
peak_tracker.py) or Phase 2's registries/split/replay consumes it.
Applying it is Phase 3+ work, gated on this design being reviewed.

Why IPW is needed at all: research/peak_tracker.py's admission
controller (P16-3) does NOT admit every path-eligible token uniformly --
under budget pressure it probabilistically samples (admission_reason
"sampled_admit"/"sampled_reject", research/peak_tracker.py's
_admission_probability()). A naive average over only the ADMITTED
tokens over-represents whichever hours/conditions had slack budget and
under-represents whichever were under pressure. IPW corrects this by
weighting each admitted observation by 1/P(admitted), so the weighted
sample approximates what would have been observed if every eligible
token had been collected.

LIVE VERIFICATION (2026-08-17, git SHA 77355b1, VPS, read-only query
against logs/research_admission/admission_log.jsonl):
    n=75 total admission decisions logged so far.
    admission_reason mix: under_hourly_pace=25, daily_cap_hard_stop=23,
        sampled_admit=18, sampled_reject=9 -- confirms the
        probabilistic-decay branch (sampled_admit/sampled_reject) DOES
        occur under real operation, not just in theory (relevant to
        P2-13, not just P2-8).
    admitted n=43 (path_admitted=True). ALL 43 have
        path_sampling_probability > 0 -- ZERO admitted rows have
        probability 0.0, which is the precondition this module
        requires before any weight can be computed (a weight of 1/0 is
        undefined; if this precondition ever failed for a real row,
        compute_ipw_weight raises rather than silently producing inf).
    Admitted probabilities range from 0.333 to 1.0 (under_hourly_pace
    rows are always probability=1.0 -- deterministic admission when
    under pace; sampled_admit rows carry the real fractional value).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median

# Explicit, checkable statement of this module's status -- never silently
# assumed by a caller. No code path in this repo currently reads this
# constant to gate behavior; it exists so grepping the codebase proves
# the claim in the module docstring.
IPW_APPLIED_IN_PIPELINE = False


def compute_ipw_weight(path_sampling_probability: float) -> float:
    """weight = 1 / P(admitted). Raises on p<=0 rather than returning
    inf/nan -- a zero-or-negative probability on an ADMITTED observation
    would itself be a data-integrity bug (an event can't be observed if
    its probability of being observed was truly zero), not something to
    silently paper over with a weight."""
    if path_sampling_probability <= 0:
        raise ValueError(
            f"path_sampling_probability={path_sampling_probability} <= 0 for an "
            "admitted observation -- this is a data-integrity bug, not weightable."
        )
    if path_sampling_probability > 1:
        raise ValueError(f"path_sampling_probability={path_sampling_probability} > 1 -- invalid")
    return 1.0 / path_sampling_probability


@dataclass(frozen=True)
class IPWDiagnostics:
    n_admitted: int
    n_zero_probability: int          # must be 0 for weighting to be valid at all
    prob_min: float
    prob_max: float
    prob_mean: float
    prob_median: float
    weight_min: float
    weight_max: float
    weight_mean: float
    unweighted_n_effective: int      # == n_admitted (every unit counted once)
    weighted_n_effective: float      # sum(weights) -- the IPW-implied "true" population size


def diagnose_admission_log(admitted_rows: list[dict]) -> IPWDiagnostics:
    """
    admitted_rows: dicts with at least "path_sampling_probability" --
    e.g. rows from logs/research_admission/admission_log.jsonl filtered
    to path_admitted=True. Pure read-only diagnostic; does not write
    anything or feed into any live decision.

    Reports UNWEIGHTED (n_admitted, unweighted_n_effective) and WEIGHTED
    (weighted_n_effective, weight_* stats) numbers as clearly separate
    fields -- per P2-8, these must never be conflated into one number
    before Phase 3.
    """
    if not admitted_rows:
        raise ValueError("diagnose_admission_log: admitted_rows must not be empty")

    probs = [r["path_sampling_probability"] for r in admitted_rows]
    n_zero = sum(1 for p in probs if p <= 0)

    weights = [compute_ipw_weight(p) for p in probs if p > 0]

    return IPWDiagnostics(
        n_admitted=len(admitted_rows),
        n_zero_probability=n_zero,
        prob_min=min(probs),
        prob_max=max(probs),
        prob_mean=mean(probs),
        prob_median=median(probs),
        weight_min=min(weights) if weights else float("nan"),
        weight_max=max(weights) if weights else float("nan"),
        weight_mean=mean(weights) if weights else float("nan"),
        unweighted_n_effective=len(admitted_rows),
        weighted_n_effective=sum(weights),
    )
