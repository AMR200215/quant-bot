"""
research/v8_readiness_engine.py — V8-FILTER-DERIVATION Phase 2 (P2-11):
data-readiness engine, one report per (entry candidate x exit candidate)
combination.

NOT a universal n>=100 gate applied blindly everywhere. Each threshold
below is cited to a specific piece of existing project evidence, and the
engine reports WHICH ones. Per the explicit P2-11 instruction: thresholds
must be evidence-specific, never invented low just to force a READY
verdict.

Threshold provenance:
  MIN_ENTRY_N = 100            -- research/analysis/path_stats.py's own
      long-standing --min-n default across every A-H analysis in this
      repo; reused, not reinvented, so this engine's bar matches what
      the rest of the codebase already treats as "enough to report a
      real number" rather than INSUFFICIENT.
  MIN_UNIQUE_MINTS = 50        -- FD5's measured 22.5%/26% same-mint
      duplication rate (Phase 1 audit) means ~74-78% of rows are
      distinct mints; half of MIN_ENTRY_N is a direct, cited
      consequence of that measured rate, not a separate guess.
  MIN_UNIQUE_DAYS = 14         -- P15-2's explicit finding that the
      progress<70 population was concentrated in ~1 week, not 2, and
      flagged that as a real regime-coverage risk ("closer to one week
      than two"). 14 days is the bar P15-2 itself used to describe the
      shortfall -- reused as the actual gate, not picked fresh.
  MIN_PATH_N = 100             -- same path_stats.py convention as
      MIN_ENTRY_N.
  MIN_PATH_COVERAGE_PCT = 50.0 -- a design choice (more than half of
      admitted entries must have a usable path), explicitly NOT claimed
      to be derived from data the way the other thresholds are --
      labeled as such in THRESHOLD_PROVENANCE below.
  MIN_SPLIT_BUCKET_N = 20      -- the quick-check --min-n this session
      itself used when live-querying path_stats.py during the P2-5
      audit; reused for consistency, not re-derived.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

MIN_ENTRY_N = 100
MIN_UNIQUE_MINTS = 50
MIN_UNIQUE_DAYS = 14
MIN_PATH_N = 100
MIN_PATH_COVERAGE_PCT = 50.0
MIN_SPLIT_BUCKET_N = 20

THRESHOLD_PROVENANCE = {
    "MIN_ENTRY_N": "research/analysis/path_stats.py --min-n default (100)",
    "MIN_UNIQUE_MINTS": "half of MIN_ENTRY_N, derived from FD5's measured ~22.5-26% same-mint duplication rate",
    "MIN_UNIQUE_DAYS": "research/v8_clean_cohort.py P15-2's own 'closer to one week than two' regime-coverage finding",
    "MIN_PATH_N": "research/analysis/path_stats.py --min-n default (100)",
    "MIN_PATH_COVERAGE_PCT": "design choice (more than half) -- NOT independently data-derived, unlike the others",
    "MIN_SPLIT_BUCKET_N": "the --min-n=20 quick-check threshold used live during the P2-5 audit query",
}


@dataclass(frozen=True)
class ReadinessInputs:
    candidate_id: str
    exit_id: str
    historical_entry_n: int              # rows passing the entry candidate's PROGRESS half historically
    forward_venue_qualified_n: int       # rows also passing the venue_state half (0 until P2-0 resolves)
    unique_mints: int
    unique_days: int
    train_n: int
    validation_n: int
    holdout_n: int
    boundary_purged_n: int
    representative_path_n: int
    path_coverage_pct: float
    cost_model_available: bool
    entry_slippage_measured: bool        # False while ENTRY_SLIPPAGE_STATUS=UNMEASURED (P2-9)


@dataclass(frozen=True)
class ReadinessReport:
    candidate_id: str
    exit_id: str
    entry_data_ready: bool
    path_data_ready: bool
    execution_model_ready: bool
    full_eval_ready: bool
    reasons: list       # human-readable reasons for whichever flags are False
    execution_model_confidence: str   # "MEASURED" | "CONSERVATIVE_ONLY"


def assess_readiness(inputs: ReadinessInputs) -> ReadinessReport:
    reasons: list[str] = []

    entry_ready = (
        inputs.historical_entry_n >= MIN_ENTRY_N
        and inputs.unique_mints >= MIN_UNIQUE_MINTS
        and inputs.unique_days >= MIN_UNIQUE_DAYS
    )
    if not entry_ready:
        if inputs.historical_entry_n < MIN_ENTRY_N:
            reasons.append(f"historical_entry_n={inputs.historical_entry_n} < MIN_ENTRY_N={MIN_ENTRY_N}")
        if inputs.unique_mints < MIN_UNIQUE_MINTS:
            reasons.append(f"unique_mints={inputs.unique_mints} < MIN_UNIQUE_MINTS={MIN_UNIQUE_MINTS}")
        if inputs.unique_days < MIN_UNIQUE_DAYS:
            reasons.append(f"unique_days={inputs.unique_days} < MIN_UNIQUE_DAYS={MIN_UNIQUE_DAYS}")

    path_ready = (
        inputs.representative_path_n >= MIN_PATH_N
        and inputs.path_coverage_pct >= MIN_PATH_COVERAGE_PCT
    )
    if not path_ready:
        if inputs.representative_path_n < MIN_PATH_N:
            reasons.append(f"representative_path_n={inputs.representative_path_n} < MIN_PATH_N={MIN_PATH_N}")
        if inputs.path_coverage_pct < MIN_PATH_COVERAGE_PCT:
            reasons.append(f"path_coverage_pct={inputs.path_coverage_pct} < MIN_PATH_COVERAGE_PCT={MIN_PATH_COVERAGE_PCT}")

    execution_ready = inputs.cost_model_available
    confidence = "MEASURED" if (inputs.cost_model_available and inputs.entry_slippage_measured) else "CONSERVATIVE_ONLY"
    if not execution_ready:
        reasons.append("cost_model_available=False")

    split_not_degenerate = (
        inputs.train_n >= MIN_SPLIT_BUCKET_N
        and inputs.validation_n >= MIN_SPLIT_BUCKET_N
        and inputs.holdout_n >= MIN_SPLIT_BUCKET_N
    )
    if not split_not_degenerate:
        reasons.append(
            f"split bucket below MIN_SPLIT_BUCKET_N={MIN_SPLIT_BUCKET_N} "
            f"(train={inputs.train_n}, validation={inputs.validation_n}, holdout={inputs.holdout_n})"
        )

    full_ready = entry_ready and path_ready and execution_ready and split_not_degenerate

    return ReadinessReport(
        candidate_id=inputs.candidate_id,
        exit_id=inputs.exit_id,
        entry_data_ready=entry_ready,
        path_data_ready=path_ready,
        execution_model_ready=execution_ready,
        full_eval_ready=full_ready,
        reasons=reasons,
        execution_model_confidence=confidence,
    )


def build_readiness_matrix(inputs_list: list[ReadinessInputs]) -> list[ReadinessReport]:
    return [assess_readiness(i) for i in inputs_list]


# ── Real current-state readiness for BASELINE-0 x E0 (2026-08-17) ──────
#
# Live-cited numbers, not synthetic: research/v8_clean_cohort.py's
# CANDIDATE0_PROGRESS_HALF_N=48 (progress<0.70 clean cohort),
# CLEAN_COHORT_DATE_RANGE's progress_lt70_concentrated_in_days=7,
# CANDIDATE0_FULL_GATE_HISTORICAL_N=None (venue_state_at_signal not yet
# persisted historically -- P2-0 unresolved as of this writing).
# unique_mints/path counts are NOT yet computed by any live query in
# this codebase (would require a dedicated join query this module
# doesn't run itself, consistent with P15-9: Phase 2 builds the engine,
# it doesn't have to already have sufficient data). Reported as 0/unknown
# rather than guessed.
def current_baseline0_e0_readiness() -> ReadinessReport:
    from research.v8_clean_cohort import CANDIDATE0_PROGRESS_HALF_N, CLEAN_COHORT_DATE_RANGE

    inputs = ReadinessInputs(
        candidate_id="BASELINE-0",
        exit_id="E0",
        historical_entry_n=CANDIDATE0_PROGRESS_HALF_N,
        forward_venue_qualified_n=0,   # P2-0 unresolved -- venue_state_at_signal not yet in production schema
        unique_mints=0,                # not yet computed by any live query -- reported honestly, not guessed
        unique_days=CLEAN_COHORT_DATE_RANGE["progress_lt70_concentrated_in_days"],
        train_n=0, validation_n=0, holdout_n=0, boundary_purged_n=0,   # split not yet run against real data
        representative_path_n=0,       # not yet computed
        path_coverage_pct=0.0,
        cost_model_available=True,     # research/v8_execution_cost_model.py exists (P2-9)
        entry_slippage_measured=False,  # ENTRY_SLIPPAGE_STATUS=UNMEASURED_ENTRY_SLIPPAGE (P2-9)
    )
    return assess_readiness(inputs)
