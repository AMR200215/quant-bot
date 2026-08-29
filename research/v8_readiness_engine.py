"""
research/v8_readiness_engine.py — V8-FILTER-DERIVATION Phase 2 (P2-11),
hardened Phase 2.1 item 1 + item 4: data-readiness engine, one report
per (entry candidate x exit candidate) combination.

PHASE 2.1 CORRECTION (2026-08-17): the original version accepted
forward_venue_qualified_n as an input field but never actually used it
in entry_ready/full_eval_ready -- meaning a candidate whose binding rule
requires venue_state_at_signal == CURVE_ACTIVE could report readiness
based ONLY on its progress-only historical count, with the venue-state
half silently unchecked. Every frozen v1 candidate (BASELINE-0, P0, P1,
P3) requires venue_state_at_signal (confirmed in
research/tests/test_v8_entry_alignment.py's
test_all_frozen_v1_candidates_require_venue_state_t0_capture) -- so this
was a real gap, not a hypothetical one.

Fixed by splitting readiness into two distinct, separately-reported
flags, never conflated:

  PROGRESS_EVIDENCE_READY -- the progress-only historical population is
      large/diverse enough (same thresholds as before). This is real
      evidence about SOMETHING, but for any candidate that also
      requires venue_state_at_signal, it does NOT mean the candidate's
      actual binding rule has enough evidence.
  FULL_ENTRY_RULE_READY -- for candidates that bind on venue_state, this
      requires the CONTEMPORANEOUSLY OBSERVED, VENUE-QUALIFIED
      population (forward_venue_qualified_n and its own unique-mints/
      unique-days counts, never the progress-only ones) to independently
      clear the same thresholds. Historical rows with unknown venue
      state (from before P2-0's schema fix) can NEVER satisfy this --
      venue state is never inferred from dex_id (research/
      v8_feature_registry.yaml's dex_id entry: "PROVEN UNRELIABLE").
      For a candidate that does NOT bind on venue_state at all,
      FULL_ENTRY_RULE_READY simply equals PROGRESS_EVIDENCE_READY (there
      is no second condition to separately satisfy).

full_eval_ready now depends on FULL_ENTRY_RULE_READY, not
PROGRESS_EVIDENCE_READY -- the readiness matrix can no longer return
FULL_EVAL_READY for a strategy whose binding features were never
actually observed for the claimed sample.

PHASE 2.1 CORRECTION (item 4): the numeric floors below (MIN_ENTRY_N,
MIN_UNIQUE_MINTS, MIN_UNIQUE_DAYS, MIN_SPLIT_BUCKET_N,
MIN_PATH_COVERAGE_PCT) are ENGINEERING SANITY FLOORS -- preconditions
below which a number is definitely too thin to report at all -- NOT a
claim of statistical sufficiency for filter selection. 14 days is not
"enough" merely because Phase 1.5 happened to observe a 14-day cohort;
n=20 is not "enough" merely because an earlier quick-check used
--min-n=20. Real Phase-3 selection readiness needs to additionally
depend on observed uncertainty/stability -- effective n (after
IPW/de-duplication), independent days/blocks (not just calendar-day
count), holdout confidence intervals, block-bootstrap CIs, profit
concentration (is the result driven by one outlier trade), and regime
stability across sub-periods. None of that is implemented here --
FUTURE_STATISTICAL_READINESS_CRITERIA documents what's still missing so
it isn't silently forgotten, and READINESS_KIND labels every threshold
below as a precondition, not a sufficiency claim.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_ENTRY_N = 100
MIN_UNIQUE_MINTS = 50
MIN_UNIQUE_DAYS = 14
MIN_PATH_N = 100
MIN_PATH_COVERAGE_PCT = 50.0
MIN_SPLIT_BUCKET_N = 20

# YD-BATCH item YD2 (docs/READINESS_RESCOPE_PROPOSAL.md, 2026-08-29):
# SELECTION (entry-EV) readiness is keyed to poll-outcome coverage
# (research/outcome_poller.py -- polls price via Helius curve-account
# reads / DexScreener on a fixed T+1m/3m/.../20m schedule, independent
# of whether the token ever trades on PumpPortal at all), NOT path
# coverage -- entry EV never reads a path file. These are the SAME
# numeric floors already established for path coverage, reused
# verbatim, not re-derived: entry EV and exit-tuning both need "more
# than half of a real (>=100) sample", the specific data source
# differs, the bar does not.
MIN_POLL_OUTCOME_N = MIN_PATH_N
MIN_POLL_OUTCOME_COVERAGE_PCT = MIN_PATH_COVERAGE_PCT

READINESS_KIND = "ENGINEERING_SANITY_FLOOR"  # applies to every threshold below -- NOT "STATISTICAL_SUFFICIENCY"

THRESHOLD_PROVENANCE = {
    "MIN_ENTRY_N": "research/analysis/path_stats.py --min-n default (100)",
    "MIN_UNIQUE_MINTS": "half of MIN_ENTRY_N, derived from FD5's measured ~22.5-26% same-mint duplication rate",
    "MIN_UNIQUE_DAYS": "research/v8_clean_cohort.py P15-2's own 'closer to one week than two' regime-coverage finding",
    "MIN_PATH_N": "research/analysis/path_stats.py --min-n default (100)",
    "MIN_PATH_COVERAGE_PCT": "design choice (more than half) -- NOT independently data-derived, unlike the others",
    "MIN_SPLIT_BUCKET_N": "the --min-n=20 quick-check threshold used live during the P2-5 audit query",
    "MIN_POLL_OUTCOME_N": "== MIN_PATH_N, reused verbatim (YD2) -- not independently re-derived",
    "MIN_POLL_OUTCOME_COVERAGE_PCT": "== MIN_PATH_COVERAGE_PCT, reused verbatim (YD2) -- not independently re-derived",
}

# Phase-3 statistical-sufficiency criteria this engine does NOT yet
# implement -- listed explicitly so "we have enough data" is never
# claimed on engineering-floor grounds alone. Each is False until built.
FUTURE_STATISTICAL_READINESS_CRITERIA = {
    "effective_n_after_ipw": False,       # de-duplicated/weighted n, not raw row count
    "independent_days_or_blocks": False,  # block-level independence, not just distinct calendar days
    "holdout_confidence_interval": False,
    "block_bootstrap_confidence_interval": False,
    "profit_concentration_check": False,  # is the result driven by one outlier trade
    "regime_stability_across_subperiods": False,
}


@dataclass(frozen=True)
class ReadinessInputs:
    candidate_id: str
    exit_id: str
    requires_venue_state: bool           # does this candidate's rule bind on venue_state_at_signal?

    # Progress-only historical population (venue state possibly unknown/pre-P2-0).
    historical_entry_n: int
    unique_mints: int
    unique_days: int

    # Venue-qualified population: CONTEMPORANEOUSLY OBSERVED venue_state_at_signal
    # matching the candidate's requirement (e.g. CURVE_ACTIVE). Never inferred from
    # dex_id or any other proxy -- a row with unknown historical venue state does
    # not belong in this count, full stop.
    forward_venue_qualified_n: int
    venue_qualified_unique_mints: int
    venue_qualified_unique_days: int

    train_n: int
    validation_n: int
    holdout_n: int
    boundary_purged_n: int
    representative_path_n: int
    path_coverage_pct: float
    # YD2: entry-EV outcome coverage from research/outcome_poller.py --
    # never a path file. poll_outcome_n/pct must be computed on
    # train+validation rows ONLY (holdout never touched), same
    # discipline as every other count in this module.
    poll_outcome_n: int
    poll_outcome_coverage_pct: float
    cost_model_available: bool
    entry_slippage_measured: bool        # False while ENTRY_SLIPPAGE_STATUS=UNMEASURED (P2-9)


@dataclass(frozen=True)
class ReadinessReport:
    candidate_id: str
    exit_id: str
    progress_evidence_ready: bool
    full_entry_rule_ready: bool
    selection_data_ready: bool        # YD2: full_entry_rule_ready + poll-outcome coverage + non-degenerate split
    exit_derivation_data_ready: bool  # YD2 rename of the old path_data_ready -- still path-keyed, unchanged floor
    execution_model_ready: bool
    full_eval_ready: bool
    reasons: list       # human-readable reasons for whichever flags are False
    execution_model_confidence: str   # "MEASURED" | "CONSERVATIVE_ONLY"


def assess_readiness(inputs: ReadinessInputs) -> ReadinessReport:
    reasons: list[str] = []

    progress_ready = (
        inputs.historical_entry_n >= MIN_ENTRY_N
        and inputs.unique_mints >= MIN_UNIQUE_MINTS
        and inputs.unique_days >= MIN_UNIQUE_DAYS
    )
    if not progress_ready:
        if inputs.historical_entry_n < MIN_ENTRY_N:
            reasons.append(f"historical_entry_n={inputs.historical_entry_n} < MIN_ENTRY_N={MIN_ENTRY_N}")
        if inputs.unique_mints < MIN_UNIQUE_MINTS:
            reasons.append(f"unique_mints={inputs.unique_mints} < MIN_UNIQUE_MINTS={MIN_UNIQUE_MINTS}")
        if inputs.unique_days < MIN_UNIQUE_DAYS:
            reasons.append(f"unique_days={inputs.unique_days} < MIN_UNIQUE_DAYS={MIN_UNIQUE_DAYS}")

    if inputs.requires_venue_state:
        full_entry_ready = (
            inputs.forward_venue_qualified_n >= MIN_ENTRY_N
            and inputs.venue_qualified_unique_mints >= MIN_UNIQUE_MINTS
            and inputs.venue_qualified_unique_days >= MIN_UNIQUE_DAYS
        )
        if not full_entry_ready:
            reasons.append(
                f"forward_venue_qualified_n={inputs.forward_venue_qualified_n} < MIN_ENTRY_N={MIN_ENTRY_N} "
                "-- venue_state_at_signal must be contemporaneously observed; historical rows with "
                "unknown venue state never satisfy this, and it is never inferred from dex_id"
            )
            if inputs.venue_qualified_unique_mints < MIN_UNIQUE_MINTS:
                reasons.append(f"venue_qualified_unique_mints={inputs.venue_qualified_unique_mints} "
                                f"< MIN_UNIQUE_MINTS={MIN_UNIQUE_MINTS}")
            if inputs.venue_qualified_unique_days < MIN_UNIQUE_DAYS:
                reasons.append(f"venue_qualified_unique_days={inputs.venue_qualified_unique_days} "
                                f"< MIN_UNIQUE_DAYS={MIN_UNIQUE_DAYS}")
    else:
        full_entry_ready = progress_ready

    exit_derivation_ready = (
        inputs.representative_path_n >= MIN_PATH_N
        and inputs.path_coverage_pct >= MIN_PATH_COVERAGE_PCT
    )
    if not exit_derivation_ready:
        if inputs.representative_path_n < MIN_PATH_N:
            reasons.append(f"representative_path_n={inputs.representative_path_n} < MIN_PATH_N={MIN_PATH_N}")
        if inputs.path_coverage_pct < MIN_PATH_COVERAGE_PCT:
            reasons.append(f"path_coverage_pct={inputs.path_coverage_pct} < MIN_PATH_COVERAGE_PCT={MIN_PATH_COVERAGE_PCT}")

    # YD2: SELECTION (entry-EV) readiness -- poll-outcome coverage,
    # never a path file. Same floors as exit_derivation_ready, different
    # data source (research/outcome_poller.py).
    poll_outcome_ready = (
        inputs.poll_outcome_n >= MIN_POLL_OUTCOME_N
        and inputs.poll_outcome_coverage_pct >= MIN_POLL_OUTCOME_COVERAGE_PCT
    )
    if not poll_outcome_ready:
        if inputs.poll_outcome_n < MIN_POLL_OUTCOME_N:
            reasons.append(f"poll_outcome_n={inputs.poll_outcome_n} < MIN_POLL_OUTCOME_N={MIN_POLL_OUTCOME_N}")
        if inputs.poll_outcome_coverage_pct < MIN_POLL_OUTCOME_COVERAGE_PCT:
            reasons.append(f"poll_outcome_coverage_pct={inputs.poll_outcome_coverage_pct} "
                            f"< MIN_POLL_OUTCOME_COVERAGE_PCT={MIN_POLL_OUTCOME_COVERAGE_PCT}")

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

    # SELECTION (entry-EV) never depends on path data at all -- gated on
    # full_entry_rule_ready + poll_outcome_ready + a non-degenerate split.
    selection_ready = full_entry_ready and poll_outcome_ready and split_not_degenerate

    # full_eval_ready is gated on FULL_ENTRY_RULE_READY, never on
    # progress_ready alone -- this is the load-bearing fix. It now also
    # requires BOTH the selection and exit-derivation gates, since a
    # complete exit-registry evaluation needs both entry EV and
    # tick-level exit-tuning data.
    full_ready = (
        full_entry_ready and selection_ready and exit_derivation_ready
        and execution_ready and split_not_degenerate
    )

    return ReadinessReport(
        candidate_id=inputs.candidate_id,
        exit_id=inputs.exit_id,
        progress_evidence_ready=progress_ready,
        full_entry_rule_ready=full_entry_ready,
        selection_data_ready=selection_ready,
        exit_derivation_data_ready=exit_derivation_ready,
        execution_model_ready=execution_ready,
        full_eval_ready=full_ready,
        reasons=reasons,
        execution_model_confidence=confidence,
    )


def build_readiness_matrix(inputs_list: list[ReadinessInputs]) -> list[ReadinessReport]:
    return [assess_readiness(i) for i in inputs_list]


# ── Real current-state readiness for BASELINE-0 x E0 (2026-08-17) ──────
#
# Live-cited numbers: research/v8_clean_cohort.py's
# CANDIDATE0_PROGRESS_HALF_N=48 (progress<0.70 clean cohort, Phase-1
# snapshot; grows daily), CLEAN_COHORT_DATE_RANGE's
# progress_lt70_concentrated_in_days=7. Venue-qualified counts are from
# the live post-P2-0 query (2026-08-17): 1 row currently satisfies
# progress<0.70 AND venue_state_at_signal==CURVE_ACTIVE; venue_state
# data itself spans under 1 calendar day so far (P2-0's schema fix only
# just landed). unique_mints for both populations are NOT yet computed
# by any live query in this codebase -- reported as 0, not guessed.
def current_baseline0_e0_readiness() -> ReadinessReport:
    from research.v8_clean_cohort import CANDIDATE0_PROGRESS_HALF_N, CLEAN_COHORT_DATE_RANGE

    inputs = ReadinessInputs(
        candidate_id="BASELINE-0",
        exit_id="E0",
        requires_venue_state=True,
        historical_entry_n=CANDIDATE0_PROGRESS_HALF_N,
        unique_mints=0,
        unique_days=CLEAN_COHORT_DATE_RANGE["progress_lt70_concentrated_in_days"],
        forward_venue_qualified_n=1,      # live query, 2026-08-17, post-P2-0 recovery
        venue_qualified_unique_mints=0,   # not yet computed
        venue_qualified_unique_days=1,    # venue_state coverage spans <1 calendar day so far
        train_n=0, validation_n=0, holdout_n=0, boundary_purged_n=0,
        representative_path_n=0,
        path_coverage_pct=0.0,
        poll_outcome_n=0,          # not yet computed by any live query in this codebase
        poll_outcome_coverage_pct=0.0,
        cost_model_available=True,
        entry_slippage_measured=False,
    )
    return assess_readiness(inputs)
