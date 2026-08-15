"""
research/v8_clean_cohort.py — V8-FILTER-DERIVATION Phase 1 (FD4):
versioned clean-cohort definition for V8 candidate derivation.

This module defines the filter, not the data — it does not query
Supabase itself. Phase 2's engine applies this definition; Phase 1 only
establishes and versions it, with real counts as evidence (see
docs/RECEIPTS.md's FD-BATCH Phase 1 section and
research/v8_feature_registry.yaml for the field-level detail
behind each gate).

V8_CLEAN_COHORT_VERSION = 1

Gates, applied in order, each one narrowing the previous (live counts as
of 2026-08-15T19:05 UTC, git SHA 62253be, 33120 total research_tokens rows):

  1. chain == "solana"                                    33120 (100%)
  2. progress_data_ok == True                               1029 (3.1%)
  3. pct_change_peak IS NOT NULL (real outcome, not just
     outcome_complete=True — see note below)                 895 (2.7% of total, 87.0% of gate 2)

NOT included as a gate, despite being part of V8's real live rule:
  venue_state_at_signal — this column does not exist in the Supabase
  research_tokens schema at all (confirmed via a live query that raised
  `column research_tokens.venue_state_at_signal does not exist`). V8's
  live gate (memecoin/v8_paper.py:passes_v8_gate) checks progress AND
  venue_state == CURVE_ACTIVE; the historical clean cohort below can
  only replicate the progress half. Any candidate/backtest using this
  cohort is implicitly missing the on-curve/graduated distinction for
  historical rows — a real, unresolved gap, not silently assumed away.
  Fixing this (adding the column going forward, or building a documented
  approximation from progress_at_signal alone) is Phase 2/3 work, not
  done here.

IMPORTANT — outcome_complete=True is NOT the same as "has a usable
outcome": 33017/33120 rows (99.7%) have outcome_complete=True, but only
3419/33120 (10.3%) have a non-null pct_change_peak overall, and within
the progress-qualified subset only 895/1029 (87.0%) do. "Complete"
apparently means the polling schedule finished, not that a price was
ever successfully observed to compute a peak from. Gate 3 above uses
pct_change_peak IS NOT NULL directly, not outcome_complete.

Progress-bucket distribution WITHIN the fully-qualified 895-row cohort
(live query, same audit):
    <50%:    46  (5.1%)
    50-70%:   1  (0.1%)   <- the bucket V8's own gate actually trades in
    70-85%: 335  (37.4%)
    85%+:   513  (57.3%)

This matches (to within a fraction of a percent) an independent,
separately-sourced 850-sample measurement taken earlier in the
V8-architecture investigation (5.3% / 0.2% / 37.6% / 56.8%) — the base
rate is a stable, structural property of the Telegram alert stream, not
sampling noise.
"""

V8_CLEAN_COHORT_VERSION = 1

GATES = [
    {"order": 1, "field": "chain", "condition": "== 'solana'",
     "live_count": 33120, "pct_of_total": 1.00},
    {"order": 2, "field": "progress_data_ok", "condition": "== True",
     "live_count": 1029, "pct_of_total": 0.031},
    {"order": 3, "field": "pct_change_peak", "condition": "IS NOT NULL",
     "live_count": 895, "pct_of_total": 0.027},
]

KNOWN_GAPS = [
    "venue_state_at_signal column does not exist in Supabase schema -- "
    "cannot replicate V8's real on-curve gate against historical rows",
    "creator_holds_pct has 0% coverage in production (0/33120) -- "
    "never successfully populated, cause not yet investigated",
    "smart_money_hit/smart_money_count: SMART_MONEY_NOT_ELIGIBLE_FOR_"
    "HISTORICAL_SELECTION (FD6) -- registry provenance lost, no "
    "smart_wallets_vN.json exists on disk or in git history",
    "path data (logs/research_paths/) has only 27 real forward/"
    "naturalistic files total, and 0 files under backfill/ -- both far "
    "below any n>=100 (or even n>=30) threshold; volume gap unexplained, "
    "not yet resolved",
    "~22.6% of a 1000-row sample showed the same token_address appearing "
    "in multiple independent research_tokens rows (145/645 distinct "
    "mints) -- any future train/validation/holdout split MUST group by "
    "token_address, not treat rows as independent (FD5)",
]

DATA_CUTOFF = "2026-08-15T19:00:16+00:00"   # latest alert_time observed at audit time
AUDIT_GIT_SHA = "62253be"
