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
(live query, 2026-08-15 audit):
    <50%:    46  (5.1%)
    50-70%:   1  (0.1%)
    70-85%: 335  (37.4%)
    85%+:   513  (57.3%)

This matches (to within a fraction of a percent) an independent,
separately-sourced 850-sample measurement taken earlier in the
V8-architecture investigation (5.3% / 0.2% / 37.6% / 56.8%) — the base
rate is a stable, structural property of the Telegram alert stream, not
sampling noise.

P15-1 CORRECTION (2026-08-16, re-verified live): candidate-0's actual
gate is progress<0.70 (not just the 50-70% bucket in isolation) --
Phase 1's own receipt mis-stated "the bucket V8 trades in has 1 row" when
it should have summed <50% + 50-70%. Re-queried live the next day (data
has grown, as expected): <50%=47, 50-70%=1, progress<0.70 combined=48
(direct single-query <0.70 filter also returns 48, consistent). The
PROGRESS half of candidate-0's gate has 48 historical clean rows.
candidate0_full_gate_historical_n (progress<0.70 AND venue_state==
CURVE_ACTIVE) remains UNKNOWN -- venue_state_at_signal was never
persisted historically (see KNOWN_GAPS) and is NOT inferred from dex_id
(dex_id is a proven-unreliable graduation signal, see
research/v8_feature_registry.yaml).

P15-2 CORRECTION: the Jun 21 - Aug 15 date range reported in Phase 1
described the OVERALL 33,120-row research_tokens table, not the clean
cohort. Re-queried the clean cohort's OWN date range directly: it runs
2026-08-03 to 2026-08-16 -- 14 calendar days, not ~2 months. Within
that: the progress<70 rows (48 total) are concentrated in just 7 of
those 14 days (2026-08-09 through 2026-08-15) -- the first 6 days
(08-03 to 08-08) have zero rows in the <50/<70 buckets. The real
regime coverage behind the progress<70 candidate is closer to one week
than two, and entirely within a single continuous stretch, not spread
across independent regimes. Full per-day table in docs/RECEIPTS.md's
Phase 1.5 section.

P15-3 -- SCAFFOLD STATUS, NOT A FROZEN CONSTRAINT: progress<0.70 is
candidate-0's placeholder threshold, inherited from the original
2026-07-30 spec (see docs/RECEIPTS.md's V8-REWIRE section for the full
provenance trace) -- it was never derived from this data and must not
be treated as a settled part of the eventual V8 strategy. Phase 2's
bounded, pre-registered candidate registry MUST evaluate multiple
progress policies, not just tune around 0.70. At minimum, subject to
sample/readiness at evaluation time:
    P0 -- no numerical progress cutoff beyond a valid CURVE_ACTIVE venue
    P1 -- progress < 0.50
    P2 -- progress < 0.70   (current candidate-0)
    P3 -- progress < 0.85
Thresholds beyond these four require a genuine prior/domain rationale
(e.g. an existing research bin boundary), decided BEFORE any holdout is
touched -- never selected by inspecting holdout outcomes.

P15-8 -- explicit non-actions, confirmed: creator_holds_pct was NOT
repaired (still 0% coverage, still excluded -- Phase 1 correctly
classified its source as unavailable at V8's T0 fork; repairing it now
would be coverage-chasing, not something Phase 2 has asked for yet).
smart_money v1 was NOT reused or repaired -- SMART_MONEY_NOT_ELIGIBLE_
FOR_HISTORICAL_SELECTION stands unchanged. Any future smart-money-v2
is an explicitly separate, later experiment, not a continuation of v1.

P15-9 -- Phase 2 precondition split, for Phase 2 itself to reference:
    ENGINE_DESIGN_READY   -- the methodology (candidate registry, exit
        registry, split logic, cost model, leakage guards) can be built
        and frozen NOW, independent of how much clean data currently
        exists. Building the engine does not require the data to already
        be sufficient.
    SELECTION_DATA_READY  -- a specific candidate has enough
        representative, leakage-safe, path-backed evidence to support an
        actual filter DECISION. As of this Phase, clearly NOT met for
        anything beyond the crudest entry-only comparisons: the
        progress<70 population is 48 rows over effectively one week, and
        representative (non-case-control) path coverage is a few dozen
        files total (see docs/RECEIPTS.md's Phase 1.5 section for the
        real path-collection funnel and its root cause).
Phase 2 may proceed to build/freeze methodology
(ENGINE_DESIGN_READY) even though SELECTION_DATA_READY is not yet true.
Phase 3 must not claim an absolute full-strategy EV number until
SELECTION_DATA_READY is separately true for whichever candidate is being
evaluated -- an entry-filter-only ranking (FD-BATCH's
V8_FD_SHORTLIST_READY_EV_PARTIAL status) is legitimate before that point;
a $/day number is not.
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

# P15-1 (2026-08-16 re-verification, superseding the 2026-08-15 Phase 1
# snapshot above -- data grows daily, this is expected)
CANDIDATE0_PROGRESS_HALF_N = 48          # progress_at_signal < 0.70, clean cohort
CANDIDATE0_FULL_GATE_HISTORICAL_N = None  # UNKNOWN -- venue_state_at_signal not persisted historically; never inferred from dex_id

# P15-2 (2026-08-16 re-verification) -- the CLEAN cohort's own date range,
# distinct from the overall 33120-row table's Jun21-Aug15 span
CLEAN_COHORT_DATE_RANGE = {
    "min_alert_time": "2026-08-03T20:09:19+00:00",
    "max_alert_time": "2026-08-16T12:33:36+00:00",
    "unique_calendar_days": 14,
    "n_at_audit": 904,   # grew from 895 (Phase 1) to 904 in ~17h -- outcome resolution catching up, not new bugs
    "progress_lt70_concentrated_in_days": 7,   # 08-09 through 08-15; 08-03..08-08 have zero <70 rows
}

# P15-3 -- pre-registered progress-policy candidates for Phase 2. This
# list is NOT a ranking and NOT a selection -- see the module docstring.
PROGRESS_POLICY_CANDIDATES = [
    {"id": "P0", "rule": "no numerical progress cutoff beyond valid CURVE_ACTIVE venue"},
    {"id": "P1", "rule": "progress_at_signal < 0.50"},
    {"id": "P2", "rule": "progress_at_signal < 0.70", "note": "current candidate-0"},
    {"id": "P3", "rule": "progress_at_signal < 0.85"},
]

KNOWN_GAPS = [
    "venue_state_at_signal: forward persistence now wired (P15-4, "
    "research/tracker.py + research/supabase_schema.sql migration) -- "
    "but the SQL migration must be applied manually (this repo has no "
    "DDL-execution path from application code) before it actually lands "
    "anywhere. Historical rows before that stay NULL/UNKNOWN permanently, "
    "never fabricated.",
    "creator_holds_pct has 0% coverage in production (0/33120) -- "
    "never successfully populated, cause not yet investigated, and "
    "deliberately NOT repaired in this phase (P15-8) since Phase 2 has "
    "not defined a delayed-entry candidate that would need it",
    "smart_money_hit/smart_money_count: SMART_MONEY_NOT_ELIGIBLE_FOR_"
    "HISTORICAL_SELECTION (FD6) -- registry provenance lost, no "
    "smart_wallets_vN.json exists on disk or in git history; NOT reused "
    "or repaired (P15-8) -- a v2 is a separate future experiment",
    "path data (logs/research_paths/) root-caused (P15-5, not just "
    "counted): the daily PumpPortal message budget is exceeded on every "
    "observed real day (23-48% over), silently dropping newly-scheduled "
    "tokens for the rest of the UTC day once hit -- observed as early as "
    "04:15 UTC. This is a known, cost-bounded constraint (raising the "
    "budget spends real SOL) requiring a user decision, not an "
    "unexplained application bug. Now counted going forward "
    "(budget_dropped_tokens) and exposed to the watchdog "
    "(watchdog/checks/path_collection.py, P15-7).",
    "~22.6% of a 1000-row sample showed the same token_address appearing "
    "in multiple independent research_tokens rows (145/645 distinct "
    "mints) -- any future train/validation/holdout split MUST group by "
    "token_address, not treat rows as independent (FD5)",
]

# P15-9 -- Phase 2 precondition split (see docstring for full explanation)
ENGINE_DESIGN_READY_MEANS = (
    "methodology (candidate/exit registries, split logic, cost model, "
    "leakage guards) can be built and frozen now, independent of "
    "current data sufficiency"
)
SELECTION_DATA_READY_MEANS = (
    "a specific candidate has enough representative, leakage-safe, "
    "path-backed evidence to support an actual filter decision -- "
    "NOT currently true for anything beyond crude entry-only comparisons"
)

DATA_CUTOFF = "2026-08-16T12:33:36+00:00"   # latest clean-cohort alert_time observed at P15 audit
AUDIT_GIT_SHA = "62253be"   # Phase 1 baseline; P15 work lands on top of this
