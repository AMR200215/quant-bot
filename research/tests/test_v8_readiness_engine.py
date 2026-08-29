"""research/tests/test_v8_readiness_engine.py — V8-FILTER-DERIVATION
Phase 2 (P2-11) / Phase 2.1 item 1: per-(candidate x exit) data-readiness
engine, with PROGRESS_EVIDENCE_READY vs FULL_ENTRY_RULE_READY split.

Run: python -m pytest research/tests/test_v8_readiness_engine.py -v
"""

import unittest

from research.v8_readiness_engine import (
    ReadinessInputs, assess_readiness, build_readiness_matrix,
    current_baseline0_e0_readiness,
    MIN_ENTRY_N, MIN_UNIQUE_MINTS, MIN_UNIQUE_DAYS, MIN_PATH_N,
    MIN_PATH_COVERAGE_PCT, MIN_SPLIT_BUCKET_N, THRESHOLD_PROVENANCE,
    READINESS_KIND, FUTURE_STATISTICAL_READINESS_CRITERIA,
    MIN_POLL_OUTCOME_N, MIN_POLL_OUTCOME_COVERAGE_PCT,
)


def _abundant_inputs(**overrides) -> ReadinessInputs:
    base = dict(
        candidate_id="TEST", exit_id="E0",
        requires_venue_state=True,
        historical_entry_n=MIN_ENTRY_N, unique_mints=MIN_UNIQUE_MINTS, unique_days=MIN_UNIQUE_DAYS,
        forward_venue_qualified_n=MIN_ENTRY_N,
        venue_qualified_unique_mints=MIN_UNIQUE_MINTS, venue_qualified_unique_days=MIN_UNIQUE_DAYS,
        train_n=MIN_SPLIT_BUCKET_N, validation_n=MIN_SPLIT_BUCKET_N, holdout_n=MIN_SPLIT_BUCKET_N,
        boundary_purged_n=0,
        representative_path_n=MIN_PATH_N, path_coverage_pct=MIN_PATH_COVERAGE_PCT,
        poll_outcome_n=MIN_POLL_OUTCOME_N, poll_outcome_coverage_pct=MIN_POLL_OUTCOME_COVERAGE_PCT,
        cost_model_available=True, entry_slippage_measured=True,
    )
    base.update(overrides)
    return ReadinessInputs(**base)


class TestThresholdsCited(unittest.TestCase):

    def test_every_threshold_has_a_provenance_note(self):
        for name in ("MIN_ENTRY_N", "MIN_UNIQUE_MINTS", "MIN_UNIQUE_DAYS",
                     "MIN_PATH_N", "MIN_PATH_COVERAGE_PCT", "MIN_SPLIT_BUCKET_N",
                     "MIN_POLL_OUTCOME_N", "MIN_POLL_OUTCOME_COVERAGE_PCT"):
            self.assertIn(name, THRESHOLD_PROVENANCE)
            self.assertTrue(THRESHOLD_PROVENANCE[name])

    def test_poll_outcome_floors_reuse_path_floors_verbatim(self):
        """YD2: no new number invented -- the poll-outcome floors must be
        literally identical to the path floors, not independently set."""
        self.assertEqual(MIN_POLL_OUTCOME_N, MIN_PATH_N)
        self.assertEqual(MIN_POLL_OUTCOME_COVERAGE_PCT, MIN_PATH_COVERAGE_PCT)

    def test_readiness_kind_is_sanity_floor_not_statistical_sufficiency(self):
        """Phase 2.1 item 4: these thresholds must never be presented as
        statistically sufficient for selection."""
        self.assertEqual(READINESS_KIND, "ENGINEERING_SANITY_FLOOR")
        self.assertNotIn("SUFFICIEN", READINESS_KIND)

    def test_future_statistical_criteria_documented_and_not_implemented(self):
        expected = {"effective_n_after_ipw", "independent_days_or_blocks",
                    "holdout_confidence_interval", "block_bootstrap_confidence_interval",
                    "profit_concentration_check", "regime_stability_across_subperiods"}
        self.assertEqual(set(FUTURE_STATISTICAL_READINESS_CRITERIA.keys()), expected)
        # none of these are actually implemented yet -- must stay False, not silently flipped true
        self.assertTrue(all(v is False for v in FUTURE_STATISTICAL_READINESS_CRITERIA.values()))


class TestProgressVsFullEntryRuleSplit(unittest.TestCase):
    """Phase 2.1 item 1's core requirement."""

    def test_500_progress_rows_zero_venue_qualified_blocks_full_entry_rule(self):
        r = assess_readiness(_abundant_inputs(
            requires_venue_state=True,
            historical_entry_n=500, unique_mints=200, unique_days=30,
            forward_venue_qualified_n=0, venue_qualified_unique_mints=0, venue_qualified_unique_days=0,
        ))
        self.assertTrue(r.progress_evidence_ready)
        self.assertFalse(r.full_entry_rule_ready)
        self.assertFalse(r.full_eval_ready)
        self.assertTrue(any("forward_venue_qualified_n" in reason for reason in r.reasons))

    def test_unknown_historical_venue_cannot_satisfy_curve_active(self):
        """A candidate requiring venue_state==CURVE_ACTIVE must never be
        marked full_entry_rule_ready from progress-only evidence alone,
        no matter how large historical_entry_n is."""
        r = assess_readiness(_abundant_inputs(
            requires_venue_state=True,
            historical_entry_n=10_000, unique_mints=5_000, unique_days=365,
            forward_venue_qualified_n=0, venue_qualified_unique_mints=0, venue_qualified_unique_days=0,
        ))
        self.assertFalse(r.full_entry_rule_ready)

    def test_forward_venue_qualified_observations_counted_correctly(self):
        """When venue-qualified evidence genuinely clears the bar, full_entry_rule_ready must be True."""
        r = assess_readiness(_abundant_inputs(
            requires_venue_state=True,
            forward_venue_qualified_n=MIN_ENTRY_N,
            venue_qualified_unique_mints=MIN_UNIQUE_MINTS,
            venue_qualified_unique_days=MIN_UNIQUE_DAYS,
        ))
        self.assertTrue(r.full_entry_rule_ready)

    def test_candidate_not_requiring_venue_state_uses_progress_ready_directly(self):
        r = assess_readiness(_abundant_inputs(
            requires_venue_state=False,
            forward_venue_qualified_n=0, venue_qualified_unique_mints=0, venue_qualified_unique_days=0,
        ))
        self.assertTrue(r.progress_evidence_ready)
        self.assertTrue(r.full_entry_rule_ready)  # no venue binding -> equals progress_ready

    def test_full_eval_ready_requires_full_entry_rule_ready_not_just_progress(self):
        r = assess_readiness(_abundant_inputs(
            requires_venue_state=True,
            forward_venue_qualified_n=0, venue_qualified_unique_mints=0, venue_qualified_unique_days=0,
        ))
        self.assertTrue(r.progress_evidence_ready)
        self.assertFalse(r.full_eval_ready)


class TestAssessReadiness(unittest.TestCase):

    def test_fully_ready_at_exactly_the_thresholds(self):
        r = assess_readiness(_abundant_inputs())
        self.assertTrue(r.progress_evidence_ready)
        self.assertTrue(r.full_entry_rule_ready)
        self.assertTrue(r.selection_data_ready)
        self.assertTrue(r.exit_derivation_data_ready)
        self.assertTrue(r.execution_model_ready)
        self.assertTrue(r.full_eval_ready)
        self.assertEqual(r.reasons, [])

    def test_progress_not_ready_below_min_n(self):
        r = assess_readiness(_abundant_inputs(historical_entry_n=MIN_ENTRY_N - 1))
        self.assertFalse(r.progress_evidence_ready)
        self.assertTrue(any("historical_entry_n" in reason for reason in r.reasons))

    def test_progress_not_ready_below_min_unique_mints(self):
        r = assess_readiness(_abundant_inputs(unique_mints=MIN_UNIQUE_MINTS - 1))
        self.assertFalse(r.progress_evidence_ready)

    def test_progress_not_ready_below_min_unique_days(self):
        r = assess_readiness(_abundant_inputs(unique_days=MIN_UNIQUE_DAYS - 1))
        self.assertFalse(r.progress_evidence_ready)

    def test_path_not_ready_below_min_path_n(self):
        r = assess_readiness(_abundant_inputs(representative_path_n=MIN_PATH_N - 1))
        self.assertFalse(r.exit_derivation_data_ready)
        self.assertFalse(r.full_eval_ready)

    def test_path_not_ready_below_coverage_floor(self):
        r = assess_readiness(_abundant_inputs(path_coverage_pct=MIN_PATH_COVERAGE_PCT - 0.1))
        self.assertFalse(r.exit_derivation_data_ready)

    def test_selection_ready_independent_of_path_coverage(self):
        """YD2's core requirement: SELECTION must be able to become True
        even when path coverage (exit-derivation) is nowhere close --
        the whole point of splitting the gate."""
        r = assess_readiness(_abundant_inputs(representative_path_n=0, path_coverage_pct=0.0))
        self.assertFalse(r.exit_derivation_data_ready)
        self.assertTrue(r.selection_data_ready)
        self.assertFalse(r.full_eval_ready)   # still blocked overall by exit-derivation

    def test_selection_not_ready_below_min_poll_outcome_n(self):
        r = assess_readiness(_abundant_inputs(poll_outcome_n=MIN_POLL_OUTCOME_N - 1))
        self.assertFalse(r.selection_data_ready)
        self.assertTrue(any("poll_outcome_n" in reason for reason in r.reasons))

    def test_selection_not_ready_below_poll_outcome_coverage_floor(self):
        r = assess_readiness(_abundant_inputs(poll_outcome_coverage_pct=MIN_POLL_OUTCOME_COVERAGE_PCT - 0.1))
        self.assertFalse(r.selection_data_ready)

    def test_selection_not_ready_without_full_entry_rule_ready(self):
        r = assess_readiness(_abundant_inputs(
            requires_venue_state=True,
            forward_venue_qualified_n=0, venue_qualified_unique_mints=0, venue_qualified_unique_days=0,
        ))
        self.assertFalse(r.full_entry_rule_ready)
        self.assertFalse(r.selection_data_ready)

    def test_selection_not_ready_with_degenerate_split(self):
        r = assess_readiness(_abundant_inputs(train_n=1))
        self.assertFalse(r.selection_data_ready)

    def test_execution_model_not_ready_when_unavailable(self):
        r = assess_readiness(_abundant_inputs(cost_model_available=False))
        self.assertFalse(r.execution_model_ready)
        self.assertFalse(r.full_eval_ready)

    def test_confidence_downgraded_when_entry_slippage_unmeasured(self):
        r = assess_readiness(_abundant_inputs(entry_slippage_measured=False))
        self.assertEqual(r.execution_model_confidence, "CONSERVATIVE_ONLY")
        self.assertTrue(r.execution_model_ready)

    def test_confidence_measured_when_everything_present(self):
        r = assess_readiness(_abundant_inputs())
        self.assertEqual(r.execution_model_confidence, "MEASURED")

    def test_degenerate_split_blocks_full_eval_even_if_entry_and_path_ready(self):
        r = assess_readiness(_abundant_inputs(train_n=1))
        self.assertTrue(r.progress_evidence_ready)
        self.assertTrue(r.exit_derivation_data_ready)
        self.assertFalse(r.full_eval_ready)

    def test_not_a_single_universal_gate_thresholds_differ_per_dimension(self):
        self.assertNotEqual(MIN_UNIQUE_MINTS, MIN_ENTRY_N)
        self.assertNotEqual(MIN_SPLIT_BUCKET_N, MIN_ENTRY_N)
        self.assertNotEqual(MIN_UNIQUE_DAYS, MIN_ENTRY_N)


class TestBuildReadinessMatrix(unittest.TestCase):

    def test_matrix_covers_every_input_pair(self):
        inputs = [_abundant_inputs(candidate_id="A"), _abundant_inputs(candidate_id="B")]
        matrix = build_readiness_matrix(inputs)
        self.assertEqual(len(matrix), 2)
        self.assertEqual({r.candidate_id for r in matrix}, {"A", "B"})


class TestCurrentRealReadiness(unittest.TestCase):

    def test_baseline0_e0_not_full_eval_ready_today(self):
        r = current_baseline0_e0_readiness()
        self.assertFalse(r.full_eval_ready)
        self.assertGreater(len(r.reasons), 0)

    def test_baseline0_e0_full_entry_rule_not_ready_yet(self):
        """Only 1 row currently satisfies the FULL gate (progress<0.70 AND
        venue_state==CURVE_ACTIVE) -- must show as not
        full_entry_rule_ready, not silently treated as sufficient."""
        r = current_baseline0_e0_readiness()
        self.assertFalse(r.full_entry_rule_ready)

    def test_baseline0_e0_requires_venue_state(self):
        from research.v8_candidate_registry import CANDIDATES
        baseline = next(c for c in CANDIDATES if c["candidate_id"] == "BASELINE-0")
        self.assertIn("venue_state_at_signal", baseline["required_features"])


if __name__ == "__main__":
    unittest.main()
