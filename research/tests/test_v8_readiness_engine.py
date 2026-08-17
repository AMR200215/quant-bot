"""research/tests/test_v8_readiness_engine.py — V8-FILTER-DERIVATION
Phase 2 (P2-11): per-(candidate x exit) data-readiness engine.

Run: python -m pytest research/tests/test_v8_readiness_engine.py -v
"""

import unittest

from research.v8_readiness_engine import (
    ReadinessInputs, assess_readiness, build_readiness_matrix,
    current_baseline0_e0_readiness,
    MIN_ENTRY_N, MIN_UNIQUE_MINTS, MIN_UNIQUE_DAYS, MIN_PATH_N,
    MIN_PATH_COVERAGE_PCT, MIN_SPLIT_BUCKET_N, THRESHOLD_PROVENANCE,
)


def _abundant_inputs(**overrides) -> ReadinessInputs:
    base = dict(
        candidate_id="TEST", exit_id="E0",
        historical_entry_n=MIN_ENTRY_N, forward_venue_qualified_n=MIN_ENTRY_N,
        unique_mints=MIN_UNIQUE_MINTS, unique_days=MIN_UNIQUE_DAYS,
        train_n=MIN_SPLIT_BUCKET_N, validation_n=MIN_SPLIT_BUCKET_N, holdout_n=MIN_SPLIT_BUCKET_N,
        boundary_purged_n=0,
        representative_path_n=MIN_PATH_N, path_coverage_pct=MIN_PATH_COVERAGE_PCT,
        cost_model_available=True, entry_slippage_measured=True,
    )
    base.update(overrides)
    return ReadinessInputs(**base)


class TestThresholdsCited(unittest.TestCase):

    def test_every_threshold_has_a_provenance_note(self):
        for name in ("MIN_ENTRY_N", "MIN_UNIQUE_MINTS", "MIN_UNIQUE_DAYS",
                     "MIN_PATH_N", "MIN_PATH_COVERAGE_PCT", "MIN_SPLIT_BUCKET_N"):
            self.assertIn(name, THRESHOLD_PROVENANCE)
            self.assertTrue(THRESHOLD_PROVENANCE[name])


class TestAssessReadiness(unittest.TestCase):

    def test_fully_ready_at_exactly_the_thresholds(self):
        r = assess_readiness(_abundant_inputs())
        self.assertTrue(r.entry_data_ready)
        self.assertTrue(r.path_data_ready)
        self.assertTrue(r.execution_model_ready)
        self.assertTrue(r.full_eval_ready)
        self.assertEqual(r.reasons, [])

    def test_entry_not_ready_below_min_n(self):
        r = assess_readiness(_abundant_inputs(historical_entry_n=MIN_ENTRY_N - 1))
        self.assertFalse(r.entry_data_ready)
        self.assertFalse(r.full_eval_ready)
        self.assertTrue(any("historical_entry_n" in reason for reason in r.reasons))

    def test_entry_not_ready_below_min_unique_mints(self):
        r = assess_readiness(_abundant_inputs(unique_mints=MIN_UNIQUE_MINTS - 1))
        self.assertFalse(r.entry_data_ready)

    def test_entry_not_ready_below_min_unique_days(self):
        r = assess_readiness(_abundant_inputs(unique_days=MIN_UNIQUE_DAYS - 1))
        self.assertFalse(r.entry_data_ready)

    def test_path_not_ready_below_min_path_n(self):
        r = assess_readiness(_abundant_inputs(representative_path_n=MIN_PATH_N - 1))
        self.assertFalse(r.path_data_ready)
        self.assertFalse(r.full_eval_ready)

    def test_path_not_ready_below_coverage_floor(self):
        r = assess_readiness(_abundant_inputs(path_coverage_pct=MIN_PATH_COVERAGE_PCT - 0.1))
        self.assertFalse(r.path_data_ready)

    def test_execution_model_not_ready_when_unavailable(self):
        r = assess_readiness(_abundant_inputs(cost_model_available=False))
        self.assertFalse(r.execution_model_ready)
        self.assertFalse(r.full_eval_ready)

    def test_confidence_downgraded_when_entry_slippage_unmeasured(self):
        r = assess_readiness(_abundant_inputs(entry_slippage_measured=False))
        self.assertEqual(r.execution_model_confidence, "CONSERVATIVE_ONLY")
        # still execution_model_ready -- a cost model exists, just at lower confidence
        self.assertTrue(r.execution_model_ready)

    def test_confidence_measured_when_everything_present(self):
        r = assess_readiness(_abundant_inputs())
        self.assertEqual(r.execution_model_confidence, "MEASURED")

    def test_degenerate_split_blocks_full_eval_even_if_entry_and_path_ready(self):
        r = assess_readiness(_abundant_inputs(train_n=1))
        self.assertTrue(r.entry_data_ready)
        self.assertTrue(r.path_data_ready)
        self.assertFalse(r.full_eval_ready)

    def test_not_a_single_universal_gate_thresholds_differ_per_dimension(self):
        """P2-11: explicitly not a single blanket n>=100 rule everywhere."""
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
        """Real, current-state check: as of P2-0 being unresolved and no
        path-join query having been run yet, BASELINE-0 x E0 must NOT
        report full_eval_ready=True. If this ever flips true without a
        real query backing it, that's a false-readiness bug."""
        r = current_baseline0_e0_readiness()
        self.assertFalse(r.full_eval_ready)
        self.assertGreater(len(r.reasons), 0)

    def test_baseline0_e0_entry_not_ready_yet(self):
        """48 historical rows (or its current live-grown value) is below
        MIN_ENTRY_N=100 -- must show as not entry_data_ready, not silently
        treated as sufficient."""
        r = current_baseline0_e0_readiness()
        self.assertFalse(r.entry_data_ready)


if __name__ == "__main__":
    unittest.main()
