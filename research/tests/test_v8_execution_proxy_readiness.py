"""research/tests/test_v8_execution_proxy_readiness.py — V8 READINESS
MONITOR CORRECTION batch item 3: candidate-specific execution-proxy
readiness.

Run: python -m pytest research/tests/test_v8_execution_proxy_readiness.py -v
"""

import unittest

from research.v8_execution_proxy_readiness import (
    assess_execution_proxy_readiness, EXECUTION_PROXY_MIN_N, EXECUTION_PROXY_MIN_COVERAGE_PCT,
    EXECUTION_PROXY_THRESHOLD_PROVENANCE,
)


class TestAssessExecutionProxyReadiness(unittest.TestCase):

    def test_one_observation_is_never_sufficient(self):
        """The exact bug being fixed: entry_slippage_measured used to be
        execution_proxy_total > 0."""
        result = assess_execution_proxy_readiness(eligible_n=1000, observed_n=1, unique_days=1)
        self.assertFalse(result.ready)

    def test_ready_when_both_floors_cleared(self):
        result = assess_execution_proxy_readiness(
            eligible_n=EXECUTION_PROXY_MIN_N, observed_n=EXECUTION_PROXY_MIN_N, unique_days=10)
        self.assertTrue(result.ready)
        self.assertEqual(result.reasons, [])

    def test_not_ready_below_min_n_even_with_full_coverage(self):
        result = assess_execution_proxy_readiness(eligible_n=50, observed_n=50, unique_days=5)
        self.assertFalse(result.ready)  # observed_n=50 < EXECUTION_PROXY_MIN_N=100

    def test_not_ready_below_min_coverage_even_with_enough_n(self):
        result = assess_execution_proxy_readiness(eligible_n=1000, observed_n=150, unique_days=10)
        # coverage = 15%, below 50% floor, even though observed_n=150 > 100
        self.assertFalse(result.ready)

    def test_zero_eligible_gives_zero_coverage_not_error(self):
        result = assess_execution_proxy_readiness(eligible_n=0, observed_n=0, unique_days=0)
        self.assertEqual(result.coverage_pct, 0.0)
        self.assertFalse(result.ready)

    def test_thresholds_reuse_existing_readiness_engine_conventions(self):
        from research.v8_readiness_engine import MIN_ENTRY_N, MIN_PATH_COVERAGE_PCT
        self.assertEqual(EXECUTION_PROXY_MIN_N, MIN_ENTRY_N)
        self.assertEqual(EXECUTION_PROXY_MIN_COVERAGE_PCT, MIN_PATH_COVERAGE_PCT)

    def test_provenance_documented_for_both_thresholds(self):
        self.assertIn("EXECUTION_PROXY_MIN_N", EXECUTION_PROXY_THRESHOLD_PROVENANCE)
        self.assertIn("EXECUTION_PROXY_MIN_COVERAGE_PCT", EXECUTION_PROXY_THRESHOLD_PROVENANCE)


if __name__ == "__main__":
    unittest.main()
