"""research/tests/test_v8_statistical_selection.py — V8 DATA RECOVERY
batch item 10: statistical-selection diagnostics, synthetic fixtures
only. No holdout data anywhere in this file.

Run: python -m pytest research/tests/test_v8_statistical_selection.py -v
"""

import unittest

from research.v8_statistical_selection import (
    block_bootstrap_ci, effective_n_after_ipw, profit_concentration,
    regime_stability, candidate_degradation, max_drawdown, max_losing_streak,
    STATISTICAL_SELECTION_VERSION,
)


class TestVersion(unittest.TestCase):
    def test_version_is_1(self):
        self.assertEqual(STATISTICAL_SELECTION_VERSION, 1)


class TestBlockBootstrapCI(unittest.TestCase):

    def test_none_with_fewer_than_two_blocks(self):
        self.assertIsNone(block_bootstrap_ci({"day1": [1.0, 2.0]}))

    def test_none_with_no_data(self):
        self.assertIsNone(block_bootstrap_ci({}))

    def test_ci_contains_point_estimate_for_synthetic_data(self):
        pnl_by_day = {f"day{i}": [10.0 + (i % 3)] for i in range(20)}
        result = block_bootstrap_ci(pnl_by_day, n_bootstrap=500, seed=42)
        self.assertIsNotNone(result)
        self.assertLessEqual(result.ci_lower, result.point_estimate)
        self.assertGreaterEqual(result.ci_upper, result.point_estimate)
        self.assertEqual(result.n_blocks, 20)

    def test_wider_variance_gives_wider_ci(self):
        tight = {f"d{i}": [10.0] for i in range(15)}
        wide = {f"d{i}": [10.0 + (50 if i % 2 == 0 else -50)] for i in range(15)}
        r_tight = block_bootstrap_ci(tight, n_bootstrap=300, seed=1)
        r_wide = block_bootstrap_ci(wide, n_bootstrap=300, seed=1)
        self.assertLess(r_tight.ci_upper - r_tight.ci_lower, r_wide.ci_upper - r_wide.ci_lower)

    def test_deterministic_with_fixed_seed(self):
        pnl_by_day = {f"d{i}": [float(i)] for i in range(10)}
        r1 = block_bootstrap_ci(pnl_by_day, n_bootstrap=200, seed=7)
        r2 = block_bootstrap_ci(pnl_by_day, n_bootstrap=200, seed=7)
        self.assertEqual(r1, r2)


class TestEffectiveNAfterIPW(unittest.TestCase):

    def test_equal_weights_equals_raw_n(self):
        weights = [1.0] * 50
        self.assertAlmostEqual(effective_n_after_ipw(weights), 50.0, places=4)

    def test_unequal_weights_shrinks_effective_n(self):
        equal = [1.0] * 10
        unequal = [10.0] + [0.1] * 9
        self.assertLess(effective_n_after_ipw(unequal), effective_n_after_ipw(equal))

    def test_empty_weights_is_zero(self):
        self.assertEqual(effective_n_after_ipw([]), 0.0)


class TestProfitConcentration(unittest.TestCase):

    def test_single_dominant_winner(self):
        pnls = [100.0, 1.0, 1.0, 1.0, -5.0, -5.0]
        result = profit_concentration(pnls)
        self.assertGreater(result.top_1_contribution_pct, 90.0)

    def test_evenly_spread_profit(self):
        pnls = [10.0, 10.0, 10.0, 10.0, 10.0, -5.0]
        result = profit_concentration(pnls)
        self.assertAlmostEqual(result.top_1_contribution_pct, 20.0, delta=0.1)

    def test_no_profitable_trades_gives_none(self):
        result = profit_concentration([-1.0, -2.0, -3.0])
        self.assertIsNone(result.top_1_contribution_pct)
        self.assertEqual(result.n_profitable_trades, 0)

    def test_empty_input(self):
        result = profit_concentration([])
        self.assertIsNone(result.top_1_contribution_pct)
        self.assertEqual(result.n_trades, 0)


class TestRegimeStability(unittest.TestCase):

    def test_stable_across_similar_weeks(self):
        pnl_by_week = {"w1": [10, 12, 9], "w2": [11, 10, 10], "w3": [9, 11, 12]}
        result = regime_stability(pnl_by_week)
        self.assertEqual(result.stability_label, "STABLE")

    def test_unstable_when_one_week_dominates(self):
        pnl_by_week = {"w1": [100, 100], "w2": [-5, -3], "w3": [1, 2]}
        result = regime_stability(pnl_by_week)
        self.assertEqual(result.stability_label, "UNSTABLE")

    def test_insufficient_weeks(self):
        result = regime_stability({"w1": [10, 12]})
        self.assertEqual(result.stability_label, "INSUFFICIENT_WEEKS")


class TestCandidateDegradation(unittest.TestCase):

    def test_degraded_when_validation_worse(self):
        result = candidate_degradation(train_metric=20.0, validation_metric=5.0)
        self.assertTrue(result.degraded)
        self.assertLess(result.absolute_change, 0)

    def test_not_degraded_when_validation_holds_up(self):
        result = candidate_degradation(train_metric=20.0, validation_metric=22.0)
        self.assertFalse(result.degraded)

    def test_zero_train_metric_gives_none_relative_change(self):
        result = candidate_degradation(train_metric=0.0, validation_metric=5.0)
        self.assertIsNone(result.relative_change_pct)


class TestMaxDrawdown(unittest.TestCase):

    def test_no_decline_gives_zero(self):
        self.assertEqual(max_drawdown([1, 2, 3, 4, 5]), 0.0)

    def test_simple_decline(self):
        self.assertEqual(max_drawdown([0, 10, 5]), 5.0)

    def test_recovers_then_declines_again_tracks_worst(self):
        series = [0, 10, 2, 15, 3]   # drawdowns: 8 (10->2), 12 (15->3)
        self.assertEqual(max_drawdown(series), 12.0)

    def test_empty_series(self):
        self.assertEqual(max_drawdown([]), 0.0)


class TestMaxLosingStreak(unittest.TestCase):

    def test_all_wins_is_zero(self):
        self.assertEqual(max_losing_streak([1, 2, 3]), 0)

    def test_simple_streak(self):
        self.assertEqual(max_losing_streak([1, -1, -1, -1, 2, -1]), 3)

    def test_zero_counts_as_a_loss(self):
        self.assertEqual(max_losing_streak([0, 0, 1]), 2)

    def test_empty_is_zero(self):
        self.assertEqual(max_losing_streak([]), 0)


if __name__ == "__main__":
    unittest.main()
