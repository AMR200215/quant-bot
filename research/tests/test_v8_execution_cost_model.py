"""research/tests/test_v8_execution_cost_model.py — V8-FILTER-DERIVATION
Phase 2 (P2-9/FD20): execution cost model.

Run: python -m pytest research/tests/test_v8_execution_cost_model.py -v
"""

import unittest

from research.v8_execution_cost_model import (
    estimate_round_trip_cost, full_cost_matrix, EXECUTION_COST_MODEL_VERSION,
    ENTRY_SLIPPAGE_STATUS, NOTIONALS_USD, SCENARIOS,
    BUY_SLIPPAGE_REVERT_CEILING_PCT, SELL_LADDER_PCTS,
)


class TestExecutionCostModel(unittest.TestCase):

    def test_version_set(self):
        self.assertEqual(EXECUTION_COST_MODEL_VERSION, 1)

    def test_invalid_scenario_raises(self):
        with self.assertRaises(ValueError):
            estimate_round_trip_cost(5.0, "MADE_UP_SCENARIO")

    def test_zero_or_negative_notional_raises(self):
        with self.assertRaises(ValueError):
            estimate_round_trip_cost(0.0, "MEASURED")
        with self.assertRaises(ValueError):
            estimate_round_trip_cost(-5.0, "MEASURED")

    def test_fee_pct_is_notional_sensitive(self):
        """Fixed-SOL priority fee must be a LARGER % hit on a smaller trade."""
        c2 = estimate_round_trip_cost(2.0, "MEASURED")
        c5 = estimate_round_trip_cost(5.0, "MEASURED")
        self.assertEqual(c2.priority_fee_usd, c5.priority_fee_usd)  # same fixed SOL cost
        self.assertGreater(c2.priority_fee_pct_of_notional, c5.priority_fee_pct_of_notional)

    def test_scenario_severity_ordering(self):
        m = estimate_round_trip_cost(5.0, "MEASURED")
        c = estimate_round_trip_cost(5.0, "CONSERVATIVE")
        s = estimate_round_trip_cost(5.0, "STRESS")
        self.assertLess(m.round_trip_cost_pct, c.round_trip_cost_pct)
        self.assertLess(c.round_trip_cost_pct, s.round_trip_cost_pct)

    def test_entry_slippage_unmeasured_outside_stress(self):
        m = estimate_round_trip_cost(5.0, "MEASURED")
        c = estimate_round_trip_cost(5.0, "CONSERVATIVE")
        self.assertEqual(m.entry_slippage_status, ENTRY_SLIPPAGE_STATUS)
        self.assertEqual(c.entry_slippage_status, ENTRY_SLIPPAGE_STATUS)
        self.assertEqual(m.entry_slippage_pct, 0.0)
        self.assertEqual(c.entry_slippage_pct, 0.0)

    def test_stress_entry_slippage_uses_documented_revert_ceiling(self):
        s = estimate_round_trip_cost(5.0, "STRESS")
        self.assertEqual(s.entry_slippage_pct, BUY_SLIPPAGE_REVERT_CEILING_PCT)

    def test_sell_slippage_pulled_from_real_sell_ladder(self):
        m = estimate_round_trip_cost(5.0, "MEASURED")
        c = estimate_round_trip_cost(5.0, "CONSERVATIVE")
        s = estimate_round_trip_cost(5.0, "STRESS")
        self.assertIn(m.sell_slippage_pct, SELL_LADDER_PCTS)
        self.assertIn(c.sell_slippage_pct, SELL_LADDER_PCTS)
        self.assertIn(s.sell_slippage_pct, SELL_LADDER_PCTS)

    def test_full_cost_matrix_covers_both_notionals_all_scenarios(self):
        matrix = full_cost_matrix()
        self.assertEqual(len(matrix), len(NOTIONALS_USD) * len(SCENARIOS))
        pairs = {(c.notional_usd, c.scenario) for c in matrix}
        self.assertEqual(pairs, {(n, s) for n in NOTIONALS_USD for s in SCENARIOS})

    def test_size_handling_label_present(self):
        c = estimate_round_trip_cost(2.0, "MEASURED")
        self.assertIn("LINEAR_SIZE_PROJECTION_ONLY", c.size_handling)
        self.assertIn("PER_NOTIONAL_FEE_MODEL", c.size_handling)


if __name__ == "__main__":
    unittest.main()
