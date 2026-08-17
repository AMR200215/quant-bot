"""research/tests/test_v8_execution_cost_model.py — V8-FILTER-DERIVATION
Phase 2.1 item 3: execution cost model, CONFIGURED_TOLERANCE vs
ACTUAL_REALIZED_EXECUTION_COST separation.

Run: python -m pytest research/tests/test_v8_execution_cost_model.py -v
"""

import unittest

from research.v8_execution_cost_model import (
    EXECUTION_COST_MODEL_VERSION, EvidenceClass,
    get_configured_tolerance, get_realized_cost_components,
    build_cost_breakdown, full_cost_matrix, NOTIONALS_USD,
    COHORT_MATCHES_V8, LIVE_JOURNAL_N, SOL_USD_REFERENCE_STATUS,
    VENUE_STATE_STRATIFICATION_STATUS,
    SELL_LADDER_TOLERANCE_PCTS, BUY_SLIPPAGE_REVERT_CEILING_PCT,
)


class TestVersion(unittest.TestCase):
    def test_version_is_2(self):
        self.assertEqual(EXECUTION_COST_MODEL_VERSION, 2)


class TestConfiguredToleranceVsRealized(unittest.TestCase):

    def test_configured_tolerance_has_no_evidence_class_field(self):
        """A configured tolerance is a fact about the system's config,
        not a measurement -- it must not carry an evidence_class."""
        tol = get_configured_tolerance()
        self.assertFalse(hasattr(tol, "evidence_class"))
        self.assertEqual(tol.sell_ladder_tolerance_pcts, SELL_LADDER_TOLERANCE_PCTS)
        self.assertEqual(tol.buy_revert_ceiling_pct, BUY_SLIPPAGE_REVERT_CEILING_PCT)

    def test_sell_ladder_never_labeled_measured_realized_slippage(self):
        """The core Phase 2.1 item 3 bug: 35/60/98 must never appear as a
        realized-cost component's value with evidence_class=MEASURED."""
        components = get_realized_cost_components()
        for c in components:
            if c.value in SELL_LADDER_TOLERANCE_PCTS:
                self.fail(f"{c.name} reuses a sell-ladder tolerance value as a realized cost")

    def test_realized_components_each_carry_an_evidence_class(self):
        valid_classes = {EvidenceClass.MEASURED, EvidenceClass.PARTIALLY_MEASURED,
                          EvidenceClass.ASSUMPTION_BOUND, EvidenceClass.UNMEASURED}
        for c in get_realized_cost_components():
            self.assertIn(c.evidence_class, valid_classes)

    def test_unmeasured_components_have_none_value_not_a_fabricated_number(self):
        components = get_realized_cost_components()
        unmeasured = [c for c in components if c.evidence_class == EvidenceClass.UNMEASURED]
        self.assertGreater(len(unmeasured), 0)
        for c in unmeasured:
            self.assertIsNone(c.value)

    def test_hard_stop_overshoot_is_partially_measured_not_measured(self):
        components = get_realized_cost_components()
        overshoot = next(c for c in components if c.name == "hard_stop_overshoot_median_pp")
        self.assertEqual(overshoot.evidence_class, EvidenceClass.PARTIALLY_MEASURED)


class TestCohortMismatch(unittest.TestCase):

    def test_cohort_matches_v8_is_false(self):
        """Explicitly checked, not assumed: the 80 historical live trades
        are v7/v4 social_alert trades, never V8."""
        self.assertFalse(COHORT_MATCHES_V8)

    def test_every_realized_component_flags_cohort_mismatch(self):
        for c in get_realized_cost_components():
            self.assertFalse(c.cohort_matches_v8)
            self.assertTrue(c.note)  # must carry an explanatory note, not a bare False

    def test_venue_state_stratification_status_is_honest(self):
        self.assertEqual(VENUE_STATE_STRATIFICATION_STATUS, "IMPOSSIBLE_WITH_CURRENT_JOURNAL_SCHEMA")

    def test_live_journal_n_is_80(self):
        self.assertEqual(LIVE_JOURNAL_N, 80)


class TestSolUsdConversion(unittest.TestCase):

    def test_sol_usd_reference_labeled_static_not_time_aligned(self):
        self.assertEqual(SOL_USD_REFERENCE_STATUS, "STATIC_ASSUMPTION")


class TestCostBreakdown(unittest.TestCase):

    def test_zero_or_negative_notional_raises(self):
        with self.assertRaises(ValueError):
            build_cost_breakdown(0.0)
        with self.assertRaises(ValueError):
            build_cost_breakdown(-5.0)

    def test_fee_pct_is_notional_sensitive(self):
        c2 = build_cost_breakdown(2.0)
        c5 = build_cost_breakdown(5.0)
        self.assertEqual(c2.priority_fee_usd_at_floor, c5.priority_fee_usd_at_floor)
        self.assertGreater(c2.priority_fee_pct_of_notional, c5.priority_fee_pct_of_notional)

    def test_breakdown_includes_both_configured_and_realized_sections(self):
        c = build_cost_breakdown(5.0)
        self.assertIsNotNone(c.configured_tolerance)
        self.assertGreater(len(c.realized_components), 0)

    def test_full_cost_matrix_covers_both_notionals(self):
        matrix = full_cost_matrix()
        self.assertEqual(len(matrix), len(NOTIONALS_USD))
        self.assertEqual({c.notional_usd for c in matrix}, set(NOTIONALS_USD))

    def test_size_handling_label_present(self):
        c = build_cost_breakdown(2.0)
        self.assertIn("LINEAR_SIZE_PROJECTION_ONLY", c.size_handling)
        self.assertIn("PER_NOTIONAL_FEE_MODEL", c.size_handling)


if __name__ == "__main__":
    unittest.main()
