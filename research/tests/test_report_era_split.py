"""
test_report_era_split.py — Unit tests for RC1 era segmentation in report.py.

Tests the pure functions _era(), _era_split(), and the helpers that drive
sections 2, 7, and 10.  No Supabase calls — all rows are dicts constructed
in-place.
"""

import unittest

from research.analysis.report import (
    _era,
    _era_split,
    _ERA_CLEAN,
    _ERA_PRERF1,
)


def _row(**kwargs):
    """Build a minimal research_tokens row dict."""
    base = {
        "category": "social_alert_bc",
        "outcome_complete": True,
        "pct_change_peak": None,
    }
    base.update(kwargs)
    return base


class TestEra(unittest.TestCase):

    def test_all_null_is_prerf1(self):
        row = _row()   # no price_source_* or price_status_* set
        self.assertEqual(_era(row), _ERA_PRERF1)

    def test_price_source_t1m_set_is_clean(self):
        row = _row(price_source_t1m="curve_account")
        self.assertEqual(_era(row), _ERA_CLEAN)

    def test_price_source_t30m_set_is_clean(self):
        row = _row(price_source_t30m="dexscreener")
        self.assertEqual(_era(row), _ERA_CLEAN)

    def test_price_status_t3m_failure_reason_is_clean(self):
        # A failure reason (not NULL) means RF1 wrote provenance — row is clean
        row = _row(price_status_t3m="curve_rpc_error")
        self.assertEqual(_era(row), _ERA_CLEAN)

    def test_price_source_none_explicit_is_prerf1(self):
        row = _row(price_source_t1m=None, price_status_t1m=None)
        self.assertEqual(_era(row), _ERA_PRERF1)

    def test_mixed_intervals_any_set_is_clean(self):
        # Only t10m has provenance; rest NULL — still clean
        row = _row(price_source_t10m="jupiter", price_source_t1m=None)
        self.assertEqual(_era(row), _ERA_CLEAN)

    def test_empty_string_price_source_is_clean(self):
        # Empty string is not None — counts as set (RF1 wrote something)
        row = _row(price_source_t1m="")
        self.assertEqual(_era(row), _ERA_CLEAN)


class TestEraSplit(unittest.TestCase):

    def setUp(self):
        self.clean_row  = _row(price_source_t1m="curve_account")
        self.prerf1_row = _row()

    def test_empty_list(self):
        clean, pre = _era_split([])
        self.assertEqual(clean, [])
        self.assertEqual(pre, [])

    def test_all_clean(self):
        rows = [self.clean_row, self.clean_row]
        clean, pre = _era_split(rows)
        self.assertEqual(len(clean), 2)
        self.assertEqual(len(pre), 0)

    def test_all_prerf1(self):
        rows = [self.prerf1_row, self.prerf1_row]
        clean, pre = _era_split(rows)
        self.assertEqual(len(clean), 0)
        self.assertEqual(len(pre), 2)

    def test_mixed(self):
        rows = [self.clean_row, self.prerf1_row, self.clean_row]
        clean, pre = _era_split(rows)
        self.assertEqual(len(clean), 2)
        self.assertEqual(len(pre), 1)

    def test_split_is_exhaustive(self):
        rows = [self.clean_row, self.prerf1_row]
        clean, pre = _era_split(rows)
        self.assertEqual(len(clean) + len(pre), len(rows))

    def test_split_no_overlap(self):
        rows = [self.clean_row, self.prerf1_row]
        clean, pre = _era_split(rows)
        for r in clean:
            self.assertNotIn(r, pre)


class TestEraConstants(unittest.TestCase):
    """Verify the string constants are what the manifest's grep patterns expect."""

    def test_prerf1_constant_value(self):
        self.assertEqual(_ERA_PRERF1, "dex_conditioned_preRF1")

    def test_clean_constant_value(self):
        self.assertEqual(_ERA_CLEAN, "clean")


if __name__ == "__main__":
    unittest.main()
