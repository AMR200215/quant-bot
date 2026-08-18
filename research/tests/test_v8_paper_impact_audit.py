"""research/tests/test_v8_paper_impact_audit.py — V8 DATA RECOVERY
batch item 3: impact classification of existing V8 paper journal rows.

Run: python -m pytest research/tests/test_v8_paper_impact_audit.py -v
"""

import unittest

from research.v8_paper_impact_audit import (
    classify_journal_row, ImpactClassification,
)

# The real 6 rows from logs/memecoin_v8_journal.csv, VPS, 2026-08-19.
_REAL_ROWS = [
    {"id": "V8481249", "token_address": "AdeKS1SbF8QzF5YLgoNhHfc7VDaWJfg6PeRUBPFwpump",
     "entry_price": "13.152426723102602", "exit_price": "6.860301374297039", "pnl_pct": "-47.84"},
    {"id": "V8c65004", "token_address": "5ayszyfWLE4qbp2fJm7gkAXohbzE28ze64rmY16iqQ8z",
     "entry_price": "10.659648611973994", "exit_price": "5.320152672167394", "pnl_pct": "-50.09"},
    {"id": "V8405d17", "token_address": "5ayszyfWLE4qbp2fJm7gkAXohbzE28ze64rmY16iqQ8z",
     "entry_price": "4.614805156981595", "exit_price": "2.6970623191681042", "pnl_pct": "-41.56"},
    {"id": "V896b463", "token_address": "Fctwz2BKy1xX8ubcazCGLBdX3RXHLXcHnaawmotmpump",
     "entry_price": "11.893302307089817", "exit_price": "6.560356468315538", "pnl_pct": "-44.84"},
    {"id": "V883e8f8", "token_address": "H8TMcLVcW1WkhVsAhfVR6Mi6WQnoV3cyJbnpjpUapump",
     "entry_price": "12.549160432675297", "exit_price": "6.7615724360323615", "pnl_pct": "-46.12"},
    {"id": "V8d37e76", "token_address": "CZsG9bUMCxMgJjQKC7BBqPm9tKWo7M7SbHMmXgGSpump",
     "entry_price": "10.980560330036644", "exit_price": "11.935336070155861", "pnl_pct": "8.7"},
]


class TestRealJournalRows(unittest.TestCase):

    def test_all_six_real_rows_classified_pct_pnl_preserved(self):
        """Every real V8 paper trade so far: absolute prices are
        implausible (~$5-16/token, matching the known bug's ~1e6x
        inflation) but pnl_pct is mathematically consistent with
        entry/exit -- the bug's constant multiplicative factor cancels
        in the ratio."""
        for row in _REAL_ROWS:
            impact = classify_journal_row(row)
            self.assertEqual(impact.classification,
                              ImpactClassification.PCT_PNL_PRESERVED_ABSOLUTE_PRICE_BAD,
                              f"row {row['id']} misclassified")
            self.assertFalse(impact.absolute_price_plausible)

    def test_implied_pnl_matches_recorded_within_tolerance_for_all_rows(self):
        for row in _REAL_ROWS:
            impact = classify_journal_row(row)
            recorded = float(row["pnl_pct"])
            self.assertAlmostEqual(impact.implied_pnl_pct_from_prices, recorded, delta=0.5)


class TestClassificationLogic(unittest.TestCase):

    def test_plausible_price_is_unaffected(self):
        row = {"id": "X", "token_address": "M", "entry_price": "0.00005",
               "exit_price": "0.00006", "pnl_pct": "20.0"}
        impact = classify_journal_row(row)
        self.assertEqual(impact.classification, ImpactClassification.UNAFFECTED)

    def test_implausible_price_with_matching_ratio_is_preserved(self):
        row = {"id": "X", "token_address": "M", "entry_price": "10.0",
               "exit_price": "5.0", "pnl_pct": "-50.0"}
        impact = classify_journal_row(row)
        self.assertEqual(impact.classification, ImpactClassification.PCT_PNL_PRESERVED_ABSOLUTE_PRICE_BAD)

    def test_implausible_price_with_mismatched_ratio_is_corrupted(self):
        row = {"id": "X", "token_address": "M", "entry_price": "10.0",
               "exit_price": "5.0", "pnl_pct": "+30.0"}   # inconsistent with the actual -50% ratio
        impact = classify_journal_row(row)
        self.assertEqual(impact.classification, ImpactClassification.PCT_PNL_CORRUPTED)

    def test_missing_fields_are_unknown(self):
        row = {"id": "X", "token_address": "M"}
        impact = classify_journal_row(row)
        self.assertEqual(impact.classification, ImpactClassification.UNKNOWN)

    def test_zero_entry_price_is_unknown_not_a_crash(self):
        row = {"id": "X", "token_address": "M", "entry_price": "0", "exit_price": "5.0", "pnl_pct": "0"}
        impact = classify_journal_row(row)
        self.assertEqual(impact.classification, ImpactClassification.UNKNOWN)

    def test_never_deletes_or_mutates_input_row(self):
        row = {"id": "X", "token_address": "M", "entry_price": "10.0",
               "exit_price": "5.0", "pnl_pct": "-50.0"}
        original = dict(row)
        classify_journal_row(row)
        self.assertEqual(row, original)


if __name__ == "__main__":
    unittest.main()
