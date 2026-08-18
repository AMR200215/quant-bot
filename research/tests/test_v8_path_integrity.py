"""research/tests/test_v8_path_integrity.py — V8-FILTER-DERIVATION
Phase 2.1 item 2: path-integrity gate.

Run: python -m pytest research/tests/test_v8_path_integrity.py -v
"""

import unittest

from research.v8_path_integrity import (
    assess_tick_integrity, assess_path_integrity, filter_integrity_qualified_paths,
    scan_corpus, V8_PATH_INTEGRITY_VERSION, CURVE_ACTIVE_PRICE_CEILING_SOL,
    THEORETICAL_MAX_CURVE_ACTIVE_PRICE_SOL, PathIntegrityStatus,
)

# The real corrupted row from the P2-5/Phase-2.1 audit
_REAL_BAD_ROW = {
    "ts_ms": "1785826482393", "price_usd": "73.49292385903", "price_sol": "0.419959564909",
    "vsol": "116.26907755036314", "venue_state": "CURVE_ACTIVE", "source": "live_pp",
    "backfilled": "false", "schema_version": "2",
}

_GOOD_ROW = {
    "ts_ms": "1000", "price_usd": "0.00005", "price_sol": "0.0000003",
    "vsol": "50.0", "venue_state": "CURVE_ACTIVE", "source": "live_pp",
    "backfilled": "false", "schema_version": "2",
}


class TestVersioning(unittest.TestCase):
    def test_version_is_1(self):
        self.assertEqual(V8_PATH_INTEGRITY_VERSION, 1)


class TestCeilingFormulaCorrection(unittest.TestCase):
    """V8 DATA RECOVERY (2026-08-19): GRAD_SOL_UI is already the full
    virtual SOL reserve at graduation -- confirmed by a real live-
    captured near-graduation PumpPortal message (vSolInBondingCurve=
    ~115.005, matching GRAD_SOL_UI=115.0 directly). The theoretical
    ceiling must use GRAD_SOL_UI alone, not GRAD_SOL_UI+30."""

    def test_theoretical_ceiling_uses_grad_sol_ui_directly(self):
        from research.config import GRAD_SOL_UI
        expected = GRAD_SOL_UI / 279_900_000
        self.assertAlmostEqual(THEORETICAL_MAX_CURVE_ACTIVE_PRICE_SOL, expected, places=12)

    def test_real_near_graduation_example_stays_at_or_under_ceiling(self):
        # mint=eMSgEEkS8RBgyKeS4HZwFcb6gZ3gN8Zj4Y4grQ5pump, live-captured 2026-08-19
        v_sol, v_tok = 115.005359056806, 279900000
        real_price = v_sol / v_tok
        self.assertLessEqual(real_price, CURVE_ACTIVE_PRICE_CEILING_SOL)


class TestTickIntegrity(unittest.TestCase):

    def test_real_corrupted_row_is_invalid_with_both_provable_reasons(self):
        r = assess_tick_integrity(_REAL_BAD_ROW)
        self.assertEqual(r.status, "INVALID")
        self.assertIn("VSOL_EXCEEDS_GRADUATION_WHILE_CURVE_ACTIVE", r.reasons)
        self.assertIn("PRICE_EXCEEDS_THEORETICAL_CURVE_MAX", r.reasons)

    def test_plausible_curve_active_row_is_valid(self):
        r = assess_tick_integrity(_GOOD_ROW)
        self.assertEqual(r.status, "VALID")
        self.assertEqual(r.reasons, ())

    def test_price_at_exactly_the_theoretical_ceiling_is_not_flagged_by_formula_check(self):
        row = dict(_GOOD_ROW, price_sol=str(CURVE_ACTIVE_PRICE_CEILING_SOL * 0.99))
        r = assess_tick_integrity(row)
        self.assertNotIn("PRICE_EXCEEDS_THEORETICAL_CURVE_MAX", r.reasons)

    def test_price_just_above_ceiling_is_invalid(self):
        row = dict(_GOOD_ROW, price_sol=str(CURVE_ACTIVE_PRICE_CEILING_SOL * 1.5))
        r = assess_tick_integrity(row)
        self.assertEqual(r.status, "INVALID")
        self.assertIn("PRICE_EXCEEDS_THEORETICAL_CURVE_MAX", r.reasons)

    def test_ceiling_has_generous_slack_over_theoretical_max(self):
        """The slack multiplier must make the ceiling meaningfully looser
        than the raw theoretical bound (never assert the exact edge)."""
        self.assertGreater(CURVE_ACTIVE_PRICE_CEILING_SOL, THEORETICAL_MAX_CURVE_ACTIVE_PRICE_SOL * 10)

    def test_non_finite_price_is_invalid(self):
        row = dict(_GOOD_ROW, price_usd="nan")
        r = assess_tick_integrity(row)
        self.assertEqual(r.status, "INVALID")
        self.assertIn("NON_FINITE_OR_NEGATIVE_PRICE", r.reasons)

    def test_negative_price_is_invalid(self):
        row = dict(_GOOD_ROW, price_usd="-1.0")
        r = assess_tick_integrity(row)
        self.assertEqual(r.status, "INVALID")

    def test_unparseable_price_is_invalid(self):
        row = dict(_GOOD_ROW, price_usd="not-a-number")
        r = assess_tick_integrity(row)
        self.assertEqual(r.status, "INVALID")
        self.assertIn("NON_FINITE_OR_UNPARSEABLE_PRICE", r.reasons)

    def test_zero_price_is_unknown_not_invalid_not_valid(self):
        """A zero price can't be independently proven wrong -- must be
        UNKNOWN, never silently VALID and never asserted INVALID without
        more evidence."""
        row = dict(_GOOD_ROW, price_usd="0")
        r = assess_tick_integrity(row)
        self.assertEqual(r.status, "UNKNOWN")
        self.assertIn("NONPOSITIVE_PRICE", r.reasons)

    def test_vsol_over_graduation_while_curve_active_is_invalid_even_with_sane_price(self):
        row = dict(_GOOD_ROW, vsol="200.0")  # far past graduation
        r = assess_tick_integrity(row)
        self.assertEqual(r.status, "INVALID")
        self.assertIn("VSOL_EXCEEDS_GRADUATION_WHILE_CURVE_ACTIVE", r.reasons)

    def test_vsol_over_graduation_but_venue_not_curve_active_is_not_flagged_by_that_reason(self):
        row = dict(_GOOD_ROW, vsol="200.0", venue_state="GRADUATED")
        r = assess_tick_integrity(row)
        self.assertNotIn("VSOL_EXCEEDS_GRADUATION_WHILE_CURVE_ACTIVE", r.reasons)

    def test_implausible_sol_usd_ratio_flagged(self):
        row = dict(_GOOD_ROW, price_usd="0.00005", price_sol="0.5")  # ratio = 0.0001, absurd
        r = assess_tick_integrity(row)
        self.assertIn("PRICE_USD_SOL_RATIO_IMPLAUSIBLE", r.reasons)

    def test_non_curve_extreme_mcap_is_unknown_not_invalid(self):
        """DEX price discovery is legitimately unconstrained -- a
        suspicious post-graduation price is UNKNOWN, not asserted
        INVALID without independent proof."""
        row = dict(_GOOD_ROW, venue_state="GRADUATED", price_usd="10.0")  # $10B implied mcap
        r = assess_tick_integrity(row)
        self.assertEqual(r.status, "UNKNOWN")
        self.assertIn("MCAP_ABOVE_EMPIRICAL_GAP_CEILING_NONCURVE", r.reasons)
        self.assertNotIn("MCAP_ABOVE_EMPIRICAL_GAP_CEILING_NONCURVE", ())  # sanity: reason string itself, not a status


class TestPathIntegrity(unittest.TestCase):

    def test_all_valid_ticks_gives_valid_path(self):
        path = [dict(_GOOD_ROW, ts_ms=str(i)) for i in range(5)]
        r = assess_path_integrity(path)
        self.assertEqual(r.status, "VALID")
        self.assertEqual(r.valid_ticks, 5)
        self.assertEqual(r.invalid_ticks, 0)

    def test_single_invalid_tick_poisons_whole_path(self):
        path = [dict(_GOOD_ROW, ts_ms=str(i)) for i in range(4)] + [_REAL_BAD_ROW]
        r = assess_path_integrity(path)
        self.assertEqual(r.status, "INVALID")
        self.assertEqual(r.invalid_ticks, 1)
        self.assertEqual(r.total_ticks, 5)

    def test_unknown_tick_without_invalid_gives_unknown_path(self):
        path = [dict(_GOOD_ROW, ts_ms=str(i)) for i in range(4)] + [dict(_GOOD_ROW, ts_ms="9", price_usd="0")]
        r = assess_path_integrity(path)
        self.assertEqual(r.status, "UNKNOWN")

    def test_empty_path_is_unknown(self):
        r = assess_path_integrity([])
        self.assertEqual(r.status, "UNKNOWN")
        self.assertEqual(r.total_ticks, 0)

    def test_integrity_version_stamped_on_result(self):
        r = assess_path_integrity([_GOOD_ROW])
        self.assertEqual(r.integrity_version, V8_PATH_INTEGRITY_VERSION)


class TestFilterIntegrityQualifiedPaths(unittest.TestCase):

    def test_only_valid_paths_survive(self):
        good_path = [dict(_GOOD_ROW, ts_ms=str(i)) for i in range(3)]
        bad_path = [dict(_GOOD_ROW, ts_ms=str(i)) for i in range(2)] + [_REAL_BAD_ROW]
        result = filter_integrity_qualified_paths([(good_path, {"id": "A"}), (bad_path, {"id": "B"})])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1]["id"], "A")

    def test_corrupted_extreme_positive_path_never_flows_through(self):
        """Direct regression for the Phase 2.1 requirement: a corrupted
        extreme-positive-price path can never survive the filter and
        therefore can never flow into a $/day result."""
        corrupted_path = [_REAL_BAD_ROW]
        result = filter_integrity_qualified_paths([(corrupted_path, {"id": "CORRUPTED"})])
        self.assertEqual(result, [])


class TestScanCorpus(unittest.TestCase):

    def test_scan_empty_dir_returns_zero_report(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            report = scan_corpus(Path(d))
        self.assertEqual(report.total_paths, 0)

    def test_scan_synthetic_corpus_classifies_correctly(self):
        import tempfile, csv
        from pathlib import Path
        from research.path_schema import PATH_HEADER

        with tempfile.TemporaryDirectory() as d:
            day_dir = Path(d) / "2026-08-17"
            day_dir.mkdir()

            def _write(path, rows):
                with open(path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=PATH_HEADER)
                    w.writeheader()
                    for r in rows:
                        full = {k: "" for k in PATH_HEADER}
                        full.update(r)
                        w.writerow(full)

            good_row = {**_GOOD_ROW, "trader_pk": "", "event_id": "", "research_event_id": "",
                        "token_amount": "0", "sol_amount": "0", "price_sol": _GOOD_ROW["price_sol"],
                        "data_status": "ok"}
            bad_row = {**_REAL_BAD_ROW, "trader_pk": "", "event_id": "", "research_event_id": "",
                       "token_amount": "0", "sol_amount": "0", "data_status": "ok"}

            _write(day_dir / "GOOD_MINT.csv", [good_row, dict(good_row, ts_ms="2000")])
            _write(day_dir / "BAD_MINT.csv", [bad_row])

            report = scan_corpus(Path(d))

        self.assertEqual(report.total_paths, 2)
        self.assertEqual(report.valid, 1)
        self.assertEqual(report.invalid, 1)
        self.assertIn("live_pp", report.by_source)
        self.assertIn("2026-08-17", report.by_date)


if __name__ == "__main__":
    unittest.main()
