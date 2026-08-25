"""
N7(b) path_stats additions — unit tests for the pure-logic pieces of
analyses E (peak-mcap), F (conditional continuation), G (buyer velocity),
H (sniper density). No network, no Supabase.

Run: python -m pytest research/tests/test_path_stats_n7b.py -v
"""

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from research.analysis.path_stats import (
    _mcap_zone,
    _first_trough,
    _unique_buyers_by,
    _has_trader_pk_data,
    _MCAP_ZONES,
    _TROUGH_MIN_DEPTH_PCT,
    _integrity_gate,
    _load_path,
    _shakeout_depth_for_target,
    _time_to_target_minutes,
)


def _row(ts_ms, price_usd, side="unknown", trader_pk=""):
    return {"ts_ms": ts_ms, "price_usd": price_usd, "side": side,
            "sol_amount": 0.0, "vsol": 0.0, "source": "live_pp",
            "backfilled": "false", "data_status": "ok", "trader_pk": trader_pk}


class TestMcapZone(unittest.TestCase):

    def test_zones_cover_full_range_without_gaps(self):
        # every zone's hi == next zone's lo (or +inf for the last)
        for i in range(len(_MCAP_ZONES) - 1):
            self.assertEqual(_MCAP_ZONES[i][1], _MCAP_ZONES[i + 1][0])
        self.assertEqual(_MCAP_ZONES[-1][1], float("inf"))

    def test_boundary_values_go_to_upper_zone(self):
        self.assertEqual(_mcap_zone(10_000), "$10-25k")
        self.assertEqual(_mcap_zone(9_999), "<$10k")

    def test_very_large_mcap_falls_in_last_zone(self):
        self.assertEqual(_mcap_zone(10_000_000), "$250k+")

    def test_zero_mcap_in_first_zone(self):
        self.assertEqual(_mcap_zone(0), "<$10k")


class TestFirstTrough(unittest.TestCase):

    def test_no_trough_below_threshold_returns_none(self):
        # 5% pullback only — below the 10% trough threshold
        rows = [_row(0, 100), _row(1000, 110), _row(2000, 105)]
        self.assertIsNone(_first_trough(rows))

    def test_qualifying_trough_detected(self):
        rows = [_row(0, 100), _row(1000, 100), _row(2000, 80), _row(3000, 90)]
        result = _first_trough(rows)
        self.assertIsNotNone(result)
        high_row, trough_row, depth = result
        self.assertEqual(high_row["price_usd"], 100)
        self.assertEqual(trough_row["price_usd"], 80)
        self.assertAlmostEqual(depth, 20.0, places=1)
        self.assertGreaterEqual(depth, _TROUGH_MIN_DEPTH_PCT)

    def test_only_first_qualifying_trough_returned(self):
        # two dips >=10%; must return the first one, not the deepest
        rows = [
            _row(0, 100),
            _row(1000, 85),    # first trough: 15% depth
            _row(2000, 120),   # new running high
            _row(3000, 90),    # second trough: 25% depth from 120 — should NOT be returned
        ]
        _high, trough_row, depth = _first_trough(rows)
        self.assertEqual(trough_row["price_usd"], 85)
        self.assertAlmostEqual(depth, 15.0, places=1)

    def test_empty_rows_returns_none(self):
        self.assertIsNone(_first_trough([]))

    def test_monotonic_rise_returns_none(self):
        rows = [_row(i * 1000, 100 + i * 10) for i in range(5)]
        self.assertIsNone(_first_trough(rows))


class TestBuyerFeatures(unittest.TestCase):

    def test_unique_buyers_by_dedupes_same_trader(self):
        rows = [
            _row(0,    1.0, side="buy",  trader_pk="A"),
            _row(1000, 1.1, side="buy",  trader_pk="A"),   # same trader — not double counted
            _row(2000, 1.2, side="buy",  trader_pk="B"),
            _row(3000, 1.1, side="sell", trader_pk="C"),   # sell — excluded
        ]
        self.assertEqual(_unique_buyers_by(rows, cutoff_ts_ms=3000), 2)

    def test_unique_buyers_by_respects_cutoff(self):
        rows = [
            _row(0,    1.0, side="buy", trader_pk="A"),
            _row(6000, 1.1, side="buy", trader_pk="B"),   # after 5s cutoff
        ]
        self.assertEqual(_unique_buyers_by(rows, cutoff_ts_ms=5000), 1)

    def test_blank_trader_pk_excluded(self):
        rows = [_row(0, 1.0, side="buy", trader_pk="")]
        self.assertEqual(_unique_buyers_by(rows, cutoff_ts_ms=1000), 0)

    def test_has_trader_pk_data_true_when_any_present(self):
        rows = [_row(0, 1.0, trader_pk=""), _row(1000, 1.0, trader_pk="X")]
        self.assertTrue(_has_trader_pk_data(rows))

    def test_has_trader_pk_data_false_for_pre_n7a_path(self):
        rows = [_row(0, 1.0, trader_pk=""), _row(1000, 1.0, trader_pk="")]
        self.assertFalse(_has_trader_pk_data(rows))


class TestIntegrityGate(unittest.TestCase):
    """--valid-only: the price-outlier-cleaning pass research/
    v8_exit_registry.py's audit said analyses B and F were blocked on,
    now available by reusing the already-tested v8_path_integrity.py
    classifier (not a new heuristic)."""

    def _write_csv(self, path, rows):
        import csv
        from research.path_schema import PATH_HEADER
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=PATH_HEADER)
            w.writeheader()
            for r in rows:
                full = {k: "" for k in PATH_HEADER}
                full.update(r)
                w.writerow(full)

    def _clean_row(self, ts_ms):
        return {
            "schema_version": "3", "ts_ms": str(ts_ms), "price_usd": "0.00005",
            "price_sol": "0.0000003", "vsol": "50.0", "vtok": "1000000000",
            "venue_state": "CURVE_ACTIVE", "source": "live_pp", "backfilled": "false",
            "data_status": "ok",
        }

    def _corrupted_row(self, ts_ms):
        # Same VSOL_EXCEEDS_GRADUATION_WHILE_CURVE_ACTIVE fixture pattern
        # used in research/tests/test_v8_final_state.py -- vsol implies
        # the token graduated while venue_state still claims CURVE_ACTIVE.
        return {
            "schema_version": "3", "ts_ms": str(ts_ms), "price_usd": "73.49",
            "price_sol": "0.42", "vsol": "116.27", "vtok": "279900000",
            "venue_state": "CURVE_ACTIVE", "source": "live_pp", "backfilled": "false",
            "data_status": "ok",
        }

    def test_valid_only_false_never_gates_or_preloads(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "MINT_A.csv"
            self._write_csv(p, [self._corrupted_row(0), self._corrupted_row(1000)])
            included, raw_rows = _integrity_gate(p, valid_only=False)
        self.assertTrue(included)
        self.assertIsNone(raw_rows)

    def test_valid_only_excludes_corrupted_path(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "MINT_BAD.csv"
            self._write_csv(p, [self._clean_row(0), self._corrupted_row(1000),
                                 self._clean_row(2000)])
            included, raw_rows = _integrity_gate(p, valid_only=True)
        self.assertFalse(included)
        self.assertIsNone(raw_rows)

    def test_valid_only_includes_clean_path_and_returns_reusable_rows(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "MINT_GOOD.csv"
            self._write_csv(p, [self._clean_row(0), self._clean_row(1000)])
            included, raw_rows = _integrity_gate(p, valid_only=True)
        self.assertTrue(included)
        self.assertEqual(len(raw_rows), 2)

    def test_load_path_reuses_preloaded_raw_rows_without_rereading(self):
        """_load_path(path, raw_rows=...) must produce the same typed
        output as a normal load, without needing the file to be
        re-readable (proves it never re-opens the path when raw_rows is
        given -- deleting the file after preloading must not matter)."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "MINT_C.csv"
            self._write_csv(p, [self._clean_row(0), self._clean_row(1000)])
            from research.path_schema import load_path_file
            raw_rows, _w = load_path_file(p)
            p.unlink()  # gone -- _load_path must not try to reopen it
            typed = _load_path(p, raw_rows=raw_rows)
        self.assertEqual(len(typed), 2)
        self.assertEqual(typed[0]["price_usd"], 0.00005)


class TestShakeoutDepthForTarget(unittest.TestCase):
    """Extracted from _analyse_shakeout (Analysis A) -- E3's hard_stop was
    derived from this exact function, so it must be independently correct."""

    def test_never_reaches_target_returns_none(self):
        rows = [_row(0, 100), _row(1000, 105)]
        self.assertIsNone(_shakeout_depth_for_target(rows, 30))

    def test_computes_max_drawdown_before_target_hit(self):
        rows = [_row(0, 100), _row(1000, 70), _row(2000, 130)]  # +30% target = 130
        depth = _shakeout_depth_for_target(rows, 30)
        self.assertAlmostEqual(depth, 30.0, places=1)  # (100-70)/100

    def test_drawdown_after_target_hit_is_not_counted(self):
        # Deep drop happens AFTER the target is already reached -- must be excluded.
        rows = [_row(0, 100), _row(1000, 130), _row(2000, 10)]
        depth = _shakeout_depth_for_target(rows, 30)
        self.assertAlmostEqual(depth, 0.0, places=1)

    def test_zero_or_negative_entry_price_returns_none(self):
        rows = [_row(0, 0), _row(1000, 130)]
        self.assertIsNone(_shakeout_depth_for_target(rows, 30))

    def test_empty_rows_returns_none(self):
        self.assertIsNone(_shakeout_depth_for_target([], 30))


class TestTimeToTargetMinutes(unittest.TestCase):

    def test_never_reaches_target_returns_none(self):
        rows = [_row(0, 100), _row(1000, 105)]
        self.assertIsNone(_time_to_target_minutes(rows, 30))

    def test_computes_minutes_to_first_target_hit(self):
        rows = [_row(0, 100), _row(60_000, 110), _row(120_000, 130)]  # +30% at t=120s
        self.assertAlmostEqual(_time_to_target_minutes(rows, 30), 2.0, places=2)

    def test_empty_rows_returns_none(self):
        self.assertIsNone(_time_to_target_minutes([], 30))


if __name__ == "__main__":
    unittest.main()
