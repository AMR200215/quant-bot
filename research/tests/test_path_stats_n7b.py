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


if __name__ == "__main__":
    unittest.main()
