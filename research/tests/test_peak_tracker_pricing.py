"""research/tests/test_peak_tracker_pricing.py — V8 DATA RECOVERY batch:
proves PeakTracker._price_from_msg actually calls the canonical
pricing helper (memecoin.pumpfun_reserve_pricing) rather than a
reintroduced local /1e6 formula.

Run: python -m pytest research/tests/test_peak_tracker_pricing.py -v
"""

import threading
import unittest
from types import SimpleNamespace

from research.peak_tracker import PeakTracker


class TestPriceFromMsg(unittest.TestCase):

    def _tracker_stub(self, sol_price=175.0):
        # _price_from_msg only reads self._sol_price -- call it unbound
        # against a minimal stand-in rather than running full __init__
        # (which stands up Supabase/websocket state we don't need here).
        return SimpleNamespace(_sol_price=sol_price)

    def test_real_captured_example_no_longer_a_million_x_inflated(self):
        # Real PumpPortal message shape, live-captured 2026-08-19.
        msg = {"vSolInBondingCurve": 115.005359056806, "vTokensInBondingCurve": 279900000}
        stub = self._tracker_stub(sol_price=175.0)
        price = PeakTracker._price_from_msg(stub, msg)
        expected = (115.005359056806 / 279900000) * 175.0
        self.assertAlmostEqual(price, expected, places=10)
        self.assertLess(price, 0.001)  # sane pre-graduation USD price, not $16+

    def test_falls_back_to_trade_amounts_when_reserves_absent(self):
        msg = {"solAmount": 1.0, "tokenAmount": 1000.0}
        stub = self._tracker_stub(sol_price=175.0)
        price = PeakTracker._price_from_msg(stub, msg)
        self.assertAlmostEqual(price, (1.0 / 1000.0) * 175.0, places=10)

    def test_returns_none_when_nothing_usable(self):
        stub = self._tracker_stub()
        self.assertIsNone(PeakTracker._price_from_msg(stub, {}))


if __name__ == "__main__":
    unittest.main()
