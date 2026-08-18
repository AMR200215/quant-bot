"""memecoin/tests/test_pumpportal_monitor_pricing.py — V8 DATA RECOVERY
batch: proves PumpPortalMonitor._compute_price actually calls the
canonical pricing helper rather than a reintroduced local /1e6 formula
-- this exact file already had this exact bug recur once (2026-08-04
fix only touched the trade-amount fallback branch, leaving the reserve
branch still wrong).

Run: python -m pytest memecoin/tests/test_pumpportal_monitor_pricing.py -v
"""

import threading
import unittest
from types import SimpleNamespace

from memecoin.pumpportal_monitor import PumpPortalMonitor


class TestComputePrice(unittest.TestCase):

    def _monitor_stub(self, sol_price=175.0):
        return SimpleNamespace(_sol_price_lock=threading.Lock(), _sol_price=sol_price)

    def test_real_captured_example_no_longer_a_million_x_inflated(self):
        msg = {"vSolInBondingCurve": 115.005359056806, "vTokensInBondingCurve": 279900000}
        stub = self._monitor_stub(sol_price=175.0)
        price = PumpPortalMonitor._compute_price(stub, msg)
        expected = (115.005359056806 / 279900000) * 175.0
        self.assertAlmostEqual(price, expected, places=10)
        self.assertLess(price, 0.001)

    def test_falls_back_to_trade_amounts_when_reserves_absent(self):
        msg = {"solAmount": 1.0, "tokenAmount": 1000.0}
        stub = self._monitor_stub(sol_price=175.0)
        price = PumpPortalMonitor._compute_price(stub, msg)
        self.assertAlmostEqual(price, (1.0 / 1000.0) * 175.0, places=10)

    def test_returns_zero_when_nothing_usable(self):
        stub = self._monitor_stub()
        self.assertEqual(PumpPortalMonitor._compute_price(stub, {}), 0.0)


if __name__ == "__main__":
    unittest.main()
