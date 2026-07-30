"""
N6: V8 paper twin — unit tests for the pure logic in memecoin/v8_paper.py.

No network, no real Supabase/PP. pumpportal_monitor.monitor.get_screening_state
is patched directly for the gate tests.

Run: python -m pytest memecoin/tests/test_v8_paper.py -v
"""

import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from memecoin.v8_paper import (
    V8_EXIT_CONFIG,
    V8_PROGRESS_MAX,
    _check_exit,
    _new_position,
    _pnl,
    compute_progress_at_signal,
    passes_v8_gate,
)


def _signal(**overrides):
    base = dict(
        id="sig1", chain="solana", token_address="Mint1111111111111111111111111111111111111",
        token_symbol="TEST", signal_type="social_alert", strength="strong",
        price_usd=0.00001, dex_id="", _price_pp=0.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestComputeProgress(unittest.TestCase):

    def test_non_solana_returns_none(self):
        self.assertIsNone(compute_progress_at_signal("bsc", "x"))

    def test_no_screening_state_returns_none(self):
        with patch("memecoin.pumpportal_monitor.monitor") as m:
            m.get_screening_state.return_value = None
            self.assertIsNone(compute_progress_at_signal("solana", "mint"))

    def test_zero_vsol_returns_none(self):
        with patch("memecoin.pumpportal_monitor.monitor") as m:
            m.get_screening_state.return_value = SimpleNamespace(latest_vsol=0)
            self.assertIsNone(compute_progress_at_signal("solana", "mint"))

    def test_progress_computed_from_vsol(self):
        with patch("memecoin.pumpportal_monitor.monitor") as m:
            m.get_screening_state.return_value = SimpleNamespace(latest_vsol=57.5)
            progress = compute_progress_at_signal("solana", "mint")
            self.assertAlmostEqual(progress, 0.5, places=2)   # 57.5 / 115


class TestPassesV8Gate(unittest.TestCase):

    def test_unknown_progress_fails_closed(self):
        with patch("memecoin.pumpportal_monitor.monitor") as m:
            m.get_screening_state.return_value = None
            passed, reason, progress = passes_v8_gate(_signal())
            self.assertFalse(passed)
            self.assertEqual(reason, "progress_unknown")

    def test_progress_over_threshold_blocked(self):
        with patch("memecoin.pumpportal_monitor.monitor") as m:
            m.get_screening_state.return_value = SimpleNamespace(latest_vsol=100.0)  # 0.87
            passed, reason, progress = passes_v8_gate(_signal())
            self.assertFalse(passed)
            self.assertIn("over", reason)

    def test_has_dex_id_blocked_even_if_progress_low(self):
        with patch("memecoin.pumpportal_monitor.monitor") as m:
            m.get_screening_state.return_value = SimpleNamespace(latest_vsol=20.0)  # 0.17
            passed, reason, progress = passes_v8_gate(_signal(dex_id="raydium"))
            self.assertFalse(passed)
            self.assertIn("has_dex_id", reason)

    def test_low_progress_no_dex_passes(self):
        with patch("memecoin.pumpportal_monitor.monitor") as m:
            m.get_screening_state.return_value = SimpleNamespace(latest_vsol=20.0)  # 0.17
            passed, reason, progress = passes_v8_gate(_signal(dex_id=""))
            self.assertTrue(passed)
            self.assertLess(progress, V8_PROGRESS_MAX)


class TestNewPosition(unittest.TestCase):

    def test_uses_pp_price_when_available(self):
        pos = _new_position(_signal(_price_pp=0.00002, price_usd=0.00001), progress=0.2)
        self.assertEqual(pos["entry_price"], 0.00002)
        self.assertEqual(pos["entry_source"], "pp_tick")

    def test_falls_back_to_dex_price(self):
        pos = _new_position(_signal(_price_pp=0.0, price_usd=0.00001), progress=0.2)
        self.assertEqual(pos["entry_price"], 0.00001)
        self.assertEqual(pos["entry_source"], "dex_stale")

    def test_peak_price_starts_at_entry(self):
        pos = _new_position(_signal(price_usd=0.00001), progress=0.2)
        self.assertEqual(pos["peak_price"], pos["entry_price"])


class TestCheckExit(unittest.TestCase):

    def _pos(self, entry, current, peak, entry_time=None):
        return {
            "entry_price": entry, "current_price": current, "peak_price": peak,
            "entry_time": entry_time or time.time(), "size_usd": 1.0,
            "status": "open",
        }

    def test_hard_stop_fires(self):
        pos = self._pos(entry=1.0, current=0.6, peak=1.0)   # -40%, below -35% hard stop
        self.assertEqual(_check_exit(pos, V8_EXIT_CONFIG), "hard_stop")

    def test_no_exit_when_flat(self):
        pos = self._pos(entry=1.0, current=1.05, peak=1.05)
        self.assertEqual(_check_exit(pos, V8_EXIT_CONFIG), "")

    def test_trailing_stop_fires_after_tier1_activation(self):
        # peak_gain = 0.30 (tier1 activates), pulled back >25% from peak
        pos = self._pos(entry=1.0, current=0.97, peak=1.30)
        self.assertEqual(_check_exit(pos, V8_EXIT_CONFIG), "trailing_stop")

    def test_trailing_stop_not_armed_below_tier1(self):
        # peak_gain only 0.10 — below the 0.30 tier1 activation, hard_stop also not hit
        pos = self._pos(entry=1.0, current=0.95, peak=1.10)
        self.assertEqual(_check_exit(pos, V8_EXIT_CONFIG), "")

    def test_time_stop_fires_when_stale_and_flat(self):
        old_entry = time.time() - (V8_EXIT_CONFIG["time_stop_minutes"] + 1) * 60
        pos = self._pos(entry=1.0, current=1.05, peak=1.05, entry_time=old_entry)
        self.assertEqual(_check_exit(pos, V8_EXIT_CONFIG), "time_stop")

    def test_time_stop_suppressed_for_a_runner(self):
        old_entry = time.time() - (V8_EXIT_CONFIG["time_stop_minutes"] + 1) * 60
        pos = self._pos(entry=1.0, current=1.35, peak=1.40)   # peak_gain 0.40 >= 0.30
        self.assertEqual(_check_exit(pos, V8_EXIT_CONFIG), "")


class TestPnl(unittest.TestCase):

    def test_open_position_uses_current_price(self):
        pos = {"entry_price": 1.0, "current_price": 1.5, "exit_price": 0.0,
               "status": "open", "size_usd": 10.0}
        pnl_usd, pnl_pct = _pnl(pos)
        self.assertAlmostEqual(pnl_pct, 0.5)
        self.assertAlmostEqual(pnl_usd, 5.0)

    def test_closed_position_uses_exit_price(self):
        pos = {"entry_price": 1.0, "current_price": 1.9, "exit_price": 1.2,
               "status": "closed", "size_usd": 10.0}
        pnl_usd, pnl_pct = _pnl(pos)
        self.assertAlmostEqual(pnl_pct, 0.2)

    def test_zero_entry_price_returns_zero_not_raise(self):
        pos = {"entry_price": 0.0, "current_price": 1.0, "exit_price": 0.0,
               "status": "open", "size_usd": 10.0}
        self.assertEqual(_pnl(pos), (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
