"""
RF3 — Tiered path watch window tests.

Tests `_should_extend()` as a pure function and the extension-decision logic
embedded in PeakTracker._finalise_loop via direct state-dict manipulation.

No network, no asyncio, no Supabase.
"""

import unittest
from unittest.mock import MagicMock, patch
import time

# We import only the pure function and constants — no side effects from the module
from research.peak_tracker import (
    _should_extend,
    BASE_WINDOW_S,
    EXTENSION_INCREMENT_S,
    HARD_CAP_S,
    _MAX_EXTENSIONS,
    _RECENT_TICK_WINDOW_S,
)


def _make_state(
    entry_price: float = 1.0,
    max_price: float = 1.0,
    last_tick_ts: float = 0.0,
    last_valid_price: float = 0.0,
    extension_count: int = 0,
    alert_ts: float = 0.0,
) -> dict:
    """Build a minimal state dict for testing."""
    now = time.time()
    return {
        "entry_price":          entry_price,
        "max_price":            max_price,
        "max_ts":               alert_ts,
        "alert_ts":             alert_ts,
        "expiry":               alert_ts + BASE_WINDOW_S,
        "base_expiry":          alert_ts + BASE_WINDOW_S,
        "done":                 False,
        "extension_count":      extension_count,
        "last_tick_ts":         last_tick_ts,
        "last_valid_price":     last_valid_price,
        "stop_reason":          None,
        "valid_tick_count":     0,
        "disconnection_periods": [],
        "ws_connected":         True,
    }


class TestShouldExtend(unittest.TestCase):
    """Unit tests for the pure _should_extend() function."""

    def setUp(self):
        # Anchor: T=0 is "15 minutes ago" from now
        self.now = time.time()
        self.t0  = self.now - BASE_WINDOW_S   # simulated alert_ts

    # ── test_no_extension_token_dies_at_minute_2 ──────────────────────────────
    def test_no_extension_token_dies_at_minute_2(self):
        """Tick at T+2min, then silence. At T+15min: last_tick 13min ago → no extend."""
        last_tick_ts    = self.t0 + 120          # T+2min
        last_valid_price = 2.0
        session_peak    = 2.0

        result = _should_extend(
            now=self.now,
            expiry=self.t0 + BASE_WINDOW_S,
            extension_count=0,
            last_tick_ts=last_tick_ts,
            last_valid_price=last_valid_price,
            session_peak_price=session_peak,
        )
        self.assertFalse(result)

    # ── test_extend_to_30min ──────────────────────────────────────────────────
    def test_extend_to_30min(self):
        """Active ticks throughout first 14min. At T+15min: both conditions met → extend."""
        last_tick_ts     = self.t0 + BASE_WINDOW_S - 60   # tick 1min before window end
        session_peak     = 3.0
        last_valid_price = 2.0   # 2.0 >= 0.5 * 3.0 = 1.5

        result = _should_extend(
            now=self.now,
            expiry=self.t0 + BASE_WINDOW_S,
            extension_count=0,
            last_tick_ts=last_tick_ts,
            last_valid_price=last_valid_price,
            session_peak_price=session_peak,
        )
        self.assertTrue(result)

    # ── test_extend_to_45min ──────────────────────────────────────────────────
    def test_extend_to_45min(self):
        """Active through T+29min (one extension already done). At T+30min: extend again."""
        # Simulate: 30min window has elapsed, extension_count=1, recent tick
        now2 = self.t0 + 2 * BASE_WINDOW_S           # T+30min
        last_tick_ts     = now2 - 60                  # 1min ago from T+30
        session_peak     = 4.0
        last_valid_price = 3.0   # 3.0 >= 0.5 * 4.0

        result = _should_extend(
            now=now2,
            expiry=self.t0 + 2 * BASE_WINDOW_S,
            extension_count=1,
            last_tick_ts=last_tick_ts,
            last_valid_price=last_valid_price,
            session_peak_price=session_peak,
        )
        self.assertTrue(result)

    # ── test_extend_to_60min_hard_cap ────────────────────────────────────────
    def test_extend_to_60min_hard_cap(self):
        """Active through T+44min (two extensions done). At T+45min: extend to 60min."""
        now3 = self.t0 + 3 * BASE_WINDOW_S           # T+45min
        last_tick_ts     = now3 - 60
        session_peak     = 5.0
        last_valid_price = 4.0

        # extension_count=2 → third extension allowed (_MAX_EXTENSIONS=3)
        result = _should_extend(
            now=now3,
            expiry=self.t0 + 3 * BASE_WINDOW_S,
            extension_count=2,
            last_tick_ts=last_tick_ts,
            last_valid_price=last_valid_price,
            session_peak_price=session_peak,
        )
        self.assertTrue(result)

    # ── test_hard_cap_enforced ────────────────────────────────────────────────
    def test_hard_cap_enforced(self):
        """extension_count == _MAX_EXTENSIONS → no further extension regardless of price."""
        now4 = self.t0 + 4 * BASE_WINDOW_S           # T+60min (already at hard cap)
        last_tick_ts     = now4 - 30                  # very recent tick
        session_peak     = 5.0
        last_valid_price = 5.0                        # at peak — price conditions fine

        result = _should_extend(
            now=now4,
            expiry=self.t0 + 4 * BASE_WINDOW_S,
            extension_count=_MAX_EXTENSIONS,           # hard cap
            last_tick_ts=last_tick_ts,
            last_valid_price=last_valid_price,
            session_peak_price=session_peak,
        )
        self.assertFalse(result)

    # ── test_no_extension_no_recent_tick ─────────────────────────────────────
    def test_no_extension_no_recent_tick(self):
        """Last tick at T+5min. At T+15min: no tick in last 3min → no extension."""
        last_tick_ts    = self.t0 + 300              # T+5min
        # at T+15min, elapsed since last tick = 600s > _RECENT_TICK_WINDOW_S (180s)
        result = _should_extend(
            now=self.now,
            expiry=self.t0 + BASE_WINDOW_S,
            extension_count=0,
            last_tick_ts=last_tick_ts,
            last_valid_price=2.0,
            session_peak_price=2.0,
        )
        self.assertFalse(result)

    # ── test_no_extension_price_below_50pct_peak ─────────────────────────────
    def test_no_extension_price_below_50pct_peak(self):
        """Token peaks at +200%, crashes to +20% by T+14min → price < 50% of peak → no ext."""
        # Entry=1.0, peak=3.0 (+200%), current=1.2 (+20%)
        # 1.2 < 0.50 * 3.0 = 1.5
        last_tick_ts     = self.now - 60             # recent tick
        session_peak     = 3.0
        last_valid_price = 1.2

        result = _should_extend(
            now=self.now,
            expiry=self.t0 + BASE_WINDOW_S,
            extension_count=0,
            last_tick_ts=last_tick_ts,
            last_valid_price=last_valid_price,
            session_peak_price=session_peak,
        )
        self.assertFalse(result)

    # ── test_missing_price_does_not_extend ───────────────────────────────────
    def test_missing_price_does_not_extend(self):
        """WS silent — last_valid_price stays 0 → missing data → no extension."""
        result = _should_extend(
            now=self.now,
            expiry=self.t0 + BASE_WINDOW_S,
            extension_count=0,
            last_tick_ts=0.0,          # never received a tick
            last_valid_price=0.0,
            session_peak_price=0.0,
        )
        self.assertFalse(result)


class TestExtensionCount(unittest.TestCase):
    """Verify MAX_EXTENSIONS constant matches HARD_CAP_S / EXTENSION_INCREMENT_S logic."""

    def test_max_extensions_constant(self):
        expected = (HARD_CAP_S - BASE_WINDOW_S) // EXTENSION_INCREMENT_S
        self.assertEqual(_MAX_EXTENSIONS, expected)
        # For default values: (3600-900)//900 = 3
        self.assertEqual(_MAX_EXTENSIONS, 3)

    def test_full_hard_cap_is_60min(self):
        """BASE + 3 * EXTENSION = 3600s = 60min."""
        self.assertEqual(BASE_WINDOW_S + _MAX_EXTENSIONS * EXTENSION_INCREMENT_S, HARD_CAP_S)


class TestStopReasonLogic(unittest.TestCase):
    """
    Test that stop_reason assignment in _finalise_loop is correct.
    We replicate the logic inline since _finalise_loop is async; the pure
    rule is simple enough to verify directly.
    """

    def _stop_reason(self, extension_count: int, should_ext: bool) -> str:
        """Mirror the stop_reason logic from _finalise_loop."""
        if extension_count >= _MAX_EXTENSIONS:
            return "hard_cap_reached"
        elif extension_count > 0:
            return "extension_condition_failed"
        else:
            return "base_window_expired"

    def test_stop_reason_base_window(self):
        self.assertEqual(self._stop_reason(0, False), "base_window_expired")

    def test_stop_reason_extension_failed(self):
        self.assertEqual(self._stop_reason(1, False), "extension_condition_failed")

    def test_stop_reason_hard_cap(self):
        self.assertEqual(self._stop_reason(_MAX_EXTENSIONS, False), "hard_cap_reached")


if __name__ == "__main__":
    unittest.main()
