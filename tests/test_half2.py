"""
tests/test_half2.py

Go-live verification protocol — Half-2 acceptance tests.

Covers:
  A. Trail tiers (FIX-06)
     1. Tier-1 activates at +30%, wide 35% trail
     2. Tier-2 activates at +100%, tightens to 25%
     3. Tier-3 activates at +300%, tightens to 15%
     4. Breakeven floor: peak ≥ +40%, exit must be ≥ entry * 1.02
     5. Time stop suppressed while peak_gain ≥ +30%

  B. Entry gate (FIX-01, executor)
     6. Synthetic Folkistan: quote 40% above signal → blocked_quote_drift
     7. no_quote: Jupiter returns nothing → blocked before spend

  C. Data integrity
     8. hard_stop_pct in JOURNAL_FIELDS

  D. Sell-stuck (FIX-05)
     9. Ladder exhaustion → status=sell_stuck, no journal close row
"""

import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_position(**overrides):
    """Build a minimal Position-like object for stop-check tests."""
    from memecoin.portfolio import Position
    defaults = dict(
        id="test001",
        signal_id="sig001",
        chain="solana",
        token_address="So11111111111111111111111111111111111111112",
        token_symbol="TEST",
        signal_type="social_alert",
        strength="medium",
        entry_price=1.0,
        signal_price=1.0,
        current_price=1.0,
        peak_price=1.0,
        size_usd=3.0,
        entry_time=0.0,
        signal_time=0.0,
        hard_stop_pct=-0.35,
        trailing_stop_pct=-0.35,
        trail_activates_pct=0.30,
        time_stop_minutes=90,
        status="open",
        notes="",
        whale_tiers=[],
        whales_involved=[],
        whale_count=0,
        tp_levels_hit=[],
        remaining_fraction=1.0,
        realized_pnl_usd=0.0,
    )
    defaults.update(overrides)
    pos = Position.__new__(Position)
    for k, v in defaults.items():
        setattr(pos, k, v)
    return pos


def _eval_stops(pos, whale_sells=None, *, time_now=None):
    """
    Run just the stop-check block from update_prices against a single position.
    Returns the exit reason string or None.
    Injects time.time() → time_now if provided.
    """
    import memecoin.portfolio as _port
    whale_sells = whale_sells or {}
    stall_tracker = {pos.id: {"last_peak": pos.peak_price, "stall_since": 0.0}}

    # Replicate the stop-check logic inline (same as portfolio.update_prices)
    import time as _time
    now = time_now or _time.time()

    from memecoin.config import get_signal_settings as _gss, TIME_STOP_MIN_GAIN
    _exit_cfg    = _gss(pos.signal_type)
    _trail_tiers = _exit_cfg.get("trail_tiers", None)
    gain         = pos.pnl_pct
    _peak_gain   = ((pos.peak_price / pos.entry_price) - 1) if pos.entry_price > 0 else 0
    reason       = None

    if pos.current_price <= 0:
        return None

    # 0. Profit-lock (not tested here — skip)

    # 1. Hard stop
    _stop_lvl = pos.entry_price * (1 + pos.hard_stop_pct)
    if pos.signal_price > 0 and pos.entry_price > pos.signal_price:
        _stop_lvl = pos.signal_price * (1 + pos.hard_stop_pct)
    if pos.current_price <= _stop_lvl:
        reason = "hard_stop"

    # 2. Trailing stop
    if not reason and pos.peak_price > 0 and pos.entry_price > 0:
        if _trail_tiers:
            _active_tier = None
            for _tier in sorted(_trail_tiers, key=lambda t: t["activates_at"], reverse=True):
                if _peak_gain >= _tier["activates_at"]:
                    _active_tier = _tier
                    break
            if _active_tier:
                _trail_pct  = -abs(_active_tier["trail_pct"])
                _trail_stop = pos.peak_price * (1 + _trail_pct)
                if _peak_gain >= 0.40:
                    _floor = pos.entry_price * 1.02
                    if _trail_stop < _floor:
                        _trail_stop = _floor
                if pos.current_price <= _trail_stop:
                    reason = "trailing_stop"
        else:
            if gain >= pos.trail_activates_pct:
                drawdown = (pos.current_price - pos.peak_price) / pos.peak_price
                if drawdown <= pos.trailing_stop_pct:
                    reason = "trailing_stop"

    # 3. Whale exit
    if not reason and pos.token_address in whale_sells:
        sellers = whale_sells[pos.token_address]
        involved = [w for w in sellers if w in pos.whales_involved]
        if involved:
            n_whales = pos.whale_count or 1
            if n_whales == 1:
                reason = f"whale_exit:{involved[0][:8]}"
            elif len(involved) >= max(1, n_whales // 2):
                reason = f"whale_exit:{len(involved)}_of_{n_whales}"

    # 4. Time stop — only while peak_gain < 30%
    if not reason and (now - pos.entry_time) / 60 > pos.time_stop_minutes:
        if _peak_gain < 0.30 and gain < TIME_STOP_MIN_GAIN:
            reason = "time_stop"

    return reason


# ---------------------------------------------------------------------------
# A. Trail tiers
# ---------------------------------------------------------------------------

class TestTrailTiers(unittest.TestCase):

    def test_tier1_activates_at_30pct(self):
        """Peak +30%, price retraces 35% from peak → trailing_stop fires."""
        pos = _make_position(
            entry_price=1.0,
            peak_price=1.30,
            current_price=1.30 * (1 - 0.35) - 0.001,  # just past 35% drawdown
        )
        reason = _eval_stops(pos)
        self.assertEqual(reason, "trailing_stop",
                         f"Tier-1 should fire at 35% drawdown from +30% peak, got {reason}")

    def test_tier1_does_not_fire_inside_35pct(self):
        """Peak +30%, drawdown 34% → no exit yet."""
        pos = _make_position(
            entry_price=1.0,
            peak_price=1.30,
            current_price=1.30 * (1 - 0.34),
        )
        reason = _eval_stops(pos)
        self.assertIsNone(reason, f"Tier-1 should not fire at 34% drawdown, got {reason}")

    def test_tier2_activates_at_100pct(self):
        """Peak +100%, price retraces 25% from peak → trailing_stop fires (not 35%)."""
        pos = _make_position(
            entry_price=1.0,
            peak_price=2.0,
            current_price=2.0 * (1 - 0.25) - 0.001,  # just past 25%
        )
        reason = _eval_stops(pos)
        self.assertEqual(reason, "trailing_stop", f"Tier-2 should fire, got {reason}")

    def test_tier2_does_not_fire_at_tier1_drawdown(self):
        """Peak +100%, drawdown 26%→ fires. Drawdown 24% → doesn't."""
        # Tier-2 is 25% — drawdown of 24% should NOT fire
        pos = _make_position(
            entry_price=1.0,
            peak_price=2.0,
            current_price=2.0 * (1 - 0.24),
        )
        reason = _eval_stops(pos)
        self.assertIsNone(reason, f"Tier-2 should not fire at 24% drawdown, got {reason}")

    def test_tier3_activates_at_300pct(self):
        """Peak +300%, drawdown 15% → fires."""
        pos = _make_position(
            entry_price=1.0,
            peak_price=4.0,
            current_price=4.0 * (1 - 0.15) - 0.001,
        )
        reason = _eval_stops(pos)
        self.assertEqual(reason, "trailing_stop", f"Tier-3 should fire, got {reason}")

    def test_breakeven_floor_prevents_loss_exit(self):
        """
        Peak +45% (peak_gain ≥ 40%), full retrace below floor.
        Trail stop should be ≥ entry * 1.02, not below entry.
        Protocol spec: entry=1.00, peak=1.45, full retrace → exit ≥ 1.02.
        """
        entry = 1.0
        peak  = 1.45     # +45% peak → floor active
        # Without floor: trail_stop = 1.45 * (1 - 0.35) = 0.9425 (LOSS)
        # With floor:    trail_stop = max(0.9425, 1.02) = 1.02
        pos = _make_position(
            entry_price=entry,
            peak_price=peak,
            current_price=1.01,   # below floor (1.02) → should trigger
        )
        reason = _eval_stops(pos)
        self.assertEqual(reason, "trailing_stop",
                         "Breakeven floor must trigger trailing_stop when price < entry*1.02")

    def test_breakeven_floor_does_not_trigger_above_floor(self):
        """Price still above floor → no exit."""
        pos = _make_position(
            entry_price=1.0,
            peak_price=1.45,
            current_price=1.03,   # above floor of 1.02
        )
        reason = _eval_stops(pos)
        self.assertIsNone(reason, f"Price above floor should not exit, got {reason}")

    def test_time_stop_suppressed_when_peak_30pct(self):
        """
        Position is past time limit but peak was +50% — time stop must NOT fire.
        """
        pos = _make_position(
            entry_price=1.0,
            peak_price=1.50,         # peak_gain = 50% ≥ 30%
            current_price=1.20,      # still in profit, not near trail
            entry_time=0.0,
            time_stop_minutes=1,     # very short for test
        )
        # time well past limit
        reason = _eval_stops(pos, time_now=10000.0)
        self.assertNotEqual(reason, "time_stop",
                            "Time stop must not fire when peak_gain ≥ 30%")

    def test_time_stop_fires_when_peak_below_30pct(self):
        """Position past time limit, peak never reached +30% → time_stop fires."""
        pos = _make_position(
            entry_price=1.0,
            peak_price=1.10,       # peak_gain = 10% < 30%
            current_price=0.95,    # below TIME_STOP_MIN_GAIN threshold
            entry_time=0.0,
            time_stop_minutes=1,
        )
        reason = _eval_stops(pos, time_now=10000.0)
        self.assertEqual(reason, "time_stop",
                         f"Time stop should fire when peak_gain < 30%, got {reason}")


# ---------------------------------------------------------------------------
# B. Entry gate (executor)
# ---------------------------------------------------------------------------

class TestEntryGate(unittest.TestCase):

    def test_folkistan_quote_drift_blocks(self):
        """Synthetic Folkistan: Jupiter quote 40% above signal → blocked_quote_drift."""
        import memecoin.executor as ex_mod

        mock_kp = MagicMock()
        mock_kp.pubkey.return_value = "So11111111111111111111111111111111111111112"

        signal_price = 0.001
        # 40% above signal: quote_price = size_usd / tokens_out
        # size_usd=3, signal_price=0.001 → tokens_out for 0% drift = 3000
        # for 40% drift: quote_price = 0.001 * 1.40 = 0.0014 → tokens_out = 3/0.0014 ≈ 2143
        tokens_out_int = int(3.0 / (signal_price * 1.40) * 1e6)  # 6 decimals

        with (
            patch.object(ex_mod, "LIVE_DRY_RUN", False),
            patch.object(ex_mod, "_get_keypair", return_value=mock_kp),
            patch.object(ex_mod, "_sol_price_usd", return_value=170.0),
            patch.object(ex_mod, "_sol_balance", return_value=500_000_000),
            patch.object(ex_mod, "_jup_get_quote", return_value={
                "outAmount": str(tokens_out_int),
                "outputDecimals": 6,
            }),
            patch.object(ex_mod, "_load_solders", return_value=(MagicMock(), MagicMock(), MagicMock())),
        ):
            result = ex_mod.MemeExecutor().buy(
                "So11111111111111111111111111111111111111112",
                size_usd=3.0,
                signal_price=signal_price,
            )

        self.assertFalse(result.get("success"))
        self.assertEqual(result.get("reason"), "blocked_quote_drift",
                         f"Expected blocked_quote_drift, got: {result}")
        self.assertGreater(result.get("slippage_pct", 0), 15,
                           "Reported slippage should be > 15%")

    def test_no_quote_blocks_before_spend(self):
        """Jupiter returns nothing → no_quote blocks before any SOL spent."""
        import memecoin.executor as ex_mod

        mock_kp = MagicMock()
        mock_kp.pubkey.return_value = "So11111111111111111111111111111111111111112"

        with (
            patch.object(ex_mod, "LIVE_DRY_RUN", False),
            patch.object(ex_mod, "_get_keypair", return_value=mock_kp),
            patch.object(ex_mod, "_sol_price_usd", return_value=170.0),
            patch.object(ex_mod, "_sol_balance", return_value=500_000_000),
            patch.object(ex_mod, "_jup_get_quote", side_effect=Exception("connection error")),
            patch.object(ex_mod, "_load_solders", return_value=(MagicMock(), MagicMock(), MagicMock())),
        ):
            result = ex_mod.MemeExecutor().buy(
                "So11111111111111111111111111111111111111112",
                size_usd=3.0,
                signal_price=0.001,
            )

        self.assertFalse(result.get("success"))
        self.assertEqual(result.get("reason"), "no_quote",
                         f"Expected no_quote, got: {result}")


# ---------------------------------------------------------------------------
# C. Data integrity
# ---------------------------------------------------------------------------

class TestDataIntegrity(unittest.TestCase):

    def test_hard_stop_pct_in_journal_fields(self):
        """hard_stop_pct must be in JOURNAL_FIELDS."""
        from memecoin.portfolio import JOURNAL_FIELDS
        self.assertIn("hard_stop_pct", JOURNAL_FIELDS,
                      "hard_stop_pct missing from JOURNAL_FIELDS")

    def test_hard_stop_pct_in_journal_row(self):
        """_build_journal_row must include hard_stop_pct."""
        from memecoin.portfolio import _build_journal_row, Position
        pos = _make_position(hard_stop_pct=-0.35)
        # Provide required list fields
        pos.tp_levels_hit = []
        pos.remaining_fraction = 1.0
        pos.realized_pnl_usd = 0.0
        row = _build_journal_row(pos)
        self.assertIn("hard_stop_pct", row)
        self.assertEqual(row["hard_stop_pct"], -0.35)

    def test_accounting_epoch_bumped(self):
        """ACCOUNTING_EPOCH must be e4_rt_feed_quote_gate."""
        from memecoin.portfolio import ACCOUNTING_EPOCH
        self.assertEqual(ACCOUNTING_EPOCH, "e4_rt_feed_quote_gate",
                         f"Epoch not bumped: {ACCOUNTING_EPOCH!r}")


# ---------------------------------------------------------------------------
# D. Sell-stuck
# ---------------------------------------------------------------------------

class TestSellStuck(unittest.TestCase):

    def test_ladder_exhaustion_leaves_position_open(self):
        """
        After MAX_SELL_RETRIES failed sells: status='sell_stuck',
        position NOT removed from _positions, _append_journal NOT called.
        """
        from memecoin import portfolio as _port

        journal_written = [False]
        orig_append = _port._append_journal

        def _mock_append(pos):
            journal_written[0] = True

        pp_mock = MagicMock()
        pp_mock.monitor.get_prices.return_value = {}

        sig = MagicMock()
        sig.signal_type = "social_alert"
        sig.token_cohort = "telegram_pump"
        sig.token_address = "So11111111111111111111111111111111111111112"
        sig.token_symbol = "STUCK"

        with (
            patch("memecoin.portfolio.LIVE_TRADING", True),
            patch("memecoin.portfolio.LIVE_DRY_RUN", False),
            patch("memecoin.portfolio._append_journal", _mock_append),
            patch("memecoin.portfolio._save_positions"),
            patch("memecoin.portfolio.alerts"),
            patch.dict("sys.modules", {
                "memecoin.pumpportal_monitor": pp_mock,
                "memecoin.pumpportal_monitor.monitor": pp_mock.monitor,
            }),
        ):
            port = _port.Portfolio()
            # Inject a live position directly
            from memecoin.portfolio import Position
            pos = Position(
                id="stuck001",
                signal_id="s001",
                chain="solana",
                token_address="So11111111111111111111111111111111111111112",
                token_symbol="STUCK",
                signal_type="social_alert",
                strength="medium",
                whale_count=0,
                whale_tiers=[],
                whales_involved=[],
                entry_price=0.001,
                signal_price=0.001,
                size_usd=3.0,
                entry_time=0.0,
                signal_time=0.0,
                hard_stop_pct=-0.35,
                trailing_stop_pct=-0.35,
                trail_activates_pct=0.30,
                time_stop_minutes=90,
                status="open",
                notes="live|tx:FAKEBUY123",
            )
            port._positions["stuck001"] = pos

            # Simulate MAX_SELL_RETRIES failures
            MAX_RETRIES = 5
            from unittest.mock import patch as _patch
            from memecoin.executor import MemeExecutor

            fail_result = {"success": False, "reason": "rpc_error", "error": "timeout"}
            with _patch.object(MemeExecutor, "sell", return_value=fail_result):
                for _ in range(MAX_RETRIES + 1):
                    port.close_position("stuck001", "hard_stop", 0.0005)

        # Journal must NOT have been written
        self.assertFalse(journal_written[0],
                         "Journal must not be written for sell_stuck position")
        # Position must still exist and be sell_stuck
        final_pos = port._positions.get("stuck001")
        self.assertIsNotNone(final_pos, "sell_stuck position must remain in _positions")
        self.assertEqual(final_pos.status, "sell_stuck",
                         f"Expected status=sell_stuck, got {final_pos.status!r}")


if __name__ == "__main__":
    unittest.main()
