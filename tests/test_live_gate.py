"""
tests/test_live_gate.py

Verifies that the live gate opens/closes based on token_cohort, not signal_type
or dex_id.  Covers:

1. telegram_pump cohort → live-eligible (type-2 signal)
2. pumpfun_stream cohort → live-eligible (type-1 signal, previously paper-only)
3. No cohort → paper only (old copy_trade / volume_breakout signals)
4. Unknown cohort → paper only
5. LIVE_TRADING=False → always paper regardless of cohort
6. LIVE_DRY_RUN=True → executor.buy() returns dry_run=True without sending tx
"""

import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(
    signal_type: str = "social_alert",
    token_cohort: str = "telegram_pump",
    dex_id: str = "pumpfun",
):
    sig = MagicMock()
    sig.signal_type    = signal_type
    sig.token_cohort   = token_cohort
    sig.dex_id         = dex_id
    sig.id             = "test0001"
    sig.chain          = "solana"
    sig.token_address  = "So11111111111111111111111111111111111111112"
    sig.token_symbol   = "TEST"
    sig.strength       = "medium"
    sig.whale_count    = 0
    sig.whale_tiers    = []
    sig.whales_involved = []
    sig.composite_score = 0.5
    sig.creator_wallet  = "creator123"
    sig.price_usd       = 0.0001
    sig._price_pp       = 0.0001
    sig._price_dex      = 0.0001
    sig._price_source   = "pp"
    for attr in ("price_change_5m", "price_change_1h", "price_change_6h",
                 "buys_5m", "sells_5m", "buys_h1", "sells_h1",
                 "buy_sell_ratio_5m", "buy_sell_ratio_h1",
                 "volume_5m", "volume_h1", "volume_h6",
                 "liquidity_usd", "mcap_usd", "fdv", "age_minutes",
                 "safety_score", "momentum_score",
                 "rugcheck_score", "buy_tax", "sell_tax"):
        setattr(sig, attr, 0.0)
    for attr in ("has_twitter", "has_telegram", "has_website"):
        setattr(sig, attr, False)
    sig.dexscreener_url = ""
    sig._t_tg_receive  = 0
    sig._t_screen_end  = 0
    return sig


def _open_pos_with_live_gate(signal, *, live_trading: bool = True):
    """
    Call portfolio.open_position() and capture whether _open_live_position was called.
    Returns (paper_pos, live_called: bool).
    """
    from memecoin import portfolio as _port

    live_called = [False]
    orig_open_live = _port.Portfolio._open_live_position

    def _mock_open_live(self, sig, paper_pos):
        live_called[0] = True

    pp_mock = MagicMock()
    pp_mock.monitor.get_prices.return_value = {signal.token_address: signal.price_usd}

    with (
        patch("memecoin.portfolio.LIVE_TRADING", live_trading),
        patch("memecoin.portfolio.LIVE_DRY_RUN", True),
        patch("memecoin.portfolio._save_positions"),
        patch("memecoin.portfolio.alerts"),
        patch.object(_port.Portfolio, "_open_live_position", _mock_open_live),
        patch.dict("sys.modules", {
            "memecoin.pumpportal_monitor": pp_mock,
            "memecoin.pumpportal_monitor.monitor": pp_mock.monitor,
        }),
    ):
        port = _port.Portfolio()
        paper_pos = port.open_position(signal)

    return paper_pos, live_called[0]


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestLiveGateCohort(unittest.TestCase):

    def test_telegram_pump_cohort_is_live_eligible(self):
        """telegram_pump cohort → _open_live_position called."""
        sig = _make_signal(
            signal_type="social_alert",
            token_cohort="telegram_pump",
        )
        _, live_called = _open_pos_with_live_gate(sig)
        self.assertTrue(live_called, "telegram_pump cohort should be live-eligible")

    def test_pumpfun_stream_cohort_is_live_eligible(self):
        """pumpfun_stream cohort (type-1) → _open_live_position called."""
        sig = _make_signal(
            signal_type="pumpportal_screen",
            token_cohort="pumpfun_stream",
        )
        _, live_called = _open_pos_with_live_gate(sig)
        self.assertTrue(live_called, "pumpfun_stream cohort should be live-eligible")

    def test_no_cohort_is_paper_only(self):
        """No token_cohort (copy_trade / volume_breakout etc.) → paper only."""
        sig = _make_signal(
            signal_type="copy_trade",
            token_cohort="",
        )
        _, live_called = _open_pos_with_live_gate(sig)
        self.assertFalse(live_called, "signals with no cohort should be paper only")

    def test_unknown_cohort_is_paper_only(self):
        """Unknown cohort value → paper only (no unknown strings in gate)."""
        sig = _make_signal(
            signal_type="volume_breakout",
            token_cohort="some_other_cohort",
        )
        _, live_called = _open_pos_with_live_gate(sig)
        self.assertFalse(live_called, "unrecognised cohort should be paper only")

    def test_live_trading_false_always_paper(self):
        """LIVE_TRADING=False → paper only regardless of cohort."""
        sig = _make_signal(
            signal_type="social_alert",
            token_cohort="telegram_pump",
        )
        _, live_called = _open_pos_with_live_gate(sig, live_trading=False)
        self.assertFalse(live_called, "LIVE_TRADING=False should suppress all live trades")

    def test_dex_id_alone_does_not_open_gate(self):
        """
        Legacy: dex_id contains 'pump' but no token_cohort → paper only.
        Confirms we no longer key off dex_id.
        """
        sig = _make_signal(
            signal_type="social_alert",
            token_cohort="",      # no cohort
            dex_id="pumpfun",    # dex_id still has pump — old gate would have passed
        )
        _, live_called = _open_pos_with_live_gate(sig)
        self.assertFalse(live_called,
                         "dex_id alone should not open live gate — cohort is the criterion")


class TestDryRunExecutor(unittest.TestCase):
    """Item 6: executor.buy() returns dry_run=True without sending tx."""

    def test_buy_dry_run_returns_synthetic_success(self):
        """
        With LIVE_DRY_RUN=True, executor.buy() returns success+dry_run without
        touching the blockchain.
        """
        import memecoin.executor as ex_mod

        mock_keypair = MagicMock()
        mock_keypair.pubkey.return_value = "So11111111111111111111111111111111111111112"

        with (
            patch.object(ex_mod, "LIVE_DRY_RUN", True),
            patch.object(ex_mod, "_get_keypair", return_value=mock_keypair),
            patch.object(ex_mod, "_sol_price_usd", return_value=170.0),
            patch.object(ex_mod, "_sol_balance", return_value=500_000_000),  # 0.5 SOL
            patch.object(ex_mod, "_jup_get_quote", return_value={
                # size_usd=$3, signal_price=$0.003 → ~1000 tokens at 6 decimals
                "outAmount": "1000000000",
                "outputDecimals": 6,
            }),
            patch.object(ex_mod, "_load_solders", return_value=(MagicMock(), MagicMock(), MagicMock())),
        ):
            result = ex_mod.MemeExecutor().buy(
                "So11111111111111111111111111111111111111112",
                size_usd=3.0,
                signal_price=0.003,
            )

        self.assertTrue(result.get("success"), f"Expected success, got: {result}")
        self.assertTrue(result.get("dry_run"), "Expected dry_run=True")
        self.assertNotIn("real", str(result.get("tx_sig", "")),
                         "tx_sig should be synthetic DRY_RUN_* string")
        self.assertTrue(str(result.get("tx_sig", "")).startswith("DRY_RUN_"),
                        f"tx_sig should start with DRY_RUN_: {result.get('tx_sig')}")

    def test_sell_dry_run_returns_synthetic_success(self):
        """
        With LIVE_DRY_RUN=True, executor.sell() returns dry_run=True without
        touching the blockchain.
        """
        import memecoin.executor as ex_mod

        mock_keypair = MagicMock()
        mock_keypair.pubkey.return_value = "So11111111111111111111111111111111111111112"

        with (
            patch.object(ex_mod, "LIVE_DRY_RUN", True),
            patch.object(ex_mod, "_get_keypair", return_value=mock_keypair),
            patch.object(ex_mod, "_load_solders", return_value=(MagicMock(), MagicMock(), MagicMock())),
        ):
            result = ex_mod.MemeExecutor().sell(
                "So11111111111111111111111111111111111111112",
                size_usd=3.0,
                entry_price=0.003,
            )

        self.assertTrue(result.get("success"), f"Expected success, got: {result}")
        self.assertTrue(result.get("dry_run"), "Expected dry_run=True on sell")
        self.assertTrue(str(result.get("tx_sig", "")).startswith("DRY_RUN_"),
                        f"tx_sig should start with DRY_RUN_: {result.get('tx_sig')}")

    def test_insufficient_sol_blocks_before_dry_run(self):
        """
        Pre-buy balance check fires even in DRY_RUN mode: insufficient SOL
        returns insufficient_free_sol before reaching the DRY_RUN return.
        """
        import memecoin.executor as ex_mod

        mock_keypair = MagicMock()
        mock_keypair.pubkey.return_value = "So11111111111111111111111111111111111111112"

        with (
            patch.object(ex_mod, "LIVE_DRY_RUN", True),
            patch.object(ex_mod, "_get_keypair", return_value=mock_keypair),
            patch.object(ex_mod, "_sol_price_usd", return_value=170.0),
            patch.object(ex_mod, "_sol_balance", return_value=100),   # near-zero
            patch.object(ex_mod, "_load_solders", return_value=(MagicMock(), MagicMock(), MagicMock())),
        ):
            result = ex_mod.MemeExecutor().buy(
                "So11111111111111111111111111111111111111112",
                size_usd=3.0,
                signal_price=0.003,
            )

        self.assertFalse(result.get("success"))
        self.assertEqual(result.get("reason"), "insufficient_free_sol")


if __name__ == "__main__":
    unittest.main()
