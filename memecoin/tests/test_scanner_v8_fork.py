"""
memecoin/tests/test_scanner_v8_fork.py — V8-REWIRE VR1/VR5/VR18.

Replaces test_scanner_v8_dedup.py (V8-TWIN-FIX VF7 #8/#9), whose entire
premise -- that V8 only ever becomes reachable through _add_signal()'s
dedup gate, after V7's screen_token() has already run -- is exactly the
"V8 conditional on V7 screening" architecture bug the rewire fixes. That
premise is no longer true and a test asserting it would now be a stale
mock of the kind watchdog/checks/test_drift.py exists to catch.

The load-bearing proof for the whole rewire: memecoin.v8_paper.book.
maybe_open_from_alert() is reached for EVERY raw Telegram alert that
reaches _on_telegram_signal(), independent of whether V7's screen_token()
accepts or rejects it. Exercises the real function (not a re-implementation
of its logic) with screen_token mocked to reject and network prefetch
disabled, so no real HTTP calls happen.

Run: python -m pytest memecoin/tests/test_scanner_v8_fork.py -v
"""

import unittest
from unittest.mock import MagicMock, patch

import memecoin.scanner as scanner


class TestV8ForkIndependentOfV7Screening(unittest.TestCase):

    def setUp(self):
        # _submit_prefetch returning None is the module's own documented
        # "pool saturated" fallback -- screen_token() (itself mocked below)
        # would do its own sync fetch in real life, but since it's mocked
        # here, this just guarantees no real dex_get_token/rugcheck_sol
        # network calls happen during the test.
        self._prefetch_patch = patch.object(scanner, "_submit_prefetch", return_value=None)
        self._prefetch_patch.start()
        self._v8_book_patch = patch("memecoin.v8_paper.book")
        self.mock_v8_book = self._v8_book_patch.start()
        # Silence unrelated best-effort side channels (research snapshot
        # writes, telemetry, creator fetch) so the test only exercises the
        # V7-screen vs V8-fork independence question.
        self._creator_patch = patch.object(scanner, "_start_creator_fetch", return_value=(None, [None]))
        self._creator_patch.start()

    def tearDown(self):
        self._creator_patch.stop()
        self._v8_book_patch.stop()
        self._prefetch_patch.stop()

    def test_v7_rejection_still_reaches_v8_fork(self):
        """The exact bug this rewire fixes: a token V7's screen_token()
        rejects must still reach V8's fork. Pre-rewire, this token would
        never have reached maybe_open() at all."""
        with patch.object(scanner, "screen_token",
                           return_value={"reason": "low_buy_pressure", "buy_sell_ratio_5m": 0.1,
                                         "volume_5m": 5000, "volume_h1": 1000,
                                         "price_change_5m": 0, "mcap_usd": 10000}):
            scanner._on_telegram_signal("solana", "V7RejectMint111111111111111111111111", "test msg")

        self.mock_v8_book.maybe_open_from_alert.assert_called_once()
        event = self.mock_v8_book.maybe_open_from_alert.call_args[0][0]
        self.assertEqual(event.token_address, "V7RejectMint111111111111111111111111")
        self.assertEqual(event.chain, "solana")
        self.assertTrue(event.event_id)

    def test_v7_no_dex_data_rejection_still_reaches_v8_fork(self):
        with patch.object(scanner, "screen_token", return_value={"reason": "no_dex_data"}):
            try:
                scanner._on_telegram_signal("solana", "NoDexMint1111111111111111111111111111", "test msg")
            except scanner._NoDexData:
                pass   # expected -- propagates for TelegramMonitor's 45s retry, unrelated to V8
        self.mock_v8_book.maybe_open_from_alert.assert_called_once()

    def test_v7_pass_also_reaches_v8_fork_exactly_once(self):
        """V8 must not be double-invoked when V7 also happens to pass —
        the fork call site fires once, unconditionally, before V7
        branches at all."""
        with patch.object(scanner, "screen_token",
                           return_value={"reason": "", "buy_sell_ratio_5m": 5.0,
                                         "volume_5m": 5000, "volume_h1": 1000,
                                         "price_change_5m": 10, "mcap_usd": 10000,
                                         "price_usd": 0.00001, "dex_id": "pumpfun"}), \
             patch.object(scanner, "make_social_alert_signal") as mock_make_sig, \
             patch.object(scanner, "_add_signal") as mock_add_signal:
            mock_make_sig.return_value = MagicMock(
                chain="solana", token_address="V7PassMint1111111111111111111111111111",
                token_symbol="PASS", price_usd=0.00001,
            )
            scanner._on_telegram_signal("solana", "V7PassMint1111111111111111111111111111", "test msg")

        self.mock_v8_book.maybe_open_from_alert.assert_called_once()
        mock_add_signal.assert_called_once()   # V7's own downstream still runs normally

    def test_add_signal_no_longer_calls_v8_paper(self):
        # Call-like pattern, not a bare substring -- _add_signal()'s own
        # explanatory comment legitimately names v8_paper.py in prose.
        import inspect
        src = inspect.getsource(scanner._add_signal)
        self.assertNotIn("v8_paper.book", src,
            "_add_signal() must not call into v8_paper.book at all post-rewire — "
            "V8 is only reached via the fork call site in _on_telegram_signal()")
        self.assertNotIn("from memecoin.v8_paper import", src)


if __name__ == "__main__":
    unittest.main()
