"""
memecoin/tests/test_scanner_v8_dedup.py — V8-TWIN-FIX VF7 tests #8, #9.

Verifies _add_signal()'s dedup gate actually prevents maybe_open() from
being called twice for the same (chain, address, signal_type) within the
900s cooldown, and that a normal non-duplicate signal reaches maybe_open()
exactly once. This is the direct proof for VF3's question ("a TG PASS
alone is NOT proof that maybe_open ran") — dedup happens inside
_add_signal, after TG PASS and Signal construction, before V8 ever sees
the signal.

Run: python -m pytest memecoin/tests/test_scanner_v8_dedup.py -v
"""

import unittest
from unittest.mock import patch

import memecoin.scanner as scanner
from memecoin.signals import Signal


def _fresh_signal(event_id="ev1", token_address="TestMintXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"):
    return Signal(
        id="sig1", timestamp=0, chain="solana", token_address=token_address,
        token_name="Test", token_symbol="TEST", signal_type="social_alert",
        strength="weak", price_usd=0.00001, liquidity_usd=0, mcap_usd=0,
        volume_h1=0, volume_h24=0, age_minutes=1.0, event_id=event_id, dex_id="",
    )


class TestAddSignalDedup(unittest.TestCase):

    def setUp(self):
        scanner._seen.clear()
        scanner._traded_today.clear()

    def test_8_duplicate_signal_does_not_reach_maybe_open(self):
        sig = _fresh_signal()
        with patch.object(scanner.portfolio, "open_positions", return_value=[]), \
             patch("memecoin.v8_paper.book") as mock_book:
            scanner._add_signal(sig)
            scanner._add_signal(sig)   # same (chain, address, type) within 900s
            self.assertEqual(mock_book.maybe_open.call_count, 1,
                "duplicate signal must not cause a second maybe_open() call")

    def test_9_nonduplicate_signal_calls_maybe_open_exactly_once(self):
        sig = _fresh_signal(event_id="ev2", token_address="TestMintYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY")
        with patch.object(scanner.portfolio, "open_positions", return_value=[]), \
             patch("memecoin.v8_paper.book") as mock_book:
            scanner._add_signal(sig)
            self.assertEqual(mock_book.maybe_open.call_count, 1)
            mock_book.maybe_open.assert_called_once_with(sig)


if __name__ == "__main__":
    unittest.main()
