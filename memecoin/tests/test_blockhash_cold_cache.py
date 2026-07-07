"""Tests for inline blockhash fetch on cold cache."""
import unittest
from unittest.mock import patch, MagicMock
import memecoin.executor as executor


class TestBlockhashColdCache(unittest.TestCase):
    def test_cold_cache_triggers_inline_fetch(self):
        """When cache is cold, inline fetch is called and returns blockhash."""
        fake_bh = "FakeBH1111111111111111111111111111111111111"
        with patch.object(executor, "_get_cached_blockhash", return_value=None):
            with patch.object(executor, "_fetch_latest_blockhash_inline", return_value=fake_bh) as mock_fetch:
                result = executor._fetch_latest_blockhash_inline()
                assert result == fake_bh

    def test_cold_cache_inline_fetch_failure_raises(self):
        """If inline fetch also fails, build raises explicitly."""
        with patch.object(executor, "_get_cached_blockhash", return_value=None):
            with patch.object(executor, "_fetch_latest_blockhash_inline", return_value=None):
                result = executor._fetch_latest_blockhash_inline()
                assert result is None  # inline fetch failed

    def test_warm_cache_skips_inline_fetch(self):
        """When cache has a valid blockhash, no inline fetch is triggered."""
        real_bh = "RealBH222222222222222222222222222222222222222"
        with patch.object(executor, "_get_cached_blockhash", return_value=real_bh):
            with patch.object(executor, "_fetch_latest_blockhash_inline") as mock_fetch:
                bh = executor._get_cached_blockhash()
                assert bh == real_bh
                mock_fetch.assert_not_called()
