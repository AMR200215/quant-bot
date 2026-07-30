"""
F4 — Classifier UNKNOWN TTL (B6a).

Verifies that UNKNOWN/error results from classify_mint() expire after 60s
so a simultaneous 429 on all RPC tiers does not permanently block a mint.

Test:
  1. Mock all 3 RPC tiers to return 429 → classify_mint → UNKNOWN
  2. Advance clock 61s
  3. Mock RPC to return healthy SPL owner
  4. classify_mint → must return SPL (cache TTL expired, not stuck on UNKNOWN)
"""
import sys
import os
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from memecoin import mint_classifier  # noqa: E402


_MINT = "TestMint1111111111111111111111111111111111"


def _rpc_429(*args, **kwargs):
    """Simulate all tiers returning 429 (connection error equivalent)."""
    return {"owner": None, "data": {}, "error": "rpc_429_or_503 url=all_tiers", "commitment": "confirmed"}


def _rpc_spl(*args, **kwargs):
    """Simulate healthy RPC returning SPL token program."""
    return {
        "owner": mint_classifier.SPL_TOKEN_PROGRAM_ID,
        "data": {},
        "error": None,
        "commitment": "confirmed",
    }


class TestF4ClassifierTTL(unittest.TestCase):

    def setUp(self):
        mint_classifier.clear_cache()

    def test_unknown_expires_after_60s(self):
        """UNKNOWN result from all-tiers-429 must not be cached beyond 60s."""
        # Step 1: all tiers 429 → UNKNOWN
        with patch.object(mint_classifier, "_fetch_mint_owner", side_effect=_rpc_429):
            result1 = mint_classifier.classify_mint(_MINT)

        self.assertEqual(result1.token_program, "UNKNOWN")
        self.assertEqual(result1.detection_source, "rpc_live")

        # Step 2: advance clock 61s past the cache entry's TTL
        # _put_cached stores cache_until = time.time() + 60.0
        # We need _get_cached to see time > cache_until.
        cache_entry = mint_classifier._cache.get(_MINT)
        self.assertIsNotNone(cache_entry, "UNKNOWN result must be cached with 60s TTL")
        _, cache_until = cache_entry
        self.assertNotEqual(cache_until, float("inf"), "UNKNOWN must NOT be cached permanently")
        self.assertAlmostEqual(cache_until - time.time(), 60.0, delta=2.0)

        # Patch time.time to advance 61s
        original_time = time.time()
        with patch("memecoin.mint_classifier.time") as mock_time:
            mock_time.time.return_value = original_time + 61.0
            mock_time.monotonic.return_value = time.monotonic() + 61.0

            # Step 3: healthy RPC now available
            with patch.object(mint_classifier, "_fetch_mint_owner", side_effect=_rpc_spl):
                result2 = mint_classifier.classify_mint(_MINT)

        # Step 4: must get SPL, not stuck on UNKNOWN
        self.assertEqual(result2.token_program, "SPL",
                         f"After TTL expiry, expected SPL but got {result2.token_program}")
        self.assertTrue(result2.is_tradeable)
        self.assertEqual(result2.detection_source, "rpc_live",
                         "Should re-hit RPC after TTL expiry, not serve stale cache")

    def test_spl_cached_permanently(self):
        """Definitive SPL result must be cached permanently (float('inf'))."""
        with patch.object(mint_classifier, "_fetch_mint_owner", side_effect=_rpc_spl):
            mint_classifier.classify_mint(_MINT)

        _, cache_until = mint_classifier._cache[_MINT]
        self.assertEqual(cache_until, float("inf"), "SPL must be cached permanently")

    def test_unknown_retried_before_ttl_returns_cached(self):
        """Second call within 60s must return cached UNKNOWN (no extra RPC hit)."""
        call_count = [0]

        def _counting_429(*a, **k):
            call_count[0] += 1
            return _rpc_429()

        with patch.object(mint_classifier, "_fetch_mint_owner", side_effect=_counting_429):
            mint_classifier.classify_mint(_MINT)
            result2 = mint_classifier.classify_mint(_MINT)   # should hit cache

        self.assertEqual(call_count[0], 1, "RPC should be called only once within TTL window")
        self.assertEqual(result2.detection_source, "cache")
        self.assertEqual(result2.token_program, "UNKNOWN")


if __name__ == "__main__":
    unittest.main(verbosity=2)
