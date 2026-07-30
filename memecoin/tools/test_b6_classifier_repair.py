"""B6: Classifier TTL, allowlist, and integration."""
import sys, os, time, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class TestB6ClassifierRepair(unittest.TestCase):

    def setUp(self):
        from memecoin.mint_classifier import clear_cache
        clear_cache()

    def test_spl_cached_permanently(self):
        """SPL results have infinite TTL."""
        from unittest.mock import patch
        from memecoin.mint_classifier import (
            classify_mint, clear_cache,
            SPL_TOKEN_PROGRAM_ID, _cache, _cache_lock
        )
        clear_cache()
        with patch("memecoin.mint_classifier._fetch_mint_owner",
                   return_value={"owner": SPL_TOKEN_PROGRAM_ID, "data": {}, "error": None}):
            classify_mint("mintSPLperm1111111111111111111111111111111111")
        import threading
        with _cache_lock:
            entry = _cache.get("mintSPLperm1111111111111111111111111111111111")
        self.assertIsNotNone(entry)
        _, cache_until = entry
        self.assertEqual(cache_until, float("inf"), "SPL must be cached permanently")

    def test_unknown_cached_with_ttl(self):
        """UNKNOWN results have 60s TTL (not permanent)."""
        from unittest.mock import patch
        from memecoin.mint_classifier import classify_mint, _cache, _cache_lock
        with patch("memecoin.mint_classifier._fetch_mint_owner",
                   return_value={"owner": None, "data": {}, "error": "timeout"}):
            classify_mint("mintUNKtll111111111111111111111111111111111111")
        with _cache_lock:
            entry = _cache.get("mintUNKtll111111111111111111111111111111111111")
        self.assertIsNotNone(entry)
        _, cache_until = entry
        self.assertLess(cache_until, float("inf"), "UNKNOWN must have finite TTL")
        self.assertAlmostEqual(cache_until - time.time(), 60, delta=2)

    def test_unknown_cache_expires(self):
        """UNKNOWN cache entry expires after TTL."""
        from unittest.mock import patch
        from memecoin.mint_classifier import classify_mint, _cache, _cache_lock

        call_count = [0]
        def _fetch(mint):
            call_count[0] += 1
            return {"owner": None, "data": {}, "error": "timeout"}

        # Manually inject a stale entry
        from memecoin.mint_classifier import MintClassification, _put_cached
        stale = MintClassification(
            mint="mintEXP11111111111111111111111111111111111111",
            mint_owner_program="",
            token_program="UNKNOWN",
            token_extensions=[],
            unsupported_extensions=[],
            transfer_hook_present=False,
            transfer_fee_present=False,
            policy_category="6_UNKNOWN_token_program",
            detection_source="rpc_live",
            detection_timestamp_wall=time.time() - 120,
            detection_timestamp_monotonic=0,
            rpc_commitment="confirmed",
            error="timeout",
        )
        with _cache_lock:
            _cache["mintEXP11111111111111111111111111111111111111"] = (stale, time.time() - 1)

        with patch("memecoin.mint_classifier._fetch_mint_owner", side_effect=_fetch):
            classify_mint("mintEXP11111111111111111111111111111111111111")
        self.assertEqual(call_count[0], 1, "Expired UNKNOWN entry must trigger re-fetch")

    def test_unknown_extension_not_tradeable(self):
        """Unrecognized extension produces UNKNOWN_EXTENSION policy, not tradeable."""
        from unittest.mock import patch
        from memecoin.mint_classifier import classify_mint, TOKEN_2022_PROGRAM_ID
        data = {"parsed": {"info": {"extensions": [{"extension": "WeirdNewExtension2026"}]}}}
        with patch("memecoin.mint_classifier._fetch_mint_owner",
                   return_value={"owner": TOKEN_2022_PROGRAM_ID, "data": data, "error": None}):
            r = classify_mint("mintWEIRD111111111111111111111111111111111111")
        self.assertFalse(r.is_tradeable, "Unknown extension must block trading")
        self.assertIn("unknown_extension", (r.error or "").lower())

    def test_known_extension_not_blocked(self):
        """MetadataPointer is a known safe extension — not blocked."""
        from unittest.mock import patch
        from memecoin.mint_classifier import classify_mint, TOKEN_2022_PROGRAM_ID
        data = {"parsed": {"info": {"extensions": [{"extension": "MetadataPointer"}, {"extension": "TokenMetadata"}]}}}
        with patch("memecoin.mint_classifier._fetch_mint_owner",
                   return_value={"owner": TOKEN_2022_PROGRAM_ID, "data": data, "error": None}):
            r = classify_mint("mintMETA1111111111111111111111111111111111111")
        self.assertTrue(r.is_tradeable, "Known extensions must not block trading")

if __name__ == "__main__":
    unittest.main(verbosity=2)
