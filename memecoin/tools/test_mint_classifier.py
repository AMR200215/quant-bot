"""Basic tests for authoritative mint classification (Part 3)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import unittest
from unittest.mock import patch

from memecoin.mint_classifier import (
    classify_mint, get_token_program, clear_cache,
    SPL_TOKEN_PROGRAM_ID, TOKEN_2022_PROGRAM_ID,
)


def _mock_fetch(owner, extensions=None, error=None):
    data = {}
    if extensions and owner == TOKEN_2022_PROGRAM_ID:
        data = {"parsed": {"info": {"extensions": [{"extension": e} for e in extensions]}}}
    return {"owner": owner, "data": data, "error": error, "commitment": "confirmed"}


class TestMintClassifier(unittest.TestCase):

    def setUp(self):
        clear_cache()

    def test_spl_detection(self):
        with patch("memecoin.mint_classifier._fetch_mint_owner",
                   return_value=_mock_fetch(SPL_TOKEN_PROGRAM_ID)):
            r = classify_mint("mintSPL111111111111111111111111111111111111")
        self.assertEqual(r.token_program, "SPL")
        self.assertTrue(r.is_spl)
        self.assertFalse(r.is_t22)
        self.assertTrue(r.is_tradeable)

    def test_t22_detection(self):
        with patch("memecoin.mint_classifier._fetch_mint_owner",
                   return_value=_mock_fetch(TOKEN_2022_PROGRAM_ID)):
            r = classify_mint("mintT22_111111111111111111111111111111111111")
        self.assertEqual(r.token_program, "T22")
        self.assertTrue(r.is_t22)
        self.assertTrue(r.is_tradeable)

    def test_unknown_detection(self):
        with patch("memecoin.mint_classifier._fetch_mint_owner",
                   return_value=_mock_fetch("SomeRandomProgram111111111111111111111111")):
            r = classify_mint("mintUNK111111111111111111111111111111111111")
        self.assertEqual(r.token_program, "UNKNOWN")
        self.assertFalse(r.is_tradeable)

    def test_cache_reused_same_mint(self):
        calls = 0
        def _fetch(mint):
            nonlocal calls
            calls += 1
            return _mock_fetch(SPL_TOKEN_PROGRAM_ID)
        with patch("memecoin.mint_classifier._fetch_mint_owner", side_effect=_fetch):
            classify_mint("mintCACHE11111111111111111111111111111111111")
            classify_mint("mintCACHE11111111111111111111111111111111111")
        self.assertEqual(calls, 1, "Cache must be reused for same mint")

    def test_cache_not_reused_across_mints(self):
        calls = 0
        def _fetch(mint):
            nonlocal calls
            calls += 1
            return _mock_fetch(SPL_TOKEN_PROGRAM_ID)
        with patch("memecoin.mint_classifier._fetch_mint_owner", side_effect=_fetch):
            classify_mint("mintAAAA1111111111111111111111111111111111111")
            classify_mint("mintBBBB1111111111111111111111111111111111111")
        self.assertEqual(calls, 2, "Cache must NOT be reused across different mints")

    def test_suffix_cannot_override_rpc(self):
        """pump suffix does NOT mean SPL — only RPC owner matters."""
        with patch("memecoin.mint_classifier._fetch_mint_owner",
                   return_value=_mock_fetch(TOKEN_2022_PROGRAM_ID)):
            r = classify_mint("HEb9cW7Q9KsHauTrA8QzpFnZHu1o93t6QWCXNsyMpump")
        # Even though it ends with "pump", RPC says T22
        self.assertEqual(r.token_program, "T22")

    def test_transfer_hook_blocks_trading(self):
        with patch("memecoin.mint_classifier._fetch_mint_owner",
                   return_value=_mock_fetch(TOKEN_2022_PROGRAM_ID, ["TransferHook"])):
            r = classify_mint("mintHOOK1111111111111111111111111111111111111")
        self.assertTrue(r.transfer_hook_present)
        self.assertFalse(r.is_tradeable)
        self.assertEqual(r.policy_category, "4_T22_transfer_hook_unsupported")

    def test_transfer_fee_supported(self):
        with patch("memecoin.mint_classifier._fetch_mint_owner",
                   return_value=_mock_fetch(TOKEN_2022_PROGRAM_ID, ["TransferFeeConfig"])):
            r = classify_mint("mintFEE11111111111111111111111111111111111111")
        self.assertTrue(r.transfer_fee_present)
        # Transfer fee doesn't block trading (executor handles it)
        self.assertTrue(r.is_tradeable)
        self.assertEqual(r.policy_category, "3_T22_transfer_fee")

    def test_rpc_error_yields_unknown(self):
        with patch("memecoin.mint_classifier._fetch_mint_owner",
                   return_value={"owner": None, "data": {}, "error": "timeout"}):
            r = classify_mint("mintERR11111111111111111111111111111111111111")
        self.assertEqual(r.token_program, "UNKNOWN")
        self.assertFalse(r.is_tradeable)
        self.assertIsNotNone(r.error)

    def test_extensions_parsed_and_persisted(self):
        exts = ["TransferFeeConfig", "MetadataPointer", "TokenMetadata"]
        with patch("memecoin.mint_classifier._fetch_mint_owner",
                   return_value=_mock_fetch(TOKEN_2022_PROGRAM_ID, exts)):
            r = classify_mint("mintEXT11111111111111111111111111111111111111")
        self.assertIn("TransferFeeConfig", r.token_extensions)
        self.assertIn("MetadataPointer", r.token_extensions)

    def test_unknown_extension_explicit(self):
        """Unknown extension produces explicit policy, never silently treated as SPL."""
        with patch("memecoin.mint_classifier._fetch_mint_owner",
                   return_value=_mock_fetch(TOKEN_2022_PROGRAM_ID, ["ConfidentialTransferMint"])):
            r = classify_mint("mintCONF1111111111111111111111111111111111111")
        self.assertFalse(r.is_tradeable)
        self.assertIn("ConfidentialTransferMint", r.unsupported_extensions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
