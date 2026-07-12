"""
Tests for C2 — evaluate_live_entry_program_gate() in memecoin.portfolio

10 tests. All call the production function directly (R6).
"""

import time
import unittest

from memecoin.mint_classifier import MintClassification
from memecoin.portfolio import evaluate_live_entry_program_gate


def _make_classification(
    token_program="SPL",
    unsupported_extensions=None,
    error=None,
    transfer_hook=False,
    transfer_fee=False,
):
    """Helper: build a MintClassification with the given fields."""
    if unsupported_extensions is None:
        unsupported_extensions = []
    return MintClassification(
        mint="Fake1111111111111111111111111111111111111111",
        mint_owner_program="TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
        token_program=token_program,
        token_extensions=[],
        unsupported_extensions=unsupported_extensions,
        transfer_hook_present=transfer_hook,
        transfer_fee_present=transfer_fee,
        policy_category="1_SPL_supported",
        detection_source="rpc_live",
        detection_timestamp_wall=time.time(),
        detection_timestamp_monotonic=time.monotonic(),
        rpc_commitment="confirmed",
        error=error,
    )


class TestEvaluateLiveEntryProgramGate(unittest.TestCase):

    # 1. SPL classification → allowed
    def test_spl_allowed(self):
        cls = _make_classification(token_program="SPL")
        result = evaluate_live_entry_program_gate(cls)
        self.assertTrue(result["allowed"])

    # 2. T22 classification → blocked with token_program="T22"
    def test_t22_blocked(self):
        cls = _make_classification(token_program="T22")
        result = evaluate_live_entry_program_gate(cls)
        self.assertFalse(result["allowed"])
        self.assertEqual(result["token_program"], "T22")

    # 3. UNKNOWN token_program → blocked
    def test_unknown_token_program_blocked(self):
        cls = _make_classification(token_program="UNKNOWN")
        result = evaluate_live_entry_program_gate(cls)
        self.assertFalse(result["allowed"])
        self.assertEqual(result["token_program"], "UNKNOWN")

    # 4. classification is None → blocked with reason containing "unknown"
    def test_none_classification_blocked(self):
        result = evaluate_live_entry_program_gate(None)
        self.assertFalse(result["allowed"])
        self.assertIn("unknown", result["reason"].lower())

    # 5. classification has error set → blocked
    def test_error_set_blocked(self):
        cls = _make_classification(token_program="SPL", error="rpc_timeout")
        result = evaluate_live_entry_program_gate(cls)
        self.assertFalse(result["allowed"])

    # 6. is_tradeable=False (unsupported extensions) → blocked
    def test_unsupported_extensions_blocked(self):
        cls = _make_classification(
            token_program="T22",
            unsupported_extensions=["TransferHook"],
            transfer_hook=True,
        )
        # is_tradeable = False because unsupported_extensions is non-empty
        self.assertFalse(cls.is_tradeable)
        result = evaluate_live_entry_program_gate(cls)
        self.assertFalse(result["allowed"])

    # 7. is_tradeable=True with SPL → allowed
    def test_spl_tradeable_allowed(self):
        cls = _make_classification(token_program="SPL", unsupported_extensions=[])
        self.assertTrue(cls.is_tradeable)
        result = evaluate_live_entry_program_gate(cls)
        self.assertTrue(result["allowed"])

    # 8. T22 + complete=True in curve_observation → still blocked (no T22 buys)
    def test_t22_with_complete_curve_still_blocked(self):
        cls = _make_classification(token_program="T22")
        curve_obs = {"complete": True}
        result = evaluate_live_entry_program_gate(cls, curve_observation=curve_obs)
        self.assertFalse(result["allowed"])
        self.assertEqual(result["token_program"], "T22")

    # 9. SPL + complete=True in curve_observation → allowed (graduation state doesn't affect entry gate)
    def test_spl_with_complete_curve_allowed(self):
        cls = _make_classification(token_program="SPL")
        curve_obs = {"complete": True}
        result = evaluate_live_entry_program_gate(cls, curve_observation=curve_obs)
        self.assertTrue(result["allowed"])

    # 10. evaluate_live_entry_program_gate is importable from memecoin.portfolio
    def test_importable_from_portfolio(self):
        from memecoin.portfolio import evaluate_live_entry_program_gate as _fn
        self.assertTrue(callable(_fn))


if __name__ == "__main__":
    unittest.main()
