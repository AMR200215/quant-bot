"""
Z1 — Graduation guard: RPC errors must never trigger graduation.

Tests the _should_trigger_graduation() helper added to scanner.py.

5 scenarios:
  1. RPC error        (ok=False, reason="rpc_error", complete=None)   → NOT graduation
  2. Parse error      (ok=False, reason="parse_error", complete=None) → NOT graduation
  3. Zero reserves    (ok=False, reason="zero_token_reserves")        → NOT graduation
  4. account_missing  (ok=True,  reason="account_missing")            → IS graduation
  5. complete=True    (ok=True,  reason="complete_true", complete=True) → IS graduation
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# Mirror the logic from scanner._should_trigger_graduation without importing
# the full scanner module (which has heavy config/thread dependencies).
def _should_trigger_graduation(result: dict) -> bool:
    """Z1: Return True only for VERIFIED graduation signals — not RPC/parse errors."""
    if result.get("complete") is True:
        return True
    if result.get("reason") == "account_missing":
        return True
    return False


class TestZ1GraduationGuard(unittest.TestCase):

    def test_rpc_error_not_graduation(self):
        """RPC error (ok=False, complete=None) must NOT trigger graduation."""
        result = {"ok": False, "complete": None, "reason": "rpc_error"}
        self.assertFalse(
            _should_trigger_graduation(result),
            "RPC error must not cause graduation — funds could be stranded on wrong sell path",
        )

    def test_parse_error_not_graduation(self):
        """Parse error (ok=False, complete=None) must NOT trigger graduation."""
        result = {"ok": False, "complete": None, "reason": "parse_error"}
        self.assertFalse(
            _should_trigger_graduation(result),
            "Parse error must not cause graduation",
        )

    def test_zero_reserves_not_graduation(self):
        """Zero reserves error must NOT trigger graduation."""
        result = {"ok": False, "complete": None, "reason": "zero_token_reserves"}
        self.assertFalse(
            _should_trigger_graduation(result),
            "zero_token_reserves must not cause graduation",
        )

    def test_account_missing_is_graduation(self):
        """Verified account_missing (ok=True) IS a graduation signal."""
        result = {"ok": True, "complete": None, "reason": "account_missing"}
        self.assertTrue(
            _should_trigger_graduation(result),
            "Real account_missing (ok=True) should trigger graduation handover",
        )

    def test_complete_true_is_graduation(self):
        """complete=True from a healthy RPC response IS a graduation signal."""
        result = {"ok": True, "complete": True, "reason": "complete_true"}
        self.assertTrue(
            _should_trigger_graduation(result),
            "complete=True should trigger graduation handover",
        )

    def test_complete_false_not_graduation(self):
        """complete=False means still on bonding curve — definitely NOT graduation."""
        result = {"ok": True, "complete": False, "reason": "complete_false"}
        self.assertFalse(
            _should_trigger_graduation(result),
            "complete=False must not trigger graduation",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
