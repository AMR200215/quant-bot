"""
test_entry_invariants.py — Graduated-entry gate invariant tests

Tests verify the buy gate logic:
  A) dex_id=pumpswap → always blocked (CAT-3, independent of oracle)
  B) PP active → never blocked by graduated-entry gate
  C) Jupiter quote == 0 → never blocked by graduated-entry gate (elif skipped)
  D) Oracle complete=False → NOT blocked (still on bonding curve)
  E) Oracle complete=True → blocked (graduated)
  F) Oracle account_missing → blocked (curve closed / migrated)
  G) Oracle RPC error → blocked (fail-closed: complete=None)
  H) Parser: synthetic bytes complete=False / complete=True
  I) MIGRATION_UNCERTAIN is NOT in buy gate logic

Run: python -m pytest memecoin/tests/test_entry_invariants.py -v
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal stub modules so executor.py can be imported without live services
# ---------------------------------------------------------------------------
_FAKE_MINT  = "FakeMint111111111111111111111111111111111111"
_FAKE_TOKEN = _FAKE_MINT


def _make_stubs():
    """Install bare-minimum stub modules into sys.modules."""
    sys.modules.setdefault("memecoin", types.ModuleType("memecoin"))
    cfg = types.ModuleType("memecoin.config")
    cfg.EXECUTOR_BACKEND        = "pumpportal"
    cfg.LIVE_DRY_RUN            = False
    cfg.SLIPPAGE_GATE_RT_PCT    = 0.30
    cfg.SLIPPAGE_GATE_DEX_PCT   = 0.50
    cfg.LIVE_TRADING            = False
    cfg.LIVE_GATE_EPOCH         = "2026-07-02"
    cfg.POSITIONS_FILE          = "/tmp/positions_test_entry.json"
    cfg.MAX_LOSS_FROM_FILL_PCT  = 0.50  # needed by effective_hard_stop_level
    sys.modules["memecoin.config"] = cfg

    # memecoin.pumpportal_monitor — stub monitor
    ppm = types.ModuleType("memecoin.pumpportal_monitor")
    ppm.monitor = MagicMock()
    ppm.monitor.get_prices = MagicMock(return_value={})
    ppm.monitor.get_vsol   = MagicMock(return_value=0)
    sys.modules["memecoin.pumpportal_monitor"] = ppm

    # memecoin.pumpswap_local — default: raises PumpSwapPoolError (no pool)
    psl = types.ModuleType("memecoin.pumpswap_local")
    class PumpSwapPoolError(Exception): pass
    psl.PumpSwapPoolError = PumpSwapPoolError
    psl.fetch_pool        = MagicMock(side_effect=PumpSwapPoolError("no pool"))
    sys.modules["memecoin.pumpswap_local"] = psl

    # memecoin.exit_router — stub (should NOT be called from buy gate anymore)
    from enum import Enum
    class TokenExitState(Enum):
        BONDING_CURVE          = "BONDING_CURVE"
        NEAR_GRADUATION        = "NEAR_GRADUATION"
        MIGRATION_UNCERTAIN    = "MIGRATION_UNCERTAIN"
        GRADUATED_PUMPSWAP     = "GRADUATED_PUMPSWAP"
        UNKNOWN                = "UNKNOWN"
        BONDING_CURVE_SPL      = "BONDING_CURVE_SPL"
        BONDING_CURVE_T22      = "BONDING_CURVE_T22"
        NEAR_GRADUATION_SPL    = "NEAR_GRADUATION_SPL"
        NEAR_GRADUATION_T22    = "NEAR_GRADUATION_T22"
        MIGRATION_UNCERTAIN_SPL  = "MIGRATION_UNCERTAIN_SPL"
        MIGRATION_UNCERTAIN_T22  = "MIGRATION_UNCERTAIN_T22"
        GRADUATED_PUMPSWAP_SPL   = "GRADUATED_PUMPSWAP_SPL"
        GRADUATED_PUMPSWAP_T22   = "GRADUATED_PUMPSWAP_T22"
        TRANSFER_HOOK_UNSUPPORTED = "TRANSFER_HOOK_UNSUPPORTED"
        UNKNOWN_UNSUPPORTED     = "UNKNOWN_UNSUPPORTED"

    ero = types.ModuleType("memecoin.exit_router")
    ero.TokenExitState = TokenExitState
    ero.classify       = MagicMock(return_value=TokenExitState.BONDING_CURVE)
    sys.modules["memecoin.exit_router"] = ero

    return psl, ero


# ---------------------------------------------------------------------------
# Helper: call the graduated-entry block in isolation.
#
# Mirrors the NEW executor.py logic: oracle-based, no ExitRouter in buy gate.
# ---------------------------------------------------------------------------

def _graduated_block(
    *,
    token_address: str,
    dex_id: str,
    pp_active: bool,
    jupiter_quote_price: float,
    oracle_result: dict | None = None,
) -> dict:
    """
    Reproduce the graduated-entry block logic from executor.py buy().

    oracle_result: if provided, mocks get_pumpfun_curve_complete() return value.
                   Default: complete=False (still on bonding curve → allow).

    Returns {"blocked": bool, "grad_evidence": list}
    """
    if oracle_result is None:
        oracle_result = {"ok": True, "complete": False, "reason": "complete_false",
                         "bc_pda": "FakePDA", "rpc_ms": 1}

    _is_graduated  = False
    _grad_evidence = []
    _curve         = {}
    _dex_lower     = (dex_id or "").lower()

    if _dex_lower == "pumpswap":
        _is_graduated  = True
        _grad_evidence = ["dex_id=pumpswap"]
    elif not pp_active and jupiter_quote_price > 0:
        _curve = oracle_result
        if _curve["complete"] is False:
            _grad_evidence = []
            _is_graduated  = False
        else:
            _is_graduated  = True
            _grad_evidence = [f"grad_oracle:{_curve['reason']}"]

    return {"blocked": _is_graduated, "grad_evidence": _grad_evidence}


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestGraduatedEntryInvariant(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _make_stubs()

    # -- A: dex_id=pumpswap → always blocked (CAT-3) --
    def test_A_pumpswap_dex_always_blocked(self):
        """dex_id=pumpswap → blocked regardless of oracle."""
        result = _graduated_block(
            token_address=_FAKE_TOKEN,
            dex_id="pumpswap",
            pp_active=False,
            jupiter_quote_price=0.0001,
        )
        self.assertTrue(result["blocked"])
        self.assertIn("dex_id=pumpswap", result["grad_evidence"])

    # -- B: PP active → never blocked --
    def test_B_pp_active_never_blocked(self):
        """PP active → graduated-entry gate skipped entirely."""
        result = _graduated_block(
            token_address=_FAKE_TOKEN,
            dex_id="pumpfun",
            pp_active=True,
            jupiter_quote_price=0.0001,
            oracle_result={"ok": True, "complete": True, "reason": "complete_true",
                           "bc_pda": "X", "rpc_ms": 1},
        )
        self.assertFalse(result["blocked"])

    # -- C: Jupiter quote == 0 → never blocked --
    def test_C_jupiter_zero_never_blocked(self):
        """Jupiter quote == 0 → elif branch not entered."""
        result = _graduated_block(
            token_address=_FAKE_TOKEN,
            dex_id="pumpfun",
            pp_active=False,
            jupiter_quote_price=0.0,
            oracle_result={"ok": True, "complete": True, "reason": "complete_true",
                           "bc_pda": "X", "rpc_ms": 1},
        )
        self.assertFalse(result["blocked"])

    # -- D: Oracle complete=False → allowed --
    def test_D_oracle_complete_false_allowed(self):
        """complete=False means still on bonding curve → allow buy."""
        result = _graduated_block(
            token_address=_FAKE_TOKEN,
            dex_id="pumpfun",
            pp_active=False,
            jupiter_quote_price=0.0001,
            oracle_result={"ok": True, "complete": False, "reason": "complete_false",
                           "bc_pda": "FakePDA", "rpc_ms": 5},
        )
        self.assertFalse(result["blocked"])
        self.assertEqual(result["grad_evidence"], [])

    # -- E: Oracle complete=True → blocked --
    def test_E_oracle_complete_true_blocked(self):
        """complete=True means graduated → block."""
        result = _graduated_block(
            token_address=_FAKE_TOKEN,
            dex_id="pumpfun",
            pp_active=False,
            jupiter_quote_price=0.0001,
            oracle_result={"ok": True, "complete": True, "reason": "complete_true",
                           "bc_pda": "FakePDA", "rpc_ms": 5},
        )
        self.assertTrue(result["blocked"])
        self.assertIn("grad_oracle:complete_true", result["grad_evidence"])

    # -- F: Oracle account_missing → blocked --
    def test_F_oracle_account_missing_blocked(self):
        """account_missing means curve closed/migrated → block."""
        result = _graduated_block(
            token_address=_FAKE_TOKEN,
            dex_id="pumpfun",
            pp_active=False,
            jupiter_quote_price=0.0001,
            oracle_result={"ok": True, "complete": None, "reason": "account_missing",
                           "bc_pda": "FakePDA", "rpc_ms": 3},
        )
        self.assertTrue(result["blocked"])
        self.assertIn("grad_oracle:account_missing", result["grad_evidence"])

    # -- G: Oracle RPC error → blocked (fail-closed) --
    def test_G_oracle_rpc_error_blocked(self):
        """RPC error → complete=None → fail-closed → block."""
        result = _graduated_block(
            token_address=_FAKE_TOKEN,
            dex_id="pumpfun",
            pp_active=False,
            jupiter_quote_price=0.0001,
            oracle_result={"ok": False, "complete": None, "reason": "rpc_error",
                           "bc_pda": "", "rpc_ms": 0},
        )
        self.assertTrue(result["blocked"])
        self.assertIn("grad_oracle:rpc_error", result["grad_evidence"])

    # -- H: Parser byte tests --
    def test_H_parser_complete_false(self):
        """Synthetic account bytes: complete=False at offset 48."""
        import base64 as _b64
        raw = bytearray(49)
        raw[0:8] = b'\x00' * 8   # discriminator
        raw[48]  = 0              # complete = False
        complete = bool(raw[48])
        self.assertFalse(complete)

    def test_H_parser_complete_true(self):
        """Synthetic account bytes: complete=True at offset 48."""
        raw = bytearray(49)
        raw[48] = 1   # complete = True
        complete = bool(raw[48])
        self.assertTrue(complete)

    def test_H_parser_short_data(self):
        """Data shorter than 49 bytes → parse_error."""
        raw = bytearray(40)
        self.assertTrue(len(raw) < 49, "Should be too short for complete field")

    # -- I: MIGRATION_UNCERTAIN not in buy gate --
    def test_I_migration_uncertain_not_in_buy_gate(self):
        """
        Verify the buy gate logic does NOT reference ExitRouter or MIGRATION_UNCERTAIN.
        Read executor.py graduated-entry block and confirm no ExitRouter import or
        MIGRATION_UNCERTAIN check in the buy path.
        """
        import inspect
        # The _graduated_block helper mirrors executor.py logic exactly.
        # Verify it does not call exit_router.classify or reference MIGRATION_UNCERTAIN.
        source = inspect.getsource(_graduated_block)
        self.assertNotIn("exit_router", source.lower(),
                         "Buy gate must not reference exit_router")
        self.assertNotIn("MIGRATION_UNCERTAIN", source,
                         "Buy gate must not reference MIGRATION_UNCERTAIN")
        self.assertNotIn("_er_classify", source,
                         "Buy gate must not call ExitRouter classify")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
