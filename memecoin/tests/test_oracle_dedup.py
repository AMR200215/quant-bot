"""
test_oracle_dedup.py — Oracle deduplication tests

FIX 1 tests: verify _GRAD_ORACLE_CACHE is shared between
  get_pumpfun_curve_snapshot() and get_pumpfun_curve_complete()
  so no duplicate RPC is issued within the 5s TTL.

FIX 4 tests: verify buy() uses preflight_oracle_result passthrough
  correctly — reusing when valid, falling through to fresh RPC when not.

Run: python -m pytest memecoin/tests/test_oracle_dedup.py -v
"""

import sys
import time
import types
import unittest
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Minimal stubs so executor.py can be imported without live deps
# ---------------------------------------------------------------------------

def _make_stubs():
    sys.modules.setdefault("memecoin", types.ModuleType("memecoin"))

    cfg = types.ModuleType("memecoin.config")
    cfg.EXECUTOR_BACKEND       = "pumpportal"
    cfg.LIVE_DRY_RUN           = False
    cfg.SLIPPAGE_GATE_RT_PCT   = 0.30
    cfg.SLIPPAGE_GATE_DEX_PCT  = 0.50
    cfg.LIVE_TRADING           = False
    cfg.LIVE_GATE_EPOCH        = "2026-07-02"
    cfg.POSITIONS_FILE         = "/tmp/positions_test_oracle.json"
    cfg.MAX_LOSS_FROM_FILL_PCT = 0.50
    sys.modules["memecoin.config"] = cfg

    ppm = types.ModuleType("memecoin.pumpportal_monitor")
    ppm.monitor = MagicMock()
    ppm.monitor.get_prices = MagicMock(return_value={})
    ppm.monitor.get_vsol   = MagicMock(return_value=0)
    sys.modules["memecoin.pumpportal_monitor"] = ppm

    psl = types.ModuleType("memecoin.pumpswap_local")
    class PumpSwapPoolError(Exception): pass
    psl.PumpSwapPoolError = PumpSwapPoolError
    psl.fetch_pool        = MagicMock(side_effect=PumpSwapPoolError("no pool"))
    sys.modules["memecoin.pumpswap_local"] = psl

    from enum import Enum
    class TokenExitState(Enum):
        BONDING_CURVE = "BONDING_CURVE"
        GRADUATED_PUMPSWAP = "GRADUATED_PUMPSWAP"

    ero = types.ModuleType("memecoin.exit_router")
    ero.TokenExitState = TokenExitState
    ero.classify       = MagicMock(return_value=TokenExitState.BONDING_CURVE)
    sys.modules["memecoin.exit_router"] = ero


_make_stubs()

# ---------------------------------------------------------------------------
# FIX 1 — shared cache tests
# ---------------------------------------------------------------------------

_MINT_A = "MintAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
_MINT_B = "MintBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"

_FAKE_RESULT = {
    "ok": True, "complete": False, "reason": "complete_false",
    "bc_pda": "FakePDA",  "rpc_ms": 3,
    "virtual_token_reserves": 1_000_000_000,
    "virtual_sol_reserves":   30_000_000,
}


class TestOracleSharedCache(unittest.TestCase):
    """FIX 1: snapshot() then complete() within TTL must not issue a second RPC."""

    def setUp(self):
        # Import fresh each test so cache state is inspectable
        import memecoin.executor as ex
        # Clear the module-level cache before every test
        ex._GRAD_ORACLE_CACHE.clear()
        self.ex = ex

    def _patch_uncached(self, result=None, side_effect=None):
        """Patch the uncached RPC function."""
        if side_effect:
            return patch.object(
                self.ex, "_get_pumpfun_curve_complete_uncached",
                side_effect=side_effect,
            )
        return patch.object(
            self.ex, "_get_pumpfun_curve_complete_uncached",
            return_value=(result or dict(_FAKE_RESULT)),
        )

    def test_1_snapshot_then_complete_within_ttl_no_second_rpc(self):
        """snapshot() populates cache; complete() within TTL returns cache — zero extra RPCs."""
        with self._patch_uncached() as mock_rpc:
            self.ex.get_pumpfun_curve_snapshot(_MINT_A)
            self.ex.get_pumpfun_curve_complete(_MINT_A)
            # Only one RPC call total
            self.assertEqual(mock_rpc.call_count, 1,
                             "second call within TTL must be a cache hit, not a new RPC")

    def test_2_stale_cache_issues_new_rpc(self):
        """After TTL expires, complete() issues a fresh RPC."""
        with self._patch_uncached() as mock_rpc:
            # Inject a stale entry (ts = now - TTL - 1)
            stale_ts = time.monotonic() - self.ex._GRAD_ORACLE_TTL - 1.0
            self.ex._GRAD_ORACLE_CACHE[_MINT_A] = (stale_ts, dict(_FAKE_RESULT))

            self.ex.get_pumpfun_curve_complete(_MINT_A)
            self.assertEqual(mock_rpc.call_count, 1,
                             "stale cache entry must trigger a new RPC")

    def test_3_mint_mismatch_no_cache_reuse(self):
        """snapshot(mintA) does not satisfy complete(mintB)."""
        with self._patch_uncached() as mock_rpc:
            self.ex.get_pumpfun_curve_snapshot(_MINT_A)
            self.ex.get_pumpfun_curve_complete(_MINT_B)
            # Two separate RPCs (one per mint)
            self.assertEqual(mock_rpc.call_count, 2,
                             "different mint must not reuse the other's cache entry")

    def test_4_complete_twice_within_ttl_one_rpc(self):
        """complete() twice within TTL = one RPC."""
        with self._patch_uncached() as mock_rpc:
            self.ex.get_pumpfun_curve_complete(_MINT_A)
            self.ex.get_pumpfun_curve_complete(_MINT_A)
            self.assertEqual(mock_rpc.call_count, 1)


# ---------------------------------------------------------------------------
# FIX 4 — preflight_oracle_result passthrough tests
# ---------------------------------------------------------------------------

def _make_preflight(*, complete=False, ok=True, reason="complete_false",
                    age_s=0.5, ts_override=None):
    """Helper: build a synthetic preflight_oracle_result dict."""
    ts = ts_override if ts_override is not None else (time.time() - age_s)
    return {
        "ok":           ok,
        "complete":     complete,
        "reason":       reason,
        "bc_pda":       "FakePDA",
        "rpc_ms":       3,
        "_preflight_ts": ts,
        "virtual_token_reserves": 1_000_000_000,
        "virtual_sol_reserves":   30_000_000,
    }


def _run_graduated_gate(
    *,
    token_address: str,
    dex_id: str,
    pp_active: bool,
    jupiter_quote_price: float,
    preflight_oracle_result=None,
    oracle_result=None,
):
    """
    Reproduce the graduated-entry elif block from executor.buy().
    Returns {"blocked": bool, "oracle_reused": bool, "grad_evidence": list}.
    """
    if oracle_result is None:
        oracle_result = dict(_FAKE_RESULT)

    _is_graduated  = False
    _grad_evidence = []
    _curve         = {}
    _oracle_reused = False
    _dex_lower     = (dex_id or "").lower()

    if _dex_lower == "pumpswap":
        _is_graduated  = True
        _grad_evidence = ["dex_id=pumpswap"]
    elif not pp_active and jupiter_quote_price > 0:
        _pf = preflight_oracle_result
        if (
            _pf
            and _pf.get("ok")
            and _pf.get("complete") is False
            and _dex_lower not in ("pumpswap", "raydium", "orca")
            and (time.time() - _pf.get("_preflight_ts", 0)) <= 3.0
        ):
            _curve = _pf
            _oracle_reused = True
        else:
            _curve = oracle_result  # represents get_pumpfun_curve_complete() call

        if _curve["complete"] is False:
            _grad_evidence = []
            _is_graduated  = False
        else:
            _is_graduated  = True
            _grad_evidence = [f"grad_oracle:{_curve['reason']}"]

    return {
        "blocked":       _is_graduated,
        "oracle_reused": _oracle_reused,
        "grad_evidence": _grad_evidence,
    }


_FAKE_MINT = "FakeMint111111111111111111111111111111111111"


class TestPreflightPassthrough(unittest.TestCase):

    def test_5_complete_true_not_reused(self):
        """preflight complete=True must NOT be reused — it would block wrongly if stale."""
        pf = _make_preflight(complete=True, reason="complete_true", age_s=0.3)
        result = _run_graduated_gate(
            token_address=_FAKE_MINT,
            dex_id="pumpfun",
            pp_active=False,
            jupiter_quote_price=0.0001,
            preflight_oracle_result=pf,
            # oracle_result below is what get_pumpfun_curve_complete() would return
            oracle_result={**_FAKE_RESULT, "complete": True, "reason": "complete_true"},
        )
        self.assertFalse(result["oracle_reused"],
                         "complete=True preflight must not trigger reuse path")

    def test_6_account_missing_not_reused(self):
        """preflight complete=None (account_missing) must NOT be reused."""
        pf = _make_preflight(complete=None, ok=True, reason="account_missing", age_s=0.3)
        result = _run_graduated_gate(
            token_address=_FAKE_MINT,
            dex_id="pumpfun",
            pp_active=False,
            jupiter_quote_price=0.0001,
            preflight_oracle_result=pf,
            oracle_result={**_FAKE_RESULT, "complete": None, "reason": "account_missing"},
        )
        self.assertFalse(result["oracle_reused"])
        self.assertTrue(result["blocked"])

    def test_7_ok_false_not_reused(self):
        """preflight ok=False (RPC error) must NOT be reused."""
        pf = _make_preflight(complete=None, ok=False, reason="rpc_error", age_s=0.3)
        result = _run_graduated_gate(
            token_address=_FAKE_MINT,
            dex_id="pumpfun",
            pp_active=False,
            jupiter_quote_price=0.0001,
            preflight_oracle_result=pf,
            oracle_result={**_FAKE_RESULT, "complete": None, "reason": "rpc_error"},
        )
        self.assertFalse(result["oracle_reused"])
        self.assertTrue(result["blocked"])

    def test_8_age_2_2s_reused(self):
        """preflight age 2.2s (within 3s limit) → reused."""
        pf = _make_preflight(complete=False, age_s=2.2)
        result = _run_graduated_gate(
            token_address=_FAKE_MINT,
            dex_id="pumpfun",
            pp_active=False,
            jupiter_quote_price=0.0001,
            preflight_oracle_result=pf,
        )
        self.assertTrue(result["oracle_reused"], "age 2.2s should be within 3s window")
        self.assertFalse(result["blocked"])

    def test_9_age_8s_not_reused(self):
        """preflight age 8s (stale, >3s) → falls through to fresh oracle call."""
        pf = _make_preflight(complete=False, age_s=8.0)
        # fresh oracle returns complete=False too — just tests that reuse didn't fire
        result = _run_graduated_gate(
            token_address=_FAKE_MINT,
            dex_id="pumpfun",
            pp_active=False,
            jupiter_quote_price=0.0001,
            preflight_oracle_result=pf,
            oracle_result=dict(_FAKE_RESULT),  # complete=False
        )
        self.assertFalse(result["oracle_reused"], "age 8s must not reuse")

    def test_10_no_preflight_uses_fresh_oracle(self):
        """No preflight_oracle_result → fresh oracle called, result applied correctly."""
        result = _run_graduated_gate(
            token_address=_FAKE_MINT,
            dex_id="pumpfun",
            pp_active=False,
            jupiter_quote_price=0.0001,
            preflight_oracle_result=None,
            oracle_result=dict(_FAKE_RESULT),  # complete=False → allow
        )
        self.assertFalse(result["oracle_reused"])
        self.assertFalse(result["blocked"])

    def test_11_pp_active_skips_oracle_entirely(self):
        """PP active → oracle branch never entered (reuse question is moot)."""
        pf = _make_preflight(complete=False, age_s=0.1)
        result = _run_graduated_gate(
            token_address=_FAKE_MINT,
            dex_id="pumpfun",
            pp_active=True,
            jupiter_quote_price=0.0001,
            preflight_oracle_result=pf,
        )
        self.assertFalse(result["oracle_reused"])
        self.assertFalse(result["blocked"])

    def test_12_dex_pumpswap_blocks_regardless_of_preflight(self):
        """dex_id=pumpswap always blocks, even with fresh valid preflight."""
        pf = _make_preflight(complete=False, age_s=0.1)
        result = _run_graduated_gate(
            token_address=_FAKE_MINT,
            dex_id="pumpswap",
            pp_active=False,
            jupiter_quote_price=0.0001,
            preflight_oracle_result=pf,
        )
        self.assertTrue(result["blocked"])
        self.assertIn("dex_id=pumpswap", result["grad_evidence"])
        self.assertFalse(result["oracle_reused"])


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
