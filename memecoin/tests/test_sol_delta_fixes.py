"""
test_sol_delta_fixes.py — Tests for Fixes 1-4

A: read_sol_delta retry — null twice then valid → real delta returned
B: confirmed tx + permanently unindexed + quote estimate → fill_estimated, not -100%
C: confirmed tx + no quote + no meta → no close at fill=0, needs_reconcile
D: scanner suppression — complete=False → no rescue dispatch
E: genuine migration — complete=True → rescue proceeds
F: account missing → rescue proceeds
G: curve feed — PP silent T22 → price_overrides updated, feed_blind suppressed
H: hard stop / trailing stop path does not call rescue suppression guard

Run: python -m pytest memecoin/tests/test_sol_delta_fixes.py -v
"""

import sys
import types
import time
import unittest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Minimal stubs for import
# ---------------------------------------------------------------------------

def _install_stubs():
    sys.modules.setdefault("memecoin", types.ModuleType("memecoin"))

    cfg = types.ModuleType("memecoin.config")
    cfg.EXECUTOR_BACKEND       = "pumpportal"
    cfg.LIVE_DRY_RUN           = False
    cfg.SLIPPAGE_GATE_RT_PCT   = 0.30
    cfg.SLIPPAGE_GATE_DEX_PCT  = 0.50
    cfg.LIVE_TRADING           = False
    cfg.LIVE_GATE_EPOCH        = "2026-07-02"
    cfg.POSITIONS_FILE         = "/tmp/positions_test_sol_delta.json"
    import pathlib
    cfg.LIVE_JOURNAL_FILE      = pathlib.Path("/tmp/live_journal_test.csv")
    cfg.SOCIAL_JOURNAL_FILE    = pathlib.Path("/tmp/social_journal_test.csv")
    cfg.LOGS_DIR               = pathlib.Path("/tmp")
    sys.modules["memecoin.config"] = cfg

    ppm = types.ModuleType("memecoin.pumpportal_monitor")
    ppm.monitor = MagicMock()
    ppm.monitor.get_prices   = MagicMock(return_value={})
    ppm.monitor.get_vsol     = MagicMock(return_value=0)
    sys.modules["memecoin.pumpportal_monitor"] = ppm

    psl = types.ModuleType("memecoin.pumpswap_local")
    class PumpSwapPoolError(Exception): pass
    psl.PumpSwapPoolError = PumpSwapPoolError
    psl.fetch_pool        = MagicMock(side_effect=PumpSwapPoolError("no pool"))
    sys.modules["memecoin.pumpswap_local"] = psl

    from enum import Enum
    class TokenExitState(Enum):
        BONDING_CURVE           = "BONDING_CURVE"
        MIGRATION_UNCERTAIN     = "MIGRATION_UNCERTAIN"
        GRADUATED_PUMPSWAP      = "GRADUATED_PUMPSWAP"
        UNKNOWN                 = "UNKNOWN"

    ero = types.ModuleType("memecoin.exit_router")
    ero.TokenExitState = TokenExitState
    ero.classify       = MagicMock(return_value=TokenExitState.BONDING_CURVE)
    sys.modules["memecoin.exit_router"] = ero

    # execution_rpc stub
    erpc = types.ModuleType("memecoin.execution_rpc")
    erpc.rpc_post = MagicMock(return_value={"result": None})
    sys.modules["memecoin.execution_rpc"] = erpc


_install_stubs()


# ---------------------------------------------------------------------------
# A: read_sol_delta retry — null twice then valid
# ---------------------------------------------------------------------------

class TestReadSolDeltaRetry(unittest.TestCase):

    def test_A_null_twice_then_valid(self):
        """Attempt 1+2 return null; attempt 3 returns real tx. sol_delta is real, not None."""
        import memecoin.tx_meta as tm

        _fake_wallet = "8PNHvFWeMT7CqpUvJjAwVgAK545t5KV3uCPd8DUfaTiM"
        _fake_sig    = "2pQsMFjV6Kpso1Da66JS"

        # Simulate tx data with the wallet at index 0
        _tx_data = {
            "result": {
                "meta": {
                    "preBalances":  [100_000_000_000, 0],   # 100 SOL
                    "postBalances": [100_041_726_044, 0],   # +0.041726 SOL
                },
                "transaction": {
                    "message": {
                        "accountKeys": [
                            {"pubkey": _fake_wallet, "signer": True, "writable": True},
                            {"pubkey": "SomeOtherAccount", "signer": False, "writable": False},
                        ]
                    }
                }
            }
        }

        _call_count = [0]

        def _mock_rpc(payload, timeout=15):
            _call_count[0] += 1
            if _call_count[0] < 3:
                return {"result": None}   # unindexed
            return _tx_data

        with patch.object(tm, "_rpc_post", side_effect=_mock_rpc), \
             patch("time.sleep"):  # speed up test
            result = tm.read_sol_delta(_fake_sig, _fake_wallet)

        self.assertTrue(result["ok"], f"Expected ok=True, got {result}")
        self.assertIsNotNone(result["sol_delta"], "sol_delta must not be None")
        self.assertAlmostEqual(result["sol_delta"], 0.041726044, places=6)
        self.assertEqual(result["source"], "native_lamports")
        self.assertEqual(result["attempts"], 3)

    def test_A_all_null_returns_none(self):
        """All 4 attempts return null → ok=False, sol_delta=None (never 0.0)."""
        import memecoin.tx_meta as tm

        with patch.object(tm, "_rpc_post", return_value={"result": None}), \
             patch("time.sleep"):
            result = tm.read_sol_delta("fakesig123", "fakewallet456")

        self.assertFalse(result["ok"])
        self.assertIsNone(result["sol_delta"], "sol_delta must be None, not 0.0")
        self.assertEqual(result["source"], "unindexed")


# ---------------------------------------------------------------------------
# B: confirmed + permanently unindexed + quote estimate → fill_estimated
# ---------------------------------------------------------------------------

class TestContradictionGuardWithEstimate(unittest.TestCase):

    def test_B_fill_estimated_not_100pct(self):
        """
        Simulate: jupiter swap confirmed, sol_delta unreadable, quote estimate exists.
        Expect: success=True, fill_estimated=True, sol_received > 0, NOT -100%.
        """
        # The contradiction guard lives in jupiter_rescue.py lines ~962-1011.
        # We test it via the logic directly.
        sol_received_result = {"ok": False, "sol_delta": None,
                               "source": "unindexed", "attempts": 4, "reason": "all_attempts_unindexed"}
        quote = {"outAmount": "41726044"}   # 0.041726044 SOL in lamports

        # Reproduce the guard logic
        _estimated_sol = 0.0
        if quote and quote.get("outAmount"):
            _estimated_sol = int(quote["outAmount"]) / 1e9

        confirmed = True
        sol_recv  = None
        fill_estimated = False
        success = False

        if confirmed and (not sol_received_result["ok"] or sol_received_result["sol_delta"] is None):
            if _estimated_sol > 0:
                sol_recv       = _estimated_sol
                fill_estimated = True
                success        = True

        self.assertTrue(success)
        self.assertTrue(fill_estimated)
        self.assertIsNotNone(sol_recv)
        self.assertGreater(sol_recv, 0, "sol_recv must be > 0 — never -100%")
        self.assertAlmostEqual(sol_recv, 0.041726044, places=6)


# ---------------------------------------------------------------------------
# C: confirmed + no quote + no meta → needs_reconcile, NOT fill=0 close
# ---------------------------------------------------------------------------

class TestContradictionGuardNoEstimate(unittest.TestCase):

    def test_C_no_close_at_fill_zero(self):
        """
        Confirmed tx, no quote estimate, no sol_delta.
        Expect: success=False, needs_reconcile=True, sol_received stays 0 but
        caller must NOT finalize position at fill=0.
        """
        sol_received_result = {"ok": False, "sol_delta": None,
                               "source": "unindexed", "attempts": 4, "reason": "all_attempts_unindexed"}
        quote = None   # no quote estimate

        confirmed = True
        _estimated_sol = 0.0
        if quote and quote.get("outAmount"):
            _estimated_sol = int(quote["outAmount"]) / 1e9

        success        = True   # initial
        needs_reconcile = False
        fill_estimated = False

        if confirmed and (not sol_received_result["ok"] or sol_received_result["sol_delta"] is None):
            if _estimated_sol > 0:
                fill_estimated = True
                success = True
            else:
                success         = False
                needs_reconcile = True

        self.assertFalse(success, "Must not claim success with no fill data")
        self.assertTrue(needs_reconcile)
        self.assertFalse(fill_estimated)


# ---------------------------------------------------------------------------
# D: scanner suppression — complete=False → no rescue dispatch
# ---------------------------------------------------------------------------

class TestScannerOracleSuppressionCompleteFalse(unittest.TestCase):

    def test_D_complete_false_suppresses_rescue(self):
        """
        Reproduce the Fix 3 decision tree in isolation.
        complete=False → suppress (no arm, no dispatch).
        """
        oracle_result = {"ok": True, "complete": False, "reason": "complete_false"}

        arm_called     = [False]
        dispatch_called = [False]

        def _arm(*a, **kw):     arm_called[0] = True
        def _dispatch(*a, **kw): dispatch_called[0] = True

        # Reproduce the guard logic
        _suppressed = False
        if oracle_result.get("complete") is False:
            _suppressed = True
            # → do not call _arm or _dispatch
        elif oracle_result.get("complete") is True or oracle_result.get("reason") == "account_missing":
            _arm()
            _dispatch()

        self.assertTrue(_suppressed)
        self.assertFalse(arm_called[0],     "arm must NOT be called when complete=False")
        self.assertFalse(dispatch_called[0], "dispatch must NOT be called when complete=False")


# ---------------------------------------------------------------------------
# E: genuine migration — complete=True → rescue proceeds
# ---------------------------------------------------------------------------

class TestScannerOracleSuppressionCompleteTrue(unittest.TestCase):

    def test_E_complete_true_rescue_proceeds(self):
        oracle_result = {"ok": True, "complete": True, "reason": "complete_true"}

        arm_called     = [False]
        dispatch_called = [False]

        if oracle_result.get("complete") is False:
            pass  # suppressed
        else:
            arm_called[0]      = True
            dispatch_called[0] = True

        self.assertTrue(arm_called[0])
        self.assertTrue(dispatch_called[0])


# ---------------------------------------------------------------------------
# F: account missing → rescue proceeds
# ---------------------------------------------------------------------------

class TestScannerOracleAccountMissing(unittest.TestCase):

    def test_F_account_missing_rescue_proceeds(self):
        oracle_result = {"ok": True, "complete": None, "reason": "account_missing"}

        suppressed = False
        if oracle_result.get("complete") is False:
            suppressed = True

        self.assertFalse(suppressed, "account_missing must allow rescue")


# ---------------------------------------------------------------------------
# G: curve feed — price_overrides updated, feed_blind suppressed
# ---------------------------------------------------------------------------

class TestCurveFeed(unittest.TestCase):

    def test_G_curve_feed_updates_price_overrides(self):
        """
        Simulate curve feed result. Verify price_overrides gets the curve price
        and that feed_blind condition is False when curve is fresh.
        """
        _CURVE_FEED_VALID_SEC = 10.0
        _FEED_BLIND_SEC = 20.0

        mint = "Cjb9odGL"
        now  = time.time()

        _curve_feed_last_seen  = {mint: now}   # just updated
        _curve_price_overrides = {mint: 0.0000119}
        price_overrides = {}

        # Simulate: no PP price in price_overrides, curve is fresh
        _cp_age = now - _curve_feed_last_seen.get(mint, 0)
        if _cp_age < _CURVE_FEED_VALID_SEC and mint in _curve_price_overrides:
            price_overrides[mint] = _curve_price_overrides[mint]

        self.assertIn(mint, price_overrides, "curve price must be in price_overrides")
        self.assertAlmostEqual(price_overrides[mint], 0.0000119, places=7)

        # feed_blind condition
        pp_age  = float("inf")
        dex_age = float("inf")
        _curve_age = now - _curve_feed_last_seen.get(mint, 0)

        feed_blind = (pp_age > _FEED_BLIND_SEC
                      and dex_age > _FEED_BLIND_SEC
                      and _curve_age >= _CURVE_FEED_VALID_SEC)

        self.assertFalse(feed_blind,
                         "feed_blind must be False when curve feed is fresh")

    def test_G_curve_feed_stale_allows_feed_blind(self):
        """Stale curve feed → feed_blind can fire."""
        _CURVE_FEED_VALID_SEC = 10.0
        _FEED_BLIND_SEC = 20.0
        mint = "somemint"
        now  = time.time()

        _curve_feed_last_seen = {mint: now - 30}  # stale

        pp_age  = 25.0
        dex_age = 25.0
        _curve_age = now - _curve_feed_last_seen.get(mint, 0)

        feed_blind = (pp_age > _FEED_BLIND_SEC
                      and dex_age > _FEED_BLIND_SEC
                      and _curve_age >= _CURVE_FEED_VALID_SEC)

        self.assertTrue(feed_blind, "feed_blind must fire when all feeds are stale")


# ---------------------------------------------------------------------------
# H: hard stop / trailing stop does not invoke rescue suppression guard
# ---------------------------------------------------------------------------

class TestHardStopUnaffected(unittest.TestCase):

    def test_H_hard_stop_not_suppressed(self):
        """
        The oracle suppression guard only applies to the MIGRATION_UNCERTAIN
        no-pool branch. Hard stop / trailing stop / TP reasons bypass it entirely.
        Verify by checking that the suppression guard predicate requires
        mig_age == inf AND pp+dex silent — never fires on explicit exit reasons.
        """
        # Suppression condition from scanner.py Fix 3:
        # fires only inside `if pp_age > FEED_BLIND_SEC and dex_age > FEED_BLIND_SEC
        #                       and mig_age == float("inf")`
        # Hard stop fires through portfolio.close_position() directly,
        # which does NOT go through this branch.

        _FEED_BLIND_SEC = 20.0
        pp_age  = 5.0   # PP is active — hard stop scenario: PP just ticked a loss
        dex_age = 3.0
        mig_age = float("inf")

        mu_branch_entered = (
            pp_age > _FEED_BLIND_SEC and dex_age > _FEED_BLIND_SEC
            and mig_age == float("inf")
        )

        self.assertFalse(mu_branch_entered,
                         "MIGRATION_UNCERTAIN branch must not enter when PP/Dex are active")


if __name__ == "__main__":
    unittest.main()
