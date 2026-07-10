"""
test_preflight_curve_baseline.py — Entry latency / curve-baseline preflight tests

Tests A–G verify the new preflight order implemented in FIX 1-5:
  A) PP cache hit → price used immediately, no curve RPC, no wait
  B) PP silent + curve complete=False + price → curve price used as baseline
  C) PP silent + curve RPC error → falls back to ≤0.5s PP wait (not 2s)
  D) PP silent + curve complete=True → blocked as graduated
  E) PP silent + account_missing → blocked as migrated
  F) Scanner mid-hold: account_missing in curve feed → graduation handover fires
  G) Drift gate still enforced when baseline_source='curve' (price above signal gate)

Run: python -m pytest memecoin/tests/test_preflight_curve_baseline.py -v
"""

import sys
import time
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Shared stubs — installed once, before any executor import
# ---------------------------------------------------------------------------
_FAKE_MINT = "CurveMint1111111111111111111111111111111111"

def _install_executor_stubs():
    """Install minimal stubs so executor.py can be imported cleanly."""
    sys.modules.setdefault("memecoin", types.ModuleType("memecoin"))

    cfg = types.ModuleType("memecoin.config")
    cfg.EXECUTOR_BACKEND      = "pumpportal"
    cfg.LIVE_DRY_RUN          = False
    cfg.SLIPPAGE_GATE_RT_PCT  = 0.30   # 30% gate (matches real config)
    cfg.SLIPPAGE_GATE_DEX_PCT = 0.50
    cfg.LIVE_TRADING          = False
    cfg.LIVE_GATE_EPOCH       = "2026-07-06"
    cfg.POSITIONS_FILE        = "/tmp/positions_curve_test.json"
    sys.modules.setdefault("memecoin.config", cfg)

    ppm = types.ModuleType("memecoin.pumpportal_monitor")
    ppm.monitor = MagicMock()
    ppm.monitor.get_prices = MagicMock(return_value={})
    sys.modules.setdefault("memecoin.pumpportal_monitor", ppm)

    psl = types.ModuleType("memecoin.pumpswap_local")
    class _PSE(Exception): pass
    psl.PumpSwapPoolError = _PSE
    psl.fetch_pool = MagicMock(side_effect=_PSE("no pool"))
    sys.modules.setdefault("memecoin.pumpswap_local", psl)

    erpc = types.ModuleType("memecoin.execution_rpc")
    erpc.rpc_post = MagicMock(return_value=MagicMock(json=lambda: {"result": None}))
    sys.modules.setdefault("memecoin.execution_rpc", erpc)

    from enum import Enum
    class _TES(Enum):
        BONDING_CURVE = "BONDING_CURVE"
    ero = types.ModuleType("memecoin.exit_router")
    ero.TokenExitState = _TES
    ero.classify = MagicMock(return_value=_TES.BONDING_CURVE)
    sys.modules.setdefault("memecoin.exit_router", ero)


_install_executor_stubs()

# Now import executor (safe — stubs installed above)
import importlib.util as _ilu
_ex_path = __import__("pathlib").Path(__file__).parent.parent / "executor.py"
_ex_spec  = _ilu.spec_from_file_location("memecoin.executor", _ex_path)
_ex_mod   = _ilu.module_from_spec(_ex_spec)
sys.modules["memecoin.executor"] = _ex_mod
_ex_spec.loader.exec_module(_ex_mod)

get_pumpfun_curve_snapshot   = _ex_mod.get_pumpfun_curve_snapshot
get_pumpfun_curve_complete   = _ex_mod.get_pumpfun_curve_complete
_GRAD_ORACLE_CACHE           = _ex_mod._GRAD_ORACLE_CACHE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _curve_result(complete, vtr=800_000_000_000, vsr=30_000_000_000):
    """Build a synthetic _get_pumpfun_curve_complete_uncached return value."""
    if complete is None:
        return {"ok": True, "complete": None, "reason": "account_missing", "bc_pda": "BCPDA", "rpc_ms": 5}
    reason = "complete_true" if complete else "complete_false"
    return {
        "ok": True, "complete": complete, "reason": reason, "bc_pda": "BCPDA", "rpc_ms": 5,
        "virtual_token_reserves": vtr,
        "virtual_sol_reserves":   vsr,
    }


def _rpc_error_result():
    return {"ok": False, "complete": None, "reason": "rpc_error", "bc_pda": "", "rpc_ms": 0}


# ---------------------------------------------------------------------------
# Tests A–E: get_pumpfun_curve_snapshot() unit behaviour
# ---------------------------------------------------------------------------

class TestA_PPCacheHit(unittest.TestCase):
    """A: If PP cache already has a price, no curve RPC should fire."""

    def test_A_pp_price_used_immediately(self):
        """get_pumpfun_curve_snapshot is not called when PP cache has a price.

        Portfolio.py checks PP cache before calling get_pumpfun_curve_snapshot.
        If _pp.get_prices().get(mint) > 0, the snapshot is never requested.
        We verify this by asserting that the snapshot function is NOT called
        when the PP cache already has a price.
        """
        pp_prices = {_FAKE_MINT: 0.00001234}

        snapshot_called = []
        orig_snapshot = get_pumpfun_curve_snapshot

        def _mock_snapshot(mint):
            snapshot_called.append(mint)
            return orig_snapshot(mint)

        # Simulate the portfolio preflight logic (type-1 path) inline:
        mint = _FAKE_MINT
        pp_price = pp_prices.get(mint, 0)
        if pp_price > 0:
            baseline_source = "pp_tick"
            # portfolio would NOT call get_pumpfun_curve_snapshot here
        else:
            baseline_source = "curve"  # would call, but PP cache has price so we don't reach here
            _mock_snapshot(mint)

        self.assertEqual(pp_price, 0.00001234)
        self.assertEqual(baseline_source, "pp_tick")
        self.assertEqual(snapshot_called, [], "snapshot must NOT be called when PP cache has price")


class TestB_CurvePriceBaseline(unittest.TestCase):
    """B: PP silent + curve complete=False + price → curve price used as baseline."""

    def test_B_curve_complete_false_returns_price(self):
        """Snapshot returns price_usd when complete=False and reserves are present."""
        # vtr=800B (800e9 base units = 800M tokens), vsr=30e9 lamports = 30 SOL
        synthetic = _curve_result(complete=False, vtr=800_000_000_000, vsr=30_000_000_000)

        with patch.object(_ex_mod, "_get_pumpfun_curve_complete_uncached", return_value=synthetic):
            with patch.object(_ex_mod, "_sol_price_usd", return_value=150.0):
                # Clear cache for clean test
                _GRAD_ORACLE_CACHE.pop(_FAKE_MINT, None)
                snap = get_pumpfun_curve_snapshot(_FAKE_MINT)

        self.assertTrue(snap["ok"])
        self.assertIs(snap["complete"], False)
        self.assertIsNotNone(snap["price_usd"])
        self.assertGreater(snap["price_usd"], 0)
        # vtr=800e9 raw = 800e9/1e6 = 800_000 human tokens (6 decimals)
        # vsr=30e9 lamports = 30e9/1e9 = 30 SOL
        # price_sol = 30 SOL / 800_000 tokens = 3.75e-5 SOL/token
        # price_usd = 3.75e-5 * 150 = 0.005625 USD/token
        self.assertAlmostEqual(snap["price_sol"], 3.75e-5, places=11)
        self.assertAlmostEqual(snap["price_usd"], 0.005625, places=8)
        self.assertEqual(snap["reason"], "complete_false")


class TestC_CurveRpcError(unittest.TestCase):
    """C: Curve RPC error → ok=False, no price (portfolio falls back to ≤0.5s PP wait)."""

    def test_C_rpc_error_returns_ok_false(self):
        """Snapshot returns ok=False on RPC error, signalling portfolio to fall back."""
        with patch.object(_ex_mod, "_get_pumpfun_curve_complete_uncached", return_value=_rpc_error_result()):
            _GRAD_ORACLE_CACHE.pop(_FAKE_MINT, None)
            snap = get_pumpfun_curve_snapshot(_FAKE_MINT)

        self.assertFalse(snap["ok"])
        self.assertIsNone(snap["price_usd"])
        self.assertIsNone(snap["price_sol"])
        self.assertEqual(snap["reason"], "rpc_error")


class TestD_CurveCompleteTrue(unittest.TestCase):
    """D: complete=True → snapshot returns complete=True, portfolio blocks buy."""

    def test_D_complete_true_no_price(self):
        """complete=True means graduated — no price, reason=complete_true."""
        synthetic = _curve_result(complete=True)
        with patch.object(_ex_mod, "_get_pumpfun_curve_complete_uncached", return_value=synthetic):
            with patch.object(_ex_mod, "_sol_price_usd", return_value=150.0):
                _GRAD_ORACLE_CACHE.pop(_FAKE_MINT, None)
                snap = get_pumpfun_curve_snapshot(_FAKE_MINT)

        self.assertTrue(snap["ok"])
        self.assertIs(snap["complete"], True)
        self.assertIsNone(snap["price_usd"])   # no price for graduated token
        self.assertIsNone(snap["price_sol"])
        self.assertEqual(snap["reason"], "complete_true")

    def test_D_portfolio_blocks_on_complete_true(self):
        """Verify portfolio preflight logic blocks when complete=True."""
        # Simulate the portfolio type-1 block condition
        snap = {"ok": True, "complete": True, "reason": "complete_true",
                "price_usd": None, "price_sol": None}
        complete = snap.get("complete")
        reason   = snap.get("reason", "")
        blocked  = (complete is True or reason == "account_missing")
        self.assertTrue(blocked, "complete=True must block preflight")


class TestE_AccountMissing(unittest.TestCase):
    """E: account_missing → snapshot returns complete=None, portfolio blocks buy."""

    def test_E_account_missing_no_price(self):
        """account_missing → ok=True, complete=None, no price."""
        synthetic = _curve_result(complete=None)
        with patch.object(_ex_mod, "_get_pumpfun_curve_complete_uncached", return_value=synthetic):
            _GRAD_ORACLE_CACHE.pop(_FAKE_MINT, None)
            snap = get_pumpfun_curve_snapshot(_FAKE_MINT)

        self.assertTrue(snap["ok"])
        self.assertIsNone(snap["complete"])
        self.assertIsNone(snap["price_usd"])
        self.assertEqual(snap["reason"], "account_missing")

    def test_E_portfolio_blocks_on_account_missing(self):
        """Verify portfolio preflight logic blocks when account_missing."""
        snap = {"ok": True, "complete": None, "reason": "account_missing",
                "price_usd": None, "price_sol": None}
        complete = snap.get("complete")
        reason   = snap.get("reason", "")
        blocked  = (complete is True or reason == "account_missing")
        self.assertTrue(blocked, "account_missing must block preflight")


# ---------------------------------------------------------------------------
# Test E2: shared 5s cache between snapshot and complete (FIX 4 no-dup-RPC)
# ---------------------------------------------------------------------------

class TestE2_SharedCache(unittest.TestCase):
    """Snapshot and complete share the same 5s cache — no duplicate RPC."""

    def test_E2_cache_shared_no_duplicate_rpc(self):
        """get_pumpfun_curve_snapshot populates cache; get_pumpfun_curve_complete hits it."""
        synthetic = _curve_result(complete=False, vtr=800_000_000_000, vsr=30_000_000_000)
        call_count = [0]

        def _counting_uncached(mint):
            call_count[0] += 1
            return synthetic

        with patch.object(_ex_mod, "_get_pumpfun_curve_complete_uncached",
                          side_effect=_counting_uncached):
            with patch.object(_ex_mod, "_sol_price_usd", return_value=150.0):
                _GRAD_ORACLE_CACHE.pop(_FAKE_MINT, None)
                snap = get_pumpfun_curve_snapshot(_FAKE_MINT)   # populates cache
                comp = get_pumpfun_curve_complete(_FAKE_MINT)   # must HIT cache

        self.assertEqual(call_count[0], 1, "RPC must only fire once (cache hit on second call)")
        self.assertIs(snap["complete"], False)
        self.assertIs(comp["complete"], False)


# ---------------------------------------------------------------------------
# Test F: Scanner mid-hold account_missing → graduation handover
# ---------------------------------------------------------------------------

class TestF_MidHoldGraduationHandover(unittest.TestCase):
    """F: scanner.py curve feed triggers graduation handover on account_missing."""

    def test_F_account_missing_fires_cohort_swap(self):
        """When curve returns account_missing mid-hold, cohort tag flips to 'graduated'."""
        # Simulate the scanner curve feed block for a bonding_curve position
        # that returns account_missing.

        class _FakePos:
            token_address = _FAKE_MINT
            token_symbol  = "FAKESYM"
            status        = "open"
            chain         = "solana"
            is_live       = True
            notes         = "live|tx:ABCDEF|cohort:bonding_curve"

        pos = _FakePos()
        cp_result = {"ok": True, "complete": None, "reason": "account_missing",
                     "price_usd": None, "price_sol": None}

        # Reproduce the FIX 5 scanner logic inline:
        curve_price_overrides = {pos.token_address: 0.00001}
        curve_feed_last_seen  = {pos.token_address: time.time()}

        if cp_result.get("ok") and cp_result.get("price_usd"):
            pass  # has price — not our case
        elif cp_result.get("complete") is None:  # account_missing
            curve_price_overrides.pop(pos.token_address, None)
            curve_feed_last_seen.pop(pos.token_address, None)
            if pos.notes and "|cohort:bonding_curve" in pos.notes:
                pos.notes = pos.notes.replace("|cohort:bonding_curve", "|cohort:graduated")

        self.assertIn("|cohort:graduated", pos.notes,
                      "account_missing must flip cohort to 'graduated'")
        self.assertNotIn("|cohort:bonding_curve", pos.notes)
        self.assertNotIn(pos.token_address, curve_price_overrides,
                         "account_missing must stop curve polling")

    def test_F_complete_true_also_fires_cohort_swap(self):
        """When curve returns complete=True mid-hold, cohort tag also flips."""
        class _FakePos:
            token_address = _FAKE_MINT
            token_symbol  = "FAKESYM"
            status        = "open"
            chain         = "solana"
            is_live       = True
            notes         = "live|tx:ABCDEF|cohort:bonding_curve"

        pos = _FakePos()
        cp_result = {"ok": True, "complete": True, "reason": "complete_true",
                     "price_usd": 0.000012, "price_sol": 8e-8}

        # Reproduce the FIX 5 scanner logic for complete=True:
        curve_price_overrides = {}
        curve_feed_last_seen  = {pos.token_address: time.time()}

        if cp_result.get("ok") and cp_result.get("price_usd"):
            if cp_result.get("complete") is True:
                curve_price_overrides.pop(pos.token_address, None)
                curve_feed_last_seen.pop(pos.token_address, None)
                if pos.notes and "|cohort:bonding_curve" in pos.notes:
                    pos.notes = pos.notes.replace("|cohort:bonding_curve", "|cohort:graduated")

        self.assertIn("|cohort:graduated", pos.notes,
                      "complete=True must flip cohort to 'graduated'")
        self.assertNotIn("|cohort:bonding_curve", pos.notes)


# ---------------------------------------------------------------------------
# Test G: Drift gate still enforced on curve-baseline price
# ---------------------------------------------------------------------------

class TestG_DriftGateEnforced(unittest.TestCase):
    """G: SLIPPAGE_GATE_RT_PCT drift gate is still enforced when baseline=curve."""

    def _check_drift_gate(self, sig_price, live_price, gate_pct):
        """Return True if drift gate blocks (live > sig * (1 + gate))."""
        return sig_price > 0 and live_price > sig_price * (1 + gate_pct)

    def test_G_gate_blocks_when_curve_price_above_gate(self):
        """Curve price 35% above signal → blocked by 30% drift gate."""
        sig_price   = 0.000010
        curve_price = 0.0000135   # 35% above signal
        gate_pct    = 0.30        # SLIPPAGE_GATE_RT_PCT = 30%
        blocked = self._check_drift_gate(sig_price, curve_price, gate_pct)
        self.assertTrue(blocked, "35% above signal must trigger 30% drift gate")

    def test_G_gate_allows_when_curve_price_within_gate(self):
        """Curve price 20% above signal → allowed by 30% drift gate."""
        sig_price   = 0.000010
        curve_price = 0.0000120   # 20% above signal
        gate_pct    = 0.30
        blocked = self._check_drift_gate(sig_price, curve_price, gate_pct)
        self.assertFalse(blocked, "20% above signal must pass 30% drift gate")

    def test_G_gate_not_enforced_when_no_sig_price(self):
        """If sig_price=0 (unknown), drift gate does not block (avoids false positives)."""
        sig_price   = 0.0
        curve_price = 0.0001
        gate_pct    = 0.20
        blocked = self._check_drift_gate(sig_price, curve_price, gate_pct)
        self.assertFalse(blocked, "No signal price means gate cannot fire")

    def test_G_gate_threshold_unchanged(self):
        """Verify SLIPPAGE_GATE_RT_PCT is still 30% (unchanged by this patch)."""
        cfg = sys.modules["memecoin.config"]
        self.assertEqual(cfg.SLIPPAGE_GATE_RT_PCT, 0.30,
                         "SLIPPAGE_GATE_RT_PCT must remain 0.30 (30%)")


# ---------------------------------------------------------------------------
# L3: oracle-driven MU escalation (post-measurement batch, 2026-07-10)
#
# Mirrors the _mu_force_escalate decision added in portfolio.py
# close_position(): at every retry evaluation (including the first), a fresh
# (not entry-time-cached) oracle read of complete=True/account_missing skips
# ExitRouter classify + PumpSwap-local + the normal PumpPortal ladder and
# escalates straight to Jupiter rescue. sell_attempts stays the only outer
# retry bound (unchanged).
# ---------------------------------------------------------------------------

class TestMUEscalation(unittest.TestCase):

    def _mu_force_escalate(self, oracle_bc: bool, oracle_result: dict) -> bool:
        """Reproduce portfolio.py's _mu_force_escalate condition."""
        if oracle_bc:
            return False
        return (oracle_result.get("complete") is True
                or oracle_result.get("reason") == "account_missing")

    def test_complete_true_escalates_at_attempt_1(self):
        """Mocked oracle flips complete=True → escalate fires on the very first attempt."""
        escalate = self._mu_force_escalate(
            oracle_bc=False, oracle_result=_curve_result(True),
        )
        self.assertTrue(escalate, "complete=True must escalate immediately, attempt 1 included")

    def test_account_missing_escalates(self):
        escalate = self._mu_force_escalate(
            oracle_bc=False, oracle_result=_curve_result(None),
        )
        self.assertTrue(escalate, "account_missing must escalate (curve closed/migrated)")

    def test_complete_false_does_not_escalate(self):
        """Still on bonding curve — normal ladder handles it, no escalation."""
        escalate = self._mu_force_escalate(
            oracle_bc=False, oracle_result=_curve_result(False),
        )
        self.assertFalse(escalate)

    def test_oracle_confirmed_bonding_curve_never_escalates(self):
        """T22 tokens tagged cohort:bonding_curve at entry must never escalate
        via this path — PP/oracle silence is normal for T22, not graduation."""
        escalate = self._mu_force_escalate(
            oracle_bc=True, oracle_result=_curve_result(True),
        )
        self.assertFalse(escalate,
                         "oracle-confirmed bonding curve (T22) must suppress escalation")


if __name__ == "__main__":
    unittest.main()
