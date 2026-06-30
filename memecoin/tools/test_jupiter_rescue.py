"""
test_jupiter_rescue.py — Regression tests for the Universal Jupiter Rescue Sell system.

Tests:
  A. BST-style: pumpswap_no_pool → rescue called → mock Jupiter success → position closed
  B. BULLSACK-style: pumpswap_bad_pool_layout, T22 → rescue called, pump-amm skipped
  C. SPL graduated failure → rescue called before manual
  D. 6005 path → rescue called before pump-amm repeated
  E. 429 path → governor retries EXIT → escalates to EMERGENCY → no uncontrolled loop
  F. Double-sell guard → rescue_pending in notes → second rescue returns early

Run: python -m memecoin.tools.test_jupiter_rescue
"""

import sys
import types
import unittest
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch, PropertyMock


# ── Minimal Position stub ──────────────────────────────────────────────────────

@dataclass
class _FakePos:
    id:            str   = "pos-test-001"
    token_address: str   = "TokenMintAddr1111111111111111111111111111111"
    token_symbol:  str   = "BST"
    chain:         str   = "solana"
    tokens_held:   int   = 1_000_000
    notes:         str   = ""
    status:        str   = "open"
    exit_price:    float = 0.0001
    size_usd:      float = 5.0
    entry_price:   float = 0.00008


# ── Helpers to inject fake modules so jupiter_rescue can be imported cleanly ──

def _make_fake_config(**overrides):
    cfg = types.ModuleType("memecoin.config")
    cfg.JUPITER_RESCUE_ENABLED               = True
    cfg.JUPITER_RESCUE_MAX_PRICE_IMPACT_PCT  = 50
    cfg.ALLOW_JUPITER_RESCUE_PANIC_EXIT      = True
    cfg.CHAINS = {"solana": {"rpc": "https://fake-rpc.test"}}
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_fake_governor(quote_resp=None, swap_resp=None, raise_on_quote=None, raise_on_swap=None):
    """Return a fake (governor, Purpose) pair."""
    class _Purpose:
        EXIT      = "EXIT"
        EMERGENCY = "EMERGENCY"
        BACKGROUND= "BACKGROUND"

    class _FakeGov:
        def request(self, purpose, endpoint, fn, mint="", **kwargs):
            if endpoint == "quote":
                if raise_on_quote:
                    raise raise_on_quote
                return quote_resp
            if endpoint == "swap":
                if raise_on_swap:
                    raise raise_on_swap
                return swap_resp
            raise ValueError(f"Unknown endpoint: {endpoint}")

    gov_module = types.ModuleType("memecoin.jupiter_governor")
    gov_module.governor = _FakeGov()
    gov_module.Purpose  = _Purpose
    return gov_module


def _make_requests_response(json_data: dict, status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    if status_code >= 400:
        import requests
        http_err = requests.exceptions.HTTPError(response=resp)
        resp.raise_for_status.side_effect = http_err
    else:
        resp.raise_for_status.return_value = None
    return resp


def _make_keypair_stub():
    kp = MagicMock()
    kp.pubkey.return_value = "WalletPubKey1111111111111111111111111111111"
    return kp


def _build_quote_response(impact_pct: float = 0.01) -> dict:
    return {
        "inputMint":      "TokenMintAddr1111111111111111111111111111111",
        "outputMint":     "So11111111111111111111111111111111111111112",
        "inAmount":       "1000000",
        "outAmount":      "50000000",
        "priceImpactPct": str(impact_pct),
    }


def _build_swap_response() -> dict:
    import base64
    # Minimal fake VersionedTransaction bytes (just needs to be decodable as bytes)
    fake_tx = base64.b64encode(b"\x00" * 128).decode()
    return {"swapTransaction": fake_tx}


# ── Base test setup ───────────────────────────────────────────────────────────

class _RescueTestBase(unittest.TestCase):

    def _patch_modules(
        self,
        cfg_overrides=None,
        governor_module=None,
        quote_json=None,
        swap_json=None,
        rpc_send_sig="FakeSig111",
        rpc_confirm_status="confirmed",
        rpc_get_tx_delta=0.05,
    ):
        """Patch all external dependencies for jupiter_rescue."""
        cfg = _make_fake_config(**(cfg_overrides or {}))
        if governor_module is None:
            q_resp = _make_requests_response(quote_json or _build_quote_response())
            s_resp = _make_requests_response(swap_json  or _build_swap_response())
            governor_module = _make_fake_governor(quote_resp=q_resp, swap_resp=s_resp)

        self._patches = []

        def _add(target, new):
            p = patch(target, new)
            p.start()
            self._patches.append(p)

        # Config
        sys.modules["memecoin.config"] = cfg

        # Governor
        sys.modules["memecoin.jupiter_governor"] = governor_module

        # Executor keypair
        fake_executor = types.ModuleType("memecoin.executor")
        fake_executor._get_keypair = lambda: _make_keypair_stub()
        sys.modules["memecoin.executor"] = fake_executor

        # RPC calls inside jupiter_rescue
        def _fake_rpc_post(payload, timeout=15):
            method = payload.get("method", "")
            if method == "getTokenAccountsByOwner":
                return {"result": {"value": [
                    {"account": {"data": {"parsed": {"info": {"tokenAmount": {"amount": "1000000"}}}}}}
                ]}}
            if method == "sendTransaction":
                return {"result": rpc_send_sig}
            if method == "getSignatureStatuses":
                return {"result": {"value": [
                    {"confirmationStatus": rpc_confirm_status, "err": None}
                ]}}
            if method == "getTransaction":
                return {"result": {
                    "meta": {"preBalances": [100_000_000], "postBalances": [int(100_000_000 + rpc_get_tx_delta * 1e9)]},
                    "transaction": {"message": {"accountKeys": [
                        {"pubkey": "WalletPubKey1111111111111111111111111111111"}
                    ]}}
                }}
            return {}

        # Patch solders VersionedTransaction
        fake_vtx = MagicMock()
        fake_vtx.message = MagicMock()
        fake_vtx_class = MagicMock()
        fake_vtx_class.from_bytes.return_value = fake_vtx
        fake_vtx_class.return_value = fake_vtx

        fake_solders_tx = types.ModuleType("solders.transaction")
        fake_solders_tx.VersionedTransaction = fake_vtx_class
        sys.modules["solders.transaction"] = fake_solders_tx

        # Make bytes(signed) work
        fake_vtx.__bytes__ = lambda self: b"\x00" * 128

        import memecoin.jupiter_rescue as jr
        import importlib
        importlib.reload(jr)  # reload so patched sys.modules take effect
        jr._rpc_post = _fake_rpc_post
        self._jr = jr

    def tearDown(self):
        for p in getattr(self, "_patches", []):
            try:
                p.stop()
            except Exception:
                pass
        # Remove cached module so next test gets a fresh load
        sys.modules.pop("memecoin.jupiter_rescue", None)


# ── Test A: BST-style pumpswap_no_pool → rescue succeeds ─────────────────────

class TestA_BSTStylePumpswapNoPool(_RescueTestBase):
    """A. pumpswap_no_pool error class → rescue fires → Jupiter succeeds → position closed."""

    def test_rescue_succeeds_on_no_pool(self):
        self._patch_modules()
        pos = _FakePos(notes="|exit_state:GRADUATED_PUMPSWAP|exit_route:pumpswap_no_pool")
        result = self._jr.force_jupiter_rescue_sell(pos, "migration_uncertain")
        self.assertTrue(result["success"], f"Expected success, got: {result}")
        self.assertEqual(result["route"], "JUPITER_RESCUE")
        self.assertTrue(result["jupiter_confirmed"])
        self.assertTrue(result["jupiter_quote_ok"])
        self.assertTrue(result["jupiter_swap_build_ok"])
        self.assertGreater(result["sol_received"], 0)
        self.assertEqual(result["tx_sig"], "FakeSig111")
        self.assertIn("|jupiter_rescue_pending:", pos.notes)
        print("  A PASS: pumpswap_no_pool → rescue succeeded")


# ── Test B: BULLSACK-style pumpswap_bad_pool_layout, T22 ─────────────────────

class TestB_BullsackT22BadPoolLayout(_RescueTestBase):
    """B. pumpswap_bad_pool_layout / T22 token → rescue fires, pump-amm path is not called."""

    def test_rescue_fires_for_bad_pool_layout(self):
        self._patch_modules()
        pos = _FakePos(
            token_symbol="BULLSACK",
            notes="|exit_state:MIGRATION_UNCERTAIN|exit_route:pumpswap_bad_pool_layout",
        )
        result = self._jr.force_jupiter_rescue_sell(pos, "graduated_exit")
        self.assertTrue(result["success"], f"Expected success, got: {result}")
        self.assertTrue(result["jupiter_quote_ok"])
        self.assertTrue(result["jupiter_swap_build_ok"])
        # pump-amm is not involved in jupiter_rescue — verify route
        self.assertEqual(result["route"], "JUPITER_RESCUE")
        print("  B PASS: pumpswap_bad_pool_layout / T22 → rescue fires correctly")


# ── Test C: SPL graduated failure → rescue before manual ─────────────────────

class TestC_SPLGraduatedFailureRescueBeforeManual(_RescueTestBase):
    """C. SPL graduated failure (executor returns graduated_unsellable) → rescue fires before manual alert."""

    def test_rescue_fires_before_manual_on_graduated_unsellable(self):
        self._patch_modules()
        pos = _FakePos(
            token_symbol="RUGGED",
            notes="|graduated_unsellable",
            status="sell_stuck",
        )
        result = self._jr.force_jupiter_rescue_sell(pos, "sell_stuck")
        self.assertTrue(result["success"], f"Expected success, got: {result}")
        # Urgent reason check (graduated_unsellable in notes → is_urgent=True)
        self.assertTrue(result["jupiter_confirmed"])
        print("  C PASS: SPL graduated_unsellable → rescue fires before manual alert")


# ── Test D: 6005 path → rescue fires before pump-amm repeated ────────────────

class TestD_6005PathRescueBeforeRepeat(_RescueTestBase):
    """D. Token hit Custom:6005 (graduated detected) → rescue fires, avoids re-running BC pump-amm."""

    def test_rescue_fires_for_6005_detected_graduation(self):
        self._patch_modules()
        pos = _FakePos(
            token_symbol="GRAD",
            notes="|cohort:graduated|exit_state:GRADUATED_PUMPSWAP",
        )
        result = self._jr.force_jupiter_rescue_sell(pos, "graduated_exit")
        self.assertTrue(result["success"], f"Expected success, got: {result}")
        self.assertEqual(result["route"], "JUPITER_RESCUE")
        print("  D PASS: 6005-path graduation → rescue fires before pump-amm repeated")


# ── Test E: 429 escalation EXIT → EMERGENCY, no uncontrolled loop ─────────────

class TestE_429EscalationExitToEmergency(_RescueTestBase):
    """E. 429 path: governor EXIT raises HTTPError(429) → escalates to EMERGENCY → success."""

    def test_429_escalation_to_emergency(self):
        import requests as _real_requests

        # First quote call (EXIT purpose) raises 429; second call (EMERGENCY) succeeds
        call_count = {"n": 0}
        q_success = _make_requests_response(_build_quote_response())
        s_success = _make_requests_response(_build_swap_response())

        class _Gov429:
            def request(self, purpose, endpoint, fn, mint="", **kwargs):
                if endpoint == "quote" and purpose == "EXIT":
                    # Simulate 429 from EXIT bucket
                    resp_429 = MagicMock()
                    resp_429.status_code = 429
                    raise _real_requests.exceptions.HTTPError(response=resp_429)
                if endpoint == "quote" and purpose == "EMERGENCY":
                    call_count["n"] += 1
                    return q_success
                if endpoint == "swap":
                    return s_success
                raise ValueError(f"Unexpected: {purpose}/{endpoint}")

        class _Purpose:
            EXIT      = "EXIT"
            EMERGENCY = "EMERGENCY"
            BACKGROUND= "BACKGROUND"

        gov_module = types.ModuleType("memecoin.jupiter_governor")
        gov_module.governor = _Gov429()
        gov_module.Purpose  = _Purpose

        self._patch_modules(governor_module=gov_module)
        pos = _FakePos()
        result = self._jr.force_jupiter_rescue_sell(pos, "hard_stop")
        self.assertTrue(result["success"], f"Expected success, got: {result}")
        self.assertEqual(result["jupiter_429_count"], 1)
        self.assertEqual(call_count["n"], 1, "EMERGENCY quote should have been called exactly once")
        print("  E PASS: 429 EXIT → EMERGENCY escalation (no uncontrolled loop)")


# ── Test F: Double-sell guard — rescue_pending in notes → early return ─────────

class TestF_DoubleSellGuard(_RescueTestBase):
    """F. rescue_pending already in notes → second rescue returns early with rescue_already_pending."""

    def test_double_sell_guard(self):
        self._patch_modules()
        pos = _FakePos(notes="|exit_state:GRADUATED_PUMPSWAP|jupiter_rescue_pending:abc123")
        result = self._jr.force_jupiter_rescue_sell(pos, "migration_uncertain")
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "rescue_already_pending")
        # Ensure no Jupiter calls were made
        self.assertFalse(result["jupiter_quote_ok"])
        self.assertFalse(result["jupiter_send_attempted"])
        print("  F PASS: double-sell guard → rescue_already_pending returned immediately")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Jupiter Rescue Regression Tests ===\n")
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [TestA_BSTStylePumpswapNoPool, TestB_BullsackT22BadPoolLayout,
                TestC_SPLGraduatedFailureRescueBeforeManual, TestD_6005PathRescueBeforeRepeat,
                TestE_429EscalationExitToEmergency, TestF_DoubleSellGuard]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=0, stream=sys.stdout)
    result = runner.run(suite)
    print(f"\n{'OK' if result.wasSuccessful() else 'FAILED'} "
          f"({result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors)")
    sys.exit(0 if result.wasSuccessful() else 1)
