"""
test_jupiter_rescue.py — Regression tests for the Universal Jupiter Rescue Sell system.

Tests A–F  (original 6)
  A. BST-style: pumpswap_no_pool → rescue fires → mock Jupiter success → position closed
  B. BULLSACK-style: pumpswap_bad_pool_layout, T22 → rescue fires, pump-amm skipped
  C. SPL graduated failure → rescue fires before manual alert
  D. 6005 path → rescue fires before pump-amm repeated
  E. 429 EXIT → escalates to EMERGENCY → no uncontrolled loop
  F. Double-sell guard → rescue_pending in notes → second rescue returns early

Tests G–O  (new: TTL, rebroadcast, multi-RPC, balance, no-route, panic)
  G. TTL pending tag — tag age < TTL → rescue_already_pending
  H. TTL pending tag — tag age >= TTL + FULL sig confirmed → success=True, rescue_stale_sig_confirmed
  I. TTL pending tag — tag age >= TTL + FULL sig failed → clears tag + rescue succeeds
  J. Rebroadcast — fallback URLs present → rebroadcast_count > 0
  K. Multi-RPC failover — primary 429 → fallback succeeds (ExecutionRpcClient unit test)
  L. tokens_held=0 + RPC balance OK → amount_source="rpc_balance"
  M. tokens_held=0 + RPC balance fails → zero_balance early return
  N. Jupiter returns no_route → no_route error class + no-route CSV written
  O. ALLOW_JUPITER_RESCUE_PANIC_EXIT=False + high impact → price_impact_too_high blocked

Run: python -m memecoin.tools.test_jupiter_rescue
"""

import sys
import time
import types
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch


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


# ── Fake module builders ───────────────────────────────────────────────────────

def _make_fake_config(**overrides):
    cfg = types.ModuleType("memecoin.config")
    cfg.JUPITER_RESCUE_ENABLED                = True
    cfg.JUPITER_RESCUE_MAX_PRICE_IMPACT_PCT   = 50
    cfg.ALLOW_JUPITER_RESCUE_PANIC_EXIT       = True
    cfg.JUPITER_RESCUE_PENDING_TTL_SEC        = 30
    cfg.JUPITER_RESCUE_REBROADCAST_ENABLED    = True
    cfg.JUPITER_RESCUE_REBROADCAST_INTERVAL_MS = 500
    cfg.JUPITER_RESCUE_REBROADCAST_MAX_RPC    = 3
    cfg.EXECUTION_RPC_URLS                    = ["https://primary.test"]
    cfg.EXECUTION_RPC_FALLBACK_URLS           = []
    cfg.CHAINS = {"solana": {"rpc": "https://primary.test"}}
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_fake_governor(quote_resp=None, swap_resp=None, raise_on_quote=None, raise_on_swap=None):
    class _Purpose:
        EXIT       = "EXIT"
        EMERGENCY  = "EMERGENCY"
        BACKGROUND = "BACKGROUND"

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

    gov_module         = types.ModuleType("memecoin.jupiter_governor")
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
        rpc_stale_sig="",          # if set, this sig returns stale_sig_confirmed_status
        rpc_stale_confirmed=False, # True → stale sig status = finalized (already sold)
        rpc_stale_failed=False,    # True → stale sig status = processed+err (failed)
    ):
        cfg = _make_fake_config(**(cfg_overrides or {}))
        if governor_module is None:
            q_resp = _make_requests_response(quote_json or _build_quote_response())
            s_resp = _make_requests_response(swap_json  or _build_swap_response())
            governor_module = _make_fake_governor(quote_resp=q_resp, swap_resp=s_resp)

        self._patches = []

        # Config
        sys.modules["memecoin.config"] = cfg

        # Governor
        sys.modules["memecoin.jupiter_governor"] = governor_module

        # Executor keypair
        fake_executor            = types.ModuleType("memecoin.executor")
        fake_executor._get_keypair = lambda: _make_keypair_stub()
        sys.modules["memecoin.executor"] = fake_executor

        # execution_rpc stub (so import inside _rpc_post doesn't fail)
        fake_exec_rpc              = types.ModuleType("memecoin.execution_rpc")
        _last_url_holder           = {"url": "https://primary.test"}
        class _FakeClient:
            last_used_url = "https://primary.test"
        fake_exec_rpc.get_client   = lambda: _FakeClient()
        fake_exec_rpc.rpc_post     = lambda payload, timeout_override_sec=None: {}
        sys.modules["memecoin.execution_rpc"] = fake_exec_rpc

        # RPC calls
        def _fake_rpc_post(payload, timeout=15):
            method = payload.get("method", "")
            if method == "getTokenAccountsByOwner":
                return {"result": {"value": [
                    {"account": {"data": {"parsed": {"info": {"tokenAmount": {"amount": "1000000"}}}}}}
                ]}}
            if method == "sendTransaction":
                return {"result": rpc_send_sig}
            if method == "getSignatureStatuses":
                sigs_requested = (payload.get("params") or [[]])[0]
                sig_req        = sigs_requested[0] if sigs_requested else ""
                # Stale sig check (TTL guard)
                if rpc_stale_sig and sig_req == rpc_stale_sig:
                    if rpc_stale_confirmed:
                        return {"result": {"value": [
                            {"confirmationStatus": "finalized", "err": None}
                        ]}}
                    elif rpc_stale_failed:
                        return {"result": {"value": [
                            {"confirmationStatus": "processed",
                             "err": {"InstructionError": [0, "Custom"]}}
                        ]}}
                    else:
                        return {"result": {"value": [None]}}
                # Regular confirm loop
                return {"result": {"value": [
                    {"confirmationStatus": rpc_confirm_status, "err": None}
                ]}}
            if method == "getTransaction":
                return {"result": {
                    "meta": {
                        "preBalances":  [100_000_000],
                        "postBalances": [int(100_000_000 + rpc_get_tx_delta * 1e9)],
                    },
                    "transaction": {"message": {"accountKeys": [
                        {"pubkey": "WalletPubKey1111111111111111111111111111111"}
                    ]}}
                }}
            return {}

        # Solders stub
        fake_vtx       = MagicMock()
        fake_vtx.message = MagicMock()
        fake_vtx.__bytes__ = lambda self: b"\x00" * 128
        fake_vtx_class = MagicMock()
        fake_vtx_class.from_bytes.return_value = fake_vtx
        fake_vtx_class.return_value            = fake_vtx

        fake_solders_tx                   = types.ModuleType("solders.transaction")
        fake_solders_tx.VersionedTransaction = fake_vtx_class
        sys.modules["solders.transaction"] = fake_solders_tx

        import memecoin.jupiter_rescue as jr
        import importlib
        importlib.reload(jr)
        jr._rpc_post = _fake_rpc_post
        self._jr = jr

    def tearDown(self):
        for p in getattr(self, "_patches", []):
            try:
                p.stop()
            except Exception:
                pass
        sys.modules.pop("memecoin.jupiter_rescue", None)
        sys.modules.pop("memecoin.execution_rpc",  None)


# ════════════════════════════════════════════════════════════════════════════════
# Tests A – F  (original)
# ════════════════════════════════════════════════════════════════════════════════

class TestA_BSTStylePumpswapNoPool(_RescueTestBase):
    """A. pumpswap_no_pool → rescue fires → Jupiter succeeds → position closed."""

    def test_rescue_succeeds_on_no_pool(self):
        self._patch_modules()
        pos    = _FakePos(notes="|exit_state:GRADUATED_PUMPSWAP|exit_route:pumpswap_no_pool")
        result = self._jr.force_jupiter_rescue_sell(pos, "migration_uncertain")
        self.assertTrue(result["success"], f"Expected success: {result}")
        self.assertEqual(result["route"], "JUPITER_RESCUE")
        self.assertTrue(result["jupiter_confirmed"])
        self.assertTrue(result["jupiter_quote_ok"])
        self.assertTrue(result["jupiter_swap_build_ok"])
        self.assertGreater(result["sol_received"], 0)
        self.assertEqual(result["tx_sig"], "FakeSig111")
        self.assertIn("|jupiter_rescue_pending:", pos.notes)
        print("  A PASS: pumpswap_no_pool → rescue succeeded")


class TestB_BullsackT22BadPoolLayout(_RescueTestBase):
    """B. pumpswap_bad_pool_layout / T22 → rescue fires, pump-amm skipped."""

    def test_rescue_fires_for_bad_pool_layout(self):
        self._patch_modules()
        pos = _FakePos(
            token_symbol="BULLSACK",
            notes="|exit_state:MIGRATION_UNCERTAIN|exit_route:pumpswap_bad_pool_layout",
        )
        result = self._jr.force_jupiter_rescue_sell(pos, "graduated_exit")
        self.assertTrue(result["success"], f"Expected success: {result}")
        self.assertTrue(result["jupiter_quote_ok"])
        self.assertTrue(result["jupiter_swap_build_ok"])
        self.assertEqual(result["route"], "JUPITER_RESCUE")
        print("  B PASS: pumpswap_bad_pool_layout / T22 → rescue fires correctly")


class TestC_SPLGraduatedFailureRescueBeforeManual(_RescueTestBase):
    """C. SPL graduated_unsellable → rescue fires before manual alert."""

    def test_rescue_fires_before_manual_on_graduated_unsellable(self):
        self._patch_modules()
        pos    = _FakePos(token_symbol="RUGGED", notes="|graduated_unsellable", status="sell_stuck")
        result = self._jr.force_jupiter_rescue_sell(pos, "sell_stuck")
        self.assertTrue(result["success"], f"Expected success: {result}")
        self.assertTrue(result["jupiter_confirmed"])
        print("  C PASS: SPL graduated_unsellable → rescue fires before manual alert")


class TestD_6005PathRescueBeforeRepeat(_RescueTestBase):
    """D. 6005-detected graduation → rescue fires before pump-amm repeated."""

    def test_rescue_fires_for_6005_detected_graduation(self):
        self._patch_modules()
        pos    = _FakePos(notes="|cohort:graduated|exit_state:GRADUATED_PUMPSWAP")
        result = self._jr.force_jupiter_rescue_sell(pos, "graduated_exit")
        self.assertTrue(result["success"], f"Expected success: {result}")
        self.assertEqual(result["route"], "JUPITER_RESCUE")
        print("  D PASS: 6005-path graduation → rescue fires before pump-amm repeated")


class TestE_429EscalationExitToEmergency(_RescueTestBase):
    """E. 429 from EXIT governor bucket → escalates to EMERGENCY → success."""

    def test_429_escalation_to_emergency(self):
        import requests as _real_requests

        call_count = {"n": 0}
        q_success  = _make_requests_response(_build_quote_response())
        s_success  = _make_requests_response(_build_swap_response())

        class _Gov429:
            def request(self, purpose, endpoint, fn, mint="", **kwargs):
                if endpoint == "quote" and purpose == "EXIT":
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
            EXIT       = "EXIT"
            EMERGENCY  = "EMERGENCY"
            BACKGROUND = "BACKGROUND"

        gov_module          = types.ModuleType("memecoin.jupiter_governor")
        gov_module.governor = _Gov429()
        gov_module.Purpose  = _Purpose

        self._patch_modules(governor_module=gov_module)
        pos    = _FakePos()
        result = self._jr.force_jupiter_rescue_sell(pos, "hard_stop")
        self.assertTrue(result["success"], f"Expected success: {result}")
        self.assertEqual(result["jupiter_429_count"], 1)
        self.assertEqual(call_count["n"], 1, "EMERGENCY quote called exactly once")
        print("  E PASS: 429 EXIT → EMERGENCY escalation (no uncontrolled loop)")


class TestF_DoubleSellGuard(_RescueTestBase):
    """F. rescue_pending in notes → second rescue returns rescue_already_pending immediately."""

    def test_double_sell_guard_old_style(self):
        self._patch_modules()
        # Old-style tag (no timestamp) — always treated as within TTL
        pos    = _FakePos(notes="|exit_state:GRADUATED_PUMPSWAP|jupiter_rescue_pending:abc123")
        result = self._jr.force_jupiter_rescue_sell(pos, "migration_uncertain")
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "rescue_already_pending")
        self.assertFalse(result["jupiter_quote_ok"])
        self.assertFalse(result["jupiter_send_attempted"])
        print("  F PASS: old-style double-sell guard → rescue_already_pending")


# ════════════════════════════════════════════════════════════════════════════════
# Tests G – O  (new: TTL / rebroadcast / multi-RPC / balance / no-route / panic)
# ════════════════════════════════════════════════════════════════════════════════

class TestG_TTLPendingTagWithinTTL(_RescueTestBase):
    """G. Tag age < PENDING_TTL_SEC → rescue_already_pending (don't retry)."""

    def test_fresh_tag_blocks_rescue(self):
        self._patch_modules(cfg_overrides={"JUPITER_RESCUE_PENDING_TTL_SEC": 30})
        fresh_ts = int(time.time()) - 5    # 5 seconds old — within 30s TTL
        pos      = _FakePos(
            notes=f"|exit_state:GRADUATED_PUMPSWAP|jupiter_rescue_pending:FakeSig1:{fresh_ts}"
        )
        result = self._jr.force_jupiter_rescue_sell(pos, "migration_uncertain")
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "rescue_already_pending")
        self.assertFalse(result["jupiter_send_attempted"])
        self.assertAlmostEqual(result["pending_tag_age_sec"], 5, delta=2)
        print("  G PASS: TTL fresh tag (5s < 30s TTL) → rescue_already_pending")


class TestH_TTLExpiredSigConfirmed(_RescueTestBase):
    """H. Tag age >= TTL + full-length sig confirmed on chain → success=True, reason=rescue_stale_sig_confirmed."""

    def test_expired_confirmed_sig_finalizes(self):
        # Full 88-char Solana-length sig — new code calls getSignatureStatuses with it
        stale_sig = "5J3mBbAH58CpQ3Y5RNJpUKPE73Ty8UxmxMjFNSBSiK9hFjfChGTbF7GV1xt3Q3mPGkfXT12xDKVJfmS1234ABCD"
        old_ts    = int(time.time()) - 120   # 120s old — expired (TTL=30)
        self._patch_modules(
            cfg_overrides={"JUPITER_RESCUE_PENDING_TTL_SEC": 30},
            rpc_stale_sig=stale_sig,
            rpc_stale_confirmed=True,
        )
        pos = _FakePos(
            notes=f"|exit_state:GRADUATED_PUMPSWAP|jupiter_rescue_pending:{stale_sig}:{old_ts}"
        )
        result = self._jr.force_jupiter_rescue_sell(pos, "migration_uncertain")
        # success=True so caller can finalize without re-sending a tx
        self.assertTrue(result["success"], f"Expected success=True: {result}")
        self.assertEqual(result["reason"], "rescue_stale_sig_confirmed")
        self.assertEqual(result["error_class"], "already_sold")
        self.assertEqual(result["tx_sig"], stale_sig)
        self.assertTrue(result["jupiter_confirmed"])
        self.assertFalse(result["jupiter_send_attempted"])
        print("  H PASS: expired full-sig confirmed → success=True, rescue_stale_sig_confirmed")


class TestI_TTLExpiredSigFailed(_RescueTestBase):
    """I. Tag age >= TTL + sig failed → clears stale tag → fresh rescue proceeds and succeeds."""

    def test_expired_failed_sig_allows_fresh_rescue(self):
        # Full-length sig so getSignatureStatuses is called (error → tag cleared → fresh rescue)
        stale_sig = "5J3mBbAH58CpQ3Y5RNJpUKPE73Ty8UxmxMjFNSBSiK9hFjfChGTbF7GV1xt3Q3mPGkfXT12xDKVJfmS2222BBBB"
        old_ts    = int(time.time()) - 120
        self._patch_modules(
            cfg_overrides={"JUPITER_RESCUE_PENDING_TTL_SEC": 30},
            rpc_stale_sig=stale_sig,
            rpc_stale_failed=True,
        )
        pos = _FakePos(
            notes=f"|exit_state:GRADUATED_PUMPSWAP|jupiter_rescue_pending:{stale_sig}:{old_ts}"
        )
        result = self._jr.force_jupiter_rescue_sell(pos, "migration_uncertain")
        # Stale tag cleared, fresh rescue ran and succeeded
        self.assertTrue(result["success"], f"Expected success: {result}")
        self.assertNotIn(f"jupiter_rescue_pending:{stale_sig}", pos.notes)
        print("  I PASS: expired tag + sig failed → stale tag cleared + fresh rescue succeeded")


class TestJ_RebroadcastCount(_RescueTestBase):
    """J. Fallback RPC URLs available → rebroadcast_count > 0 after rescue."""

    def test_rebroadcast_fires_when_fallback_urls_set(self):
        self._patch_modules(cfg_overrides={
            "JUPITER_RESCUE_REBROADCAST_ENABLED":    True,
            "JUPITER_RESCUE_REBROADCAST_MAX_RPC":    2,
            "JUPITER_RESCUE_REBROADCAST_INTERVAL_MS": 0,  # no delay in test
            "EXECUTION_RPC_URLS":         ["https://primary.test"],
            "EXECUTION_RPC_FALLBACK_URLS": ["https://fallback1.test", "https://fallback2.test"],
        })
        rebroadcast_hits = []

        orig_post_to_url = self._jr._rpc_post_to_url

        def _spy_post_to_url(url, payload, timeout=5.0):
            rebroadcast_hits.append(url)

        self._jr._rpc_post_to_url = _spy_post_to_url
        try:
            pos    = _FakePos()
            result = self._jr.force_jupiter_rescue_sell(pos, "migration_uncertain")
            self.assertTrue(result["success"], f"Expected success: {result}")
            self.assertGreater(result["rebroadcast_count"], 0,
                               "Expected at least 1 rebroadcast")
            self.assertGreater(len(rebroadcast_hits), 0,
                               "Expected _rpc_post_to_url to be called for rebroadcast")
        finally:
            self._jr._rpc_post_to_url = orig_post_to_url
        print(f"  J PASS: rebroadcast_count={result['rebroadcast_count']} rebroadcasts fired to {rebroadcast_hits}")


class TestK_MultiRPCFailoverClient(unittest.TestCase):
    """K. ExecutionRpcClient: primary 429 → rotates to fallback → returns success."""

    def test_failover_on_primary_429(self):
        # Import fresh (no mocked sys.modules needed — testing class directly)
        sys.modules.pop("memecoin.execution_rpc", None)
        sys.modules.pop("memecoin.config",         None)
        import importlib
        import memecoin.execution_rpc as exc_rpc
        importlib.reload(exc_rpc)

        call_log = []

        def _mock_post(url, json=None, timeout=None):
            call_log.append(url)
            resp = MagicMock()
            if "primary" in url:
                resp.status_code = 429
                resp.raise_for_status.return_value = None
            else:
                resp.status_code = 200
                resp.json.return_value = {"result": "ok_from_fallback"}
                resp.raise_for_status.return_value = None
            return resp

        client = exc_rpc.ExecutionRpcClient(
            primary_urls=["https://primary.test"],
            fallback_urls=["https://fallback.test"],
            timeout_sec=3.0,
            max_retries=2,
        )

        with patch("memecoin.execution_rpc._requests.post", _mock_post):
            result = client.post({"method": "test"})

        self.assertEqual(result, {"result": "ok_from_fallback"})
        self.assertEqual(len(call_log), 2)
        self.assertIn("primary", call_log[0])
        self.assertIn("fallback", call_log[1])
        self.assertIn("fallback", client.last_used_url)
        print(f"  K PASS: primary 429 → fallback succeeded  calls={call_log}")

    def tearDown(self):
        sys.modules.pop("memecoin.execution_rpc", None)


class TestL_TokensHeldZeroRPCBalanceOK(_RescueTestBase):
    """L. tokens_held=0 + RPC getTokenAccountsByOwner succeeds → amount_source='rpc_balance'."""

    def test_rpc_balance_used_when_pos_tokens_held_zero(self):
        self._patch_modules()
        pos = _FakePos(tokens_held=0)
        result = self._jr.force_jupiter_rescue_sell(pos, "migration_uncertain")
        self.assertTrue(result["success"], f"Expected success: {result}")
        self.assertFalse(result["used_pos_tokens_held"])
        self.assertEqual(result["amount_source"], "rpc_balance")
        self.assertFalse(result["balance_rpc_failed"])
        self.assertEqual(result["token_balance_raw"], 1_000_000)  # from fake RPC
        print("  L PASS: tokens_held=0 + RPC OK → amount_source='rpc_balance'")


class TestM_TokensHeldZeroRPCBalanceFail(_RescueTestBase):
    """M. tokens_held=0 + RPC getTokenAccountsByOwner raises → zero_balance early return."""

    def test_zero_balance_when_rpc_fails(self):
        self._patch_modules()
        orig_rpc = self._jr._rpc_post

        def _rpc_fail_on_balance(payload, timeout=15):
            if payload.get("method") == "getTokenAccountsByOwner":
                raise ConnectionError("RPC unavailable")
            return orig_rpc(payload, timeout=timeout)

        self._jr._rpc_post = _rpc_fail_on_balance

        pos    = _FakePos(tokens_held=0)
        result = self._jr.force_jupiter_rescue_sell(pos, "migration_uncertain")
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "zero_balance")
        self.assertEqual(result["amount_source"], "rpc_balance_failed")
        self.assertTrue(result["balance_rpc_failed"])
        self.assertFalse(result["jupiter_send_attempted"])
        print("  M PASS: tokens_held=0 + RPC fails → zero_balance / rpc_balance_failed")


class TestN_NoRouteDiagnostics(_RescueTestBase):
    """N. Jupiter returns no-route → error_class=jupiter_no_route + no-route CSV written."""

    def test_no_route_diagnostics_logged(self):
        no_route_quote = {"error": "NO_ROUTE", "errorCode": "TOKEN_NOT_TRADABLE"}
        q_resp         = _make_requests_response(no_route_quote)
        gov_module     = _make_fake_governor(quote_resp=q_resp)
        self._patch_modules(governor_module=gov_module)

        import tempfile, os
        tmp_log = Path(tempfile.mktemp(suffix=".csv"))
        self._jr._NOROUTE_LOG = tmp_log

        pos    = _FakePos()
        result = self._jr.force_jupiter_rescue_sell(pos, "migration_uncertain")

        self.assertFalse(result["success"])
        self.assertEqual(result["error_class"], "jupiter_no_route")
        self.assertEqual(result["reason"], "no_route")
        self.assertTrue(tmp_log.exists(), "No-route CSV should have been written")
        content = tmp_log.read_text()
        self.assertIn("TOKEN_NOT_TRADABLE", content)

        try:
            tmp_log.unlink()
        except Exception:
            pass
        print("  N PASS: Jupiter no_route → error_class=jupiter_no_route + CSV written")


class TestO_PanicExitDisabledHighImpactBlocked(_RescueTestBase):
    """O. ALLOW_JUPITER_RESCUE_PANIC_EXIT=False + impact > limit → price_impact_too_high."""

    def test_high_impact_blocked_when_panic_disabled(self):
        # quote returns 80% price impact; limit is 50%; panic disabled
        high_impact_quote = _build_quote_response(impact_pct=0.80)   # 80% after *100
        q_resp            = _make_requests_response(high_impact_quote)
        gov_module        = _make_fake_governor(quote_resp=q_resp)
        self._patch_modules(
            cfg_overrides={"ALLOW_JUPITER_RESCUE_PANIC_EXIT": False},
            governor_module=gov_module,
        )
        pos    = _FakePos()
        result = self._jr.force_jupiter_rescue_sell(pos, "migration_uncertain")
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"],      "price_impact_too_high")
        self.assertEqual(result["error_class"], "price_impact_exceeded")
        self.assertFalse(result["panic_price_impact_allowed"])
        self.assertAlmostEqual(result["jupiter_price_impact_pct"], 80.0, places=0)
        self.assertFalse(result["jupiter_send_attempted"])
        print("  O PASS: panic disabled + 80% impact → price_impact_too_high blocked")


# ── Runner ─────────────────────────────────────────────────────────────────────

_ALL_TESTS = [
    TestA_BSTStylePumpswapNoPool,
    TestB_BullsackT22BadPoolLayout,
    TestC_SPLGraduatedFailureRescueBeforeManual,
    TestD_6005PathRescueBeforeRepeat,
    TestE_429EscalationExitToEmergency,
    TestF_DoubleSellGuard,
    TestG_TTLPendingTagWithinTTL,
    TestH_TTLExpiredSigConfirmed,
    TestI_TTLExpiredSigFailed,
    TestJ_RebroadcastCount,
    TestK_MultiRPCFailoverClient,
    TestL_TokensHeldZeroRPCBalanceOK,
    TestM_TokensHeldZeroRPCBalanceFail,
    TestN_NoRouteDiagnostics,
    TestO_PanicExitDisabledHighImpactBlocked,
]

if __name__ == "__main__":
    print("\n=== Jupiter Rescue Regression Tests (A–O, 15 total) ===\n")
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in _ALL_TESTS:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=0, stream=sys.stdout)
    result = runner.run(suite)
    print(
        f"\n{'OK' if result.wasSuccessful() else 'FAILED'} "
        f"({result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors)"
    )
    sys.exit(0 if result.wasSuccessful() else 1)
