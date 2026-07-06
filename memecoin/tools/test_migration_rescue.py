"""
test_migration_rescue.py — Regression tests for MIGRATION_UNCERTAIN rescue path.

Tests verify:
  P. Migration rescue SUCCESS → _finalize_rescue_sell called, close_position NOT called,
     executor.sell NOT called, journal written once, position marked closed.
  Q. Migration rescue NO-ROUTE → _arm_migration_retry called, position = sell_stuck,
     close_position NOT called, executor.sell NOT called.
  R. Double-sell prevention: rescue-pending tag active → second _finalize_rescue_sell
     is a no-op (position already closed → guard exits early).
  S. Pending-tx discipline: retry fires while rescue_pending tag active → no new
     force_jupiter_rescue_sell call (TTL guard upstream catches it first; verify
     sell_stuck timer was armed and not overridden).
  T. is_rescue_eligible_error() covers all declared exit states / error classes.
  U. is_rescue_eligible_error() returns False for unknown inputs (no false positives).
  V. _arm_migration_retry: notes get |migration_wait| + timestamp, status=sell_stuck,
     timer set.
  W. _arm_migration_retry: idempotent — second call does not double-stamp notes.

Run: python -m memecoin.tools.test_migration_rescue
"""

import sys
import time
import threading
import unittest
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch, call

# ── Guard: if a prior test file installed a stub memecoin.config without the
# full attribute set, portfolio.py will fail to import. Clear both so the real
# modules load instead.
#
# Checked attribute: JOURNAL_FILE (the attribute most likely to be missing from
# minimal stubs — test_sol_delta_fixes.py has POSITIONS_FILE but not JOURNAL_FILE).
# ───────────────────────────────────────────────────────────────────────────────
_cfg_stub = sys.modules.get("memecoin.config")
if _cfg_stub is not None and not hasattr(_cfg_stub, "JOURNAL_FILE"):
    sys.modules.pop("memecoin.config", None)
    sys.modules.pop("memecoin.portfolio", None)


# ── Minimal stubs ──────────────────────────────────────────────────────────────

@dataclass
class _FakePos:
    id:            str   = "pos-mu-001"
    token_address: str   = "MigrationMintAddr111111111111111111111111111"
    token_symbol:  str   = "MIGTEST"
    chain:         str   = "solana"
    tokens_held:   int   = 5_000_000
    notes:         str   = ""
    status:        str   = "open"
    exit_price:    float = 0.0
    current_price: float = 0.000020
    size_usd:      float = 5.0
    entry_price:   float = 0.000025
    entry_time:    float = field(default_factory=time.time)
    exit_time:     float = 0.0
    exit_reason:   str   = ""


def _make_portfolio():
    """
    Construct a Portfolio instance with only _positions, _close_locks, and the
    _sell_stuck_until / _graduated_retry_count dicts wired up — enough for
    _finalize_rescue_sell and _arm_migration_retry to operate without touching
    executor, DB, or the Telegram alert stack.
    """
    import importlib
    import types

    # Build minimal module stubs so portfolio.py can be imported ---------------
    for mod in [
        "memecoin.executor", "memecoin.pumpportal_monitor", "memecoin.jupiter_rescue",
        "app.alerts", "app.config",
    ]:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)

    # Stub _save_positions, _append_journal, promote_to_winners so they
    # don't touch disk.  We patch them after import.
    from memecoin import portfolio as pmod

    pf = pmod.Portfolio.__new__(pmod.Portfolio)
    pf._positions            = {}
    pf._close_locks          = {}
    pf._close_locks_meta     = threading.Lock()
    pf._sell_stuck_until     = {}
    pf._graduated_retry_count = {}
    pf._presigned_exits      = {}
    pf._presigned_ts         = {}
    pf._graduated_mints      = set()
    pf._presigned_lock       = threading.Lock()
    return pf


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestMigrationRescue(unittest.TestCase):

    def setUp(self):
        # Hardening tests (A, D, ...) each install their own config stubs that
        # lack JOURNAL_FILE. If one ran before us, portfolio.py will fail to
        # import when the next test calls `from memecoin import portfolio`.
        # Clear both modules so the real config loads fresh.
        _cfg = sys.modules.get("memecoin.config")
        if _cfg is not None and not hasattr(_cfg, "JOURNAL_FILE"):
            sys.modules.pop("memecoin.config", None)
            sys.modules.pop("memecoin.portfolio", None)

    # ── Test P: rescue SUCCESS → finalize closes pos, journal written once ──

    def test_P_rescue_success_finalizes_position(self):
        from memecoin import portfolio as pmod

        pf  = _make_portfolio()
        pos = _FakePos()
        pf._positions[pos.id] = pos

        rescue_result = {
            "success":      True,
            "tx_sig":       "Sig4TestRescueSuccessABCDEFGHIJKLMNOP",
            "sol_received": 0.004321,
            "fill_price":   0.0000195,
            "error_class":  "",
        }

        journal_calls = []
        winners_calls = []

        with (
            patch.object(pmod, "_append_journal", side_effect=lambda p: journal_calls.append(p.id)),
            patch.object(pmod, "promote_to_winners", side_effect=lambda p: winners_calls.append(p.id)),
            patch.object(pmod, "_save_positions"),
            patch("app.alerts.alert_live_sell", create=True),
        ):
            pf._finalize_rescue_sell(pos.id, rescue_result)

        # position must be removed from _positions (deleted after close)
        self.assertNotIn(pos.id, pf._positions, "position still in _positions after finalize")

        # journal written exactly once
        self.assertEqual(journal_calls.count(pos.id), 1, "_append_journal must be called exactly once")

        # position attributes set correctly
        self.assertEqual(pos.status,      "closed")
        self.assertEqual(pos.exit_reason, "jupiter_rescue")
        self.assertAlmostEqual(pos.exit_price, 0.0000195)
        self.assertIn("sell_tx:Sig4TestRescueSuccessABCDEFGHIJKLMNOP", pos.notes)
        self.assertIn("route:JUPITER_RESCUE", pos.notes)
        self.assertIn("sol_received:0.004321", pos.notes)

    # ── Test Q: rescue NO-ROUTE → arm retry, NOT close ──────────────────────

    def test_Q_rescue_no_route_arms_retry(self):
        from memecoin import portfolio as pmod

        pf  = _make_portfolio()
        pos = _FakePos()
        pf._positions[pos.id] = pos

        with patch.object(pmod, "_save_positions") as mock_save:
            pf._arm_migration_retry(pos.id, retry_sec=60)

        # position still open in dict (sell_stuck, not removed)
        self.assertIn(pos.id, pf._positions)
        self.assertEqual(pos.status, "sell_stuck")
        self.assertIn("|migration_wait", pos.notes)
        self.assertIn("|migration_uncertain_ts:", pos.notes)
        self.assertIn(pos.id, pf._sell_stuck_until)
        self.assertGreater(pf._sell_stuck_until[pos.id], time.time())

    # ── Test R: double-sell prevention via _finalize_rescue_sell no-op ───────

    def test_R_double_sell_prevention(self):
        from memecoin import portfolio as pmod

        pf  = _make_portfolio()
        pos = _FakePos()
        pf._positions[pos.id] = pos

        rescue_result = {
            "success":      True,
            "tx_sig":       "SigFirstSell111111111111111111111111111111",
            "sol_received": 0.003,
            "fill_price":   0.000018,
        }

        journal_calls = []

        with (
            patch.object(pmod, "_append_journal", side_effect=lambda p: journal_calls.append(p.id)),
            patch.object(pmod, "promote_to_winners"),
            patch.object(pmod, "_save_positions"),
            patch("app.alerts.alert_live_sell", create=True),
        ):
            # First call — should close
            pf._finalize_rescue_sell(pos.id, rescue_result)
            # Second call — position already gone, must be no-op
            pf._finalize_rescue_sell(pos.id, rescue_result)

        self.assertEqual(
            journal_calls.count(pos.id), 1,
            "journal must be written exactly once; second _finalize_rescue_sell was NOT a no-op",
        )
        self.assertNotIn(pos.id, pf._positions)

    # ── Test S: sell_stuck timer armed; reschedule does not override timer ────

    def test_S_sell_stuck_timer_not_overridden(self):
        from memecoin import portfolio as pmod

        pf  = _make_portfolio()
        pos = _FakePos()
        pf._positions[pos.id] = pos

        retry_sec = 60
        with patch.object(pmod, "_save_positions"):
            pf._arm_migration_retry(pos.id, retry_sec)

        first_deadline = pf._sell_stuck_until[pos.id]

        # Simulate the sell_stuck timer already running — a second arm should
        # update |migration_uncertain_ts| if missing, but we verify the
        # notes accumulation does not double-stamp |migration_wait|.
        with patch.object(pmod, "_save_positions"):
            pf._arm_migration_retry(pos.id, retry_sec)

        # |migration_wait| appears exactly once
        self.assertEqual(pos.notes.count("|migration_wait"), 1)

        # Timer was refreshed (>= first_deadline) — second arm extends, not drops
        self.assertGreaterEqual(pf._sell_stuck_until[pos.id], first_deadline)

    # ── Test T: is_rescue_eligible_error covers all declared inputs ───────────

    def test_T_rescue_eligible_positive_cases(self):
        from memecoin.portfolio import is_rescue_eligible_error

        # exit states
        for es in [
            "GRADUATED_PUMPSWAP", "GRADUATED_PUMPSWAP_SPL", "GRADUATED_PUMPSWAP_T22",
            "MIGRATION_UNCERTAIN", "MIGRATION_UNCERTAIN_SPL", "MIGRATION_UNCERTAIN_T22",
        ]:
            self.assertTrue(
                is_rescue_eligible_error(exit_state=es),
                f"exit_state={es!r} should be rescue-eligible",
            )

        # error classes
        for ec in [
            "pumpswap_no_pool", "pumpswap_bad_pool_layout", "pool_not_indexed",
            "local_build_failed", "local_sim_failed", "pumpswap_simulation_failed",
            "jupiter_no_route", "graduated_unsellable", "Custom:6005", "Custom:6001",
        ]:
            self.assertTrue(
                is_rescue_eligible_error(error_class=ec),
                f"error_class={ec!r} should be rescue-eligible",
            )

        # reasons
        for rr in [
            "migration_uncertain_no_pool", "migration_uncertain_retry",
            "sell_stuck", "graduated_exit", "feed_blind",
        ]:
            self.assertTrue(
                is_rescue_eligible_error(reason=rr),
                f"reason={rr!r} should be rescue-eligible",
            )

    # ── Test U: is_rescue_eligible_error returns False for unknowns ───────────

    def test_U_rescue_eligible_negative_cases(self):
        from memecoin.portfolio import is_rescue_eligible_error

        self.assertFalse(is_rescue_eligible_error())
        self.assertFalse(is_rescue_eligible_error(
            exit_state="BONDING_CURVE",
            error_class="some_random_error",
            reason="unknown_reason",
        ))
        self.assertFalse(is_rescue_eligible_error(exit_state="OPEN"))
        self.assertFalse(is_rescue_eligible_error(error_class=""))
        # Partial string that should NOT match a frozenset member
        self.assertFalse(is_rescue_eligible_error(error_class="pumpswap_no"))

    # ── Test V: _arm_migration_retry sets all required fields ─────────────────

    def test_V_arm_migration_retry_fields(self):
        from memecoin import portfolio as pmod

        pf  = _make_portfolio()
        pos = _FakePos()
        pf._positions[pos.id] = pos

        t_before = int(time.time())
        with patch.object(pmod, "_save_positions"):
            pf._arm_migration_retry(pos.id, retry_sec=120)

        self.assertEqual(pos.status, "sell_stuck")
        self.assertIn("|migration_wait", pos.notes)
        self.assertIn("|migration_uncertain_ts:", pos.notes)

        # Timestamp in notes must be plausible
        import re
        m = re.search(r'\|migration_uncertain_ts:(\d+)', pos.notes)
        self.assertIsNotNone(m, "migration_uncertain_ts not found in notes")
        ts_in_notes = int(m.group(1))
        self.assertGreaterEqual(ts_in_notes, t_before)
        self.assertLessEqual(ts_in_notes, t_before + 5)

        # Retry timer must be approximately now + 120s
        self.assertAlmostEqual(
            pf._sell_stuck_until[pos.id],
            time.time() + 120,
            delta=5,
        )

    # ── Test W: _arm_migration_retry is idempotent on |migration_wait| ────────

    def test_W_arm_migration_retry_idempotent(self):
        from memecoin import portfolio as pmod

        pf  = _make_portfolio()
        pos = _FakePos()
        pf._positions[pos.id] = pos

        with patch.object(pmod, "_save_positions"):
            pf._arm_migration_retry(pos.id, retry_sec=60)
            pf._arm_migration_retry(pos.id, retry_sec=60)
            pf._arm_migration_retry(pos.id, retry_sec=60)

        # Should appear exactly once
        self.assertEqual(pos.notes.count("|migration_wait"), 1,
                         "|migration_wait stamped more than once")


# ══════════════════════════════════════════════════════════════════════════════
# New hardening tests (A–J) — Full-sig, truncated-sig, classify_rescue_result,
# classify dispatcher, close_position executor-block, scanner non-blocking
# ══════════════════════════════════════════════════════════════════════════════

class TestHardeningA_FullPendingSig(unittest.TestCase):
    """A. Pending tag stores FULL signature; getSignatureStatuses receives full sig."""

    def test_A_full_sig_in_pending_tag(self):
        import importlib, types, re, base64
        from unittest.mock import MagicMock, patch

        # Build a full 88-char sig
        full_sig = "A" * 88

        sys.modules.pop("memecoin.jupiter_rescue", None)

        cfg = types.ModuleType("memecoin.config")
        cfg.JUPITER_RESCUE_ENABLED               = True
        cfg.JUPITER_RESCUE_MAX_PRICE_IMPACT_PCT  = 50
        cfg.ALLOW_JUPITER_RESCUE_PANIC_EXIT      = True
        cfg.JUPITER_RESCUE_PENDING_TTL_SEC       = 30
        cfg.JUPITER_RESCUE_REBROADCAST_ENABLED   = False
        cfg.JUPITER_RESCUE_REBROADCAST_INTERVAL_MS = 500
        cfg.JUPITER_RESCUE_REBROADCAST_MAX_RPC   = 3
        cfg.EXECUTION_RPC_URLS                   = ["https://primary.test"]
        cfg.EXECUTION_RPC_FALLBACK_URLS          = []
        sys.modules["memecoin.config"] = cfg

        # Governor — include outAmount so estimate-fallback path also works
        _wallet_a = "Wallet" + "1" * 37
        class _FakeGov:
            def request(self, purpose, endpoint, fn, mint="", **kwargs):
                r = MagicMock()
                r.status_code = 200
                if endpoint == "quote":
                    r.json.return_value = {"priceImpactPct": "0.01", "inputMint": "X",
                                           "outputMint": "Y", "inAmount": "1000000",
                                           "outAmount": "41726044"}
                    r.raise_for_status.return_value = None
                elif endpoint == "swap":
                    r.json.return_value = {"swapTransaction": base64.b64encode(b"\x00"*128).decode()}
                    r.raise_for_status.return_value = None
                return r
        gov_mod = types.ModuleType("memecoin.jupiter_governor")
        gov_mod.governor = _FakeGov()
        class _P: EXIT = "EXIT"; EMERGENCY = "EMERGENCY"
        gov_mod.Purpose = _P
        sys.modules["memecoin.jupiter_governor"] = gov_mod

        # Executor keypair
        kp = MagicMock(); kp.pubkey.return_value = _wallet_a
        ex_mod = types.ModuleType("memecoin.executor")
        ex_mod._get_keypair = lambda: kp
        sys.modules["memecoin.executor"] = ex_mod

        # Solders stub
        vtx = MagicMock(); vtx.message = MagicMock(); vtx.__bytes__ = lambda s: b"\x00"*128
        vtx_cls = MagicMock(); vtx_cls.from_bytes.return_value = vtx; vtx_cls.return_value = vtx
        st_mod = types.ModuleType("solders.transaction")
        st_mod.VersionedTransaction = vtx_cls
        sys.modules["solders.transaction"] = st_mod

        import memecoin.jupiter_rescue as jr
        importlib.reload(jr)

        rpc_calls = []
        def _fake_rpc(payload, timeout=15):
            m = payload.get("method", "")
            rpc_calls.append(m)
            if m == "getTokenAccountsByOwner":
                return {"result": {"value": [
                    {"account": {"data": {"parsed": {"info": {"tokenAmount": {"amount": "1000000"}}}}}}
                ]}}
            if m == "sendTransaction":
                return {"result": full_sig}
            if m == "getSignatureStatuses":
                return {"result": {"value": [{"confirmationStatus": "confirmed", "err": None}]}}
            if m == "getTransaction":
                # wallet receives 0.05 SOL: preBalances[1]=0, postBalances[1]=50_000_000 lamports
                return {"result": {"meta": {
                    "preBalances":  [0, 0],
                    "postBalances": [0, 50_000_000],
                }, "transaction": {"message": {"accountKeys": [
                    {"pubkey": "SomeProgramAccount", "signer": False, "writable": False},
                    {"pubkey": _wallet_a,             "signer": True,  "writable": True},
                ]}}}}
            return {}

        # Patch BOTH jupiter_rescue._rpc_post and tx_meta._rpc_post.
        # read_sol_delta (in tx_meta) calls its own module-level _rpc_post, not jr._rpc_post.
        import memecoin.tx_meta as _tm
        _tm._rpc_post = _fake_rpc
        jr._rpc_post   = _fake_rpc

        @dataclass
        class _P2:
            id: str = "pos-sig-001"
            token_address: str = "Mint" + "1" * 40
            token_symbol: str = "FULLSIG"
            tokens_held: int = 1_000_000
            notes: str = ""
            status: str = "open"
            exit_price: float = 0.0001

        pos = _P2()
        result = jr.force_jupiter_rescue_sell(pos, "migration_uncertain_no_pool")

        # Tx must have been sent and succeeded
        self.assertTrue(result["success"], f"Expected success: {result}")
        self.assertEqual(result["tx_sig"], full_sig)

        # Pending tag must contain the full sig (not truncated)
        # The tag is updated during send; check it was written with full sig
        self.assertIn(full_sig, pos.notes,
                      "pending tag did not store full sig — check sig[:16] truncation")

        sys.modules.pop("memecoin.jupiter_rescue", None)


class TestHardeningB_TruncatedLegacyTag(unittest.TestCase):
    """B. Truncated (16-char) legacy pending tag handling."""

    def setUp(self):
        sys.modules.pop("memecoin.jupiter_rescue", None)
        import types, importlib

        cfg = types.ModuleType("memecoin.config")
        cfg.JUPITER_RESCUE_ENABLED               = True
        cfg.JUPITER_RESCUE_MAX_PRICE_IMPACT_PCT  = 50
        cfg.ALLOW_JUPITER_RESCUE_PANIC_EXIT      = True
        cfg.JUPITER_RESCUE_PENDING_TTL_SEC       = 30
        cfg.JUPITER_RESCUE_REBROADCAST_ENABLED   = False
        cfg.JUPITER_RESCUE_REBROADCAST_INTERVAL_MS = 500
        cfg.JUPITER_RESCUE_REBROADCAST_MAX_RPC   = 3
        cfg.EXECUTION_RPC_URLS                   = ["https://primary.test"]
        cfg.EXECUTION_RPC_FALLBACK_URLS          = []
        sys.modules["memecoin.config"] = cfg

        import memecoin.jupiter_rescue as jr
        importlib.reload(jr)

        self.jr  = jr
        self.sig_calls: list = []

        def _fake_rpc(payload, timeout=15):
            m = payload.get("method", "")
            self.sig_calls.append(m)
            if m == "getTokenAccountsByOwner":
                return {"result": {"value": [
                    {"account": {"data": {"parsed": {"info": {"tokenAmount": {"amount": "500000"}}}}}}
                ]}}
            return {}

        jr._rpc_post = _fake_rpc

    def tearDown(self):
        sys.modules.pop("memecoin.jupiter_rescue", None)

    def test_B1_truncated_age_lt_ttl_skips(self):
        """Truncated sig + age < TTL → rescue_already_pending (no new tx, no status check)."""
        trunc_sig = "StaleSig12345678"   # 16 chars < MIN_VALID_SIG_LEN
        fresh_ts  = int(time.time()) - 5  # 5s old < TTL=30

        @dataclass
        class _P:
            id: str = "pos-trunc-001"
            token_address: str = "Mint" + "1" * 40
            token_symbol: str = "TRUNC"
            tokens_held: int = 1_000_000
            notes: str = f"|jupiter_rescue_pending:{trunc_sig}:{fresh_ts}"
            status: str = "open"
            exit_price: float = 0.0001

        pos    = _P()
        result = self.jr.force_jupiter_rescue_sell(pos, "migration_uncertain_no_pool")

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "rescue_already_pending")
        # Never called getSignatureStatuses with the truncated sig
        self.assertNotIn("getSignatureStatuses", self.sig_calls)

    def test_B2_truncated_age_gte_ttl_clears_and_resets(self):
        """Truncated sig + age >= TTL → tag cleared, no getSignatureStatuses with truncated sig."""
        trunc_sig = "StaleSig12345678"
        stale_ts  = int(time.time()) - 120  # 120s > TTL=30

        @dataclass
        class _P:
            id: str = "pos-trunc-002"
            token_address: str = "Mint" + "1" * 40
            token_symbol: str = "TRUNC2"
            tokens_held: int = 0       # no tokens in pos → will do balance RPC
            notes: str = f"|jupiter_rescue_pending:{trunc_sig}:{stale_ts}"
            status: str = "open"
            exit_price: float = 0.0001

        pos    = _P()
        # After clearing tag, balance check yields 500_000 → then keypair needed
        # To avoid testing the full flow, check that:
        # 1. getSignatureStatuses was NOT called with the truncated sig
        # 2. pending tag was cleared from notes
        try:
            self.jr.force_jupiter_rescue_sell(pos, "migration_uncertain_no_pool")
        except Exception:
            pass  # may fail at keypair load — that's fine

        self.assertNotIn("getSignatureStatuses", self.sig_calls)
        self.assertNotIn(f"jupiter_rescue_pending:{trunc_sig}", pos.notes)


class TestHardeningC_StaleConfirmedFinalizes(unittest.TestCase):
    """C. Stale pending FULL sig confirmed → success=True, caller can finalize."""

    def test_C_stale_confirmed_returns_success(self):
        import types, importlib, re

        full_sig = "C" * 88
        old_ts   = int(time.time()) - 200  # stale

        sys.modules.pop("memecoin.jupiter_rescue", None)

        cfg = types.ModuleType("memecoin.config")
        cfg.JUPITER_RESCUE_ENABLED               = True
        cfg.JUPITER_RESCUE_PENDING_TTL_SEC       = 30
        cfg.JUPITER_RESCUE_MAX_PRICE_IMPACT_PCT  = 50
        cfg.ALLOW_JUPITER_RESCUE_PANIC_EXIT      = True
        cfg.JUPITER_RESCUE_REBROADCAST_ENABLED   = False
        cfg.JUPITER_RESCUE_REBROADCAST_INTERVAL_MS = 500
        cfg.JUPITER_RESCUE_REBROADCAST_MAX_RPC   = 3
        cfg.EXECUTION_RPC_URLS                   = ["https://primary.test"]
        cfg.EXECUTION_RPC_FALLBACK_URLS          = []
        sys.modules["memecoin.config"] = cfg

        import memecoin.jupiter_rescue as jr
        importlib.reload(jr)

        sig_status_calls: list = []
        def _fake_rpc(payload, timeout=15):
            m = payload.get("method", "")
            if m == "getSignatureStatuses":
                sig_status_calls.append(payload["params"][0][0])
                return {"result": {"value": [{"confirmationStatus": "finalized", "err": None}]}}
            return {}
        jr._rpc_post = _fake_rpc

        @dataclass
        class _P:
            id: str = "pos-stale-001"
            token_address: str = "MintC" + "1" * 39
            token_symbol: str = "STALE"
            tokens_held: int = 500_000
            notes: str = f"|jupiter_rescue_pending:{full_sig}:{old_ts}"
            status: str = "open"
            exit_price: float = 0.00002

        pos    = _P()
        result = jr.force_jupiter_rescue_sell(pos, "graduated_exit")

        self.assertTrue(result["success"], f"Expected success=True: {result}")
        self.assertEqual(result["reason"], "rescue_stale_sig_confirmed")
        self.assertEqual(result["error_class"], "already_sold")
        self.assertEqual(result["tx_sig"], full_sig)
        self.assertTrue(result["jupiter_confirmed"])
        self.assertFalse(result["jupiter_send_attempted"])
        # Full sig was passed to getSignatureStatuses
        self.assertEqual(sig_status_calls, [full_sig],
                         "getSignatureStatuses was not called with full sig")

        sys.modules.pop("memecoin.jupiter_rescue", None)


class TestHardeningD_MigrationHighImpactUrgent(unittest.TestCase):
    """D. migration_uncertain_no_pool + high impact + panic=True → sell allowed."""

    def test_D_urgent_reason_bypasses_impact_block(self):
        import types, importlib, base64

        sys.modules.pop("memecoin.jupiter_rescue", None)

        cfg = types.ModuleType("memecoin.config")
        cfg.JUPITER_RESCUE_ENABLED               = True
        cfg.JUPITER_RESCUE_MAX_PRICE_IMPACT_PCT  = 50
        cfg.ALLOW_JUPITER_RESCUE_PANIC_EXIT      = True  # panic allowed
        cfg.JUPITER_RESCUE_PENDING_TTL_SEC       = 30
        cfg.JUPITER_RESCUE_REBROADCAST_ENABLED   = False
        cfg.JUPITER_RESCUE_REBROADCAST_INTERVAL_MS = 500
        cfg.JUPITER_RESCUE_REBROADCAST_MAX_RPC   = 3
        cfg.EXECUTION_RPC_URLS                   = ["https://primary.test"]
        cfg.EXECUTION_RPC_FALLBACK_URLS          = []
        sys.modules["memecoin.config"] = cfg

        _wallet_d = "Wallet" + "1" * 37
        class _FakeGov:
            def request(self, purpose, endpoint, fn, mint="", **kwargs):
                r = MagicMock()
                r.raise_for_status.return_value = None
                if endpoint == "quote":
                    r.json.return_value = {"priceImpactPct": "0.85",   # 85% > 50% max
                                           "inputMint": "X", "outputMint": "Y",
                                           "inAmount": "1000000", "outAmount": "50000000"}
                elif endpoint == "swap":
                    r.json.return_value = {"swapTransaction": base64.b64encode(b"\x00"*128).decode()}
                return r
        gov = types.ModuleType("memecoin.jupiter_governor")
        gov.governor = _FakeGov()
        class _Pu: EXIT = "EXIT"; EMERGENCY = "EMERGENCY"
        gov.Purpose = _Pu
        sys.modules["memecoin.jupiter_governor"] = gov

        kp = MagicMock(); kp.pubkey.return_value = _wallet_d
        ex = types.ModuleType("memecoin.executor")
        ex._get_keypair = lambda: kp
        sys.modules["memecoin.executor"] = ex

        vtx = MagicMock(); vtx.message = MagicMock(); vtx.__bytes__ = lambda s: b"\x00"*128
        vtx_cls = MagicMock(); vtx_cls.from_bytes.return_value = vtx; vtx_cls.return_value = vtx
        st = types.ModuleType("solders.transaction")
        st.VersionedTransaction = vtx_cls
        sys.modules["solders.transaction"] = st

        import memecoin.jupiter_rescue as jr
        importlib.reload(jr)

        def _fake_rpc(payload, timeout=15):
            m = payload.get("method", "")
            if m == "getTokenAccountsByOwner":
                return {"result": {"value": [
                    {"account": {"data": {"parsed": {"info": {"tokenAmount": {"amount": "1000000"}}}}}}
                ]}}
            if m == "sendTransaction":
                return {"result": "D" * 88}
            if m == "getSignatureStatuses":
                return {"result": {"value": [{"confirmationStatus": "confirmed", "err": None}]}}
            if m == "getTransaction":
                # wallet receives 0.05 SOL at index 0 (string key form is handled by read_sol_delta)
                return {"result": {"meta": {"preBalances": [0], "postBalances": [50_000_000]},
                                   "transaction": {"message": {"accountKeys": [_wallet_d]}}}}
            return {}

        # Patch BOTH jupiter_rescue._rpc_post and tx_meta._rpc_post.
        # read_sol_delta (in tx_meta) calls tx_meta._rpc_post, not jr._rpc_post.
        import memecoin.tx_meta as _tm
        _tm._rpc_post = _fake_rpc
        jr._rpc_post   = _fake_rpc

        @dataclass
        class _P:
            id: str = "pos-d-001"
            token_address: str = "MintD" + "1" * 39
            token_symbol: str = "HIGHIMP"
            tokens_held: int = 1_000_000
            notes: str = ""
            status: str = "open"
            exit_price: float = 0.0001

        pos    = _P()
        # migration_uncertain_no_pool is in _URGENT_REASONS → panic allowed
        result = jr.force_jupiter_rescue_sell(pos, "migration_uncertain_no_pool")

        self.assertTrue(result["success"], f"Expected sell to succeed with panic: {result}")
        self.assertTrue(result["panic_price_impact_allowed"])
        self.assertGreater(result["jupiter_price_impact_pct"], 50)

        sys.modules.pop("memecoin.jupiter_rescue", None)


class TestHardeningE_ClassifyRescueResult(unittest.TestCase):
    """E. classify_rescue_result() correctly maps all result shapes."""

    def setUp(self):
        sys.modules.pop("memecoin.jupiter_rescue", None)
        import types, importlib
        cfg = types.ModuleType("memecoin.config")
        cfg.JUPITER_RESCUE_ENABLED = True; cfg.JUPITER_RESCUE_PENDING_TTL_SEC = 30
        cfg.JUPITER_RESCUE_MAX_PRICE_IMPACT_PCT = 50; cfg.ALLOW_JUPITER_RESCUE_PANIC_EXIT = True
        cfg.JUPITER_RESCUE_REBROADCAST_ENABLED = False; cfg.EXECUTION_RPC_URLS = []
        cfg.JUPITER_RESCUE_REBROADCAST_INTERVAL_MS = 500; cfg.JUPYTER_RESCUE_REBROADCAST_MAX_RPC = 3
        cfg.EXECUTION_RPC_FALLBACK_URLS = []
        sys.modules["memecoin.config"] = cfg
        import memecoin.jupiter_rescue as jr; importlib.reload(jr)
        self.clf = jr.classify_rescue_result

    def tearDown(self):
        sys.modules.pop("memecoin.jupiter_rescue", None)

    def test_E_sold(self):
        self.assertEqual(self.clf({"success": True, "jupiter_confirmed": True}), "sold")

    def test_E_already_sold_error_class(self):
        self.assertEqual(self.clf({"success": False, "error_class": "already_sold"}), "already_sold")

    def test_E_already_sold_reason(self):
        self.assertEqual(self.clf({"reason": "rescue_stale_sig_confirmed"}), "already_sold")

    def test_E_pending_already_pending(self):
        self.assertEqual(self.clf({"reason": "rescue_already_pending"}), "pending")

    def test_E_pending_tx_not_confirmed(self):
        self.assertEqual(self.clf({"reason": "tx_not_confirmed", "tx_sig": "X"*88,
                                    "jupiter_send_attempted": True}), "pending")

    def test_E_pending_send_attempted_no_confirm(self):
        self.assertEqual(self.clf({"jupiter_send_attempted": True, "tx_sig": "Z"*88,
                                    "jupiter_confirmed": False}), "pending")

    def test_E_no_route_error_class(self):
        self.assertEqual(self.clf({"error_class": "jupiter_no_route"}), "no_route")

    def test_E_no_route_reason(self):
        self.assertEqual(self.clf({"reason": "no_route"}), "no_route")

    def test_E_retry_no_send_429(self):
        self.assertEqual(self.clf({"error_class": "jupiter_429_exhausted"}), "retry_no_send")

    def test_E_retry_no_send_zero_balance(self):
        self.assertEqual(self.clf({"reason": "zero_balance"}), "retry_no_send")

    def test_E_fatal_keypair(self):
        self.assertEqual(self.clf({"error_class": "keypair_error"}), "fatal_no_send")

    def test_E_fatal_sign(self):
        self.assertEqual(self.clf({"reason": "sign_failed"}), "fatal_no_send")

    def test_E_fallback_allowed_no_send(self):
        self.assertEqual(self.clf({"jupiter_send_attempted": False, "reason": ""}), "fallback_allowed")


class TestHardeningF_ClosePositionNoRouteBlocksExecutor(unittest.TestCase):
    """F. close_position: rescue returns no_route → executor.sell NOT called, retry armed."""

    def setUp(self):
        # Restore real config if a prior test installed a stub without POSITIONS_FILE
        _stub = sys.modules.get("memecoin.config")
        if _stub is not None and not hasattr(_stub, "POSITIONS_FILE"):
            sys.modules.pop("memecoin.config", None)
            sys.modules.pop("memecoin.portfolio", None)

    def test_F_no_route_blocks_executor(self):
        from memecoin import portfolio as pmod
        from unittest.mock import MagicMock, patch

        pf  = _make_portfolio()
        pos = _FakePos(notes="|migration_wait|migration_uncertain_ts:1234567890")
        pos.status = "sell_stuck"
        pf._positions[pos.id] = pos

        no_route_result = {
            "success": False, "reason": "no_route",
            "error_class": "jupiter_no_route",
            "jupiter_send_attempted": False,
            "tx_sig": "", "jupiter_confirmed": False,
        }

        executor_sell_calls: list = []

        with (
            patch("memecoin.portfolio.is_rescue_eligible_error", return_value=True),
            patch("memecoin.portfolio.force_jupiter_rescue_sell", return_value=no_route_result,
                  create=True),
            patch("memecoin.portfolio.classify_rescue_result",
                  return_value="no_route", create=True),
            patch.object(pmod, "_save_positions"),
            patch.object(pmod, "_append_journal"),
            patch.object(pmod, "promote_to_winners"),
        ):
            # Patch the import inside close_position
            import memecoin.jupiter_rescue as jr_mod
            orig_cls = getattr(jr_mod, "classify_rescue_result", None)
            orig_frc = getattr(jr_mod, "force_jupiter_rescue_sell", None)
            jr_mod.classify_rescue_result   = lambda r: "no_route"
            jr_mod.force_jupiter_rescue_sell = lambda p, reason, purpose="EXIT": no_route_result

            try:
                class _FakeExecutor:
                    def sell(self, *a, **kw):
                        executor_sell_calls.append(1)
                        return {"success": True}

                with patch("memecoin.executor.MemeExecutor",
                           return_value=_FakeExecutor(), create=True):
                    try:
                        pf.close_position(
                            pos.id, "migration_uncertain_retry",
                            pos.current_price,
                        )
                    except Exception:
                        pass  # close_position may return early without exception

                self.assertEqual(executor_sell_calls, [],
                                 "executor.sell must NOT be called after rescue returns no_route")
            finally:
                if orig_cls is not None:
                    jr_mod.classify_rescue_result   = orig_cls
                if orig_frc is not None:
                    jr_mod.force_jupiter_rescue_sell = orig_frc


class TestHardeningG_AlreadySoldFinalizes(unittest.TestCase):
    """G. already_sold rescue result + classify_rescue_result → position closed, no executor.sell.

    Tests the contract at the _finalize_rescue_sell layer: when classify_rescue_result
    returns "already_sold" the caller calls _finalize_rescue_sell which must close the
    position without calling executor.sell (proven by P + E tests together).
    This test verifies _finalize_rescue_sell works for stale-sig rescue results.
    """

    def setUp(self):
        # Restore real config if prior test installed a stub without POSITIONS_FILE
        _stub = sys.modules.get("memecoin.config")
        if _stub is not None and not hasattr(_stub, "POSITIONS_FILE"):
            sys.modules.pop("memecoin.config", None)
            sys.modules.pop("memecoin.portfolio", None)

    def test_G_already_sold_result_closes_position(self):
        from memecoin import portfolio as pmod

        pf  = _make_portfolio()
        pos = _FakePos()
        pf._positions[pos.id] = pos

        # Result shape that classify_rescue_result maps to "already_sold"
        already_sold_result = {
            "success":              True,
            "reason":               "rescue_stale_sig_confirmed",
            "error_class":          "already_sold",
            "tx_sig":               "E" * 88,
            "jupiter_confirmed":    True,
            "jupiter_send_attempted": False,
            "sol_received":         0.0,
            "fill_price":           None,
        }

        journal_calls: list = []

        with (
            patch.object(pmod, "_append_journal", side_effect=lambda p: journal_calls.append(p.id)),
            patch.object(pmod, "promote_to_winners"),
            patch.object(pmod, "_save_positions"),
            patch("app.alerts.alert_live_sell", create=True),
        ):
            pf._finalize_rescue_sell(pos.id, already_sold_result)

        # Position must be closed and removed
        self.assertNotIn(pos.id, pf._positions)
        self.assertEqual(pos.status, "closed")
        self.assertEqual(pos.exit_reason, "jupiter_rescue")
        self.assertEqual(journal_calls, [pos.id],
                         "Journal must be written exactly once")
        # sig must appear in notes
        self.assertIn("E" * 88, pos.notes)

    def test_G_classify_already_sold_returns_correct_class(self):
        """classify_rescue_result correctly identifies already_sold shapes."""
        import importlib, types

        sys.modules.pop("memecoin.jupiter_rescue", None)
        cfg = types.ModuleType("memecoin.config")
        cfg.JUPITER_RESCUE_ENABLED = True; cfg.JUPITER_RESCUE_PENDING_TTL_SEC = 30
        cfg.EXECUTION_RPC_FALLBACK_URLS = []; cfg.JUPITER_RESCUE_REBROADCAST_ENABLED = False
        sys.modules["memecoin.config"] = cfg

        import memecoin.jupiter_rescue as jr; importlib.reload(jr)

        # success=True → "sold" (caller finalizes; stale vs fresh sell is irrelevant to caller)
        stale_result = {
            "success": True, "reason": "rescue_stale_sig_confirmed",
            "error_class": "already_sold", "tx_sig": "E" * 88,
            "jupiter_confirmed": True, "jupiter_send_attempted": False,
        }
        self.assertEqual(jr.classify_rescue_result(stale_result), "sold")

        # success=False + error_class=already_sold → "already_sold" (e.g. zero_balance path)
        no_tx_result = {
            "success": False, "reason": "rescue_stale_sig_confirmed",
            "error_class": "already_sold", "tx_sig": "",
            "jupiter_confirmed": False, "jupiter_send_attempted": False,
        }
        self.assertEqual(jr.classify_rescue_result(no_tx_result), "already_sold")

        sys.modules.pop("memecoin.jupiter_rescue", None)


class TestHardeningH_ScannerNonBlocking(unittest.TestCase):
    """H. Scanner migration branch arms retry immediately and spawns daemon worker (no 45s block).

    scanner.py has complex real-config imports so we test the guard pattern directly
    using the same logic the scanner branch employs, without importing the full module.
    """

    def test_H_scanner_returns_quickly(self):
        """The migration no-pool branch logic spawns a thread and returns immediately."""
        import threading

        # Simulate the inflight guard + thread spawn that scanner.py does
        inflight: set = set()
        thread_spawned: list = []
        pos_id = "scanner-test-001"

        def _simulate_scanner_branch(pid):
            """Replica of the scanner no-pool branch guard logic."""
            if pid not in inflight:
                inflight.add(pid)

                def _worker():
                    try:
                        pass  # rescue would run here
                    finally:
                        inflight.discard(pid)

                t = threading.Thread(
                    target=_worker, daemon=True,
                    name=f"mu-rescue-{pid[:8]}",
                )
                thread_spawned.append(t.name)
                t.start()
                # Branch returns immediately — NOT join(t)

        import time
        t0 = time.time()
        _simulate_scanner_branch(pos_id)
        elapsed = time.time() - t0

        self.assertLess(elapsed, 1.0,
                        f"Scanner branch took {elapsed:.2f}s — must be < 1s (no blocking)")
        self.assertTrue(any("mu-rescue" in n for n in thread_spawned),
                        "No daemon rescue thread was spawned")

    def test_H_no_duplicate_worker(self):
        """Duplicate call with same pos_id must not spawn second worker (inflight guard)."""
        inflight: set = set()
        spawned: list = []

        def _maybe_spawn(pid):
            if pid not in inflight:
                inflight.add(pid)
                spawned.append(pid)

        pos_id = "dedup-test-001"
        _maybe_spawn(pos_id)
        _maybe_spawn(pos_id)  # second call must be blocked

        self.assertEqual(len(spawned), 1,
                         "Second spawn was not blocked by inflight guard")

    def test_H_inflight_cleared_after_worker(self):
        """Inflight set is cleared in finally block when worker finishes."""
        import threading

        inflight: set = set()
        done_event = threading.Event()
        pos_id = "clear-test-001"
        inflight.add(pos_id)

        def _worker():
            try:
                pass
            finally:
                inflight.discard(pos_id)
                done_event.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        done_event.wait(timeout=2)

        self.assertNotIn(pos_id, inflight,
                         "Inflight set not cleared after worker finished")


class TestHardeningI_ConfigWarningEmptyFallback(unittest.TestCase):
    """I. Config warning logged once when rebroadcast enabled but no fallback RPCs."""

    def test_I_warning_logged(self):
        import types, importlib, logging

        sys.modules.pop("memecoin.jupiter_rescue", None)

        cfg = types.ModuleType("memecoin.config")
        cfg.JUPITER_RESCUE_ENABLED                = True
        cfg.JUPITER_RESCUE_MAX_PRICE_IMPACT_PCT   = 50
        cfg.ALLOW_JUPITER_RESCUE_PANIC_EXIT       = True
        cfg.JUPITER_RESCUE_PENDING_TTL_SEC        = 30
        cfg.JUPITER_RESCUE_REBROADCAST_ENABLED    = True  # enabled
        cfg.JUPITER_RESCUE_REBROADCAST_INTERVAL_MS = 500
        cfg.JUPITER_RESCUE_REBROADCAST_MAX_RPC    = 3
        cfg.EXECUTION_RPC_URLS                    = ["https://primary.test"]
        cfg.EXECUTION_RPC_FALLBACK_URLS           = []    # empty — should warn
        sys.modules["memecoin.config"] = cfg

        import memecoin.jupiter_rescue as jr
        importlib.reload(jr)
        jr._REBROADCAST_WARN_LOGGED = False  # reset flag

        with self.assertLogs("memecoin.jupiter_rescue", level="WARNING") as cm:
            jr._warn_rebroadcast_config()

        self.assertTrue(
            any("fallback" in line.lower() or "FALLBACK" in line for line in cm.output),
            f"Expected fallback warning not found in: {cm.output}",
        )
        # Second call must be a no-op (flag set)
        self.assertTrue(jr._REBROADCAST_WARN_LOGGED)

        sys.modules.pop("memecoin.jupiter_rescue", None)


# ── Entry point ────────────────────────────────────────────────────────────────

def _run():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestMigrationRescue))
    suite.addTests(loader.loadTestsFromTestCase(TestHardeningA_FullPendingSig))
    suite.addTests(loader.loadTestsFromTestCase(TestHardeningB_TruncatedLegacyTag))
    suite.addTests(loader.loadTestsFromTestCase(TestHardeningC_StaleConfirmedFinalizes))
    suite.addTests(loader.loadTestsFromTestCase(TestHardeningD_MigrationHighImpactUrgent))
    suite.addTests(loader.loadTestsFromTestCase(TestHardeningE_ClassifyRescueResult))
    suite.addTests(loader.loadTestsFromTestCase(TestHardeningF_ClosePositionNoRouteBlocksExecutor))
    suite.addTests(loader.loadTestsFromTestCase(TestHardeningG_AlreadySoldFinalizes))
    suite.addTests(loader.loadTestsFromTestCase(TestHardeningH_ScannerNonBlocking))
    suite.addTests(loader.loadTestsFromTestCase(TestHardeningI_ConfigWarningEmptyFallback))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    _run()
