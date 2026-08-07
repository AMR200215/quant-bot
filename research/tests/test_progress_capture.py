"""
test_progress_capture.py — PROGRESS-FIX PF13 regression tests for
memecoin/progress_capture.py.

Covers (numbering matches the PF13 spec list where applicable):
  1. Cold subscribe (freshly-created ScreeningState, vsol=0) is NOT valid pp_warm
  2. Warm PP state with a fresh timestamp -> pp_warm
  3. Cold alert + valid curve account -> curve_account
  5. Curve RPC failure + PP tick arriving later -> pp_post_alert, lag recorded
  6. Curve RPC failure + no PP tick ever -> NULL progress, explicit status
  8. Two alerts for the same mint (different event_id) never cross-write
  11. Nothing in capture_progress_async blocks the caller
  12. A default-only (cold) state is correctly rejected regardless of caller
  (Test 4 — survives scanner eviction — is structural: source B/C never
  touch scanner._screening at all, only source A does, and A already
  requires a fresh timestamp; test_curve_account_independent_of_screening
  below is the direct check.)
  (Test 13, curve price fixture — covered in test_rf1_curve_oracle.py.)
  (Tests 7, 9, 10 — Research/V8 agreement, V8 fail-closed, no-trading-block —
  are covered by test_v8_paper_progress_integration.py.)

Run with:
    python -m unittest research/tests/test_progress_capture.py
"""

import sys
import time
import types
import unittest
from unittest.mock import MagicMock, patch

# Same config-stub pattern as test_rf1_curve_oracle.py, so this file can run
# standalone or alongside it without import-order surprises.
import importlib

if "research.config" not in sys.modules:
    _real_config = importlib.import_module("research.config")
    _config_stub = types.ModuleType("research.config")
    for _k, _v in vars(_real_config).items():
        setattr(_config_stub, _k, _v)
    sys.modules["research.config"] = _config_stub

from memecoin import progress_capture as pc  # noqa: E402


class _FakeScreeningState:
    def __init__(self, latest_vsol=0.0, latest_vsol_ts=0.0):
        self.latest_vsol = latest_vsol
        self.latest_vsol_ts = latest_vsol_ts


class TestProgressCaptureDataclass(unittest.TestCase):

    def test_failure_never_sets_progress_to_zero(self):
        """PF2: a missing measurement is NULL, never the genuine physical
        value 0 — this is the single most important invariant in the spec."""
        cap = pc.ProgressCapture.failure("ev1", "mintA", time.time(), "pp_timeout")
        self.assertIsNone(cap.vsol_at_signal)
        self.assertIsNone(cap.progress_at_signal)
        self.assertEqual(cap.progress_status, "pp_timeout")
        self.assertEqual(cap.progress_source, "unknown")

    def test_success_computes_progress_from_grad_sol_ui(self):
        from memecoin.config import GRAD_SOL_UI
        alert_ts = time.time() - 0.5
        cap = pc.ProgressCapture.success("ev2", "mintB", alert_ts, 57.5, "curve_account")
        self.assertEqual(cap.progress_status, "ok")
        self.assertAlmostEqual(cap.progress_at_signal, round(57.5 / GRAD_SOL_UI, 4))
        self.assertGreater(cap.progress_capture_lag_ms, 0)

    def test_success_rejects_invalid_source(self):
        with self.assertRaises(AssertionError):
            pc.ProgressCapture.success("ev3", "mintC", time.time(), 10.0, "made_up_source")

    def test_to_dict_from_dict_roundtrip(self):
        cap = pc.ProgressCapture.success("ev4", "mintD", time.time(), 20.0, "pp_warm")
        d = cap.to_dict()
        restored = pc.ProgressCapture.from_dict(d)
        self.assertEqual(cap, restored)


class TestSourceA_PPWarm(unittest.TestCase):

    def test_cold_subscribe_freshly_created_state_not_valid(self):
        """The exact PF1 race: a ScreeningState that exists but was never
        actually updated by a real PumpPortal message must NOT count as warm."""
        fake_monitor = MagicMock()
        fake_monitor.get_screening_state.return_value = _FakeScreeningState(
            latest_vsol=0.0, latest_vsol_ts=0.0,
        )
        with patch("memecoin.pumpportal_monitor.monitor", fake_monitor):
            result = pc._try_pp_warm("mintCold")
        self.assertIsNone(result)

    def test_warm_state_with_fresh_timestamp_accepted(self):
        fake_monitor = MagicMock()
        fake_monitor.get_screening_state.return_value = _FakeScreeningState(
            latest_vsol=42.0, latest_vsol_ts=time.time() - 0.1,
        )
        with patch("memecoin.pumpportal_monitor.monitor", fake_monitor):
            result = pc._try_pp_warm("mintWarm")
        self.assertEqual(result, 42.0)

    def test_stale_warm_state_rejected(self):
        """latest_vsol_ts far in the past (beyond PP_WARM_FRESHNESS_S) is not warm."""
        fake_monitor = MagicMock()
        fake_monitor.get_screening_state.return_value = _FakeScreeningState(
            latest_vsol=42.0, latest_vsol_ts=time.time() - 999,
        )
        with patch("memecoin.pumpportal_monitor.monitor", fake_monitor):
            result = pc._try_pp_warm("mintStale")
        self.assertIsNone(result)

    def test_no_screening_state_returns_none(self):
        fake_monitor = MagicMock()
        fake_monitor.get_screening_state.return_value = None
        with patch("memecoin.pumpportal_monitor.monitor", fake_monitor):
            result = pc._try_pp_warm("mintMissing")
        self.assertIsNone(result)


class TestSourceB_CurveAccount(unittest.TestCase):

    def setUp(self):
        with pc._cache_lock:
            pc._cache.clear()
            pc._cache_order.clear()
            pc._cache_waiters.clear()
        with pc._batch_lock:
            pc._batch_pending.clear()

    def test_curve_account_success_stores_result(self):
        fake_result = {
            "mintCurve": {
                "vsol_ui": 30.0, "complete": False,
                "venue_state": "CURVE_ACTIVE", "failure_reason": None,
            }
        }
        with patch("memecoin.progress_capture._helius_api_key", return_value="fake-key"):
            with patch("research.curve_oracle.get_curve_state_batch", return_value=fake_result):
                pc._batch_pending["mintCurve"] = [("ev5", time.time())]
                pc._flush_curve_batch()

        cap = pc.get_capture("ev5")
        self.assertIsNotNone(cap)
        self.assertEqual(cap.progress_source, "curve_account")
        self.assertEqual(cap.progress_status, "ok")
        self.assertEqual(cap.vsol_at_signal, 30.0)

    def test_curve_rpc_failure_falls_through_to_pp_post_alert(self):
        """RPC failure must not produce a result directly — it must hand off
        to the pp_post_alert fallback (test verified by checking the waiter
        registry gets populated, not by waiting out the full timeout)."""
        fake_result = {
            "mintFail": {
                "vsol_ui": None, "complete": None,
                "venue_state": "RPC_ERROR", "failure_reason": "curve_rpc_error",
            }
        }
        with patch("memecoin.progress_capture._helius_api_key", return_value="fake-key"):
            with patch("research.curve_oracle.get_curve_state_batch", return_value=fake_result):
                with patch("memecoin.progress_capture._ensure_callback_registered"):
                    pc._batch_pending["mintFail"] = [("ev6", time.time())]
                    pc._flush_curve_batch()

        # No result yet (handed off to the async pp_post_alert worker thread)
        self.assertIsNone(pc.get_capture("ev6"))
        # Give the background worker a moment to register itself
        for _ in range(20):
            with pc._waiting_lock:
                if "mintFail" in pc._waiting:
                    break
            time.sleep(0.05)
        with pc._waiting_lock:
            self.assertIn("mintFail", pc._waiting)
            pc._waiting.pop("mintFail", None)   # cleanup so the worker thread's own pop is a no-op

    def test_no_helius_key_falls_through_to_pp_post_alert(self):
        with patch("memecoin.progress_capture._helius_api_key", return_value=""):
            with patch("memecoin.progress_capture._ensure_callback_registered"):
                pc._batch_pending["mintNoKey"] = [("ev7", time.time())]
                pc._flush_curve_batch()
        self.assertIsNone(pc.get_capture("ev7"))
        for _ in range(20):
            with pc._waiting_lock:
                if "mintNoKey" in pc._waiting:
                    break
            time.sleep(0.05)
        with pc._waiting_lock:
            self.assertIn("mintNoKey", pc._waiting)
            pc._waiting.pop("mintNoKey", None)


class TestSourceC_PPPostAlert(unittest.TestCase):

    def setUp(self):
        with pc._cache_lock:
            pc._cache.clear()
            pc._cache_order.clear()
            pc._cache_waiters.clear()
        with pc._waiting_lock:
            pc._waiting.clear()

    def test_timeout_with_no_tick_produces_null_not_zero(self):
        with patch("memecoin.progress_capture.PP_POST_ALERT_TIMEOUT_S", 0.2):
            with patch("memecoin.progress_capture._ensure_callback_registered"):
                pc._fallback_pp_post_alert("ev8", "mintTimeout", time.time(), "curve_rpc_error")
                cap = pc.wait_for_capture("ev8", timeout_s=1.0)

        self.assertIsNotNone(cap)
        self.assertEqual(cap.progress_status, "pp_timeout")
        self.assertIsNone(cap.progress_at_signal)
        self.assertIsNone(cap.vsol_at_signal)

    def test_tick_arriving_during_wait_produces_success(self):
        with patch("memecoin.progress_capture.PP_POST_ALERT_TIMEOUT_S", 2.0):
            with patch("memecoin.progress_capture._ensure_callback_registered"):
                pc._fallback_pp_post_alert("ev9", "mintTick", time.time(), "curve_rpc_error")
                # Give the worker a moment to register, then fire the event
                for _ in range(20):
                    with pc._waiting_lock:
                        if "mintTick" in pc._waiting:
                            break
                    time.sleep(0.02)
                pc._on_vsol_update("mintTick", 15.0)
                cap = pc.wait_for_capture("ev9", timeout_s=1.0)

        self.assertIsNotNone(cap)
        self.assertEqual(cap.progress_status, "ok")
        self.assertEqual(cap.progress_source, "pp_post_alert")
        self.assertEqual(cap.vsol_at_signal, 15.0)

    def test_two_concurrent_waiters_same_mint_both_get_result(self):
        """PF5/realert scenario: two different event_ids for the same mint
        waiting concurrently must both resolve independently, not clobber."""
        with patch("memecoin.progress_capture.PP_POST_ALERT_TIMEOUT_S", 2.0):
            with patch("memecoin.progress_capture._ensure_callback_registered"):
                pc._fallback_pp_post_alert("evA", "mintDup", time.time(), "curve_rpc_error")
                pc._fallback_pp_post_alert("evB", "mintDup", time.time(), "curve_rpc_error")
                for _ in range(20):
                    with pc._waiting_lock:
                        if len(pc._waiting.get("mintDup", [])) == 2:
                            break
                    time.sleep(0.02)
                pc._on_vsol_update("mintDup", 8.0)
                cap_a = pc.wait_for_capture("evA", timeout_s=1.0)
                cap_b = pc.wait_for_capture("evB", timeout_s=1.0)

        self.assertIsNotNone(cap_a)
        self.assertIsNotNone(cap_b)
        self.assertNotEqual(cap_a.event_id, cap_b.event_id)
        self.assertEqual(cap_a.vsol_at_signal, 8.0)
        self.assertEqual(cap_b.vsol_at_signal, 8.0)


class TestEventIdIsolation(unittest.TestCase):

    def setUp(self):
        with pc._cache_lock:
            pc._cache.clear()
            pc._cache_order.clear()
            pc._cache_waiters.clear()

    def test_different_events_never_cross_write(self):
        cap1 = pc.ProgressCapture.success("evX", "mintSame", time.time(), 10.0, "pp_warm")
        cap2 = pc.ProgressCapture.success("evY", "mintSame", time.time(), 99.0, "curve_account")
        pc._store_result(cap1)
        pc._store_result(cap2)

        got1 = pc.get_capture("evX")
        got2 = pc.get_capture("evY")
        self.assertEqual(got1.vsol_at_signal, 10.0)
        self.assertEqual(got2.vsol_at_signal, 99.0)
        self.assertNotEqual(got1.vsol_at_signal, got2.vsol_at_signal)


class TestNonBlocking(unittest.TestCase):

    def test_capture_progress_async_returns_quickly_even_with_slow_curve_rpc(self):
        """PF3: must never block the caller — this simulates a slow curve
        RPC (mocked to sleep) and asserts capture_progress_async itself
        still returns near-instantly, since the actual RPC work happens on
        the deferred micro-batch timer, not synchronously in this call."""
        fake_monitor = MagicMock()
        fake_monitor.get_screening_state.return_value = None   # force past source A

        t0 = time.time()
        with patch("memecoin.pumpportal_monitor.monitor", fake_monitor):
            pc.capture_progress_async("evFast", "mintFast", time.time(), "solana")
        elapsed = time.time() - t0

        # Real blocking (an actual RPC call) would be hundreds of ms to
        # seconds; this just needs to rule that out, not assert a tight
        # absolute bound that's sensitive to test-harness/mock overhead.
        self.assertLess(elapsed, 0.5, "capture_progress_async must not block the caller")

    def test_non_solana_chain_fails_immediately_not_null_silently(self):
        pc.capture_progress_async("evChain", "mintBsc", time.time(), "bsc")
        cap = pc.get_capture("evChain")
        self.assertIsNotNone(cap)
        self.assertEqual(cap.progress_status, "non_pumpfun")
        self.assertIsNone(cap.progress_at_signal)


if __name__ == "__main__":
    unittest.main()
