"""
test_v8_paper_progress_integration.py — PROGRESS-FIX PF13 regression tests
#7, #9, #10 (referenced from test_progress_capture.py's docstring).

  7.  Research (research/tracker.py, file-based) and V8-paper
      (memecoin/v8_paper.py, in-process cache) resolve the SAME
      ProgressCapture result for the same event_id — both read the one
      canonical value memecoin/progress_capture.py produced, never two
      independent measurements (PF6).
  9.  V8's gate stays fail-closed ("progress_unknown") when a measurement
      genuinely never resolves — it never falls through to treating
      missing progress as passing/failing the < 0.70 threshold.
  10. No LIVE trading path (memecoin/portfolio.py, memecoin/executor.py)
      references progress_capture at all — only the paper-only V8 gate
      does, so a slow/failed capture can never block or slow a live
      buy/sell decision.

Run with:
    python -m unittest research/tests/test_v8_paper_progress_integration.py
"""

import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import importlib

if "research.config" not in sys.modules:
    _real_config = importlib.import_module("research.config")
    _config_stub = types.ModuleType("research.config")
    for _k, _v in vars(_real_config).items():
        setattr(_config_stub, _k, _v)
    sys.modules["research.config"] = _config_stub

from memecoin import progress_capture as pc  # noqa: E402
from memecoin import v8_paper  # noqa: E402
import research.tracker as tracker  # noqa: E402


class _FakeSignal:
    def __init__(self, event_id, chain="solana", token_address="mintX",
                 dex_id="", price_usd=1.0):
        self.event_id = event_id
        self.chain = chain
        self.token_address = token_address
        self.dex_id = dex_id
        self.price_usd = price_usd


class TestResearchV8SameEvent(unittest.TestCase):
    """PF13 #7."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._snap_path = Path(self._tmpdir.name) / "progress_snapshots.jsonl"
        with pc._cache_lock:
            pc._cache.clear()
            pc._cache_order.clear()
            pc._cache_waiters.clear()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_research_and_v8_read_identical_progress_for_same_event(self):
        with patch("memecoin.progress_capture.SNAPSHOT_PATH", self._snap_path):
            cap = pc.ProgressCapture.success(
                "evShared", "mintShared", time.time() - 0.3, 40.25, "curve_account",
            )
            pc._store_result(cap)   # writes both the in-process cache AND the file

            # V8's path: in-process cache lookup (no file I/O)
            v8_cap = pc.wait_for_capture("evShared", timeout_s=1.0)

            # Research's path: reads the durable file, as a separate process would
            with patch("research.tracker._PROGRESS_SNAPSHOTS_PATH", self._snap_path):
                research_progress_dict = tracker._read_progress_snapshot(
                    "evShared", max_wait_s=0.2)

        self.assertIsNotNone(v8_cap)
        self.assertTrue(research_progress_dict)
        self.assertEqual(v8_cap.progress_at_signal,
                          research_progress_dict["progress_at_signal"])
        self.assertEqual(v8_cap.progress_source,
                          research_progress_dict["progress_source"])
        self.assertEqual(v8_cap.progress_at_signal, round(40.25 / v8_paper._GRAD_SOL, 4))


class TestV8FailClosed(unittest.TestCase):
    """PF13 #9."""

    def setUp(self):
        with pc._cache_lock:
            pc._cache.clear()
            pc._cache_order.clear()
            pc._cache_waiters.clear()

    def test_gate_returns_progress_unknown_when_capture_never_resolves(self):
        sig = _FakeSignal(event_id="evNeverResolves")
        # No ProgressCapture ever stored for this event_id — wait_for_capture
        # will time out and return None.
        with patch("memecoin.v8_paper._GATE_CAPTURE_WAIT_S", 0.05):
            passed, reason, progress = v8_paper.passes_v8_gate(sig)

        self.assertFalse(passed)
        self.assertEqual(reason, "progress_unknown")
        self.assertIsNone(progress)

    def test_gate_does_not_treat_missing_progress_as_under_threshold(self):
        """A missing measurement must never accidentally satisfy
        progress < V8_PROGRESS_MAX just because None compares oddly."""
        sig = _FakeSignal(event_id="evNeverResolves2")
        with patch("memecoin.v8_paper._GATE_CAPTURE_WAIT_S", 0.05):
            passed, reason, progress = v8_paper.passes_v8_gate(sig)
        self.assertNotEqual(reason, "ok")
        self.assertFalse(passed)


class TestNoLiveTradingPathDependsOnCapture(unittest.TestCase):
    """PF13 #10 — static/structural check: the live buy/sell paths never
    import or call anything from progress_capture, so a slow or failed
    capture can never add latency to, or block, a live decision. (V8-paper
    is paper-only by construction — memecoin/v8_paper.py explicitly does
    not share state with memecoin.portfolio.Position.)"""

    def test_portfolio_module_does_not_reference_progress_capture(self):
        import memecoin.portfolio as portfolio
        src = Path(portfolio.__file__).read_text()
        self.assertNotIn("progress_capture", src,
            "memecoin/portfolio.py (live position lifecycle) must never "
            "import or call progress_capture — that would put a paper-only "
            "measurement worker on the live trading critical path.")

    def test_executor_module_does_not_reference_progress_capture(self):
        import memecoin.executor as executor
        src = Path(executor.__file__).read_text()
        self.assertNotIn("progress_capture", src,
            "memecoin/executor.py (live buy/sell execution) must never "
            "import or call progress_capture.")


if __name__ == "__main__":
    unittest.main()
