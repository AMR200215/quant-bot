"""
memecoin/tests/test_v8_paper.py — V8-TWIN-FIX VF7 deterministic tests.

Rewritten 2026-08-11: the previous version tested a pre-PROGRESS-FIX
implementation of compute_progress_at_signal() (mocked
pumpportal_monitor.monitor.get_screening_state directly). PF6
(2026-08-08) rewrote it to read the shared ProgressCapture cache
instead, and this file was never updated to match — 4 of its tests had
been silently failing ever since (confirmed by running it before this
rewrite). Replaced with tests against the real, current gate logic:
progress_at_signal < V8_PROGRESS_MAX AND venue_state_at_signal ==
CURVE_ACTIVE — see docs/RECEIPTS.md's "V8-TWIN-FIX" section for why
dex_id was never the right signal (DexScreener indexes pump.fun
bonding-curve tokens as dex_id="pumpfun" long before graduation).

Covers VF7 tests 1-7 (gate logic) and 12-14 (position persistence,
journal-on-close, V7/live isolation). Tests 8-9 (dedup) are in
test_scanner_v8_dedup.py; test 11 (executor exception surfacing) is in
test_telegram_executor_observability.py — both need their own module
scope.

Run: python -m pytest memecoin/tests/test_v8_paper.py -v
"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from memecoin.progress_capture import ProgressCapture
from memecoin.v8_paper import (
    V8_PROGRESS_MAX,
    V8PaperBook,
    _new_position,
    passes_v8_gate,
)


def _signal(**overrides):
    base = dict(
        id="sig1", chain="solana", token_address="Mint1111111111111111111111111111111111111",
        token_symbol="TEST", signal_type="social_alert", strength="strong",
        price_usd=0.00001, dex_id="", _price_pp=0.0, event_id="ev1",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _cap(progress, venue_state, status="ok", source="curve_account"):
    return ProgressCapture(
        event_id="ev1", token_address="MintX", alert_ts=time.time(),
        vsol_at_signal=(progress * 115.0) if progress is not None else None,
        progress_at_signal=progress,
        progress_source=source,
        progress_observed_at=time.time(),
        progress_capture_lag_ms=400.0,
        progress_status=status,
        venue_state_at_signal=venue_state,
    )


# _get_capture_for_gate does `from memecoin.progress_capture import
# wait_for_capture` as a LOCAL import inside the function body, so
# patching the source (memecoin.progress_capture.wait_for_capture) is
# what actually takes effect — patching memecoin.v8_paper.wait_for_capture
# would silently do nothing, since that name never exists in v8_paper's
# module namespace.
_PATCH_TARGET = "memecoin.progress_capture.wait_for_capture"


class TestPassesV8GateVenueState(unittest.TestCase):
    """VF7 tests 1-7."""

    def test_1_low_progress_dex_id_pumpfun_curve_active_passes(self):
        with patch(_PATCH_TARGET, return_value=_cap(0.50, "CURVE_ACTIVE")):
            passed, reason, progress = passes_v8_gate(_signal(dex_id="pumpfun"))
        self.assertTrue(passed)
        self.assertEqual(reason, "ok")
        self.assertEqual(progress, 0.50)

    def test_2_low_progress_no_dex_id_curve_active_passes(self):
        with patch(_PATCH_TARGET, return_value=_cap(0.50, "CURVE_ACTIVE")):
            passed, reason, progress = passes_v8_gate(_signal(dex_id=""))
        self.assertTrue(passed)

    def test_3_low_progress_graduated_venue_rejects(self):
        with patch(_PATCH_TARGET, return_value=_cap(0.50, "GRADUATED")):
            passed, reason, progress = passes_v8_gate(_signal(dex_id="pumpswap"))
        self.assertFalse(passed)
        self.assertIn("venue_state", reason)
        self.assertIn("GRADUATED", reason)

    def test_4_dex_active_venue_rejects(self):
        with patch(_PATCH_TARGET, return_value=_cap(0.50, "DEX_ACTIVE")):
            passed, reason, progress = passes_v8_gate(_signal())
        self.assertFalse(passed)
        self.assertIn("DEX_ACTIVE", reason)

    def test_5_unknown_venue_fails_closed(self):
        with patch(_PATCH_TARGET, return_value=_cap(0.50, "UNKNOWN")):
            passed, reason, progress = passes_v8_gate(_signal())
        self.assertFalse(passed)
        self.assertIn("UNKNOWN", reason)

    def test_6_progress_over_threshold_rejects(self):
        with patch(_PATCH_TARGET, return_value=_cap(0.80, "CURVE_ACTIVE")):
            passed, reason, progress = passes_v8_gate(_signal())
        self.assertFalse(passed)
        self.assertIn("over", reason)
        self.assertLess(V8_PROGRESS_MAX, 1.0)   # sanity: threshold unchanged (0.70)
        self.assertEqual(V8_PROGRESS_MAX, 0.70)

    def test_7_no_capture_rejects_progress_unknown(self):
        with patch(_PATCH_TARGET, return_value=None):
            passed, reason, progress = passes_v8_gate(_signal())
        self.assertFalse(passed)
        self.assertEqual(reason, "progress_unknown")
        self.assertIsNone(progress)

    def test_10_pumpportal_screening_signal_dex_id_pumpfun_can_pass(self):
        """VF7 #10: the gate is source-agnostic — a signal built via the
        PumpPortal-native screening path (memecoin/scanner.py's
        _fire_screening_entry(), which hardcodes dex_id="pumpfun") must
        pass exactly like a Telegram-sourced one does in test #1, since
        both go through the same passes_v8_gate()."""
        pp_native_signal = _signal(
            signal_type="pumpportal_screen", dex_id="pumpfun", token_cohort="pumpfun_stream",
        )
        with patch(_PATCH_TARGET, return_value=_cap(0.30, "CURVE_ACTIVE")):
            passed, reason, progress = passes_v8_gate(pp_native_signal)
        self.assertTrue(passed)
        self.assertEqual(reason, "ok")


class TestV8BookPersistenceAndIsolation(unittest.TestCase):
    """VF7 tests 12-14."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._journal = Path(self._tmpdir.name) / "v8_journal.csv"
        self._positions = Path(self._tmpdir.name) / "v8_positions.json"
        self._paths_patch = patch("memecoin.v8_paper._paths",
                                   return_value=(self._journal, self._positions))
        self._paths_patch.start()

    def tearDown(self):
        self._paths_patch.stop()
        self._tmpdir.cleanup()

    def test_12_open_creates_and_persists_position(self):
        """VF7 #12: V8 open creates/persists position state."""
        book = V8PaperBook()
        with patch(_PATCH_TARGET, return_value=_cap(0.40, "CURVE_ACTIVE")):
            book.maybe_open(_signal(token_address="MintPersist111111111111111111111111111"))
        self.assertTrue(self._positions.exists())
        data = json.loads(self._positions.read_text())
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["token_address"], "MintPersist111111111111111111111111111")
        self.assertEqual(data[0]["status"], "open")
        self.assertAlmostEqual(data[0]["progress_at_signal"], 0.40)

        # A second book instance loading from the same paths sees it too —
        # proves this is real persistence, not just in-memory state.
        book2 = V8PaperBook()
        self.assertEqual(len(book2.open_positions()), 1)

    def test_13_close_writes_journal(self):
        """VF7 #13: V8 close writes journal."""
        book = V8PaperBook()
        with patch(_PATCH_TARGET, return_value=_cap(0.40, "CURVE_ACTIVE")):
            book.maybe_open(_signal(token_address="MintClose1111111111111111111111111111"))
        self.assertFalse(self._journal.exists())   # not written until close
        pos_id = next(iter(book._positions))
        book._close(pos_id, price=0.00002, reason="test_close")
        self.assertTrue(self._journal.exists())
        content = self._journal.read_text()
        self.assertIn("test_close", content)
        self.assertIn("MintClose1111111111111111111111111111", content)

    def test_14_no_v8_action_affects_v7_portfolio(self):
        """VF7 #14: no V8 action can affect V7/live Portfolio state."""
        import memecoin.scanner as scanner
        before = list(scanner.portfolio._positions) if hasattr(scanner.portfolio, "_positions") else None
        book = V8PaperBook()
        with patch(_PATCH_TARGET, return_value=_cap(0.40, "CURVE_ACTIVE")):
            book.maybe_open(_signal(token_address="MintIsolation11111111111111111111111"))
        after = list(scanner.portfolio._positions) if hasattr(scanner.portfolio, "_positions") else None
        self.assertEqual(before, after,
            "V8's maybe_open() must never mutate memecoin.scanner.portfolio (V7's live/paper book)")
        # Structural check: v8_paper.py's own source never *imports or
        # calls into* memecoin.portfolio / scanner.portfolio (docstring
        # mentions of the concept are fine — only real usage matters).
        src = Path(__import__("memecoin.v8_paper", fromlist=["x"]).__file__).read_text()
        self.assertNotIn("import memecoin.portfolio", src)
        self.assertNotIn("from memecoin.portfolio", src)
        self.assertNotIn("from memecoin import portfolio", src)
        self.assertNotIn("scanner.portfolio.", src)


if __name__ == "__main__":
    unittest.main()
