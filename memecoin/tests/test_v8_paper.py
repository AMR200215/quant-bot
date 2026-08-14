"""
memecoin/tests/test_v8_paper.py — V8-TWIN-FIX VF7 deterministic tests,
updated for V8-REWIRE (2026-08-14).

Gate-logic tests (1-7, 10) are unchanged: passes_v8_gate() is duck-typed
on .chain/.token_address/.event_id, so it works identically whether fed a
V7 Signal-shaped object or a real memecoin.alert_event.TelegramAlertEvent
— that's the point (see VR3/VR4 note in v8_paper.py). Book tests (12-14)
are rewritten against maybe_open_from_alert()/_evaluate_alert(), which
take a TelegramAlertEvent, not a Signal — V8 no longer accepts a V7
Signal anywhere in this module. New tests cover the rewire's structural
guarantees: async dispatch never blocks the caller, V8's own transport
dedup is independent of V7's, and an unpriced-but-passing gate produces
an explicit terminal state rather than a silent drop.

Run: python -m pytest memecoin/tests/test_v8_paper.py -v
"""

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from memecoin.alert_event import TelegramAlertEvent
from memecoin.progress_capture import ProgressCapture
from memecoin.v8_paper import (
    V8_PROGRESS_MAX,
    V8PaperBook,
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


def _event(**overrides):
    base = dict(
        event_id="ev1", chain="solana",
        token_address="Mint1111111111111111111111111111111111111",
        alert_ts=time.time(), message_text="", token_symbol="TEST",
    )
    base.update(overrides)
    return TelegramAlertEvent(**base)


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
    """VF7 tests 1-7. Unaffected by V8-REWIRE: passes_v8_gate() only ever
    reads .chain/.token_address/.event_id, so a SimpleNamespace signal
    stand-in still exercises the exact same code a real
    TelegramAlertEvent would."""

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

    def test_gate_accepts_real_telegram_alert_event_not_just_signal_stand_in(self):
        """V8-REWIRE: the real call path passes a TelegramAlertEvent, not
        the SimpleNamespace stand-in used above -- prove the gate works
        against the real dataclass too, not just a mock shaped like one."""
        with patch(_PATCH_TARGET, return_value=_cap(0.50, "CURVE_ACTIVE")):
            passed, reason, progress = passes_v8_gate(_event())
        self.assertTrue(passed)
        self.assertEqual(reason, "ok")


class TestV8BookPersistenceAndIsolation(unittest.TestCase):
    """VF7 tests 12-14, rewritten against maybe_open_from_alert()/
    _evaluate_alert() (V8-REWIRE). Tests call _evaluate_alert() directly
    (the synchronous worker) rather than maybe_open_from_alert() (which
    only dispatches a thread) so results are deterministic without
    thread-timing waits -- the dispatch behavior itself is covered
    separately below."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._journal = Path(self._tmpdir.name) / "v8_journal.csv"
        self._positions = Path(self._tmpdir.name) / "v8_positions.json"
        self._paths_patch = patch("memecoin.v8_paper._paths",
                                   return_value=(self._journal, self._positions))
        self._paths_patch.start()
        self._price_patch = patch("memecoin.v8_paper._resolve_entry_price",
                                   return_value=(0.00002, "pp_tick_v8_fork"))
        self._price_patch.start()
        # Never touch the real repo's logs/watchdog/v8_rewire_deploy_ts.txt
        # from a test run -- _current_era() is covered on its own below.
        self._era_patch = patch("memecoin.v8_paper._current_era", return_value="TEST_ERA")
        self._era_patch.start()

    def tearDown(self):
        self._era_patch.stop()
        self._price_patch.stop()
        self._paths_patch.stop()
        self._tmpdir.cleanup()

    def test_12_open_creates_and_persists_position(self):
        """VF7 #12: V8 open creates/persists position state."""
        book = V8PaperBook()
        with patch(_PATCH_TARGET, return_value=_cap(0.40, "CURVE_ACTIVE")):
            book._evaluate_alert(_event(token_address="MintPersist111111111111111111111111111",
                                         event_id="ev_persist"))
        self.assertTrue(self._positions.exists())
        data = json.loads(self._positions.read_text())
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["token_address"], "MintPersist111111111111111111111111111")
        self.assertEqual(data[0]["status"], "open")
        self.assertAlmostEqual(data[0]["progress_at_signal"], 0.40)
        self.assertEqual(data[0]["entry_price"], 0.00002)
        self.assertEqual(data[0]["entry_source"], "pp_tick_v8_fork")

        # A second book instance loading from the same paths sees it too —
        # proves this is real persistence, not just in-memory state.
        book2 = V8PaperBook()
        self.assertEqual(len(book2.open_positions()), 1)

    def test_13_close_writes_journal(self):
        """VF7 #13: V8 close writes journal."""
        book = V8PaperBook()
        with patch(_PATCH_TARGET, return_value=_cap(0.40, "CURVE_ACTIVE")):
            book._evaluate_alert(_event(token_address="MintClose1111111111111111111111111111",
                                         event_id="ev_close"))
        self.assertFalse(self._journal.exists())   # not written until close
        pos_id = next(iter(book._positions))
        book._close(pos_id, price=0.00003, reason="test_close")
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
            book._evaluate_alert(_event(token_address="MintIsolation11111111111111111111111",
                                         event_id="ev_isolation"))
        after = list(scanner.portfolio._positions) if hasattr(scanner.portfolio, "_positions") else None
        self.assertEqual(before, after,
            "V8's book must never mutate memecoin.scanner.portfolio (V7's live/paper book)")
        # Structural check: v8_paper.py's own source never *imports or
        # calls into* memecoin.portfolio / scanner.portfolio, and never
        # imports memecoin.scanner.Signal (V8-REWIRE: V8 must not even be
        # able to construct or type-depend on a V7 Signal).
        src = Path(__import__("memecoin.v8_paper", fromlist=["x"]).__file__).read_text()
        self.assertNotIn("import memecoin.portfolio", src)
        self.assertNotIn("from memecoin.portfolio", src)
        self.assertNotIn("from memecoin import portfolio", src)
        self.assertNotIn("scanner.portfolio.", src)
        self.assertNotIn("from memecoin.scanner import", src)
        self.assertNotIn("memecoin.scanner.Signal", src)

    def test_gate_fail_produces_no_position_and_no_journal_row(self):
        book = V8PaperBook()
        with patch(_PATCH_TARGET, return_value=_cap(0.90, "CURVE_ACTIVE")):  # over threshold
            book._evaluate_alert(_event(token_address="MintReject111111111111111111111111111",
                                         event_id="ev_reject"))
        self.assertEqual(len(book._positions), 0)
        self.assertFalse(self._journal.exists())

    def test_unpriced_pass_opens_no_position_but_is_a_distinct_outcome(self):
        """V8-REWIRE VR8: gate passes but no independent price arrived —
        must not open a position (nothing to price PnL against), and must
        not be indistinguishable from a gate rejection at the telemetry
        layer (covered by the emit() call inside _evaluate_alert; here we
        just confirm the book-state behavior: no position, no crash)."""
        book = V8PaperBook()
        with patch("memecoin.v8_paper._resolve_entry_price", return_value=(0.0, "pp_unpriced")):
            with patch(_PATCH_TARGET, return_value=_cap(0.40, "CURVE_ACTIVE")):
                book._evaluate_alert(_event(token_address="MintUnpriced11111111111111111111111",
                                             event_id="ev_unpriced"))
        self.assertEqual(len(book._positions), 0)

    def test_already_open_position_blocks_second_open_same_token(self):
        book = V8PaperBook()
        with patch(_PATCH_TARGET, return_value=_cap(0.40, "CURVE_ACTIVE")):
            book._evaluate_alert(_event(token_address="MintDup11111111111111111111111111111",
                                         event_id="ev_dup_1"))
            book._evaluate_alert(_event(token_address="MintDup11111111111111111111111111111",
                                         event_id="ev_dup_2"))
        self.assertEqual(len(book._positions), 1)


class TestV8TransportDedupIndependentOfV7(unittest.TestCase):
    """V8-REWIRE VR5/VR6: V8's own dedup must never depend on
    memecoin.scanner._is_duplicate() / _seen / _traded_today / V7's
    portfolio.open_positions() — none of those are imported or read by
    v8_paper.py at all."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._paths_patch = patch(
            "memecoin.v8_paper._paths",
            return_value=(Path(self._tmpdir.name) / "j.csv", Path(self._tmpdir.name) / "p.json"),
        )
        self._paths_patch.start()
        self._price_patch = patch("memecoin.v8_paper._resolve_entry_price",
                                   return_value=(0.00002, "pp_tick_v8_fork"))
        self._price_patch.start()
        self._era_patch = patch("memecoin.v8_paper._current_era", return_value="TEST_ERA")
        self._era_patch.start()

    def tearDown(self):
        self._era_patch.stop()
        self._price_patch.stop()
        self._paths_patch.stop()
        self._tmpdir.cleanup()

    def test_same_event_id_evaluated_twice_only_opens_once(self):
        import memecoin.v8_paper as v8_paper
        book = v8_paper.V8PaperBook()
        event = _event(token_address="MintTransport1111111111111111111111", event_id="ev_transport_dup")
        with patch(_PATCH_TARGET, return_value=_cap(0.40, "CURVE_ACTIVE")):
            book._evaluate_alert(event)
            book._evaluate_alert(event)   # exact same event_id, simulating a double-invocation
        self.assertEqual(len(book._positions), 1)

    def test_v8_paper_module_never_imports_scanner_dedup_state(self):
        # AST-based, not a substring scan -- v8_paper.py's own prose
        # comments legitimately mention "memecoin.scanner._on_telegram_
        # signal()" and similar in plain English when explaining what V8
        # must NOT depend on, which a naive substring check false-
        # positives on. Only real import statements count here.
        import ast
        src = Path(__import__("memecoin.v8_paper", fromlist=["x"]).__file__).read_text()
        tree = ast.parse(src)
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
        for forbidden in ("memecoin.scanner", "memecoin.portfolio"):
            self.assertFalse(
                any(m == forbidden or m.startswith(forbidden + ".") for m in imported_modules),
                f"v8_paper.py must never import {forbidden} (found in: {sorted(imported_modules)})",
            )


class TestEraBootstrapping(unittest.TestCase):
    """V8-REWIRE VR12/VR13: the deploy-cutover stamp is self-bootstrapping,
    not a hand-set constant. Each test gets a fresh temp stamp path AND a
    cleared in-process cache, since _independent_validation_start() caches
    on its own function attribute."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._stamp = Path(self._tmpdir.name) / "v8_rewire_deploy_ts.txt"
        self._path_patch = patch("memecoin.v8_paper._deploy_stamp_path", return_value=self._stamp)
        self._path_patch.start()
        import memecoin.v8_paper as v8_paper
        self._v8_paper = v8_paper
        if hasattr(v8_paper._independent_validation_start, "_cached"):
            del v8_paper._independent_validation_start._cached

    def tearDown(self):
        if hasattr(self._v8_paper._independent_validation_start, "_cached"):
            del self._v8_paper._independent_validation_start._cached
        self._path_patch.stop()
        self._tmpdir.cleanup()

    def test_first_call_creates_stamp_file_with_real_timestamp(self):
        before = time.time()
        ts = self._v8_paper._independent_validation_start()
        after = time.time()
        self.assertTrue(self._stamp.exists())
        self.assertGreaterEqual(ts, before)
        self.assertLessEqual(ts, after)

    def test_second_call_reuses_the_same_stamp_not_a_fresh_timestamp(self):
        ts1 = self._v8_paper._independent_validation_start()
        del self._v8_paper._independent_validation_start._cached   # force a real re-read from disk
        time.sleep(0.01)
        ts2 = self._v8_paper._independent_validation_start()
        self.assertEqual(ts1, ts2)

    def test_era_before_and_after_cutover(self):
        stamp_ts = time.time() + 100   # cutover in the "future" relative to now
        self._stamp.write_text(str(stamp_ts))
        with patch("memecoin.v8_paper.time.time", return_value=stamp_ts - 1):
            self.assertEqual(self._v8_paper._current_era(), self._v8_paper.V8_ERA_PRE_REWIRE)
        del self._v8_paper._independent_validation_start._cached
        with patch("memecoin.v8_paper.time.time", return_value=stamp_ts + 1):
            self.assertEqual(self._v8_paper._current_era(), self._v8_paper.V8_ERA_INDEPENDENT)


class TestMaybeOpenFromAlertDispatchesAsync(unittest.TestCase):
    """V8-REWIRE VR1: maybe_open_from_alert() must return immediately —
    the entry-price wait (up to _PRICE_WAIT_S) must never block the
    caller, which is on V7's synchronous, latency-budgeted signal path."""

    def test_returns_before_evaluate_alert_completes(self):
        import threading
        import memecoin.v8_paper as v8_paper

        release = threading.Event()
        started = threading.Event()

        def _slow_evaluate(self, event):
            started.set()
            release.wait(timeout=2.0)

        book = v8_paper.V8PaperBook.__new__(v8_paper.V8PaperBook)
        book._positions = {}
        book._lock = threading.Lock()

        with patch.object(v8_paper.V8PaperBook, "_evaluate_alert", _slow_evaluate):
            t0 = time.time()
            book.maybe_open_from_alert(_event())
            elapsed = time.time() - t0
        self.assertLess(elapsed, 0.5, "maybe_open_from_alert must dispatch and return immediately")
        self.assertTrue(started.wait(timeout=1.0), "the worker thread must actually run")
        release.set()


if __name__ == "__main__":
    unittest.main()
