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
    """VF7 tests 1-7 (test 6 rewritten 2026-09-10 for the V8-P0 gate
    switch -- see its own docstring). Unaffected by V8-REWIRE itself:
    passes_v8_gate() only ever reads .chain/.token_address/.event_id, so
    a SimpleNamespace signal stand-in still exercises the exact same code
    a real TelegramAlertEvent would."""

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

    def test_6_high_progress_still_passes_v8p0_has_no_progress_condition(self):
        """SWITCHED 2026-09-10: V8-P0 (research/v8_candidate_registry.py)
        has no progress_at_signal condition at all -- only venue_state_
        at_signal == CURVE_ACTIVE gates entry. This is the regression
        test that would catch a progress gate silently creeping back in.
        High progress (0.80, close to graduation) must still PASS as long
        as the venue is CURVE_ACTIVE."""
        with patch(_PATCH_TARGET, return_value=_cap(0.80, "CURVE_ACTIVE")):
            passed, reason, progress = passes_v8_gate(_signal())
        self.assertTrue(passed)
        self.assertEqual(reason, "ok")
        self.assertEqual(progress, 0.80)

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

    def test_open_durably_subscribes_for_ongoing_price_updates(self):
        """2026-09-17: root-caused live that 173/179 open positions never
        received a single price update after entry — v8_paper only ever
        touched pumpportal_monitor's bounded, LRU-evictable screening
        subscription, never the durable `subscribe()` set V7's
        portfolio.py uses for exactly this reason. A newly-opened position
        must get a durable subscription so the monitor loop can actually
        update it."""
        fake_monitor = SimpleNamespace(subscribe=lambda mints: None)
        with patch("memecoin.pumpportal_monitor.monitor", fake_monitor), \
             patch.object(fake_monitor, "subscribe") as mock_subscribe, \
             patch(_PATCH_TARGET, return_value=_cap(0.40, "CURVE_ACTIVE")):
            book = V8PaperBook()
            book._evaluate_alert(_event(
                token_address="MintSub1111111111111111111111111111111",
                event_id="ev_sub"))
        mock_subscribe.assert_called_once_with({"MintSub1111111111111111111111111111111"})

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
        # SWITCHED 2026-09-10: V8-P0 has no progress condition, so a high
        # progress value alone no longer fails the gate (see test_6) --
        # GRADUATED venue is the genuinely gate-failing scenario now.
        book = V8PaperBook()
        with patch(_PATCH_TARGET, return_value=_cap(0.90, "GRADUATED")):
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


class TestCurveFallbackPricing(unittest.TestCase):
    """2026-09-12: _resolve_entry_price had no fallback at all when no
    PumpPortal tick arrived in budget (root-caused live: 40/73 real
    signals over 41h ended pp_unpriced). These test the new on-chain
    curve-read fallback and its SOL/USD freshness tracking."""

    def setUp(self):
        import memecoin.v8_paper as v8_paper
        self._v8_paper = v8_paper
        self._orig_cache = dict(v8_paper._sol_price_cache)
        v8_paper._sol_price_cache["price"] = 0.0
        v8_paper._sol_price_cache["ts"] = 0.0

    def tearDown(self):
        self._v8_paper._sol_price_cache.update(self._orig_cache)

    def test_sol_price_success_updates_cache_with_fresh_age(self):
        v8p = self._v8_paper
        resp = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"outAmount": "170000000"},
        )
        with patch("requests.get", return_value=resp):
            price, age = v8p._fresh_sol_price_usd()
        self.assertEqual(price, 170.0)
        self.assertEqual(age, 0.0)
        self.assertEqual(v8p._sol_price_cache["price"], 170.0)

    def test_sol_price_failure_does_not_falsely_refresh_age(self):
        """The bug this guards against: executor._sol_price_usd() bumps
        its own timestamp even on failure, so a staleness check against
        it can never detect sustained fetch failure. This cache must not
        repeat that mistake — age must reflect real elapsed time since
        the last genuine success, staying inf if there never was one."""
        v8p = self._v8_paper
        with patch("requests.get", side_effect=RuntimeError("429")):
            price, age = v8p._fresh_sol_price_usd()
        self.assertEqual(price, 0.0)
        self.assertEqual(age, float("inf"))

    def test_sol_price_falls_back_to_coingecko_when_jupiter_fails(self):
        """2026-09-14: found live that Jupiter alone wasn't enough — it
        was 429ing at the exact moment this fallback needed a price
        (same congestion as executor.py's own SOL price fetches).
        CoinGecko is a different provider/rate-limit pool."""
        v8p = self._v8_paper
        jup_fail = RuntimeError("429 from Jupiter")
        cg_resp = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"solana": {"usd": 172.5}},
        )
        with patch("requests.get", side_effect=[jup_fail, cg_resp]):
            price, age = v8p._fresh_sol_price_usd()
        self.assertEqual(price, 172.5)
        self.assertEqual(age, 0.0)

    def test_curve_fallback_no_helius_key(self):
        v8p = self._v8_paper
        with patch.dict("os.environ", {"HELIUS_API_KEY": ""}):
            price, source = v8p._curve_fallback_price("Mint111")
        self.assertEqual(price, 0.0)
        self.assertEqual(source, "curve_fallback_no_key")

    def test_curve_fallback_success(self):
        v8p = self._v8_paper
        with patch.dict("os.environ", {"HELIUS_API_KEY": "fake_key"}), \
             patch.object(v8p, "_fresh_sol_price_usd", return_value=(170.0, 0.0)), \
             patch("research.curve_oracle.get_curve_prices_batch",
                   return_value={"Mint111": {"price_usd": 0.0000123, "failure_reason": None}}):
            price, source = v8p._curve_fallback_price("Mint111")
        self.assertEqual(price, 0.0000123)
        self.assertEqual(source, "curve_fallback")

    def test_curve_fallback_reports_specific_failure_reason(self):
        v8p = self._v8_paper
        with patch.dict("os.environ", {"HELIUS_API_KEY": "fake_key"}), \
             patch.object(v8p, "_fresh_sol_price_usd", return_value=(170.0, 0.0)), \
             patch("research.curve_oracle.get_curve_prices_batch",
                   return_value={"Mint111": {"price_usd": None, "failure_reason": "curve_account_missing"}}):
            price, source = v8p._curve_fallback_price("Mint111")
        self.assertEqual(price, 0.0)
        self.assertEqual(source, "curve_fallback_curve_account_missing")

    def test_resolve_entry_price_falls_back_to_curve_when_pp_never_ticks(self):
        v8p = self._v8_paper
        fake_monitor = SimpleNamespace(get_prices=lambda: {})
        with patch.dict("sys.modules", {}), \
             patch("memecoin.pumpportal_monitor.monitor", fake_monitor), \
             patch.object(v8p, "_PRICE_WAIT_S", 0.01), \
             patch.object(v8p, "_PRICE_POLL_INTERVAL_S", 0.005), \
             patch.object(v8p, "_curve_fallback_price", return_value=(0.0000456, "curve_fallback")):
            price, source = v8p._resolve_entry_price("solana", "Mint111")
        self.assertEqual(price, 0.0000456)
        self.assertEqual(source, "curve_fallback")

    def test_resolve_entry_price_preserves_curve_fallback_failure_reason(self):
        """2026-09-14: previously collapsed every fallback failure to the
        generic 'pp_unpriced', which made 56/60 real live failures
        undiagnosable from the journal/log alone -- must preserve the
        specific reason string instead."""
        v8p = self._v8_paper
        fake_monitor = SimpleNamespace(get_prices=lambda: {})
        with patch("memecoin.pumpportal_monitor.monitor", fake_monitor), \
             patch.object(v8p, "_PRICE_WAIT_S", 0.01), \
             patch.object(v8p, "_PRICE_POLL_INTERVAL_S", 0.005), \
             patch.object(v8p, "_curve_fallback_price", return_value=(0.0, "curve_fallback_no_key")):
            price, source = v8p._resolve_entry_price("solana", "Mint111")
        self.assertEqual(price, 0.0)
        self.assertEqual(source, "curve_fallback_no_key")

    def test_resolve_entry_price_falls_back_to_pp_unpriced_if_no_reason_at_all(self):
        v8p = self._v8_paper
        fake_monitor = SimpleNamespace(get_prices=lambda: {})
        with patch("memecoin.pumpportal_monitor.monitor", fake_monitor), \
             patch.object(v8p, "_PRICE_WAIT_S", 0.01), \
             patch.object(v8p, "_PRICE_POLL_INTERVAL_S", 0.005), \
             patch.object(v8p, "_curve_fallback_price", return_value=(0.0, "")):
            price, source = v8p._resolve_entry_price("solana", "Mint111")
        self.assertEqual(price, 0.0)
        self.assertEqual(source, "pp_unpriced")


class TestCurveFallbackPricesBatch(unittest.TestCase):
    """2026-09-17: batched ongoing-monitoring fallback, distinct from the
    one-shot _curve_fallback_price used only at entry. Multiple scenarios
    per the user's explicit request for thorough stress testing."""

    def setUp(self):
        import memecoin.v8_paper as v8_paper
        self._v8_paper = v8_paper

    def test_empty_input_returns_empty_without_any_calls(self):
        v8p = self._v8_paper
        with patch.dict("os.environ", {"HELIUS_API_KEY": "fake_key"}), \
             patch("research.curve_oracle.get_curve_prices_batch") as mock_batch:
            result = v8p._curve_fallback_prices_batch([])
        self.assertEqual(result, {})
        mock_batch.assert_not_called()

    def test_no_helius_key_returns_empty(self):
        v8p = self._v8_paper
        with patch.dict("os.environ", {"HELIUS_API_KEY": ""}):
            result = v8p._curve_fallback_prices_batch(["MintA", "MintB"])
        self.assertEqual(result, {})

    def test_no_sol_price_returns_empty(self):
        v8p = self._v8_paper
        with patch.dict("os.environ", {"HELIUS_API_KEY": "fake_key"}), \
             patch.object(v8p, "_fresh_sol_price_usd", return_value=(0.0, float("inf"))):
            result = v8p._curve_fallback_prices_batch(["MintA"])
        self.assertEqual(result, {})

    def test_partial_success_only_returns_successful_mints(self):
        """Some mints price, some fail (missing/graduated/parse error) --
        the failures must not appear in the result at all, not as 0.0."""
        v8p = self._v8_paper
        with patch.dict("os.environ", {"HELIUS_API_KEY": "fake_key"}), \
             patch.object(v8p, "_fresh_sol_price_usd", return_value=(170.0, 0.0)), \
             patch("research.curve_oracle.get_curve_prices_batch", return_value={
                 "MintA": {"price_usd": 0.0000123, "failure_reason": None},
                 "MintB": {"price_usd": None, "failure_reason": "curve_account_missing"},
                 "MintC": {"price_usd": 0.0000456, "failure_reason": None},
                 "MintD": {"price_usd": 0.0, "failure_reason": None},
             }):
            result = v8p._curve_fallback_prices_batch(["MintA", "MintB", "MintC", "MintD"])
        self.assertEqual(result, {"MintA": 0.0000123, "MintC": 0.0000456})
        self.assertNotIn("MintB", result)
        self.assertNotIn("MintD", result)

    def test_missing_mint_in_results_is_simply_absent(self):
        v8p = self._v8_paper
        with patch.dict("os.environ", {"HELIUS_API_KEY": "fake_key"}), \
             patch.object(v8p, "_fresh_sol_price_usd", return_value=(170.0, 0.0)), \
             patch("research.curve_oracle.get_curve_prices_batch", return_value={}):
            result = v8p._curve_fallback_prices_batch(["MintA", "MintB"])
        self.assertEqual(result, {})

    def test_exception_during_batch_call_returns_empty_not_raises(self):
        v8p = self._v8_paper
        with patch.dict("os.environ", {"HELIUS_API_KEY": "fake_key"}), \
             patch.object(v8p, "_fresh_sol_price_usd", return_value=(170.0, 0.0)), \
             patch("research.curve_oracle.get_curve_prices_batch",
                   side_effect=RuntimeError("RPC exploded")):
            result = v8p._curve_fallback_prices_batch(["MintA"])
        self.assertEqual(result, {})

    def test_all_mints_succeed(self):
        v8p = self._v8_paper
        with patch.dict("os.environ", {"HELIUS_API_KEY": "fake_key"}), \
             patch.object(v8p, "_fresh_sol_price_usd", return_value=(170.0, 0.0)), \
             patch("research.curve_oracle.get_curve_prices_batch", return_value={
                 "MintA": {"price_usd": 1e-5, "failure_reason": None},
                 "MintB": {"price_usd": 2e-5, "failure_reason": None},
             }):
            result = v8p._curve_fallback_prices_batch(["MintA", "MintB"])
        self.assertEqual(result, {"MintA": 1e-5, "MintB": 2e-5})


class TestStalePositionExitsAndDeadman(unittest.TestCase):
    """2026-09-17: _evaluate_stale_position_exits is what lets time_stop
    fire for positions that never got a fresh price this cycle, and
    stale_deadman is the safety net for winners (peak_gain>=0.30) that go
    permanently dark before ever pulling back through their trail level.
    Multiple scenarios per the user's explicit request."""

    def setUp(self):
        import memecoin.v8_paper as v8_paper
        self._v8_paper = v8_paper
        self._tmpdir = tempfile.TemporaryDirectory()
        self._journal = Path(self._tmpdir.name) / "v8_journal.csv"
        self._positions = Path(self._tmpdir.name) / "v8_positions.json"
        self._paths_patch = patch("memecoin.v8_paper._paths",
                                   return_value=(self._journal, self._positions))
        self._paths_patch.start()
        self.book = v8_paper.V8PaperBook()

    def tearDown(self):
        self._paths_patch.stop()
        self._tmpdir.cleanup()

    def _make_pos(self, **overrides):
        now = time.time()
        base = dict(
            id="V8test1", signal_id="sig1", chain="solana",
            token_address="MintStale111111111111111111111111111111",
            token_symbol="STALE", signal_type="social_alert", strength="v8_fork",
            signal_price=1e-5, signal_time=now, entry_price=1e-5, entry_time=now,
            last_priced_at=now, size_usd=1.0, current_price=1e-5, peak_price=1e-5,
            status="open", exit_price=0.0, exit_time=0.0, exit_reason="",
            progress_at_signal=0.5, dex_id="", entry_source="pp_tick_v8_fork",
            era="V8_TELEGRAM_INDEPENDENT_V1", notes="",
        )
        base.update(overrides)
        return base

    def test_fresh_position_stays_open(self):
        pos = self._make_pos()
        self.book._positions = {pos["id"]: pos}
        self.book._evaluate_stale_position_exits()
        self.assertEqual(self.book._positions[pos["id"]]["status"], "open")

    def test_time_stop_fires_for_a_position_never_repriced_this_cycle(self):
        """This is the core VR fix: a position with no gain, past
        time_stop_minutes, that never got a fresh price this specific
        cycle must still close -- previously it would never even be
        evaluated."""
        v8_cfg_time_stop_min = self._v8_paper.V8_EXIT_CONFIG["time_stop_minutes"]
        old_ts = time.time() - (v8_cfg_time_stop_min + 5) * 60
        pos = self._make_pos(entry_time=old_ts, last_priced_at=old_ts,
                              current_price=1e-5, peak_price=1e-5)  # zero gain
        self.book._positions = {pos["id"]: pos}
        self.book._evaluate_stale_position_exits()
        closed = self.book._positions[pos["id"]]
        self.assertEqual(closed["status"], "closed")
        self.assertEqual(closed["exit_reason"], "time_stop")

    def test_stale_deadman_fires_for_a_winner_gone_dark(self):
        """The actual gap this session found: peak_gain >= 0.30 disables
        time_stop by design ("never interrupt a runner"), so a position
        that ran up then went permanently dark must be caught by the
        deadman instead, under its own distinct reason."""
        deadman_s = self._v8_paper._STALE_DATA_DEADMAN_S
        old_priced = time.time() - (deadman_s + 60)
        pos = self._make_pos(
            entry_price=1e-5, current_price=2e-5, peak_price=2e-5,   # +100% peak gain
            last_priced_at=old_priced,
        )
        self.book._positions = {pos["id"]: pos}
        self.book._evaluate_stale_position_exits()
        closed = self.book._positions[pos["id"]]
        self.assertEqual(closed["status"], "closed")
        self.assertEqual(closed["exit_reason"], "stale_deadman")

    def test_deadman_does_not_fire_for_actively_priced_winner(self):
        """A position still getting real price ticks every cycle must
        never be at risk from the deadman, no matter how long it's open."""
        pos = self._make_pos(
            entry_price=1e-5, current_price=2e-5, peak_price=2e-5,
            entry_time=time.time() - 999999,   # very old, but...
            last_priced_at=time.time(),        # ...still being priced right now
        )
        self.book._positions = {pos["id"]: pos}
        self.book._evaluate_stale_position_exits()
        self.assertEqual(self.book._positions[pos["id"]]["status"], "open")

    def test_legacy_position_missing_last_priced_at_falls_back_to_entry_time(self):
        """Positions opened before 2026-09-17 have no last_priced_at key
        at all. Must fail closed (treat as stale) using entry_time, not
        crash with a KeyError."""
        old_ts = time.time() - (self._v8_paper._STALE_DATA_DEADMAN_S + 60)
        pos = self._make_pos(entry_time=old_ts, current_price=1e-5, peak_price=1e-5)
        del pos["last_priced_at"]
        self.book._positions = {pos["id"]: pos}
        self.book._evaluate_stale_position_exits()   # must not raise
        self.assertEqual(self.book._positions[pos["id"]]["status"], "closed")
        self.assertEqual(self.book._positions[pos["id"]]["exit_reason"], "stale_deadman")

    def test_hard_stop_still_takes_priority_over_deadman_when_price_is_fresh(self):
        pos = self._make_pos(entry_price=1e-5, current_price=6e-6, peak_price=1e-5,
                              last_priced_at=time.time())   # -40%, fresh
        self.book._positions = {pos["id"]: pos}
        self.book._evaluate_stale_position_exits()
        closed = self.book._positions[pos["id"]]
        self.assertEqual(closed["status"], "closed")
        self.assertEqual(closed["exit_reason"], "hard_stop")

    def test_mixed_batch_only_the_correct_positions_close(self):
        """Multiple positions in one pass -- verifies no cross-contamination
        between positions and that closed ones are skipped entirely."""
        fresh_ok = self._make_pos(id="V8fresh", token_address="MintFresh1111111111111111111111111111",
                                   last_priced_at=time.time())
        deadman_hit = self._make_pos(
            id="V8dead", token_address="MintDead11111111111111111111111111111",
            entry_price=1e-5, current_price=2e-5, peak_price=2e-5,
            last_priced_at=time.time() - (self._v8_paper._STALE_DATA_DEADMAN_S + 1),
        )
        already_closed = self._make_pos(id="V8closed", status="closed", exit_reason="hard_stop")
        self.book._positions = {
            fresh_ok["id"]: fresh_ok,
            deadman_hit["id"]: deadman_hit,
            already_closed["id"]: already_closed,
        }
        self.book._evaluate_stale_position_exits()
        self.assertEqual(self.book._positions["V8fresh"]["status"], "open")
        self.assertEqual(self.book._positions["V8dead"]["status"], "closed")
        self.assertEqual(self.book._positions["V8dead"]["exit_reason"], "stale_deadman")
        # Already-closed position must be left completely untouched.
        self.assertEqual(self.book._positions["V8closed"]["exit_reason"], "hard_stop")

    def test_empty_positions_dict_is_a_safe_noop(self):
        self.book._positions = {}
        self.book._evaluate_stale_position_exits()   # must not raise
        self.assertEqual(self.book._positions, {})

    def test_new_position_seeds_last_priced_at_equal_to_entry_time(self):
        v8p = self._v8_paper
        with patch(_PATCH_TARGET, return_value=_cap(0.40, "CURVE_ACTIVE")), \
             patch.object(v8p, "_resolve_entry_price", return_value=(0.00002, "pp_tick_v8_fork")), \
             patch.object(v8p, "_current_era", return_value="TEST_ERA"), \
             patch("memecoin.pumpportal_monitor.monitor", SimpleNamespace(subscribe=lambda m: None)):
            self.book._evaluate_alert(_event(
                token_address="MintSeed1111111111111111111111111111111",
                event_id="ev_seed"))
        opened = [p for p in self.book._positions.values() if p["status"] == "open"]
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0]["last_priced_at"], opened[0]["entry_time"])

    def test_update_price_refreshes_last_priced_at(self):
        pos = self._make_pos(last_priced_at=time.time() - 500)
        self.book._positions = {pos["id"]: pos}
        before = self.book._positions[pos["id"]]["last_priced_at"]
        self.book.update_price(pos["token_address"], 1.1e-5)
        after = self.book._positions[pos["id"]]["last_priced_at"]
        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()
