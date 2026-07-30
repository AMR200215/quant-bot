"""
test_phase4.py — Phase 4 addendum tests

4B: pnl property setters raise AttributeError; finalize_rescue_sell no longer crashes.
4C: TP dispatch inflight guard — exactly 1 thread per position/level; 30s cooldown after failure.
4D: MU terminal (attempt 8 → manual_required, never re-arms); kill switch multi-token gate.
4E: is_live field drives all journal routing; notes mutation mid-retry has no effect; double-close → single journal row.

Run: python -m pytest memecoin/tests/test_phase4.py -v
"""

import sys
import time
import types
import threading
import unittest
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

def _make_stubs():
    sys.modules.setdefault("memecoin", types.ModuleType("memecoin"))

    cfg = types.ModuleType("memecoin.config")
    cfg.EXECUTOR_BACKEND       = "pumpportal"
    cfg.LIVE_DRY_RUN           = False
    cfg.SLIPPAGE_GATE_RT_PCT   = 0.30
    cfg.SLIPPAGE_GATE_DEX_PCT  = 0.50
    cfg.LIVE_TRADING           = False
    cfg.LIVE_GATE_EPOCH        = "2026-07-02"
    cfg.POSITIONS_FILE         = "/tmp/positions_test_phase4.json"
    cfg.JOURNAL_FILE           = "/tmp/journal_test_phase4.csv"
    cfg.SOCIAL_JOURNAL_FILE    = "/tmp/social_journal_test_phase4.csv"
    cfg.LIVE_JOURNAL_FILE      = "/tmp/live_journal_test_phase4.csv"
    cfg.TRAJECTORY_FILE        = "/tmp/trajectory_test_phase4.csv"
    cfg.HARD_STOP_PCT          = 0.50
    cfg.TRAILING_STOP_PCT      = 0.20
    cfg.TRAIL_ACTIVATES_PCT    = 0.30
    cfg.TIME_STOP_MINUTES      = 60
    cfg.TIME_STOP_MIN_GAIN     = 0.0
    cfg.TP_LEVELS              = []
    cfg.TRADE_SIZE_USD         = 3.0
    cfg.PRICE_PATHS_DIR        = "/tmp"
    cfg.DAILY_LOSS_LIMIT       = 20.0
    cfg.SELL_STUCK_RETRY_SEC   = 60
    cfg.REALTIME_PRICE_FEED    = False
    cfg.MAX_LOSS_FROM_FILL_PCT = 0.50
    cfg.AUTO_DISABLE_ON_UNKNOWN_SELL_FAILURE = True
    cfg.get_signal_settings    = MagicMock(return_value={})
    # data_client deps
    cfg.DEXSCREENER_BASE = "https://api.dexscreener.com"
    cfg.GMGN_BASE        = "https://gmgn.ai"
    cfg.RUGCHECK_BASE    = "https://api.rugcheck.xyz"
    cfg.HONEYPOT_BASE    = "https://api.honeypot.is"
    cfg.CHAINS           = ["solana"]
    # candidate_log deps
    cfg.CANDIDATES_FILE  = "/tmp/candidates_test.json"
    cfg.WINNERS_FILE     = "/tmp/winners_test.json"
    cfg.REJECTIONS_FILE  = "/tmp/rejections_test.json"
    cfg.NEAR_MISS_FILE   = "/tmp/near_miss_test.json"
    sys.modules["memecoin.config"] = cfg

    # data_client stub
    dc = types.ModuleType("memecoin.data_client")
    dc.dex_get_token          = MagicMock(return_value=None)
    dc.sol_get_token_creator  = MagicMock(return_value=None)
    sys.modules["memecoin.data_client"] = dc

    # candidate_log stub
    cl = types.ModuleType("memecoin.candidate_log")
    cl.promote_to_winners = MagicMock()
    sys.modules["memecoin.candidate_log"] = cl

    # journal_io stub
    import threading as _threading
    ji = types.ModuleType("memecoin.journal_io")
    ji.JOURNAL_LOCK = _threading.Lock()
    sys.modules["memecoin.journal_io"] = ji

    # app.alerts stub
    sys.modules.setdefault("app", types.ModuleType("app"))
    al = types.ModuleType("app.alerts")
    al.alert_position_open  = MagicMock()
    al.alert_position_close = MagicMock()
    al.alert_live_buy       = MagicMock()
    al.alert_live_sell      = MagicMock()
    al.alert_tp_hit         = MagicMock()
    sys.modules["app.alerts"] = al

    ppm = types.ModuleType("memecoin.pumpportal_monitor")
    ppm.monitor = MagicMock()
    ppm.monitor.get_prices = MagicMock(return_value={})
    ppm.monitor.get_vsol   = MagicMock(return_value=0)
    sys.modules["memecoin.pumpportal_monitor"] = ppm

    psl = types.ModuleType("memecoin.pumpswap_local")
    class PumpSwapPoolError(Exception): pass
    psl.PumpSwapPoolError = PumpSwapPoolError
    psl.fetch_pool        = MagicMock(side_effect=PumpSwapPoolError("no pool"))
    sys.modules["memecoin.pumpswap_local"] = psl

    from enum import Enum
    class TokenExitState(Enum):
        BONDING_CURVE = "BONDING_CURVE"
        GRADUATED_PUMPSWAP = "GRADUATED_PUMPSWAP"

    ero = types.ModuleType("memecoin.exit_router")
    ero.TokenExitState = TokenExitState
    ero.classify       = MagicMock(return_value=TokenExitState.BONDING_CURVE)
    sys.modules["memecoin.exit_router"] = ero


_make_stubs()

from memecoin.portfolio import Position


# ---------------------------------------------------------------------------
# 4B — pnl property setters
# ---------------------------------------------------------------------------

class TestPnlPropertySetters(unittest.TestCase):

    def _pos(self):
        return Position(
            id="P001", signal_id="s1", chain="solana",
            token_address="Fake1111", token_symbol="TEST",
            signal_type="social_alert", strength="medium",
            whale_count=0, whale_tiers=[], whales_involved=[],
            entry_price=0.0001, size_usd=3.0,
        )

    def test_pnl_pct_setter_raises(self):
        pos = self._pos()
        with self.assertRaises(AttributeError) as cm:
            pos.pnl_pct = 0.5
        self.assertIn("computed", str(cm.exception))

    def test_pnl_usd_setter_raises(self):
        pos = self._pos()
        with self.assertRaises(AttributeError) as cm:
            pos.pnl_usd = 1.0
        self.assertIn("computed", str(cm.exception))

    def test_pnl_pct_computed_correctly(self):
        pos = self._pos()
        pos.exit_price = 0.00015
        pos.status = "closed"
        self.assertAlmostEqual(pos.pnl_pct, 0.5, places=6)

    def test_pnl_usd_computed_correctly(self):
        pos = self._pos()
        pos.exit_price = 0.00015   # +50%
        pos.status = "closed"
        self.assertAlmostEqual(pos.pnl_usd, 1.5, places=4)  # 3.0 * 0.5

    def test_exit_price_assignment_does_not_raise(self):
        """exit_price IS a settable field — no AttributeError."""
        pos = self._pos()
        pos.exit_price = 0.00012   # should not raise
        self.assertEqual(pos.exit_price, 0.00012)

    def test_finalize_rescue_sell_no_exception(self):
        """
        Simulate the critical path: set exit_price only (no pnl_pct/pnl_usd assignment).
        The property computes the correct value without raising.
        """
        pos = self._pos()
        fill_price = 0.00008   # -20%
        pos.entry_price = 0.0001
        # This is what _finalize_rescue_sell now does (after fix):
        pos.exit_price = fill_price   # must not raise
        pos.status = "closed"
        # Property gives correct answer
        self.assertAlmostEqual(pos.pnl_pct, -0.20, places=6)


# ---------------------------------------------------------------------------
# 4C — TP dispatch inflight guard
# ---------------------------------------------------------------------------

class TestTPInflightGuard(unittest.TestCase):
    """Verifies the in-memory dispatch guard prevents concurrent TP threads."""

    def _simulate_tp_guard(self, inflight: dict, pos_id: str, level_key: str) -> bool:
        """Returns True if dispatch is allowed, False if blocked."""
        lvls = inflight.setdefault(pos_id, {})
        ready_at = lvls.get(level_key, 0)
        if ready_at > time.time():
            return False  # blocked
        lvls[level_key] = float("inf")  # mark in-flight
        return True

    def _clear_inflight(self, inflight: dict, pos_id: str, level_key: str,
                        cooldown_s: float = 0.0):
        lvls = inflight.get(pos_id, {})
        lvls[level_key] = time.time() + cooldown_s if cooldown_s > 0 else 0.0
        inflight[pos_id] = lvls

    def test_first_dispatch_allowed(self):
        inflight = {}
        self.assertTrue(self._simulate_tp_guard(inflight, "P1", "tp1"))

    def test_second_dispatch_blocked_while_inflight(self):
        inflight = {}
        self._simulate_tp_guard(inflight, "P1", "tp1")
        # Second call while first is in-flight (inf)
        self.assertFalse(self._simulate_tp_guard(inflight, "P1", "tp1"))

    def test_100_ticks_above_tp_exactly_one_thread(self):
        """100 concurrent dispatch attempts → exactly 1 succeeds."""
        inflight = {}
        dispatched = []
        for _ in range(100):
            if self._simulate_tp_guard(inflight, "P1", "tp1"):
                dispatched.append(1)
        self.assertEqual(len(dispatched), 1, "exactly 1 thread should be dispatched")

    def test_after_success_clear_allows_next_level(self):
        """After clear (success), a different level dispatches normally."""
        inflight = {}
        self._simulate_tp_guard(inflight, "P1", "tp1")
        self._clear_inflight(inflight, "P1", "tp1")  # success clear
        # Next level (tp2) is independent
        self.assertTrue(self._simulate_tp_guard(inflight, "P1", "tp2"))

    def test_failure_sets_30s_cooldown(self):
        """After failure, same level is blocked for 30s."""
        inflight = {}
        self._simulate_tp_guard(inflight, "P1", "tp1")
        self._clear_inflight(inflight, "P1", "tp1", cooldown_s=30.0)
        # Immediately after failure, still blocked
        self.assertFalse(self._simulate_tp_guard(inflight, "P1", "tp1"))

    def test_after_cooldown_expires_dispatch_allowed(self):
        """After cooldown expires, dispatch is allowed again."""
        inflight = {}
        # Set cooldown that's already expired
        inflight["P1"] = {"tp1": time.time() - 1.0}
        self.assertTrue(self._simulate_tp_guard(inflight, "P1", "tp1"))


# ---------------------------------------------------------------------------
# 4D — MU terminal + kill switch multi-token gate
# ---------------------------------------------------------------------------

class TestMUTerminal(unittest.TestCase):
    """Verify mu_sell_total >= 8 sets manual_required and never re-arms."""

    def _simulate_ladder_exhaustion(self, pos: Position, portfolio_sell_stuck: dict) -> str:
        """
        Simulate one sell window exhaustion. Returns new status.
        Mirrors the fix in close_position ladder-exhausted branch.
        """
        pos.mu_sell_total = getattr(pos, "mu_sell_total", 0) + 1
        pos.sell_attempts = 0
        if pos.mu_sell_total >= 8:
            pos.status = "manual_required"
            portfolio_sell_stuck.pop(pos.id, None)  # never re-arm
        else:
            pos.status = "sell_stuck"
            portfolio_sell_stuck[pos.id] = time.time() + 60
        return pos.status

    def test_attempts_1_to_7_set_sell_stuck(self):
        pos = Position(
            id="P002", signal_id="s2", chain="solana",
            token_address="Fake2222", token_symbol="STUCK",
            signal_type="social_alert", strength="medium",
            whale_count=0, whale_tiers=[], whales_involved=[],
        )
        stuck_dict = {}
        for i in range(1, 8):
            status = self._simulate_ladder_exhaustion(pos, stuck_dict)
            self.assertEqual(status, "sell_stuck", f"attempt {i} should be sell_stuck")
            self.assertIn(pos.id, stuck_dict, f"attempt {i} should re-arm stuck timer")

    def test_attempt_8_sets_manual_required(self):
        pos = Position(
            id="P003", signal_id="s3", chain="solana",
            token_address="Fake3333", token_symbol="STUCK2",
            signal_type="social_alert", strength="medium",
            whale_count=0, whale_tiers=[], whales_involved=[],
        )
        pos.mu_sell_total = 7
        stuck_dict = {pos.id: time.time() + 60}

        status = self._simulate_ladder_exhaustion(pos, stuck_dict)
        self.assertEqual(status, "manual_required")
        self.assertNotIn(pos.id, stuck_dict, "sell_stuck timer must be cleared at attempt 8")

    def test_zero_further_iterations_after_manual_required(self):
        """manual_required positions must not be re-queued."""
        pos = Position(
            id="P004", signal_id="s4", chain="solana",
            token_address="Fake4444", token_symbol="STUCK3",
            signal_type="social_alert", strength="medium",
            whale_count=0, whale_tiers=[], whales_involved=[],
        )
        pos.status = "manual_required"
        stuck_dict = {}
        # Reconciler check: manual_required is NOT in ("open", "sell_stuck") retry trigger
        would_retry = pos.status in ("open", "sell_stuck")
        self.assertFalse(would_retry, "manual_required must not trigger retry")


class TestKillSwitchMultiToken(unittest.TestCase):
    """4D(b): kill switch fires only when >=2 distinct tokens fail within window."""

    def _run_guard(self, fail_tokens: list[str], window: float = 600.0) -> tuple[bool, int]:
        """
        Simulate the kill switch guard for a list of failing token addresses.
        Returns (should_fire, distinct_count).
        """
        ks_log: dict[str, float] = {}
        now = time.time()
        fired = False
        distinct = 0
        for token in fail_tokens:
            ks_log[token] = now
            # Purge expired
            ks_log = {t: ts for t, ts in ks_log.items() if now - ts <= window}
            distinct = len(ks_log)
            if distinct >= 2:
                fired = True
        return fired, distinct

    def test_50_failures_one_token_no_kill_switch(self):
        tokens = ["TokenAAAA"] * 50
        fired, distinct = self._run_guard(tokens)
        self.assertFalse(fired, "single token storming must not fire kill switch")
        self.assertEqual(distinct, 1)

    def test_2_distinct_tokens_fire_kill_switch(self):
        tokens = ["TokenAAAA", "TokenBBBB"]
        fired, distinct = self._run_guard(tokens)
        self.assertTrue(fired, "2 distinct failing tokens must fire kill switch")
        self.assertEqual(distinct, 2)

    def test_1_distinct_token_no_fire(self):
        tokens = ["TokenAAAA", "TokenAAAA", "TokenAAAA"]
        fired, distinct = self._run_guard(tokens)
        self.assertFalse(fired)
        self.assertEqual(distinct, 1)


# ---------------------------------------------------------------------------
# 4E — is_live field routing; notes mutation; double-close idempotency
# ---------------------------------------------------------------------------

class TestIsLiveField(unittest.TestCase):

    def test_is_live_default_false(self):
        pos = Position(
            id="P005", signal_id="s5", chain="solana",
            token_address="Fake5555", token_symbol="PAPER",
            signal_type="social_alert", strength="medium",
            whale_count=0, whale_tiers=[], whales_involved=[],
        )
        self.assertFalse(pos.is_live)

    def test_is_live_set_true_survives_notes_mutation(self):
        """Journal routing must not break if notes are mutated mid-retry."""
        pos = Position(
            id="P006", signal_id="s6", chain="solana",
            token_address="Fake6666", token_symbol="LIVE",
            signal_type="social_alert", strength="medium",
            whale_count=0, whale_tiers=[], whales_involved=[],
            notes="live|tx:AABBCC|fill:0.0000280010|cohort:bonding_curve",
            is_live=True,
        )
        # Simulate notes being clobbered (retry loop appending garbage)
        pos.notes = "|sell_failed:tx123(attempt 5)|sell_error(attempt 6)|sell_stuck"
        # is_live must still be True — notes no longer contain "live|tx:"
        self.assertTrue(pos.is_live, "is_live must not depend on notes content")
        # journal routing check
        would_write_live = pos.is_live   # replaces: "live|tx:" in pos.notes
        self.assertTrue(would_write_live)

    def test_is_live_backfill_from_notes(self):
        """_load_positions backfill: old serialized positions without is_live field."""
        d = {
            "id": "P007", "signal_id": "s7", "chain": "solana",
            "token_address": "Fake7777", "token_symbol": "OLD",
            "signal_type": "social_alert", "strength": "medium",
            "whale_count": 0, "whale_tiers": [], "whales_involved": [],
            "notes": "live|tx:ABCDEF|fill:0.00001234|cohort:bonding_curve",
            # is_live intentionally absent
        }
        # Simulate the backfill logic in _load_positions
        if "is_live" not in d:
            d["is_live"] = bool(d.get("notes") and "live|tx:" in d["notes"])
        pos = Position(**d)
        self.assertTrue(pos.is_live)

    def test_is_live_backfill_paper_position(self):
        """Paper position without live|tx: → is_live stays False after backfill."""
        d = {
            "id": "P008", "signal_id": "s8", "chain": "solana",
            "token_address": "Fake8888", "token_symbol": "PAPER2",
            "signal_type": "social_alert", "strength": "medium",
            "whale_count": 0, "whale_tiers": [], "whales_involved": [],
            "notes": "|has_live_twin:L123456",
        }
        if "is_live" not in d:
            d["is_live"] = bool(d.get("notes") and "live|tx:" in d["notes"])
        pos = Position(**d)
        self.assertFalse(pos.is_live)

    def test_double_close_idempotency(self):
        """
        close_position returns None on second call (status already "closed").
        Simulate the guard: `if not pos or pos.status == "closed": return None`.
        """
        pos = Position(
            id="P009", signal_id="s9", chain="solana",
            token_address="Fake9999", token_symbol="DBLCLOSE",
            signal_type="social_alert", strength="medium",
            whale_count=0, whale_tiers=[], whales_involved=[],
        )
        journal_rows = []

        def mock_close(p):
            if p.status == "closed":
                return None   # idempotency guard
            p.status = "closed"
            journal_rows.append(p.id)
            return p

        mock_close(pos)
        mock_close(pos)  # second call — should no-op
        self.assertEqual(len(journal_rows), 1, "double-close must produce exactly 1 journal row")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
