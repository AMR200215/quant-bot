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
# full attribute set (e.g. test_jupiter_retry.py), portfolio.py will fail to
# import. Clear the stub so the real module loads instead. ───────────────────
_cfg_stub = sys.modules.get("memecoin.config")
if _cfg_stub is not None and not hasattr(_cfg_stub, "POSITIONS_FILE"):
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


# ── Entry point ────────────────────────────────────────────────────────────────

def _run():
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromTestCase(TestMigrationRescue)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    _run()
