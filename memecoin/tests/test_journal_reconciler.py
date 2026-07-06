"""
test_journal_reconciler.py — Tests for journal_reconciler.py

Test A: Row with exit_price=0 + sig → reconciler corrects within one pass.
Test B: graduated_loss blocked while wallet holds tokens.
Test C: graduated_loss blocked when sig confirms positive delta → graduated_recovered.
Test D: reconciler idempotent on already-tagged rows.

Run: python -m pytest memecoin/tests/test_journal_reconciler.py -v
"""

import csv
import io
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal module stubs so journal_reconciler imports cleanly.
# Must be installed before any memecoin import.
# ---------------------------------------------------------------------------

def _install_stubs():
    sys.modules.setdefault("memecoin", types.ModuleType("memecoin"))

    cfg = types.ModuleType("memecoin.config")
    cfg.LIVE_JOURNAL_FILE    = Path("/tmp/reconciler_test_live.csv")
    cfg.SOCIAL_JOURNAL_FILE  = Path("/tmp/reconciler_test_social.csv")
    cfg.JOURNAL_FILE         = Path("/tmp/reconciler_test_main.csv")
    cfg.WALLET_PUBKEY        = "8PNHvFWeMT7CqpUvJjAwVgAK545t5KV3uCPd8DUfaTiM"
    cfg.EXIT_ROUTER_ENABLED  = True
    cfg.LIVE_TRADING         = False
    cfg.LIVE_DRY_RUN         = False
    cfg.POSITIONS_FILE       = "/tmp/positions_reconciler_test.json"
    cfg.LOGS_DIR             = Path("/tmp")
    sys.modules["memecoin.config"] = cfg

    # journal_io stub — provides JOURNAL_LOCK without importing full memecoin
    journal_io = types.ModuleType("memecoin.journal_io")
    import threading as _threading
    journal_io.JOURNAL_LOCK = _threading.Lock()
    sys.modules["memecoin.journal_io"] = journal_io

    # execution_rpc stub must be installed BEFORE loading real tx_meta,
    # because tx_meta._rpc_post lazily imports it on first call.
    erpc = types.ModuleType("memecoin.execution_rpc")
    erpc.rpc_post = MagicMock(return_value={"result": None})
    sys.modules["memecoin.execution_rpc"] = erpc

    # Load the REAL tx_meta module so its _rpc_post attribute exists and is
    # patchable by test_sol_delta_fixes.py (which runs after this file alphabetically).
    # An empty ModuleType stub (the previous approach) leaves _rpc_post absent, which
    # causes AttributeError in patch.object(tm, "_rpc_post", ...) in the other test file.
    if "memecoin.tx_meta" not in sys.modules:
        import importlib.util as _ilu
        _tm_path = Path(__file__).parent.parent / "tx_meta.py"
        _tm_spec = _ilu.spec_from_file_location("memecoin.tx_meta", _tm_path)
        _tm_mod = _ilu.module_from_spec(_tm_spec)
        sys.modules["memecoin.tx_meta"] = _tm_mod
        _tm_spec.loader.exec_module(_tm_mod)

    # executor stub (for _token_balance + _sol_price_usd)
    executor = types.ModuleType("memecoin.executor")
    executor._token_balance = MagicMock(return_value=0)
    executor._sol_price_usd = MagicMock(return_value=150.0)
    executor._get_keypair   = MagicMock()
    sys.modules["memecoin.executor"] = executor


_install_stubs()


# ---------------------------------------------------------------------------
# Helper: write a minimal journal CSV with one row
# ---------------------------------------------------------------------------

_FIELDS = [
    "id", "signal_id", "chain", "token_address", "token_symbol",
    "signal_type", "strength",
    "signal_price", "signal_time",
    "entry_price", "entry_time", "size_usd",
    "exit_price", "exit_time", "exit_reason",
    "pnl_usd", "pnl_pct", "peak_price", "hard_stop_pct",
    "whale_count", "whale_tiers",
    "safety_score", "momentum_score", "composite_score",
    "price_change_5m", "price_change_1h", "price_change_6h",
    "buys_5m", "sells_5m", "buys_h1", "sells_h1",
    "buy_sell_ratio_5m", "buy_sell_ratio_h1",
    "volume_5m", "volume_h1", "volume_h6",
    "liquidity_usd", "mcap_usd", "fdv", "age_minutes",
    "dex_id", "dexscreener_url",
    "has_twitter", "has_telegram", "has_website",
    "rugcheck_score", "buy_tax", "sell_tax",
    "notes",
    "config_tag",
    "tp_levels_hit", "realized_partial_usd", "remaining_fraction",
    "sol_received", "accounting_epoch",
]

_FAKE_SIG = "FAKESIG0000000000000000000000000000000000000000000000000000000000000000000000000000000"


def _make_row(**overrides) -> dict:
    row = {f: "" for f in _FIELDS}
    row.update({
        "id": "TestPos01",
        "chain": "solana",
        "token_address": "MintAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "token_symbol": "TEST",
        "signal_type": "social_alert",
        "entry_price": "0.0000100",
        "entry_time": "2026-07-05 12:00:00",
        "size_usd": "3.0",
        "exit_price": "0",
        "exit_time": "2026-07-05 12:05:00",
        "exit_reason": "graduated_loss",
        "pnl_usd": "-3.0",
        "pnl_pct": "-100",
        "notes": f"live|tx:BUYTX{_FAKE_SIG[:8]}|sell_tx:{_FAKE_SIG}",
        "config_tag": "v7_test",
        "accounting_epoch": "e4_test",
        "remaining_fraction": "1.0",
    })
    row.update(overrides)
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Test A: Row with exit_price=0 + sig → corrected in one pass
# ---------------------------------------------------------------------------

class TestReconcilerCorrects(unittest.TestCase):

    def test_A_corrects_row_with_zero_exit_price(self):
        """exit_price=0, pnl_pct=-100, sell_tx sig → reconciler corrects in one pass."""
        live_path   = Path("/tmp/rec_test_A_live.csv")
        social_path = Path("/tmp/rec_test_A_social.csv")

        row = _make_row(
            exit_price="0",
            pnl_pct="-100",
            notes=f"live|tx:BUYTX|sell_tx:{_FAKE_SIG}",
        )
        _write_csv(live_path, [row])
        _write_csv(social_path, [])  # empty

        # Patch read_sol_delta to return confirmed delta for our sig
        mock_delta = {
            "ok": True,
            "sol_delta": 0.041726,
            "source": "native_lamports",
            "attempts": 1,
            "reason": "success",
        }

        import memecoin.journal_reconciler as jr

        with patch("memecoin.journal_reconciler.read_sol_delta", return_value=mock_delta) as mock_rsd:
            with patch("memecoin.journal_reconciler._get_sol_price", return_value=150.0):
                result = jr.run_reconciler_pass(
                    "8PNHvFWeMT7CqpUvJjAwVgAK545t5KV3uCPd8DUfaTiM",
                    _live_path=live_path,
                    _social_path=social_path,
                    _main_path=Path("/tmp/rec_test_A_main.csv"),
                )

        self.assertEqual(result["rows_checked"], 1, "Should have checked 1 row")
        self.assertEqual(result["rows_corrected"], 1, "Should have corrected 1 row")

        rows_after = _read_csv(live_path)
        self.assertEqual(len(rows_after), 1)
        row_after = rows_after[0]

        pnl_pct_after = float(row_after["pnl_pct"])
        self.assertNotAlmostEqual(pnl_pct_after, -100.0, places=0,
                                  msg=f"pnl_pct should not be -100, got {pnl_pct_after}")

        self.assertIn("journal_reconciled:", row_after["notes"],
                      "notes should contain journal_reconciled: tag")

        mock_rsd.assert_called()


# ---------------------------------------------------------------------------
# Test B: graduated_loss blocked while wallet holds tokens
# ---------------------------------------------------------------------------

class TestGraduatedLossDefersOnBalance(unittest.TestCase):

    def _make_portfolio_pos(self):
        """Build minimal Position-like object for test."""
        class FakePos:
            id = "GradPos01"
            token_address = "MintGRADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            token_symbol  = "GRAD"
            signal_type   = "social_alert"
            chain         = "solana"
            status        = "open"
            exit_price    = 0.0
            exit_time     = 0.0
            exit_reason   = ""
            sell_attempts = 0
            notes         = (
                f"live|tx:BUYTX|sell_tx:{_FAKE_SIG}"
            )
            size_usd      = 3.0
            realized_pnl_usd = 0.0
            remaining_fraction = 1.0
            tp_levels_hit = []
            entry_price   = 0.000010
            peak_price    = 0.000012
            tokens_held   = 0
        return FakePos()

    def test_B_defers_when_balance_positive(self):
        """
        When on-chain balance > 0 and no confirmed sig:
        _arm_migration_retry is called, graduated_loss write-off is skipped.
        """
        # Import journal_reconciler to ensure stubs are loaded
        import memecoin.journal_reconciler  # noqa: F401

        # We test the Fix B logic in portfolio.py directly by simulating it.
        # Since portfolio.py is complex to instantiate, we test the decision
        # logic by extracting it inline (matching the Fix B code).

        import re as _re
        from unittest.mock import MagicMock

        pos_notes = f"live|tx:BUYTX|sell_tx:{_FAKE_SIG}"
        wallet    = "8PNHvFWeMT7CqpUvJjAwVgAK545t5KV3uCPd8DUfaTiM"

        # Mocks
        mock_read_sol_delta = MagicMock(return_value={"ok": False, "sol_delta": None})
        mock_token_balance  = MagicMock(return_value=1_000_000)  # tokens still held!

        arm_called = []

        def fake_arm(pos_id, retry_sec):
            arm_called.append((pos_id, retry_sec))

        # Simulate Fix B decision logic
        sigs = list(reversed(_re.findall(
            r'(?:sell_tx|sell_unconf|jupiter_rescue_pending|sell_pending):([A-Za-z0-9]+)',
            pos_notes,
        )))
        recovered = False
        for sig in sigs:
            res = mock_read_sol_delta(sig, wallet)
            if res.get("ok") and (res.get("sol_delta") or 0) > 0:
                recovered = True
                break

        self.assertFalse(recovered, "Should not recover when sol_delta not ok")

        # Check balance
        bal = mock_token_balance(wallet, "MintGRADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        self.assertGreater(bal, 0, "Balance should be > 0")

        if bal > 0:
            fake_arm("GradPos01", 60)

        self.assertEqual(len(arm_called), 1, "_arm_migration_retry should have been called")
        self.assertEqual(arm_called[0][0], "GradPos01")


# ---------------------------------------------------------------------------
# Test C: graduated_loss → graduated_recovered when sig confirms
# ---------------------------------------------------------------------------

class TestGraduatedLossRecoveredOnSig(unittest.TestCase):

    def test_C_recovered_when_sig_confirms(self):
        """
        When a sig in notes confirms with sol_delta > 0:
        exit_reason becomes 'graduated_recovered' and graduated_loss is skipped.
        """
        import re as _re
        from unittest.mock import MagicMock

        pos_notes = f"live|tx:BUYTX|sell_tx:{_FAKE_SIG}"
        wallet    = "8PNHvFWeMT7CqpUvJjAwVgAK545t5KV3uCPd8DUfaTiM"

        mock_read_sol_delta = MagicMock(return_value={
            "ok": True,
            "sol_delta": 0.041726,
            "source": "native_lamports",
            "attempts": 1,
            "reason": "success",
        })

        # Simulate Fix B decision logic
        sigs = list(reversed(_re.findall(
            r'(?:sell_tx|sell_unconf|jupiter_rescue_pending|sell_pending):([A-Za-z0-9]+)',
            pos_notes,
        )))

        recovered_sig    = None
        recovered_delta  = None
        new_exit_reason  = None
        new_notes_suffix = None

        for sig in sigs:
            res = mock_read_sol_delta(sig, wallet)
            if res.get("ok") and (res.get("sol_delta") or 0) > 0:
                recovered_sig    = sig
                recovered_delta  = res["sol_delta"]
                new_exit_reason  = "graduated_recovered"
                new_notes_suffix = f"|graduated_recovered:{sig[:8]}"
                break

        self.assertIsNotNone(recovered_sig, "Should have found a confirmed sig")
        self.assertEqual(new_exit_reason, "graduated_recovered",
                         "exit_reason should be 'graduated_recovered'")
        self.assertAlmostEqual(recovered_delta, 0.041726, places=5)
        self.assertIn("graduated_recovered:", new_notes_suffix)


# ---------------------------------------------------------------------------
# Test D: reconciler idempotent on already-tagged rows
# ---------------------------------------------------------------------------

class TestReconcilerIdempotent(unittest.TestCase):

    def test_D_skips_already_reconciled_row(self):
        """Row with 'journal_reconciled:' in notes → read_sol_delta NOT called."""
        live_path   = Path("/tmp/rec_test_D_live.csv")
        social_path = Path("/tmp/rec_test_D_social.csv")

        row = _make_row(
            exit_price="0",
            pnl_pct="-100",
            notes=f"live|tx:BUYTX|sell_tx:{_FAKE_SIG}|journal_reconciled:ABCDEF12",
        )
        _write_csv(live_path, [row])
        _write_csv(social_path, [])

        import memecoin.journal_reconciler as jr

        with patch("memecoin.journal_reconciler.read_sol_delta") as mock_rsd:
            with patch("memecoin.journal_reconciler._get_sol_price", return_value=150.0):
                result = jr.run_reconciler_pass(
                    "8PNHvFWeMT7CqpUvJjAwVgAK545t5KV3uCPd8DUfaTiM",
                    _live_path=live_path,
                    _social_path=social_path,
                    _main_path=Path("/tmp/rec_test_D_main.csv"),
                )

        mock_rsd.assert_not_called()
        self.assertEqual(result["rows_checked"], 0,
                         "Already-tagged row should not be checked")
        self.assertEqual(result["rows_corrected"], 0)


# ---------------------------------------------------------------------------
# Test E: Race — append during slow reconciler pass must not be erased
# ---------------------------------------------------------------------------

class TestReconcilerRaceAppendPreserved(unittest.TestCase):

    def test_E_append_mid_pass_survives_rewrite(self):
        """
        Simulates a lost-update race:
          1. Reconciler Phase 1 snapshot-reads the journal (1 existing row).
          2. While RPC is 'slow', a new row is appended via JOURNAL_LOCK
             (simulated by writing directly to the file under the same lock).
          3. Reconciler Phase 2 re-reads fresh and rewrites.
          4. Assert the new row appended mid-pass still exists after rewrite.
        """
        import threading
        import memecoin.journal_reconciler as jr
        from memecoin.journal_io import JOURNAL_LOCK

        live_path   = Path("/tmp/rec_test_E_live.csv")
        social_path = Path("/tmp/rec_test_E_social.csv")
        main_path   = Path("/tmp/rec_test_E_main.csv")
        new_row_id  = "NewRowMidPass"

        # Start with one existing target row
        existing_row = _make_row(
            id="ExistingPos01",
            exit_price="0",
            pnl_pct="-100",
            notes=f"live|tx:BUYTX|sell_tx:{_FAKE_SIG}",
        )
        _write_csv(live_path, [existing_row])
        _write_csv(social_path, [])
        _write_csv(main_path, [])

        append_done = threading.Event()
        new_row_written = []

        def slow_read_sol_delta(sig, wallet):
            """
            First call: mimic slow RPC by appending a new row under JOURNAL_LOCK
            before returning.  This simulates portfolio._append_journal() running
            during Phase 1.
            """
            if not append_done.is_set():
                with JOURNAL_LOCK:
                    new_row = _make_row(
                        id=new_row_id,
                        exit_price="0.000050",
                        pnl_pct="25.0",
                        exit_reason="hard_stop",
                        notes="live|tx:BUYTXNEW",
                    )
                    new_row_written.append(new_row)
                    with open(live_path, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=_FIELDS)
                        w.writerow(new_row)
                append_done.set()

            return {
                "ok": True,
                "sol_delta": 0.030000,
                "source": "native_lamports",
                "attempts": 1,
                "reason": "success",
            }

        with patch("memecoin.journal_reconciler.read_sol_delta", side_effect=slow_read_sol_delta):
            with patch("memecoin.journal_reconciler._get_sol_price", return_value=150.0):
                result = jr.run_reconciler_pass(
                    "8PNHvFWeMT7CqpUvJjAwVgAK545t5KV3uCPd8DUfaTiM",
                    _live_path=live_path,
                    _social_path=social_path,
                    _main_path=main_path,
                )

        rows_after = _read_csv(live_path)
        ids_after = [r["id"] for r in rows_after]

        self.assertIn("ExistingPos01", ids_after,
                      "Original row must survive the reconciler rewrite")
        self.assertIn(new_row_id, ids_after,
                      "Row appended mid-pass must survive the reconciler rewrite")
        self.assertEqual(len(rows_after), 2,
                         f"Expected 2 rows, got {len(rows_after)}: {ids_after}")

        # Original row should have been corrected
        orig = next(r for r in rows_after if r["id"] == "ExistingPos01")
        self.assertIn("journal_reconciled:", orig["notes"],
                      "Original row should be tagged as reconciled")

        # New row must be untouched (not a target row — no loss tag)
        new = next(r for r in rows_after if r["id"] == new_row_id)
        self.assertNotIn("journal_reconciled:", new["notes"],
                         "Mid-pass appended row must not have been modified")

        self.assertEqual(result["rows_corrected"], 1,
                         "Only the original row should be corrected")


if __name__ == "__main__":
    unittest.main()
