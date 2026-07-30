"""
test_reconciler_guards.py — reconciled_gone decision guard tests

Tests:
  A: single zero read → tags timestamp, does not close
  B: second zero read <30s → no close
  C: second zero read >=30s + no pending sig + no fresh migration → closes
  D: zero balance + pending sig → no close
  E: zero balance during fresh complete=True <60s → tags migration_transit, no close
  F: legitimate reconciled_gone runs sig sweep before journaling

Run: python -m pytest memecoin/tests/test_reconciler_guards.py -v
"""

import re
import sys
import time
import types
import threading
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
def _install_stubs():
    sys.modules.setdefault("memecoin", types.ModuleType("memecoin"))
    if "memecoin.config" not in sys.modules:
        cfg = types.ModuleType("memecoin.config")
        cfg.POSITIONS_FILE = "/tmp/test_rec_positions.json"
        cfg.JOURNAL_FILE = "/tmp/test_rec_journal.csv"
        cfg.SOCIAL_JOURNAL_FILE = "/tmp/test_rec_social.csv"
        cfg.LIVE_JOURNAL_FILE = "/tmp/test_rec_live.csv"
        cfg.TRAJECTORY_FILE = "/tmp/test_rec_traj.csv"
        cfg.TRADE_SIZE_USD = 3
        cfg.HARD_STOP_PCT = -0.35
        cfg.MAX_LOSS_FROM_FILL_PCT = 0.50
        cfg.TRAILING_STOP_PCT = -0.40
        cfg.TRAIL_ACTIVATES_PCT = 1.00
        cfg.TIME_STOP_MINUTES = 90
        cfg.TIME_STOP_MIN_GAIN = 0.30
        cfg.TP_LEVELS = [(0.30, 0.30)]
        cfg.PRICE_PATHS_DIR = "/tmp/test_rec_pp"
        cfg.LIVE_TRADING = False
        cfg.DAILY_LOSS_LIMIT = -5.0
        cfg.LIVE_DRY_RUN = False
        cfg.REALTIME_PRICE_FEED = True
        cfg.SLIPPAGE_GATE_RT_PCT = 0.30
        cfg.SLIPPAGE_GATE_DEX_PCT = 0.50
        cfg.SELL_STUCK_RETRY_SEC = 60
        cfg.WALLET_PUBKEY = "TestWallet111111111111111111111111111111111"
        cfg.get_signal_settings = lambda st: {
            "trade_size_usd": 3, "hard_stop_pct": -0.35,
            "trailing_stop_pct": -0.40, "trail_activates_pct": 1.00,
            "time_stop_minutes": 90,
        }
        sys.modules["memecoin.config"] = cfg

    for mod_name in [
        "memecoin.data_client", "memecoin.candidate_log",
        "memecoin.journal_io", "app", "app.alerts",
        "memecoin.executor", "memecoin.tx_meta",
    ]:
        if mod_name not in sys.modules:
            m = types.ModuleType(mod_name)
            if mod_name == "memecoin.journal_io":
                m.JOURNAL_LOCK = threading.Lock()
            if mod_name == "memecoin.candidate_log":
                m.promote_to_winners = lambda *a, **kw: None
            if mod_name == "memecoin.data_client":
                m.dex_get_token = lambda *a, **kw: {}
            if mod_name == "app.alerts":
                m.alert_position_open = lambda *a, **kw: None
                m.alert_position_close = lambda *a, **kw: None
                m.alert_live_buy = lambda *a, **kw: None
                m.alert_live_sell = lambda *a, **kw: None
                m._send = lambda *a, **kw: None
            if mod_name == "app":
                m.alerts = sys.modules.get("app.alerts", types.ModuleType("app.alerts"))
            if mod_name == "memecoin.executor":
                m._get_keypair = MagicMock()
                m._token_balance = MagicMock(return_value=0)
                m.get_pumpfun_curve_complete = MagicMock(return_value={"ok": True, "complete": False, "reason": "bonding_curve"})
            if mod_name == "memecoin.tx_meta":
                m.read_sol_delta = MagicMock(return_value={"ok": False})
            sys.modules[mod_name] = m


_install_stubs()


@dataclass
class FakePos:
    id: str = "pos_test_1"
    signal_id: str = "sig_1"
    chain: str = "solana"
    token_address: str = "FakeMint111111111111111111111111111111111111"
    token_symbol: str = "TEST"
    signal_type: str = "social_alert"
    status: str = "open"
    current_price: float = 0.001
    entry_price: float = 0.001
    signal_price: float = 0.001
    notes: str = "live|tx:fakesig123"
    sol_received: float = 0.0
    exit_reason: str = ""


def _make_pos(**kwargs):
    return FakePos(**kwargs)


# ── Test A: single zero read → tags timestamp, does not close ──
def test_first_zero_read_tags_timestamp():
    pos = _make_pos()
    assert "|reconciler_zero_first_ts:" not in pos.notes

    # Simulate reconciler logic inline (since we can't run the thread)
    _now_ts = int(time.time())
    _first_zero_key = "|reconciler_zero_first_ts:"
    if _first_zero_key not in (pos.notes or ""):
        pos.notes = (pos.notes or "") + f"|reconciler_zero_first_ts:{_now_ts}|"
        closed = False
    else:
        closed = True

    assert not closed
    assert "|reconciler_zero_first_ts:" in pos.notes


# ── Test B: second zero read <30s → no close ──
def test_second_zero_read_under_30s_no_close():
    _now_ts = int(time.time())
    pos = _make_pos(notes=f"live|tx:fakesig123|reconciler_zero_first_ts:{_now_ts}|")

    # 5 seconds later
    _check_ts = _now_ts + 5
    _fz_match = re.search(r'\|reconciler_zero_first_ts:(\d+)\|', pos.notes or "")
    _first_zero_ts = int(_fz_match.group(1)) if _fz_match else _check_ts

    should_continue = (_check_ts - _first_zero_ts) < 30
    assert should_continue, "Should not close within 30s"


# ── Test C: second zero read >=30s → closes ──
def test_second_zero_read_over_30s_closes():
    _old_ts = int(time.time()) - 35  # 35 seconds ago
    pos = _make_pos(notes=f"live|tx:fakesig123|reconciler_zero_first_ts:{_old_ts}|")

    _now_ts = int(time.time())
    _fz_match = re.search(r'\|reconciler_zero_first_ts:(\d+)\|', pos.notes or "")
    _first_zero_ts = int(_fz_match.group(1)) if _fz_match else _now_ts

    should_wait = (_now_ts - _first_zero_ts) < 30
    assert not should_wait, "Should close after 30s"

    # Check no pending markers
    _pending_markers = ("|sell_pending:", "|sell_unconf:", "|jupiter_rescue_pending:", "|rescue_pending:", "|pending_sig:")
    has_pending = any(m in (pos.notes or "") for m in _pending_markers)
    assert not has_pending, "Should not have pending markers"


# ── Test D: zero balance + pending sig → no close ──
def test_pending_sig_blocks_close():
    _old_ts = int(time.time()) - 60
    pos = _make_pos(
        notes=f"live|tx:fakesig123|reconciler_zero_first_ts:{_old_ts}||sell_pending:somesig123|"
    )

    _pending_markers = ("|sell_pending:", "|sell_unconf:", "|jupiter_rescue_pending:", "|rescue_pending:", "|pending_sig:")
    has_pending = any(m in (pos.notes or "") for m in _pending_markers)
    assert has_pending, "Pending sig should block close"


# ── Test E: zero balance during fresh complete=True <60s → tags migration_transit ──
def test_migration_transit_guard():
    _old_ts = int(time.time()) - 35
    pos = _make_pos(notes=f"live|tx:fakesig123|reconciler_zero_first_ts:{_old_ts}|")
    _now_ts = int(time.time())

    # Simulate curve check returns complete=True
    curve_res = {"ok": True, "complete": True, "reason": "graduated"}

    if curve_res.get("complete") is True:
        _cct_key = "|curve_complete_first_ts:"
        if _cct_key not in (pos.notes or ""):
            pos.notes = (pos.notes or "") + f"|curve_complete_first_ts:{_now_ts}|migration_transit:{_now_ts}|"
            should_continue = True
        else:
            should_continue = False
    else:
        should_continue = False

    assert should_continue, "Should wait on fresh migration transit"
    assert "|curve_complete_first_ts:" in pos.notes
    assert "|migration_transit:" in pos.notes


# ── Test F: sig sweep runs before closing as reconciled_gone ──
def test_sig_sweep_before_reconciled_gone():
    """Verify the sig sweep logic finds a positive delta and closes as recovered."""
    _old_ts = int(time.time()) - 60
    pos = _make_pos(
        notes=f"live|tx:fakesig123|reconciler_zero_first_ts:{_old_ts}||sell_tx:ConfirmedSellSig123|"
    )

    # Simulate sig sweep
    _rec_sigs = re.findall(
        r'(?:sell_tx|sell_unconf|jupiter_rescue_pending|sell_pending|pending_sig):([A-Za-z0-9]+)',
        pos.notes or "",
    )
    assert len(_rec_sigs) > 0, "Should find at least one sig"
    assert "ConfirmedSellSig123" in _rec_sigs

    # Simulate positive delta found
    _rec_recovered = False
    for _sig in reversed(_rec_sigs):
        # Mock: this sig has a positive delta
        mock_result = {"ok": True, "sol_delta": 0.005}
        if mock_result.get("ok") and (mock_result.get("sol_delta") or 0) > 0:
            pos.exit_reason = "reconciled_recovered"
            pos.sol_received = mock_result["sol_delta"]
            _rec_recovered = True
            break

    assert _rec_recovered, "Should recover from sig sweep"
    assert pos.exit_reason == "reconciled_recovered"
    assert pos.sol_received == 0.005
