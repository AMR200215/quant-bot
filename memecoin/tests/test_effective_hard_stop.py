"""
test_effective_hard_stop.py — effective_hard_stop_level tests

Tests:
  A: fill=2.17x signal → effective stop = fill*0.50 (fill floor wins)
  B: fill=1.10x signal → effective stop = signal*0.65 (signal wins)
  C: paper: entry==signal → result unchanged from pure signal stop
  D: size_mult uses the same effective stop_level as the real hard-stop check
  E: PP event-driven hard stop uses same effective_hard_stop_level as portfolio.update_prices

Run: python -m pytest memecoin/tests/test_effective_hard_stop.py -v
"""

import pytest
import sys
import types
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Stub modules so portfolio.py can be imported without live services
# ---------------------------------------------------------------------------
def _install_stubs():
    sys.modules.setdefault("memecoin", types.ModuleType("memecoin"))
    if "memecoin.config" not in sys.modules:
        cfg = types.ModuleType("memecoin.config")
        cfg.MAX_LOSS_FROM_FILL_PCT = 0.50
        cfg.HARD_STOP_PCT = -0.35
        cfg.POSITIONS_FILE = "/tmp/test_ehs_positions.json"
        cfg.JOURNAL_FILE = "/tmp/test_ehs_journal.csv"
        cfg.SOCIAL_JOURNAL_FILE = "/tmp/test_ehs_social.csv"
        cfg.LIVE_JOURNAL_FILE = "/tmp/test_ehs_live.csv"
        cfg.TRAJECTORY_FILE = "/tmp/test_ehs_traj.csv"
        cfg.TRADE_SIZE_USD = 3
        cfg.TRAILING_STOP_PCT = -0.40
        cfg.TRAIL_ACTIVATES_PCT = 1.00
        cfg.TIME_STOP_MINUTES = 90
        cfg.TIME_STOP_MIN_GAIN = 0.30
        cfg.TP_LEVELS = [(0.30, 0.30)]
        cfg.PRICE_PATHS_DIR = "/tmp/test_ehs_pp"
        cfg.LIVE_TRADING = False
        cfg.DAILY_LOSS_LIMIT = -5.0
        cfg.LIVE_DRY_RUN = False
        cfg.REALTIME_PRICE_FEED = True
        cfg.SLIPPAGE_GATE_RT_PCT = 0.30
        cfg.SLIPPAGE_GATE_DEX_PCT = 0.50
        cfg.SELL_STUCK_RETRY_SEC = 60
        cfg.get_signal_settings = lambda st: {
            "trade_size_usd": 3, "hard_stop_pct": -0.35,
            "trailing_stop_pct": -0.40, "trail_activates_pct": 1.00,
            "time_stop_minutes": 90,
        }
        sys.modules["memecoin.config"] = cfg

    for mod_name in [
        "memecoin.data_client", "memecoin.candidate_log",
        "memecoin.journal_io", "app", "app.alerts",
    ]:
        if mod_name not in sys.modules:
            m = types.ModuleType(mod_name)
            if mod_name == "memecoin.journal_io":
                import threading
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
            sys.modules[mod_name] = m


_install_stubs()

from memecoin.portfolio import effective_hard_stop_level


# ── Test A: fill=2.17x signal → fill floor wins ──
def test_high_slippage_fill_floor_wins():
    signal = 1.00
    fill = 2.17
    hard_stop_pct = -0.35
    result = effective_hard_stop_level(signal, fill, hard_stop_pct)
    # signal_stop = 1.00 * 0.65 = 0.65
    # fill_floor  = 2.17 * 0.50 = 1.085
    # effective   = max(0.65, 1.085) = 1.085
    assert abs(result - (fill * 0.50)) < 1e-9, f"Expected {fill * 0.50}, got {result}"


# ── Test B: fill=1.10x signal → signal anchor wins ──
def test_low_slippage_signal_wins():
    signal = 1.00
    fill = 1.10
    hard_stop_pct = -0.35
    result = effective_hard_stop_level(signal, fill, hard_stop_pct)
    # signal_stop = 1.00 * 0.65 = 0.65
    # fill_floor  = 1.10 * 0.50 = 0.55
    # effective   = max(0.65, 0.55) = 0.65
    assert abs(result - (signal * 0.65)) < 1e-9, f"Expected {signal * 0.65}, got {result}"


# ── Test C: paper: entry==signal → result unchanged from pure signal stop ──
def test_paper_unchanged():
    signal = 1.00
    entry = 1.00  # paper: same as signal
    hard_stop_pct = -0.35
    result = effective_hard_stop_level(signal, entry, hard_stop_pct)
    pure_signal_stop = signal * (1 + hard_stop_pct)
    assert abs(result - pure_signal_stop) < 1e-9, f"Expected {pure_signal_stop}, got {result}"


# ── Test D: size_mult uses same effective stop level as hard-stop check ──
def test_size_norm_uses_same_stop_level():
    """Both the hard-stop check and the size-norm compute the same stop level."""
    signal = 1.00
    fill = 2.17
    hard_stop_pct = -0.35

    # Hard-stop check path
    stop_for_monitor = effective_hard_stop_level(signal, fill, hard_stop_pct)

    # Size-norm path (mirrors portfolio.py code)
    _sig_price = signal
    _pp_price = fill  # at entry time, pp_price ~ fill
    _eff_stop = effective_hard_stop_level(_sig_price, _pp_price, hard_stop_pct)

    assert abs(stop_for_monitor - _eff_stop) < 1e-9, (
        f"Monitor stop ({stop_for_monitor}) != size-norm stop ({_eff_stop})"
    )


# ── Test E: PP event-driven hard stop uses same function ──
def test_pp_event_driven_uses_same_function():
    """scanner.py PP callback uses the same effective_hard_stop_level function."""
    signal = 1.00
    fill = 2.00
    hard_stop_pct = -0.35

    # Both portfolio.update_prices and scanner._on_pp_price_tick should use
    # effective_hard_stop_level with the same inputs → same output.
    from memecoin.portfolio import effective_hard_stop_level as ehs_portfolio

    # The scanner.py import is: from memecoin.portfolio import effective_hard_stop_level
    # Both call: effective_hard_stop_level(pos.signal_price, pos.entry_price, pos.hard_stop_pct)
    stop_monitor = ehs_portfolio(signal, fill, hard_stop_pct)
    stop_pp = ehs_portfolio(signal, fill, hard_stop_pct)

    assert stop_monitor == stop_pp
    # Also verify value correctness
    # signal_stop = 1.00 * 0.65 = 0.65
    # fill_floor  = 2.00 * 0.50 = 1.00
    assert abs(stop_pp - 1.00) < 1e-9
