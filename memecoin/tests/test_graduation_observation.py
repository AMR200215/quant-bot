"""
Tests for graduation_state.py — 10 tests covering select_progress_observation
and decide_graduation_action.

R6: all tests import and call production functions directly; no inline reimplementation.
"""

import time
import pytest

from memecoin.graduation_state import (
    GraduationAction,
    ProgressObservation,
    decide_graduation_action,
    select_progress_observation,
)

MINT = "So11111111111111111111111111111111111111112"
NOW = time.monotonic()
GRAD_SOL = 115.0
TRIGGER_PCT = 0.85
TRIGGER_SOL = GRAD_SOL * TRIGGER_PCT  # 97.75


def _make_obs(source="pp", age=0.0, vsol_ui=50.0, complete=False, reason="normal"):
    return ProgressObservation(
        mint=MINT,
        source=source,
        sample_monotonic=NOW - age,
        vsol_ui=vsol_ui,
        complete=complete,
        reason=reason,
        price_usd=0.0,
    )


# ---------------------------------------------------------------------------
# select_progress_observation
# ---------------------------------------------------------------------------

def test_pp_preferred_over_curve_when_both_fresh():
    """Test 1: PP fresh observation preferred over curve."""
    pp = _make_obs(source="pp", age=1.0, vsol_ui=60.0)
    curve = _make_obs(source="curve", age=1.0, vsol_ui=55.0)
    result = select_progress_observation(pp, curve, NOW, freshness_sec=5.0)
    assert result is pp


def test_stale_pp_falls_back_to_curve():
    """Test 2: Stale PP observation (>freshness_sec) → falls back to curve."""
    pp = _make_obs(source="pp", age=10.0, vsol_ui=60.0)
    curve = _make_obs(source="curve", age=1.0, vsol_ui=55.0)
    result = select_progress_observation(pp, curve, NOW, freshness_sec=5.0)
    assert result is curve


def test_pp_vsol_zero_treated_as_absent():
    """Test 3: PP with vsol_ui=0 treated as absent → falls back to curve."""
    pp = _make_obs(source="pp", age=1.0, vsol_ui=0.0)
    curve = _make_obs(source="curve", age=1.0, vsol_ui=55.0)
    result = select_progress_observation(pp, curve, NOW, freshness_sec=5.0)
    assert result is curve


def test_both_none_returns_none():
    """Test 4: Both None → select returns None."""
    result = select_progress_observation(None, None, NOW, freshness_sec=5.0)
    assert result is None


def test_both_stale_returns_none():
    """Test 5: Both stale → select returns None."""
    pp = _make_obs(source="pp", age=10.0, vsol_ui=60.0)
    curve = _make_obs(source="curve", age=10.0, vsol_ui=55.0)
    result = select_progress_observation(pp, curve, NOW, freshness_sec=5.0)
    assert result is None


# ---------------------------------------------------------------------------
# decide_graduation_action
# ---------------------------------------------------------------------------

def test_decide_none_when_vsol_below_threshold():
    """Test 6: Returns NONE when vSOL below threshold."""
    obs = _make_obs(vsol_ui=50.0, complete=False, reason="normal")
    action = decide_graduation_action(obs, GRAD_SOL, TRIGGER_PCT)
    assert action is GraduationAction.NONE


def test_decide_pregrad_exit_at_exactly_trigger_pct():
    """Test 7: Returns PRE_GRAD_EXIT when vSOL at exactly trigger pct."""
    obs = _make_obs(vsol_ui=TRIGGER_SOL, complete=False, reason="normal")
    action = decide_graduation_action(obs, GRAD_SOL, TRIGGER_PCT)
    assert action is GraduationAction.PRE_GRAD_EXIT


def test_decide_pregrad_exit_above_threshold_not_complete():
    """Test 8: Returns PRE_GRAD_EXIT when vSOL above threshold but complete=False."""
    obs = _make_obs(vsol_ui=TRIGGER_SOL + 5.0, complete=False, reason="normal")
    action = decide_graduation_action(obs, GRAD_SOL, TRIGGER_PCT)
    assert action is GraduationAction.PRE_GRAD_EXIT


def test_decide_grad_exit_when_complete_true():
    """Test 9: Returns GRAD_EXIT when complete=True regardless of vsol_ui."""
    obs = _make_obs(vsol_ui=10.0, complete=True, reason="curve_complete")
    action = decide_graduation_action(obs, GRAD_SOL, TRIGGER_PCT)
    assert action is GraduationAction.GRAD_EXIT


def test_decide_already_gone_when_account_missing():
    """Test 10: Returns ALREADY_GONE when reason='account_missing'."""
    obs = _make_obs(vsol_ui=0.0, complete=False, reason="account_missing")
    action = decide_graduation_action(obs, GRAD_SOL, TRIGGER_PCT)
    assert action is GraduationAction.ALREADY_GONE
