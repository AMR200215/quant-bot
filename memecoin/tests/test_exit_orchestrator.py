"""
test_exit_orchestrator.py — 9 tests for ExitOrchestrator (R3/R4/R5).

R6: calls ExitOrchestrator production methods directly.
"""

import pytest

from memecoin.exit_orchestrator import (
    ExitOrchestrator,
    RouteOutcome,
    RouteResult,
    VenueState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_executor(
    success=False,
    sig=None,
    error_class=None,
    fill_price=None,
    sol_received=None,
):
    """Return a zero-argument callable that returns the given executor dict."""
    def _fn(*args, **kwargs):
        return {
            "success": success,
            "sig": sig,
            "error_class": error_class,
            "fill_price": fill_price,
            "sol_received": sol_received,
        }
    return _fn


# ---------------------------------------------------------------------------
# Test 1: CONFIRMED_SUCCESS when executor returns success=True
# ---------------------------------------------------------------------------

def test_dispatch_confirmed_success():
    orch = ExitOrchestrator(pos_id="pos-001")
    fn = make_executor(success=True, sig="abc123", fill_price=0.0042, sol_received=0.5)
    result = orch.dispatch("pump_amm", fn)
    assert result.outcome == RouteOutcome.CONFIRMED_SUCCESS
    assert result.venue == "pump_amm"
    assert result.sig == "abc123"
    assert result.fill_price == pytest.approx(0.0042)
    assert result.sol_received == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Test 2: CONFIRMED_REVERT when error_class="revert"
# ---------------------------------------------------------------------------

def test_dispatch_confirmed_revert():
    orch = ExitOrchestrator(pos_id="pos-002")
    fn = make_executor(success=False, error_class="revert")
    result = orch.dispatch("pump_amm", fn)
    assert result.outcome == RouteOutcome.CONFIRMED_REVERT


# ---------------------------------------------------------------------------
# Test 3: dispatch sets _global_pending_sig when outcome is SENT_PENDING
# ---------------------------------------------------------------------------

def test_dispatch_sets_global_pending_sig_on_sent_pending():
    orch = ExitOrchestrator(pos_id="pos-003")
    fn = make_executor(success=False, error_class="pending", sig="sig-xyz")
    result = orch.dispatch("jupiter", fn)
    assert result.outcome == RouteOutcome.SENT_PENDING
    assert orch.pending_sig() == "sig-xyz"


# ---------------------------------------------------------------------------
# Test 4: dispatch returns NO_SEND when _global_pending_sig already set (R5)
# ---------------------------------------------------------------------------

def test_dispatch_blocked_by_global_pending_sig_r5():
    orch = ExitOrchestrator(pos_id="pos-004")
    # First call sets pending
    fn_pending = make_executor(success=False, error_class="pending", sig="sig-pending")
    orch.dispatch("jupiter", fn_pending)
    assert orch.pending_sig() == "sig-pending"

    # Second call on a different venue must be blocked
    fn_second = make_executor(success=True)
    result = orch.dispatch("pump_amm", fn_second)
    assert result.outcome == RouteOutcome.NO_SEND
    assert "R5" in (result.error or "")


# ---------------------------------------------------------------------------
# Test 5: error_class="no_route" returns NO_SEND, not CONFIRMED_REVERT (R3)
# ---------------------------------------------------------------------------

def test_no_route_returns_no_send_r3():
    orch = ExitOrchestrator(pos_id="pos-005")
    fn = make_executor(success=False, error_class="no_route")
    result = orch.dispatch("jupiter", fn)
    assert result.outcome == RouteOutcome.NO_SEND
    # Ensure the global pending sig was NOT set (failure must not cross-contaminate)
    assert orch.pending_sig() is None


# ---------------------------------------------------------------------------
# Test 6: clear_pending() clears the sig; subsequent dispatch is not blocked
# ---------------------------------------------------------------------------

def test_clear_pending_unblocks_dispatch():
    orch = ExitOrchestrator(pos_id="pos-006")
    fn_pending = make_executor(success=False, error_class="pending", sig="sig-block")
    orch.dispatch("jupiter", fn_pending)
    assert orch.pending_sig() == "sig-block"

    # Clear the pending sig externally (e.g. after polling confirms/reverts)
    orch.clear_pending()
    assert orch.pending_sig() is None

    # Now a new dispatch on another venue must proceed normally
    fn_success = make_executor(success=True, sig="sig-new")
    result = orch.dispatch("pump_amm", fn_success)
    assert result.outcome == RouteOutcome.CONFIRMED_SUCCESS


# ---------------------------------------------------------------------------
# Test 7: venue_attempts increments per venue independently
# ---------------------------------------------------------------------------

def test_venue_attempts_increment_independently():
    orch = ExitOrchestrator(pos_id="pos-007")
    fn = make_executor(success=False, error_class="revert")

    assert orch.venue_attempts("pump_amm") == 0
    assert orch.venue_attempts("jupiter") == 0

    orch.dispatch("pump_amm", fn)
    orch.dispatch("pump_amm", fn)
    orch.dispatch("jupiter", fn)

    assert orch.venue_attempts("pump_amm") == 2
    assert orch.venue_attempts("jupiter") == 1
    assert orch.venue_attempts("bc_t22") == 0


# ---------------------------------------------------------------------------
# Test 8: FATAL_PRE_SEND for error_class="build_failed"
# ---------------------------------------------------------------------------

def test_fatal_pre_send_for_build_failed():
    orch = ExitOrchestrator(pos_id="pos-008")
    fn = make_executor(success=False, error_class="build_failed")
    result = orch.dispatch("pump_amm", fn)
    assert result.outcome == RouteOutcome.FATAL_PRE_SEND
    # A pre-send fatal must not set a global pending sig
    assert orch.pending_sig() is None


# ---------------------------------------------------------------------------
# Test 9: ZERO_BALANCE for error_class="zero_balance"
# ---------------------------------------------------------------------------

def test_zero_balance_outcome():
    orch = ExitOrchestrator(pos_id="pos-009")
    fn = make_executor(success=False, error_class="zero_balance")
    result = orch.dispatch("bc_t22", fn)
    assert result.outcome == RouteOutcome.ZERO_BALANCE
    assert orch.pending_sig() is None
