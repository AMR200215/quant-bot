"""
exit_orchestrator.py — Single-venue exit controller.

Enforces R3/R4/R5:
  R3: venue failures never cross-contaminate
  R4: exactly one outbound call per orchestration step
  R5: SENT_PENDING blocks all venues until resolved

Public API:
    RouteOutcome   — enum
    RouteResult    — dataclass
    VenueState     — dataclass (per-position per-venue state)
    ExitOrchestrator — main class

Usage:
    orch = ExitOrchestrator(pos_id)
    result = orch.dispatch(venue, executor_fn, *args, **kwargs)
"""

import dataclasses
import enum
import threading
import time


class RouteOutcome(enum.Enum):
    NO_SEND           = "no_send"            # not attempted (pre-send block, cooldown)
    SENT_PENDING      = "sent_pending"       # sig sent, not yet confirmed/reverted
    CONFIRMED_SUCCESS = "confirmed_success"  # on-chain confirmed, tokens sold
    CONFIRMED_REVERT  = "confirmed_revert"   # on-chain revert (Custom:6001 etc)
    ZERO_BALANCE      = "zero_balance"       # tokens already gone, nothing to sell
    FATAL_PRE_SEND    = "fatal_pre_send"     # pre-send error (build failed, no keypair)


@dataclasses.dataclass
class RouteResult:
    outcome: RouteOutcome
    venue: str              # "pump_amm" | "jupiter" | "bc_t22" | "pumpportal_bc"
    sig: str | None = None
    fill_price: float | None = None
    sol_received: float | None = None
    error: str | None = None
    attempt_num: int = 1
    elapsed_ms: float = 0.0


@dataclasses.dataclass
class VenueState:
    venue: str
    attempts: int = 0
    last_outcome: RouteOutcome | None = None
    cooldown_until: float = 0.0      # monotonic timestamp
    pending_sig: str | None = None   # if SENT_PENDING, the unresolved sig


def _map_executor_result(result: dict) -> RouteOutcome:
    """Map executor result dict to RouteOutcome.

    Mapping rules (in priority order):
      - success=True                                    → CONFIRMED_SUCCESS
      - error_class == "pending" OR (sig present and success=False) → SENT_PENDING
      - error_class in ("revert", "custom_6001", "custom_error")   → CONFIRMED_REVERT
      - error_class in ("no_route", "no_liquidity")                → NO_SEND  (R3)
      - error_class == "zero_balance"                              → ZERO_BALANCE
      - error_class in ("build_failed", "no_keypair", "fatal")     → FATAL_PRE_SEND
      - otherwise                                                  → NO_SEND
    """
    success = result.get("success", False)
    error_class = result.get("error_class")
    sig = result.get("sig")

    if success:
        return RouteOutcome.CONFIRMED_SUCCESS

    if error_class == "pending" or (sig and not success):
        return RouteOutcome.SENT_PENDING

    if error_class in ("revert", "custom_6001", "custom_error"):
        return RouteOutcome.CONFIRMED_REVERT

    if error_class in ("no_route", "no_liquidity"):
        # R3: Jupiter no_route must never block pump-amm or bonding curve.
        # Return NO_SEND so the failure stays scoped to this venue only.
        return RouteOutcome.NO_SEND

    if error_class == "zero_balance":
        return RouteOutcome.ZERO_BALANCE

    if error_class in ("build_failed", "no_keypair", "fatal"):
        return RouteOutcome.FATAL_PRE_SEND

    return RouteOutcome.NO_SEND


class ExitOrchestrator:
    """Single-venue exit controller enforcing R3/R4/R5."""

    def __init__(self, pos_id: str):
        self._pos_id = pos_id
        self._venue_states: dict[str, VenueState] = {}
        self._lock = threading.Lock()
        self._global_pending_sig: str | None = None  # R5 sentinel

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dispatch(self, venue: str, executor_fn, *args, **kwargs) -> RouteResult:
        """Single-venue dispatch with R3/R4/R5 enforcement.

        R4: only one outbound call per invocation.
        R5: if _global_pending_sig is set, returns NO_SEND immediately.

        executor_fn(*args, **kwargs) must return a dict with keys:
            success: bool
            sig: str | None
            error_class: str | None  ("revert" | "no_route" | "pending" | ...)
            fill_price: float | None
            sol_received: float | None

        Maps executor result → RouteOutcome.
        If outcome is SENT_PENDING, sets _global_pending_sig.
        If outcome is CONFIRMED_SUCCESS/REVERT/ZERO_BALANCE, clears _global_pending_sig.
        """
        with self._lock:
            # R5: block all venues if there is an unresolved pending sig
            if self._global_pending_sig is not None:
                state = self._get_or_create_venue_state(venue)
                return RouteResult(
                    outcome=RouteOutcome.NO_SEND,
                    venue=venue,
                    attempt_num=state.attempts,
                    error="blocked: global_pending_sig unresolved (R5)",
                )

            state = self._get_or_create_venue_state(venue)

        # R4: exactly one outbound call — invoke executor_fn outside the lock
        # so other threads are not blocked during network I/O, but the
        # state mutation that follows re-acquires the lock.
        t_start = time.monotonic()
        try:
            exec_result = executor_fn(*args, **kwargs)
        except Exception as exc:
            exec_result = {
                "success": False,
                "sig": None,
                "error_class": "fatal",
                "fill_price": None,
                "sol_received": None,
            }
            error_str = str(exc)
        else:
            error_str = exec_result.get("error_class")

        elapsed_ms = (time.monotonic() - t_start) * 1000.0
        outcome = _map_executor_result(exec_result)
        sig = exec_result.get("sig")

        with self._lock:
            state = self._get_or_create_venue_state(venue)
            state.attempts += 1
            state.last_outcome = outcome

            if outcome == RouteOutcome.SENT_PENDING:
                # R5: lock all other venues until this sig resolves
                self._global_pending_sig = sig
                state.pending_sig = sig
            elif outcome in (
                RouteOutcome.CONFIRMED_SUCCESS,
                RouteOutcome.CONFIRMED_REVERT,
                RouteOutcome.ZERO_BALANCE,
            ):
                # Terminal outcome — clear any pending sentinel
                self._global_pending_sig = None
                state.pending_sig = None

            return RouteResult(
                outcome=outcome,
                venue=venue,
                sig=sig,
                fill_price=exec_result.get("fill_price"),
                sol_received=exec_result.get("sol_received"),
                error=error_str if not exec_result.get("success") else None,
                attempt_num=state.attempts,
                elapsed_ms=elapsed_ms,
            )

    def get_venue_state(self, venue: str) -> VenueState:
        """Return VenueState for venue, creating default if absent."""
        with self._lock:
            return self._get_or_create_venue_state(venue)

    def pending_sig(self) -> str | None:
        """Return the unresolved global pending sig, or None."""
        with self._lock:
            return self._global_pending_sig

    def clear_pending(self) -> None:
        """Clear the global pending sig (call after sig confirmed/reverted)."""
        with self._lock:
            self._global_pending_sig = None

    def venue_attempts(self, venue: str) -> int:
        """Return attempt count for venue."""
        with self._lock:
            state = self._venue_states.get(venue)
            return state.attempts if state else 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create_venue_state(self, venue: str) -> VenueState:
        """Must be called with self._lock held."""
        if venue not in self._venue_states:
            self._venue_states[venue] = VenueState(venue=venue)
        return self._venue_states[venue]
