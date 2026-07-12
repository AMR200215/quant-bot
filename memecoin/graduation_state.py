"""
graduation_state.py — Pure functions for pre-graduation progress observation.

Replaces B1/B2 inline logic in scanner.py with a clean, testable API.

Public API:
    ProgressObservation  — frozen dataclass
    select_progress_observation(pp_obs, curve_obs, now_monotonic, freshness_sec) → ProgressObservation | None
    decide_graduation_action(obs, grad_sol_ui, pregrad_trigger_pct) → GraduationAction
"""

import enum
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ProgressObservation:
    """Snapshot of bonding curve progress at a point in time."""
    mint: str
    source: str              # "pp" | "curve"
    sample_monotonic: float  # time.monotonic() when sample was taken
    vsol_ui: float           # virtual SOL reserves in UI units (not lamports)
    complete: bool           # True if oracle says bonding curve complete
    reason: str              # "normal" | "account_missing" | "curve_complete" | "pp_complete" etc.
    price_usd: float         # 0.0 if unknown


class GraduationAction(enum.Enum):
    NONE = "none"                    # below trigger threshold, nothing to do
    PRE_GRAD_EXIT = "pre_grad_exit"  # vSOL crossed threshold, close pre-graduation
    GRAD_EXIT = "grad_exit"          # oracle confirmed complete=True
    ALREADY_GONE = "already_gone"    # account_missing


def select_progress_observation(
    pp_obs: Optional[ProgressObservation],
    curve_obs: Optional[ProgressObservation],
    now_monotonic: float,
    freshness_sec: float = 5.0,
) -> Optional[ProgressObservation]:
    """
    Return the most useful fresh observation.

    Priority:
    1. pp_obs if fresh (within freshness_sec) AND vsol_ui > 0
    2. curve_obs if fresh
    3. None

    Never raises.
    """
    try:
        if (
            pp_obs is not None
            and (now_monotonic - pp_obs.sample_monotonic) <= freshness_sec
            and pp_obs.vsol_ui > 0
        ):
            return pp_obs

        if (
            curve_obs is not None
            and (now_monotonic - curve_obs.sample_monotonic) <= freshness_sec
        ):
            return curve_obs

        return None
    except Exception:
        return None


def decide_graduation_action(
    obs: ProgressObservation,
    grad_sol_ui: float,
    pregrad_trigger_pct: float,
) -> GraduationAction:
    """
    Decide what action to take given a progress observation.

    Args:
        obs: ProgressObservation (caller must ensure not None)
        grad_sol_ui: graduation threshold in SOL UI (e.g. 115.0)
        pregrad_trigger_pct: fraction of grad_sol_ui that triggers pre-grad exit (e.g. 0.85)

    Returns:
        GraduationAction
    """
    if obs.reason == "account_missing":
        return GraduationAction.ALREADY_GONE

    if obs.complete:
        return GraduationAction.GRAD_EXIT

    if obs.vsol_ui >= grad_sol_ui * pregrad_trigger_pct:
        return GraduationAction.PRE_GRAD_EXIT

    return GraduationAction.NONE
