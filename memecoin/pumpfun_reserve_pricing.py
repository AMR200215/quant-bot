"""
memecoin/pumpfun_reserve_pricing.py — the ONE canonical function for
computing a pump.fun bonding-curve price from PumpPortal's real-time
reserve fields (vSolInBondingCurve / vTokensInBondingCurve).

ROOT CAUSE (V8 DATA RECOVERY batch, 2026-08-19, proven live, not
assumed): research/peak_tracker.py's _price_from_msg and memecoin/
pumpportal_monitor.py's _compute_price both computed
`(vsol / (vtok / 1e6)) * sol_price` -- an erroneous /1e6 applied to
vTokensInBondingCurve, on the belief that PumpPortal delivers this
field in raw (undecimaled) base units the way the on-chain bonding-
curve ACCOUNT does (memecoin/executor.py's get_pumpfun_curve_price,
which correctly divides raw on-chain virtual_token_reserves by 1e6).

PROVEN WRONG by direct live capture (8 real subscribeNewToken +
subscribeTokenTrade "create" events, VPS, 2026-08-19):
  - A brand-new "create" event showed vTokensInBondingCurve=
    1069434660.764585 -- matching pump.fun's PUBLICLY KNOWN initial
    virtual token reserve (~1,073,000,000 UI tokens) almost exactly,
    decremented by the tiny initial buy. If this field were raw base
    units, it would read ~1.07e15, not ~1.07e9.
  - A near-graduation event showed vSolInBondingCurve=115.005359...,
    vTokensInBondingCurve=279900000 -- EXACTLY matching
    1,073,000,000 - 793,100,000 = 279,900,000, the theoretical minimum
    virtual token reserve at the graduation instant (using pump.fun's
    own public real-tokens-sold-at-graduation constant). This is
    non-coincidental, exact confirmation of two things at once: (1)
    vTokensInBondingCurve arrives in UI units, needing NO conversion,
    and (2) research/v8_path_integrity.py's PUMPFUN_* protocol
    constants are independently, empirically correct.

The fix: vTokensInBondingCurve needs NO /1e6 -- PumpPortal delivers
BOTH vSolInBondingCurve and vTokensInBondingCurve already in
human-readable UI units, exactly like tokenAmount/solAmount on regular
trade messages (peak_tracker.py's own docstring already correctly
identified THOSE as UI units -- it just wrongly treated the reserve
fields as a different, raw-unit convention with no supporting evidence).

This function is the ONLY place in the repo permitted to compute a
price from PumpPortal reserve fields. research/peak_tracker.py and
memecoin/pumpportal_monitor.py both now call this instead of keeping
their own copies -- the exact drift this batch was asked to make
structurally impossible.
"""

from __future__ import annotations

from typing import Optional

# GRAD_SOL_UI (research/config.py) already represents the FULL virtual
# SOL reserve at graduation, not an incremental "real SOL added" amount
# on top of a separate 30 SOL baseline -- confirmed by the same live
# capture above (a near-graduation event read vSolInBondingCurve=
# ~115.005, matching GRAD_SOL_UI=115.0 directly, not ~145). Phase 2.1's
# v8_path_integrity.py THEORETICAL_MAX_CURVE_ACTIVE_PRICE_SOL wrongly
# added a +30 SOL baseline on top of GRAD_SOL_UI -- fixed in the same
# batch as this module, for the same reason.
PUMPFUN_INITIAL_VIRTUAL_TOKEN_RESERVES = 1_073_000_000
PUMPFUN_REAL_TOKEN_RESERVES_AT_GRADUATION = 793_100_000
MIN_VIRTUAL_TOKEN_RESERVES_ON_CURVE = (
    PUMPFUN_INITIAL_VIRTUAL_TOKEN_RESERVES - PUMPFUN_REAL_TOKEN_RESERVES_AT_GRADUATION
)  # 279,900,000 -- empirically confirmed exact via live capture, not just theoretical


def price_sol_from_pp_reserves(v_sol_in_bonding_curve, v_tokens_in_bonding_curve) -> Optional[float]:
    """
    Both inputs are PumpPortal's raw message fields, already in UI units
    -- v_sol_in_bonding_curve in SOL, v_tokens_in_bonding_curve in whole
    (6-decimal-denominated but already-divided) tokens. Neither needs any
    further unit conversion. Returns None on invalid/missing input
    (never 0.0, which would be indistinguishable from a real zero price).
    """
    try:
        vsol = float(v_sol_in_bonding_curve)
        vtok = float(v_tokens_in_bonding_curve)
    except (TypeError, ValueError):
        return None
    if vsol <= 0 or vtok <= 0:
        return None
    return vsol / vtok


def price_usd_from_pp_reserves(v_sol_in_bonding_curve, v_tokens_in_bonding_curve, sol_price_usd: float) -> Optional[float]:
    """price_sol_from_pp_reserves() converted to USD using the caller's
    own SOL/USD rate (this function does not fetch or assume one)."""
    price_sol = price_sol_from_pp_reserves(v_sol_in_bonding_curve, v_tokens_in_bonding_curve)
    if price_sol is None:
        return None
    try:
        rate = float(sol_price_usd)
    except (TypeError, ValueError):
        return None
    if rate <= 0:
        return None
    return price_sol * rate


def venue_state_from_pp_reserves(v_sol_in_bonding_curve, v_tokens_in_bonding_curve) -> str:
    """
    V8 DATA RECOVERY item 4: research/peak_tracker.py previously wrote
    venue_state="CURVE_ACTIVE" unconditionally on every tick -- the
    direct cause of Phase 2.1's VSOL_EXCEEDS_GRADUATION_WHILE_
    CURVE_ACTIVE findings (a genuinely-graduated token kept getting
    every later tick mislabeled).

    Reserve fields present (the same presence check
    price_sol_from_pp_reserves already uses to decide its formula
    branch) is the only evidence-grounded signal available from this
    message shape: their presence means the token is still on the
    bonding curve; their absence means it left, but this shape alone
    cannot prove GRADUATED vs DEX_ACTIVE -- reported "UNKNOWN" rather
    than asserted, per this project's "prove, don't infer" discipline
    (no live-captured evidence of a genuinely-graduated message's exact
    shape exists yet to justify a more specific claim).
    """
    price = price_sol_from_pp_reserves(v_sol_in_bonding_curve, v_tokens_in_bonding_curve)
    return "CURVE_ACTIVE" if price is not None else "UNKNOWN"
