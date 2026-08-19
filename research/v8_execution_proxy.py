"""
research/v8_execution_proxy.py — V8 DATA RECOVERY batch, item 7:
paper/research-only $2/$5 execution-cost proxy collector.

Phase 2.1 found V8 entry slippage genuinely UNMEASURED and V7/V4 live
execution data NOT proven transportable to V8's population. Rather than
wait until final-filter time to discover execution cost is still
unknown, this collects a best-effort executable-cost OBSERVATION for
every forward V8-research-eligible event, going forward, using the
SAME exact math the real system already relies on:

  CURVE_ACTIVE: the standard constant-product AMM formula pump.fun's
    bonding curve actually implements (fee taken from input, output =
    token_reserve * amount_in_after_fee / (sol_reserve +
    amount_in_after_fee)) -- applied to the SAME vSolInBondingCurve/
    vTokensInBondingCurve reserves this batch's root-cause work already
    validated against live on-chain reads. PUMPFUN_TRADING_FEE_RATE is
    a cited public protocol constant (1% on bonding-curve trades), not
    independently re-verified against this project's own on-chain fee
    receipts in this pass -- labeled MODEL, not MEASURED.
  Non-CURVE_ACTIVE (graduated/DEX): reuses memecoin/executor.py's
    _jup_get_quote -- a genuinely non-binding Jupiter quote call that
    sends no transaction -- rather than inventing a second, approximate
    DEX-impact formula.
  Neither available (no reserves AND no quote): EXECUTION_PROXY_UNAVAILABLE,
    never a fabricated cost.

Sends no transaction, ever. Never touches live_buys_enabled/
live_sells_enabled or any executor state. Intended to be called as a
best-effort, exception-isolated side observation alongside existing
research enrichment (e.g. where ProgressCapture already runs in
research/tracker.py) -- never on the live-trading critical path, which
lives entirely in memecoin/, not research/.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

EXECUTION_PROXY_MODEL_VERSION = 1

# Public pump.fun protocol constant (bonding-curve trading fee) --
# widely documented, NOT independently re-verified against this
# project's own on-chain fee receipts in this pass. Same evidence tier
# as PUMPFUN_INITIAL_VIRTUAL_TOKEN_RESERVES etc.
# (memecoin/pumpfun_reserve_pricing.py) -- cited, not measured.
PUMPFUN_TRADING_FEE_RATE = 0.01

EXECUTION_PROXY_UNAVAILABLE = "EXECUTION_PROXY_UNAVAILABLE"


@dataclass(frozen=True)
class CurveBuySimResult:
    reference_spot_price_sol: float
    executable_buy_price_sol: float
    price_impact_pct: float
    fee_sol: float
    tokens_received: float
    new_vsol: float
    new_vtok: float


def simulate_curve_buy(vsol: float, vtok: float, notional_sol: float,
                        fee_rate: float = PUMPFUN_TRADING_FEE_RATE) -> Optional[CurveBuySimResult]:
    """
    Exact constant-product AMM math (the same formula the real pump.fun
    program implements), applied to real reserve snapshots. Returns
    None on invalid input rather than a fabricated result.
    """
    if vsol <= 0 or vtok <= 0 or notional_sol <= 0 or not (0 <= fee_rate < 1):
        return None

    reference_spot_price_sol = vsol / vtok
    fee_sol = notional_sol * fee_rate
    amount_in_after_fee = notional_sol - fee_sol
    tokens_received = (vtok * amount_in_after_fee) / (vsol + amount_in_after_fee)
    if tokens_received <= 0:
        return None

    executable_buy_price_sol = notional_sol / tokens_received
    price_impact_pct = (executable_buy_price_sol / reference_spot_price_sol - 1) * 100

    new_vsol = vsol + amount_in_after_fee
    new_vtok = vtok - tokens_received

    return CurveBuySimResult(
        reference_spot_price_sol=reference_spot_price_sol,
        executable_buy_price_sol=executable_buy_price_sol,
        price_impact_pct=round(price_impact_pct, 6),
        fee_sol=fee_sol,
        tokens_received=tokens_received,
        new_vsol=new_vsol,
        new_vtok=new_vtok,
    )


@dataclass(frozen=True)
class RoundTripSimResult:
    buy: CurveBuySimResult
    sell_proceeds_sol: float
    sell_fee_sol: float
    round_trip_value_sol: float   # sell_proceeds_sol net of sell fee
    round_trip_pct: float         # vs notional_sol, immediate buy+sell, no time passing


def simulate_round_trip(vsol: float, vtok: float, notional_sol: float,
                         fee_rate: float = PUMPFUN_TRADING_FEE_RATE) -> Optional[RoundTripSimResult]:
    """Immediate buy-then-sell through the SAME shifted curve state the
    buy leg produced -- both legs use the identical exact formula, fee
    applied on each leg (matching how the real program charges it on
    every trade, not just entries)."""
    buy = simulate_curve_buy(vsol, vtok, notional_sol, fee_rate)
    if buy is None:
        return None

    # Sell leg: sell buy.tokens_received back into the post-buy curve state.
    sell_fee_rate = fee_rate
    tokens_in = buy.tokens_received
    sol_out_gross = (buy.new_vsol * tokens_in) / (buy.new_vtok + tokens_in)
    sell_fee_sol = sol_out_gross * sell_fee_rate
    sell_proceeds_sol = sol_out_gross - sell_fee_sol

    round_trip_pct = (sell_proceeds_sol / notional_sol - 1) * 100

    return RoundTripSimResult(
        buy=buy,
        sell_proceeds_sol=sell_proceeds_sol,
        sell_fee_sol=sell_fee_sol,
        round_trip_value_sol=sell_proceeds_sol,
        round_trip_pct=round(round_trip_pct, 6),
    )


@dataclass(frozen=True)
class ExecutionProxyObservation:
    event_id: str
    token_address: str
    notional_usd: float
    observed_at: str
    lag_from_alert_ms: Optional[int]
    venue: str                       # "CURVE_ACTIVE" | "GRADUATED_OR_DEX" | "UNKNOWN"
    data_source: str                 # "pp_reserve_snapshot" | "jupiter_quote" | "unavailable"
    model_version: int
    reference_spot_price_usd: Optional[float]
    executable_buy_price_usd: Optional[float]
    price_impact_pct: Optional[float]
    fee_usd: Optional[float]
    round_trip_value_usd: Optional[float]
    round_trip_pct: Optional[float]
    curve_reserves_used: Optional[dict]   # {"vsol": ..., "vtok": ...} or None
    status: str                      # "OK" | EXECUTION_PROXY_UNAVAILABLE


def build_curve_observation(
    event_id: str, token_address: str, notional_usd: float, sol_price_usd: float,
    vsol: float, vtok: float, alert_ts_ms: Optional[int] = None, observed_ts_ms: Optional[int] = None,
) -> ExecutionProxyObservation:
    """Builds one $-notional observation for a CURVE_ACTIVE token from
    real reserve fields. Never sends a transaction."""
    observed_at = datetime.now(timezone.utc).isoformat()
    lag_ms = None
    if alert_ts_ms is not None and observed_ts_ms is not None:
        lag_ms = observed_ts_ms - alert_ts_ms

    if sol_price_usd <= 0:
        return ExecutionProxyObservation(
            event_id=event_id, token_address=token_address, notional_usd=notional_usd,
            observed_at=observed_at, lag_from_alert_ms=lag_ms, venue="CURVE_ACTIVE",
            data_source="unavailable", model_version=EXECUTION_PROXY_MODEL_VERSION,
            reference_spot_price_usd=None, executable_buy_price_usd=None, price_impact_pct=None,
            fee_usd=None, round_trip_value_usd=None, round_trip_pct=None,
            curve_reserves_used=None, status=EXECUTION_PROXY_UNAVAILABLE,
        )

    notional_sol = notional_usd / sol_price_usd
    rt = simulate_round_trip(vsol, vtok, notional_sol)
    if rt is None:
        return ExecutionProxyObservation(
            event_id=event_id, token_address=token_address, notional_usd=notional_usd,
            observed_at=observed_at, lag_from_alert_ms=lag_ms, venue="CURVE_ACTIVE",
            data_source="unavailable", model_version=EXECUTION_PROXY_MODEL_VERSION,
            reference_spot_price_usd=None, executable_buy_price_usd=None, price_impact_pct=None,
            fee_usd=None, round_trip_value_usd=None, round_trip_pct=None,
            curve_reserves_used={"vsol": vsol, "vtok": vtok}, status=EXECUTION_PROXY_UNAVAILABLE,
        )

    return ExecutionProxyObservation(
        event_id=event_id, token_address=token_address, notional_usd=notional_usd,
        observed_at=observed_at, lag_from_alert_ms=lag_ms, venue="CURVE_ACTIVE",
        data_source="pp_reserve_snapshot", model_version=EXECUTION_PROXY_MODEL_VERSION,
        reference_spot_price_usd=rt.buy.reference_spot_price_sol * sol_price_usd,
        executable_buy_price_usd=rt.buy.executable_buy_price_sol * sol_price_usd,
        price_impact_pct=rt.buy.price_impact_pct,
        fee_usd=rt.buy.fee_sol * sol_price_usd,
        round_trip_value_usd=rt.round_trip_value_sol * sol_price_usd,
        round_trip_pct=rt.round_trip_pct,
        curve_reserves_used={"vsol": vsol, "vtok": vtok},
        status="OK",
    )


def build_unavailable_observation(event_id: str, token_address: str, notional_usd: float,
                                   venue: str = "UNKNOWN") -> ExecutionProxyObservation:
    """Explicit, honest non-observation -- e.g. no reserves and the
    Jupiter quote path also failed/is inapplicable. Never fabricates a
    cost."""
    return ExecutionProxyObservation(
        event_id=event_id, token_address=token_address, notional_usd=notional_usd,
        observed_at=datetime.now(timezone.utc).isoformat(), lag_from_alert_ms=None,
        venue=venue, data_source="unavailable", model_version=EXECUTION_PROXY_MODEL_VERSION,
        reference_spot_price_usd=None, executable_buy_price_usd=None, price_impact_pct=None,
        fee_usd=None, round_trip_value_usd=None, round_trip_pct=None,
        curve_reserves_used=None, status=EXECUTION_PROXY_UNAVAILABLE,
    )
