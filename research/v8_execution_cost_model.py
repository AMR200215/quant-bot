"""
research/v8_execution_cost_model.py — V8-FILTER-DERIVATION Phase 2
(P2-9/FD20): execution cost model, built from real project evidence, not
assumed round numbers.

EXECUTION_COST_MODEL_VERSION = 1

Evidence audited live (2026-08-17, git SHA 5830704, VPS):

  Code constants (memecoin/executor.py, grep-confirmed):
    SLIPPAGE_BUY_PCT = 30          -- buy reverts (no fill) above this
    SELL_LADDER = [(35,'High'), (60,'VeryHigh'), (98,'UnsafeMax')]
    PRIORITY_FEE_SOL floor = 0.0005 / 0.0015 / 0.005 SOL per ladder rung
        (dynamic Helius estimate used when higher -- floor is a lower
        bound on the real cost, not the typical cost)
    abort_tripwire / SLIPPAGE_GATE_RT_PCT = 30% (fill vs signal price)
    Reference SOL price used in-repo for fee comments: ~$170/SOL
        (memecoin/executor.py:58 comment) -- used here ONLY to convert
        SOL-denominated fees to USD, not as a live price source.

  Live journal (logs/memecoin_live_journal.csv, n=80 real live trades,
  queried live via VPS):
    sell_failed:      8/80  = 10.0%
    abort_tripwire:   5/80  =  6.25%
    exit_reason mix:  hard_stop 42 (52.5%), feed_blind 9 (11.25%),
        jupiter_rescue 8 (10.0%), trailing_stop 7 (8.75%),
        sell_stuck_retry 3 (3.75%), reconciled_gone 3 (3.75%),
        abort_tripwire 5 (6.25%), manual/manual_sell 2 (2.5%),
        graduated_loss 1 (1.25%)
    hard_stop overshoot (realized pnl_pct - configured hard_stop_pct):
        n=9 usable rows (most historical rows don't carry both fields),
        mean=-50.2pp, median=-51.3pp, range=[-87.47, -5.26]pp. NOT
        promoted to MEASURED: n is small AND spans both the pre- and
        post-MU-retry-ladder eras (deployed 2026-07-07, commit
        538132f) -- CLAUDE.md explicitly locks abort-threshold
        recalibration until 10+ trades exist under the current latency
        profile. Recorded as CONSERVATIVE-informative only.
    entry-side fill-vs-baseline slippage: NOT measurable from the
        current journal export in this pass (signal_dex_price/
        baseline_curve_price were empty for every "fill:" row checked)
        -- UNMEASURED_ENTRY_SLIPPAGE, not invented. SLIPPAGE_BUY_PCT
        (30%, a hard revert ceiling, not an average) is used only as
        the STRESS-scenario bound below, explicitly labeled as such.

Scenario tiers (FD20's explicit MEASURED/CONSERVATIVE/STRESS split,
used where evidence strength differs per cost component -- never a
single blended number pretending uniform confidence):
  MEASURED     -- directly observed in code constants or the live
                  journal, cited above.
  CONSERVATIVE -- worst directly-observed value promoted to a
                  planning assumption (e.g. the 98% terminal sell-ladder
                  rung, the small-n hard-stop overshoot median).
  STRESS       -- a documented ceiling (e.g. the 30% buy-revert cap)
                  used as a pessimistic bound where no real average
                  exists yet.
"""

from __future__ import annotations

from dataclasses import dataclass

EXECUTION_COST_MODEL_VERSION = 1

# ── Real, cited constants ───────────────────────────────────────────────
SOL_USD_REFERENCE = 170.0   # memecoin/executor.py:58 comment -- conversion only

PRIORITY_FEE_SOL_HIGH = 0.0005        # SELL_LADDER rung 1 floor
PRIORITY_FEE_SOL_VERY_HIGH = 0.0015   # SELL_LADDER rung 2 floor
PRIORITY_FEE_SOL_UNSAFE_MAX = 0.005   # SELL_LADDER rung 3 floor

BUY_SLIPPAGE_REVERT_CEILING_PCT = 30.0   # SLIPPAGE_BUY_PCT -- revert above this, not an avg
SELL_LADDER_PCTS = [35.0, 60.0, 98.0]    # escalation rungs, memecoin/executor.py SELL_LADDER

LIVE_JOURNAL_N = 80
SELL_FAILED_RATE = 8 / 80
ABORT_TRIPWIRE_RATE = 5 / 80
HARD_STOP_OVERSHOOT_MEDIAN_PP = -51.3   # n=9, CONSERVATIVE-informative only (see module docstring)

ENTRY_SLIPPAGE_STATUS = "UNMEASURED_ENTRY_SLIPPAGE"


@dataclass(frozen=True)
class CostBreakdown:
    scenario: str
    notional_usd: float
    priority_fee_usd: float
    priority_fee_pct_of_notional: float
    sell_slippage_pct: float
    entry_slippage_pct: float
    entry_slippage_status: str
    sell_failure_rate: float
    round_trip_cost_pct: float   # priority_fee_pct + sell_slippage_pct + entry_slippage_pct
    size_handling: str           # "PER_NOTIONAL_FEE_MODEL" | "LINEAR_SIZE_PROJECTION_ONLY"


def _priority_fee_usd(scenario: str) -> float:
    if scenario == "MEASURED":
        return PRIORITY_FEE_SOL_HIGH * SOL_USD_REFERENCE
    if scenario == "CONSERVATIVE":
        return PRIORITY_FEE_SOL_VERY_HIGH * SOL_USD_REFERENCE
    if scenario == "STRESS":
        return PRIORITY_FEE_SOL_UNSAFE_MAX * SOL_USD_REFERENCE
    raise ValueError(f"unknown scenario: {scenario}")


def _sell_slippage_pct(scenario: str) -> float:
    if scenario == "MEASURED":
        return SELL_LADDER_PCTS[0]   # 35% -- the rung actually used absent escalation
    if scenario == "CONSERVATIVE":
        return SELL_LADDER_PCTS[1]   # 60% -- escalation rung, real and observed in SELL_LADDER
    if scenario == "STRESS":
        return SELL_LADDER_PCTS[2]   # 98% -- terminal rung, real ceiling
    raise ValueError(f"unknown scenario: {scenario}")


def estimate_round_trip_cost(notional_usd: float, scenario: str) -> CostBreakdown:
    """
    Priority fee is a FIXED SOL cost -- genuinely size-sensitive, modeled
    per-notional exactly (a $0.085 fee is a real, different % hit on $2
    vs $5). Entry/sell slippage is curve-depth-driven; this repo has no
    curve-depth simulation, so it is applied as the SAME percentage
    regardless of notional (LINEAR_SIZE_PROJECTION_ONLY) -- explicitly
    NOT claimed to be notional-aware.
    """
    if scenario not in ("MEASURED", "CONSERVATIVE", "STRESS"):
        raise ValueError(f"unknown scenario: {scenario}")
    if notional_usd <= 0:
        raise ValueError("notional_usd must be > 0")

    fee_usd = _priority_fee_usd(scenario)
    fee_pct = fee_usd / notional_usd * 100

    sell_slip = _sell_slippage_pct(scenario)
    # Entry slippage stays unmeasured -- 0.0 is NOT a claim of zero cost,
    # it's an explicit absence, distinguished by entry_slippage_status.
    entry_slip = 0.0 if scenario != "STRESS" else BUY_SLIPPAGE_REVERT_CEILING_PCT

    round_trip = fee_pct + sell_slip + entry_slip

    return CostBreakdown(
        scenario=scenario,
        notional_usd=notional_usd,
        priority_fee_usd=fee_usd,
        priority_fee_pct_of_notional=round(fee_pct, 3),
        sell_slippage_pct=sell_slip,
        entry_slippage_pct=entry_slip,
        entry_slippage_status=ENTRY_SLIPPAGE_STATUS,
        sell_failure_rate=SELL_FAILED_RATE,
        round_trip_cost_pct=round(round_trip, 3),
        size_handling="PER_NOTIONAL_FEE_MODEL+LINEAR_SIZE_PROJECTION_ONLY_SLIPPAGE",
    )


# $2 and $5 handled separately per P2-9 (fee component is notional-aware;
# slippage component is not -- see estimate_round_trip_cost's docstring).
NOTIONALS_USD = [2.0, 5.0]
SCENARIOS = ["MEASURED", "CONSERVATIVE", "STRESS"]


def full_cost_matrix() -> list[CostBreakdown]:
    return [estimate_round_trip_cost(n, s) for n in NOTIONALS_USD for s in SCENARIOS]
