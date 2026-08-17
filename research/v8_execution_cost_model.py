"""
research/v8_execution_cost_model.py — V8-FILTER-DERIVATION Phase 2.1
item 3: execution cost model, REBUILT to separate CONFIGURED_TOLERANCE
from ACTUAL_REALIZED_EXECUTION_COST.

EXECUTION_COST_MODEL_VERSION = 2

PHASE 2.1 CORRECTION (2026-08-17): the v1 model labeled
SELL_LADDER's 35/60/98% rungs as "MEASURED" scenario slippage. That was
wrong -- those percentages are the executor's SLIPPAGE TOLERANCE
settings (memecoin/executor.py's SELL_LADDER: "on revert, escalate
slippage and fee immediately" -- the maximum the executor is WILLING to
accept before a swap reverts), not a measurement of what slippage
actually occurs. An executor configured to tolerate up to 98% slippage
does not mean trades typically realize 98% slippage. This module now
keeps CONFIGURED_TOLERANCE (from code constants) and
ACTUAL_REALIZED_EXECUTION_COST (from real journal evidence, each tagged
with an honest evidence class) as two structurally separate sections
that are never merged into one number.

REAL EVIDENCE AUDITED (2026-08-17, VPS):

  CONFIGURED_TOLERANCE (memecoin/executor.py, grep-confirmed):
    SLIPPAGE_BUY_PCT=30 (buy reverts, no fill, above this)
    SELL_LADDER=[(35,'High'),(60,'VeryHigh'),(98,'UnsafeMax')]
    PRIORITY_FEE_SOL floors: 0.0005/0.0015/0.005 SOL per rung (dynamic
      Helius estimate used when higher -- floor is a LOWER bound on
      real cost, not the typical cost)

  ACTUAL_REALIZED_EXECUTION_COST (logs/memecoin_live_journal.csv, n=80
  real live trades):
    sell_failed: 8/80 = 10.0%  (MEASURED -- a real outcome rate)
    abort_tripwire: 5/80 = 6.25%  (MEASURED)
    hard_stop overshoot (realized pnl_pct - configured hard_stop_pct):
      n=9 usable rows, mean=-50.2pp -- PARTIALLY_MEASURED, not MEASURED:
      too few usable rows AND spans both the pre- and post-MU-retry-
      ladder eras (deployed 2026-07-07, commit 538132f); CLAUDE.md
      already locks abort-threshold recalibration pending 10+ trades
      under the current latency profile.
    entry-side fill-vs-baseline slippage: UNMEASURED. Re-checked this
      pass -- signal_dex_price and baseline_curve_price are 0/80
      populated, fill_price_field/entry_source only 6/80 populated.
      Genuinely not recoverable from this journal export.
    priority fee actually paid per trade: UNMEASURED at the per-trade
      level (Helius dynamic estimate isn't logged); ASSUMPTION_BOUND at
      the configured floor.

  CRITICAL COHORT CHECK (explicitly required by Phase 2.1 item 3, not
  previously done): are the 80 historical live trades exchangeable with
  V8's population? NO -- confirmed, not assumed. Every one of the 80
  trades carries config_tag in {'v7_entry_filters_2026-06-06' (44),
  'v4_2026-05-13' (36)}, signal_type='social_alert' (80/80). V8 has
  never been live-traded (CLAUDE.md: "Live trading is PAUSED ...
  paper + research only until the user says go-live post-V8"). Venue-
  state stratification (CURVE_ACTIVE vs graduated/DEX) was also
  attempted and could NOT be done: dex_id is proven unreliable (research/
  v8_feature_registry.yaml's dex_id entry) and no venue_state field
  exists in the live-trading journal schema at all. Every
  ACTUAL_REALIZED_EXECUTION_COST value below therefore carries
  cohort_matches_v8=False and must be treated as ASSUMPTION_BOUND
  evidence for V8, regardless of its own evidence_class for V7/V4.

  SOL/USD CONVERSION: re-audited rather than kept as a static code
  comment. A time-aligned market-price source does NOT exist anywhere
  in this project's data -- checked directly: the price capture
  pipeline itself computes price_usd = price_sol * FIXED_RATE, not from
  an independently time-varying market feed (confirmed via 202,385
  integrity-qualified real ticks: price_usd/price_sol = 175.0 almost
  exactly, p10-p90 range 174.9998-175.0002 -- a fixed constant, not
  market variation). SOL_USD_REFERENCE is updated from the prior ~$170
  comment-based guess to this more precisely confirmed $175 constant,
  but STILL explicitly labeled STATIC_ASSUMPTION, never claimed to be
  time-aligned market evidence, because it isn't.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

EXECUTION_COST_MODEL_VERSION = 2


class EvidenceClass:
    MEASURED = "MEASURED"
    PARTIALLY_MEASURED = "PARTIALLY_MEASURED"
    ASSUMPTION_BOUND = "ASSUMPTION_BOUND"
    UNMEASURED = "UNMEASURED"


# ── SOL/USD conversion ───────────────────────────────────────────────────
SOL_USD_REFERENCE = 175.0
SOL_USD_REFERENCE_STATUS = "STATIC_ASSUMPTION"   # never claim time-aligned -- none exists in this project's data
SOL_USD_REFERENCE_PROVENANCE = (
    "confirmed as the fixed constant research/peak_tracker.py's price capture pipeline "
    "actually uses (price_usd = price_sol * 175.0, verified against 202,385 integrity-"
    "qualified real ticks, p10-p90 range 174.9998-175.0002) -- NOT a time-varying market "
    "SOL/USD feed, because none exists anywhere in this project's captured data"
)

# ── CONFIGURED_TOLERANCE (memecoin/executor.py, grep-confirmed) ────────
BUY_SLIPPAGE_REVERT_CEILING_PCT = 30.0
SELL_LADDER_TOLERANCE_PCTS = [35.0, 60.0, 98.0]
PRIORITY_FEE_SOL_FLOOR_HIGH = 0.0005
PRIORITY_FEE_SOL_FLOOR_VERY_HIGH = 0.0015
PRIORITY_FEE_SOL_FLOOR_UNSAFE_MAX = 0.005

# ── ACTUAL_REALIZED_EXECUTION_COST (logs/memecoin_live_journal.csv) ────
LIVE_JOURNAL_N = 80
LIVE_JOURNAL_COHORT = "v7_entry_filters_2026-06-06 (44) + v4_2026-05-13 (36), social_alert"
COHORT_MATCHES_V8 = False   # confirmed, not assumed -- see module docstring
VENUE_STATE_STRATIFICATION_STATUS = "IMPOSSIBLE_WITH_CURRENT_JOURNAL_SCHEMA"

SELL_FAILED_RATE = 8 / 80
SELL_FAILED_RATE_EVIDENCE_CLASS = EvidenceClass.MEASURED

ABORT_TRIPWIRE_RATE = 5 / 80
ABORT_TRIPWIRE_RATE_EVIDENCE_CLASS = EvidenceClass.MEASURED

HARD_STOP_OVERSHOOT_MEDIAN_PP = -51.3
HARD_STOP_OVERSHOOT_N = 9
HARD_STOP_OVERSHOOT_EVIDENCE_CLASS = EvidenceClass.PARTIALLY_MEASURED

ENTRY_SLIPPAGE_EVIDENCE_CLASS = EvidenceClass.UNMEASURED
PRIORITY_FEE_ACTUAL_EVIDENCE_CLASS = EvidenceClass.ASSUMPTION_BOUND


@dataclass(frozen=True)
class ConfiguredTolerance:
    buy_revert_ceiling_pct: float
    sell_ladder_tolerance_pcts: list
    priority_fee_floor_sol_by_rung: dict


@dataclass(frozen=True)
class RealizedCostComponent:
    name: str
    value: Optional[float]     # None when genuinely UNMEASURED
    evidence_class: str
    cohort_matches_v8: bool
    note: str


@dataclass(frozen=True)
class CostBreakdown:
    notional_usd: float
    configured_tolerance: ConfiguredTolerance
    realized_components: list          # list[RealizedCostComponent]
    priority_fee_usd_at_floor: float   # ASSUMPTION_BOUND -- floor, not a measured actual
    priority_fee_pct_of_notional: float
    size_handling: str


def get_configured_tolerance() -> ConfiguredTolerance:
    return ConfiguredTolerance(
        buy_revert_ceiling_pct=BUY_SLIPPAGE_REVERT_CEILING_PCT,
        sell_ladder_tolerance_pcts=list(SELL_LADDER_TOLERANCE_PCTS),
        priority_fee_floor_sol_by_rung={
            "High": PRIORITY_FEE_SOL_FLOOR_HIGH,
            "VeryHigh": PRIORITY_FEE_SOL_FLOOR_VERY_HIGH,
            "UnsafeMax": PRIORITY_FEE_SOL_FLOOR_UNSAFE_MAX,
        },
    )


def get_realized_cost_components() -> list:
    """Every component here describes V7/V4's social_alert population,
    NOT V8's -- COHORT_MATCHES_V8 is False for all of them. Applying any
    of these to a V8 EV estimate is an explicit, labeled transport
    assumption, never a proven equivalence."""
    cohort_note = (
        f"measured against {LIVE_JOURNAL_N} real live trades, cohort={LIVE_JOURNAL_COHORT} -- "
        "NOT V8's population (V8 has never been live-traded); venue-state stratification "
        f"is {VENUE_STATE_STRATIFICATION_STATUS} (dex_id proven unreliable, no venue_state "
        "field in this journal's schema)"
    )
    return [
        RealizedCostComponent(
            name="sell_failed_rate", value=SELL_FAILED_RATE,
            evidence_class=SELL_FAILED_RATE_EVIDENCE_CLASS,
            cohort_matches_v8=COHORT_MATCHES_V8, note=cohort_note,
        ),
        RealizedCostComponent(
            name="abort_tripwire_rate", value=ABORT_TRIPWIRE_RATE,
            evidence_class=ABORT_TRIPWIRE_RATE_EVIDENCE_CLASS,
            cohort_matches_v8=COHORT_MATCHES_V8, note=cohort_note,
        ),
        RealizedCostComponent(
            name="hard_stop_overshoot_median_pp", value=HARD_STOP_OVERSHOOT_MEDIAN_PP,
            evidence_class=HARD_STOP_OVERSHOOT_EVIDENCE_CLASS,
            cohort_matches_v8=COHORT_MATCHES_V8,
            note=cohort_note + f" (n={HARD_STOP_OVERSHOOT_N}, spans pre- and post-MU-retry-ladder eras)",
        ),
        RealizedCostComponent(
            name="entry_side_slippage_pct", value=None,
            evidence_class=ENTRY_SLIPPAGE_EVIDENCE_CLASS,
            cohort_matches_v8=COHORT_MATCHES_V8,
            note="signal_dex_price/baseline_curve_price are 0/80 populated in the journal -- "
                 "not recoverable from this export",
        ),
        RealizedCostComponent(
            name="priority_fee_actual_usd", value=None,
            evidence_class=PRIORITY_FEE_ACTUAL_EVIDENCE_CLASS,
            cohort_matches_v8=COHORT_MATCHES_V8,
            note="per-trade Helius dynamic fee estimate is not logged -- bound at the "
                 "configured floor, not independently measured",
        ),
    ]


def build_cost_breakdown(notional_usd: float) -> CostBreakdown:
    if notional_usd <= 0:
        raise ValueError("notional_usd must be > 0")

    fee_usd = PRIORITY_FEE_SOL_FLOOR_HIGH * SOL_USD_REFERENCE
    fee_pct = fee_usd / notional_usd * 100

    return CostBreakdown(
        notional_usd=notional_usd,
        configured_tolerance=get_configured_tolerance(),
        realized_components=get_realized_cost_components(),
        priority_fee_usd_at_floor=fee_usd,
        priority_fee_pct_of_notional=round(fee_pct, 3),
        size_handling="PER_NOTIONAL_FEE_MODEL (fixed-SOL fee, genuinely size-sensitive) + "
                       "LINEAR_SIZE_PROJECTION_ONLY (no curve-depth model for slippage/impact)",
    )


NOTIONALS_USD = [2.0, 5.0]


def full_cost_matrix() -> list:
    return [build_cost_breakdown(n) for n in NOTIONALS_USD]
