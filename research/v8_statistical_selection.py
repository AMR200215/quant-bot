"""
research/v8_statistical_selection.py — V8 DATA RECOVERY batch, item 10:
statistical-selection diagnostics Phase 2.1 identified as still missing
(FUTURE_STATISTICAL_READINESS_CRITERIA in research/v8_readiness_engine.py),
built now against TRAIN/VALIDATION/synthetic fixtures only. The locked
final holdout is never touched by this module -- every function here
takes plain lists/dicts of already-computed PnL values as input; none
of them read from research_tokens, a path file, or any holdout split.

Standard, established statistical techniques -- nothing novel:
  block_bootstrap_ci        -- day-block bootstrap confidence interval
  effective_n_after_ipw     -- (sum(w))^2 / sum(w^2), the standard
                                Kish effective-sample-size formula
  profit_concentration      -- top-1 / top-5 trade share of total profit
  regime_stability          -- coefficient of variation of per-week means
  candidate_degradation     -- train -> validation metric change
  max_drawdown              -- peak-to-trough on a cumulative PnL series
  max_losing_streak         -- longest consecutive run of losing trades
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Optional

STATISTICAL_SELECTION_VERSION = 1


@dataclass(frozen=True)
class BootstrapCIResult:
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_blocks: int
    n_bootstrap: int


def block_bootstrap_ci(
    pnl_by_day: dict, n_bootstrap: int = 1000, ci_level: float = 0.90, seed: Optional[int] = None,
) -> Optional[BootstrapCIResult]:
    """
    pnl_by_day: {day_label: [pnl_pct, ...]} -- one block per calendar
    day (or other independence unit), never per-trade -- resampling
    individual trades would understate real uncertainty when trades on
    the same day are correlated (shared regime, shared token launches).

    Returns None if fewer than 2 day-blocks exist (a CI from 1 block is
    not meaningful).
    """
    days = list(pnl_by_day.keys())
    if len(days) < 2:
        return None

    rng = random.Random(seed)
    all_pnls = [p for day_pnls in pnl_by_day.values() for p in day_pnls]
    if not all_pnls:
        return None
    point_estimate = mean(all_pnls)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sampled_days = [rng.choice(days) for _ in range(len(days))]
        resampled = [p for d in sampled_days for p in pnl_by_day[d]]
        if resampled:
            bootstrap_means.append(mean(resampled))

    if not bootstrap_means:
        return None

    bootstrap_means.sort()
    alpha = 1 - ci_level
    lower_idx = int(len(bootstrap_means) * (alpha / 2))
    upper_idx = int(len(bootstrap_means) * (1 - alpha / 2)) - 1
    upper_idx = min(upper_idx, len(bootstrap_means) - 1)

    return BootstrapCIResult(
        point_estimate=round(point_estimate, 4),
        ci_lower=round(bootstrap_means[lower_idx], 4),
        ci_upper=round(bootstrap_means[upper_idx], 4),
        ci_level=ci_level,
        n_blocks=len(days),
        n_bootstrap=len(bootstrap_means),
    )


def effective_n_after_ipw(weights: list) -> float:
    """Kish effective sample size: (sum w)^2 / sum(w^2). Equals raw n
    when all weights are 1.0; shrinks as weights become more unequal
    (a few heavily-upweighted observations dominate less)."""
    if not weights:
        return 0.0
    sum_w = sum(weights)
    sum_w2 = sum(w * w for w in weights)
    if sum_w2 == 0:
        return 0.0
    return round((sum_w ** 2) / sum_w2, 4)


@dataclass(frozen=True)
class ProfitConcentrationResult:
    total_profit: float
    top_1_contribution_pct: Optional[float]
    top_5_contribution_pct: Optional[float]
    n_trades: int
    n_profitable_trades: int


def profit_concentration(pnls: list) -> ProfitConcentrationResult:
    """What fraction of TOTAL PROFIT (summed only over winning trades)
    comes from the single best trade, and the top 5. High concentration
    means the result is fragile -- driven by one outlier, not a
    repeatable edge."""
    profitable = sorted([p for p in pnls if p > 0], reverse=True)
    total_profit = sum(profitable)

    if total_profit <= 0 or not profitable:
        return ProfitConcentrationResult(
            total_profit=round(total_profit, 4), top_1_contribution_pct=None,
            top_5_contribution_pct=None, n_trades=len(pnls), n_profitable_trades=len(profitable),
        )

    top_1_pct = round(profitable[0] / total_profit * 100, 2)
    top_5_pct = round(sum(profitable[:5]) / total_profit * 100, 2)

    return ProfitConcentrationResult(
        total_profit=round(total_profit, 4), top_1_contribution_pct=top_1_pct,
        top_5_contribution_pct=top_5_pct, n_trades=len(pnls), n_profitable_trades=len(profitable),
    )


@dataclass(frozen=True)
class RegimeStabilityResult:
    per_week_means: dict
    coefficient_of_variation: Optional[float]
    stability_label: str   # "STABLE" | "UNSTABLE" | "INSUFFICIENT_WEEKS"
    n_weeks: int


def regime_stability(pnl_by_week: dict, cv_unstable_threshold: float = 1.0) -> RegimeStabilityResult:
    """Coefficient of variation across per-week mean PnL -- a candidate
    whose edge appears in only one week (high CV) is not evidence of a
    stable, repeatable strategy."""
    weeks = list(pnl_by_week.keys())
    if len(weeks) < 2:
        return RegimeStabilityResult(
            per_week_means={w: round(mean(v), 4) for w, v in pnl_by_week.items() if v},
            coefficient_of_variation=None, stability_label="INSUFFICIENT_WEEKS", n_weeks=len(weeks),
        )

    per_week_means = {w: mean(v) for w, v in pnl_by_week.items() if v}
    means = list(per_week_means.values())
    if len(means) < 2:
        return RegimeStabilityResult(
            per_week_means={w: round(m, 4) for w, m in per_week_means.items()},
            coefficient_of_variation=None, stability_label="INSUFFICIENT_WEEKS", n_weeks=len(weeks),
        )

    m = mean(means)
    if m == 0:
        cv = float("inf")
    else:
        cv = abs(stdev(means) / m)

    label = "UNSTABLE" if cv > cv_unstable_threshold else "STABLE"
    return RegimeStabilityResult(
        per_week_means={w: round(v, 4) for w, v in per_week_means.items()},
        coefficient_of_variation=round(cv, 4) if cv != float("inf") else None,
        stability_label=label, n_weeks=len(weeks),
    )


@dataclass(frozen=True)
class DegradationResult:
    train_metric: float
    validation_metric: float
    absolute_change: float
    relative_change_pct: Optional[float]
    degraded: bool


def candidate_degradation(train_metric: float, validation_metric: float) -> DegradationResult:
    """Classic overfitting check: does performance measured on TRAIN
    hold up on VALIDATION (never holdout -- both of these are pre-
    holdout splits)."""
    absolute_change = validation_metric - train_metric
    relative_change_pct = None
    if train_metric != 0:
        relative_change_pct = round(absolute_change / abs(train_metric) * 100, 2)
    return DegradationResult(
        train_metric=round(train_metric, 4), validation_metric=round(validation_metric, 4),
        absolute_change=round(absolute_change, 4), relative_change_pct=relative_change_pct,
        degraded=validation_metric < train_metric,
    )


def max_drawdown(cumulative_pnl_series: list) -> float:
    """Peak-to-trough decline on a cumulative PnL series. Returns a
    positive number (the size of the largest decline), 0.0 if the
    series never declines."""
    if not cumulative_pnl_series:
        return 0.0
    peak = cumulative_pnl_series[0]
    max_dd = 0.0
    for v in cumulative_pnl_series:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 4)


def max_losing_streak(pnls: list) -> int:
    """Longest consecutive run of pnl <= 0, in trade-sequence order (the
    order the list is given in -- caller is responsible for passing
    chronological order, this function doesn't sort)."""
    longest = current = 0
    for p in pnls:
        if p <= 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest
