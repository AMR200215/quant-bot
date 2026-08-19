"""
research/v8_execution_proxy_readiness.py — V8 READINESS MONITOR
CORRECTION batch, item 3: candidate-specific execution-proxy coverage
+ EXECUTION_PROXY_READY.

Fixes: research/v8_forward_readiness_report.py previously set
entry_slippage_measured = execution_proxy_total > 0 -- one observation
is not sufficient evidence of anything. Coverage was also a GLOBAL sum
across candidates whose eligible populations overlap (a single
Telegram event can qualify for both V8-P0 and V8-P3); summing per-
candidate counts into one global denominator would double-count that
event. This module reports per-candidate numbers only -- callers must
never sum them across candidates.

EXECUTION_PROXY_MIN_N / EXECUTION_PROXY_MIN_COVERAGE_PCT reuse the SAME
cited conventions research/v8_readiness_engine.py already uses
(MIN_ENTRY_N/MIN_PATH_N from research/analysis/path_stats.py's own
--min-n default; MIN_PATH_COVERAGE_PCT as the same explicitly-labeled
design-choice floor) -- no project-specific "execution cost is
estimable at N observations" threshold exists, so this deliberately
does not invent a new number; it is the same evidence tier as the
existing path-coverage floor, labeled the same way: an engineering
prerequisite, not a statistical-sufficiency claim.
"""

from __future__ import annotations

from dataclasses import dataclass

EXECUTION_PROXY_MIN_N = 100
EXECUTION_PROXY_MIN_COVERAGE_PCT = 50.0

EXECUTION_PROXY_THRESHOLD_PROVENANCE = {
    "EXECUTION_PROXY_MIN_N": "same as research/v8_readiness_engine.py MIN_ENTRY_N/MIN_PATH_N "
                              "-- research/analysis/path_stats.py's own --min-n default",
    "EXECUTION_PROXY_MIN_COVERAGE_PCT": "same as research/v8_readiness_engine.py MIN_PATH_COVERAGE_PCT "
                                         "-- a design choice (more than half), NOT independently data-derived",
}


@dataclass(frozen=True)
class ExecutionProxyCoverage:
    eligible_n: int
    observed_n: int             # distinct events (by event_id, never mint alone) with >=1 observation
    coverage_pct: float
    unique_days: int
    ready: bool
    reasons: list


def assess_execution_proxy_readiness(eligible_n: int, observed_n: int, unique_days: int) -> ExecutionProxyCoverage:
    """
    eligible_n / observed_n / unique_days must all be computed for ONE
    candidate's own eligible population -- never a cross-candidate sum.
    Never becomes ready from a single observation: both the minimum
    count AND the minimum coverage percentage must independently clear
    their floors.
    """
    reasons: list = []
    coverage_pct = round(observed_n / eligible_n * 100, 2) if eligible_n else 0.0

    if observed_n < EXECUTION_PROXY_MIN_N:
        reasons.append(f"observed_n={observed_n} < EXECUTION_PROXY_MIN_N={EXECUTION_PROXY_MIN_N}")
    if coverage_pct < EXECUTION_PROXY_MIN_COVERAGE_PCT:
        reasons.append(f"coverage_pct={coverage_pct} < EXECUTION_PROXY_MIN_COVERAGE_PCT={EXECUTION_PROXY_MIN_COVERAGE_PCT}")

    ready = not reasons
    return ExecutionProxyCoverage(
        eligible_n=eligible_n, observed_n=observed_n, coverage_pct=coverage_pct,
        unique_days=unique_days, ready=ready, reasons=reasons,
    )
