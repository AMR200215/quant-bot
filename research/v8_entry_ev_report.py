"""
research/v8_entry_ev_report.py — first entry-EV (SELECTION) evaluation.

Answers: given a frozen entry candidate's rule, what does the poll-
tracked outcome (research/outcome_poller.py's pct_change_peak) actually
look like? TRAIN + VALIDATION ONLY -- holdout is never read here, not
even indirectly. This is the first module in the project that reads
real pct_change_peak VALUES (not just counts) -- everything before this
(v8_forward_readiness_report.py, v8_collection_yield.py,
v8_path_predictability.py) deliberately stopped short of that. This one
crosses that line ON PURPOSE, but only for the train+validation split,
enforced the same way _compute_diagnostics_feasibility already does
(grouped_chronological_split, then only ever touch result.train +
result.validation).

"Winner" threshold (pct_change_peak >= 50%) is not invented here -- it's
research/analysis/path_stats.py's own existing _WINNER_THRESHOLD_PCT
convention, reused verbatim.

Only reports candidates already gated by research/v8_readiness_engine.py's
selection_data_ready (via research/v8_forward_readiness_report.py) as
having a real, floor-clearing sample -- others are still shown, but
explicitly labeled BELOW_FLOOR / informational only, never presented as
equally trustworthy.

Read-only. Does not touch any frozen registry, does not open holdout,
does not change any threshold.

Run:
    python -m research.v8_entry_ev_report
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import mean, median, quantiles
from typing import Optional

from research.v8_candidate_registry import CANDIDATES
from research.v8_split import grouped_chronological_split
from research.v8_forward_readiness_report import _fetch_all_clean_events, _filter_candidate_events

WINNER_THRESHOLD_PCT = 50.0  # research/analysis/path_stats.py's _WINNER_THRESHOLD_PCT, reused verbatim


def _alert_time_to_epoch(alert_time: str) -> Optional[float]:
    if not alert_time:
        return None
    try:
        return datetime.fromisoformat(alert_time.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


@dataclass(frozen=True)
class EntryEvResult:
    candidate_id: str
    n: int                    # train+validation venue-qualified rows with a non-null pct_change_peak
    below_floor: bool         # True if this candidate did not clear selection_data_ready (see caller)
    win_rate_pct: Optional[float]
    mean_pct_change_peak: Optional[float]
    median_pct_change_peak: Optional[float]
    p25: Optional[float]
    p75: Optional[float]
    p90: Optional[float]


def _train_validation_outcome_rows(candidate_events: list) -> list:
    """Same pattern as v8_forward_readiness_report.py's
    _compute_diagnostics_feasibility / _compute_poll_outcome_coverage:
    split via grouped_chronological_split, then only ever read the
    train and validation buckets. The holdout bucket is never consulted
    here in any way."""
    venue_qualified = [r for r in candidate_events if r.get("venue_state_at_signal") == "CURVE_ACTIVE"]
    events_with_time = [r for r in venue_qualified if r.get("alert_time")]
    rows_for_split = [
        {"token_address": r["token_address"], "_epoch": _alert_time_to_epoch(r["alert_time"]), "_orig": r}
        for r in events_with_time
    ]
    rows_for_split = [r for r in rows_for_split if r["_epoch"] is not None]
    if len(rows_for_split) < 2:
        return []
    try:
        result = grouped_chronological_split(
            rows_for_split, lambda r: r["token_address"], lambda r: r["_epoch"],
        )
    except ValueError:
        return []
    train_validation_rows = [r["_orig"] for r in (result.train + result.validation)]
    return [r for r in train_validation_rows if r.get("pct_change_peak") is not None]


def compute_entry_ev(candidate_id: str, outcome_rows: list, below_floor: bool) -> EntryEvResult:
    n = len(outcome_rows)
    if n == 0:
        return EntryEvResult(candidate_id=candidate_id, n=0, below_floor=below_floor,
                              win_rate_pct=None, mean_pct_change_peak=None, median_pct_change_peak=None,
                              p25=None, p75=None, p90=None)

    values = [r["pct_change_peak"] for r in outcome_rows]
    wins = sum(1 for v in values if v >= WINNER_THRESHOLD_PCT)
    win_rate = round(100 * wins / n, 2)
    mean_v = round(mean(values), 2)
    median_v = round(median(values), 2)

    if n >= 4:
        qs = quantiles(sorted(values), n=100)
        p25, p75, p90 = round(qs[24], 2), round(qs[74], 2), round(qs[89], 2)
    else:
        p25 = p75 = p90 = None

    return EntryEvResult(
        candidate_id=candidate_id, n=n, below_floor=below_floor,
        win_rate_pct=win_rate, mean_pct_change_peak=mean_v, median_pct_change_peak=median_v,
        p25=p25, p75=p75, p90=p90,
    )


def build_report(sb, selection_ready_candidates: Optional[set] = None) -> list:
    all_events = _fetch_all_clean_events(sb)
    results = []
    for candidate in CANDIDATES:
        candidate_events = _filter_candidate_events(all_events, candidate)
        outcome_rows = _train_validation_outcome_rows(candidate_events)
        below_floor = (
            selection_ready_candidates is not None
            and candidate["candidate_id"] not in selection_ready_candidates
        )
        results.append(compute_entry_ev(candidate["candidate_id"], outcome_rows, below_floor))
    return results


def print_report(results: list) -> None:
    print(f"\n{'=' * 78}")
    print(f"  V8 ENTRY-EV REPORT (train+validation only -- holdout never read)")
    print(f"{'=' * 78}")
    print(f"  Winner = pct_change_peak >= +{WINNER_THRESHOLD_PCT:.0f}% "
          f"(research/analysis/path_stats.py's own convention, reused)")
    print(f"\n  {'candidate':<12} {'n':>6} {'win_rate':>10} {'mean':>10} {'median':>10} "
          f"{'p25':>8} {'p75':>8} {'p90':>8}")
    for r in results:
        flag = "  (BELOW FLOOR -- informational only)" if r.below_floor else ""
        if r.n == 0:
            print(f"  {r.candidate_id:<12} {r.n:>6}  INSUFFICIENT{flag}")
            continue
        p25 = f"{r.p25:>7.1f}%" if r.p25 is not None else "    n/a"
        p75 = f"{r.p75:>7.1f}%" if r.p75 is not None else "    n/a"
        p90 = f"{r.p90:>7.1f}%" if r.p90 is not None else "    n/a"
        print(f"  {r.candidate_id:<12} {r.n:>6} {r.win_rate_pct:>9.1f}% {r.mean_pct_change_peak:>9.1f}% "
              f"{r.median_pct_change_peak:>9.1f}% {p25} {p75} {p90}{flag}")
    print(f"\n{'=' * 78}")
    print("  holdout_evaluated = False (structurally always -- this report never reads result.holdout)")
    print(f"{'=' * 78}\n")


def main():
    from supabase import create_client
    from research.config import SUPABASE_URL, SUPABASE_KEY
    from research.v8_forward_readiness_report import build_report as build_readiness_report
    from pathlib import Path

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    readiness = build_readiness_report(sb, Path(__file__).parent.parent)
    results = build_report(sb, selection_ready_candidates=set(readiness.selection_data_ready_candidates))
    print_report(results)


if __name__ == "__main__":
    main()
