"""
research/v8_forward_readiness_report.py — V8 DATA RECOVERY batch, item 9:
ONE automatic, read-only forward-readiness report for the frozen v1
experiment against CURRENT FORWARD DATA. Never opens/prints holdout
results -- this report never computes PnL or runs any candidate against
price/outcome data at all; it only measures DATA SUFFICIENCY (counts,
integrity, coverage), which is categorically distinct from holdout
evaluation in the FD-BATCH sense.

Reuses the frozen v1 registries and the existing readiness engine
(research/v8_readiness_engine.py) rather than redesigning them -- this
module is a live-data POPULATOR for ReadinessInputs plus an
accumulation-velocity estimator, not a new readiness policy.

Run:
    python -m research.v8_forward_readiness_report
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from research.v8_candidate_registry import CANDIDATES
from research.v8_exit_registry import EXIT_CANDIDATES
from research.v8_readiness_engine import ReadinessInputs, ReadinessReport, assess_readiness
from research.v8_path_integrity import scan_corpus, CorpusIntegrityReport

FORWARD_READINESS_REPORT_VERSION = 1

_ACCUMULATION_WINDOW_DAYS = 7  # trailing window used to estimate daily accumulation rate
_MIN_DAYS_FOR_VELOCITY_ESTIMATE = 2  # below this, rate is too noisy to report as anything but INSUFFICIENT_HISTORY
_VELOCITY_CV_UNSTABLE_THRESHOLD = 0.75  # coefficient of variation above this -> too unstable to project


@dataclass(frozen=True)
class AccumulationVelocity:
    new_venue_qualified_per_day: Optional[float]
    new_valid_representative_paths_per_day: Optional[float]
    new_execution_proxy_observations_per_day: Optional[float]
    stability_label: str   # "STABLE" | "TOO_VARIABLE_TO_PROJECT" | "INSUFFICIENT_HISTORY"
    window_days: int
    note: str


@dataclass(frozen=True)
class CandidateForwardEvidence:
    candidate_id: str
    historical_entry_n: int
    unique_mints: int
    unique_days: int
    forward_venue_qualified_n: int
    venue_qualified_unique_mints: int
    venue_qualified_unique_days: int
    progress_distribution: dict   # {"<50%": n, "50-70%": n, "70-85%": n, "85%+": n}


@dataclass(frozen=True)
class ForwardReadinessReport:
    generated_at: str
    report_version: int
    candidate_evidence: list        # list[CandidateForwardEvidence]
    readiness_matrix: list          # list[ReadinessReport], one per (candidate, exit)
    path_integrity: CorpusIntegrityReport
    execution_proxy_observations_total: int
    execution_proxy_coverage_pct: float
    accumulation_velocity: AccumulationVelocity
    selection_data_ready: bool
    holdout_evaluated: bool = False   # structurally always False -- this report never touches holdout


def _progress_bucket(p: Optional[float]) -> str:
    if p is None:
        return "unknown"
    if p < 0.50:
        return "<50%"
    if p < 0.70:
        return "50-70%"
    if p < 0.85:
        return "70-85%"
    return "85%+"


def _query_candidate_evidence(sb, candidate: dict) -> CandidateForwardEvidence:
    """Live Supabase query for ONE frozen candidate's real current
    counts. Never touches holdout -- these are pure data-sufficiency
    counts, not any PnL/outcome computation."""
    base = sb.table("research_tokens").select(
        "token_address,alert_time,progress_at_signal,venue_state_at_signal"
    ).eq("chain", "solana").eq("progress_data_ok", True)

    progress_cond = next((c for c in candidate["conditions"] if c["feature"] == "progress_at_signal"), None)
    if progress_cond is not None:
        base = base.lt("progress_at_signal", progress_cond["value"])

    rows, offset, batch = [], 0, 1000
    while True:
        resp = base.range(offset, offset + batch - 1).execute()
        chunk = resp.data or []
        rows.extend(chunk)
        if len(chunk) < batch:
            break
        offset += batch

    historical_entry_n = len(rows)
    unique_mints = len({r["token_address"] for r in rows})
    unique_days = len({r["alert_time"][:10] for r in rows if r.get("alert_time")})

    venue_qualified = [r for r in rows if r.get("venue_state_at_signal") == "CURVE_ACTIVE"]
    forward_venue_qualified_n = len(venue_qualified)
    venue_qualified_unique_mints = len({r["token_address"] for r in venue_qualified})
    venue_qualified_unique_days = len({r["alert_time"][:10] for r in venue_qualified if r.get("alert_time")})

    dist = {"<50%": 0, "50-70%": 0, "70-85%": 0, "85%+": 0, "unknown": 0}
    for r in rows:
        dist[_progress_bucket(r.get("progress_at_signal"))] += 1

    return CandidateForwardEvidence(
        candidate_id=candidate["candidate_id"],
        historical_entry_n=historical_entry_n,
        unique_mints=unique_mints,
        unique_days=unique_days,
        forward_venue_qualified_n=forward_venue_qualified_n,
        venue_qualified_unique_mints=venue_qualified_unique_mints,
        venue_qualified_unique_days=venue_qualified_unique_days,
        progress_distribution=dist,
    )


def _read_execution_proxy_log(log_path: Path) -> list:
    if not log_path.exists():
        return []
    import json
    out = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _read_admission_log(log_path: Path) -> list:
    if not log_path.exists():
        return []
    import json
    out = []
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _estimate_velocity(events_by_day: dict, window_days: int) -> tuple:
    """Returns (rate_per_day, stability_label). events_by_day:
    {date_str: count}. Never predicts a date -- callers only get a
    rate + stability label."""
    if len(events_by_day) < _MIN_DAYS_FOR_VELOCITY_ESTIMATE:
        return None, "INSUFFICIENT_HISTORY"

    counts = list(events_by_day.values())
    mean = sum(counts) / len(counts)
    if mean == 0:
        return 0.0, "INSUFFICIENT_HISTORY"
    variance = sum((c - mean) ** 2 for c in counts) / len(counts)
    stdev = variance ** 0.5
    cv = stdev / mean if mean > 0 else float("inf")

    if cv > _VELOCITY_CV_UNSTABLE_THRESHOLD:
        return round(mean, 2), "TOO_VARIABLE_TO_PROJECT"
    return round(mean, 2), "STABLE"


def compute_accumulation_velocity(
    venue_qualified_rows: list, valid_path_dates: list, execution_proxy_rows: list,
    window_days: int = _ACCUMULATION_WINDOW_DAYS,
) -> AccumulationVelocity:
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)

    venue_by_day: dict = {}
    for r in venue_qualified_rows:
        at = r.get("alert_time")
        if not at:
            continue
        try:
            dt = datetime.fromisoformat(at.replace("Z", "+00:00"))
        except ValueError:
            continue
        if dt < cutoff:
            continue
        venue_by_day[dt.strftime("%Y-%m-%d")] = venue_by_day.get(dt.strftime("%Y-%m-%d"), 0) + 1

    path_by_day: dict = {}
    for d in valid_path_dates:
        path_by_day[d] = path_by_day.get(d, 0) + 1

    proxy_by_day: dict = {}
    for r in execution_proxy_rows:
        obs_at = r.get("observed_at")
        if not obs_at:
            continue
        day = obs_at[:10]
        proxy_by_day[day] = proxy_by_day.get(day, 0) + 1

    venue_rate, venue_stability = _estimate_velocity(venue_by_day, window_days)
    path_rate, path_stability = _estimate_velocity(path_by_day, window_days)
    proxy_rate, proxy_stability = _estimate_velocity(proxy_by_day, window_days)

    labels = [venue_stability, path_stability, proxy_stability]
    if "INSUFFICIENT_HISTORY" in labels and all(l in ("INSUFFICIENT_HISTORY", "STABLE") for l in labels):
        overall = "INSUFFICIENT_HISTORY" if labels.count("INSUFFICIENT_HISTORY") == len(labels) else "STABLE"
    elif "TOO_VARIABLE_TO_PROJECT" in labels:
        overall = "TOO_VARIABLE_TO_PROJECT"
    else:
        overall = "STABLE"

    return AccumulationVelocity(
        new_venue_qualified_per_day=venue_rate,
        new_valid_representative_paths_per_day=path_rate,
        new_execution_proxy_observations_per_day=proxy_rate,
        stability_label=overall,
        window_days=window_days,
        note="Descriptive rate only -- no completion date is projected when the "
             "observed rate is unstable or history is too short.",
    )


def build_report(sb, repo_root: Optional[Path] = None) -> ForwardReadinessReport:
    root = repo_root or Path(__file__).parent.parent

    candidate_evidence = [_query_candidate_evidence(sb, c) for c in CANDIDATES]

    path_report = scan_corpus(root / "logs" / "research_paths")

    execution_proxy_rows = _read_execution_proxy_log(
        root / "logs" / "research_execution_proxy" / "execution_proxy_log.jsonl")
    execution_proxy_total = len(execution_proxy_rows)
    total_venue_qualified = sum(e.forward_venue_qualified_n for e in candidate_evidence)
    execution_proxy_coverage_pct = (
        round(execution_proxy_total / total_venue_qualified * 100, 2) if total_venue_qualified else 0.0
    )

    readiness_matrix = []
    for cev, candidate in zip(candidate_evidence, CANDIDATES):
        requires_venue = any(c["feature"] == "venue_state_at_signal" for c in candidate["conditions"])
        for exit_c in EXIT_CANDIDATES:
            inputs = ReadinessInputs(
                candidate_id=cev.candidate_id, exit_id=exit_c["exit_id"],
                requires_venue_state=requires_venue,
                historical_entry_n=cev.historical_entry_n, unique_mints=cev.unique_mints,
                unique_days=cev.unique_days,
                forward_venue_qualified_n=cev.forward_venue_qualified_n,
                venue_qualified_unique_mints=cev.venue_qualified_unique_mints,
                venue_qualified_unique_days=cev.venue_qualified_unique_days,
                train_n=0, validation_n=0, holdout_n=0, boundary_purged_n=0,
                representative_path_n=path_report.valid,
                path_coverage_pct=0.0,
                cost_model_available=True,
                entry_slippage_measured=execution_proxy_total > 0,
            )
            readiness_matrix.append(assess_readiness(inputs))

    # Accumulation velocity needs raw venue-qualified rows with alert_time,
    # re-fetched compactly (candidate_evidence only has counts).
    resp = (sb.table("research_tokens").select("alert_time")
            .eq("chain", "solana").eq("venue_state_at_signal", "CURVE_ACTIVE")
            .execute())
    venue_qualified_rows = resp.data or []

    valid_path_dates = []
    for d, counts in path_report.by_date.items():
        valid_path_dates.extend([d] * counts.get("VALID", 0))

    velocity = compute_accumulation_velocity(venue_qualified_rows, valid_path_dates, execution_proxy_rows)

    selection_data_ready = any(r.full_eval_ready for r in readiness_matrix)

    return ForwardReadinessReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        report_version=FORWARD_READINESS_REPORT_VERSION,
        candidate_evidence=candidate_evidence,
        readiness_matrix=readiness_matrix,
        path_integrity=path_report,
        execution_proxy_observations_total=execution_proxy_total,
        execution_proxy_coverage_pct=execution_proxy_coverage_pct,
        accumulation_velocity=velocity,
        selection_data_ready=selection_data_ready,
    )


def print_report(report: ForwardReadinessReport) -> None:
    print(f"\n{'=' * 72}")
    print(f"  V8 FORWARD READINESS REPORT v{report.report_version} — {report.generated_at}")
    print(f"{'=' * 72}")
    print(f"  holdout_evaluated: {report.holdout_evaluated}  (structurally always False here)")

    print(f"\n{'─' * 72}\n  Per-candidate forward evidence\n{'─' * 72}")
    for e in report.candidate_evidence:
        print(f"\n  {e.candidate_id}")
        print(f"    historical_entry_n={e.historical_entry_n}  unique_mints={e.unique_mints}  unique_days={e.unique_days}")
        print(f"    forward_venue_qualified_n={e.forward_venue_qualified_n}  "
              f"venue_qualified_unique_mints={e.venue_qualified_unique_mints}  "
              f"venue_qualified_unique_days={e.venue_qualified_unique_days}")
        print(f"    progress_distribution={e.progress_distribution}")

    print(f"\n{'─' * 72}\n  Path integrity\n{'─' * 72}")
    print(f"  total={report.path_integrity.total_paths}  valid={report.path_integrity.valid}  "
          f"invalid={report.path_integrity.invalid}  unknown={report.path_integrity.unknown}")

    print(f"\n{'─' * 72}\n  Execution-proxy coverage\n{'─' * 72}")
    print(f"  total observations={report.execution_proxy_observations_total}  "
          f"coverage={report.execution_proxy_coverage_pct}% of venue-qualified events")

    print(f"\n{'─' * 72}\n  Accumulation velocity ({report.accumulation_velocity.window_days}d window)\n{'─' * 72}")
    v = report.accumulation_velocity
    print(f"  new_venue_qualified/day={v.new_venue_qualified_per_day}  "
          f"new_valid_paths/day={v.new_valid_representative_paths_per_day}  "
          f"new_execution_proxy_obs/day={v.new_execution_proxy_observations_per_day}")
    print(f"  stability: {v.stability_label}  -- {v.note}")

    print(f"\n{'─' * 72}\n  Readiness matrix ({len(report.readiness_matrix)} candidate x exit pairs)\n{'─' * 72}")
    for r in report.readiness_matrix:
        print(f"  {r.candidate_id:<12} x {r.exit_id:<4}  progress_ready={r.progress_evidence_ready}  "
              f"full_entry_rule_ready={r.full_entry_rule_ready}  full_eval_ready={r.full_eval_ready}")

    print(f"\n{'=' * 72}")
    print(f"  SELECTION_DATA_READY = {report.selection_data_ready}")
    print(f"{'=' * 72}\n")


def main():
    from supabase import create_client
    from research.config import SUPABASE_URL, SUPABASE_KEY
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    report = build_report(sb)
    print_report(report)


if __name__ == "__main__":
    main()
