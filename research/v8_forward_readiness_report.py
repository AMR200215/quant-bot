"""
research/v8_forward_readiness_report.py — V8 DATA RECOVERY batch item
9, CORRECTED by the V8 READINESS MONITOR CORRECTION batch: ONE
automatic, read-only forward-readiness report for the frozen v1
experiment against CURRENT FORWARD DATA. Never opens/prints holdout
results -- this report only computes data-sufficiency counts (venue-
qualified n, unique mints/days, progress distribution, real train/
validation/holdout/boundary-purge split counts, candidate-specific
path coverage, candidate-specific execution-proxy coverage), never
holdout OUTCOME/PnL data (counting how many eligible rows fall in the
holdout window is fine; reading their pct_change_peak is not, and this
module never does).

READINESS MONITOR CORRECTION (2026-08-19) fixed three real bugs in the
original version:

1. train_n/validation_n/holdout_n/path_coverage_pct were hardcoded to 0
   for every candidate, meaning SELECTION_DATA_READY could never
   become true no matter how much data accumulated. Now computed for
   real via research/v8_split.py's frozen grouped_chronological_split,
   applied to each candidate's venue-qualified population.
2. representative_path_n used ALL VALID paths in the WHOLE corpus as
   every candidate's number -- wrong, since candidates have different
   eligible populations. Now computed per candidate via research/
   v8_candidate_path_coverage.py, joining through research_tokens'
   own path_file column with ambiguous-mint exclusion and real entry-
   time alignment.
3. entry_slippage_measured was execution_proxy_total > 0 -- one
   observation is not evidence. Now research/
   v8_execution_proxy_readiness.py's EXECUTION_PROXY_READY, requiring
   both a minimum count AND minimum coverage, computed per candidate
   (never summed across candidates -- an event qualifying for both
   V8-P0 and V8-P3 is not double-counted in any global denominator
   because there is no global denominator here at all).

Reuses the frozen v1 registries and the existing readiness engine
(research/v8_readiness_engine.py) as a live-data populator, not a
redesign.

Run:
    python -m research.v8_forward_readiness_report
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from research.v8_candidate_registry import CANDIDATES
from research.v8_exit_registry import EXIT_CANDIDATES
from research.v8_readiness_engine import (
    ReadinessInputs, ReadinessReport, assess_readiness,
    MIN_POLL_OUTCOME_N, MIN_POLL_OUTCOME_COVERAGE_PCT,
)
from research.v8_path_integrity import scan_corpus, CorpusIntegrityReport
from research.v8_split import grouped_chronological_split
from research.v8_candidate_path_coverage import compute_candidate_path_coverage, CandidatePathCoverage
from research.v8_execution_proxy_readiness import assess_execution_proxy_readiness, ExecutionProxyCoverage
from research.v8_collection_yield import compute_collection_yield, CollectionYield, load_admission_log_by_mint

FORWARD_READINESS_REPORT_VERSION = 3

_ACCUMULATION_WINDOW_DAYS = 7
_MIN_DAYS_FOR_VELOCITY_ESTIMATE = 2
_VELOCITY_CV_UNSTABLE_THRESHOLD = 0.75

# Minimum independent VALID day-blocks for the statistical machinery
# (research/v8_statistical_selection.py's block_bootstrap_ci) to be
# able to compute anything at all.
_MIN_DAY_BLOCKS_FOR_STATISTICS = 2


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


def _alert_time_to_epoch(alert_time: str) -> Optional[float]:
    if not alert_time:
        return None
    try:
        dt = datetime.fromisoformat(alert_time.replace("Z", "+00:00"))
        return dt.timestamp()
    except ValueError:
        return None


@dataclass(frozen=True)
class AccumulationVelocity:
    new_venue_qualified_per_day: Optional[float]
    new_valid_representative_paths_per_day: Optional[float]
    new_execution_proxy_observations_per_day: Optional[float]
    stability_label: str
    window_days: int
    note: str


@dataclass(frozen=True)
class SplitCounts:
    train_n: int
    validation_n: int
    holdout_n: int
    boundary_purged_n: int
    group_count: int


@dataclass(frozen=True)
class DiagnosticsFeasibility:
    computable: bool
    train_validation_outcome_n: int
    train_validation_valid_day_blocks: int
    reasons: list


@dataclass(frozen=True)
class CandidateForwardEvidence:
    candidate_id: str
    historical_entry_n: int
    unique_mints: int
    unique_days: int
    forward_venue_qualified_n: int
    venue_qualified_unique_mints: int
    venue_qualified_unique_days: int
    progress_distribution: dict
    split_counts: SplitCounts
    path_coverage: CandidatePathCoverage           # population-denominator, all history (context only)
    execution_proxy_coverage: ExecutionProxyCoverage  # population-denominator, all history (context only)
    collection_yield: CollectionYield              # era/admission-restricted -- feeds exit-derivation gating
    poll_outcome_coverage: PollOutcomeCoverage     # YD2: train+val only -- feeds SELECTION gating
    diagnostics_feasibility: DiagnosticsFeasibility


@dataclass(frozen=True)
class ForwardReadinessReport:
    generated_at: str
    report_version: int
    candidate_evidence: list
    readiness_matrix: list
    path_integrity: CorpusIntegrityReport
    accumulation_velocity: AccumulationVelocity
    selection_data_ready: bool
    selection_data_ready_candidates: list   # candidate_ids for which every gate passed
    holdout_evaluated: bool = False


def _fetch_all_clean_events(sb) -> list:
    """The full superset every candidate's eligible population is drawn
    from -- chain=solana, progress_data_ok=True, no progress-value
    filter. Reused for ambiguous-mint detection across ALL candidates
    (ambiguity is a property of a mint's total alert history, not any
    one candidate's filtered subset)."""
    rows, offset, batch = [], 0, 1000
    while True:
        resp = (sb.table("research_tokens")
                .select("event_id,token_address,alert_time,progress_at_signal,"
                        "venue_state_at_signal,path_file,progress_capture_lag_ms,pct_change_peak")
                .eq("chain", "solana").eq("progress_data_ok", True)
                .range(offset, offset + batch - 1).execute())
        chunk = resp.data or []
        rows.extend(chunk)
        if len(chunk) < batch:
            break
        offset += batch
    return rows


def _filter_candidate_events(all_events: list, candidate: dict) -> list:
    progress_cond = next((c for c in candidate["conditions"] if c["feature"] == "progress_at_signal"), None)
    if progress_cond is None:
        return list(all_events)
    return [r for r in all_events if r.get("progress_at_signal") is not None
            and r["progress_at_signal"] < progress_cond["value"]]


def _compute_split_counts(venue_qualified_events: list) -> SplitCounts:
    events_with_time = [r for r in venue_qualified_events if r.get("alert_time")]
    if len(events_with_time) < 2:
        return SplitCounts(train_n=0, validation_n=0, holdout_n=0, boundary_purged_n=0, group_count=0)

    rows_for_split = [
        {"token_address": r["token_address"], "_epoch": _alert_time_to_epoch(r["alert_time"])}
        for r in events_with_time
    ]
    rows_for_split = [r for r in rows_for_split if r["_epoch"] is not None]
    if len(rows_for_split) < 2:
        return SplitCounts(train_n=0, validation_n=0, holdout_n=0, boundary_purged_n=0, group_count=0)

    try:
        result = grouped_chronological_split(
            rows_for_split, lambda r: r["token_address"], lambda r: r["_epoch"],
        )
    except ValueError:
        return SplitCounts(train_n=0, validation_n=0, holdout_n=0, boundary_purged_n=0, group_count=0)

    return SplitCounts(
        train_n=len(result.train), validation_n=len(result.validation), holdout_n=len(result.holdout),
        boundary_purged_n=result.purged_rows, group_count=result.group_count,
    )


def _compute_diagnostics_feasibility(venue_qualified_events: list) -> DiagnosticsFeasibility:
    """
    Feasibility check ONLY -- counts whether train+validation rows have
    enough non-null outcome data spread across enough days for research/
    v8_statistical_selection.py's diagnostics to be computable. NEVER
    reads/prints/uses the actual pct_change_peak VALUES, and never
    touches holdout rows at all (the split's holdout bucket is not
    consulted here in any way).
    """
    reasons = []
    events_with_time = [r for r in venue_qualified_events if r.get("alert_time")]
    rows_for_split = [
        {"token_address": r["token_address"], "_epoch": _alert_time_to_epoch(r["alert_time"]), "_orig": r}
        for r in events_with_time
    ]
    rows_for_split = [r for r in rows_for_split if r["_epoch"] is not None]

    if len(rows_for_split) < 2:
        reasons.append("fewer than 2 venue-qualified events with parseable alert_time")
        return DiagnosticsFeasibility(computable=False, train_validation_outcome_n=0,
                                       train_validation_valid_day_blocks=0, reasons=reasons)

    try:
        result = grouped_chronological_split(
            rows_for_split, lambda r: r["token_address"], lambda r: r["_epoch"],
        )
    except ValueError as e:
        reasons.append(str(e))
        return DiagnosticsFeasibility(computable=False, train_validation_outcome_n=0,
                                       train_validation_valid_day_blocks=0, reasons=reasons)

    train_validation_rows = result.train + result.validation
    outcome_rows = [r["_orig"] for r in train_validation_rows if r["_orig"].get("pct_change_peak") is not None]
    outcome_n = len(outcome_rows)

    days_with_outcome = {r["alert_time"][:10] for r in outcome_rows if r.get("alert_time")}
    n_day_blocks = len(days_with_outcome)

    if outcome_n == 0:
        reasons.append("zero train/validation rows have a non-null pct_change_peak")
    if n_day_blocks < _MIN_DAY_BLOCKS_FOR_STATISTICS:
        reasons.append(f"only {n_day_blocks} day-block(s) with outcome data -- "
                        f"block_bootstrap_ci needs >= {_MIN_DAY_BLOCKS_FOR_STATISTICS}")

    computable = outcome_n > 0 and n_day_blocks >= _MIN_DAY_BLOCKS_FOR_STATISTICS
    return DiagnosticsFeasibility(
        computable=computable, train_validation_outcome_n=outcome_n,
        train_validation_valid_day_blocks=n_day_blocks, reasons=reasons,
    )


@dataclass(frozen=True)
class PollOutcomeCoverage:
    """YD2 (docs/READINESS_RESCOPE_PROPOSAL.md): SELECTION (entry-EV)
    readiness evidence -- research/outcome_poller.py's poll-based
    pct_change_peak, never a path file. Computed on train+validation
    rows ONLY (holdout never touched), same discipline as
    _compute_diagnostics_feasibility."""
    eligible_n: int      # train+validation venue-qualified rows
    observed_n: int       # of those, pct_change_peak IS NOT NULL
    coverage_pct: float
    ready: bool
    reasons: list


def _compute_poll_outcome_coverage(venue_qualified_events: list) -> PollOutcomeCoverage:
    events_with_time = [r for r in venue_qualified_events if r.get("alert_time")]
    rows_for_split = [
        {"token_address": r["token_address"], "_epoch": _alert_time_to_epoch(r["alert_time"]), "_orig": r}
        for r in events_with_time
    ]
    rows_for_split = [r for r in rows_for_split if r["_epoch"] is not None]

    if len(rows_for_split) < 2:
        return PollOutcomeCoverage(eligible_n=0, observed_n=0, coverage_pct=0.0, ready=False,
                                    reasons=["fewer than 2 venue-qualified events with parseable alert_time"])

    try:
        result = grouped_chronological_split(
            rows_for_split, lambda r: r["token_address"], lambda r: r["_epoch"],
        )
    except ValueError as e:
        return PollOutcomeCoverage(eligible_n=0, observed_n=0, coverage_pct=0.0, ready=False, reasons=[str(e)])

    train_validation_rows = [r["_orig"] for r in (result.train + result.validation)]
    eligible_n = len(train_validation_rows)
    observed_n = sum(1 for r in train_validation_rows if r.get("pct_change_peak") is not None)
    coverage_pct = round(100 * observed_n / eligible_n, 2) if eligible_n else 0.0

    reasons = []
    if observed_n < MIN_POLL_OUTCOME_N:
        reasons.append(f"poll_outcome_n={observed_n} < MIN_POLL_OUTCOME_N={MIN_POLL_OUTCOME_N}")
    if coverage_pct < MIN_POLL_OUTCOME_COVERAGE_PCT:
        reasons.append(f"poll_outcome_coverage_pct={coverage_pct} < MIN_POLL_OUTCOME_COVERAGE_PCT={MIN_POLL_OUTCOME_COVERAGE_PCT}")

    return PollOutcomeCoverage(
        eligible_n=eligible_n, observed_n=observed_n, coverage_pct=coverage_pct,
        ready=not reasons, reasons=reasons,
    )


def _read_jsonl(log_path: Path) -> list:
    if not log_path.exists():
        return []
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


def _compute_execution_proxy_coverage(venue_qualified_events: list, execution_proxy_rows: list) -> ExecutionProxyCoverage:
    # READINESS DENOMINATOR AUDIT: eligible_n was event-count while
    # observed_n (below) is mint-count -- not commensurate whenever a
    # mint has more than one alert in the population. admission_log.jsonl
    # and execution_proxy_log.jsonl are both keyed by token_address only
    # (no event_id carried), so observed_n can only ever be mint-level;
    # eligible_n is now made mint-level too so numerator and denominator
    # are never mixed units. This is a population-level count (all
    # history, unchanged in scope) -- see research/v8_collection_yield.py
    # for the era/admission-restricted collector-yield view used for
    # actual readiness gating.
    eligible_mints = {r["token_address"] for r in venue_qualified_events}
    eligible_n = len(eligible_mints)

    # Dedup by (token_address) since the collector fires once per mint --
    # event_id isn't carried on every proxy row reliably yet, so mint is
    # the best available identity here; ambiguous-mint concerns don't
    # apply the same way for this coverage count (a mint appearing twice
    # in venue_qualified_events is already reflected in eligible_n).
    observed_mints = {r.get("token_address") for r in execution_proxy_rows
                       if r.get("token_address") in eligible_mints and r.get("status") == "OK"}
    observed_n = len(observed_mints)

    observed_days = {r.get("observed_at", "")[:10] for r in execution_proxy_rows
                      if r.get("token_address") in eligible_mints and r.get("status") == "OK"}
    unique_days = len({d for d in observed_days if d})

    return assess_execution_proxy_readiness(eligible_n=eligible_n, observed_n=observed_n, unique_days=unique_days)


def _query_candidate_evidence(candidate: dict, all_events: list, execution_proxy_rows: list,
                               repo_root: Path, admission_log_by_mint: Optional[dict] = None) -> CandidateForwardEvidence:
    candidate_events = _filter_candidate_events(all_events, candidate)

    historical_entry_n = len(candidate_events)
    unique_mints = len({r["token_address"] for r in candidate_events})
    unique_days = len({r["alert_time"][:10] for r in candidate_events if r.get("alert_time")})

    venue_qualified = [r for r in candidate_events if r.get("venue_state_at_signal") == "CURVE_ACTIVE"]
    forward_venue_qualified_n = len(venue_qualified)
    venue_qualified_unique_mints = len({r["token_address"] for r in venue_qualified})
    venue_qualified_unique_days = len({r["alert_time"][:10] for r in venue_qualified if r.get("alert_time")})

    dist = {"<50%": 0, "50-70%": 0, "70-85%": 0, "85%+": 0, "unknown": 0}
    for r in candidate_events:
        dist[_progress_bucket(r.get("progress_at_signal"))] += 1

    split_counts = _compute_split_counts(venue_qualified)
    path_coverage = compute_candidate_path_coverage(venue_qualified, all_events, candidate, repo_root)
    execution_proxy_coverage = _compute_execution_proxy_coverage(venue_qualified, execution_proxy_rows)
    collection_yield = compute_collection_yield(
        venue_qualified, all_events, candidate, repo_root,
        admission_log_by_mint=admission_log_by_mint, execution_proxy_rows=execution_proxy_rows,
    )
    poll_outcome_coverage = _compute_poll_outcome_coverage(venue_qualified)
    diagnostics = _compute_diagnostics_feasibility(venue_qualified)

    return CandidateForwardEvidence(
        candidate_id=candidate["candidate_id"],
        historical_entry_n=historical_entry_n, unique_mints=unique_mints, unique_days=unique_days,
        forward_venue_qualified_n=forward_venue_qualified_n,
        venue_qualified_unique_mints=venue_qualified_unique_mints,
        venue_qualified_unique_days=venue_qualified_unique_days,
        progress_distribution=dist,
        split_counts=split_counts, path_coverage=path_coverage,
        execution_proxy_coverage=execution_proxy_coverage, collection_yield=collection_yield,
        poll_outcome_coverage=poll_outcome_coverage,
        diagnostics_feasibility=diagnostics,
    )


def _estimate_velocity(events_by_day: dict, window_days: int) -> tuple:
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
        day = dt.strftime("%Y-%m-%d")
        venue_by_day[day] = venue_by_day.get(day, 0) + 1

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
    if "TOO_VARIABLE_TO_PROJECT" in labels:
        overall = "TOO_VARIABLE_TO_PROJECT"
    elif all(l == "INSUFFICIENT_HISTORY" for l in labels):
        overall = "INSUFFICIENT_HISTORY"
    else:
        overall = "STABLE"

    return AccumulationVelocity(
        new_venue_qualified_per_day=venue_rate,
        new_valid_representative_paths_per_day=path_rate,
        new_execution_proxy_observations_per_day=proxy_rate,
        stability_label=overall, window_days=window_days,
        note="Descriptive rate only -- no completion date is projected when the "
             "observed rate is unstable or history is too short.",
    )


def _collection_yield_execution_proxy_ready(cy: CollectionYield) -> ExecutionProxyCoverage:
    """Re-evaluates the SAME, unmodified execution-proxy thresholds
    (EXECUTION_PROXY_MIN_N / EXECUTION_PROXY_MIN_COVERAGE_PCT) against
    the corrected, era/admission-restricted denominator instead of the
    whole historical population."""
    return assess_execution_proxy_readiness(
        eligible_n=cy.execution_proxy_collection_eligible_n,
        observed_n=cy.execution_proxy_observed_n,
        unique_days=cy.unique_forward_days,
    )


def _candidate_selection_ready(evidence: CandidateForwardEvidence, requires_venue: bool) -> bool:
    """YD2 (docs/READINESS_RESCOPE_PROPOSAL.md): SELECTION (entry-EV)
    readiness now comes directly from the engine's own
    selection_data_ready flag -- poll-outcome-keyed
    (evidence.poll_outcome_coverage, train+validation only), never
    path-keyed. representative_path_n/path_coverage_pct are still
    passed through (still needed for exit_derivation_data_ready, kept
    for context/downstream use) but no longer gate SELECTION at all.
    diagnostics_feasibility.computable is kept as an additional
    requirement since it is ALSO poll/outcome-based (train+validation
    pct_change_peak spread across enough day-blocks for research/
    v8_statistical_selection.py), not path-based -- it belongs on the
    SELECTION side, not the exit-derivation side."""
    cy = evidence.collection_yield
    poc = evidence.poll_outcome_coverage

    inputs = ReadinessInputs(
        candidate_id=evidence.candidate_id, exit_id="E0", requires_venue_state=requires_venue,
        historical_entry_n=evidence.historical_entry_n, unique_mints=evidence.unique_mints,
        unique_days=evidence.unique_days,
        forward_venue_qualified_n=evidence.forward_venue_qualified_n,
        venue_qualified_unique_mints=evidence.venue_qualified_unique_mints,
        venue_qualified_unique_days=evidence.venue_qualified_unique_days,
        train_n=evidence.split_counts.train_n, validation_n=evidence.split_counts.validation_n,
        holdout_n=evidence.split_counts.holdout_n, boundary_purged_n=evidence.split_counts.boundary_purged_n,
        representative_path_n=cy.admitted_with_valid_usable_path_n,
        path_coverage_pct=cy.admitted_path_yield_pct,
        poll_outcome_n=poc.observed_n,
        poll_outcome_coverage_pct=poc.coverage_pct,
        cost_model_available=True,
        entry_slippage_measured=_collection_yield_execution_proxy_ready(cy).ready,
    )
    engineering_report = assess_readiness(inputs)

    return engineering_report.selection_data_ready and evidence.diagnostics_feasibility.computable


def build_report(sb, repo_root: Optional[Path] = None) -> ForwardReadinessReport:
    root = repo_root or Path(__file__).parent.parent

    all_events = _fetch_all_clean_events(sb)
    execution_proxy_rows = _read_jsonl(root / "logs" / "research_execution_proxy" / "execution_proxy_log.jsonl")
    admission_log_by_mint = load_admission_log_by_mint(root)

    candidate_evidence = [
        _query_candidate_evidence(c, all_events, execution_proxy_rows, root, admission_log_by_mint)
        for c in CANDIDATES
    ]

    path_report = scan_corpus(root / "logs" / "research_paths")

    readiness_matrix = []
    selection_ready_candidates = []
    for cev, candidate in zip(candidate_evidence, CANDIDATES):
        requires_venue = any(c["feature"] == "venue_state_at_signal" for c in candidate["conditions"])
        candidate_selection_ready = _candidate_selection_ready(cev, requires_venue)
        if candidate_selection_ready:
            selection_ready_candidates.append(cev.candidate_id)

        cy = cev.collection_yield
        proxy_ready = _collection_yield_execution_proxy_ready(cy)
        for exit_c in EXIT_CANDIDATES:
            inputs = ReadinessInputs(
                candidate_id=cev.candidate_id, exit_id=exit_c["exit_id"],
                requires_venue_state=requires_venue,
                historical_entry_n=cev.historical_entry_n, unique_mints=cev.unique_mints,
                unique_days=cev.unique_days,
                forward_venue_qualified_n=cev.forward_venue_qualified_n,
                venue_qualified_unique_mints=cev.venue_qualified_unique_mints,
                venue_qualified_unique_days=cev.venue_qualified_unique_days,
                train_n=cev.split_counts.train_n, validation_n=cev.split_counts.validation_n,
                holdout_n=cev.split_counts.holdout_n, boundary_purged_n=cev.split_counts.boundary_purged_n,
                representative_path_n=cy.admitted_with_valid_usable_path_n,
                path_coverage_pct=cy.admitted_path_yield_pct,
                poll_outcome_n=cev.poll_outcome_coverage.observed_n,
                poll_outcome_coverage_pct=cev.poll_outcome_coverage.coverage_pct,
                cost_model_available=True,
                entry_slippage_measured=proxy_ready.ready,
            )
            readiness_matrix.append(assess_readiness(inputs))

    resp = (sb.table("research_tokens").select("alert_time")
            .eq("chain", "solana").eq("venue_state_at_signal", "CURVE_ACTIVE").execute())
    venue_qualified_rows = resp.data or []

    valid_path_dates = []
    for d, counts in path_report.by_date.items():
        valid_path_dates.extend([d] * counts.get("VALID", 0))

    velocity = compute_accumulation_velocity(venue_qualified_rows, valid_path_dates, execution_proxy_rows)

    return ForwardReadinessReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        report_version=FORWARD_READINESS_REPORT_VERSION,
        candidate_evidence=candidate_evidence,
        readiness_matrix=readiness_matrix,
        path_integrity=path_report,
        accumulation_velocity=velocity,
        selection_data_ready=len(selection_ready_candidates) > 0,
        selection_data_ready_candidates=selection_ready_candidates,
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
        sc = e.split_counts
        print(f"    split: train_n={sc.train_n} validation_n={sc.validation_n} "
              f"holdout_n={sc.holdout_n} boundary_purged_n={sc.boundary_purged_n} group_count={sc.group_count}")
        pc = e.path_coverage
        print(f"    path_coverage: eligible_n={pc.eligible_n} usable_n={pc.usable_n} "
              f"coverage_pct={pc.coverage_pct}  (ambiguous={pc.ambiguous_excluded_n} "
              f"no_path_file={pc.no_path_file_n} missing={pc.path_file_missing_or_empty_n} "
              f"alignment_excluded={pc.alignment_excluded_n} integrity_bad={pc.integrity_invalid_or_unknown_n})")
        ec = e.execution_proxy_coverage
        print(f"    execution_proxy (population): eligible_n={ec.eligible_n} observed_n={ec.observed_n} "
              f"coverage_pct={ec.coverage_pct} unique_days={ec.unique_days} ready={ec.ready}")
        if ec.reasons:
            print(f"      reasons: {ec.reasons}")

        cy = e.collection_yield
        if cy.era_undetermined:
            print(f"    collection_yield: era UNDETERMINED (no funded-collector output observed yet)")
        else:
            print(f"    collection_yield: era_start={cy.era_start}")
            print(f"      candidate_venue_qualified_n={cy.candidate_venue_qualified_n}  "
                  f"ambiguous_excluded_mints_n={cy.ambiguous_excluded_mints_n}  "
                  f"path_collection_eligible_n={cy.path_collection_eligible_n}  "
                  f"no_admission_record_n={cy.no_admission_record_n}")
            print(f"      path_admitted_n={cy.path_admitted_n}  admitted_with_tick_n={cy.admitted_with_tick_n}  "
                  f"admitted_with_valid_usable_path_n={cy.admitted_with_valid_usable_path_n}  "
                  f"admitted_path_yield_pct={cy.admitted_path_yield_pct}  ipw_effective_n={cy.ipw_effective_n}")
            proxy_ready = _collection_yield_execution_proxy_ready(cy)
            print(f"      execution_proxy (collector-yield): eligible_n={cy.execution_proxy_collection_eligible_n} "
                  f"observed_n={cy.execution_proxy_observed_n} coverage_pct={cy.execution_proxy_coverage_pct} "
                  f"unique_forward_days={cy.unique_forward_days} ready={proxy_ready.ready}")
            if proxy_ready.reasons:
                print(f"        reasons: {proxy_ready.reasons}")
        poc = e.poll_outcome_coverage
        print(f"    poll_outcome (SELECTION, train+val only): eligible_n={poc.eligible_n} "
              f"observed_n={poc.observed_n} coverage_pct={poc.coverage_pct} ready={poc.ready}")
        if poc.reasons:
            print(f"      reasons: {poc.reasons}")

        df = e.diagnostics_feasibility
        print(f"    diagnostics_feasibility: computable={df.computable} "
              f"train_validation_outcome_n={df.train_validation_outcome_n} "
              f"day_blocks={df.train_validation_valid_day_blocks}")
        if df.reasons:
            print(f"      reasons: {df.reasons}")

    print(f"\n{'─' * 72}\n  Path integrity (full corpus)\n{'─' * 72}")
    print(f"  total={report.path_integrity.total_paths}  valid={report.path_integrity.valid}  "
          f"invalid={report.path_integrity.invalid}  unknown={report.path_integrity.unknown}")

    print(f"\n{'─' * 72}\n  Accumulation velocity ({report.accumulation_velocity.window_days}d window)\n{'─' * 72}")
    v = report.accumulation_velocity
    print(f"  new_venue_qualified/day={v.new_venue_qualified_per_day}  "
          f"new_valid_paths/day={v.new_valid_representative_paths_per_day}  "
          f"new_execution_proxy_obs/day={v.new_execution_proxy_observations_per_day}")
    print(f"  stability: {v.stability_label}  -- {v.note}")

    print(f"\n{'─' * 72}\n  Readiness matrix ({len(report.readiness_matrix)} candidate x exit pairs)\n{'─' * 72}")
    for r in report.readiness_matrix:
        print(f"  {r.candidate_id:<12} x {r.exit_id:<4}  progress_ready={r.progress_evidence_ready}  "
              f"full_entry_rule_ready={r.full_entry_rule_ready}  "
              f"selection_data_ready={r.selection_data_ready}  "
              f"exit_derivation_data_ready={r.exit_derivation_data_ready}  "
              f"full_eval_ready={r.full_eval_ready}")

    print(f"\n{'=' * 72}")
    print(f"  SELECTION_DATA_READY = {report.selection_data_ready}")
    print(f"  ready candidates: {report.selection_data_ready_candidates}")
    print(f"{'=' * 72}\n")


def main():
    from supabase import create_client
    from research.config import SUPABASE_URL, SUPABASE_KEY
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    report = build_report(sb)
    print_report(report)


if __name__ == "__main__":
    main()
