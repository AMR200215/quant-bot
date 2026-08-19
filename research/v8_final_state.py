"""
research/v8_final_state.py — V8 DATA RECOVERY batch item 11, CORRECTED
by the V8 READINESS MONITOR CORRECTION batch: the ONE final state
machine. No further micro-phases after this -- exactly three booleans,
precisely defined, never conflated:

  ENGINE_READY
      = the already-frozen filter/exit/replay/statistics machinery is
        complete. Checked structurally.

  FORWARD_DATA_PIPELINE_HEALTHY
      = venue state, progress, representative path, corrected price and
        execution-proxy data are ALL CURRENTLY flowing without a known
        systematic corruption.

      READINESS MONITOR CORRECTION (2026-08-19): the original version
      only checked recent live_pp path integrity -- one component out
      of five. A silently-dead execution-proxy collector, or a stalled
      progress/venue_state capture pipeline, could not have been
      detected even though the definition explicitly names all five
      streams. Now genuinely independent per-component checks:
        A. progress_at_signal flow
        B. venue_state_at_signal flow
        C. post-fix representative live_pp paths (existence/volume)
        D. path integrity / corrected prices (quality, on the same rows as C)
        E. execution-proxy observation flow
      Aggregation: any UNHEALTHY -> UNHEALTHY; no UNHEALTHY but any
      UNKNOWN -> UNKNOWN; all HEALTHY -> HEALTHY. Low natural yield
      produces UNKNOWN for that component, never UNHEALTHY -- but it
      still prevents the OVERALL status from reaching HEALTHY, per the
      explicit "a silent/dead execution-proxy collector must prevent
      HEALTHY" instruction.

  SELECTION_DATA_READY
      = at least one frozen (candidate, exit) pair has enough forward,
        venue-qualified, integrity-valid, representative, execution-
        cost-evidenced data to justify the one-shot holdout selection
        experiment.

      READINESS MONITOR CORRECTION: now delegates to research/
      v8_forward_readiness_report.py's real (not hardcoded-zero) split
      counts, candidate-specific path coverage, and per-candidate
      EXECUTION_PROXY_READY -- the report's own selection_data_ready
      field already encodes every required gate (FULL_ENTRY_RULE_READY,
      PATH_DATA_READY, EXECUTION_PROXY_READY, non-degenerate train/
      validation/holdout counts, and pre-holdout statistical-diagnostics
      feasibility). This module does not recompute that logic a second
      time -- it reads the one, single, already-correct source.

Does NOT open, evaluate, or print holdout results. Never will -- no
function in this module accepts a holdout dataset as an argument, and
research/v8_forward_readiness_report.py's own diagnostics-feasibility
check never reads its split result's holdout bucket either (see that
module's own tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Post-fix live capture (V8 DATA RECOVERY batch, 2026-08-19): 10/10 real
# fresh CURVE_ACTIVE ticks were VALID. Pre-fix baseline (Phase 2.1):
# 51.5% invalid. Set well below any plausible "the bug recurred" rate.
RECENT_INVALID_RATE_UNHEALTHY_THRESHOLD_PCT = 15.0
RECENT_WINDOW_HOURS = 24
MIN_RECENT_ROWS_TO_JUDGE = 5   # below this, report UNKNOWN rather than guess, for every component below


@dataclass(frozen=True)
class ComponentHealth:
    name: str
    status: str          # "HEALTHY" | "UNHEALTHY" | "UNKNOWN"
    n_checked: int
    detail: str
    note: str


@dataclass(frozen=True)
class EngineReadyCheck:
    ready: bool
    entry_registry_ok: bool
    exit_registry_ok: bool
    holdout_lock_ok: bool
    statistical_module_ok: bool
    reasons: list


@dataclass(frozen=True)
class ForwardPipelineHealthCheck:
    status: str   # overall: "HEALTHY" | "UNHEALTHY" | "UNKNOWN"
    components: list   # list[ComponentHealth], A-E in order
    window_hours: int


@dataclass(frozen=True)
class V8FinalState:
    generated_at: str
    engine_ready: bool
    forward_data_pipeline_healthy: Optional[bool]   # None means UNKNOWN, never silently True
    selection_data_ready: bool
    engine_detail: EngineReadyCheck
    pipeline_detail: ForwardPipelineHealthCheck
    selection_data_ready_candidates: list
    holdout_evaluated: bool = False


def check_engine_ready() -> EngineReadyCheck:
    reasons = []
    entry_ok = exit_ok = lock_ok = stats_ok = False

    try:
        from research.v8_candidate_registry import assert_registry_frozen
        assert_registry_frozen()
        entry_ok = True
    except Exception as e:
        reasons.append(f"entry registry not frozen/importable: {e}")

    try:
        from research.v8_exit_registry import assert_registry_frozen as assert_exit_frozen
        assert_exit_frozen()
        exit_ok = True
    except Exception as e:
        reasons.append(f"exit registry not frozen/importable: {e}")

    try:
        from research.v8_experiment_manifest import ExperimentManifest, assert_holdout_not_evaluated
        m = ExperimentManifest()
        assert_holdout_not_evaluated(m)
        lock_ok = True
    except Exception as e:
        reasons.append(f"holdout lock mechanism not importable/working: {e}")

    try:
        import research.v8_statistical_selection  # noqa: F401
        stats_ok = True
    except Exception as e:
        reasons.append(f"statistical-selection module not importable: {e}")

    ready = entry_ok and exit_ok and lock_ok and stats_ok
    return EngineReadyCheck(
        ready=ready, entry_registry_ok=entry_ok, exit_registry_ok=exit_ok,
        holdout_lock_ok=lock_ok, statistical_module_ok=stats_ok, reasons=reasons,
    )


def _recent_dirs(paths_dir: Path) -> list:
    if not paths_dir.exists():
        return []
    today = datetime.now(timezone.utc)
    return [paths_dir / today.strftime("%Y-%m-%d"),
            paths_dir / (today - timedelta(days=1)).strftime("%Y-%m-%d")]


def check_live_pp_paths_flow(repo_root: Path) -> ComponentHealth:
    """Component C: are representative live_pp paths being produced at
    all recently (existence/volume, not quality -- that's component D)."""
    from research.path_schema import load_path_file

    cutoff_ms = int((datetime.now(timezone.utc) - timedelta(hours=RECENT_WINDOW_HOURS)).timestamp() * 1000)
    checked = 0
    for d in _recent_dirs(repo_root / "logs" / "research_paths"):
        if not d.exists():
            continue
        for fp in list(d.glob("*.csv")) + list(d.glob("*.csv.gz")):
            rows, _ = load_path_file(fp)
            for r in rows:
                if r.get("source") != "live_pp":
                    continue
                try:
                    ts = int(r.get("ts_ms") or 0)
                except (TypeError, ValueError):
                    continue
                if ts >= cutoff_ms:
                    checked += 1

    if checked < MIN_RECENT_ROWS_TO_JUDGE:
        return ComponentHealth(
            name="live_pp_paths_flow", status="UNKNOWN", n_checked=checked,
            detail=f"only {checked} recent live_pp rows in the last {RECENT_WINDOW_HOURS}h",
            note="natural yield is low -- UNKNOWN, not UNHEALTHY, per policy",
        )
    return ComponentHealth(
        name="live_pp_paths_flow", status="HEALTHY", n_checked=checked,
        detail=f"{checked} recent live_pp rows in the last {RECENT_WINDOW_HOURS}h", note="",
    )


def check_path_integrity_quality(repo_root: Path) -> ComponentHealth:
    """Component D: of the paths that ARE flowing, what fraction pass
    integrity -- the corrected-prices check."""
    from research.path_schema import load_path_file
    from research.v8_path_integrity import assess_tick_integrity

    cutoff_ms = int((datetime.now(timezone.utc) - timedelta(hours=RECENT_WINDOW_HOURS)).timestamp() * 1000)
    checked = invalid = 0
    for d in _recent_dirs(repo_root / "logs" / "research_paths"):
        if not d.exists():
            continue
        for fp in list(d.glob("*.csv")) + list(d.glob("*.csv.gz")):
            rows, _ = load_path_file(fp)
            for r in rows:
                if r.get("source") != "live_pp":
                    continue
                try:
                    ts = int(r.get("ts_ms") or 0)
                except (TypeError, ValueError):
                    continue
                if ts < cutoff_ms:
                    continue
                checked += 1
                if assess_tick_integrity(r).status == "INVALID":
                    invalid += 1

    if checked < MIN_RECENT_ROWS_TO_JUDGE:
        return ComponentHealth(
            name="path_integrity_quality", status="UNKNOWN", n_checked=checked,
            detail=f"only {checked} recent live_pp rows to judge",
            note="natural yield is low -- UNKNOWN, not UNHEALTHY, per policy",
        )

    invalid_rate = round(invalid / checked * 100, 2)
    status = "UNHEALTHY" if invalid_rate > RECENT_INVALID_RATE_UNHEALTHY_THRESHOLD_PCT else "HEALTHY"
    return ComponentHealth(
        name="path_integrity_quality", status=status, n_checked=checked,
        detail=f"invalid_rate={invalid_rate}% (threshold={RECENT_INVALID_RATE_UNHEALTHY_THRESHOLD_PCT}%)",
        note="post-fix live capture measured 0%; pre-fix baseline was 51.5%",
    )


def _check_supabase_column_flow(sb, column: str) -> ComponentHealth:
    """Shared logic for components A (progress_at_signal) and B
    (venue_state_at_signal): among recently-alerted, progress_data_ok
    rows, what fraction have this column populated."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=RECENT_WINDOW_HOURS)).isoformat()
    resp = (sb.table("research_tokens")
            .select(f"id,{column}")
            .eq("chain", "solana").eq("progress_data_ok", True)
            .gte("alert_time", cutoff)
            .execute())
    rows = resp.data or []
    checked = len(rows)
    populated = sum(1 for r in rows if r.get(column) is not None)

    if checked < MIN_RECENT_ROWS_TO_JUDGE:
        return ComponentHealth(
            name=f"{column}_flow", status="UNKNOWN", n_checked=checked,
            detail=f"only {checked} recent progress_data_ok rows in the last {RECENT_WINDOW_HOURS}h",
            note="natural yield is low -- UNKNOWN, not UNHEALTHY, per policy",
        )

    rate_pct = round(populated / checked * 100, 2)
    # progress_data_ok=True rows should almost always carry a populated
    # value for both of these fields (they're captured together, see
    # memecoin/progress_capture.py) -- a collapse well below that is a
    # real signal, not natural yield noise.
    status = "HEALTHY" if rate_pct >= 50.0 else "UNHEALTHY"
    return ComponentHealth(
        name=f"{column}_flow", status=status, n_checked=checked,
        detail=f"{populated}/{checked} ({rate_pct}%) populated in the last {RECENT_WINDOW_HOURS}h", note="",
    )


def check_progress_at_signal_flow(sb) -> ComponentHealth:
    return _check_supabase_column_flow(sb, "progress_at_signal")


def check_venue_state_at_signal_flow(sb) -> ComponentHealth:
    return _check_supabase_column_flow(sb, "venue_state_at_signal")


def check_execution_proxy_flow(repo_root: Path) -> ComponentHealth:
    """Component E: is the execution-proxy collector still producing
    observations. A silently-dead collector must show up here -- and
    per the aggregation rule, UNKNOWN here prevents overall HEALTHY
    even if every price/path component looks fine."""
    import json
    log_path = repo_root / "logs" / "research_execution_proxy" / "execution_proxy_log.jsonl"
    cutoff = datetime.now(timezone.utc) - timedelta(hours=RECENT_WINDOW_HOURS)

    checked = 0
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            obs_at = row.get("observed_at")
            if not obs_at:
                continue
            try:
                dt = datetime.fromisoformat(obs_at.replace("Z", "+00:00"))
            except ValueError:
                continue
            if dt >= cutoff:
                checked += 1

    if checked < MIN_RECENT_ROWS_TO_JUDGE:
        return ComponentHealth(
            name="execution_proxy_flow", status="UNKNOWN", n_checked=checked,
            detail=f"only {checked} recent execution-proxy observations in the last {RECENT_WINDOW_HOURS}h",
            note="natural yield is low OR the collector may be silently dead -- "
                 "UNKNOWN either way, never assumed healthy from silence",
        )
    return ComponentHealth(
        name="execution_proxy_flow", status="HEALTHY", n_checked=checked,
        detail=f"{checked} recent execution-proxy observations in the last {RECENT_WINDOW_HOURS}h", note="",
    )


def _combine_component_statuses(components: list) -> str:
    statuses = [c.status for c in components]
    if any(s == "UNHEALTHY" for s in statuses):
        return "UNHEALTHY"
    if any(s == "UNKNOWN" for s in statuses):
        return "UNKNOWN"
    return "HEALTHY"


def check_forward_pipeline_healthy(sb=None, repo_root: Optional[Path] = None) -> ForwardPipelineHealthCheck:
    root = repo_root or Path(__file__).parent.parent

    components = []
    if sb is not None:
        components.append(check_progress_at_signal_flow(sb))
        components.append(check_venue_state_at_signal_flow(sb))
    else:
        components.append(ComponentHealth("progress_at_signal_flow", "UNKNOWN", 0, "no supabase client provided", ""))
        components.append(ComponentHealth("venue_state_at_signal_flow", "UNKNOWN", 0, "no supabase client provided", ""))

    components.append(check_live_pp_paths_flow(root))
    components.append(check_path_integrity_quality(root))
    components.append(check_execution_proxy_flow(root))

    overall = _combine_component_statuses(components)
    return ForwardPipelineHealthCheck(status=overall, components=components, window_hours=RECENT_WINDOW_HOURS)


def build_final_state(repo_root: Optional[Path] = None) -> V8FinalState:
    from supabase import create_client
    from research.config import SUPABASE_URL, SUPABASE_KEY
    from research.v8_forward_readiness_report import build_report

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    engine = check_engine_ready()
    pipeline = check_forward_pipeline_healthy(sb, repo_root)
    forward_report = build_report(sb, repo_root)

    return V8FinalState(
        generated_at=datetime.now(timezone.utc).isoformat(),
        engine_ready=engine.ready,
        forward_data_pipeline_healthy=(None if pipeline.status == "UNKNOWN" else pipeline.status == "HEALTHY"),
        selection_data_ready=forward_report.selection_data_ready,
        engine_detail=engine, pipeline_detail=pipeline,
        selection_data_ready_candidates=forward_report.selection_data_ready_candidates,
    )


def print_final_state(state: V8FinalState) -> None:
    print(f"\n{'=' * 72}")
    print(f"  V8 FINAL STATE — {state.generated_at}")
    print(f"{'=' * 72}")
    print(f"  ENGINE_READY = {state.engine_ready}")
    for r in state.engine_detail.reasons:
        print(f"    - {r}")

    print(f"\n  FORWARD_DATA_PIPELINE_HEALTHY = {state.forward_data_pipeline_healthy}  "
          f"(overall: {state.pipeline_detail.status})")
    for c in state.pipeline_detail.components:
        print(f"    [{c.status:<9}] {c.name:<26} n={c.n_checked:<6} {c.detail}")
        if c.note:
            print(f"               {c.note}")

    print(f"\n  SELECTION_DATA_READY = {state.selection_data_ready}")
    print(f"    ready candidates: {state.selection_data_ready_candidates}")

    print(f"\n  holdout_evaluated = {state.holdout_evaluated}  (structurally always False)")
    print(f"{'=' * 72}\n")


def main():
    state = build_final_state()
    print_final_state(state)


if __name__ == "__main__":
    main()
