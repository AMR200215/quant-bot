"""
research/v8_final_state.py — V8 DATA RECOVERY batch, item 11: the ONE
final state machine. No further micro-phases after this -- exactly
three booleans, precisely defined, never conflated:

  ENGINE_READY
      = the already-frozen filter/exit/replay/statistics machinery is
        complete. Checked structurally: the frozen v1 entry/exit
        registries load and hash-match their frozen snapshots, the
        holdout lock mechanism is importable and enforces its
        invariants, and the statistical-selection module (item 10) is
        importable. This is a code-completeness check, not a data
        check.

  FORWARD_DATA_PIPELINE_HEALTHY
      = venue state, progress, representative path, corrected price and
        execution-proxy data are CURRENTLY flowing without a KNOWN
        SYSTEMATIC corruption. Computed from a real, current path-
        integrity scan restricted to recently-written rows (not the
        full historical corpus, which still contains pre-fix corrupted
        rows by design -- those stay INVALID forever, per item 5).
        UNKNOWN (not True) when there isn't yet enough recent data to
        judge -- never assumed healthy from silence.

  SELECTION_DATA_READY
      = at least one frozen candidate has enough forward, venue-
        qualified, integrity-valid, representative evidence to justify
        the one-shot holdout selection experiment. Requires BOTH the
        engineering-readiness floors (research/v8_readiness_engine.py)
        AND a real statistical-readiness check -- crossing the
        engineering floors alone is explicitly NOT sufficient (Phase
        2.1 item 4's whole point). This module checks for a non-trivial
        number of day-blocks (>=2, the block_bootstrap_ci minimum) as a
        proxy for "the statistical diagnostics in
        research/v8_statistical_selection.py would produce a real
        answer, not None" -- it does not run those diagnostics against
        holdout data, which stays locked.

Does NOT open, evaluate, or print holdout results. Never will -- no
function in this module accepts a holdout dataset as an argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Post-fix live capture (item 6, 2026-08-19): 10/10 real fresh
# CURVE_ACTIVE ticks were VALID. Pre-fix baseline (Phase 2.1): 51.5%
# invalid. This threshold is set well below any plausible "the bug
# recurred" rate while tolerant of the residual UNKNOWN rate from
# post-graduation ticks (never asserted CURVE_ACTIVE without proof --
# see memecoin/pumpfun_reserve_pricing.venue_state_from_pp_reserves).
RECENT_INVALID_RATE_UNHEALTHY_THRESHOLD_PCT = 15.0
RECENT_WINDOW_HOURS = 24
MIN_RECENT_LIVE_ROWS_TO_JUDGE = 5   # below this, report UNKNOWN rather than guess


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
    status: str   # "HEALTHY" | "UNHEALTHY" | "UNKNOWN"
    recent_live_rows_checked: int
    recent_invalid_rate_pct: Optional[float]
    window_hours: int
    note: str


@dataclass(frozen=True)
class SelectionDataReadyCheck:
    ready: bool
    any_candidate_engineering_ready: bool
    sufficient_day_blocks_for_statistics: bool
    reasons: list


@dataclass(frozen=True)
class V8FinalState:
    generated_at: str
    engine_ready: bool
    forward_data_pipeline_healthy: Optional[bool]   # None means UNKNOWN, never silently True
    selection_data_ready: bool
    engine_detail: EngineReadyCheck
    pipeline_detail: ForwardPipelineHealthCheck
    selection_detail: SelectionDataReadyCheck
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
        from research.v8_experiment_manifest import (
            ExperimentManifest, assert_holdout_not_evaluated, unlock_holdout_for_phase3,
        )
        m = ExperimentManifest()
        assert_holdout_not_evaluated(m)   # must not raise on a fresh manifest
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


def check_forward_pipeline_healthy(repo_root: Optional[Path] = None) -> ForwardPipelineHealthCheck:
    from research.path_schema import load_path_file
    from research.v8_path_integrity import assess_tick_integrity

    root = repo_root or Path(__file__).parent.parent
    paths_dir = root / "logs" / "research_paths"
    cutoff = datetime.now(timezone.utc) - timedelta(hours=RECENT_WINDOW_HOURS)
    cutoff_ms = int(cutoff.timestamp() * 1000)

    checked_rows = 0
    invalid_rows = 0

    if paths_dir.exists():
        # Only today's + yesterday's date dirs can possibly hold rows
        # inside the recent window -- avoids scanning the whole corpus.
        today = datetime.now(timezone.utc)
        candidate_dirs = [paths_dir / today.strftime("%Y-%m-%d"),
                          paths_dir / (today - timedelta(days=1)).strftime("%Y-%m-%d")]
        for d in candidate_dirs:
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
                    checked_rows += 1
                    result = assess_tick_integrity(r)
                    if result.status == "INVALID":
                        invalid_rows += 1

    if checked_rows < MIN_RECENT_LIVE_ROWS_TO_JUDGE:
        return ForwardPipelineHealthCheck(
            status="UNKNOWN", recent_live_rows_checked=checked_rows,
            recent_invalid_rate_pct=None, window_hours=RECENT_WINDOW_HOURS,
            note=f"fewer than {MIN_RECENT_LIVE_ROWS_TO_JUDGE} recent live_pp rows -- "
                 "not enough data to judge, not assumed healthy",
        )

    invalid_rate = round(invalid_rows / checked_rows * 100, 2)
    status = "UNHEALTHY" if invalid_rate > RECENT_INVALID_RATE_UNHEALTHY_THRESHOLD_PCT else "HEALTHY"
    return ForwardPipelineHealthCheck(
        status=status, recent_live_rows_checked=checked_rows,
        recent_invalid_rate_pct=invalid_rate, window_hours=RECENT_WINDOW_HOURS,
        note=f"threshold={RECENT_INVALID_RATE_UNHEALTHY_THRESHOLD_PCT}% "
             "(post-fix live capture measured 0%; pre-fix baseline was 51.5%)",
    )


def check_selection_data_ready(forward_report=None) -> SelectionDataReadyCheck:
    """
    forward_report: an already-built research.v8_forward_readiness_report.
        ForwardReadinessReport, or None to build one live.
    """
    reasons = []
    if forward_report is None:
        from supabase import create_client
        from research.config import SUPABASE_URL, SUPABASE_KEY
        from research.v8_forward_readiness_report import build_report
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        forward_report = build_report(sb)

    any_engineering_ready = any(r.full_eval_ready for r in forward_report.readiness_matrix)
    if not any_engineering_ready:
        reasons.append("no (candidate, exit) pair is engineering-readiness-ready yet")

    # Proxy for "the statistical diagnostics would produce a real answer":
    # at least 2 distinct calendar days of VALID representative paths,
    # the block_bootstrap_ci minimum -- NOT a run of the diagnostics
    # against holdout, which never happens here.
    valid_dates = {d for d, counts in forward_report.path_integrity.by_date.items()
                   if counts.get("VALID", 0) > 0}
    sufficient_blocks = len(valid_dates) >= 2
    if not sufficient_blocks:
        reasons.append(f"only {len(valid_dates)} day(s) with VALID representative paths -- "
                        "statistical diagnostics (block bootstrap etc.) need >= 2")

    ready = any_engineering_ready and sufficient_blocks
    return SelectionDataReadyCheck(
        ready=ready, any_candidate_engineering_ready=any_engineering_ready,
        sufficient_day_blocks_for_statistics=sufficient_blocks, reasons=reasons,
    )


def build_final_state(repo_root: Optional[Path] = None) -> V8FinalState:
    engine = check_engine_ready()
    pipeline = check_forward_pipeline_healthy(repo_root)

    from supabase import create_client
    from research.config import SUPABASE_URL, SUPABASE_KEY
    from research.v8_forward_readiness_report import build_report
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    forward_report = build_report(sb)
    selection = check_selection_data_ready(forward_report)

    return V8FinalState(
        generated_at=datetime.now(timezone.utc).isoformat(),
        engine_ready=engine.ready,
        forward_data_pipeline_healthy=(None if pipeline.status == "UNKNOWN" else pipeline.status == "HEALTHY"),
        selection_data_ready=selection.ready,
        engine_detail=engine, pipeline_detail=pipeline, selection_detail=selection,
    )


def print_final_state(state: V8FinalState) -> None:
    print(f"\n{'=' * 72}")
    print(f"  V8 FINAL STATE — {state.generated_at}")
    print(f"{'=' * 72}")
    print(f"  ENGINE_READY = {state.engine_ready}")
    if state.engine_detail.reasons:
        for r in state.engine_detail.reasons:
            print(f"    - {r}")
    print(f"\n  FORWARD_DATA_PIPELINE_HEALTHY = {state.forward_data_pipeline_healthy}")
    print(f"    status={state.pipeline_detail.status}  "
          f"recent_rows={state.pipeline_detail.recent_live_rows_checked}  "
          f"invalid_rate={state.pipeline_detail.recent_invalid_rate_pct}%")
    print(f"    {state.pipeline_detail.note}")
    print(f"\n  SELECTION_DATA_READY = {state.selection_data_ready}")
    for r in state.selection_detail.reasons:
        print(f"    - {r}")
    print(f"\n  holdout_evaluated = {state.holdout_evaluated}  (structurally always False)")
    print(f"{'=' * 72}\n")


def main():
    state = build_final_state()
    print_final_state(state)


if __name__ == "__main__":
    main()
