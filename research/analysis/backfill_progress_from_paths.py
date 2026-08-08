"""
research/analysis/backfill_progress_from_paths.py — PROGRESS-FIX PF10.

Bounded historical recovery of progress_at_signal for research_tokens rows
that predate the live capture mechanism (PF2-PF9), using each token's own
recorded path file — a frozen historical record, never present-day curve
or PumpPortal state.

Recovery source: the nearest tick with a genuine live vsol observation
(vsol > 0 — path_schema.py's own convention: vsol is 0 for backfill-sourced
ticks, which never saw real-time PumpPortal data and cannot be used here)
to the row's original alert_time. A tick with vsol=0 is not a candidate at
any distance.

Threshold selection is pre-registered: run with --report first to see
recoverable-row counts at several candidate lag thresholds. Pick ONE using
coverage/cleanliness only — never by checking which threshold makes
downstream backtest numbers look best (that would be p-hacking the
recovery itself). Only then run --apply --threshold-s <chosen>.

Scope: only rows with progress_schema_version IS NULL — i.e. never touched
by the live PF2-PF9 mechanism at all. A row that already has an honest
progress_status from a real live capture attempt (even "capture_missing"
or "pp_timeout") is left alone; overwriting a genuine live-measurement
failure with a path-tick guess would mask real capture-pipeline problems
that PF12's monitoring needs visibility into.

Run:
    python -m research.analysis.backfill_progress_from_paths --report
    python -m research.analysis.backfill_progress_from_paths --apply --threshold-s 30
"""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

from research.config import SUPABASE_URL, SUPABASE_KEY, RESEARCH_PATHS_DIR, GRAD_SOL_UI
from research.analysis.path_stats import _load_path as load_path_file

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Pre-registered BEFORE looking at any outcome/performance data.
_CANDIDATE_THRESHOLDS_S = [5, 15, 30, 60, 120]


def _discover_path_for_mint(mint: str, research_paths_dir: Path) -> Path | None:
    for p in research_paths_dir.rglob(f"{mint}.csv"):
        return p
    for p in research_paths_dir.rglob(f"{mint}.csv.gz"):
        return p
    return None


def _nearest_live_tick(rows: list[dict], alert_ts_ms: int):
    """Return (tick, lag_ms) for the closest tick with vsol > 0, or None if
    the path has no genuine live observation anywhere in it."""
    best, best_lag = None, None
    for r in rows:
        vsol = r.get("vsol") or 0
        if vsol <= 0:
            continue
        lag = abs(r["ts_ms"] - alert_ts_ms)
        if best is None or lag < best_lag:
            best, best_lag = r, lag
    if best is None:
        return None
    return best, best_lag


def _fetch_recoverable_rows(sb) -> list[dict]:
    """Rows never touched by the live PF2-PF9 mechanism, Solana only (path
    files are Solana-only), with a usable alert_time."""
    rows, offset, batch = [], 0, 1000
    while True:
        resp = (
            sb.table("research_tokens")
            .select("id,token_address,alert_time,progress_schema_version")
            .eq("chain", "solana")
            .is_("progress_schema_version", "null")
            .range(offset, offset + batch - 1)
            .execute()
        )
        chunk = resp.data or []
        rows.extend(chunk)
        if len(chunk) < batch:
            break
        offset += batch
    return [r for r in rows if r.get("alert_time")]


def _match_candidates(rows: list[dict], research_paths_dir: Path):
    """For each candidate row, find its nearest live tick (if any). Returns
    (matches, no_path_count, no_live_tick_count)."""
    matches = []
    no_path = 0
    no_live_tick = 0
    for row in rows:
        mint = row["token_address"]
        path = _discover_path_for_mint(mint, research_paths_dir)
        if path is None:
            no_path += 1
            continue
        path_rows = load_path_file(path)
        if not path_rows:
            no_path += 1
            continue
        alert_dt = datetime.fromisoformat(row["alert_time"].replace("Z", "+00:00"))
        alert_ts_ms = int(alert_dt.timestamp() * 1000)
        found = _nearest_live_tick(path_rows, alert_ts_ms)
        if found is None:
            no_live_tick += 1
            continue
        tick, lag_ms = found
        matches.append((row, tick, lag_ms))
    return matches, no_path, no_live_tick


def _print_report(rows, matches, no_path, no_live_tick):
    print(f"\n{'=' * 72}")
    print(f"  PF10 — historical progress_at_signal recovery: coverage report")
    print(f"{'=' * 72}")
    print(f"  Candidate rows (progress_schema_version IS NULL, solana):  {len(rows)}")
    print(f"  No matching path file:                                    {no_path}")
    print(f"  Path found but zero live (vsol>0) ticks in it:            {no_live_tick}")
    print(f"  Candidates with a nearest-live-tick match:                {len(matches)}")
    print(f"\n  Coverage by candidate lag threshold")
    print(f"  (pre-registered — chosen for coverage/cleanliness, NEVER by outcome performance):")
    print(f"  {'threshold':>10}  {'recoverable':>12}  {'% of all candidates':>22}")
    for th in _CANDIDATE_THRESHOLDS_S:
        n = sum(1 for _, _, lag_ms in matches if lag_ms <= th * 1000)
        pct = (n / len(rows) * 100) if rows else 0.0
        print(f"  {th:>9}s  {n:>12}  {pct:>21.1f}%")
    if matches:
        lags = sorted(lag_ms for _, _, lag_ms in matches)
        p50 = lags[len(lags) // 2]
        p90 = lags[min(int(len(lags) * 0.90), len(lags) - 1)]
        print(f"\n  Lag distribution among matched candidates: p50={p50:.0f}ms  p90={p90:.0f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="PF10: bounded historical progress_at_signal recovery from path ticks")
    parser.add_argument("--report", action="store_true",
                         help="print coverage-by-threshold table only; no writes")
    parser.add_argument("--apply", action="store_true",
                         help="write recovered rows to Supabase")
    parser.add_argument("--threshold-s", type=float, default=None,
                         help="chosen lag threshold in seconds (required with --apply; "
                              "must be a value already printed by --report)")
    args = parser.parse_args()

    if not args.report and not (args.apply and args.threshold_s):
        parser.error("pass --report to see coverage, or --apply --threshold-s <seconds> to write")

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("SUPABASE_URL/SUPABASE_KEY not configured — cannot run.")
        return

    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    rows = _fetch_recoverable_rows(sb)
    matches, no_path, no_live_tick = _match_candidates(rows, RESEARCH_PATHS_DIR)
    _print_report(rows, matches, no_path, no_live_tick)

    if not args.apply:
        return

    threshold_ms = args.threshold_s * 1000
    applied, skipped_lag = 0, 0
    for row, tick, lag_ms in matches:
        if lag_ms > threshold_ms:
            skipped_lag += 1
            continue
        vsol = tick["vsol"]
        progress = round(vsol / GRAD_SOL_UI, 4)
        observed_at = datetime.fromtimestamp(tick["ts_ms"] / 1000, tz=timezone.utc).isoformat()
        update = {
            "vsol_at_signal":          vsol,
            "progress_at_signal":      progress,
            "progress_source":         "pc_path_nearest_tick",
            "progress_observed_at":    observed_at,
            "progress_capture_lag_ms": round(lag_ms, 1),
            "progress_status":         "ok",
            "progress_data_ok":        True,
            "progress_schema_version": 1,
        }
        sb.table("research_tokens").update(update).eq("id", row["id"]).execute()
        applied += 1

    print(f"\n{'=' * 72}")
    print(f"  Applied threshold={args.threshold_s}s: recovered {applied} rows, "
          f"skipped {skipped_lag} (lag over threshold), "
          f"{len(rows) - applied - skipped_lag} left NULL (no usable path data)")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
