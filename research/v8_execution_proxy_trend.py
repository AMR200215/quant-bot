"""
research/v8_execution_proxy_trend.py — YD-BATCH item YD3.

Recomputes execution-proxy coverage on the funded-era denominator only
(same population v8_collection_yield.py's execution_proxy_collection_
eligible_n uses -- admitted, non-ambiguous, funded-era CURVE_ACTIVE
mints; V8-P0's scope, since it has no progress_at_signal restriction
and is therefore the broadest/most general population, shared as an
unfiltered superset across all four frozen candidates) and reports the
day-by-day CUMULATIVE trajectory, not just a single snapshot number --
answering "is this rising toward the 50% floor naturally, or plateaued
below it" from real data, without touching the floor itself.

Read-only. Does not modify research/v8_collection_yield.py, does not
change EXECUTION_PROXY_MIN_COVERAGE_PCT or any other threshold.

Run:
    python -m research.v8_execution_proxy_trend
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from research.v8_collection_yield import trustworthy_collection_era_start, load_admission_log_by_mint
from research.v8_entry_alignment import find_ambiguous_mints

TREND_WINDOW_DAYS = 7


def _read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _fetch_clean_events(sb) -> list:
    rows, offset, batch = [], 0, 1000
    while True:
        resp = (sb.table("research_tokens")
                .select("token_address,alert_time,venue_state_at_signal")
                .eq("chain", "solana").eq("progress_data_ok", True)
                .range(offset, offset + batch - 1).execute())
        chunk = resp.data or []
        rows.extend(chunk)
        if len(chunk) < batch:
            break
        offset += batch
    return rows


@dataclass(frozen=True)
class DayPoint:
    day: str                    # UTC date, YYYY-MM-DD, end-of-day cutoff
    cumulative_admitted_n: int
    cumulative_observed_n: int
    coverage_pct: float


def compute_trend(sb, repo_root: Path, window_days: int = TREND_WINDOW_DAYS) -> list[DayPoint]:
    all_events = _fetch_clean_events(sb)
    ambiguous_mints = find_ambiguous_mints(all_events)

    era_start = trustworthy_collection_era_start(repo_root)
    if era_start is None:
        return []
    era_epoch = era_start.timestamp()

    venue_qualified = [r for r in all_events if r.get("venue_state_at_signal") == "CURVE_ACTIVE"]
    by_mint = {}
    for r in venue_qualified:
        mint = r.get("token_address")
        if mint in ambiguous_mints or mint in by_mint:
            continue
        at = r.get("alert_time")
        if not at:
            continue
        try:
            dt = datetime.fromisoformat(at.replace("Z", "+00:00"))
        except ValueError:
            continue
        if dt.timestamp() < era_epoch:
            continue
        by_mint[mint] = r

    eligible_mints = set(by_mint.keys())

    admission_by_mint = load_admission_log_by_mint(repo_root)
    # earliest admitting-decision timestamp per eligible, admitted mint
    admit_ts_by_mint: dict = {}
    for mint in eligible_mints:
        rows = admission_by_mint.get(mint, [])
        admit_row = next((r for r in rows if r.get("path_admitted")), None)
        if admit_row is not None:
            admit_ts_by_mint[mint] = admit_row.get("ts")

    proxy_rows = _read_jsonl(repo_root / "logs" / "research_execution_proxy" / "execution_proxy_log.jsonl")
    # earliest OK observation timestamp per mint (mint-level, consistent with
    # v8_collection_yield.py's own execution_proxy_observed_n units)
    observed_ts_by_mint: dict = {}
    for r in proxy_rows:
        if r.get("status") != "OK":
            continue
        mint = r.get("token_address")
        if mint not in admit_ts_by_mint:
            continue
        obs_at = r.get("observed_at")
        if not obs_at:
            continue
        try:
            ts = datetime.fromisoformat(obs_at.replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
        if mint not in observed_ts_by_mint or ts < observed_ts_by_mint[mint]:
            observed_ts_by_mint[mint] = ts

    today = datetime.now(timezone.utc).date()
    points = []
    for i in range(window_days - 1, -1, -1):
        day = today - timedelta(days=i)
        cutoff = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=timezone.utc).timestamp()
        admitted_n = sum(1 for ts in admit_ts_by_mint.values() if ts is not None and ts <= cutoff)
        observed_n = sum(1 for ts in observed_ts_by_mint.values() if ts <= cutoff)
        coverage_pct = round(100 * observed_n / admitted_n, 2) if admitted_n else 0.0
        points.append(DayPoint(day=day.isoformat(), cumulative_admitted_n=admitted_n,
                                cumulative_observed_n=observed_n, coverage_pct=coverage_pct))
    return points


def classify_trend(points: list[DayPoint]) -> str:
    """Descriptive only -- RISING / PLATEAUED / FALLING / INSUFFICIENT_HISTORY,
    never a projected date or a threshold change."""
    usable = [p for p in points if p.cumulative_admitted_n > 0]
    if len(usable) < 3:
        return "INSUFFICIENT_HISTORY"
    first, last = usable[0].coverage_pct, usable[-1].coverage_pct
    delta = last - first
    if delta >= 5.0:
        return "RISING"
    if delta <= -5.0:
        return "FALLING"
    return "PLATEAUED"


def print_report(points: list[DayPoint]) -> None:
    print(f"\n{'=' * 72}")
    print(f"  EXECUTION-PROXY COVERAGE TREND (YD3) — {datetime.now(timezone.utc).isoformat()}")
    print(f"{'=' * 72}")
    if not points:
        print("  era undetermined or no data -- cannot compute trend")
        return
    print(f"  {'day':<12} {'admitted_n':>11} {'observed_n':>11} {'coverage_pct':>13}")
    for p in points:
        print(f"  {p.day:<12} {p.cumulative_admitted_n:>11} {p.cumulative_observed_n:>11} {p.coverage_pct:>12.2f}%")
    trend = classify_trend(points)
    print(f"\n  trend: {trend}  (floor=50.0%, this script does not change it)")
    print(f"{'=' * 72}\n")


def main():
    from supabase import create_client
    from research.config import SUPABASE_URL, SUPABASE_KEY
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    root = Path(__file__).parent.parent
    points = compute_trend(sb, root)
    print_report(points)


if __name__ == "__main__":
    main()
