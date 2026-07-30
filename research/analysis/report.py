"""
Research analytics report.

Queries research_tokens (outcome_complete=True) and prints:
  1. Win rate + median/peak pct by category (bc vs grad vs unknown)
  2. Peak pct distribution by entry-feature buckets:
       buy_sell_ratio_5m, volume_5m, pp_vsol, top10_holder_pct
  3. Screener pass/fail vs outcome (v7 filter recomputed at query time)
  4. v7_traded overlap: did v7 trade it, and how did it do vs the full set?
  5. Tick-level peak (pct_change_peak_3m) vs poll-based peak comparison
  6. [W3a] Missed-winners: screener-rejected tokens that peaked ≥+50%
  7. [W3b] progress_at_signal buckets: n, %win, time-to-peak by BC progress
  8. [W3c] Readiness verdicts: clean-n + days-to-300 for candidate V8 rules

Excludes data_partial=True rows from pct analysis by default.
All Supabase queries are paginated (no silent 1000-row truncation).

Usage:
    python -m research.analysis.report
    python -m research.analysis.report --include-partial --output results.csv
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, mean, quantiles
from typing import Optional

# research.config loads .env automatically
from research.config import (
    SUPABASE_URL, SUPABASE_KEY,
    SCREENER_MIN_LIQUIDITY_USD,
    SCREENER_MAX_MCAP_USD,
    SCREENER_MIN_BUY_SELL_RATIO_5M,
    SCREENER_MIN_VOL_5M,
    SCREENER_MAX_VOL_5M,
    SCREENER_MAX_PRICE_CHANGE_5M,
    SCREENER_MAX_RUGCHECK_SCORE,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fetch_all(sb, include_partial: bool = False) -> list[dict]:
    """Paginated fetch — avoids the silent 1000-row Supabase cap."""
    rows: list = []
    offset, batch = 0, 1000
    while True:
        q = (
            sb.table("research_tokens")
            .select("*")
            .eq("outcome_complete", True)
        )
        if not include_partial and _HAS_DATA_PARTIAL:
            q = q.or_("data_partial.eq.false,data_partial.is.null")
        chunk = (q.range(offset, offset + batch - 1).execute().data) or []
        rows.extend(chunk)
        if len(chunk) < batch:
            break
        offset += batch
    return rows


_HAS_DATA_PARTIAL: bool = True   # detected at first query; set False if column missing


def _fetch_all_for_report(sb, include_partial: bool = False) -> list[dict]:
    """
    Fetch ALL rows (not just outcome_complete) for missed-winner analysis.
    Gracefully handles missing data_partial column (column was added via migration).
    """
    global _HAS_DATA_PARTIAL
    rows: list = []
    offset, batch = 0, 1000
    while True:
        q = sb.table("research_tokens").select("*")
        if not include_partial and _HAS_DATA_PARTIAL:
            q = q.or_("data_partial.eq.false,data_partial.is.null")
        try:
            chunk = (q.range(offset, offset + batch - 1).execute().data) or []
        except Exception as e:
            if "data_partial" in str(e) and not include_partial:
                # Column doesn't exist yet — run without the filter
                _HAS_DATA_PARTIAL = False
                print("  NOTE: data_partial column missing — migration needed; "
                      "treating all rows as non-partial")
                chunk = (
                    sb.table("research_tokens")
                    .select("*")
                    .range(offset, offset + batch - 1)
                    .execute().data
                ) or []
            else:
                raise
        rows.extend(chunk)
        if len(chunk) < batch:
            break
        offset += batch
    return rows


def _screener_failed_filters(row: dict) -> list:
    """
    Return list of filter names that block this row.
    Priority order matches live screener.
    """
    failed = []
    liq  = row.get("liquidity_usd") or 0
    mcap = row.get("mcap_usd") or 0
    bsr  = row.get("buy_sell_ratio_5m") or 0
    vol5 = row.get("volume_5m") or 0
    pc5  = abs(row.get("price_change_5m") or 0)
    rug  = row.get("rugcheck_score") or 0
    if liq  < SCREENER_MIN_LIQUIDITY_USD:      failed.append("liq<8k")
    if mcap > SCREENER_MAX_MCAP_USD:           failed.append("mcap>8M")
    if bsr  < SCREENER_MIN_BUY_SELL_RATIO_5M:  failed.append("bsr<0.55")
    if vol5 < SCREENER_MIN_VOL_5M:             failed.append("vol<2k")
    if vol5 > SCREENER_MAX_VOL_5M:             failed.append("vol>50k")
    if pc5  > SCREENER_MAX_PRICE_CHANGE_5M:    failed.append("pc5>500%")
    if rug  > SCREENER_MAX_RUGCHECK_SCORE:     failed.append("rug>500")
    return failed


def _screener_passed(row: dict) -> bool:
    """Recompute v7's filter at query time using config thresholds."""
    liq  = row.get("liquidity_usd") or 0
    mcap = row.get("mcap_usd") or 0
    bsr  = row.get("buy_sell_ratio_5m") or 0
    vol5 = row.get("volume_5m") or 0
    pc5  = abs(row.get("price_change_5m") or 0)
    rug  = row.get("rugcheck_score") or 0

    if liq  < SCREENER_MIN_LIQUIDITY_USD:      return False
    if mcap > SCREENER_MAX_MCAP_USD:           return False
    if bsr  < SCREENER_MIN_BUY_SELL_RATIO_5M:  return False
    if vol5 < SCREENER_MIN_VOL_5M:             return False
    if vol5 > SCREENER_MAX_VOL_5M:             return False
    if pc5  > SCREENER_MAX_PRICE_CHANGE_5M:    return False
    if rug  > SCREENER_MAX_RUGCHECK_SCORE:     return False
    return True


def _peak(row: dict) -> Optional[float]:
    return row.get("pct_change_peak")


def _bucket(val, edges: list) -> str:
    if val is None:
        return "  NULL"
    for e in edges:
        if val < e:
            return f"<{e}"
    return f">={edges[-1]}"


# ── RC1 era segmentation ───────────────────────────────────────────────────────
# "clean"               — row was polled under RF1; price_source/status provenance written
# "dex_conditioned_preRF1" — polled before RF1; BC tokens' DexScreener NULLs bias outcomes

_ERA_CLEAN    = "clean"
_ERA_PRERF1   = "dex_conditioned_preRF1"


def _alert_dt(row: dict) -> Optional[datetime]:
    """Parse row['alert_time'] (ISO string, possibly 'Z'-suffixed) to a datetime.
    Returns None if missing/unparseable. Used by section 11 (N7c) hour/weekday
    bucketing and available standalone for tests."""
    t = row.get("alert_time")
    if not t:
        return None
    try:
        return datetime.fromisoformat(str(t).replace("Z", "+00:00"))
    except Exception:
        return None


def _era(row: dict) -> str:
    """
    Return the measurement era for a row.
    Clean: at least one price_source or price_status column is non-NULL
    (means RF1 ran and wrote provenance for this row's polls).
    PreRF1: all provenance columns are NULL/absent — collected before RF1 deployment.
    """
    for interval in ("t1m", "t3m", "t5m", "t10m", "t15m", "t20m", "t30m"):
        if row.get(f"price_source_{interval}") is not None:
            return _ERA_CLEAN
        if row.get(f"price_status_{interval}") is not None:
            return _ERA_CLEAN
    return _ERA_PRERF1


def _era_split(rows: list) -> tuple[list, list]:
    """Return (clean_rows, preRF1_rows)."""
    clean   = [r for r in rows if _era(r) == _ERA_CLEAN]
    pre_rf1 = [r for r in rows if _era(r) == _ERA_PRERF1]
    return clean, pre_rf1


def _stats_era(label: str, clean: list, pre: list, indent: str = "    ") -> None:
    """Print stats for clean era; note excluded preRF1 count."""
    if pre:
        print(f"{indent}  [RC1] era=clean n={len(clean)}  "
              f"excl. dex_conditioned_preRF1 n={len(pre)}")
    _stats(label, clean, indent=indent)


def _stats(label: str, rows: list, indent: str = "    ") -> None:
    pcts = [_peak(r) for r in rows if _peak(r) is not None]
    wins  = [p for p in pcts if p > 0]
    s50   = [p for p in pcts if p > 50]
    s200  = [p for p in pcts if p > 200]
    if not pcts:
        print(f"{indent}[{label}]  n={len(rows)}  no price data")
        return
    print(f"{indent}[{label}]  n={len(rows):4d}  priced={len(pcts):4d}  "
          f"win={len(wins)/len(pcts)*100:5.1f}%  "
          f">50%={len(s50)/len(pcts)*100:4.1f}%  "
          f">200%={len(s200)/len(pcts)*100:4.1f}%  "
          f"med={median(pcts):+7.1f}%  "
          f"max={max(pcts):+8.1f}%")


def _bucket_table_era(clean: list, pre: list, field: str, edges: list, label: str) -> None:
    """
    Print a bucket table for the clean era with a side-by-side note for preRF1.
    The clean-era table is printed in full; preRF1 is summarised as a single
    exclusion line so the table isn't doubled in length.
    """
    pre_priced = [r for r in pre if _peak(r) is not None]
    if pre:
        print(f"  [RC1] era=clean n={len(clean)}  "
              f"excl. dex_conditioned_preRF1 n={len(pre)} ({len(pre_priced)} priced)")
    _bucket_table(clean, field, edges, label)


def _bucket_table(rows: list, field: str, edges: list, label: str) -> None:
    buckets: dict = defaultdict(list)
    for r in rows:
        p = _peak(r)
        if p is None:
            continue
        buckets[_bucket(r.get(field), edges)].append(p)
    if not buckets:
        print(f"    (no data for {field})")
        return
    print(f"\n  {label}:")
    # Sort numerically where possible
    def _sort_key(k):
        if k.strip() == "NULL":
            return (1, 0)
        try:
            return (0, float(k.lstrip("<>=").split()[0]))
        except Exception:
            return (0, 0)
    for bkt in sorted(buckets, key=_sort_key):
        pcts = buckets[bkt]
        wins = [p for p in pcts if p > 0]
        med  = median(pcts) if pcts else 0
        wr   = len(wins) / len(pcts) * 100 if pcts else 0
        print(f"    {bkt:>10}  n={len(pcts):4d}  win={wr:5.1f}%  med={med:+7.1f}%")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Research pipeline analytics report")
    parser.add_argument("--include-partial", action="store_true",
                        help="Include rows where some polls had NULL prices")
    parser.add_argument("--output", metavar="FILE",
                        help="Write enriched CSV with screener_passed column")
    args = parser.parse_args()

    try:
        from supabase import create_client
    except ImportError:
        print("supabase-py not installed — run: pip install supabase")
        sys.exit(1)

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("SUPABASE_URL / SUPABASE_KEY not set in .env")
        sys.exit(1)

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    print(f"Fetching research_tokens "
          f"({'including' if args.include_partial else 'excluding'} partial)…")
    # Fetch ALL rows for missed-winner analysis, then split into complete/all
    all_rows = _fetch_all_for_report(sb, include_partial=args.include_partial)
    rows     = [r for r in all_rows if r.get("outcome_complete")]
    print(f"  {len(all_rows)} total rows  |  {len(rows)} outcome_complete\n")

    if not rows:
        print("No complete rows yet — wait for tokens to finish their poll windows.")
        return

    partial_n = sum(1 for r in rows if r.get("data_partial"))
    print(f"  {len(rows) - partial_n} full  |  {partial_n} partial (data_partial=True)\n")

    # RC1: era split applied globally — used by sections 2, 7, and section 10
    clean_rows, preRF1_rows = _era_split(rows)
    all_clean,  all_preRF1  = _era_split(all_rows)
    _era_note = (f"  [RC1] era split: {len(clean_rows)} clean "
                 f"/ {len(preRF1_rows)} dex_conditioned_preRF1 "
                 f"(of {len(rows)} outcome_complete)")
    print(_era_note + "\n")

    # ── 1. By category ────────────────────────────────────────────────────────
    sep = "=" * 70
    print(sep)
    print("1. WIN RATE & PEAK BY CATEGORY")
    print(sep)
    cats = ("social_alert_bc", "social_alert_grad", "unknown")
    for cat in cats:
        cat_rows = [r for r in rows if r.get("category") == cat]
        if cat_rows:
            _stats(cat, cat_rows)
            ic = Counter(r.get("peak_interval") for r in cat_rows
                         if r.get("peak_interval"))
            if ic:
                print(f"      peak intervals: {dict(ic.most_common(5))}")
    _stats("ALL", rows)

    # ── 2. Entry-feature bucket analysis ──────────────────────────────────────
    print(f"\n{sep}")
    print("2. PEAK PCT BY ENTRY FEATURE BUCKETS  [RC1: clean era only]")
    print(sep)
    _bucket_table_era(clean_rows, preRF1_rows, "buy_sell_ratio_5m",
                      [0.4, 0.55, 0.65, 0.75, 0.85], "Buy/sell ratio 5m (BSR)")
    _bucket_table_era(clean_rows, preRF1_rows, "volume_5m",
                      [500, 2_000, 5_000, 10_000, 20_000], "Volume 5m (USD)")
    _bucket_table_era(clean_rows, preRF1_rows, "pp_vsol",
                      [5, 20, 40, 60, 79],
                      "PP vSol (bonding-curve SOL, 0→graduation at ~85)")
    _bucket_table_era(clean_rows, preRF1_rows, "top10_holder_pct",
                      [20, 40, 60, 80], "Top-10 holder concentration (%)")

    # ── 3. Screener pass/fail ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print("3. SCREENER PASS/FAIL VS OUTCOME  (v7 filter at query time)")
    print(sep)
    passed = [r for r in rows if _screener_passed(r)]
    failed = [r for r in rows if not _screener_passed(r)]
    _stats("PASS", passed)
    _stats("FAIL", failed)

    # How many rows have enough data for the screener at all?
    has_liq = sum(1 for r in rows if r.get("liquidity_usd"))
    print(f"\n  Note: {has_liq}/{len(rows)} rows have liquidity_usd "
          f"(screener requires DexScreener snapshot)")

    # ── 4. v7_traded overlap ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print("4. V7 TRADED OVERLAP")
    print(sep)
    v7_yes = [r for r in rows if r.get("v7_traded")]
    v7_no  = [r for r in rows if not r.get("v7_traded")]
    _stats("V7 TRADED",     v7_yes)
    _stats("NOT V7 TRADED", v7_no)

    # ── 5. Tick-level peak vs poll-based peak ─────────────────────────────────
    tick_rows = [r for r in rows if r.get("pct_change_peak_3m") is not None]
    if tick_rows:
        print(f"\n{sep}")
        print("5. TICK-LEVEL PEAK (3m window) vs POLL-BASED PEAK")
        print(sep)
        tick_pcts = [r["pct_change_peak_3m"] for r in tick_rows]
        poll_pcts = [r["pct_change_peak"] for r in tick_rows
                     if r.get("pct_change_peak") is not None]
        print(f"  Tokens with tick data:  {len(tick_rows)}")
        print(f"  Tick peak  — med={median(tick_pcts):+.1f}%  max={max(tick_pcts):+.1f}%")
        if poll_pcts:
            gains = [t - p for t, p in
                     zip(tick_pcts, [r.get("pct_change_peak") or 0 for r in tick_rows])
                     if r.get("pct_change_peak") is not None]
            print(f"  Poll peak  — med={median(poll_pcts):+.1f}%  max={max(poll_pcts):+.1f}%")
            print(f"  Avg tick uplift vs poll:  {mean(gains):+.1f}pp")
        t_peaks = [r["t_peak_3m_s"] for r in tick_rows if r.get("t_peak_3m_s") is not None]
        if t_peaks:
            early = sum(1 for t in t_peaks if t < 60)
            print(f"  Peak timing: {early}/{len(t_peaks)} peaked before T+60s "
                  f"({early/len(t_peaks)*100:.0f}%)")
    else:
        print(f"\n  (No tick-level peak data yet — PeakTracker running)")

    # ── 6. [W3a] Missed-winners ───────────────────────────────────────────────
    print(f"\n{sep}")
    print("6. MISSED WINNERS (screener-rejected, pct_change_peak >= +50%)")
    print(sep)
    # Use ALL outcome-complete rows regardless of partial flag for missed-winner accuracy
    complete_rows = [r for r in all_rows if r.get("outcome_complete")]
    missed = [
        r for r in complete_rows
        if not _screener_passed(r) and (_peak(r) or 0) >= 50
    ]
    print(f"  Total missed winners (>=+50%):  {len(missed)}")
    if missed:
        # Aggregate by binding filter (first failing filter in priority order)
        by_filter: dict = defaultdict(list)
        for r in missed:
            filters = _screener_failed_filters(r)
            binding = filters[0] if filters else "no_data"
            by_filter[binding].append(_peak(r))

        print(f"\n  {'Filter':<14} {'Missed':>6}  {'Med peak':>10}  {'Max peak':>10}  {'>=+100%':>7}")
        print(f"  {'-'*14}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*7}")
        for filt, peaks in sorted(by_filter.items(), key=lambda x: -len(x[1])):
            peaks_nn = [p for p in peaks if p is not None]
            med  = median(peaks_nn) if peaks_nn else 0
            mx   = max(peaks_nn)    if peaks_nn else 0
            ge100 = sum(1 for p in peaks_nn if p >= 100)
            print(f"  {filt:<14}  {len(peaks):>6}  {med:>+10.1f}%  {mx:>+10.1f}%  {ge100:>7}")

        # Tokens blocked by only ONE filter (single-filter block — most actionable)
        single_block = [
            r for r in missed
            if len(_screener_failed_filters(r)) == 1
        ]
        print(f"\n  Single-filter blocks:  {len(single_block)}/{len(missed)}  "
              f"({len(single_block)/len(missed)*100:.0f}% removable by relaxing one rule)")

    # ── 7. [W3b] progress_at_signal buckets ──────────────────────────────────
    print(f"\n{sep}")
    print("7. PROGRESS_AT_SIGNAL BUCKETS  [RC1: clean era only]  (pp_vsol / 115)")
    print(sep)
    # Compute on-the-fly from pp_vsol if progress_at_signal column is missing
    def _progress(r):
        p = r.get("progress_at_signal")
        if p is not None:
            return p
        vsol = r.get("pp_vsol")
        return round(vsol / 115.0, 4) if vsol else None

    _PROG_EDGES = [(0.50, "<50%"), (0.70, "50-70%"), (0.85, "70-85%"), (1.01, "85%+")]

    def _prog_bucket(p):
        if p is None:
            return "  NULL"
        for edge, label in _PROG_EDGES:
            if p < edge:
                return label
        return "85%+"

    # Use clean-era rows for this gradient analysis
    prog_all   = [r for r in rows       if _progress(r) is not None and _peak(r) is not None]
    prog_rows  = [r for r in clean_rows if _progress(r) is not None and _peak(r) is not None]
    prog_pre   = [r for r in preRF1_rows if _progress(r) is not None and _peak(r) is not None]
    print(f"  Rows with pp_vsol data: {len(prog_all)} total  "
          f"({len(prog_rows)} clean / {len(prog_pre)} preRF1 excluded)")
    if prog_rows or prog_pre:
        buckets_p: dict = defaultdict(list)
        for r in prog_rows:
            bkt = _prog_bucket(_progress(r))
            buckets_p[bkt].append(r)

        # Time-to-peak: use peak_interval → minutes, or t_peak_3m_s → seconds
        def _ttp_min(r):
            pi = r.get("peak_interval")
            ttp_map = {"T1m": 1, "T3m": 3, "T5m": 5, "T10m": 10,
                       "T15m": 15, "T20m": 20, "T30m": 30}
            return ttp_map.get(pi)

        print(f"\n  {'Bucket':<10} {'n':>5}  {'med peak':>10}  {'>=+30%':>7}  "
              f"{'>=+50%':>7}  {'p25/p50/p75 TTP (min)':>22}")
        print(f"  {'-'*10}  {'-'*5}  {'-'*10}  {'-'*7}  {'-'*7}  {'-'*22}")
        bkt_order = ["<50%", "50-70%", "70-85%", "85%+", "  NULL"]
        for bkt in bkt_order:
            bkt_rows = buckets_p.get(bkt, [])
            if not bkt_rows:
                continue
            peaks    = [_peak(r) for r in bkt_rows if _peak(r) is not None]
            ttp_vals = [_ttp_min(r) for r in bkt_rows if _ttp_min(r) is not None]
            med      = median(peaks) if peaks else 0
            ge30     = sum(1 for p in peaks if p >= 30)
            ge50     = sum(1 for p in peaks if p >= 50)
            if ttp_vals and len(ttp_vals) >= 3:
                q = quantiles(ttp_vals, n=4)
                ttp_str = f"{q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}"
            elif ttp_vals:
                m = median(ttp_vals)
                ttp_str = f"-/{m:.0f}/-"
            else:
                ttp_str = "n/a"
            print(f"  {bkt:<10}  {len(peaks):>5}  {med:>+10.1f}%  {ge30:>7}  "
                  f"{ge50:>7}  {ttp_str:>22}")

    # ── 8. [W3c] Readiness verdicts ───────────────────────────────────────────
    print(f"\n{sep}")
    print("8. READINESS VERDICTS (clean-n + days-to-n≥300 for candidate V8 rules)")
    print(sep)

    # Estimate daily alert rate from date range of all_rows
    dates = sorted(
        r["alert_time"][:10]
        for r in all_rows
        if r.get("alert_time")
    )
    if len(dates) >= 2:
        first_day = datetime.fromisoformat(dates[0]).replace(tzinfo=timezone.utc)
        last_day  = datetime.now(timezone.utc)
        span_days = max((last_day - first_day).days, 1)
        daily_rate = len(all_rows) / span_days
    else:
        span_days, daily_rate = 1, 1.0

    complete_nopart = [r for r in rows if not r.get("data_partial")]
    print(f"  Collection span:  {span_days} days  ({daily_rate:.0f} alerts/day)")
    print(f"  Complete rows:    {len(rows)}  ({len(complete_nopart)} non-partial)")

    def _verdict(label: str, subset: list, target: int = 300):
        n = len(subset)
        if n >= target:
            days_str = "READY"
        else:
            remaining = target - n
            days_needed = remaining / daily_rate if daily_rate > 0 else 9999
            days_str = f"{days_needed:.0f}d to go"
        pcts  = [_peak(r) for r in subset if _peak(r) is not None]
        med   = f"{median(pcts):+.1f}%" if pcts else "n/a"
        wins  = f"{sum(1 for p in pcts if p>0)/len(pcts)*100:.0f}%" if pcts else "n/a"
        print(f"  {label:<40} n={n:>5}  med={med:>8}  wr={wins:>6}  [{days_str}]")

    print()
    # Baseline
    _verdict("ALL complete non-partial",            complete_nopart)
    _verdict("social_alert_bc only",
             [r for r in complete_nopart if r.get("category") == "social_alert_bc"])
    _verdict("snapshot_ok=True (DexScreener data)",
             [r for r in complete_nopart if r.get("snapshot_ok")])
    _verdict("pp_vsol available (BC real-time)",
             [r for r in complete_nopart if r.get("pp_vsol")])
    _verdict("progress_at_signal < 0.5 (early BC)",
             [r for r in complete_nopart
              if _progress(r) is not None and _progress(r) < 0.5])
    _verdict("progress_at_signal 0.5-0.70",
             [r for r in complete_nopart
              if _progress(r) is not None and 0.5 <= _progress(r) < 0.70])
    _verdict("progress_at_signal 0.70-0.85",
             [r for r in complete_nopart
              if _progress(r) is not None and 0.70 <= _progress(r) < 0.85])
    _verdict("screener_passed (v7 filter)",
             [r for r in complete_nopart if _screener_passed(r)])
    _verdict("smart_money_hit=True",
             [r for r in complete_nopart if r.get("smart_money_hit")])
    _verdict("top10_holder_pct available",
             [r for r in complete_nopart if r.get("top10_holder_pct") is not None])
    _verdict("creator_holds_pct available",
             [r for r in complete_nopart if r.get("creator_holds_pct") is not None])

    # ── 9. [RF4] Realert feature analysis ────────────────────────────────────
    realert_rows = [r for r in all_rows if r.get("realert_count") is not None]
    if realert_rows:
        print(f"\n{sep}")
        print("9. [RF4] REALERT FEATURE ANALYSIS")
        print(sep)

        def _realert_bucket(count):
            if count is None or count == 0:
                return "0"
            if count == 1:
                return "1"
            if count == 2:
                return "2"
            return "3+"

        # Complete rows only for outcome analysis
        complete_realert = [r for r in realert_rows if r.get("outcome_complete")]
        buckets_ra: dict = defaultdict(list)
        for r in complete_realert:
            bkt = _realert_bucket(r.get("realert_count", 0))
            buckets_ra[bkt].append(r)

        print(f"\n  Rows with realert_count field: {len(realert_rows)} total, "
              f"{len(complete_realert)} outcome_complete")

        if complete_realert:
            print(f"\n  {'Bucket':>5}  {'n':>5}  {'valid%':>7}  {'med peak':>10}  "
                  f"{'>=+30%':>6}  {'>=+50%':>6}  {'>=+100%':>7}  {'med TTP':>8}  {'med prog':>9}")
            print(f"  {'-----':>5}  {'-----':>5}  {'------':>7}  {'-'*10}  "
                  f"{'------':>6}  {'------':>6}  {'-------':>7}  {'-------':>8}  {'--------':>9}")

            ttp_map = {"T1m": 1, "T3m": 3, "T5m": 5, "T10m": 10,
                       "T15m": 15, "T20m": 20, "T30m": 30}

            for bkt in ["0", "1", "2", "3+"]:
                bkt_rows = buckets_ra.get(bkt, [])
                if not bkt_rows:
                    continue
                pcts  = [_peak(r) for r in bkt_rows if _peak(r) is not None]
                valid_pct = f"{len(pcts)/len(bkt_rows)*100:.0f}%" if bkt_rows else "n/a"
                med   = f"{median(pcts):+.1f}%" if pcts else "n/a"
                ge30  = sum(1 for p in pcts if p >= 30)
                ge50  = sum(1 for p in pcts if p >= 50)
                ge100 = sum(1 for p in pcts if p >= 100)
                ttp_vals = [ttp_map[r["peak_interval"]] for r in bkt_rows
                            if r.get("peak_interval") and r["peak_interval"] in ttp_map]
                med_ttp = f"{median(ttp_vals):.0f}min" if ttp_vals else "n/a"
                prog_vals = [r["progress_at_signal"] for r in bkt_rows
                             if r.get("progress_at_signal") is not None]
                med_prog = f"{median(prog_vals):.2f}" if prog_vals else "n/a"
                print(f"  {bkt:>5}  {len(bkt_rows):>5}  {valid_pct:>7}  {med:>10}  "
                      f"{ge30:>6}  {ge50:>6}  {ge100:>7}  {med_ttp:>8}  {med_prog:>9}")

            # Overall any_realert vs no_realert
            any_realert  = [r for r in complete_realert if (r.get("realert_count") or 0) > 0]
            no_realert   = [r for r in complete_realert if (r.get("realert_count") or 0) == 0]
            print()
            _stats("any_realert  (count>=1)", any_realert)
            _stats("no_realert   (count==0)", no_realert)

            # realert_times distribution — median gap between alert and first realert
            gaps = []
            for r in any_realert:
                times = r.get("realert_times") or []
                alert_t = r.get("alert_time")
                if times and alert_t:
                    try:
                        t0 = datetime.fromisoformat(alert_t.replace("Z", "+00:00"))
                        t1 = datetime.fromisoformat(str(times[0]).replace("Z", "+00:00"))
                        gaps.append(abs((t1 - t0).total_seconds() / 60))
                    except Exception:
                        pass
            if gaps:
                print(f"\n  Median time to first realert: {median(gaps):.0f}min  "
                      f"(n={len(gaps)} tokens with realert)")
        else:
            print("  No outcome_complete rows with realert data yet.")
    else:
        print(f"\n  (No realert_count data yet — RF4 migration may be pending)")

    # ── 10. [RC1] Era data-quality summary ───────────────────────────────────
    print(f"\n{sep}")
    print("10. [RC1] ERA DATA-QUALITY SUMMARY")
    print(sep)
    print(f"  {'Era':<30}  {'rows':>6}  {'priced':>7}  {'null_t1m':>9}  "
          f"{'null_t3m':>9}  {'null_t10m':>10}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*7}  {'-'*9}  {'-'*9}  {'-'*10}")

    def _null_rate(era_rows, col):
        if not era_rows:
            return "n/a"
        n_null = sum(1 for r in era_rows if r.get(col) is None)
        return f"{n_null/len(era_rows)*100:.0f}%"

    for era_label, era_set in [(_ERA_CLEAN, clean_rows), (_ERA_PRERF1, preRF1_rows)]:
        priced = sum(1 for r in era_set if _peak(r) is not None)
        n1  = _null_rate(era_set, "price_t1m")
        n3  = _null_rate(era_set, "price_t3m")
        n10 = _null_rate(era_set, "price_t10m")
        print(f"  {era_label:<30}  {len(era_set):>6}  {priced:>7}  {n1:>9}  "
              f"{n3:>9}  {n10:>10}")

    # BC-token null drill-down
    bc_clean  = [r for r in clean_rows  if r.get("category") == "social_alert_bc"]
    bc_pre    = [r for r in preRF1_rows if r.get("category") == "social_alert_bc"]
    print()
    print(f"  BC tokens only (social_alert_bc):")
    for era_label, era_set in [(_ERA_CLEAN, bc_clean), (_ERA_PRERF1, bc_pre)]:
        priced = sum(1 for r in era_set if _peak(r) is not None)
        n1  = _null_rate(era_set, "price_t1m")
        n3  = _null_rate(era_set, "price_t3m")
        n10 = _null_rate(era_set, "price_t10m")
        print(f"    {era_label:<28}  {len(era_set):>6}  {priced:>7}  {n1:>9}  "
              f"{n3:>9}  {n10:>10}")

    if not clean_rows:
        print("\n  NOTE: No clean-era rows yet. All outcome data is dex_conditioned_preRF1.")
        print("        Re-run after RF1 has polled its first completed tokens (~30min window).")

    # ── 11. [N7c] Hour-of-day / day-of-week outcome + crowding ───────────────
    print(f"\n{sep}")
    print("11. [N7c] HOUR-OF-DAY / DAY-OF-WEEK OUTCOME + ALERT CROWDING")
    print(sep)
    print("  alerts/hour = volume proxy for signal crowding at that hour/weekday;")
    print("  a high-volume, low-win-rate cell suggests the desk is overloaded there.")

    timed_all      = [(r, dt) for r in all_rows      if (dt := _alert_dt(r)) is not None]
    timed_complete = [(r, dt) for r in complete_nopart if (dt := _alert_dt(r)) is not None]

    if not timed_all:
        print("\n  No rows with a parseable alert_time — skipping.")
    else:
        _DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        def _hour_table(keyfn, labels, title):
            print(f"\n  {title}")
            print(f"  {'bucket':<6} {'alerts':>7} {'alerts/hr':>10} {'n_priced':>9}  "
                  f"{'win%':>6}  {'>=+30%':>7}  {'>=+50%':>7}  {'med peak':>10}")
            counts_all: dict = defaultdict(int)
            for _r, dt in timed_all:
                counts_all[keyfn(dt)] += 1
            rows_c: dict = defaultdict(list)
            for r, dt in timed_complete:
                rows_c[keyfn(dt)].append(r)
            # normaliser: for hour-of-day, 1 bucket = 1 hour every day in span;
            # for day-of-week, 1 bucket = that weekday's occurrences in span.
            span_occurrences = max(span_days / 7.0, 1.0) if labels is _DOW_LABELS else max(span_days, 1.0)
            for key in labels:
                total = counts_all.get(key, 0)
                cell_rows = rows_c.get(key, [])
                peaks = [_peak(r) for r in cell_rows if _peak(r) is not None]
                rate = total / span_occurrences
                if len(peaks) < 30:
                    priced_str = f"{len(peaks)}" if peaks else "0"
                    print(f"  {key:<6} {total:>7} {rate:>9.1f}  {priced_str:>9}  "
                          f"{'INSUF':>6}  {'INSUF':>7}  {'INSUF':>7}  {'INSUF':>10}")
                    continue
                win  = sum(1 for p in peaks if p > 0) / len(peaks) * 100
                ge30 = sum(1 for p in peaks if p >= 30) / len(peaks) * 100
                ge50 = sum(1 for p in peaks if p >= 50) / len(peaks) * 100
                med  = median(peaks)
                print(f"  {key:<6} {total:>7} {rate:>9.1f}  {len(peaks):>9}  "
                      f"{win:>5.0f}%  {ge30:>6.0f}%  {ge50:>6.0f}%  {med:>+9.1f}%")

        _hour_table(lambda dt: f"{dt.hour:02d}h", [f"{h:02d}h" for h in range(24)],
                    "By hour of day (UTC)")
        _hour_table(lambda dt: _DOW_LABELS[dt.weekday()], _DOW_LABELS,
                    "By day of week (UTC)")

    # ── CSV output ────────────────────────────────────────────────────────────
    if args.output:
        out = Path(args.output)
        enriched = [{**r, "screener_passed": _screener_passed(r)} for r in rows]
        if enriched:
            fields = list(enriched[0].keys())
            with open(out, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(enriched)
            print(f"\nCSV written → {out}  ({len(enriched)} rows)")


if __name__ == "__main__":
    main()
