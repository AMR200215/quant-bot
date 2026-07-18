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
        if not include_partial:
            # Exclude rows where data_partial is True
            # Use .or_ workaround: data_partial=False OR data_partial IS NULL
            q = q.or_("data_partial.eq.false,data_partial.is.null")
        chunk = (q.range(offset, offset + batch - 1).execute().data) or []
        rows.extend(chunk)
        if len(chunk) < batch:
            break
        offset += batch
    return rows


def _fetch_all_for_report(sb, include_partial: bool = False) -> list[dict]:
    """
    Fetch ALL rows (not just outcome_complete) for missed-winner analysis.
    Returns (complete_rows, all_rows).
    """
    rows: list = []
    offset, batch = 0, 1000
    while True:
        q = sb.table("research_tokens").select("*")
        if not include_partial:
            q = q.or_("data_partial.eq.false,data_partial.is.null")
        chunk = (q.range(offset, offset + batch - 1).execute().data) or []
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
    print("2. PEAK PCT BY ENTRY FEATURE BUCKETS  (excludes NULL pct rows)")
    print(sep)
    _bucket_table(rows, "buy_sell_ratio_5m", [0.4, 0.55, 0.65, 0.75, 0.85],
                  "Buy/sell ratio 5m (BSR)")
    _bucket_table(rows, "volume_5m",         [500, 2_000, 5_000, 10_000, 20_000],
                  "Volume 5m (USD)")
    _bucket_table(rows, "pp_vsol",           [5, 20, 40, 60, 79],
                  "PP vSol (bonding-curve SOL, 0→graduation at ~85)")
    _bucket_table(rows, "top10_holder_pct",  [20, 40, 60, 80],
                  "Top-10 holder concentration (%)")

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
    print("7. PROGRESS_AT_SIGNAL BUCKETS  (pp_vsol / 115, bonding-curve completion)")
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

    prog_rows = [r for r in rows if _progress(r) is not None and _peak(r) is not None]
    print(f"  Rows with pp_vsol data: {len(prog_rows)} / {len(rows)}")
    if prog_rows:
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
