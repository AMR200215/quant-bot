"""
Research analytics report.

Queries research_tokens (outcome_complete=True) and prints:
  1. Win rate + median/peak pct by category (bc vs grad vs unknown)
  2. Peak pct distribution by entry-feature buckets:
       buy_sell_ratio_5m, volume_5m, pp_vsol, top10_holder_pct
  3. Screener pass/fail vs outcome (v7 filter recomputed at query time)
  4. v7_traded overlap: did v7 trade it, and how did it do vs the full set?
  5. Tick-level peak (pct_change_peak_3m) vs poll-based peak comparison

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
from pathlib import Path
from statistics import median, mean
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
    print(f"Fetching complete research_tokens "
          f"({'including' if args.include_partial else 'excluding'} partial)…")
    rows = _fetch_all(sb, include_partial=args.include_partial)
    print(f"  {len(rows)} rows loaded\n")

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
