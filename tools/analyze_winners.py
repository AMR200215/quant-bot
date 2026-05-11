"""
Winner vs Loser analysis tool.

Compares signal_candidates.csv (all copy_trade signals) against
winners_journal.csv (profitable closes) to identify which features
at entry time best predict a winning trade.

Usage:
    python tools/analyze_winners.py
    python tools/analyze_winners.py --min-pnl 50  # winners must be +50% or more
"""

import argparse
import csv
import sys
from pathlib import Path

CANDIDATES_FILE = Path(__file__).parent.parent / "logs" / "signal_candidates.csv"
WINNERS_FILE    = Path(__file__).parent.parent / "logs" / "winners_journal.csv"

# Numeric features to compare
NUMERIC_FEATURES = [
    "safety_score", "momentum_score", "composite_score",
    "price_change_5m", "price_change_1h", "price_change_6h",
    "buy_sell_ratio_5m", "buy_sell_ratio_h1",
    "buys_5m", "sells_5m", "buys_h1", "sells_h1",
    "volume_5m", "volume_h1", "volume_h6",
    "liquidity_usd", "mcap_usd", "age_minutes",
    "rugcheck_score",
]

# Categorical features to compare
CATEGORICAL_FEATURES = [
    "strength", "whale_tiers",
    "has_twitter", "has_telegram", "has_website",
    "mint_disabled", "freeze_disabled",
]


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(val) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return float("nan")


def mean(vals: list[float]) -> float:
    clean = [v for v in vals if v == v]  # drop NaN
    return sum(clean) / len(clean) if clean else float("nan")


def median(vals: list[float]) -> float:
    clean = sorted(v for v in vals if v == v)
    if not clean:
        return float("nan")
    n = len(clean)
    mid = n // 2
    return clean[mid] if n % 2 else (clean[mid - 1] + clean[mid]) / 2


def analyze(min_pnl_pct: float = 0.0, since: str = ""):
    candidates = load_csv(CANDIDATES_FILE)
    winners    = load_csv(WINNERS_FILE)

    # Filter to signals after a cutoff date (e.g. "2026-05-11" to exclude pre-fix data)
    if since:
        candidates = [c for c in candidates if c.get("signal_time", "") >= since]
        winners    = [w for w in winners    if w.get("signal_time", "") >= since]
        print(f"Filtered to signal_time >= {since}")

    print(f"\nLoaded {len(candidates)} candidates, {len(winners)} winner records")

    # Filter winners by min PnL
    winners = [w for w in winners if to_float(w.get("pnl_pct", 0)) >= min_pnl_pct]
    print(f"Winners after min_pnl={min_pnl_pct}%: {len(winners)}")

    # Build winner signal_id set
    winner_ids = {w["signal_id"] for w in winners if w.get("signal_id")}

    # Split candidates
    wins   = [c for c in candidates if c.get("signal_id") in winner_ids]
    losses = [c for c in candidates if c.get("signal_id") not in winner_ids]

    print(f"\nIn candidates: {len(wins)} winners, {len(losses)} losers")
    print(f"Win rate: {len(wins)/(len(wins)+len(losses))*100:.1f}%" if wins or losses else "")

    if not wins or not losses:
        print("\nNot enough data yet — need signals in both winners and candidates.")
        return

    # -----------------------------------------------------------------------
    # Numeric feature comparison
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print(f"{'FEATURE':<28} {'WINNERS avg':>12} {'LOSERS avg':>12} {'DIFF %':>8}")
    print("="*70)

    diffs = []
    for feat in NUMERIC_FEATURES:
        w_vals = [to_float(r.get(feat)) for r in wins]
        l_vals = [to_float(r.get(feat)) for r in losses]
        w_avg  = mean(w_vals)
        l_avg  = mean(l_vals)
        if l_avg != 0 and l_avg == l_avg and w_avg == w_avg:
            diff_pct = (w_avg - l_avg) / abs(l_avg) * 100
        else:
            diff_pct = float("nan")
        diffs.append((feat, w_avg, l_avg, diff_pct))

    # Sort by absolute diff
    diffs.sort(key=lambda x: abs(x[3]) if x[3] == x[3] else 0, reverse=True)

    for feat, w_avg, l_avg, diff_pct in diffs:
        if w_avg != w_avg:
            continue
        marker = " <<<" if abs(diff_pct) >= 20 else ""
        print(f"{feat:<28} {w_avg:>12.3f} {l_avg:>12.3f} {diff_pct:>7.1f}%{marker}")

    # -----------------------------------------------------------------------
    # Categorical feature comparison
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print(f"{'FEATURE':<28} {'WINNERS %':>12} {'LOSERS %':>12}")
    print("="*70)

    for feat in CATEGORICAL_FEATURES:
        w_true = sum(1 for r in wins   if str(r.get(feat, "")).lower() in ("true", "1", "yes"))
        l_true = sum(1 for r in losses if str(r.get(feat, "")).lower() in ("true", "1", "yes"))
        w_pct  = w_true / len(wins)   * 100 if wins   else 0
        l_pct  = l_true / len(losses) * 100 if losses else 0
        marker = " <<<" if abs(w_pct - l_pct) >= 15 else ""
        print(f"{feat:<28} {w_pct:>11.1f}% {l_pct:>11.1f}%{marker}")

    # -----------------------------------------------------------------------
    # Strength distribution
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("STRENGTH DISTRIBUTION")
    print("="*70)
    for s in ("strong", "medium", "weak"):
        wc = sum(1 for r in wins   if r.get("strength") == s)
        lc = sum(1 for r in losses if r.get("strength") == s)
        print(f"  {s:<10}  winners: {wc:>4} ({wc/len(wins)*100:.0f}%)   "
              f"losers: {lc:>4} ({lc/len(losses)*100:.0f}%)")

    # -----------------------------------------------------------------------
    # Whale tier distribution
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("WHALE TIER (winners vs losers — which tiers actually win?)")
    print("="*70)
    for tier in ("1", "2", "3"):
        wc = sum(1 for r in wins   if tier in str(r.get("whale_tiers", "")))
        lc = sum(1 for r in losses if tier in str(r.get("whale_tiers", "")))
        print(f"  Tier {tier}  winners: {wc:>4} ({wc/len(wins)*100:.0f}%)   "
              f"losers: {lc:>4} ({lc/len(losses)*100:.0f}%)")

    # -----------------------------------------------------------------------
    # Top 5 actionable thresholds
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("TOP DIFFERENTIATING FEATURES (>>> = strong signal)")
    print("="*70)
    top = [(f, w, l, d) for f, w, l, d in diffs if d == d and abs(d) >= 15][:5]
    for feat, w_avg, l_avg, diff_pct in top:
        direction = "higher" if w_avg > l_avg else "lower"
        print(f"  {feat}: winners avg {w_avg:.3f} vs losers {l_avg:.3f} "
              f"({direction} by {abs(diff_pct):.0f}%)")

    print("\nSuggested next step: use top features as additional entry filters.")
    print("Run again with --min-pnl 100 to study only the big winners.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-pnl", type=float, default=0.0,
                        help="Minimum winner PnL%% to include (default: 0 = any profit)")
    parser.add_argument("--since", type=str, default="",
                        help="Only include signals on or after this date, e.g. 2026-05-11")
    args = parser.parse_args()
    analyze(args.min_pnl, args.since)
