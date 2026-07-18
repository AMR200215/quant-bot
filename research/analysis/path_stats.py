"""
Path Statistics — four analyses on per-token trade-path CSVs.

Loads path files from logs/research_paths/ (live + backfill/YYYY-MM-DD dirs).
Joins with Supabase for token metadata (progress_at_signal, pct_change_peak).

Analyses:
  A — Shakeout depth: max drawdown from entry before first reaching +30/+50/+100%,
      by BC-progress bucket. P25/50/75/90 per cell. n<MIN_N → INSUFFICIENT.
  B — Post-peak decay: price retention at peak+1m/+3m/+5m, by progress bucket.
      "Retention" = price_at_offset / peak_price.
  C — Pre-dump order flow: net SOL flow (buys−sells) in the 10s BEFORE any ≥40%
      price drop vs matched random 10s windows. Reports Cohen's d + directional
      verdict (negative net flow = sell pressure precedes dumps → TRUE/FALSE).
  D — Graduation velocity: d(vsol)/dt for live paths where vsol crosses 85% of
      graduation threshold (~97.75 SOL). Backfill paths excluded (vsol=0).

Progress buckets (progress_at_signal = pp_vsol / 115):
  0–0.25, 0.25–0.50, 0.50–0.75, 0.75–0.90, 0.90+

Run:
    python -m research.analysis.path_stats
    python -m research.analysis.path_stats --min-n 50 --live-only
"""

import argparse
import csv
import gzip
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, quantiles, stdev

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Constants ──────────────────────────────────────────────────────────────────

_GRAD_SOL           = 115.0          # bonding curve graduation threshold
_GRAD_85_VSOL       = _GRAD_SOL * 0.85   # ~97.75 SOL
_GRAD_70_VSOL       = _GRAD_SOL * 0.70   # ~80.5 SOL

_SHAKEOUT_TARGETS   = [30, 50, 100]  # % gain levels before which we measure drawdown
_DECAY_OFFSETS_S    = [60, 180, 300] # 1min, 3min, 5min post-peak
_DUMP_THRESHOLD     = 0.40           # 40% price drop defines a dump
_DUMP_WINDOW_S      = 10             # seconds to detect a dump
_PRE_FLOW_WINDOW_S  = 10             # seconds of order-flow before dump

_PROGRESS_BUCKETS   = [(0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 0.90), (0.90, 2.0)]
_BUCKET_LABELS      = ["0–25%", "25–50%", "50–75%", "75–90%", "90%+"]


# ── Path file loader ───────────────────────────────────────────────────────────

def _open_path_file(p: Path):
    """Return an open text stream for a .csv or .csv.gz file."""
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, "r", encoding="utf-8", newline="")


def _load_path(p: Path) -> list[dict]:
    """
    Load a path CSV into a list of row dicts.
    Columns: ts_ms, price_usd, side, sol_amount, vsol [, source]
    Sorted by ts_ms ascending.
    """
    rows = []
    try:
        with _open_path_file(p) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rows.append({
                        "ts_ms":      int(row["ts_ms"]),
                        "price_usd":  float(row["price_usd"]),
                        "side":       row.get("side", ""),
                        "sol_amount": float(row.get("sol_amount") or 0),
                        "vsol":       float(row.get("vsol") or 0),
                        "source":     row.get("source", "live"),
                    })
                except (ValueError, KeyError):
                    pass
    except Exception as e:
        log.warning("Failed to load %s: %s", p.name, e)
        return []
    rows.sort(key=lambda r: r["ts_ms"])
    return rows


def _discover_paths(research_paths_dir: Path, live_only: bool) -> dict[str, Path]:
    """
    Walk research_paths_dir and return {mint: path} for all .csv / .csv.gz files.
    mint = filename stem (strip .csv or .csv.gz).
    """
    mint_to_path: dict[str, Path] = {}
    if not research_paths_dir.exists():
        return mint_to_path
    for p in research_paths_dir.rglob("*.csv"):
        if live_only and "backfill" in str(p):
            continue
        mint = p.stem   # e.g. "AbCdEfGh..."
        mint_to_path[mint] = p
    for p in research_paths_dir.rglob("*.csv.gz"):
        if live_only and "backfill" in str(p):
            continue
        mint = p.name[:-len(".csv.gz")]
        mint_to_path[mint] = p
    return mint_to_path


# ── Supabase metadata loader ───────────────────────────────────────────────────

def _load_metadata(sb) -> dict[str, dict]:
    """
    Fetch token metadata keyed by token_address.
    Returns {token_address: {progress_at_signal, pct_change_peak, path_file, ...}}
    """
    rows, offset, batch = [], 0, 1000
    while True:
        resp = (
            sb.table("research_tokens")
            .select("token_address,progress_at_signal,pct_change_peak,path_file,symbol")
            .eq("outcome_complete", True)
            .eq("chain", "solana")
            .range(offset, offset + batch - 1)
            .execute()
        )
        chunk = resp.data or []
        rows.extend(chunk)
        if len(chunk) < batch:
            break
        offset += batch
    return {r["token_address"]: r for r in rows}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _bucket_index(progress: float) -> int:
    for i, (lo, hi) in enumerate(_PROGRESS_BUCKETS):
        if lo <= progress < hi:
            return i
    return len(_PROGRESS_BUCKETS) - 1


def _pct_ile(vals: list[float], n: int) -> list[str]:
    """Return [p25, p50, p75, p90] strings or INSUFFICIENT if n < threshold."""
    if len(vals) < n:
        return ["INSUF"] * 4
    qs = quantiles(vals, n=100)
    return [f"{qs[24]:.1f}", f"{qs[49]:.1f}", f"{qs[74]:.1f}", f"{qs[89]:.1f}"]


def _hline(label: str):
    print(f"\n{'─' * 72}")
    print(f"  {label}")
    print("─" * 72)


def _insufficient(label: str, n: int, min_n: int):
    print(f"  {label}: INSUFFICIENT (n={n}, need ≥{min_n})")


# ── Analysis A: Shakeout depth ────────────────────────────────────────────────

def _analyse_shakeout(
    path_meta: list[tuple],   # [(rows, progress_at_signal), ...]
    min_n: int,
):
    """
    For each target (+30/+50/+100%): compute the max drawdown FROM ENTRY up until
    the target is first reached. If token never reaches target → excluded.
    Group by progress bucket.
    """
    _hline("A — Shakeout depth before reaching target (max drawdown from entry)")
    print(f"  Definition: max(entry_price − low) / entry_price × 100, measured")
    print(f"  on ticks from t=0 until price first hits +30/+50/+100%.")

    for target in _SHAKEOUT_TARGETS:
        print(f"\n  Target: >{target:+d}%")
        bucket_vals: list[list] = [[] for _ in _PROGRESS_BUCKETS]

        for rows, progress in path_meta:
            if not rows:
                continue
            if progress is None:
                progress = 0.0
            entry = rows[0]["price_usd"]
            if entry <= 0:
                continue
            target_price  = entry * (1 + target / 100)
            target_hit_idx = next(
                (i for i, r in enumerate(rows) if r["price_usd"] >= target_price),
                None,
            )
            if target_hit_idx is None:
                continue   # never reached target
            # Max drawdown from entry up to (and including) target hit
            window = rows[:target_hit_idx + 1]
            min_price  = min(r["price_usd"] for r in window)
            drawdown   = (entry - min_price) / entry * 100
            bkt = _bucket_index(progress)
            bucket_vals[bkt].append(drawdown)

        # Print table
        print(f"  {'Bucket':<12} {'n':>5}  {'p25':>7}  {'p50':>7}  {'p75':>7}  {'p90':>7}")
        for i, label in enumerate(_BUCKET_LABELS):
            vals = bucket_vals[i]
            if len(vals) < min_n:
                print(f"  {label:<12} {len(vals):>5}  INSUFFICIENT (need ≥{min_n})")
            else:
                ps = _pct_ile(vals, min_n)
                print(f"  {label:<12} {len(vals):>5}  "
                      f"{ps[0]:>7}  {ps[1]:>7}  {ps[2]:>7}  {ps[3]:>7}")


# ── Analysis B: Post-peak decay ────────────────────────────────────────────────

def _analyse_decay(path_meta: list[tuple], min_n: int):
    """
    For each token: find global peak price and its timestamp.
    Then find actual price at peak+1m, +3m, +5m (interpolate nearest tick).
    Retention = price_at_offset / peak_price.
    """
    _hline("B — Post-peak price retention (time-stop curve)")
    print(f"  Retention = price_at_offset / peak_price. 1.0 = held. 0.5 = halved.")
    print(f"\n  {'Bucket':<12} {'n':>5}  {'ret@1m':>8}  {'ret@3m':>8}  {'ret@5m':>8}")

    bucket_retentions: list[list[list]] = [
        [[] for _ in _DECAY_OFFSETS_S] for _ in _PROGRESS_BUCKETS
    ]

    for rows, progress in path_meta:
        if not rows:
            continue
        if progress is None:
            progress = 0.0

        # Global peak
        peak_row = max(rows, key=lambda r: r["price_usd"])
        peak_price  = peak_row["price_usd"]
        peak_ts_ms  = peak_row["ts_ms"]
        if peak_price <= 0:
            continue

        bkt = _bucket_index(progress)
        for j, offset_s in enumerate(_DECAY_OFFSETS_S):
            target_ts = peak_ts_ms + offset_s * 1000
            # Find closest tick at or after target_ts
            after = [r for r in rows if r["ts_ms"] >= target_ts]
            if not after:
                continue
            price_at_offset = after[0]["price_usd"]
            retention = price_at_offset / peak_price
            bucket_retentions[bkt][j].append(retention)

    for i, label in enumerate(_BUCKET_LABELS):
        medians = []
        n_vals  = []
        for j in range(len(_DECAY_OFFSETS_S)):
            vals = bucket_retentions[i][j]
            n_vals.append(len(vals))
            medians.append(median(vals) if vals else None)
        n = min(n_vals) if n_vals else 0
        if n < min_n:
            print(f"  {label:<12} {n:>5}  INSUFFICIENT (need ≥{min_n})")
        else:
            row_s = "  " + f"{label:<12} {n:>5}"
            for m in medians:
                row_s += f"  {m:>8.3f}" if m is not None else f"  {'n/a':>8}"
            print(row_s)


# ── Analysis C: Pre-dump order flow ───────────────────────────────────────────

def _net_sol_flow(rows: list[dict], from_ts_ms: int, to_ts_ms: int) -> float:
    """Net SOL flow in [from_ts_ms, to_ts_ms): buys positive, sells negative."""
    total = 0.0
    for r in rows:
        if from_ts_ms <= r["ts_ms"] < to_ts_ms:
            amt = r["sol_amount"]
            if r["side"] == "buy":
                total += amt
            elif r["side"] == "sell":
                total -= amt
    return total


def _cohens_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled_std = math.sqrt((stdev(a) ** 2 + stdev(b) ** 2) / 2)
    if pooled_std == 0:
        return 0.0
    return (mean(a) - mean(b)) / pooled_std


def _analyse_predump_flow(all_paths: list[tuple], min_n: int):
    """
    For each path: detect ≥40% price drops over any DUMP_WINDOW_S window.
    Compute net SOL flow in the PRE_FLOW_WINDOW_S seconds BEFORE each drop start.
    Compare to random windows drawn from the same path.
    """
    _hline("C — Pre-dump order flow (net SOL, 10s before ≥40% drop)")
    print(f"  Dump: price drops ≥{_DUMP_THRESHOLD*100:.0f}% in {_DUMP_WINDOW_S}s.")
    print(f"  Pre-window: {_PRE_FLOW_WINDOW_S}s of order flow immediately before dump start.")
    print(f"  Random baseline: same-length windows drawn from non-dump periods.")

    dump_flows:   list[float] = []
    random_flows: list[float] = []

    for rows, _progress in all_paths:
        if len(rows) < 10:
            continue
        ts_list  = [r["ts_ms"] for r in rows]
        t_start  = ts_list[0]
        t_end    = ts_list[-1]

        # Detect dump windows
        dump_starts: list[int] = []   # ts_ms of dump start
        step_ms = 1000   # check every 1s
        i = 0
        while i < len(rows) - 1:
            start_price = rows[i]["price_usd"]
            if start_price <= 0:
                i += 1
                continue
            start_ts = rows[i]["ts_ms"]
            end_ts   = start_ts + _DUMP_WINDOW_S * 1000
            # Find prices in the window
            window_prices = [
                r["price_usd"] for r in rows
                if start_ts <= r["ts_ms"] <= end_ts
            ]
            if window_prices:
                min_in_window = min(window_prices)
                drop = (start_price - min_in_window) / start_price
                if drop >= _DUMP_THRESHOLD:
                    dump_starts.append(start_ts)
                    # Skip ahead to avoid double-counting the same dump
                    i += max(1, int(_DUMP_WINDOW_S * 1000 / (
                        (ts_list[-1] - ts_list[0]) / max(len(rows) - 1, 1)
                    )))
                    continue
            i += 1

        # Compute pre-dump flow
        for ds in dump_starts:
            pre_from = ds - _PRE_FLOW_WINDOW_S * 1000
            pre_to   = ds
            if pre_from < t_start:
                continue
            flow = _net_sol_flow(rows, pre_from, pre_to)
            dump_flows.append(flow)

        # Random baseline: sample N random windows (N = number of dumps found)
        import random as _random
        n_dump = len(dump_starts)
        if n_dump == 0 or t_end - t_start < _PRE_FLOW_WINDOW_S * 2000:
            continue
        for _ in range(n_dump):
            rnd_start = _random.randint(
                t_start,
                t_end - _PRE_FLOW_WINDOW_S * 1000,
            )
            flow = _net_sol_flow(rows, rnd_start, rnd_start + _PRE_FLOW_WINDOW_S * 1000)
            random_flows.append(flow)

    n_dumps = len(dump_flows)
    n_rand  = len(random_flows)

    if n_dumps < min_n:
        _insufficient(f"Pre-dump windows (need ≥{min_n})", n_dumps, min_n)
        return

    d  = _cohens_d(dump_flows, random_flows)
    m_dump = mean(dump_flows)
    m_rand = mean(random_flows) if random_flows else 0.0
    direction = "SELL pressure precedes dumps" if m_dump < m_rand else "No consistent sell pressure"
    verdict   = "TRUE" if m_dump < m_rand and abs(d) >= 0.2 else "FALSE"

    print(f"\n  Dump windows:   n={n_dumps}  mean net SOL={m_dump:+.4f}")
    print(f"  Random windows: n={n_rand}  mean net SOL={m_rand:+.4f}")
    print(f"  Cohen's d:      {d:.3f}")
    print(f"  Signal:         {direction}")
    print(f"  Verdict (|d|≥0.2 + correct direction): {verdict}")


# ── Analysis D: Graduation velocity ───────────────────────────────────────────

def _analyse_grad_velocity(all_paths: list[tuple], min_n: int):
    """
    Live paths only (vsol > 0). For tokens crossing 85% BC progress.
    Compute d(vsol)/dt = (vsol_at_85% − vsol_at_70%) / elapsed_seconds.
    """
    _hline("D — Graduation velocity d(vsol)/dt for tokens crossing 85% BC")
    print(f"  Measures rate of bonding-curve fill (SOL/sec) from 70%→85% progress.")
    print(f"  Backfill paths excluded (vsol=0 in history).")
    print(f"  Graduation threshold: {_GRAD_SOL} SOL  |  70%={_GRAD_70_VSOL:.1f}  85%={_GRAD_85_VSOL:.1f}")

    rates: list[float] = []

    for rows, _progress in all_paths:
        # Skip backfill paths (all vsol=0)
        if not rows or all(r["vsol"] == 0 for r in rows):
            continue

        # Find first tick where vsol crosses 70% and 85%
        t_70 = next((r for r in rows if r["vsol"] >= _GRAD_70_VSOL), None)
        t_85 = next((r for r in rows if r["vsol"] >= _GRAD_85_VSOL), None)

        if t_70 is None or t_85 is None:
            continue
        if t_85["ts_ms"] == t_70["ts_ms"]:
            continue   # same tick → degenerate

        elapsed_s = (t_85["ts_ms"] - t_70["ts_ms"]) / 1000
        if elapsed_s <= 0:
            continue
        rate = (t_85["vsol"] - t_70["vsol"]) / elapsed_s
        rates.append(rate)

    if len(rates) < min_n:
        _insufficient("Graduation velocity", len(rates), min_n)
        return

    qs = quantiles(rates, n=100)
    print(f"\n  n={len(rates)}")
    print(f"  {'Metric':<20}  {'SOL/sec':>10}")
    print(f"  {'p10':<20}  {qs[9]:>10.3f}")
    print(f"  {'p25':<20}  {qs[24]:>10.3f}")
    print(f"  {'median':<20}  {qs[49]:>10.3f}")
    print(f"  {'p75':<20}  {qs[74]:>10.3f}")
    print(f"  {'p90':<20}  {qs[89]:>10.3f}")
    print(f"  {'mean':<20}  {mean(rates):>10.3f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trade-path statistical analysis")
    parser.add_argument("--min-n",      type=int, default=100,
                        help="minimum sample size per cell (default: 100)")
    parser.add_argument("--live-only",  action="store_true",
                        help="exclude backfill paths")
    parser.add_argument("--no-db",      action="store_true",
                        help="don't query Supabase; all tokens treated as progress=None")
    args = parser.parse_args()

    from research.config import SUPABASE_URL, SUPABASE_KEY, RESEARCH_PATHS_DIR

    # Discover path files
    mint_to_path = _discover_paths(RESEARCH_PATHS_DIR, live_only=args.live_only)
    log.info("Found %d path files", len(mint_to_path))
    if not mint_to_path:
        print("No path files found. Run PeakTracker for live data or backfill_paths.py.")
        sys.exit(0)

    # Load metadata from Supabase
    meta_by_mint: dict = {}
    if not args.no_db and SUPABASE_URL and SUPABASE_KEY:
        try:
            from supabase import create_client
            sb = create_client(SUPABASE_URL, SUPABASE_KEY)
            meta_by_mint = _load_metadata(sb)
            log.info("Loaded metadata for %d tokens from Supabase", len(meta_by_mint))
        except Exception as e:
            log.warning("Supabase metadata load failed: %s — progress_at_signal will be None", e)

    # Build (rows, progress_at_signal) pairs
    path_meta: list[tuple] = []
    loaded = 0
    skipped = 0
    for mint, path in mint_to_path.items():
        rows = _load_path(path)
        if not rows:
            skipped += 1
            continue
        meta = meta_by_mint.get(mint, {})
        progress = meta.get("progress_at_signal")   # None if missing
        path_meta.append((rows, progress))
        loaded += 1

    log.info("Loaded %d paths (%d skipped/empty)", loaded, skipped)

    print(f"\n{'=' * 72}")
    print(f"  TRADE-PATH STATISTICS  —  {loaded} tokens  —  min_n={args.min_n}")
    print(f"{'=' * 72}")

    # Filter to paths with progress_at_signal set for bucket analyses
    with_progress = [(r, p) for r, p in path_meta if p is not None]
    log.info("%d/%d paths have progress_at_signal metadata", len(with_progress), len(path_meta))

    _analyse_shakeout(with_progress, args.min_n)
    _analyse_decay(with_progress, args.min_n)
    _analyse_predump_flow(path_meta, args.min_n)
    _analyse_grad_velocity(path_meta, args.min_n)

    print(f"\n{'=' * 72}")
    print(f"  END PATH STATS")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
