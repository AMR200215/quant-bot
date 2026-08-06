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
  E — Peak-mcap distribution ("where do they turn"): mcap = price_usd × 1e9 at
      the global peak tick, distributed into named stall zones, overall + per
      progress bucket. n<MIN_N → INSUFFICIENT.
  F — Conditional continuation ("how high after the turn"): first trough depth
      (≥10% pullback from a local high) and the mcap band it occurred at, vs
      the subsequent extension (peak reached after the trough, relative to the
      trough price). Grouped by mcap band at the trough.
  G — Unique-buyer velocity curve: cumulative distinct trader_pk with side=buy
      at t=5/15/30/60s, vs outcome (winner = pct_change_peak >= +50%). Requires
      trader_pk (N7a) — rows written before 2026-07-30 have trader_pk="" and are
      excluded from n, not silently zero-filled.
  H — Sniper density: distinct buyer count in the first 5s after the path's
      first tick, vs outcome. Same trader_pk requirement/exclusion as G.

Progress buckets (progress_at_signal = pp_vsol / 115):
  0–0.25, 0.25–0.50, 0.50–0.75, 0.75–0.90, 0.90+

Run:
    python -m research.analysis.path_stats
    python -m research.analysis.path_stats --min-n 50 --live-only
"""

import argparse
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, quantiles, stdev

from research.path_schema import load_path_file

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


# ── Path file loader (RF5: delegates to canonical load_path_file) ──────────────

def _load_path(p: Path) -> list[dict]:
    """
    Load a path CSV into a list of row dicts using the canonical schema loader.
    Returns only rows with data_status != 'partial' (complete rows only).
    Partial-row warnings are counted and logged but rows are not omitted entirely
    from caller — consumers still receive them; analysis functions decide further.
    Sorted by ts_ms ascending (done by load_path_file).
    """
    rows, warnings = load_path_file(p)
    if warnings:
        partial_count = sum(1 for w in warnings if "partial" in w or "missing" in w)
        log.debug("load_path_file %s: %d warnings (%d partial-row)",
                  p.name, len(warnings), partial_count)

    # Convert canonical string fields to typed values expected by analysis functions
    typed: list[dict] = []
    for row in rows:
        try:
            typed.append({
                "ts_ms":      int(row["ts_ms"]),
                "price_usd":  float(row["price_usd"]),
                "side":       row.get("side", "unknown"),
                "sol_amount": float(row.get("sol_amount") or 0),
                "vsol":       float(row.get("vsol") or 0),
                "source":     row.get("source", "unknown"),
                "backfilled": row.get("backfilled", "false"),
                "data_status": row.get("data_status", "ok"),
                "trader_pk":  row.get("trader_pk", ""),   # N7(a); "" for pre-2026-07-30 rows
            })
        except (ValueError, KeyError):
            pass
    return typed


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
                    # Skip ahead to avoid double-counting the same dump.
                    # Guard: sparse/short paths (common in the 2026-08-06
                    # backfill re-run — some tokens only got 1-3 recovered
                    # ticks) can have ts_list[-1] == ts_list[0], making the
                    # avg-tick-interval term 0 -> ZeroDivisionError.
                    avg_tick_interval_ms = (ts_list[-1] - ts_list[0]) / max(len(rows) - 1, 1)
                    if avg_tick_interval_ms > 0:
                        i += max(1, int(_DUMP_WINDOW_S * 1000 / avg_tick_interval_ms))
                    else:
                        i += 1
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


# ── Analysis E: Peak-mcap distribution ("where do they turn") ────────────────

# mcap = price_usd × 1e9 (spec formula — pump.fun 1e9 fixed supply convention).
_MCAP_ZONES = [
    (0,        10_000,   "<$10k"),
    (10_000,   25_000,   "$10-25k"),
    (25_000,   50_000,   "$25-50k"),
    (50_000,   100_000,  "$50-100k"),
    (100_000,  250_000,  "$100-250k"),
    (250_000,  float("inf"), "$250k+"),
]


def _mcap_zone(mcap: float) -> str:
    for lo, hi, label in _MCAP_ZONES:
        if lo <= mcap < hi:
            return label
    return _MCAP_ZONES[-1][2]


def _analyse_peak_mcap(path_meta: list[tuple], min_n: int):
    """
    Global-peak mcap per token, distributed into named zones. "Where do they
    turn" — which mcap band tokens tend to peak in, overall and per BC-progress
    bucket entered at.
    """
    _hline("E — Peak-mcap distribution (\"where do they turn\")")
    print(f"  mcap = price_usd × 1e9 at the token's global-peak tick.")

    overall: list[float] = []
    by_bucket: list[list[float]] = [[] for _ in _PROGRESS_BUCKETS]

    for rows, progress in path_meta:
        if not rows:
            continue
        peak_row = max(rows, key=lambda r: r["price_usd"])
        if peak_row["price_usd"] <= 0:
            continue
        mcap = peak_row["price_usd"] * 1e9
        overall.append(mcap)
        if progress is not None:
            by_bucket[_bucket_index(progress)].append(mcap)

    def _zone_table(vals: list[float], label: str):
        if len(vals) < min_n:
            _insufficient(label, len(vals), min_n)
            return
        zone_counts: dict[str, int] = defaultdict(int)
        for v in vals:
            zone_counts[_mcap_zone(v)] += 1
        qs = quantiles(vals, n=100)
        print(f"\n  {label}  (n={len(vals)})")
        print(f"  median=${qs[49]:,.0f}  p25=${qs[24]:,.0f}  p75=${qs[74]:,.0f}")
        for _, _, zone_label in _MCAP_ZONES:
            cnt = zone_counts.get(zone_label, 0)
            pct = cnt / len(vals) * 100
            print(f"    {zone_label:<12} n={cnt:>5}  ({pct:5.1f}%)")

    _zone_table(overall, "Overall")
    for i, label in enumerate(_BUCKET_LABELS):
        _zone_table(by_bucket[i], f"Entered at progress {label}")


# ── Analysis F: Conditional continuation ("how high after the turn") ─────────

_TROUGH_MIN_DEPTH_PCT = 10.0   # minimum pullback from a running local high to count as a trough


def _first_trough(rows: list[dict]) -> tuple[dict, dict, float] | None:
    """
    Walk ticks tracking the running high. First time price pulls back from that
    high by >= _TROUGH_MIN_DEPTH_PCT, that low tick is "the first trough".
    Returns (running_high_row, trough_row, depth_pct) or None if no such trough.
    """
    if not rows:
        return None
    running_high = rows[0]
    for r in rows[1:]:
        if r["price_usd"] > running_high["price_usd"]:
            running_high = r
            continue
        if running_high["price_usd"] <= 0:
            continue
        depth = (running_high["price_usd"] - r["price_usd"]) / running_high["price_usd"] * 100
        if depth >= _TROUGH_MIN_DEPTH_PCT:
            return running_high, r, depth
    return None


def _analyse_conditional_continuation(path_meta: list[tuple], min_n: int):
    """
    For each token: find the first trough (>=10% pullback from a running high).
    Record its depth% and the mcap band at that trough. Then measure subsequent
    extension = (max price after the trough − trough price) / trough price.
    Grouped by mcap band at the trough — "given it pulled back this much at this
    mcap, how much further did it go afterward."
    """
    _hline("F — Conditional continuation (\"how high after the turn\")")
    print(f"  Trough: first pullback >= {_TROUGH_MIN_DEPTH_PCT:.0f}% from a running high.")
    print(f"  Extension = (max price after trough − trough price) / trough price.")

    by_zone: dict[str, list[float]] = defaultdict(list)
    depths: list[float] = []

    for rows, _progress in path_meta:
        trough = _first_trough(rows)
        if trough is None:
            continue
        _high_row, trough_row, depth = trough
        depths.append(depth)
        trough_idx = rows.index(trough_row)
        after = rows[trough_idx + 1:]
        if not after or trough_row["price_usd"] <= 0:
            continue
        max_after = max(r["price_usd"] for r in after)
        extension = (max_after - trough_row["price_usd"]) / trough_row["price_usd"] * 100
        mcap_at_trough = trough_row["price_usd"] * 1e9
        by_zone[_mcap_zone(mcap_at_trough)].append(extension)

    n_troughs = len(depths)
    if n_troughs < min_n:
        _insufficient("Tokens with a qualifying trough", n_troughs, min_n)
        return

    print(f"\n  n={n_troughs} tokens had a qualifying trough  "
          f"(median depth={median(depths):.1f}%)")
    print(f"\n  {'mcap band @ trough':<16} {'n':>5}  {'p25 ext':>9}  {'p50 ext':>9}  {'p75 ext':>9}")
    for _, _, zone_label in _MCAP_ZONES:
        vals = by_zone.get(zone_label, [])
        if len(vals) < min_n:
            print(f"  {zone_label:<16} {len(vals):>5}  INSUFFICIENT (need ≥{min_n})")
            continue
        qs = quantiles(vals, n=100)
        print(f"  {zone_label:<16} {len(vals):>5}  "
              f"{qs[24]:>8.1f}%  {qs[49]:>8.1f}%  {qs[74]:>8.1f}%")


# ── Analyses G/H: buyer-count features (require trader_pk, N7a) ──────────────

_BUYER_VELOCITY_OFFSETS_S = [5, 15, 30, 60]
_SNIPER_WINDOW_S          = 5
_WINNER_THRESHOLD_PCT     = 50.0   # pct_change_peak >= this = "winner" for G/H


def _unique_buyers_by(rows: list[dict], cutoff_ts_ms: int) -> int:
    """Distinct trader_pk with side='buy' at ts_ms <= cutoff. Blank trader_pk excluded."""
    seen = {
        r["trader_pk"] for r in rows
        if r["ts_ms"] <= cutoff_ts_ms and r["side"] == "buy" and r["trader_pk"]
    }
    return len(seen)


def _has_trader_pk_data(rows: list[dict]) -> bool:
    return any(r["trader_pk"] for r in rows)


def _analyse_buyer_velocity(path_meta_with_outcome: list[tuple], min_n: int):
    """
    G: cumulative unique buyers at t=5/15/30/60s vs outcome (winner/not).
    Only counts paths that actually have trader_pk data (post-N7a rows) —
    pre-N7a rows have trader_pk="" everywhere and are excluded from n rather
    than reported as buyer_count=0.
    """
    _hline("G — Unique-buyer velocity curve vs outcome")
    print(f"  Cumulative distinct trader_pk (side=buy) at t=5/15/30/60s after first tick.")
    print(f"  Winner = pct_change_peak >= +{_WINNER_THRESHOLD_PCT:.0f}%.")
    print(f"  Requires trader_pk (N7a, live from 2026-07-30) — pre-N7a rows excluded from n.")

    winner_vals: list[list[float]] = [[] for _ in _BUYER_VELOCITY_OFFSETS_S]
    loser_vals:  list[list[float]] = [[] for _ in _BUYER_VELOCITY_OFFSETS_S]

    n_with_trader_pk = 0
    for rows, pct_peak in path_meta_with_outcome:
        if not rows or not _has_trader_pk_data(rows):
            continue
        n_with_trader_pk += 1
        t0 = rows[0]["ts_ms"]
        is_winner = (pct_peak or 0) >= _WINNER_THRESHOLD_PCT
        bucket = winner_vals if is_winner else loser_vals
        for j, offset_s in enumerate(_BUYER_VELOCITY_OFFSETS_S):
            count = _unique_buyers_by(rows, t0 + offset_s * 1000)
            bucket[j].append(count)

    if n_with_trader_pk < min_n:
        _insufficient("Paths with trader_pk data", n_with_trader_pk, min_n)
        return

    print(f"\n  n={n_with_trader_pk} paths have trader_pk data "
          f"(winners={len(winner_vals[0])}, losers={len(loser_vals[0])})")
    print(f"\n  {'t offset':<10} {'winner median':>14}  {'loser median':>14}")
    for j, offset_s in enumerate(_BUYER_VELOCITY_OFFSETS_S):
        w = median(winner_vals[j]) if len(winner_vals[j]) >= min_n else None
        l = median(loser_vals[j])  if len(loser_vals[j])  >= min_n else None
        w_s = f"{w:.1f}" if w is not None else "INSUF"
        l_s = f"{l:.1f}" if l is not None else "INSUF"
        print(f"  +{offset_s:>3}s     {w_s:>14}  {l_s:>14}")


def _analyse_sniper_density(path_meta_with_outcome: list[tuple], min_n: int):
    """
    H: distinct buyers in the first _SNIPER_WINDOW_S seconds vs outcome.
    Same trader_pk-availability exclusion as G.
    """
    _hline("H — Sniper density (buyers in first 5s) vs outcome")
    print(f"  Distinct trader_pk (side=buy) within {_SNIPER_WINDOW_S}s of the first tick.")
    print(f"  Winner = pct_change_peak >= +{_WINNER_THRESHOLD_PCT:.0f}%.")
    print(f"  Requires trader_pk (N7a, live from 2026-07-30) — pre-N7a rows excluded from n.")

    winner_counts: list[float] = []
    loser_counts:  list[float] = []

    for rows, pct_peak in path_meta_with_outcome:
        if not rows or not _has_trader_pk_data(rows):
            continue
        t0 = rows[0]["ts_ms"]
        count = _unique_buyers_by(rows, t0 + _SNIPER_WINDOW_S * 1000)
        is_winner = (pct_peak or 0) >= _WINNER_THRESHOLD_PCT
        (winner_counts if is_winner else loser_counts).append(count)

    n_total = len(winner_counts) + len(loser_counts)
    if n_total < min_n:
        _insufficient("Paths with trader_pk data", n_total, min_n)
        return

    print(f"\n  n={n_total}  (winners={len(winner_counts)}, losers={len(loser_counts)})")
    if len(winner_counts) >= min_n:
        print(f"  winners: median={median(winner_counts):.1f}  "
              f"mean={mean(winner_counts):.2f}")
    else:
        _insufficient("  winners", len(winner_counts), min_n)
    if len(loser_counts) >= min_n:
        print(f"  losers:  median={median(loser_counts):.1f}  "
              f"mean={mean(loser_counts):.2f}")
    else:
        _insufficient("  losers", len(loser_counts), min_n)


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

    # Build (rows, progress_at_signal) pairs, and (rows, pct_change_peak) pairs
    path_meta: list[tuple] = []
    path_meta_outcome: list[tuple] = []
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
        path_meta_outcome.append((rows, meta.get("pct_change_peak")))
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
    _analyse_peak_mcap(path_meta, args.min_n)
    _analyse_conditional_continuation(path_meta, args.min_n)
    _analyse_buyer_velocity(path_meta_outcome, args.min_n)
    _analyse_sniper_density(path_meta_outcome, args.min_n)

    print(f"\n{'=' * 72}")
    print(f"  END PATH STATS")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
