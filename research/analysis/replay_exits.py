"""
Exit-rule replay harness — tick-resolution simulation on path-file CSVs.

Simulates two named exit-rule specs on each path file and compares outcomes.

Spec A (default: "v7") mirrors the live social_alert settings in memecoin/config.py:
  hard_stop=-35%, trail_tiers=[{+30%/−25%}, {+100%/−25%}, {+300%/−15%}],
  profit_lock at +40–100% if peak stalled 60s, time_stop=90min if gain<30%.

Spec B (default: "alt1") is an alternative to test:
  Same hard stop. Tighter trail at +30% (−20% vs −25%).
  Shorter time_stop=45min. Profit_lock stall=90s (vs 60s).

Execution lag (--exec-lag-ms, default 500): after a trigger tick, fill is
simulated at the price lag_ms later (nearest tick). Mimics real sell latency.

TP ladder: optional, configurable per spec. Each TP partially exits the
position; final exit price is the SOL-weighted average across all exits.

Output: per-spec table (n, win-rate, median PnL, p25/75/90, exit-reason mix)
       and a side-by-side comparison summary.

Run:
    python -m research.analysis.replay_exits
    python -m research.analysis.replay_exits --exec-lag-ms 800 --live-only
    python -m research.analysis.replay_exits --spec-b-json '{"hard_stop":-0.30,"trail_tiers":[...]}'
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, quantiles

from research.path_schema import load_path_file
from research.v8_replay_engine import replay_strategy, FixedLagExecutionModel

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Default exit specs ─────────────────────────────────────────────────────────

_V7_SPEC = {
    "name":             "v7_social_alert",
    "hard_stop":        -0.35,
    "trail_tiers":      [
        {"activates_at": 0.30, "trail_pct": 0.25},
        {"activates_at": 1.00, "trail_pct": 0.25},
        {"activates_at": 3.00, "trail_pct": 0.15},
    ],
    "tp_levels":        [],          # v7 has no TP ladder for social_alert
    "time_stop_min":    90,
    "time_stop_min_gain": 0.30,      # don't fire time stop if gain > 30%
    "profit_lock_min_gain":  0.40,
    "profit_lock_max_gain":  1.00,
    "profit_lock_stall_sec": 60,
}

_ALT1_SPEC = {
    "name":             "alt1_early_tp_heavy",
    "hard_stop":        -0.35,       # same hard stop as v7
    "trail_tiers":      [
        {"activates_at": 0.20, "trail_pct": 0.20},  # earlier: arms at +20% (was +30%)
        {"activates_at": 0.60, "trail_pct": 0.20},  # second tier at +60% (was +100%)
    ],
    "tp_levels":        [],
    "time_stop_min":    45,          # shorter time stop
    "time_stop_min_gain": 0.30,
    "profit_lock_min_gain":  0.40,
    "profit_lock_max_gain":  1.00,
    "profit_lock_stall_sec": 60,
}

_ALT2_SPEC = {
    "name":             "alt2_wide_stop",
    "hard_stop":        -0.50,       # wider hard stop — tolerates deeper shakeouts
    "trail_tiers":      [            # same as v7 (default)
        {"activates_at": 0.30, "trail_pct": 0.25},
        {"activates_at": 1.00, "trail_pct": 0.25},
        {"activates_at": 3.00, "trail_pct": 0.15},
    ],
    "tp_levels":        [],
    "time_stop_min":    120,         # longer time stop (size cut compensates risk)
    "time_stop_min_gain": 0.30,
    "profit_lock_min_gain":  0.40,
    "profit_lock_max_gain":  1.00,
    "profit_lock_stall_sec": 60,
    # note: size=0.5x is a capital-allocation decision, not an exit-rule param;
    # all % PnL figures here are pre-size; apply 0.5x when computing $ EV
}


# ── Path file loader (RF5: delegates to canonical load_path_file) ──────────────

def _load_path(p: Path) -> list[dict]:
    """
    Load a path CSV using the canonical schema loader.
    Converts canonical string fields to typed values for the replay engine.
    Partial rows (data_status='partial') are excluded from replay to avoid
    corrupting simulation with missing price data.
    Sorted by ts_ms ascending (done by load_path_file).
    """
    raw_rows, warnings = load_path_file(p)
    if warnings:
        log.debug("load_path_file %s: %d warnings", p.name, len(warnings))

    rows: list[dict] = []
    partial_skipped = 0
    for row in raw_rows:
        # Exclude partial rows from replay simulation
        if row.get("data_status") == "partial":
            partial_skipped += 1
            continue
        try:
            rows.append({
                "ts_ms":      int(row["ts_ms"]),
                "price_usd":  float(row["price_usd"]),
                "side":       row.get("side", "unknown"),
                "sol_amount": float(row.get("sol_amount") or 0),
                "vsol":       float(row.get("vsol") or 0),
            })
        except (ValueError, KeyError):
            pass

    if partial_skipped:
        log.debug("Skipped %d partial rows from %s", partial_skipped, p.name)
    return rows


def _discover_paths(research_paths_dir: Path, live_only: bool) -> list[Path]:
    paths = []
    if not research_paths_dir.exists():
        return paths
    for p in research_paths_dir.rglob("*.csv"):
        if live_only and "backfill" in str(p):
            continue
        paths.append(p)
    for p in research_paths_dir.rglob("*.csv.gz"):
        if live_only and "backfill" in str(p):
            continue
        paths.append(p)
    return paths


# ── Replay engine ──────────────────────────────────────────────────────────────
#
# P2-6/FD14: the tick-resolution simulator itself now lives in
# research/v8_replay_engine.py (replay_strategy) as a reusable interface.
# This wrapper preserves the exact prior call shape (dict in, dict out,
# entry assumed at rows[0] -- P2-7 fixes that assumption at the caller
# level, not here) so this script's CLI output is unchanged.

def _replay_one(rows: list[dict], spec: dict, exec_lag_ms: int) -> dict | None:
    """
    Simulate spec on one path. Returns:
    {exit_price, exit_reason, pnl_pct, hold_time_s, partial_exits}
    Returns None if path is too short.
    """
    result = replay_strategy(
        rows=rows,
        entry_ts=rows[0]["ts_ms"] if rows else 0,
        entry_spec={},
        exit_spec=spec,
        execution_model=FixedLagExecutionModel(exec_lag_ms=exec_lag_ms),
    )
    if result is None:
        return None
    return {
        "exit_price":    result.exit_price,
        "exit_reason":   result.exit_reason,
        "pnl_pct":       result.pnl_pct,
        "hold_time_s":   result.hold_time_s,
        "partial_exits": result.partial_exits,
    }


# ── Statistics ─────────────────────────────────────────────────────────────────

def _summarise(results: list[dict], spec_name: str) -> dict:
    pnls         = [r["pnl_pct"] for r in results]
    wins         = [p for p in pnls if p > 0]
    reasons      = Counter(r["exit_reason"] for r in results)
    n            = len(pnls)
    win_rate     = len(wins) / n * 100 if n > 0 else 0
    qs           = quantiles(pnls, n=100) if len(pnls) >= 10 else [0] * 100

    return {
        "spec":     spec_name,
        "n":        n,
        "win_rate": win_rate,
        "mean_pnl": mean(pnls) if pnls else 0,
        "med_pnl":  median(pnls) if pnls else 0,
        "p25":      qs[24],
        "p75":      qs[74],
        "p90":      qs[89],
        "reasons":  dict(reasons),
    }


def _print_summary(s: dict):
    print(f"\n  Spec: {s['spec']}")
    print(f"  n={s['n']}  win_rate={s['win_rate']:.1f}%  "
          f"mean_pnl={s['mean_pnl']:+.1f}%  median={s['med_pnl']:+.1f}%  "
          f"p25={s['p25']:+.1f}%  p75={s['p75']:+.1f}%  p90={s['p90']:+.1f}%")
    top_reasons = sorted(s["reasons"].items(), key=lambda x: -x[1])
    reason_str  = "  ".join(f"{r}:{cnt}" for r, cnt in top_reasons[:6])
    print(f"  Exit reasons: {reason_str}")


def _compare(sa: dict, sb: dict):
    print(f"\n  {'Metric':<22}  {'Spec A':>12}  {'Spec B':>12}  {'Delta (B−A)':>14}")
    print(f"  {'-'*64}")
    for label, ka, kb in [
        ("win_rate (%)",   "win_rate", "win_rate"),
        ("mean_pnl (%)",   "mean_pnl", "mean_pnl"),
        ("median_pnl (%)", "med_pnl",  "med_pnl"),
        ("p25 (%)",        "p25",      "p25"),
        ("p75 (%)",        "p75",      "p75"),
        ("p90 (%)",        "p90",      "p90"),
    ]:
        va, vb = sa[ka], sb[kb]
        delta  = vb - va
        sign   = "+" if delta >= 0 else ""
        print(f"  {label:<22}  {va:>12.1f}  {vb:>12.1f}  {sign}{delta:>13.1f}")
    print(f"\n  Winner: {'Spec B' if sb['med_pnl'] > sa['med_pnl'] else 'Spec A'} "
          f"(by median PnL)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exit-rule tick-resolution replay")
    parser.add_argument("--exec-lag-ms",  type=int, default=500,
                        help="simulated execution lag in ms (default: 500)")
    parser.add_argument("--live-only",    action="store_true",
                        help="exclude backfill paths")
    parser.add_argument("--spec-b-json",  type=str, default=None,
                        help="JSON string overriding Spec B fields")
    parser.add_argument("--max-paths",    type=int, default=0,
                        help="cap number of paths (0=all, for quick checks)")
    args = parser.parse_args()

    from research.config import RESEARCH_PATHS_DIR

    spec_a = dict(_V7_SPEC)
    spec_b = dict(_ALT1_SPEC)
    spec_c = dict(_ALT2_SPEC)

    if args.spec_b_json:
        try:
            overrides = json.loads(args.spec_b_json)
            spec_b.update(overrides)
            log.info("Spec B overridden: %s", overrides)
        except json.JSONDecodeError as e:
            log.error("Invalid --spec-b-json: %s", e)
            sys.exit(1)

    path_files = _discover_paths(RESEARCH_PATHS_DIR, live_only=args.live_only)
    if not path_files:
        print("No path files found. Run PeakTracker or backfill_paths.py first.")
        sys.exit(0)

    if args.max_paths and len(path_files) > args.max_paths:
        path_files = path_files[:args.max_paths]
        log.info("Capped to %d paths", args.max_paths)

    log.info("Replaying %d paths  exec_lag=%dms", len(path_files), args.exec_lag_ms)

    results_a: list[dict] = []
    results_b: list[dict] = []
    results_c: list[dict] = []
    skipped = 0

    for i, p in enumerate(path_files, 1):
        rows = _load_path(p)
        if not rows:
            skipped += 1
            continue
        res_a = _replay_one(rows, spec_a, args.exec_lag_ms)
        res_b = _replay_one(rows, spec_b, args.exec_lag_ms)
        res_c = _replay_one(rows, spec_c, args.exec_lag_ms)
        if res_a:
            results_a.append(res_a)
        if res_b:
            results_b.append(res_b)
        if res_c:
            results_c.append(res_c)

        if i % 50 == 0:
            log.info("  %d/%d paths processed", i, len(path_files))

    log.info("Done. Spec A: %d  Spec B: %d  Spec C: %d  skipped: %d",
             len(results_a), len(results_b), len(results_c), skipped)

    if not results_a:
        print("No results — all paths too short or empty.")
        sys.exit(0)

    sa = _summarise(results_a, spec_a["name"])
    sb = _summarise(results_b, spec_b["name"])
    sc = _summarise(results_c, spec_c["name"])

    print(f"\n{'=' * 72}")
    print(f"  EXIT REPLAY — {len(results_a)} paths  exec_lag={args.exec_lag_ms}ms")
    print(f"{'=' * 72}")

    print(f"\n── Spec A (v7 current) ──")
    _print_summary(sa)
    print(f"\n── Spec B (early-TP-heavy) ──")
    _print_summary(sb)
    print(f"\n── Spec C (wide-stop/small-size) ──")
    _print_summary(sc)

    print(f"\n{'─' * 72}")
    print(f"  Side-by-side comparison (A vs B vs C)")
    _compare(sa, sb)
    _compare(sa, sc)

    print(f"\n{'=' * 72}")
    print(f"  END REPLAY")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
