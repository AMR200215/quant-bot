"""
research/thr_run_ts.py — TS-BATCH: censoring-aware trail/TP grid
derivation over the combined (forward + reconstructed) corpus.

TS1 (censoring guard): a replay that exits via "path_end" means the
recorded path ran out before the candidate exit spec ever fired -- the
trade's true outcome is UNKNOWN, not whatever the last recorded price
happens to be (this is a right-censored observation, survival-analysis
terms). Scoring these optimistically (at face value) is exactly the
§8 measurement trap flagged in docs/V8_COMPLETE_SETUP_2026-09-10.md:
a wider/later-arming trail holds positions longer, more of them run off
the end of the recording window, and the cell looks better purely
because measurement stopped, not because the strategy improved.

Fix, reusing this project's own established pattern (YD2,
docs/RECEIPTS.md: "no-path mass imputed at... the frozen E0-E3
hard_stop values"): every censored trade is ALSO scored pessimistically,
imputed at the CANDIDATE SPEC'S OWN hard_stop value (not a fixed number
-- a spec with a wider hard_stop gets a correspondingly more lenient
floor, which is the fair, apples-to-apples comparison). A cell's
apparent advantage over E0 is only trusted if it survives this pessimism.

TS2 (the grid): trail arm in {30,40,50,75}%, width in {25,30,40,50}%.
Tier2/tier3 activation points stay fixed at E0's own +100%/+300% (only
the WIDTH scales with the grid, not the tier2/3 arm points); tier3's
width preserves E0's own tier1:tier3 ratio (0.15/0.25 = 0.6) applied to
the grid's width instead of E0's fixed 0.25. Two TP variants: no TP
(E4-grid family) and TP at +50%/+100% -- +50% is the established
WINNER_THRESHOLD_PCT (research/analysis/path_stats.py), +100% is this
project's own suggested second level
(docs/V8_COMPLETE_SETUP_2026-09-10.md §3e) -- 25% of position at each.
hard_stop/time_stop/profit_lock stay at E0's own values throughout.

Every cell is scored against BOTH corpus scopes (forward-collected-only,
and forward+reconstructed combined) computed from ONE replay pass per
cell per candidate (the combined-corpus per-token results are tagged by
source and the forward-only view is a filter over the same data, not a
separate replay run).

Read-only. Does not touch v8_exit_registry.py, does not change any
threshold, does not write to Supabase. Selection (TS3) only drafts
proposal text; does not apply anything to any registry or live config.

Run:
    python -m research.thr_run_ts
"""

from __future__ import annotations

import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path

from research.thr_run_t5 import (
    _forward_valid_mints, combined_mint_set_for_candidate, _load_path_rows,
    ROUND_TRIP_COST_PCT, WINNER_THRESHOLD_PCT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("thr_run_ts")

OUTPUT_FILE = Path(__file__).parent / "thr_ts_grid_results.json"

# --- E0's own fixed reference points, reused not reinvented ---
E0_HARD_STOP = -0.35
E0_TIME_STOP_MIN = 90
E0_TIME_STOP_MIN_GAIN = 0.30
E0_PROFIT_LOCK_MIN_GAIN = 0.40
E0_PROFIT_LOCK_MAX_GAIN = 1.00
E0_PROFIT_LOCK_STALL_SEC = 60
E0_TIER2_ARM = 1.00     # E0's own fixed "+100%" second-tier arm point
E0_TIER3_ARM = 3.00     # E0's own fixed "+300%" third-tier arm point
E0_TIER3_WIDTH_RATIO = 0.15 / 0.25   # E0's own tier1:tier3 width ratio (0.6)

ARM_VALUES = (0.30, 0.40, 0.50, 0.75)
WIDTH_VALUES = (0.25, 0.30, 0.40, 0.50)
TP_VARIANTS = {
    "notp": [],
    "tp": [(0.50, 0.25), (1.00, 0.25)],  # 50%=WINNER_THRESHOLD_PCT, 100%=this project's own §3e suggestion
}

# TS3 tolerances -- proposed here with explicit reasoning, not silently
# assumed, revisable on explicit instruction:
HARD_STOP_HIT_RATE_TOLERANCE_PP = 10.0
# Reasoning: the real winner-survival margin already established
# (docs/RECEIPTS.md CORRECTION) is 93.75%-100% for E0's own -35% stop.
# A 10-percentage-point degradation is a conservative slice of that
# margin -- large enough that small-sample noise (25-100 winners) won't
# spuriously fail a genuinely-fine cell, small enough that a cell
# materially eroding hard_stop's validated behavior gets caught.


def build_trail_tiers(arm: float, width: float) -> list:
    tier3_width = round(width * E0_TIER3_WIDTH_RATIO, 4)
    return [
        {"activates_at": arm, "trail_pct": width},
        {"activates_at": E0_TIER2_ARM, "trail_pct": width},
        {"activates_at": E0_TIER3_ARM, "trail_pct": tier3_width},
    ]


def build_exit_spec(arm: float, width: float, tp_levels: list) -> dict:
    return {
        "hard_stop": E0_HARD_STOP,
        "trail_tiers": build_trail_tiers(arm, width),
        "tp_levels": list(tp_levels),
        "time_stop_min": E0_TIME_STOP_MIN,
        "time_stop_min_gain": E0_TIME_STOP_MIN_GAIN,
        "profit_lock_min_gain": E0_PROFIT_LOCK_MIN_GAIN,
        "profit_lock_max_gain": E0_PROFIT_LOCK_MAX_GAIN,
        "profit_lock_stall_sec": E0_PROFIT_LOCK_STALL_SEC,
    }


def build_grid() -> list:
    """32 cells: 4 arms x 4 widths x 2 TP variants."""
    grid = []
    for arm in ARM_VALUES:
        for width in WIDTH_VALUES:
            for tp_name, tp_levels in TP_VARIANTS.items():
                label = f"arm{int(round(arm * 100))}_w{int(round(width * 100))}_{tp_name}"
                grid.append({
                    "label": label, "arm": arm, "width": width, "tp_variant": tp_name,
                    "spec": build_exit_spec(arm, width, tp_levels),
                })
    return grid


def prepare_population(candidate: dict, mints: set, by_mint: dict, ambiguous: set, root: Path) -> list:
    """Resolves entry alignment + peak gain for EVERY mint in the
    combined set (not just winners) -- source-tagged. This is TS1's
    combined-corpus extension: reconstructed-sourced tokens are included
    alongside forward, unlike thr_winner_exit_analysis.py's forward-only
    scope."""
    from research.v8_entry_alignment import resolve_entry_alignment, EntryAlignmentExclusion
    from research.v8_path_integrity import assess_path_integrity, PathIntegrityStatus

    population = []
    for mint in mints:
        rows, event, source = _load_path_rows(mint, by_mint, root)
        if not rows or source is None:
            continue
        integ = assess_path_integrity(rows)
        if integ.status != PathIntegrityStatus.VALID.value:
            continue
        align = resolve_entry_alignment(event, rows, candidate, ambiguous)
        if isinstance(align, EntryAlignmentExclusion):
            continue

        after = [r for r in rows if r["ts_ms"] >= align.entry_ts_ms]
        if len(after) < 2:
            continue
        entry_price = after[0]["price_usd"]
        if entry_price <= 0:
            continue

        peak_gain = 0.0
        for r in after[1:]:
            gain = (r["price_usd"] / entry_price - 1) * 100
            if gain > peak_gain:
                peak_gain = gain

        population.append({
            "mint": mint, "rows": rows, "entry_ts_ms": align.entry_ts_ms,
            "source": source, "peak_gain_pct": round(peak_gain, 2),
            "is_winner": peak_gain >= WINNER_THRESHOLD_PCT,
        })
    return population


def _mean(values: list):
    return round(statistics.mean(values), 3) if values else None


def _median(values: list):
    return round(statistics.median(values), 3) if values else None


def score_cell(population: list, exit_spec: dict) -> dict:
    """Runs the exit spec against every token in the population once,
    then aggregates twice (forward-only subset, all/combined) from that
    single set of per-token results. Returns per-scope stats."""
    from research.v8_replay_engine import replay_strategy_for_full_ev, FixedLagExecutionModel

    exec_model = FixedLagExecutionModel()
    hard_stop_pnl = exit_spec["hard_stop"] * 100 + ROUND_TRIP_COST_PCT
    records = []

    for tok in population:
        result = replay_strategy_for_full_ev(tok["rows"], tok["entry_ts_ms"], {}, exit_spec, exec_model)
        if result is None:
            continue
        net_pnl = round(result.pnl_pct + ROUND_TRIP_COST_PCT, 2)
        censored = result.exit_reason == "path_end"
        pessimistic_pnl = round(hard_stop_pnl, 2) if censored else net_pnl
        records.append({
            "source": tok["source"], "is_winner": tok["is_winner"], "peak_gain_pct": tok["peak_gain_pct"],
            "exit_reason": result.exit_reason, "net_pnl": net_pnl, "pessimistic_pnl": pessimistic_pnl,
            "censored": censored,
        })

    def _scope_stats(recs: list) -> dict:
        if not recs:
            return {"n": 0}
        n = len(recs)
        winners = [r for r in recs if r["is_winner"]]
        non_winners = [r for r in recs if not r["is_winner"]]
        censored_n = sum(1 for r in recs if r["censored"])
        hard_stop_n = sum(1 for r in recs if r["exit_reason"] == "hard_stop")

        winner_captures_opt = [r["net_pnl"] / r["peak_gain_pct"] for r in winners if r["peak_gain_pct"] > 0]
        winner_captures_pess = [r["pessimistic_pnl"] / r["peak_gain_pct"] for r in winners if r["peak_gain_pct"] > 0]

        return {
            "n": n,
            "censored_fraction": round(censored_n / n, 3),
            "hard_stop_hit_rate_pct": round(100 * hard_stop_n / n, 2),
            "net_mean_ev_pct": _mean([r["net_pnl"] for r in recs]),
            "net_mean_ev_pct_pessimistic": _mean([r["pessimistic_pnl"] for r in recs]),
            "winner_n": len(winners),
            "winner_mean_capture": _mean(winner_captures_opt),
            "winner_median_capture": _median(winner_captures_opt),
            "winner_mean_capture_pessimistic": _mean(winner_captures_pess),
            "nonwinner_n": len(non_winners),
            "nonwinner_mean_net_pnl_pct": _mean([r["net_pnl"] for r in non_winners]),
        }

    forward_only = [r for r in records if r["source"] == "forward"]
    return {"forward_only": _scope_stats(forward_only), "combined": _scope_stats(records)}


def qualifies(cell_stats: dict, e0_stats: dict, hard_stop_tolerance_pp: float = HARD_STOP_HIT_RATE_TOLERANCE_PP) -> dict:
    """TS3 selection: beats E0's pessimistic net EV on BOTH corpus scopes,
    hard_stop hit-rate not degraded beyond tolerance on either scope."""
    reasons = []
    for scope in ("forward_only", "combined"):
        c, e = cell_stats[scope], e0_stats[scope]
        if c["n"] == 0 or e["n"] == 0:
            reasons.append(f"{scope}: insufficient data")
            continue
        if c["net_mean_ev_pct_pessimistic"] <= e["net_mean_ev_pct_pessimistic"]:
            reasons.append(f"{scope}: pessimistic EV {c['net_mean_ev_pct_pessimistic']} "
                            f"does not beat E0 {e['net_mean_ev_pct_pessimistic']}")
        delta = c["hard_stop_hit_rate_pct"] - e["hard_stop_hit_rate_pct"]
        if delta > hard_stop_tolerance_pp:
            reasons.append(f"{scope}: hard_stop hit-rate degraded by {delta:.1f}pp "
                            f"(tolerance {hard_stop_tolerance_pp}pp)")
    return {"qualifies": len(reasons) == 0, "reasons": reasons}


def run():
    from research.config import SUPABASE_URL, SUPABASE_KEY
    from supabase import create_client
    from research.v8_candidate_registry import CANDIDATES
    from research.v8_exit_registry import EXIT_CANDIDATES

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    root = Path(".")

    t3 = json.loads((Path(__file__).parent / "thr_t3_results.json").read_text())
    reconstructed_valid = {r["token_address"] for r in t3["results"] if r.get("valid_usable_path")}
    forward_valid, by_mint, ambiguous = _forward_valid_mints(sb, root)
    e0_spec = next(e for e in EXIT_CANDIDATES if e["exit_id"] == "E0")["spec"]

    grid = build_grid()
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "grid_size": len(grid),
           "hard_stop_hit_rate_tolerance_pp": HARD_STOP_HIT_RATE_TOLERANCE_PP, "candidates": {}}

    for cid in ("V8-P0", "V8-P3"):
        cand = next(c for c in CANDIDATES if c["candidate_id"] == cid)
        mints = combined_mint_set_for_candidate(cand, forward_valid, reconstructed_valid, by_mint)
        population = prepare_population(cand, mints, by_mint, ambiguous, root)
        n_winners = sum(1 for p in population if p["is_winner"])
        log.info("%s: population n=%d (winners=%d)", cid, len(population), n_winners)

        e0_stats = score_cell(population, e0_spec)
        cells_out = []
        for i, cell in enumerate(grid, 1):
            stats = score_cell(population, cell["spec"])
            qual = qualifies(stats, e0_stats)
            cells_out.append({"label": cell["label"], "arm": cell["arm"], "width": cell["width"],
                               "tp_variant": cell["tp_variant"], "stats": stats, **qual})
            if i % 8 == 0:
                log.info("%s: %d/%d cells scored", cid, i, len(grid))

        out["candidates"][cid] = {
            "n_population": len(population), "n_winners": n_winners,
            "e0_baseline": e0_stats, "cells": cells_out,
        }

    OUTPUT_FILE.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", OUTPUT_FILE)
    return out


if __name__ == "__main__":
    run()
