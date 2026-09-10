"""
research/thr_winner_exit_analysis.py — same winner-survival scrutiny T5's
hard_stop correction applied, extended to trail_stop/time_stop/
profit_lock: for real winners (tokens that reach the +50% winner
threshold at some point after entry, on real forward-collected paths),
what did each exit spec ACTUALLY realize vs the eventual peak the token
went on to reach -- broken down by which rule fired.

Reuses T5's mint-set/path-loading/entry-alignment machinery verbatim
(research/thr_run_t5.py) -- this module only adds the eventual-peak
comparison and per-exit-reason breakdown on top.

Read-only. Does not touch any frozen registry, does not change any
threshold, does not write to Supabase.

Run:
    python -m research.thr_winner_exit_analysis
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
log = logging.getLogger("thr_winner_exit_analysis")

OUTPUT_FILE = Path(__file__).parent / "thr_winner_exit_results.json"


def find_real_winners(candidate: dict, mints: set, by_mint: dict, ambiguous: set, root: Path) -> list:
    """Forward-collected-only, real winners: tokens whose price ever
    reached +WINNER_THRESHOLD_PCT from the resolved entry price. Returns
    a list of dicts: {mint, event, rows, entry_ts_ms, entry_price,
    peak_gain_pct, peak_ts_ms}."""
    from research.v8_entry_alignment import resolve_entry_alignment, EntryAlignmentExclusion
    from research.v8_path_integrity import assess_path_integrity, PathIntegrityStatus

    winners = []
    for mint in mints:
        rows, event, source = _load_path_rows(mint, by_mint, root)
        if not rows or source != "forward":
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
        peak_ts_ms = after[0]["ts_ms"]
        for r in after[1:]:
            gain = (r["price_usd"] / entry_price - 1) * 100
            if gain > peak_gain:
                peak_gain = gain
                peak_ts_ms = r["ts_ms"]

        if peak_gain >= WINNER_THRESHOLD_PCT:
            winners.append({
                "mint": mint, "rows": rows, "entry_ts_ms": align.entry_ts_ms,
                "entry_price": entry_price, "peak_gain_pct": round(peak_gain, 2),
                "peak_ts_ms": peak_ts_ms,
            })
    return winners


def analyze_exit_for_winners(winners: list, exit_spec_dict: dict) -> dict:
    """For each real winner, replays the given exit spec and compares the
    REALIZED net pnl against the peak gain the token actually went on to
    reach -- grouped by which rule fired the exit."""
    from research.v8_replay_engine import replay_strategy_for_full_ev, FixedLagExecutionModel

    exec_model = FixedLagExecutionModel()
    by_reason: dict = {}
    per_winner = []

    for w in winners:
        result = replay_strategy_for_full_ev(w["rows"], w["entry_ts_ms"], {}, exit_spec_dict["spec"], exec_model)
        if result is None:
            continue
        net_pnl = round(result.pnl_pct + ROUND_TRIP_COST_PCT, 2)
        captured_fraction = round(net_pnl / w["peak_gain_pct"], 3) if w["peak_gain_pct"] > 0 else None

        by_reason.setdefault(result.exit_reason, []).append({
            "net_pnl": net_pnl, "peak_gain_pct": w["peak_gain_pct"],
            "captured_fraction": captured_fraction,
        })
        per_winner.append({
            "mint": w["mint"], "exit_reason": result.exit_reason, "net_pnl_pct": net_pnl,
            "peak_gain_pct": w["peak_gain_pct"], "captured_fraction": captured_fraction,
        })

    summary = {}
    for reason, entries in by_reason.items():
        net_pnls = [e["net_pnl"] for e in entries]
        peaks = [e["peak_gain_pct"] for e in entries]
        fractions = [e["captured_fraction"] for e in entries if e["captured_fraction"] is not None]
        summary[reason] = {
            "n": len(entries),
            "mean_net_pnl_pct": round(statistics.mean(net_pnls), 2),
            "mean_peak_gain_pct": round(statistics.mean(peaks), 2),
            "mean_captured_fraction": round(statistics.mean(fractions), 3) if fractions else None,
        }

    return {"n_winners_evaluated": len(per_winner), "by_exit_reason": summary, "per_winner": per_winner}


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
    exit_by_id = {e["exit_id"]: e for e in EXIT_CANDIDATES}

    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "candidates": {}}

    for cid in ("V8-P0", "V8-P3"):
        cand = next(c for c in CANDIDATES if c["candidate_id"] == cid)
        mints = combined_mint_set_for_candidate(cand, forward_valid, reconstructed_valid, by_mint)
        winners = find_real_winners(cand, mints, by_mint, ambiguous, root)
        log.info("%s: %d real winners found (forward-only)", cid, len(winners))

        cand_out = {"n_real_winners": len(winners), "exits": {}}
        for exit_id in ("E0", "E1", "E2", "E3"):
            exit_spec_dict = exit_by_id.get(exit_id)
            if exit_spec_dict is None:
                continue
            cand_out["exits"][exit_id] = analyze_exit_for_winners(winners, exit_spec_dict)
        out["candidates"][cid] = cand_out

    OUTPUT_FILE.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", OUTPUT_FILE)

    print(f"\n{'=' * 90}")
    print("  Real-winner exit analysis -- trail_stop/time_stop/profit_lock scrutiny")
    print("  captured_fraction = realized net pnl / eventual peak gain the token actually reached")
    print(f"{'=' * 90}")
    for cid, cdata in out["candidates"].items():
        print(f"\n  {cid}  n_real_winners={cdata['n_real_winners']}")
        for exit_id, data in cdata["exits"].items():
            print(f"    {exit_id}  (n_evaluated={data['n_winners_evaluated']}):")
            for reason, s in sorted(data["by_exit_reason"].items(), key=lambda x: -x[1]["n"]):
                cf = f"{s['mean_captured_fraction']:.2f}" if s["mean_captured_fraction"] is not None else "n/a"
                print(f"      {reason:<14} n={s['n']:>3}  mean_net_pnl={s['mean_net_pnl_pct']:>+7.2f}%  "
                      f"mean_peak={s['mean_peak_gain_pct']:>+7.2f}%  captured_fraction={cf}")
    print(f"\n{'=' * 90}\n")
    return out


if __name__ == "__main__":
    run()
