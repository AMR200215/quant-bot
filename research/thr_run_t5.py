"""
research/thr_run_t5.py — THR-BATCH T5: re-run the E0-E3 exit evaluation
on the combined (forward + reconstructed) corpus.

Gate: T3 cleared MIN_PATH_N=100 for V8-P0 (258) and V8-P3 (125) for the
first time in the project -- see docs/EXIT_EVIDENCE.md for the full
readiness-floor nuance (MIN_PATH_COVERAGE_PCT is NOT cleared under the
existing admission-funnel denominator; results here are labeled
accordingly, not claimed as passing the full formal gate).

For each candidate (V8-P0, V8-P3) and each frozen exit spec (E0-E3):
reconstructs the combined mint set (forward-valid UNION reconstructed-
valid, de-duplicated), resolves entry alignment via the existing P2-7
rules (research.v8_entry_alignment.resolve_entry_alignment -- no
path[0]==entry assumption), replays via the ONE sanctioned full-EV
entrypoint (research.v8_replay_engine.replay_strategy_for_full_ev, which
itself refuses anything that isn't path-integrity VALID), and applies
the synthetic execution-cost line: the real, already-measured -1.99%
round-trip cost (docs/RECEIPTS.md, live execution-proxy observations) --
not reinvented, reused verbatim as a flat haircut on realized pnl_pct.

$/day conversion uses $5/trade (CLAUDE.md's documented live trade size,
"~$3-5 per trade") and each candidate's own real historical signal rate
(candidate_venue_qualified_n / era_days) -- both cited, neither invented.

Read-only. Does not touch any frozen registry, does not change any
threshold, does not write to Supabase.

Run:
    python -m research.thr_run_t5
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("thr_run_t5")

ROUND_TRIP_COST_PCT = -1.99  # docs/RECEIPTS.md -- real, measured, reused verbatim
TRADE_SIZE_USD = 5.0         # CLAUDE.md's documented live trade size (~$3-5/trade)
WINNER_THRESHOLD_PCT = 50.0  # research/analysis/path_stats.py's _WINNER_THRESHOLD_PCT, reused

_EXIT_SPECS = ("E0", "E1", "E2", "E3")
T3_RESULTS_FILE = Path(__file__).parent / "thr_t3_results.json"
RECONSTRUCTED_DIR = Path("logs/research_paths/reconstructed")
OUTPUT_FILE = Path(__file__).parent / "thr_t5_results.json"


def _forward_valid_mints(sb, root: Path) -> tuple:
    """Population-wide forward-collected valid-usable-path mints, plus
    the alert-time observable table needed for per-candidate progress
    filtering. Same logic already verified live during T3/T4 prep."""
    from research.v8_collection_yield import load_admission_log_by_mint
    from research.v8_entry_alignment import find_ambiguous_mints
    from research.v8_path_integrity import assess_path_integrity, PathIntegrityStatus
    from research.path_schema import load_path_file

    rows, offset, batch = [], 0, 1000
    while True:
        resp = (sb.table("research_tokens")
                .select("event_id,token_address,alert_time,progress_at_signal,"
                        "venue_state_at_signal,path_file,progress_capture_lag_ms")
                .eq("chain", "solana").eq("progress_data_ok", True)
                .range(offset, offset + batch - 1).execute())
        chunk = resp.data or []
        rows.extend(chunk)
        if len(chunk) < batch:
            break
        offset += batch

    by_mint = {r["token_address"]: r for r in rows}
    admission_by_mint = load_admission_log_by_mint(root)
    ambiguous = find_ambiguous_mints(rows)

    valid_mints = set()
    for r in rows:
        mint = r.get("token_address")
        if mint in ambiguous:
            continue
        admit_rows = admission_by_mint.get(mint, [])
        if not any(a.get("path_admitted") for a in admit_rows):
            continue
        pf = r.get("path_file")
        if not pf:
            continue
        full = root / pf
        if not full.exists():
            gz = full.with_suffix(full.suffix + ".gz")
            full = gz if gz.exists() else None
        if full is None:
            continue
        raw, _w = load_path_file(full)
        if not raw:
            continue
        has_real = any(x.get("source") == "live_pp" and x.get("backfilled") != "true" for x in raw)
        if not has_real:
            continue
        integ = assess_path_integrity(raw)
        if integ.status != PathIntegrityStatus.VALID.value:
            continue
        valid_mints.add(mint)

    return valid_mints, by_mint, ambiguous


def combined_mint_set_for_candidate(candidate: dict, forward_valid: set, reconstructed_valid: set,
                                     by_mint: dict) -> set:
    """Forward-valid UNION reconstructed-valid, filtered to this
    candidate's progress_at_signal condition (candidates with no such
    condition match every mint in either set)."""
    prog_cond = next((c for c in candidate["conditions"] if c["feature"] == "progress_at_signal"), None)
    combined = forward_valid | reconstructed_valid
    if prog_cond is None:
        return combined
    thr = prog_cond["value"]
    return {m for m in combined if by_mint.get(m, {}).get("progress_at_signal") is not None
            and by_mint[m]["progress_at_signal"] < thr}


def _type_rows(rows: list) -> list:
    """load_path_file returns raw csv.DictReader string values -- same
    typing pass research/v8_collection_yield.py already uses before
    handing rows to resolve_entry_alignment/replay_strategy, which do
    numeric comparisons (ts_ms >= target) that fail on strings."""
    typed = []
    for r in rows:
        try:
            t = dict(r)
            t["ts_ms"] = int(r["ts_ms"])
            t["price_usd"] = float(r["price_usd"])
            typed.append(t)
        except (KeyError, ValueError, TypeError):
            continue
    return typed


def _load_path_rows(mint: str, by_mint: dict, root: Path):
    """Forward path (research_tokens.path_file) if present and loadable,
    else the reconstructed path (logs/research_paths/reconstructed/).
    Returns (rows, event, path_source) -- path_source in
    ("forward", "reconstructed", None) is load-bearing: forward and
    reconstructed paths differ enormously in typical length/duration
    (T3: reconstructed paths average ~8 ticks over seconds; forward paths
    routinely run to hundreds/thousands of ticks over 15+ minutes), so
    any aggregate replay stat that doesn't distinguish them conflates two
    very different data qualities."""
    from research.path_schema import load_path_file

    event = by_mint.get(mint)
    pf = (event or {}).get("path_file")
    if pf:
        full = root / pf
        if not full.exists():
            gz = full.with_suffix(full.suffix + ".gz")
            full = gz if gz.exists() else None
        if full is not None:
            rows, _w = load_path_file(full)
            typed = _type_rows(rows) if rows else []
            if typed:
                return typed, event, "forward"

    recon_path = RECONSTRUCTED_DIR / f"{mint}.csv.gz"
    if recon_path.exists():
        rows, _w = load_path_file(recon_path)
        typed = _type_rows(rows) if rows else []
        if typed:
            return typed, event, "reconstructed"
    return None, event, None


def _source_stats(pnls: list, exit_reasons: dict) -> dict:
    if not pnls:
        return {"n": 0}
    wins = sum(1 for p in pnls if p >= WINNER_THRESHOLD_PCT)
    return {
        "n": len(pnls), "win_rate_pct": round(100 * wins / len(pnls), 2),
        "mean_pnl_pct_net": round(statistics.mean(pnls), 2),
        "median_pnl_pct_net": round(statistics.median(pnls), 2),
        "exit_reasons": dict(exit_reasons),
    }


def replay_candidate_exit(candidate: dict, exit_spec_dict: dict, mints: set, by_mint: dict,
                           ambiguous: set, root: Path) -> dict:
    """Returns overall stats PLUS a by_source breakdown (forward vs
    reconstructed) -- required context, not optional detail: T3 found
    reconstructed paths average ~8 ticks over a few seconds (real trading
    dies fast), so replay against them routinely exits via 'path_end'
    almost immediately regardless of exit spec, while forward paths run
    long enough for hard_stop/trail_stop/time_stop to actually fire. An
    aggregate that doesn't separate these conflates two very different
    data qualities into one misleading number."""
    from research.v8_entry_alignment import resolve_entry_alignment, EntryAlignmentExclusion
    from research.v8_replay_engine import replay_strategy_for_full_ev, FixedLagExecutionModel

    exec_model = FixedLagExecutionModel()
    exclusion_reasons: dict = {}
    pnls_by_source: dict = {"forward": [], "reconstructed": []}
    exit_reasons_by_source: dict = {"forward": {}, "reconstructed": {}}

    for mint in mints:
        rows, event, source = _load_path_rows(mint, by_mint, root)
        if not rows or event is None:
            exclusion_reasons["NO_PATH_ROWS"] = exclusion_reasons.get("NO_PATH_ROWS", 0) + 1
            continue

        align = resolve_entry_alignment(event, rows, candidate, ambiguous)
        if isinstance(align, EntryAlignmentExclusion):
            exclusion_reasons[align.reason] = exclusion_reasons.get(align.reason, 0) + 1
            continue

        result = replay_strategy_for_full_ev(rows, align.entry_ts_ms, {}, exit_spec_dict["spec"], exec_model)
        if result is None:
            exclusion_reasons["REPLAY_REFUSED_OR_TOO_SHORT"] = exclusion_reasons.get("REPLAY_REFUSED_OR_TOO_SHORT", 0) + 1
            continue

        net_pnl_pct = result.pnl_pct + ROUND_TRIP_COST_PCT
        pnls_by_source[source].append(net_pnl_pct)
        er = exit_reasons_by_source[source]
        er[result.exit_reason] = er.get(result.exit_reason, 0) + 1

    all_pnls = pnls_by_source["forward"] + pnls_by_source["reconstructed"]
    n = len(all_pnls)
    combined_exit_reasons: dict = {}
    for src_reasons in exit_reasons_by_source.values():
        for k, v in src_reasons.items():
            combined_exit_reasons[k] = combined_exit_reasons.get(k, 0) + v

    result_out = {
        "n": n, "exclusion_reasons": exclusion_reasons,
        "by_source": {
            "forward": _source_stats(pnls_by_source["forward"], exit_reasons_by_source["forward"]),
            "reconstructed": _source_stats(pnls_by_source["reconstructed"], exit_reasons_by_source["reconstructed"]),
        },
    }
    if n == 0:
        return result_out

    pnls = all_pnls
    result_out["exit_reasons"] = combined_exit_reasons
    wins = sum(1 for p in pnls if p >= WINNER_THRESHOLD_PCT)
    result_out["win_rate_pct"] = round(100 * wins / n, 2)
    result_out["mean_pnl_pct_net"] = round(statistics.mean(pnls), 2)
    result_out["median_pnl_pct_net"] = round(statistics.median(pnls), 2)
    return result_out


def _era_days(era_start_iso: str) -> float:
    era_start = datetime.fromisoformat(era_start_iso)
    days = (datetime.now(timezone.utc) - era_start).total_seconds() / 86400
    return max(days, 1.0)


def candidate_venue_qualified_n(candidate: dict, by_mint: dict) -> int:
    """Real trade-frequency denominator: every venue-qualified mint this
    candidate's entry rule would have fired on, NOT just the subset that
    happens to have a valid exit path. A live strategy enters on every
    qualifying signal -- using the (much smaller) valid-path count here
    would silently understate trades/day by 5-20x (see T3's yield
    numbers)."""
    prog_cond = next((c for c in candidate["conditions"] if c["feature"] == "progress_at_signal"), None)
    qualified = [m for m, e in by_mint.items() if e.get("venue_state_at_signal") == "CURVE_ACTIVE"]
    if prog_cond is None:
        return len(qualified)
    thr = prog_cond["value"]
    return sum(1 for m in qualified if by_mint[m].get("progress_at_signal") is not None
               and by_mint[m]["progress_at_signal"] < thr)


def run():
    from research.config import SUPABASE_URL, SUPABASE_KEY
    from supabase import create_client
    from research.v8_candidate_registry import CANDIDATES
    from research.v8_exit_registry import EXIT_CANDIDATES
    from research.v8_collection_yield import trustworthy_collection_era_start

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    root = Path(".")

    t3 = json.loads(T3_RESULTS_FILE.read_text())
    reconstructed_valid = {r["token_address"] for r in t3["results"] if r.get("valid_usable_path")}

    forward_valid, by_mint, ambiguous = _forward_valid_mints(sb, root)
    era_start = trustworthy_collection_era_start(root)

    exit_by_id = {e["exit_id"]: e for e in EXIT_CANDIDATES}

    out = {"generated_at": datetime.now(timezone.utc).isoformat(),
           "round_trip_cost_pct": ROUND_TRIP_COST_PCT, "trade_size_usd": TRADE_SIZE_USD,
           "candidates": {}}

    for cand in CANDIDATES:
        cid = cand["candidate_id"]
        mints = combined_mint_set_for_candidate(cand, forward_valid, reconstructed_valid, by_mint)
        log.info("%s: combined mint set n=%d", cid, len(mints))
        if len(mints) < 100:
            out["candidates"][cid] = {"combined_n": len(mints), "below_MIN_PATH_N_floor": True}
            continue

        era_days = _era_days(era_start.isoformat())
        real_signal_n = candidate_venue_qualified_n(cand, by_mint)
        trades_per_day = real_signal_n / era_days

        cand_out = {"combined_n": len(mints), "below_MIN_PATH_N_floor": False,
                    "real_venue_qualified_n": real_signal_n, "era_days": round(era_days, 1),
                    "trades_per_day_assumed": round(trades_per_day, 3), "exits": {}}
        for exit_id in _EXIT_SPECS:
            exit_spec_dict = exit_by_id.get(exit_id)
            if exit_spec_dict is None:
                continue
            stats = replay_candidate_exit(cand, exit_spec_dict, mints, by_mint, ambiguous, root)
            if stats["n"] > 0:
                stats["dollars_per_day"] = round(
                    (stats["mean_pnl_pct_net"] / 100) * TRADE_SIZE_USD * trades_per_day, 4)
            cand_out["exits"][exit_id] = stats
        out["candidates"][cid] = cand_out

    OUTPUT_FILE.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", OUTPUT_FILE)

    print(f"\n{'=' * 90}")
    print("  THR-BATCH T5 -- E0-E3 exit evaluation, combined corpus")
    print("  (by_source split is load-bearing: reconstructed paths average ~8 ticks/"
          "few seconds -- T3 -- and mostly exit via path_end; forward paths run long")
    print("   enough for hard_stop/trail_stop/time_stop to actually fire. Don't read")
    print("   the combined row alone.)")
    print(f"{'=' * 90}")
    for cid, cdata in out["candidates"].items():
        print(f"\n  {cid}  combined_n={cdata['combined_n']}"
              f"{'  (BELOW MIN_PATH_N=100 FLOOR)' if cdata.get('below_MIN_PATH_N_floor') else ''}")
        for exit_id, stats in cdata.get("exits", {}).items():
            if stats["n"] == 0:
                print(f"    {exit_id}: n=0  exclusions={stats['exclusion_reasons']}")
                continue
            print(f"    {exit_id} COMBINED: n={stats['n']:>4}  win_rate={stats['win_rate_pct']:>6.1f}%  "
                  f"mean_net={stats['mean_pnl_pct_net']:>+7.2f}%  median_net={stats['median_pnl_pct_net']:>+7.2f}%  "
                  f"$/day={stats.get('dollars_per_day', 'n/a')}  exit_reasons={stats.get('exit_reasons')}")
            for src in ("forward", "reconstructed"):
                s = stats["by_source"].get(src, {})
                if s.get("n", 0) == 0:
                    print(f"      {src}: n=0")
                    continue
                print(f"      {src}: n={s['n']:>4}  win_rate={s['win_rate_pct']:>6.1f}%  "
                      f"mean_net={s['mean_pnl_pct_net']:>+7.2f}%  median_net={s['median_pnl_pct_net']:>+7.2f}%  "
                      f"exit_reasons={s['exit_reasons']}")
    print(f"\n{'=' * 90}\n")
    return out


def main():
    parser = argparse.ArgumentParser(description="THR-BATCH T5")
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
