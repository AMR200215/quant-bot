"""
research/thr_pilot_t1.py — THR-BATCH T1 pilot: exact curve-reserve
reconstruction + cross-validation gate, run over a small real sample.

Purpose: prove the mechanism (research/thr_reconstruct_paths.py) end-to-end
on real transactions before T2 commits to a ~500-token production run, and
derive the xval tolerance from the REAL observed mismatch distribution
(never invented ahead of seeing data).

Population for the pilot sample: clean-era (progress_data_ok=True,
chain=solana), CURVE_ACTIVE at signal, TRAIN+VALIDATION ONLY -- holdout is
hard-excluded via grouped_chronological_split and never fetched, same
split utility used everywhere else in this project. No credit-cap concern
at this scale (20 tokens); T2 handles the pre-registered larger sample and
its own dry-run cost estimate.

Reuses (does not duplicate) research/backfill_paths.py's signature-
fetching and batched-getTransaction machinery (_fetch_sigs, _parse_txs_std)
-- same K3-fixed retry/fallback/rate-limit behavior, not reimplemented.

Read-only. Writes nothing to Supabase or disk except the printed report.

Run:
    python -m research.thr_pilot_t1 [--n 20]
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("thr_pilot_t1")

_PUBLIC_RPC = "https://api.mainnet-beta.solana.com"


def _alert_time_to_epoch(alert_time: str):
    if not alert_time:
        return None
    try:
        return datetime.fromisoformat(alert_time.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _select_pilot_tokens(sb, n: int) -> list:
    """Train+validation only, CURVE_ACTIVE, most-recent-first. Holdout
    bucket is computed via the same grouped_chronological_split used
    everywhere else and its members are never included here."""
    from research.v8_split import grouped_chronological_split

    rows, offset, batch = [], 0, 1000
    while True:
        resp = (sb.table("research_tokens")
                .select("event_id,token_address,alert_time,venue_state_at_signal,"
                        "pct_change_peak,price_t1m,price_t3m,price_t5m,price_t10m,price_t20m,"
                        "vsol_at_signal,progress_capture_lag_ms")
                .eq("chain", "solana").eq("progress_data_ok", True)
                .eq("venue_state_at_signal", "CURVE_ACTIVE")
                .range(offset, offset + batch - 1).execute())
        chunk = resp.data or []
        rows.extend(chunk)
        if len(chunk) < batch:
            break
        offset += batch

    events_with_time = [r for r in rows if r.get("alert_time")]
    split_rows = [{"token_address": r["token_address"], "_epoch": _alert_time_to_epoch(r["alert_time"]), "_orig": r}
                  for r in events_with_time]
    split_rows = [r for r in split_rows if r["_epoch"] is not None]
    result = grouped_chronological_split(split_rows, lambda r: r["token_address"], lambda r: r["_epoch"])
    holdout_mints = {r["token_address"] for r in result.holdout}

    train_val = [r["_orig"] for r in (result.train + result.validation)]
    train_val.sort(key=lambda r: r["alert_time"], reverse=True)

    # dedupe by mint, keep most recent alert
    seen = set()
    picked = []
    for r in train_val:
        m = r["token_address"]
        if m in seen or m in holdout_mints:
            continue
        seen.add(m)
        picked.append(r)
        if len(picked) >= n:
            break

    assert not (seen & holdout_mints), "holdout leakage in pilot selection"
    return picked


def _get_sol_price() -> float:
    import requests
    try:
        r = requests.get("https://api.jup.ag/price/v2?ids=So11111111111111111111111111111111111111112", timeout=8)
        if r.status_code == 200:
            entry = (r.json().get("data") or {}).get("So11111111111111111111111111111111111111112")
            if entry:
                p = float(entry.get("price") or 0)
                if p > 0:
                    return p
    except Exception as e:
        log.warning("SOL price fetch failed: %s -- using $170.00", e)
    return 170.0


def run_pilot(n: int = 20):
    from research.config import SUPABASE_URL, SUPABASE_KEY
    from supabase import create_client
    from research.backfill_paths import _fetch_sigs, _parse_txs_std
    from research.thr_reconstruct_paths import (
        extract_rows_exact, compute_xval_diffs, classify_xval,
        compute_interpolated_xval, independent_reference_points,
    )
    from research.v8_path_integrity import assess_path_integrity, PathIntegrityStatus

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    sol_price = _get_sol_price()
    log.info("SOL price: $%.2f", sol_price)

    tokens = _select_pilot_tokens(sb, n)
    log.info("Selected %d pilot tokens (train+validation only, holdout excluded)", len(tokens))

    per_token_results = []
    all_diffs = []  # raw (label, pct_diff, staleness_s) triples across the whole pilot (fixed-offset gate)
    all_interp_diffs = []  # raw pct_diff values across the whole pilot (interpolated gate)
    worked_example = None

    for i, tok in enumerate(tokens, 1):
        mint = tok["token_address"]
        alert_ts = _alert_time_to_epoch(tok["alert_time"])
        log.info("[%d/%d] %s alert=%s", i, len(tokens), mint[:8], tok["alert_time"])

        sigs = _fetch_sigs(mint, _PUBLIC_RPC, alert_ts=alert_ts)
        time.sleep(0.1)
        if not sigs:
            per_token_results.append({"mint": mint, "status": "NO_SIGS"})
            continue

        tx_results = _parse_txs_std(sigs, _PUBLIC_RPC)
        exact_rows, heuristic_candidates = extract_rows_exact(tx_results, mint, sol_price,
                                                                research_event_id=tok.get("event_id", ""))

        if not exact_rows:
            per_token_results.append({"mint": mint, "status": "NO_EXACT_ROWS",
                                       "n_tx": len(tx_results), "n_heuristic_candidates": len(heuristic_candidates)})
            continue

        integ = assess_path_integrity(exact_rows)
        diffs = compute_xval_diffs(exact_rows, tok, alert_ts) if alert_ts else []
        for d in diffs:
            if d.pct_diff is not None:
                all_diffs.append((d.label, d.pct_diff, d.staleness_s))

        interp_diffs = compute_interpolated_xval(exact_rows, tok, alert_ts, sol_price) if alert_ts else []
        for d in interp_diffs:
            all_interp_diffs.append(d.pct_diff)
        n_ref_points = len(independent_reference_points(tok, alert_ts, sol_price)) if alert_ts else 0

        result = {
            "mint": mint, "status": "RECONSTRUCTED",
            "n_exact_rows": len(exact_rows), "n_heuristic_fallback_tx": len(heuristic_candidates),
            "integrity_status": integ.status,
            "diffs": diffs,
            "interp_diffs": interp_diffs, "n_ref_points": n_ref_points,
            "has_vsol_at_signal": tok.get("vsol_at_signal") is not None,
        }
        per_token_results.append(result)

        if worked_example is None and integ.status == PathIntegrityStatus.VALID.value and diffs:
            worked_example = {"mint": mint, "alert_time": tok["alert_time"],
                               "exact_rows": exact_rows, "diffs": diffs, "n_tx": len(tx_results),
                               "interp_diffs": interp_diffs}

        time.sleep(0.1)

    # --- Distribution of real mismatches (for tolerance derivation) ---
    # STALENESS_CUT_S: a transparent, stated pragmatic cut (half of the
    # smallest poll offset, T1m=60s) separating "the reconstructed
    # reference row is close enough to the target offset that a pct_diff
    # mainly reflects reconstruction (dis)agreement" from "the reference
    # row is stale and a large pct_diff more likely reflects real token
    # movement in the gap, not a reconstruction error." Not a claim that
    # 30s is the true right answer -- a stated, revisitable choice.
    STALENESS_CUT_S = 30.0
    peak_diffs = [(lbl, v, s) for lbl, v, s in all_diffs if lbl == "peak"]
    offset_diffs = [(lbl, v, s) for lbl, v, s in all_diffs if lbl != "peak"]
    fresh = [(lbl, v, s) for lbl, v, s in offset_diffs if s is not None and abs(s) <= STALENESS_CUT_S]
    stale = [(lbl, v, s) for lbl, v, s in offset_diffs if s is not None and abs(s) > STALENESS_CUT_S]

    def _dist(label, triples):
        vals = sorted(abs(v) for _, v, _ in triples)
        print(f"\n  {label} (n={len(vals)}):")
        if not vals:
            print("    (none)")
            return
        def pct(p):
            idx = min(len(vals) - 1, int(round(p / 100 * (len(vals) - 1))))
            return vals[idx]
        print(f"    p50={pct(50):.3f}%  p75={pct(75):.3f}%  p90={pct(90):.3f}%  "
              f"p95={pct(95):.3f}%  max={vals[-1]:.3f}%")
        if len(vals) >= 2:
            print(f"    mean={statistics.mean(vals):.3f}%  stdev={statistics.stdev(vals):.3f}%")

    print(f"\n{'=' * 78}")
    print("  THR-BATCH T1 PILOT REPORT")
    print(f"{'=' * 78}")
    print(f"  tokens attempted: {len(tokens)}")

    reconstructed = [r for r in per_token_results if r["status"] == "RECONSTRUCTED"]
    valid_reconstructed = [r for r in reconstructed if r["integrity_status"] == PathIntegrityStatus.VALID.value]
    print(f"  reconstructed (>=1 exact row): {len(reconstructed)}  ({100*len(reconstructed)/max(1,len(tokens)):.1f}%)")
    print(f"  integrity-VALID among those:   {len(valid_reconstructed)}  "
          f"({100*len(valid_reconstructed)/max(1,len(reconstructed)):.1f}% of reconstructed)")
    for status in ("NO_SIGS", "NO_EXACT_ROWS"):
        n = sum(1 for r in per_token_results if r["status"] == status)
        if n:
            print(f"  {status}: {n}")

    print(f"\n  [OLD] Fixed-offset mismatch distributions (|pct_diff|), staleness-conditioned "
          f"(cut={STALENESS_CUT_S:.0f}s, half the smallest poll offset T1m=60s):")
    _dist(f"FRESH (staleness<={STALENESS_CUT_S:.0f}s -- reconstruction-agreement signal)", fresh)
    _dist(f"STALE (staleness>{STALENESS_CUT_S:.0f}s -- confounded by real token movement in the gap)", stale)
    _dist("peak (reconstructed max vs poll-implied peak -- staleness not well-defined the same way)", peak_diffs)

    interp_abs = sorted(abs(v) for v in all_interp_diffs)
    tokens_with_vsol_at_signal = sum(1 for r in per_token_results
                                      if r["status"] == "RECONSTRUCTED" and r.get("has_vsol_at_signal"))
    print(f"\n  [NEW] Interpolated cross-validation (compares each reconstructed row against an "
          f"independent price interpolated AT THAT ROW'S OWN TIMESTAMP -- no staleness confound):")
    print(f"    reconstructed tokens with vsol_at_signal available: {tokens_with_vsol_at_signal}/{len(reconstructed)}")
    _dist(f"interpolated |pct_diff|", [(None, v, None) for v in all_interp_diffs])

    print(f"\n  Per-token detail:")
    for r in per_token_results:
        if r["status"] != "RECONSTRUCTED":
            print(f"    {r['mint'][:12]}  {r['status']}")
            continue
        diff_s = "  ".join(
            f"{d.label}={d.pct_diff:+.2f}%(stale={d.staleness_s:.0f}s)" if d.staleness_s is not None
            else f"{d.label}={d.pct_diff:+.2f}%"
            for d in r["diffs"] if d.pct_diff is not None
        )
        interp_s = "  ".join(f"interp={d.pct_diff:+.2f}%" for d in r["interp_diffs"])
        print(f"    {r['mint'][:12]}  rows={r['n_exact_rows']:>4}  heuristic_fallback_tx={r['n_heuristic_fallback_tx']:>3}  "
              f"integrity={r['integrity_status']:<7}  ref_points={r['n_ref_points']}  vsol_at_signal={r['has_vsol_at_signal']}")
        if diff_s:
            print(f"      [OLD] {diff_s}")
        if interp_s:
            print(f"      [NEW] {interp_s}")
        elif r["n_exact_rows"] > 0:
            print(f"      [NEW] (no reconstructed row fell within the independent reference range)")

    if worked_example:
        print(f"\n  Worked example: {worked_example['mint']}  (alert {worked_example['alert_time']})")
        print(f"    raw transactions fetched: {worked_example['n_tx']}")
        print(f"    exact on-curve rows reconstructed: {len(worked_example['exact_rows'])}")
        first, last = worked_example["exact_rows"][0], worked_example["exact_rows"][-1]
        print(f"    first tick: ts_ms={first['ts_ms']} price_usd={first['price_usd']:.10f} vsol={first['vsol']:.4f}")
        print(f"    last  tick: ts_ms={last['ts_ms']}  price_usd={last['price_usd']:.10f} vsol={last['vsol']:.4f}")
        for d in worked_example["diffs"]:
            if d.pct_diff is not None:
                stale_s = f"  staleness={d.staleness_s:.0f}s" if d.staleness_s is not None else ""
                print(f"    [OLD] xval {d.label}: reconstructed={d.reconstructed_price:.10f}  "
                      f"poll={d.poll_price:.10f}  diff={d.pct_diff:+.3f}%{stale_s}")
        for d in worked_example["interp_diffs"]:
            print(f"    [NEW] xval @ts={d.ts_ms}: reconstructed={d.reconstructed_price:.10f}  "
                  f"interpolated={d.interpolated_price:.10f}  diff={d.pct_diff:+.3f}%")

    print(f"\n{'=' * 78}\n")
    return per_token_results, all_diffs


def main():
    parser = argparse.ArgumentParser(description="THR-BATCH T1 pilot")
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    run_pilot(n=args.n)


if __name__ == "__main__":
    main()
