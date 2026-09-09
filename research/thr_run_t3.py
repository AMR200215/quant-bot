"""
research/thr_run_t3.py — THR-BATCH T3: the reconstruction run.

Reads the pre-registered T2 sample (research/thr_t2_sample.json, frozen
before this ever runs), reconstructs each token's on-curve trade history
via research/thr_reconstruct_paths.py, classifies via the existing
path-integrity checker (research/v8_path_integrity.py) AND the
interpolated xval gate (XVAL_TOLERANCE_PCT, T1), and persists VALID +
xval-PASS-or-INSUFFICIENT_DATA paths to
logs/research_paths/reconstructed/<mint>.csv.gz -- schema-compatible with
the existing forward-collection corpus (source="reconstructed_curve_exact").

A token's xval status is INSUFFICIENT_DATA when no reconstructed row
overlapped any independent reference point (missing vsol_at_signal and
all poll marks null). Per T1's own spec ("mismatch beyond tolerance ->
INVALID_XVAL"), that is not itself a mismatch, so INSUFFICIENT_DATA does
not disqualify a path -- it's reported distinctly, not silently folded
into PASS.

Writes a result manifest (research/thr_t3_results.json) so T4/T5 can find
the reconstructed corpus WITHOUT this script mutating research_tokens.
path_file, any frozen registry, or any other shared live-pipeline state.
The reconstructed corpus is additive and separate by construction.

Run:
    python -m research.thr_run_t3 [--sample-file PATH] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("thr_run_t3")

_PUBLIC_RPC = "https://api.mainnet-beta.solana.com"
SAMPLE_FILE = Path(__file__).parent / "thr_t2_sample.json"
RESULTS_FILE = Path(__file__).parent / "thr_t3_results.json"
OUT_DIR = Path("logs/research_paths/reconstructed")
_CHUNK = 200  # Supabase .in_() batch size for detail fetch


def _alert_time_to_epoch(alert_time: str):
    if not alert_time:
        return None
    try:
        return datetime.fromisoformat(alert_time.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _fetch_token_details(sb, token_addresses: list) -> dict:
    """Batch-fetch full alert-time context (poll marks, vsol_at_signal) --
    the T2 manifest only carries address/alert_time/admission info, not
    the fields the xval gate needs."""
    out: dict = {}
    for i in range(0, len(token_addresses), _CHUNK):
        chunk = token_addresses[i:i + _CHUNK]
        resp = (sb.table("research_tokens")
                .select("token_address,alert_time,pct_change_peak,price_t1m,price_t3m,"
                        "price_t5m,price_t10m,price_t20m,vsol_at_signal,progress_capture_lag_ms")
                .in_("token_address", chunk).execute())
        for r in (resp.data or []):
            out[r["token_address"]] = r
    return out


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


def _write_reconstructed_csv(mint: str, rows: list, out_dir: Path) -> str:
    from research.path_schema import PATH_HEADER
    out_dir.mkdir(parents=True, exist_ok=True)
    gz_path = out_dir / f"{mint}.csv.gz"
    with gzip.open(gz_path, "wt", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PATH_HEADER)
        for row in rows:
            writer.writerow([row.get(col, "") for col in PATH_HEADER])
    return str(gz_path)


def process_token(token_entry: dict, detail: dict, sol_price: float, out_dir: Path) -> dict:
    """Pure-ish core (network calls happen inside via the reused fetch
    helpers) -- one token in, one result dict out. Split out from run()
    so it's independently testable with a fake sig/tx fetcher."""
    from research.backfill_paths import _fetch_sigs, _parse_txs_std
    from research.thr_reconstruct_paths import extract_rows_exact, compute_interpolated_xval, classify_interpolated_xval
    from research.v8_path_integrity import assess_path_integrity, PathIntegrityStatus

    mint = token_entry["token_address"]
    alert_ts = _alert_time_to_epoch(token_entry["alert_time"])

    sigs = _fetch_sigs(mint, _PUBLIC_RPC, alert_ts=alert_ts)
    if not sigs:
        return {"token_address": mint, "status": "NO_SIGS", "valid_usable_path": False}

    tx_results = _parse_txs_std(sigs, _PUBLIC_RPC)
    exact_rows, heuristic_candidates = extract_rows_exact(
        tx_results, mint, sol_price, research_event_id=token_entry.get("event_id", ""))

    if not exact_rows:
        return {"token_address": mint, "status": "NO_EXACT_ROWS", "valid_usable_path": False,
                "n_tx": len(tx_results)}

    integ = assess_path_integrity(exact_rows)
    interp_diffs = compute_interpolated_xval(exact_rows, detail, alert_ts, sol_price) if alert_ts else []
    xval = classify_interpolated_xval(interp_diffs)

    integrity_ok = integ.status == PathIntegrityStatus.VALID.value
    xval_ok = xval.status in ("PASS", "INSUFFICIENT_DATA")
    valid = integrity_ok and xval_ok

    path_file = None
    if valid:
        path_file = _write_reconstructed_csv(mint, exact_rows, out_dir)

    return {
        "token_address": mint, "status": "RECONSTRUCTED", "valid_usable_path": valid,
        "n_exact_rows": len(exact_rows), "n_heuristic_fallback_tx": len(heuristic_candidates),
        "integrity_status": integ.status, "xval_status": xval.status,
        "xval_max_abs_pct_diff": xval.max_abs_pct_diff, "xval_n_comparable": len(interp_diffs),
        "path_file": path_file,
    }


def run(sample_file: Path = SAMPLE_FILE, results_file: Path = RESULTS_FILE, out_dir: Path = OUT_DIR):
    from research.config import SUPABASE_URL, SUPABASE_KEY
    from supabase import create_client

    manifest = json.loads(sample_file.read_text())
    tokens = manifest["tokens"]
    log.info("Loaded pre-registered sample: %d tokens (seed=%s, generated_at=%s)",
              len(tokens), manifest.get("seed"), manifest.get("generated_at"))

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    details = _fetch_token_details(sb, [t["token_address"] for t in tokens])
    sol_price = _get_sol_price()
    log.info("SOL price: $%.2f  |  detail rows fetched: %d/%d", sol_price, len(details), len(tokens))

    results = []
    for i, t in enumerate(tokens, 1):
        mint = t["token_address"]
        detail = details.get(mint, t)
        if i % 25 == 0 or i == 1:
            log.info("[%d/%d] %s", i, len(tokens), mint[:8])
        try:
            result = process_token(t, detail, sol_price, out_dir)
        except Exception as e:
            log.warning("  %s failed: %s", mint[:8], e)
            result = {"token_address": mint, "status": "ERROR", "valid_usable_path": False, "error": str(e)}
        results.append(result)
        time.sleep(0.1)

    valid_n = sum(1 for r in results if r.get("valid_usable_path"))
    reconstructed_n = sum(1 for r in results if r["status"] == "RECONSTRUCTED")

    out_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_file": str(sample_file), "sample_seed": manifest.get("seed"),
        "n_attempted": len(tokens), "n_reconstructed": reconstructed_n, "n_valid_usable_path": valid_n,
        "out_dir": str(out_dir), "results": results,
    }
    results_file.write_text(json.dumps(out_manifest, indent=2))

    log.info("=" * 78)
    log.info("T3 COMPLETE: attempted=%d reconstructed=%d (%.1f%%) valid_usable_path=%d (%.1f%% of attempted)",
              len(tokens), reconstructed_n, 100 * reconstructed_n / max(1, len(tokens)),
              valid_n, 100 * valid_n / max(1, len(tokens)))
    log.info("Results manifest: %s", results_file)
    log.info("=" * 78)
    return out_manifest


def main():
    parser = argparse.ArgumentParser(description="THR-BATCH T3 reconstruction run")
    parser.add_argument("--sample-file", type=Path, default=SAMPLE_FILE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    run(sample_file=args.sample_file, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
