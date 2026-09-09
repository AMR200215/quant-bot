"""
research/thr_sample_t2.py — THR-BATCH T2: pre-registered reconstruction
sample.

Draws the FIXED sample T3 will reconstruct against, and writes it to
research/thr_t2_sample.json BEFORE any reconstruction runs -- committed to
the repo so the sample can't be quietly redrawn after seeing results.
"Pre-registered" means this file's contents are frozen the moment it's
generated; T3 reads it back rather than re-querying, so a bad result
can't be papered over by re-sampling.

Population: clean-era (chain=solana, progress_data_ok=True), CURVE_ACTIVE
at signal, TRAIN+VALIDATION ONLY -- holdout hard-excluded via
grouped_chronological_split (same utility used everywhere else in this
project) and asserted, not just filtered.

Sampling method: uniform simple random sample, fixed seed, from the full
eligible population -- not "most recent N" (T1's pilot used that
ordering and it introduced run-to-run non-determinism against a
live-growing table; not appropriate for a pre-registered sample). A
uniform random sample over a fixed population is IPW-consistent by
construction (every unit has the same, known inclusion probability
n_sample/n_population -- no differential reweighting needed for the
sampling step itself). This is a DIFFERENT selection process from
memecoin/progress_capture.py's P16-3 budget-paced admission controller
(research/v8_ipw.py's path_sampling_probability) -- THR reconstruction
reads directly from chain history, so it is not subject to that
controller's budget constraint at all. Each sampled token's original
path_sampling_probability (if it went through admission) is still
recorded in the output file, informationally, for anyone who later wants
to combine reconstructed data with the live-admitted population.

Run:
    python -m research.thr_sample_t2 [--target-n 500] [--seed 20260909]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("thr_sample_t2")

DEFAULT_TARGET_N = 500
DEFAULT_SEED = 20260909  # fixed, pre-registered -- do not change after a run to redraw
SAMPLE_FILE = Path(__file__).parent / "thr_t2_sample.json"
ADMISSION_LOG_PATH = Path("logs/research_admission/admission_log.jsonl")


def _alert_time_to_epoch(alert_time: str):
    if not alert_time:
        return None
    try:
        return datetime.fromisoformat(alert_time.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _load_admission_probabilities(path: Path) -> dict:
    """token_address -> (path_sampling_probability, admission_reason), last write wins.
    Informational only -- see module docstring. Missing file / missing
    token both resolve to (None, None), never an error."""
    out: dict = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            addr = rec.get("token_address")
            if not addr:
                continue
            out[addr] = (rec.get("path_sampling_probability"), rec.get("admission_reason"))
    return out


def build_sample(sb, target_n: int = DEFAULT_TARGET_N, seed: int = DEFAULT_SEED,
                  admission_log_path: Path = ADMISSION_LOG_PATH) -> dict:
    from research.v8_split import grouped_chronological_split

    rows, offset, batch = [], 0, 1000
    while True:
        resp = (sb.table("research_tokens")
                .select("event_id,token_address,alert_time,venue_state_at_signal")
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

    if len(split_rows) < 2:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(), "seed": seed,
            "target_n": target_n, "population_n": 0, "sample_n": 0,
            "holdout_mints_excluded": 0, "tokens": [],
        }

    result = grouped_chronological_split(split_rows, lambda r: r["token_address"], lambda r: r["_epoch"])
    holdout_mints = {r["token_address"] for r in result.holdout}

    train_val = [r["_orig"] for r in (result.train + result.validation)]

    # dedupe by mint -- keep the earliest alert per mint, deterministic tie-break
    by_mint: dict = {}
    for r in sorted(train_val, key=lambda r: (r["alert_time"], r["token_address"])):
        m = r["token_address"]
        if m not in by_mint:
            by_mint[m] = r
    population = list(by_mint.values())

    assert not (set(by_mint) & holdout_mints), "holdout leakage in T2 population"

    rng = random.Random(seed)
    sample_n = min(target_n, len(population))
    sampled = rng.sample(population, sample_n)
    sampled.sort(key=lambda r: r["token_address"])  # deterministic output order

    admission = _load_admission_probabilities(admission_log_path)
    tokens_out = []
    for r in sampled:
        prob, reason = admission.get(r["token_address"], (None, None))
        tokens_out.append({
            "token_address": r["token_address"], "alert_time": r["alert_time"],
            "event_id": r.get("event_id", ""),
            "path_sampling_probability": prob, "admission_reason": reason,
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "target_n": target_n,
        "population_n": len(population),
        "sample_n": len(tokens_out),
        "holdout_mints_excluded": len(holdout_mints),
        "tokens": tokens_out,
    }


def main():
    parser = argparse.ArgumentParser(description="THR-BATCH T2 pre-registered sample")
    parser.add_argument("--target-n", type=int, default=DEFAULT_TARGET_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    from research.config import SUPABASE_URL, SUPABASE_KEY
    from supabase import create_client

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    manifest = build_sample(sb, target_n=args.target_n, seed=args.seed)

    SAMPLE_FILE.write_text(json.dumps(manifest, indent=2))
    log.info("Wrote %d tokens (population=%d, holdout_excluded=%d) to %s",
              manifest["sample_n"], manifest["population_n"], manifest["holdout_mints_excluded"], SAMPLE_FILE)


if __name__ == "__main__":
    main()
