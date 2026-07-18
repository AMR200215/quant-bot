"""
Backfill trade-path CSVs from Helius parsed-transaction history.

Fetches on-chain trade activity for:
  • Up to --winners  tokens with pct_change_peak ≥ --win-thresh  (default 200 / +50%)
  • Up to --losers   tokens with pct_change_peak <  0             (default 200)

Writes per-token CSVs to logs/research_paths/backfill/<mint>.csv in the same
format as live PeakTracker paths, with an extra "source=backfill" column.
Updates research_tokens.path_file for each written file.

Hard Helius credit cap (--credit-cap, default 50000):
  Helius charges 1 credit per getSignaturesForAddress call and
  1 credit per enhanced transaction parsed.
  Each token costs: 1 (sigs) + N_parsed (parse calls).
  The script stops issuing new tokens when the running total hits the cap.

Dry-run (--dry-run):
  Prints estimated credit cost based on assumed avg tx count and exits without
  touching Helius or writing any files.

Run:
    python -m research.backfill_paths [--dry-run] [--winners N] [--losers N]
    python -m research.backfill_paths --credit-cap 20000 --win-thresh 100

CSV columns: ts_ms, price_usd, side, sol_amount, vsol, source
"""

import argparse
import csv
import gzip
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("backfill_paths")

_CSV_HEADER  = ["ts_ms", "price_usd", "side", "sol_amount", "vsol", "source"]
_SOL_MINT    = "So11111111111111111111111111111111111111112"
_LAMPORTS    = 1_000_000_000
_MAX_SIGS    = 1000   # getSignaturesForAddress limit
_PARSE_BATCH = 100    # Helius enhanced-tx batch size
_RATE_SLEEP  = 0.15  # seconds between API calls (courteous, not needed on paid plan)

# Average tx count used for credit estimation in dry-run
_DRY_RUN_AVG_TXS = 200


def _get_sol_price() -> float:
    """Fetch current SOL/USD from Jupiter price API."""
    try:
        r = requests.get(
            f"https://api.jup.ag/price/v2?ids={_SOL_MINT}", timeout=8
        )
        if r.status_code == 200:
            entry = (r.json().get("data") or {}).get(_SOL_MINT)
            if entry:
                price = float(entry.get("price") or 0)
                if price > 0:
                    log.info("SOL price: $%.2f", price)
                    return price
    except Exception as e:
        log.warning("SOL price fetch failed: %s — using $170.00", e)
    return 170.0


def _load_tokens(sb, winners: int, losers: int, win_thresh: float) -> tuple:
    """
    Returns (winner_list, loser_list) of research_token rows.
    Both lists capped at the requested counts.
    """
    def _page(query):
        rows, offset, batch = [], 0, 1000
        while True:
            chunk = (
                query
                .range(offset, offset + batch - 1)
                .execute()
                .data or []
            )
            rows.extend(chunk)
            if len(chunk) < batch:
                break
            offset += batch
        return rows

    base = (
        sb.table("research_tokens")
        .select("id,token_address,symbol,pct_change_peak,alert_time,path_file")
        .eq("outcome_complete", True)
        .eq("chain", "solana")
    )

    winner_rows = _page(
        base.gte("pct_change_peak", win_thresh).order("alert_time", desc=True)
    )[:winners]

    loser_rows = _page(
        base.lt("pct_change_peak", 0).order("alert_time", desc=True)
    )[:losers]

    return winner_rows, loser_rows


def _fetch_sigs(mint: str, rpc_url: str) -> list:
    """getSignaturesForAddress → list of sig strings (oldest first, no errors)."""
    try:
        r = requests.post(
            rpc_url,
            json={
                "jsonrpc": "2.0", "id": 1,
                "method":  "getSignaturesForAddress",
                "params":  [mint, {"limit": _MAX_SIGS, "commitment": "confirmed"}],
            },
            timeout=20,
        )
        result = r.json().get("result") or []
    except Exception as e:
        log.warning("getSignaturesForAddress failed for %s: %s", mint[:8], e)
        return []
    # result is DESC (newest first); reverse to oldest-first; skip failed txs
    result.reverse()
    return [s["signature"] for s in result if not s.get("err")]


def _parse_txs(sigs: list, parse_url: str) -> list:
    """
    Batch-parse up to _PARSE_BATCH sigs via Helius enhanced tx API.
    Returns list of parsed tx dicts.
    """
    try:
        r = requests.post(
            parse_url,
            json={"transactions": sigs[:_PARSE_BATCH]},
            timeout=30,
        )
        parsed = r.json()
        return parsed if isinstance(parsed, list) else []
    except Exception as e:
        log.warning("Enhanced TX parse failed (batch %d): %s", len(sigs), e)
        return []


def _extract_rows(parsed_txs: list, mint: str, sol_price: float) -> list:
    """
    Convert Helius enhanced tx list → CSV rows.

    Price derivation:
      sol_amount = |native balance change of fee payer| in SOL
      token_amount = tokenTransfers for this mint received/sent by fee payer
      price_usd = (sol_amount / token_amount) * sol_price   [per token]
    vsol is not available from history → stored as 0.
    side: "buy" if feePayer receives token, "sell" if feePayer sends token.
    """
    rows = []
    for tx in parsed_txs:
        if not isinstance(tx, dict):
            continue
        if tx.get("type") not in ("SWAP", "UNKNOWN"):
            continue
        ts    = tx.get("timestamp") or tx.get("blockTime")
        if not ts:
            continue
        ts_ms = int(ts) * 1000
        fee_payer = tx.get("feePayer", "")

        # Determine side and token amount
        token_amount  = 0.0
        side          = ""
        for tt in (tx.get("tokenTransfers") or []):
            if tt.get("mint") != mint:
                continue
            raw_amt = float(tt.get("tokenAmount") or 0)
            if tt.get("toUserAccount") == fee_payer:
                side         = "buy"
                token_amount = raw_amt
                break
            elif tt.get("fromUserAccount") == fee_payer:
                side         = "sell"
                token_amount = raw_amt
                break
        if not side or token_amount == 0:
            continue

        # SOL amount — absolute native balance change for fee payer
        sol_amount = 0.0
        for ad in (tx.get("accountData") or []):
            if ad.get("account") == fee_payer:
                change     = float(ad.get("nativeBalanceChange") or 0)
                sol_amount = abs(change) / _LAMPORTS
                break
        # Fallback: sum nativeTransfers involving fee payer
        if sol_amount == 0:
            for nt in (tx.get("nativeTransfers") or []):
                if nt.get("fromUserAccount") == fee_payer or nt.get("toUserAccount") == fee_payer:
                    sol_amount += abs(float(nt.get("amount") or 0)) / _LAMPORTS

        price_usd = (sol_amount / token_amount * sol_price) if token_amount > 0 else 0.0

        rows.append([ts_ms, round(price_usd, 12), side, round(sol_amount, 9), 0.0, "backfill"])

    # Sort by timestamp ascending
    rows.sort(key=lambda r: r[0])
    return rows


def _write_csv(mint: str, rows: list, out_dir: Path) -> str:
    """Write rows to out_dir/<mint>.csv (gzipped). Returns relative path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gz_path = out_dir / f"{mint}.csv.gz"
    with gzip.open(gz_path, "wt", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_HEADER)
        writer.writerows(rows)
    # relative path for DB (callers check .csv or .csv.gz)
    return f"logs/research_paths/backfill/{mint}.csv.gz"


def _update_db(sb, row_id: str, path_str: str):
    try:
        sb.table("research_tokens") \
            .update({"path_file": path_str}) \
            .eq("id", row_id) \
            .execute()
    except Exception as e:
        log.debug("path_file update failed for id %s: %s", row_id, e)


def _process_token(tok: dict, rpc_url: str, parse_url: str, sol_price: float,
                   out_dir: Path, sb, credit_budget: list) -> int:
    """
    Process one token. Updates credit_budget[0] in place.
    Returns credits consumed by this token.
    """
    mint   = tok.get("token_address", "")
    sym    = tok.get("symbol") or mint[:8]
    peak   = tok.get("pct_change_peak")
    peak_s = f"{peak:+.0f}%" if peak is not None else "n/a"

    log.info("  %s (%s) peak=%s", sym, mint[:8], peak_s)

    # Fetch signatures (1 credit)
    sigs = _fetch_sigs(mint, rpc_url)
    credit_budget[0] -= 1
    time.sleep(_RATE_SLEEP)

    if not sigs:
        log.debug("  no sigs for %s", mint[:8])
        return 1

    all_rows: list = []
    total_parsed = 0
    for i in range(0, len(sigs), _PARSE_BATCH):
        batch = sigs[i:i + _PARSE_BATCH]
        if credit_budget[0] < len(batch):
            log.warning("  credit cap reached mid-token %s — stopping parse", mint[:8])
            break
        parsed = _parse_txs(batch, parse_url)
        credits_used = len(batch)
        credit_budget[0] -= credits_used
        total_parsed += credits_used
        rows = _extract_rows(parsed, mint, sol_price)
        all_rows.extend(rows)
        time.sleep(_RATE_SLEEP)
        if credit_budget[0] <= 0:
            break

    if not all_rows:
        log.debug("  no tradeable rows extracted for %s", mint[:8])
        return 1 + total_parsed

    rel_path = _write_csv(mint, all_rows, out_dir)
    _update_db(sb, tok["id"], rel_path)
    log.info("  → %d rows → %s", len(all_rows), rel_path)
    return 1 + total_parsed


def main():
    parser = argparse.ArgumentParser(
        description="Backfill trade-path CSVs from Helius transaction history"
    )
    parser.add_argument("--dry-run",     action="store_true",
                        help="print estimated credit cost only, no files written")
    parser.add_argument("--winners",     type=int, default=200,
                        help="max winner tokens to backfill (default: 200)")
    parser.add_argument("--losers",      type=int, default=200,
                        help="max loser tokens to backfill (default: 200)")
    parser.add_argument("--win-thresh",  type=float, default=50.0,
                        help="min pct_change_peak for winners (default: 50)")
    parser.add_argument("--credit-cap",  type=int, default=50_000,
                        help="hard Helius credit cap (default: 50000)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip tokens that already have a path_file in DB")
    args = parser.parse_args()

    from research.config import SUPABASE_URL, SUPABASE_KEY, HELIUS_API_KEY, RESEARCH_PATHS_DIR

    if not SUPABASE_URL or not SUPABASE_KEY:
        log.error("SUPABASE_URL and SUPABASE_KEY required")
        sys.exit(1)
    if not HELIUS_API_KEY:
        log.error("HELIUS_API_KEY required")
        sys.exit(1)

    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    log.info("Loading tokens from Supabase...")
    winner_rows, loser_rows = _load_tokens(sb, args.winners, args.losers, args.win_thresh)
    all_tokens = winner_rows + loser_rows
    log.info("Loaded %d winners + %d losers = %d tokens",
             len(winner_rows), len(loser_rows), len(all_tokens))

    if args.skip_existing:
        before = len(all_tokens)
        all_tokens = [t for t in all_tokens if not t.get("path_file")]
        log.info("Skipped %d already-pathed tokens → %d remaining",
                 before - len(all_tokens), len(all_tokens))

    if not all_tokens:
        log.warning("No tokens to process")
        sys.exit(0)

    # Dry-run: estimate credits
    if args.dry_run:
        est_sigs     = len(all_tokens)
        est_parse    = len(all_tokens) * _DRY_RUN_AVG_TXS
        est_total    = est_sigs + est_parse
        pct_of_cap   = est_total / args.credit_cap * 100
        print("\n=== Dry-run Credit Estimate ===")
        print(f"Tokens to process:      {len(all_tokens)}")
        print(f"  getSignaturesForAddress calls: {est_sigs}   ({est_sigs} credits)")
        print(f"  Enhanced TX parse (avg {_DRY_RUN_AVG_TXS} txns/token): {est_parse} credits")
        print(f"Total estimated credits: {est_total:,}  ({pct_of_cap:.1f}% of --credit-cap {args.credit_cap:,})")
        if est_total > args.credit_cap:
            n_covered = int(args.credit_cap / (1 + _DRY_RUN_AVG_TXS))
            print(f"WARNING: estimate exceeds cap. Would fully cover ~{n_covered} tokens.")
        print(f"\nTo run for real:  python -m research.backfill_paths"
              f" --winners {args.winners} --losers {args.losers}"
              f" --credit-cap {args.credit_cap}")
        return

    sol_price = _get_sol_price()
    rpc_url   = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"
    parse_url = f"https://api.helius.xyz/v0/transactions/?api-key={HELIUS_API_KEY}"
    out_dir   = RESEARCH_PATHS_DIR / "backfill"

    credit_budget = [args.credit_cap]   # mutable wrapper for in-place updates
    done = 0
    skipped = 0

    log.info("Starting backfill (credit cap: %d, SOL price: $%.2f)",
             args.credit_cap, sol_price)

    for i, tok in enumerate(all_tokens, 1):
        if credit_budget[0] <= 0:
            log.warning("Credit cap exhausted after %d tokens. Stopping.", done)
            break
        log.info("[%d/%d] Processing %s", i, len(all_tokens),
                 tok.get("symbol") or tok.get("token_address", "")[:8])
        try:
            used = _process_token(
                tok, rpc_url, parse_url, sol_price, out_dir, sb, credit_budget
            )
            done += 1
            remaining = credit_budget[0]
            log.info("  credits used this token: %d | remaining: %d", used, remaining)
        except Exception as e:
            log.warning("  failed: %s", e)
            skipped += 1

    credits_used = args.credit_cap - credit_budget[0]
    print("\n=== Backfill Path Summary ===")
    print(f"Tokens processed:    {done}")
    print(f"Tokens skipped/err:  {skipped}")
    print(f"Credits used:        {credits_used:,} / {args.credit_cap:,}")
    print(f"Output directory:    {out_dir}")


if __name__ == "__main__":
    main()
