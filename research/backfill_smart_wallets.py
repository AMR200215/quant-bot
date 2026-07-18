"""
Backfill smart-money wallet registry.

1. Queries Supabase for all outcome-complete tokens where pct_change_peak >= THRESHOLD.
2. For each winner, fetches first N buyers via Helius parsed-tx.
3. Identifies wallets that appear early in >= MIN_WINS distinct winner tokens.
4. Writes research/smart_wallets.json.

Run:
    python -m research.backfill_smart_wallets [--dry-run] [--min-wins N] [--threshold PCT]

Examples:
    python -m research.backfill_smart_wallets
    python -m research.backfill_smart_wallets --dry-run --threshold 50
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("backfill_smart_wallets")


def _load_winners(sb, threshold: float) -> list:
    """Load all outcome-complete tokens with pct_change_peak >= threshold."""
    rows   = []
    offset = 0
    batch  = 1000
    while True:
        resp = (
            sb.table("research_tokens")
            .select("id,token_address,symbol,alert_time,pct_change_peak,pct_change_peak_3m,chain")
            .eq("outcome_complete", True)
            .gte("pct_change_peak", threshold)
            .eq("chain", "solana")
            .order("alert_time", desc=False)
            .range(offset, offset + batch - 1)
            .execute()
        )
        chunk = resp.data or []
        rows.extend(chunk)
        if len(chunk) < batch:
            break
        offset += batch
    return rows


def _update_supabase_rows(sb, wallet_set: set, winner_tokens: list, dry_run: bool):
    """
    Back-fill smart_money_hit / smart_money_count on winner tokens that already
    have outcome rows — so historical data reflects who was smart money.

    Also updates any historical token rows where we have buyer data.
    This is best-effort — skipped on dry-run.
    """
    if dry_run:
        log.info("dry-run: skipping Supabase backfill of smart_money columns")
        return

    log.info("Supabase backfill: updating %d winner tokens...", len(winner_tokens))
    updated = 0
    for tok in winner_tokens:
        buyers = tok.get("_buyers", [])
        hits   = [b for b in buyers if b in wallet_set]
        if hits:
            try:
                sb.table("research_tokens").update({
                    "smart_money_hit":   True,
                    "smart_money_count": len(hits),
                }).eq("id", tok["id"]).execute()
                updated += 1
            except Exception as e:
                log.debug("update %s failed: %s", tok.get("token_address", "")[:8], e)
    log.info("Supabase backfill done: %d rows updated with smart_money_hit=True", updated)


def main():
    parser = argparse.ArgumentParser(description="Build smart_wallets.json from Supabase winners")
    parser.add_argument("--dry-run",    action="store_true",   help="print stats only, don't write files")
    parser.add_argument("--min-wins",   type=int,   default=2, help="min winner appearances (default: 2)")
    parser.add_argument("--threshold",  type=float, default=100.0, help="min pct_change_peak (default: 100)")
    parser.add_argument("--first-n",    type=int,   default=30,    help="first N buyers per token (default: 30)")
    args = parser.parse_args()

    from research.config import SUPABASE_URL, SUPABASE_KEY, HELIUS_API_KEY, SMART_WALLETS_PATH
    from research.snapshot import fetch_first_buyers

    if not SUPABASE_URL or not SUPABASE_KEY:
        log.error("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        sys.exit(1)
    if not HELIUS_API_KEY:
        log.error("HELIUS_API_KEY must be set in .env")
        sys.exit(1)

    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    log.info("Loading winner tokens (pct_change_peak >= %.0f%%)...", args.threshold)
    winners = _load_winners(sb, args.threshold)
    log.info("Found %d winner tokens", len(winners))

    if not winners:
        log.warning("No winners found — check threshold or DB contents")
        sys.exit(0)

    # Fetch first N buyers for each winner
    wallet_token_map: dict = defaultdict(list)   # wallet → [token_address, ...]
    failed = 0
    for i, tok in enumerate(winners):
        addr  = tok.get("token_address", "")
        sym   = tok.get("symbol") or addr[:8]
        peak  = tok.get("pct_change_peak") or 0
        if not addr:
            continue
        log.info("[%d/%d] Fetching buyers for %s (%s, peak=+%.0f%%)",
                 i + 1, len(winners), sym, addr[:8], peak)
        buyers = fetch_first_buyers(addr, HELIUS_API_KEY, n=args.first_n)
        tok["_buyers"] = buyers
        if not buyers:
            log.debug("  no buyers found for %s", addr[:8])
            failed += 1
        else:
            log.debug("  %d buyers found", len(buyers))
            for w in buyers:
                wallet_token_map[w].append(addr)
        # Rate-limit: Helius paid plan is generous but be courteous
        time.sleep(0.3)

    log.info("Buyer fetch complete: %d/%d tokens had data (%d failed/empty)",
             len(winners) - failed, len(winners), failed)

    # Identify smart wallets
    smart_wallets: dict = {}
    for wallet, tokens in wallet_token_map.items():
        unique_tokens = list(dict.fromkeys(tokens))   # deduplicate, preserve order
        if len(unique_tokens) >= args.min_wins:
            smart_wallets[wallet] = {
                "first_bought_count": len(unique_tokens),
                "tokens": unique_tokens[:20],   # cap stored list at 20 for file size
            }

    log.info(
        "Smart wallets identified: %d wallets appeared early in >= %d winner tokens",
        len(smart_wallets), args.min_wins,
    )

    # Summary stats
    if smart_wallets:
        counts = sorted([v["first_bought_count"] for v in smart_wallets.values()], reverse=True)
        log.info("Top counts: %s", counts[:10])

    output = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "winner_threshold": args.threshold,
        "min_wins":         args.min_wins,
        "first_n":          args.first_n,
        "winner_count":     len(winners),
        "wallets_scanned":  len(wallet_token_map),
        "wallet_count":     len(smart_wallets),
        "wallets":          smart_wallets,
    }

    if args.dry_run:
        log.info("dry-run: would write %d wallets to %s", len(smart_wallets), SMART_WALLETS_PATH)
        print(json.dumps({k: v for k, v in output.items() if k != "wallets"}, indent=2))
        return

    SMART_WALLETS_PATH.write_text(json.dumps(output, indent=2))
    log.info("Written: %s (%d smart wallets)", SMART_WALLETS_PATH, len(smart_wallets))

    # Backfill Supabase rows for historical data
    smart_wallet_set = set(smart_wallets.keys())
    _update_supabase_rows(sb, smart_wallet_set, winners, dry_run=args.dry_run)

    print("\n=== Smart Wallet Backfill Summary ===")
    print(f"Winners scanned:    {len(winners)}")
    print(f"Winners with buyers:{len(winners) - failed}")
    print(f"Smart wallets found:{len(smart_wallets)}")
    print(f"Output: {SMART_WALLETS_PATH}")


if __name__ == "__main__":
    main()
