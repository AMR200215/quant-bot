"""
Phase 1 seed — populates the wallet intelligence DB from:
  1. memecoin/data/whale_wallets_sol.json  — all known SOL wallets
  2. memecoin/data/whale_wallets_bnb.json  — all known BSC wallets
  3. memecoin/data/whale_stats.json        — ranker output (scores, win rates)
     (only present after rank_wallets_helius.py has run)

Run:
    python3 -m wallet_db.seed
    python3 -m wallet_db.seed --dry-run   (show counts without writing)
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path

from wallet_db.db import init_db, get_conn, SQLITE_PATH as DB_PATH

ROOT        = Path(__file__).parent.parent
SOL_FILE    = ROOT / "memecoin" / "data" / "whale_wallets_sol.json"
BNB_FILE    = ROOT / "memecoin" / "data" / "whale_wallets_bnb.json"
STATS_FILE  = ROOT / "memecoin" / "data" / "whale_stats.json"

# Tier cutoffs matching config.py
TIER_S_TOP  = 0.10   # top 10%
TIER_A_TOP  = 0.30   # next 20%
TIER_B_TOP  = 0.60   # next 30%
MIN_TRADES  = 3      # must have ≥3 trades to receive a tier


def _assign_tier(rank: int, total_ranked: int, trades: int) -> str | None:
    if trades < MIN_TRADES:
        return None
    pct = rank / total_ranked
    if pct <= TIER_S_TOP:
        return "S"
    if pct <= TIER_A_TOP:
        return "A"
    if pct <= TIER_B_TOP:
        return "B"
    return None


def seed(dry_run: bool = False):
    init_db()
    conn = get_conn()
    now  = int(time.time())
    today = date.today().isoformat()

    # ------------------------------------------------------------------
    # 1. Load all wallets from JSON files
    # ------------------------------------------------------------------
    sol_wallets = json.loads(SOL_FILE.read_text()) if SOL_FILE.exists() else []
    bnb_wallets = json.loads(BNB_FILE.read_text()) if BNB_FILE.exists() else []

    print(f"SOL wallets:  {len(sol_wallets)}")
    print(f"BSC wallets:  {len(bnb_wallets)}")

    wallet_rows = []
    for w in sol_wallets:
        addr = w.get("trackedWalletAddress") or w.get("address", "")
        if addr:
            wallet_rows.append((addr, "solana", w.get("name", ""), "initial", now))

    for w in bnb_wallets:
        addr = w.get("address", "")
        if addr:
            wallet_rows.append((addr, "bsc", w.get("name", ""), "initial", now))

    # ------------------------------------------------------------------
    # 2. Load ranker stats if available
    # ------------------------------------------------------------------
    stats_by_wallet: dict[str, dict] = {}
    if STATS_FILE.exists():
        stats = json.loads(STATS_FILE.read_text())
        for s in stats:
            stats_by_wallet[s["wallet"]] = s
        print(f"Ranker stats: {len(stats_by_wallet)} wallets scored")
    else:
        print("Ranker stats: not found — run rank_wallets_helius.py first")

    # Assign tiers based on rank position in stats (already sorted by score desc)
    ranked = [s for s in (json.loads(STATS_FILE.read_text()) if STATS_FILE.exists() else [])
              if s.get("trades", 0) >= MIN_TRADES and not s.get("skip")]
    total_ranked = len(ranked)
    tier_by_wallet: dict[str, str | None] = {}
    score_by_wallet: dict[str, float] = {}
    for rank, s in enumerate(ranked, 1):
        tier = _assign_tier(rank, total_ranked, s.get("trades", 0))
        tier_by_wallet[s["wallet"]] = tier
        score_by_wallet[s["wallet"]] = s.get("score", 0.0)

    # ------------------------------------------------------------------
    # 3. Upsert wallets
    # ------------------------------------------------------------------
    added = skipped = tiered = 0
    history_rows = []

    for addr, chain, name, source, ts in wallet_rows:
        stats  = stats_by_wallet.get(addr, {})
        tier   = tier_by_wallet.get(addr)
        score  = score_by_wallet.get(addr, 0.0)

        # Determine status from ranker data
        if stats.get("skip") == "inactive":
            status = "dormant"
        else:
            status = "active"

        last_trade_ts = stats.get("newest_trade") or None

        if dry_run:
            skipped += 1
            continue

        conn.execute(
            """
            INSERT INTO wallets (address, chain, name, source, first_seen_ts,
                                 last_trade_ts, current_tier, current_score, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(address, chain) DO UPDATE SET
                name           = excluded.name,
                last_trade_ts  = excluded.last_trade_ts,
                current_tier   = excluded.current_tier,
                current_score  = excluded.current_score,
                status         = excluded.status
            """,
            (addr, chain, name, source, ts,
             int(last_trade_ts) if last_trade_ts else None,
             tier, score, status),
        )
        added += 1
        if tier:
            tiered += 1

        # Write score history row if we have stats
        if stats.get("trades", 0) >= MIN_TRADES:
            history_rows.append((
                addr, chain, today, tier,
                stats.get("win_rate"),
                stats.get("avg_roi"),
                stats.get("trades"),
            ))

    if not dry_run:
        if history_rows:
            conn.executemany(
                """
                INSERT OR IGNORE INTO wallet_scores_history
                    (wallet_address, chain, score_date, tier, hit_rate, median_return, trade_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                history_rows,
            )
        conn.commit()

    conn.close()

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print(f"\n{'DRY RUN — ' if dry_run else ''}Seed complete")
    print(f"  Wallets upserted:  {added}")
    print(f"  Tiered (S/A/B):    {tiered}")
    print(f"  Score history rows: {len(history_rows)}")
    print(f"  DB: {DB_PATH}")

    if not dry_run and total_ranked > 0:
        # Print tier breakdown
        s_count = sum(1 for t in tier_by_wallet.values() if t == "S")
        a_count = sum(1 for t in tier_by_wallet.values() if t == "A")
        b_count = sum(1 for t in tier_by_wallet.values() if t == "B")
        print(f"\nTier breakdown (from ranker):")
        print(f"  S-tier: {s_count} wallets")
        print(f"  A-tier: {a_count} wallets")
        print(f"  B-tier: {b_count} wallets")
        print(f"  Untiered (insufficient trades): {total_ranked - s_count - a_count - b_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    seed(args.dry_run)
