"""
Phase 2c/2d — Discovery pipeline.

Phase 2c (backtrace): for each winner token, find early buyers via Helius,
add unknown wallets to discovery_queue.

Phase 2d (promote): wallets in discovery_queue that appear as early buyer
of 2+ distinct winners within 14d → auto-promote to wallets table.

Run:
    python3 -m wallet_db.discovery backtrace
    python3 -m wallet_db.discovery promote
    python3 -m wallet_db.discovery all     (both)
"""

import logging
import os
import sys
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from wallet_db.db import get_conn, init_db
from wallet_db.ingest import ingest_wallet, _compute_score

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HELIUS_KEY  = os.getenv("HELIUS_API_KEY", "")
HELIUS_BASE = "https://api.helius.xyz/v0"
DEX_BASE    = "https://api.dexscreener.com"
WSOL        = "So11111111111111111111111111111111111111112"
CALL_DELAY  = 0.25

# Promotion threshold
MIN_WINNER_APPEARANCES = 2   # must appear as early buyer on 2+ winners
PROMOTION_WINDOW_DAYS  = 14
MAX_PAGES_PER_TOKEN    = 8   # max Helius pages per token (800 txs)


# ---------------------------------------------------------------------------
# Helius helpers
# ---------------------------------------------------------------------------

def _helius_get(wallet_or_address: str, params: dict) -> list:
    url = f"{HELIUS_BASE}/addresses/{wallet_or_address}/transactions"
    params["api-key"] = HELIUS_KEY
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 429:
                wait = 5 * (2 ** attempt)
                log.warning("Helius rate limit — sleeping %ds (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
                continue
            if r.status_code != 200:
                return []
            return r.json() or []
        except Exception as e:
            log.debug("Helius error: %s", e)
            time.sleep(2)
    return []


def _get_pair_address(token_address: str) -> str | None:
    """Get the best pair address for a Solana token from DexScreener."""
    try:
        r = requests.get(f"{DEX_BASE}/latest/dex/tokens/{token_address}", timeout=10)
        data = r.json()
        pairs = [p for p in (data.get("pairs") or []) if p.get("chainId") == "solana"]
        if not pairs:
            return None
        pairs.sort(key=lambda p: (p.get("liquidity") or {}).get("usd", 0) or 0, reverse=True)
        return pairs[0].get("pairAddress")
    except Exception:
        return None


def _find_early_buyers(pair_address: str, launch_ts: int) -> list[str]:
    """
    Find early buyer wallets for a token pair.
    Paginates Helius until we reach launch_time + 30min or max pages.
    Returns list of unique wallet addresses (earliest buyers first).
    """
    params  = {"limit": 100, "type": "SWAP"}
    pages   = []
    before  = None
    cutoff  = launch_ts + 30 * 60   # 30 min after launch

    for page_num in range(MAX_PAGES_PER_TOKEN):
        if before:
            params["before"] = before
        batch = _helius_get(pair_address, dict(params))
        if not batch:
            break
        pages.append(batch)

        oldest_in_batch = min(tx.get("timestamp", 9e9) for tx in batch)
        if oldest_in_batch <= cutoff:
            break   # reached early enough
        if len(batch) < 100:
            break   # last page
        before = batch[-1]["signature"]
        time.sleep(CALL_DELAY)

    if not pages:
        return []

    # Take the LAST (oldest) page — these are the earliest transactions
    earliest_batch = pages[-1]
    # Sort by timestamp ascending (oldest first)
    earliest_batch.sort(key=lambda tx: tx.get("timestamp", 0))

    buyers = []
    seen   = set()
    for tx in earliest_batch:
        if tx.get("transactionError"):
            continue
        native = tx.get("nativeTransfers", [])
        tokens = tx.get("tokenTransfers", [])
        # Find wallet that received tokens (= buyer)
        for t in tokens:
            wallet = t.get("toUserAccount", "")
            mint   = t.get("mint", "")
            if wallet and mint and mint != WSOL and wallet not in seen:
                # Verify it's a buy: wallet also sent SOL
                sol_sent = sum(n["amount"] for n in native
                               if n.get("fromUserAccount") == wallet)
                if sol_sent > 0:
                    buyers.append(wallet)
                    seen.add(wallet)
                    if len(buyers) >= 30:
                        return buyers
    return buyers


# ---------------------------------------------------------------------------
# Phase 2c — Winner backtrace
# ---------------------------------------------------------------------------

def run_backtrace():
    if not HELIUS_KEY:
        raise SystemExit("HELIUS_API_KEY not set")

    init_db()
    conn = get_conn()

    # Get winner tokens updated in last 14 days
    cutoff = int(time.time()) - 14 * 86400
    winners = conn.execute(
        """
        SELECT token_address, launch_time, peak_multiple
        FROM token_outcomes
        WHERE chain = 'solana'
          AND status = 'winner'
          AND peak_multiple >= 5.0
          AND last_updated >= ?
        ORDER BY peak_multiple DESC
        LIMIT 50
        """,
        (cutoff,),
    ).fetchall()
    conn.close()

    log.info("Phase 2c backtrace — %d winner tokens to process", len(winners))

    total_queued = 0
    for row in winners:
        token_address = row["token_address"]
        launch_ts     = row["launch_time"] or 0
        peak_mult     = row["peak_multiple"]

        log.info("  Backtracing %s (%.1fx)…", token_address[:8], peak_mult)

        pair_address = _get_pair_address(token_address)
        time.sleep(CALL_DELAY)
        if not pair_address:
            log.debug("    No pair address found")
            continue

        buyers = _find_early_buyers(pair_address, launch_ts)
        if not buyers:
            log.debug("    No buyers found")
            continue

        log.info("    Found %d early buyers", len(buyers))

        conn = get_conn()
        now  = int(time.time())
        queued = 0

        for wallet in buyers:
            # Check if already in wallets table
            known = conn.execute(
                "SELECT 1 FROM wallets WHERE address = ? AND chain = 'solana'",
                (wallet,),
            ).fetchone()

            if known:
                # Known wallet — just log the win hit (don't re-queue)
                continue

            # Check if already in discovery_queue for this token
            already = conn.execute(
                """
                SELECT 1 FROM discovery_queue
                WHERE wallet_address = ? AND chain = 'solana'
                  AND context = ?
                """,
                (wallet, token_address),
            ).fetchone()

            if not already:
                conn.execute(
                    """
                    INSERT INTO discovery_queue
                        (wallet_address, chain, source, context, discovered_at)
                    VALUES (?, 'solana', 'winner_early_buyer', ?, ?)
                    """,
                    (wallet, token_address, now),
                )
                queued += 1

        conn.commit()
        conn.close()

        total_queued += queued
        log.info("    Queued %d new wallets", queued)
        time.sleep(CALL_DELAY)

    log.info("Phase 2c complete — %d wallets added to discovery_queue", total_queued)

    # Summary
    conn = get_conn()
    pending = conn.execute(
        "SELECT COUNT(*) as n FROM discovery_queue WHERE evaluated = 0"
    ).fetchone()["n"]
    conn.close()
    log.info("Discovery queue: %d pending candidates", pending)


# ---------------------------------------------------------------------------
# Phase 2d — Multi-winner promotion
# ---------------------------------------------------------------------------

def run_promote():
    if not HELIUS_KEY:
        raise SystemExit("HELIUS_API_KEY not set")

    init_db()
    now    = int(time.time())
    cutoff = now - PROMOTION_WINDOW_DAYS * 86400
    today  = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    conn = get_conn()
    # Find wallets appearing as early buyer on 2+ distinct winners within 14d
    candidates = conn.execute(
        """
        SELECT wallet_address, COUNT(DISTINCT context) as win_count
        FROM discovery_queue
        WHERE chain      = 'solana'
          AND source     = 'winner_early_buyer'
          AND evaluated  = 0
          AND discovered_at >= ?
        GROUP BY wallet_address
        HAVING COUNT(DISTINCT context) >= ?
        """,
        (cutoff, MIN_WINNER_APPEARANCES),
    ).fetchall()
    conn.close()

    log.info("Phase 2d promote — %d multi-winner candidates", len(candidates))

    promoted = 0
    for row in candidates:
        wallet    = row["wallet_address"]
        win_count = row["win_count"]

        log.info("  Evaluating %s… (%d winner hits)", wallet[:12], win_count)

        # Pull 30d history
        summary    = ingest_wallet(wallet, "solana", now - 30 * 86400)
        time.sleep(CALL_DELAY)
        score_data = _compute_score(wallet, "solana", 30)

        conn = get_conn()

        # Add to wallets table
        conn.execute(
            """
            INSERT INTO wallets
                (address, chain, name, source, first_seen_ts, current_tier,
                 current_score, status)
            VALUES (?, 'solana', '', 'discovery_winner_backtrace', ?, NULL, ?, 'active')
            ON CONFLICT (address, chain) DO UPDATE SET
                current_score = excluded.current_score,
                status        = 'active'
            """,
            (wallet, now, score_data["score"]),
        )

        # Write score history
        conn.execute(
            """
            INSERT INTO wallet_scores_history
                (wallet_address, chain, score_date, tier, hit_rate, median_return, trade_count)
            VALUES (?, 'solana', ?, NULL, ?, ?, ?)
            ON CONFLICT (wallet_address, chain, score_date) DO NOTHING
            """,
            (wallet, today, score_data["win_rate"],
             score_data["avg_roi"], score_data["trade_count"]),
        )

        # Mark all discovery_queue entries for this wallet as evaluated
        conn.execute(
            """
            UPDATE discovery_queue
            SET evaluated = 1, evaluation_result = 'promoted'
            WHERE wallet_address = ? AND chain = 'solana'
            """,
            (wallet,),
        )

        conn.commit()
        conn.close()

        promoted += 1
        log.info("    PROMOTED %s  trades=%d  win=%.0f%%  roi=%+.1f%%  score=%.2f",
                 wallet[:12], score_data["trade_count"],
                 score_data["win_rate"] * 100, score_data["avg_roi"],
                 score_data["score"])
        time.sleep(CALL_DELAY)

    # Mark single-winner wallets older than 21d as dropped
    drop_cutoff = now - 21 * 86400
    conn = get_conn()
    conn.execute(
        """
        UPDATE discovery_queue
        SET evaluated = 1, evaluation_result = 'expired'
        WHERE evaluated = 0 AND discovered_at < ?
        """,
        (drop_cutoff,),
    )
    conn.commit()
    conn.close()

    log.info("Phase 2d complete — %d wallets promoted to active pool", promoted)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("backtrace", "all"):
        run_backtrace()
    if cmd in ("promote", "all"):
        run_promote()
