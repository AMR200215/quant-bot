"""
Phase 2a — Helius ingestion for tiered wallets.

Pulls last 30d of swap activity for all S/A/B tier wallets,
parses into wallet_trades, recomputes scores, demotes inactive wallets.

Run:
    python3 -m wallet_db.ingest
    python3 -m wallet_db.ingest --days 30
    python3 -m wallet_db.ingest --wallet <address>   (single wallet test)
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from wallet_db.db import get_conn, init_db

load_dotenv()

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

HELIUS_KEY  = os.getenv("HELIUS_API_KEY", "")
HELIUS_BASE = "https://api.helius.xyz/v0"
WSOL        = "So11111111111111111111111111111111111111112"
CALL_DELAY  = 0.25   # 4 req/s — well under free tier

# ---------------------------------------------------------------------------
# Helius fetch + parse (same logic as rank_wallets_helius.py)
# ---------------------------------------------------------------------------

def _fetch_swaps(wallet: str, since_ts: int, limit: int = 100) -> list[dict]:
    """Fetch swap transactions for a wallet since a unix timestamp."""
    url    = f"{HELIUS_BASE}/addresses/{wallet}/transactions"
    params = {"api-key": HELIUS_KEY, "limit": min(limit, 100), "type": "SWAP"}
    all_txs = []
    before  = None

    while True:
        if before:
            params["before"] = before
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 401:
                raise SystemExit("Invalid HELIUS_API_KEY")
            if r.status_code == 429:
                log.warning("Helius rate limit — sleeping 5s")
                time.sleep(5)
                continue
            if r.status_code != 200:
                break
            batch = r.json()
            if not batch:
                break

            for tx in batch:
                tx_ts = tx.get("timestamp", 0)
                if tx_ts < since_ts:
                    return all_txs   # older than window — stop paginating
                all_txs.append(tx)

            if len(batch) < 100:
                break
            before = batch[-1]["signature"]
            time.sleep(CALL_DELAY)
        except SystemExit:
            raise
        except Exception as e:
            log.warning("Helius fetch error: %s", e)
            break

    return all_txs


def _parse_swap(tx: dict, wallet: str) -> dict | None:
    """Parse a Helius SWAP tx into a trade dict."""
    if tx.get("transactionError"):
        return None

    native = tx.get("nativeTransfers", [])
    tokens = tx.get("tokenTransfers", [])

    sol_out = sum(t["amount"] for t in native if t.get("fromUserAccount") == wallet) / 1e9
    sol_in  = sum(t["amount"] for t in native if t.get("toUserAccount")   == wallet) / 1e9

    tok_in  = [t for t in tokens if t.get("toUserAccount")   == wallet and t.get("mint") != WSOL]
    tok_out = [t for t in tokens if t.get("fromUserAccount") == wallet and t.get("mint") != WSOL]

    sol_net = sol_in - sol_out

    if tok_in and sol_net < -0.001:
        main = max(tok_in, key=lambda t: float(t.get("tokenAmount") or 0))
        return {
            "side":          "buy",
            "mint":          main["mint"],
            "native_amount": abs(sol_net),
            "token_amount":  float(main.get("tokenAmount") or 0),
            "tx_hash":       tx.get("signature", ""),
            "block_time":    int(tx.get("timestamp", 0)),
        }

    if tok_out and sol_net > 0.001:
        main = max(tok_out, key=lambda t: float(t.get("tokenAmount") or 0))
        return {
            "side":          "sell",
            "mint":          main["mint"],
            "native_amount": sol_net,
            "token_amount":  float(main.get("tokenAmount") or 0),
            "tx_hash":       tx.get("signature", ""),
            "block_time":    int(tx.get("timestamp", 0)),
        }

    return None


# ---------------------------------------------------------------------------
# Ingest one wallet
# ---------------------------------------------------------------------------

def ingest_wallet(wallet: str, chain: str, since_ts: int) -> dict:
    """
    Pull and store last N days of swaps for one wallet.
    Returns summary: {inserted, skipped_dup, trades_found}
    """
    txs = _fetch_swaps(wallet, since_ts)
    if not txs:
        return {"inserted": 0, "skipped_dup": 0, "trades_found": 0}

    trades   = [t for t in (_parse_swap(tx, wallet) for tx in txs) if t]
    inserted = skipped = 0

    conn = get_conn()
    for t in trades:
        try:
            conn.execute(
                """
                INSERT INTO wallet_trades
                    (wallet_address, chain, token_address, side,
                     token_amount, native_amount, block_time, tx_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (wallet, chain, t["mint"], t["side"],
                 t["token_amount"], t["native_amount"],
                 t["block_time"], t["tx_hash"]),
            )
            inserted += 1
        except Exception:
            skipped += 1   # UNIQUE tx_hash violation = already stored

    # Update last_trade_ts on wallet
    if trades:
        newest = max(t["block_time"] for t in trades)
        conn.execute(
            "UPDATE wallets SET last_trade_ts = MAX(COALESCE(last_trade_ts,0), ?) WHERE address = ? AND chain = ?",
            (newest, wallet, chain),
        )

    conn.commit()
    conn.close()

    return {"inserted": inserted, "skipped_dup": skipped, "trades_found": len(trades)}


# ---------------------------------------------------------------------------
# Recompute score from wallet_trades (30d window)
# ---------------------------------------------------------------------------

def _compute_score(wallet: str, chain: str, window_days: int = 30) -> dict:
    """
    Simple scoring from wallet_trades.
    Returns score dict including tier recommendation.
    """
    since = int(time.time()) - window_days * 86400
    conn  = get_conn()

    buys  = conn.execute(
        "SELECT * FROM wallet_trades WHERE wallet_address=? AND chain=? AND side='buy'  AND block_time>=?",
        (wallet, chain, since),
    ).fetchall()
    sells = conn.execute(
        "SELECT * FROM wallet_trades WHERE wallet_address=? AND chain=? AND side='sell' AND block_time>=?",
        (wallet, chain, since),
    ).fetchall()
    conn.close()

    trade_count = len(buys) + len(sells)
    if trade_count < 3:
        return {"score": 0.0, "trade_count": trade_count, "win_rate": 0.0,
                "avg_roi": 0.0, "low_confidence": True}

    # Match buys → sells per token
    from collections import defaultdict
    open_buys: dict[str, list] = defaultdict(list)
    closed: list[dict] = []

    all_trades = sorted(
        [{"side": r["side"], "mint": r["token_address"], "sol": r["native_amount"],
          "ts": r["block_time"]} for r in buys + sells],
        key=lambda x: x["ts"],
    )

    for t in all_trades:
        if t["side"] == "buy":
            open_buys[t["mint"]].append(t["sol"])
        elif t["side"] == "sell" and t["mint"] in open_buys:
            costs = open_buys.pop(t["mint"])
            cost  = sum(costs)
            if cost > 0.001:
                roi = (t["sol"] - cost) / cost * 100
                closed.append({"roi": roi, "win": roi > 0})

    if not closed:
        return {"score": 0.0, "trade_count": trade_count, "win_rate": 0.0,
                "avg_roi": 0.0, "low_confidence": True}

    win_rate = sum(1 for c in closed if c["win"]) / len(closed)
    avg_roi  = sum(c["roi"] for c in closed) / len(closed)
    score    = round(win_rate * max(avg_roi, 0), 4)
    low_conf = len(closed) < 10

    return {
        "score":          score,
        "trade_count":    len(closed),
        "win_rate":       round(win_rate, 4),
        "avg_roi":        round(avg_roi, 2),
        "low_confidence": low_conf,
    }


# ---------------------------------------------------------------------------
# Update tier in DB
# ---------------------------------------------------------------------------

def _update_tier(wallet: str, chain: str, score: float,
                 trade_count: int, active: bool, today: str):
    conn = get_conn()

    if not active:
        conn.execute(
            "UPDATE wallets SET status='dormant', current_tier=NULL WHERE address=? AND chain=?",
            (wallet, chain),
        )
        conn.execute(
            """INSERT OR IGNORE INTO wallet_scores_history
               (wallet_address, chain, score_date, tier, trade_count)
               VALUES (?, ?, ?, 'dormant', ?)""",
            (wallet, chain, today, trade_count),
        )
        conn.commit()
        conn.close()
        return

    conn.execute(
        "UPDATE wallets SET current_score=? WHERE address=? AND chain=?",
        (score, wallet, chain),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Main 2a job
# ---------------------------------------------------------------------------

def run_phase2a(days: int = 30, single_wallet: str = ""):
    if not HELIUS_KEY:
        raise SystemExit("HELIUS_API_KEY not set in .env")

    init_db()
    since_ts = int(time.time()) - days * 86400
    today    = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    conn = get_conn()
    if single_wallet:
        rows = conn.execute(
            "SELECT address, chain, name FROM wallets WHERE address=?",
            (single_wallet,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT address, chain, name FROM wallets WHERE current_tier IS NOT NULL"
        ).fetchall()
    conn.close()

    wallets = [dict(r) for r in rows]
    log.info("Phase 2a — ingesting %d wallets (%d day window)", len(wallets), days)

    results = []
    for w in wallets:
        addr  = w["address"]
        chain = w["chain"]
        name  = w.get("name", addr[:8])

        if chain != "solana":
            log.info("  SKIP %s (%s) — BSC ingestion not yet built", name, chain)
            continue

        log.info("  Ingesting %s (%s)…", name, addr[:12])
        summary = ingest_wallet(addr, chain, since_ts)
        time.sleep(CALL_DELAY)

        score_data = _compute_score(addr, chain, days)
        active     = summary["trades_found"] > 0 or score_data["trade_count"] > 0

        _update_tier(addr, chain, score_data["score"],
                     score_data["trade_count"], active, today)

        results.append({
            "name":       name,
            "address":    addr,
            "inserted":   summary["inserted"],
            "trades_30d": score_data["trade_count"],
            "win_rate":   score_data["win_rate"],
            "avg_roi":    score_data["avg_roi"],
            "score":      score_data["score"],
            "active":     active,
            "low_conf":   score_data["low_confidence"],
        })

        status = "ACTIVE" if active else "DORMANT→demoted"
        log.info("    %s  trades=%d  win=%.0f%%  roi=%+.1f%%  score=%.2f  [%s]",
                 status, score_data["trade_count"],
                 score_data["win_rate"] * 100, score_data["avg_roi"],
                 score_data["score"],
                 "LOW CONF" if score_data["low_confidence"] else "ok")

    # Rerank remaining active wallets and assign S/A/B
    active_results = sorted(
        [r for r in results if r["active"] and r["trades_30d"] >= 3],
        key=lambda x: x["score"], reverse=True,
    )
    total = len(active_results)

    conn = get_conn()
    for i, r in enumerate(active_results):
        pct = (i + 1) / total if total else 1
        if pct <= 0.10 and r["trades_30d"] >= 10:
            tier = "S"
        elif pct <= 0.30 and r["trades_30d"] >= 5:
            tier = "A"
        elif pct <= 0.60 and r["trades_30d"] >= 3:
            tier = "B"
        else:
            tier = None
        conn.execute(
            "UPDATE wallets SET current_tier=? WHERE address=? AND chain=?",
            (tier, r["address"], "solana"),
        )
        conn.execute(
            """INSERT OR REPLACE INTO wallet_scores_history
               (wallet_address, chain, score_date, tier, hit_rate, median_return, trade_count)
               VALUES (?, 'solana', ?, ?, ?, ?, ?)""",
            (r["address"], today, tier, r["win_rate"], r["avg_roi"], r["trades_30d"]),
        )
    conn.commit()
    conn.close()

    # Print summary table
    print(f"\n{'='*70}")
    print(f"Phase 2a complete — {days}d window — {len(results)} wallets processed")
    print(f"{'='*70}")
    print(f"{'Name':<22} {'Trades':>6} {'Win%':>6} {'AvgROI':>8} {'Score':>8} {'Status'}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        status = "DORMANT" if not r["active"] else ("LOW-CONF" if r["low_conf"] else "ok")
        print(f"{r['name']:<22} {r['trades_30d']:>6} "
              f"{r['win_rate']*100:>5.0f}%  {r['avg_roi']:>+7.1f}%  "
              f"{r['score']:>8.2f}  {status}")

    demoted = sum(1 for r in results if not r["active"])
    print(f"\nDemoted to dormant: {demoted}")
    print(f"Active with data:   {len(results) - demoted}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",   type=int, default=30)
    parser.add_argument("--wallet", type=str, default="")
    args = parser.parse_args()
    run_phase2a(args.days, args.wallet)
