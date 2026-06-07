"""
Phase 4 — Wallet scoring engine.

Computes a full 0-100 composite score per wallet using a 30d window of
wallet_trades, with hit_rate, median_return, winner_rate, sell_discipline,
and a confidence-weighted composite. Writes results to wallets and
wallet_scores_history.

Run:
    python3 -m wallet_db.score
    python3 -m wallet_db.score --wallet <address>   (single-wallet test mode)
"""

import argparse
import logging
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone

from wallet_db.db import get_conn, init_db, _USE_POSTGRES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_PH = "%s" if _USE_POSTGRES else "?"

WINDOW_DAYS      = 30
FULL_CONF_TRADES = 15    # trade_count >= 15 → confidence = 1.0
LOW_CONF_THRESH  = 5     # trade_count < 5   → flag low_confidence
DORMANT_THRESH   = 3     # score==0 and trade_count < 3 → mark dormant
WINNER_MULTIPLE  = 5.0   # peak_multiple threshold for winner_rate
MEDIAN_SCALE     = 200.0 # median_return at which median_return_norm = 1.0


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _fifo_pairs(buys: list[dict], sells: list[dict]) -> list[dict]:
    """
    Match buys → sells per token using FIFO chronologically.
    Each sell matches the oldest unmatched buy for that token.
    Only closed pairs are returned.

    Returns list of dicts:
        token_address, cost_sol, proceeds_sol, roi_pct, sell_ts
    """
    buy_queues: dict[str, list[dict]] = defaultdict(list)
    for b in sorted(buys, key=lambda x: x["block_time"]):
        buy_queues[b["token_address"]].append(b)

    pairs = []
    for s in sorted(sells, key=lambda x: x["block_time"]):
        token = s["token_address"]
        queue = buy_queues.get(token)
        if not queue:
            continue
        buy = queue.pop(0)
        if not queue:
            buy_queues.pop(token, None)

        cost = buy["native_amount"]
        if cost < 0.0001:
            continue  # dust — skip
        proceeds = s["native_amount"]
        roi_pct  = (proceeds - cost) / cost * 100.0
        pairs.append({
            "token_address": token,
            "cost_sol":      cost,
            "proceeds_sol":  proceeds,
            "roi_pct":       roi_pct,
            "sell_ts":       s["block_time"],
        })
    return pairs


def compute_wallet_score(wallet: str, chain: str, window_days: int = 30) -> dict:
    """
    Full Phase 4 scoring for one wallet over the given window.

    Return dict keys (superset of ingest._compute_score for drop-in compat):
        score, hit_rate, median_return, avg_return, sell_discipline,
        winner_rate, trade_count, recent_7d_hit_rate,
        win_rate (alias), avg_roi (alias), low_confidence
    """
    now    = int(time.time())
    since  = now - window_days * 86400
    since7 = now - 7 * 86400

    conn  = get_conn()
    buys  = conn.execute(
        f"SELECT token_address, native_amount, block_time FROM wallet_trades"
        f" WHERE wallet_address={_PH} AND chain={_PH} AND side='buy' AND block_time>={_PH}",
        (wallet, chain, since),
    ).fetchall()
    sells = conn.execute(
        f"SELECT token_address, native_amount, block_time FROM wallet_trades"
        f" WHERE wallet_address={_PH} AND chain={_PH} AND side='sell' AND block_time>={_PH}",
        (wallet, chain, since),
    ).fetchall()
    conn.close()

    buys_d  = [dict(r) for r in buys]
    sells_d = [dict(r) for r in sells]

    _zero = {
        "score": 0.0, "hit_rate": 0.0, "median_return": 0.0,
        "avg_return": 0.0, "sell_discipline": 0.0, "winner_rate": 0.0,
        "trade_count": 0, "recent_7d_hit_rate": 0.0,
        "win_rate": 0.0, "avg_roi": 0.0, "low_confidence": True,
    }

    if not buys_d:
        return _zero

    pairs       = _fifo_pairs(buys_d, sells_d)
    trade_count = len(pairs)

    if trade_count == 0:
        return _zero

    # Core return metrics
    rois       = [p["roi_pct"] for p in pairs]
    wins       = [r for r in rois if r > 0]
    hit_rate   = len(wins) / trade_count
    median_ret = statistics.median(rois)
    avg_ret    = sum(rois) / trade_count

    # Recent 7d hit rate
    pairs_7d = [p for p in pairs if p["sell_ts"] >= since7]
    if pairs_7d:
        recent_7d = sum(1 for p in pairs_7d if p["roi_pct"] > 0) / len(pairs_7d)
    else:
        recent_7d = 0.0

    # Sell discipline: distinct tokens bought that also had a sell
    bought_tokens   = {b["token_address"] for b in buys_d}
    sold_tokens     = {s["token_address"] for s in sells_d}
    sell_discipline = len(bought_tokens & sold_tokens) / len(bought_tokens)

    # Winner rate: bought tokens that hit 5x+ in token_outcomes
    winner_rate = 0.0
    if bought_tokens:
        conn     = get_conn()
        ph_list  = ",".join([_PH] * len(bought_tokens))
        winners  = conn.execute(
            f"SELECT token_address FROM token_outcomes"
            f" WHERE chain={_PH} AND peak_multiple>={_PH} AND token_address IN ({ph_list})",
            (chain, WINNER_MULTIPLE, *bought_tokens),
        ).fetchall()
        conn.close()
        winner_rate = len(winners) / len(bought_tokens)

    # Composite score (0–100)
    median_norm = min(1.0, max(0.0, median_ret / MEDIAN_SCALE))
    raw = (
        hit_rate        * 35.0 +
        median_norm     * 35.0 +
        winner_rate     * 20.0 +
        sell_discipline * 10.0
    )
    confidence = min(1.0, trade_count / FULL_CONF_TRADES)
    score      = round(raw * confidence, 2)

    return {
        "score":              score,
        "hit_rate":           round(hit_rate,         4),
        "median_return":      round(median_ret,        4),
        "avg_return":         round(avg_ret,           4),
        "sell_discipline":    round(sell_discipline,   4),
        "winner_rate":        round(winner_rate,       4),
        "trade_count":        trade_count,
        "recent_7d_hit_rate": round(recent_7d,         4),
        # backward-compat aliases
        "win_rate":           round(hit_rate,  4),
        "avg_roi":            round(avg_ret,   4),
        "low_confidence":     trade_count < LOW_CONF_THRESH,
    }


# ---------------------------------------------------------------------------
# DB write
# ---------------------------------------------------------------------------

def _write_score(wallet: str, chain: str, sd: dict, today: str) -> None:
    conn = get_conn()

    # Always update current_score — clears any stale values from old scoring formulas
    conn.execute(
        f"UPDATE wallets SET current_score={_PH} WHERE address={_PH} AND chain={_PH}",
        (sd["score"], wallet, chain),
    )
    if sd["score"] == 0.0 and sd["trade_count"] < DORMANT_THRESH:
        conn.execute(
            f"UPDATE wallets SET status='dormant' WHERE address={_PH} AND chain={_PH}",
            (wallet, chain),
        )

    conn.execute(
        f"""
        INSERT INTO wallet_scores_history
            (wallet_address, chain, score_date, tier, hit_rate, median_return, trade_count)
        VALUES ({_PH}, {_PH}, {_PH}, NULL, {_PH}, {_PH}, {_PH})
        ON CONFLICT (wallet_address, chain, score_date) DO UPDATE SET
            hit_rate      = excluded.hit_rate,
            median_return = excluded.median_return,
            trade_count   = excluded.trade_count
        """,
        (wallet, chain, today,
         sd["hit_rate"], sd["median_return"], sd["trade_count"]),
    )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Batch job
# ---------------------------------------------------------------------------

def run_scoring(single_wallet: str = "", chain: str = "solana") -> None:
    init_db()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    conn = get_conn()
    if single_wallet:
        rows = conn.execute(
            f"SELECT address, chain, name FROM wallets WHERE address={_PH} AND chain={_PH}",
            (single_wallet, chain),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT address, chain, name FROM wallets WHERE status='active' AND chain={_PH}",
            (chain,),
        ).fetchall()
    conn.close()

    wallets = [dict(r) for r in rows]
    log.info("Phase 4 scoring — %d wallets (chain=%s, window=%dd)",
             len(wallets), chain, WINDOW_DAYS)

    if not wallets:
        log.warning("No active wallets found.")
        return

    results = []
    for w in wallets:
        addr   = w["address"]
        wchain = w.get("chain") or chain
        name   = w.get("name") or addr[:10]

        log.info("  Scoring %s (%s)…", name, addr[:12])
        sd = compute_wallet_score(addr, wchain, WINDOW_DAYS)
        _write_score(addr, wchain, sd, today)

        results.append({**sd, "name": name, "address": addr})

        log.info(
            "    score=%.2f  hit=%.0f%%  med=%+.1f%%  winner=%.0f%%"
            "  sell_disc=%.0f%%  trades=%d  7d=%.0f%%  [%s]",
            sd["score"],
            sd["hit_rate"]        * 100,
            sd["median_return"],
            sd["winner_rate"]     * 100,
            sd["sell_discipline"] * 100,
            sd["trade_count"],
            sd["recent_7d_hit_rate"] * 100,
            "LOW-CONF" if sd["low_confidence"] else "ok",
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    W = 82
    print(f"\n{'='*W}")
    print(f"Phase 4 Scoring — {today} — {len(results)} wallets")
    print(f"{'='*W}")
    print(f"{'Name':<22} {'Score':>6} {'Hit%':>5} {'MedROI':>8} {'Win%':>5}"
          f" {'SD%':>5} {'Tr':>4} {'7d%':>5}  Status")
    print("-" * W)
    for r in results:
        dormant = r["score"] == 0.0 and r["trade_count"] < DORMANT_THRESH
        status  = "DORMANT" if dormant else ("LOW-CONF" if r["low_confidence"] else "ok")
        print(
            f"{r['name']:<22} {r['score']:>6.2f} "
            f"{r['hit_rate']*100:>4.0f}% "
            f"{r['median_return']:>+7.1f}% "
            f"{r['winner_rate']*100:>4.0f}% "
            f"{r['sell_discipline']*100:>4.0f}% "
            f"{r['trade_count']:>4d} "
            f"{r['recent_7d_hit_rate']*100:>4.0f}%  "
            f"{status}"
        )

    dormant_ct  = sum(1 for r in results if r["score"] == 0.0 and r["trade_count"] < DORMANT_THRESH)
    low_conf_ct = sum(1 for r in results if r["low_confidence"])
    print(f"\nActive: {len(results)-dormant_ct}  Low-conf: {low_conf_ct}  Dormant: {dormant_ct}")
    print(f"{'='*W}\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 — Wallet scoring engine")
    parser.add_argument("--wallet", type=str, default="",
                        help="Score a single wallet (test mode)")
    parser.add_argument("--chain",  type=str, default="solana")
    args = parser.parse_args()
    run_scoring(single_wallet=args.wallet, chain=args.chain)
