"""
Whale wallet ranker — uses Helius enhanced transaction API.

Helius free tier: 1M credits/month (enough for 300 wallets with room to spare).
Helius parses swaps natively — no manual SOL delta math, much faster than raw RPC.

Setup:
  1. Sign up free at helius.dev
  2. Get API key from dashboard
  3. Add to .env:  HELIUS_API_KEY=your_key_here

Run:
  python3 scripts/rank_wallets_helius.py
  python3 scripts/rank_wallets_helius.py --resume
  python3 scripts/rank_wallets_helius.py --limit 100   (fewer txs, faster test)
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
ROOT     = Path(__file__).parent.parent
WALLETS  = ROOT / "memecoin" / "data" / "whale_wallets_sol.json"
OUT_FILE = ROOT / "memecoin" / "data" / "whale_stats.json"

HELIUS_KEY   = os.getenv("HELIUS_API_KEY", "")
HELIUS_BASE  = f"https://api.helius.xyz/v0"

DEFAULT_TX_LIMIT  = 100   # per wallet — Helius is fast, 100 is plenty for signal
CALL_DELAY        = 0.2   # 5 req/sec — well under free tier limits
MIN_TRADES        = 3
WSOL              = "So11111111111111111111111111111111111111112"
MAX_INACTIVE_DAYS = 45    # skip wallets with no trades in last 45 days (dead wallets)


# ---------------------------------------------------------------------------
def _most_recent_swap_ts(wallet: str) -> float:
    """Return unix timestamp of the most recent SWAP for this wallet, or 0."""
    url = f"{HELIUS_BASE}/addresses/{wallet}/transactions"
    params = {"api-key": HELIUS_KEY, "limit": 1, "type": "SWAP"}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return 0
        txs = r.json()
        if txs:
            return float(txs[0].get("timestamp", 0))
    except Exception:
        pass
    return 0


def get_parsed_transactions(wallet: str, limit: int = 100) -> list[dict]:
    """
    Fetch last N transactions for a wallet using Helius enhanced API.
    Returns parsed transaction objects — swaps are pre-identified with
    tokenTransfers and nativeTransfers broken out cleanly.
    """
    url = f"{HELIUS_BASE}/addresses/{wallet}/transactions"
    params = {
        "api-key": HELIUS_KEY,
        "limit":   min(limit, 100),   # Helius max per page is 100
        "type":    "SWAP",            # only swap transactions
    }

    all_txs = []
    before = None

    while len(all_txs) < limit:
        if before:
            params["before"] = before

        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 401:
                print("\nERROR: Invalid HELIUS_API_KEY — check your .env file")
                raise SystemExit(1)
            if r.status_code == 429:
                print("  rate-limited, sleeping 5s…")
                time.sleep(5)
                continue
            if r.status_code != 200:
                break

            batch = r.json()
            if not batch:
                break

            all_txs.extend(batch)

            if len(batch) < 100:
                break   # last page

            before = batch[-1]["signature"]
            time.sleep(CALL_DELAY)

        except Exception:
            break

    return all_txs[:limit]


def parse_helius_swap(tx: dict, wallet: str) -> dict | None:
    """
    Parse a Helius-enriched SWAP transaction.

    Helius gives us tokenTransfers and nativeTransfers already extracted.
    A swap for our wallet looks like:
      - SOL out  (nativeTransfer from wallet) + token in  (tokenTransfer to wallet)  = BUY
      - token out (tokenTransfer from wallet) + SOL in   (nativeTransfer to wallet)  = SELL
    """
    if tx.get("transactionError"):
        return None

    native   = tx.get("nativeTransfers",  [])
    tokens   = tx.get("tokenTransfers",   [])

    # SOL flow for our wallet
    sol_out = sum(t["amount"] for t in native
                  if t.get("fromUserAccount") == wallet) / 1e9
    sol_in  = sum(t["amount"] for t in native
                  if t.get("toUserAccount") == wallet) / 1e9

    # Token flow for our wallet (exclude WSOL)
    tok_in  = [t for t in tokens
               if t.get("toUserAccount") == wallet
               and t.get("mint") != WSOL]
    tok_out = [t for t in tokens
               if t.get("fromUserAccount") == wallet
               and t.get("mint") != WSOL]

    sol_net = sol_in - sol_out   # positive = received SOL, negative = spent SOL

    if tok_in and sol_net < -0.001:
        # Spent SOL, got tokens = BUY
        # Pick the token with highest value (usually only one per swap)
        main = max(tok_in, key=lambda t: float(t.get("tokenAmount") or 0))
        return {"type": "buy", "mint": main["mint"], "sol": abs(sol_net)}

    if tok_out and sol_net > 0.001:
        # Got SOL, sent tokens = SELL
        main = max(tok_out, key=lambda t: float(t.get("tokenAmount") or 0))
        return {"type": "sell", "mint": main["mint"], "sol": sol_net}

    return None


# ---------------------------------------------------------------------------
def analyse_wallet(wallet: str, tx_limit: int) -> dict:
    # Fast activity check — skip dead wallets before fetching full history
    import time as _time
    last_ts = _most_recent_swap_ts(wallet)
    time.sleep(CALL_DELAY)
    if last_ts > 0:
        days_inactive = (_time.time() - last_ts) / 86400
        if days_inactive > MAX_INACTIVE_DAYS:
            return {"wallet": wallet, "trades": 0,
                    "inactive_days": round(days_inactive), "skip": "inactive"}

    txs = get_parsed_transactions(wallet, tx_limit)
    if not txs:
        return {"wallet": wallet, "trades": 0}

    open_trades: dict[str, list[float]] = defaultdict(list)
    closed_trades: list[dict] = []

    for tx in txs:
        swap = parse_helius_swap(tx, wallet)
        if not swap:
            continue

        mint = swap["mint"]
        sol  = swap["sol"]

        if swap["type"] == "buy":
            open_trades[mint].append(sol)

        elif swap["type"] == "sell" and mint in open_trades:
            costs = open_trades.pop(mint)
            total_cost = sum(costs)
            if total_cost > 0.001:
                pnl_pct = (sol - total_cost) / total_cost * 100
                closed_trades.append({
                    "mint":     mint,
                    "cost_sol": round(total_cost, 6),
                    "recv_sol": round(sol, 6),
                    "pnl_pct":  round(pnl_pct, 2),
                    "win":      pnl_pct > 0,
                })

    all_trades = closed_trades
    # Note: skipping unrealized for speed — closed trades are enough to rank

    # Date range from transaction timestamps
    timestamps = [t.get("timestamp", 0) for t in txs if t.get("timestamp")]
    oldest_ts = min(timestamps) if timestamps else 0
    newest_ts = max(timestamps) if timestamps else 0

    if not all_trades:
        return {"wallet": wallet, "trades": 0}

    wins     = [t for t in all_trades if t["win"]]
    losses   = [t for t in all_trades if not t["win"]]
    win_rate = len(wins) / len(all_trades)
    avg_roi  = sum(t["pnl_pct"] for t in all_trades) / len(all_trades)
    avg_win  = sum(t["pnl_pct"] for t in wins)   / len(wins)   if wins   else 0
    avg_loss = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0
    score    = win_rate * max(avg_roi, 0)

    return {
        "wallet":          wallet,
        "trades":          len(all_trades),
        "closed_trades_n": len(closed_trades),
        "win_rate":        round(win_rate, 4),
        "avg_roi":         round(avg_roi, 2),
        "avg_win":         round(avg_win, 2),
        "avg_loss":        round(avg_loss, 2),
        "score":           round(score, 4),
        "closed_trades":   closed_trades,
        "oldest_trade":    oldest_ts,
        "newest_trade":    newest_ts,
        "source":          "helius",
    }


# ---------------------------------------------------------------------------
def main():
    if not HELIUS_KEY:
        print("ERROR: HELIUS_API_KEY not set in .env\n"
              "  1. Sign up free at helius.dev\n"
              "  2. Add HELIUS_API_KEY=your_key to .env")
        raise SystemExit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",  type=int, default=DEFAULT_TX_LIMIT,
                        help=f"Max swap txs per wallet (default {DEFAULT_TX_LIMIT})")
    parser.add_argument("--resume", action="store_true",
                        help="Skip wallets already in output file")
    args = parser.parse_args()

    wallets = json.loads(WALLETS.read_text())
    print(f"Loaded {len(wallets)} wallets from {WALLETS.name}")
    print(f"Fetching last {args.limit} SWAP txs per wallet via Helius\n")

    existing: dict[str, dict] = {}
    if args.resume and OUT_FILE.exists():
        for entry in json.loads(OUT_FILE.read_text()):
            existing[entry["wallet"]] = entry
        print(f"Resuming — {len(existing)} wallets already done\n")

    results: list[dict] = list(existing.values())
    total = len(wallets)

    for i, w in enumerate(wallets):
        addr = w.get("trackedWalletAddress") or w.get("address") or ""
        name = w.get("name", addr[:8])
        if not addr:
            continue
        if args.resume and addr in existing:
            continue

        print(f"[{i+1:>3}/{total}] {name:<20} {addr[:12]}…", end="  ", flush=True)

        try:
            stats = analyse_wallet(addr, args.limit)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        stats["name"] = name
        results.append(stats)

        trades = stats.get("trades", 0)
        if stats.get("skip") == "inactive":
            print(f"SKIP — inactive {stats.get('inactive_days', '?')}d")
        elif trades >= MIN_TRADES:
            import datetime as _dt
            newest = stats.get("newest_trade", 0)
            date_str = _dt.datetime.fromtimestamp(newest).strftime("%Y-%m-%d") if newest else "?"
            wr  = stats.get("win_rate", 0) * 100
            roi = stats.get("avg_roi", 0)
            print(f"{trades} trades  {wr:.0f}% win  avg_roi {roi:+.1f}%  last={date_str}")
        else:
            print(f"{trades} closed trades (need {MIN_TRADES} to rank)")

        time.sleep(CALL_DELAY)

        # Save after every wallet
        ranked = _rank(results)
        OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUT_FILE.write_text(json.dumps(ranked, indent=2))

    ranked = _rank(results)
    OUT_FILE.write_text(json.dumps(ranked, indent=2))

    print(f"\n{'='*65}")
    print(f"Done. {len(ranked)} wallets ranked (≥{MIN_TRADES} trades each)")
    print(f"Saved → {OUT_FILE}\n")
    print(f"{'Rank':<5} {'Name':<20} {'Trades':>6} {'Win%':>6} {'AvgROI':>8} {'Score':>7}")
    print("-" * 55)
    for rank, r in enumerate(ranked[:20], 1):
        print(f"{rank:<5} {r.get('name','?'):<20} {r['trades']:>6} "
              f"{r['win_rate']*100:>5.1f}%  {r['avg_roi']:>+7.1f}%  {r['score']:>7.4f}")


def _rank(results: list[dict]) -> list[dict]:
    return sorted(
        [r for r in results if r.get("trades", 0) >= MIN_TRADES],
        key=lambda x: x.get("score", 0),
        reverse=True,
    )


if __name__ == "__main__":
    main()
