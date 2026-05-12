"""
Whale wallet ranker — pulls on-chain swap history from Solana RPC
and ranks all 300 wallets by actual trading performance.

How it works:
  1. getSignaturesForAddress  — fetch last N tx signatures per wallet
  2. getTransaction           — parse each tx for swap data
  3. Match buys → sells per token (SOL in/out as cost/proceeds)
  4. Calculate win_rate, avg_roi, trade_count per wallet
  5. Write ranked results to memecoin/data/whale_stats.json

Run:  python3 scripts/rank_wallets.py
      python3 scripts/rank_wallets.py --limit 100   (fewer txs, faster)
      python3 scripts/rank_wallets.py --resume       (skip already-done wallets)
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
ROOT     = Path(__file__).parent.parent
WALLETS  = ROOT / "memecoin" / "data" / "whale_wallets_sol.json"
OUT_FILE = ROOT / "memecoin" / "data" / "whale_stats.json"

RPC_URL  = "https://api.mainnet-beta.solana.com"
WSOL     = "So11111111111111111111111111111111111111112"

# How many txs to analyse per wallet (more = slower but better data)
DEFAULT_TX_LIMIT = 500
# Seconds between RPC calls (stay well under rate limit)
CALL_DELAY = 0.15
# Min trades needed to include wallet in ranking
MIN_TRADES = 3


# ---------------------------------------------------------------------------
def rpc(method: str, params: list, retries: int = 3) -> dict | None:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    for attempt in range(retries):
        try:
            r = requests.post(RPC_URL, json=payload, timeout=15)
            r.raise_for_status()
            result = r.json()
            if "error" in result:
                return None
            return result.get("result")
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                return None


# ---------------------------------------------------------------------------
def get_signatures(wallet: str, limit: int) -> list[str]:
    result = rpc("getSignaturesForAddress", [wallet, {"limit": limit}])
    if not result:
        return []
    # skip failed transactions
    return [s["signature"] for s in result if not s.get("err")]


def parse_swap(tx: dict, wallet: str) -> dict | None:
    """
    Parse a transaction for swap activity relative to a wallet.
    Returns {"type": "buy"|"sell", "mint": str, "sol": float} or None.

    SOL accounting:
      buy  → wallet SOL decreases, wallet token balance increases
      sell → wallet SOL increases, wallet token balance decreases (or account closes)
    """
    meta = tx.get("meta", {})
    if meta.get("err"):
        return None

    accounts = tx.get("transaction", {}).get("message", {}).get("accountKeys", [])
    wallet_idx = None
    for i, a in enumerate(accounts):
        pubkey = a.get("pubkey") if isinstance(a, dict) else a
        if pubkey == wallet:
            wallet_idx = i
            break
    if wallet_idx is None:
        return None

    pre_sol  = meta.get("preBalances",  [])
    post_sol = meta.get("postBalances", [])
    if wallet_idx >= len(pre_sol) or wallet_idx >= len(post_sol):
        return None

    # Net SOL change for wallet (lamports → SOL, accounts for fees)
    sol_delta = (post_sol[wallet_idx] - pre_sol[wallet_idx]) / 1e9

    # Build token balance maps (pre and post) for this wallet
    pre_tok: dict[str, float] = {}
    for t in meta.get("preTokenBalances", []):
        if t.get("owner") == wallet and t.get("mint") != WSOL:
            pre_tok[t["mint"]] = float(t["uiTokenAmount"].get("uiAmount") or 0)

    post_tok: dict[str, float] = {}
    for t in meta.get("postTokenBalances", []):
        if t.get("owner") == wallet and t.get("mint") != WSOL:
            post_tok[t["mint"]] = float(t["uiTokenAmount"].get("uiAmount") or 0)

    # All mints that appeared in either snapshot
    all_mints = set(pre_tok) | set(post_tok)
    if not all_mints:
        return None

    # Find the mint with the biggest absolute change
    changes = {m: post_tok.get(m, 0) - pre_tok.get(m, 0) for m in all_mints}
    if not changes:
        return None

    main_mint   = max(changes, key=lambda m: abs(changes[m]))
    token_delta = changes[main_mint]

    if token_delta == 0:
        return None

    # classify
    if sol_delta < -0.001 and token_delta > 0:
        # spent SOL, received tokens = BUY
        return {"type": "buy",  "mint": main_mint, "sol": abs(sol_delta)}

    if sol_delta > 0.001 and token_delta < 0:
        # received SOL, sent tokens = SELL
        return {"type": "sell", "mint": main_mint, "sol": sol_delta}

    # Closed token account: account vanishes from postTokenBalances (token_delta < 0)
    # but SOL barely changes (just rent). Skip — not a meaningful trade.
    return None


# ---------------------------------------------------------------------------
def analyse_wallet(wallet: str, tx_limit: int) -> dict:
    """
    Fetch and analyse up to tx_limit transactions for one wallet.
    Returns stats dict.
    """
    sigs = get_signatures(wallet, tx_limit)
    if not sigs:
        return {}

    # {mint: [list of buy SOL costs]}
    open_trades: dict[str, list[float]] = defaultdict(list)
    closed_trades: list[dict] = []

    for sig in sigs:
        time.sleep(CALL_DELAY)
        tx = rpc("getTransaction", [sig, {
            "encoding": "jsonParsed",
            "maxSupportedTransactionVersion": 0,
        }])
        if not tx:
            continue

        swap = parse_swap(tx, wallet)
        if not swap:
            continue

        mint = swap["mint"]
        sol  = swap["sol"]

        if swap["type"] == "buy":
            open_trades[mint].append(sol)

        elif swap["type"] == "sell" and mint in open_trades:
            costs = open_trades.pop(mint)
            total_cost = sum(costs)
            if total_cost > 0.001:   # skip dust
                pnl_pct = (sol - total_cost) / total_cost * 100
                closed_trades.append({
                    "mint":     mint,
                    "cost_sol": round(total_cost, 6),
                    "recv_sol": round(sol, 6),
                    "pnl_pct":  round(pnl_pct, 2),
                    "win":      pnl_pct > 0,
                })

    # --- Unrealized PnL for still-open positions ---
    # Wallets that hold without selling still tell us if they pick winners.
    # Check current price vs buy price for each open position.
    unrealized_trades = []
    if open_trades:
        for mint, costs in open_trades.items():
            total_cost = sum(costs)
            if total_cost < 0.001:
                continue
            time.sleep(CALL_DELAY)
            current_price = _get_current_price_sol(mint)
            if current_price is None:
                # Token dead/no liquidity — treat as -100% loss
                unrealized_trades.append({
                    "mint": mint, "cost_sol": round(total_cost, 6),
                    "recv_sol": 0.0, "pnl_pct": -100.0,
                    "win": False, "status": "dead",
                })
            else:
                # We don't know exact token amount bought, so estimate PnL
                # by comparing buy price approximation (cost/tokens ≈ entry)
                # Use cost as basis — current_price relative to reference
                # Simplification: mark as unrealized with price available
                unrealized_trades.append({
                    "mint": mint, "cost_sol": round(total_cost, 6),
                    "recv_sol": round(current_price, 6),
                    "pnl_pct": round((current_price - total_cost) / total_cost * 100, 2),
                    "win": current_price > total_cost,
                    "status": "holding",
                })

    all_trades = closed_trades + unrealized_trades

    if not all_trades:
        return {"wallet": wallet, "trades": 0}

    wins     = [t for t in all_trades if t["win"]]
    losses   = [t for t in all_trades if not t["win"]]
    win_rate = len(wins) / len(all_trades)
    avg_roi  = sum(t["pnl_pct"] for t in all_trades) / len(all_trades)
    avg_win  = sum(t["pnl_pct"] for t in wins)   / len(wins)   if wins   else 0
    avg_loss = sum(t["pnl_pct"] for t in losses)  / len(losses) if losses else 0

    # Composite score: win_rate * avg_roi (only positive contributions count)
    score = win_rate * max(avg_roi, 0)

    return {
        "wallet":            wallet,
        "trades":            len(all_trades),
        "closed_trades_n":   len(closed_trades),
        "unrealized_n":      len(unrealized_trades),
        "win_rate":          round(win_rate, 4),
        "avg_roi":           round(avg_roi, 2),
        "avg_win":           round(avg_win, 2),
        "avg_loss":          round(avg_loss, 2),
        "score":             round(score, 4),
        "closed_trades":     closed_trades,
        "unrealized_trades": unrealized_trades,
    }


def _get_current_price_sol(mint: str) -> float | None:
    """Fetch current token price in SOL via DexScreener. Returns None if dead."""
    try:
        r = requests.get(
            f"https://api.dexscreener.com/latest/dex/tokens/{mint}",
            timeout=6,
        )
        if r.status_code != 200:
            return None
        pairs = r.json().get("pairs") or []
        sol_pairs = [p for p in pairs if p.get("chainId") == "solana"]
        if not sol_pairs:
            return None
        best = max(sol_pairs, key=lambda p: p.get("liquidity", {}).get("usd", 0) or 0)
        price_usd = float(best.get("priceUsd") or 0)
        # Convert to SOL (rough: assume SOL = $150, good enough for relative ranking)
        SOL_PRICE_USD = 150
        return price_usd / SOL_PRICE_USD if price_usd > 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",  type=int, default=DEFAULT_TX_LIMIT,
                        help="Max transactions per wallet (default 200)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip wallets already in output file")
    args = parser.parse_args()

    wallets = json.loads(WALLETS.read_text())
    print(f"Loaded {len(wallets)} wallets from {WALLETS.name}")
    print(f"Fetching last {args.limit} txs per wallet\n")

    # Load existing results for resume
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

        trades   = stats.get("trades", 0)
        closed_n = stats.get("closed_trades_n", 0)
        unreal_n = stats.get("unrealized_n", 0)
        if trades >= MIN_TRADES:
            wr  = stats.get("win_rate", 0) * 100
            roi = stats.get("avg_roi", 0)
            print(f"{trades} trades ({closed_n} closed +{unreal_n} holding)  "
                  f"{wr:.0f}% win  avg_roi {roi:+.1f}%")
        else:
            print(f"{trades} trades found (need {MIN_TRADES} to rank)")

        # Save after every wallet so progress isn't lost
        ranked = sorted(
            [r for r in results if r.get("trades", 0) >= MIN_TRADES],
            key=lambda x: x.get("score", 0), reverse=True
        )
        OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUT_FILE.write_text(json.dumps(ranked, indent=2))

    # Final ranked output
    ranked = sorted(
        [r for r in results if r.get("trades", 0) >= MIN_TRADES],
        key=lambda x: x.get("score", 0), reverse=True
    )
    OUT_FILE.write_text(json.dumps(ranked, indent=2))

    print(f"\n{'='*60}")
    print(f"Done. {len(ranked)} wallets ranked (≥{MIN_TRADES} trades each)")
    print(f"Saved to {OUT_FILE}\n")
    print(f"{'Rank':<5} {'Name':<20} {'Trades':>6} {'Win%':>6} {'AvgROI':>8} {'Score':>7}")
    print("-"*60)
    for rank, r in enumerate(ranked[:20], 1):
        print(f"{rank:<5} {r.get('name','?'):<20} {r['trades']:>6} "
              f"{r['win_rate']*100:>5.1f}%  {r['avg_roi']:>+7.1f}%  {r['score']:>7.4f}")


if __name__ == "__main__":
    main()
