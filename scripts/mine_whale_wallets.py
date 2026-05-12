"""
Mine fresh whale wallets from Solana moonshot tokens.

Strategy:
  1. Find tokens that actually 10x+ recently on Solana (via GeckoTerminal + DexScreener)
  2. For each moonshot, get the EARLY buyers (bought before the pump)
  3. Wallets appearing as early buyers across multiple moonshots = real smart money
  4. Rank by: appearances × (1 / avg_entry_position)
  5. MERGE into existing whale_wallets_sol.json (no replacements)

Run:
  python3 scripts/mine_whale_wallets.py
  python3 scripts/mine_whale_wallets.py --tokens 100 --top 200 --min-gain 500
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

ROOT         = Path(__file__).parent.parent
SOL_WALLETS  = ROOT / "memecoin" / "data" / "whale_wallets_sol.json"
HELIUS_KEY   = os.getenv("HELIUS_API_KEY", "")
HELIUS_BASE  = "https://api.helius.xyz/v0"
DEX_BASE     = "https://api.dexscreener.com"
GECKO_BASE   = "https://api.geckoterminal.com/api/v2"

CALL_DELAY        = 0.25
WSOL              = "So11111111111111111111111111111111111111112"
EARLY_BUYER_CUTOFF = 30   # look at first 30 buyers per token


# ---------------------------------------------------------------------------
# Step 1: Find moonshot tokens
# ---------------------------------------------------------------------------

def fetch_moonshots_gecko(min_gain_pct: float = 500, limit: int = 100) -> list[str]:
    """
    GeckoTerminal free API — get top gaining Solana pools in last 24h.
    Returns list of token mint addresses.
    """
    mints = []
    print("Fetching top gainers from GeckoTerminal...")

    # Top gaining pools on Solana sorted by price_change_percentage_24h
    for page in range(1, 4):  # up to 3 pages = 60 pools
        try:
            r = requests.get(
                f"{GECKO_BASE}/networks/solana/trending_pools",
                params={"page": page},
                headers={"Accept": "application/json"},
                timeout=10,
            )
            if r.status_code != 200:
                break
            pools = r.json().get("data", [])
            if not pools:
                break
            for pool in pools:
                attrs = pool.get("attributes", {})
                change = attrs.get("price_change_percentage", {})
                h24 = float(change.get("h24") or 0)
                if h24 >= min_gain_pct:
                    # base token address
                    rels = pool.get("relationships", {})
                    base = rels.get("base_token", {}).get("data", {})
                    token_id = base.get("id", "")  # format: "solana_<mint>"
                    if token_id.startswith("solana_"):
                        mint = token_id.replace("solana_", "")
                        if mint and mint != WSOL and mint not in mints:
                            mints.append(mint)
            time.sleep(CALL_DELAY)
        except Exception:
            break

    print(f"  GeckoTerminal: {len(mints)} tokens with >{min_gain_pct}% gain")
    return mints


def fetch_moonshots_dex(min_gain_pct: float = 500, limit: int = 100) -> list[str]:
    """
    DexScreener — search for Solana tokens and filter by h24 price change.
    Uses the token-boosts endpoint (boosted = real volume/attention) then
    enriches with pair data to check actual price gains.
    """
    mints = []
    print("Fetching moonshots from DexScreener...")

    # Get a batch of Solana pairs and filter by price change
    try:
        # DexScreener doesn't have a "sort by gain" endpoint, but
        # we can search for pump.fun tokens and filter
        for query in ["pump", "sol", "moon", "cat", "dog", "pepe"]:
            r = requests.get(
                f"{DEX_BASE}/latest/dex/search?q={query}",
                timeout=10,
            )
            if r.status_code != 200:
                continue
            pairs = r.json().get("pairs") or []
            for p in pairs:
                if p.get("chainId") != "solana":
                    continue
                h24 = float((p.get("priceChange") or {}).get("h24") or 0)
                if h24 >= min_gain_pct:
                    mint = (p.get("baseToken") or {}).get("address", "")
                    if mint and mint != WSOL and mint not in mints:
                        mints.append(mint)
            time.sleep(CALL_DELAY)
            if len(mints) >= limit:
                break
    except Exception:
        pass

    print(f"  DexScreener: {len(mints)} tokens with >{min_gain_pct}% gain")
    return mints[:limit]


def fetch_pump_fun_graduated(limit: int = 50) -> list[str]:
    """
    Pump.fun graduated tokens = hit full $69K bonding curve → moved to Raydium.
    Every graduation is a proven moonshot (started near zero, hit $69K+).
    Use DexScreener to find recent Raydium pairs that originated from Pump.fun.
    """
    mints = []
    print("Fetching recently graduated Pump.fun tokens...")

    try:
        r = requests.get(
            f"{DEX_BASE}/token-profiles/latest/v1",
            timeout=10,
        )
        if r.status_code == 200:
            for item in r.json():
                if item.get("chainId") == "solana":
                    mint = item.get("tokenAddress", "")
                    if mint and mint not in mints:
                        mints.append(mint)
    except Exception:
        pass

    print(f"  Pump.fun graduated: {len(mints)} tokens")
    return mints[:limit]


# ---------------------------------------------------------------------------
# Step 2: Find early buyers of each moonshot
# ---------------------------------------------------------------------------

def get_early_buyers(mint: str, cutoff: int = EARLY_BUYER_CUTOFF) -> list[str]:
    """
    Get first N wallets that bought this token.
    Uses Helius — fetches SWAP transactions for the token mint, sorted oldest-first.
    """
    url = f"{HELIUS_BASE}/addresses/{mint}/transactions"
    params = {"api-key": HELIUS_KEY, "limit": 100, "type": "SWAP"}

    try:
        r = requests.get(url, params=params, timeout=12)
        if r.status_code != 200:
            return []
        txs = r.json()
        if not txs:
            return []

        # Oldest first = who bought earliest
        txs.sort(key=lambda t: t.get("timestamp", 0))

        buyers = []
        seen = set()
        for tx in txs:
            for transfer in tx.get("tokenTransfers", []):
                if transfer.get("mint") == mint:
                    wallet = transfer.get("toUserAccount", "")
                    if wallet and len(wallet) >= 32 and wallet not in seen:
                        if not _is_program(wallet):
                            seen.add(wallet)
                            buyers.append(wallet)
                            if len(buyers) >= cutoff:
                                return buyers
        return buyers
    except Exception:
        return []


def _is_program(wallet: str) -> bool:
    """Filter out known DEX programs and system accounts."""
    known = {
        "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
        "11111111111111111111111111111111",
        "ComputeBudget111111111111111111111111111111",
        "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJe1bS",
        "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",  # Raydium AMM
        "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P",   # Pump.fun
    }
    return wallet in known


# ---------------------------------------------------------------------------
# Step 3: Score and merge
# ---------------------------------------------------------------------------

def mine_and_merge(token_mints: list[str], top_n: int, min_appearances: int = 2):
    """Mine early buyers, score them, merge into existing wallet list."""
    wallet_appearances: dict[str, list[tuple[str, int]]] = defaultdict(list)

    for i, mint in enumerate(token_mints):
        print(f"  [{i+1:>3}/{len(token_mints)}] {mint[:14]}…", end="  ", flush=True)
        time.sleep(CALL_DELAY)

        buyers = get_early_buyers(mint)
        if not buyers:
            print("skip (no buyers)")
            continue

        print(f"{len(buyers)} early buyers")
        for pos, wallet in enumerate(buyers):
            wallet_appearances[wallet].append((mint, pos))

    # Score wallets
    scored = []
    for wallet, appearances in wallet_appearances.items():
        count = len(appearances)
        if count < min_appearances:
            continue
        avg_pos = sum(p for _, p in appearances) / count
        score = count * (1.0 / (avg_pos + 1))
        scored.append({
            "trackedWalletAddress": wallet,
            "name": f"mined_{wallet[:8]}",
            "appearances": count,
            "avg_early_position": round(avg_pos, 1),
            "score": round(score, 4),
            "source": "moonshot_miner",
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    new_wallets = scored[:top_n]

    if not new_wallets:
        print("\nNo wallets found appearing in 2+ moonshots. Try --min-gain lower or --tokens higher.")
        return

    # Load existing list and merge (no duplicates)
    existing = []
    if SOL_WALLETS.exists():
        existing = json.loads(SOL_WALLETS.read_text())

    existing_addrs = {
        w.get("trackedWalletAddress") or w.get("address") or ""
        for w in existing
    }

    added = [w for w in new_wallets if w["trackedWalletAddress"] not in existing_addrs]
    merged = existing + added

    SOL_WALLETS.write_text(json.dumps(merged, indent=2))

    print(f"\n{'='*60}")
    print(f"Mined {len(new_wallets)} smart wallets from moonshots")
    print(f"  {len(added)} NEW wallets added to whale_wallets_sol.json")
    print(f"  {len(new_wallets) - len(added)} already in list (skipped)")
    print(f"  Total wallet list: {len(merged)}")
    print()
    print(f"{'Rank':<5} {'Wallet':<16} {'Appearances':>12} {'AvgEarlyPos':>13} {'Score':>8}")
    print("-" * 60)
    for i, w in enumerate(new_wallets[:20], 1):
        marker = " (NEW)" if w in added else ""
        print(f"{i:<5} {w['trackedWalletAddress'][:14]:<16} "
              f"{w['appearances']:>12}  "
              f"{w['avg_early_position']:>12.1f}  "
              f"{w['score']:>8.4f}{marker}")

    print(f"\nNow run: python3 scripts/rank_wallets_helius.py --resume")


# ---------------------------------------------------------------------------

def main():
    if not HELIUS_KEY:
        print("ERROR: HELIUS_API_KEY not set in .env")
        raise SystemExit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens",    type=int,   default=80,
                        help="Number of moonshot tokens to mine from (default 80)")
    parser.add_argument("--top",       type=int,   default=150,
                        help="Top N wallets to add (default 150)")
    parser.add_argument("--min-gain",  type=float, default=500,
                        help="Min 24h price gain %% to qualify as moonshot (default 500)")
    args = parser.parse_args()

    print(f"Mining whale wallets from tokens with >{args.min_gain}% 24h gain\n")

    # Gather moonshots from multiple sources
    mints: list[str] = []

    gecko  = fetch_moonshots_gecko(args.min_gain, args.tokens)
    mints.extend(gecko)
    time.sleep(0.5)

    dex = fetch_moonshots_dex(args.min_gain, args.tokens)
    for m in dex:
        if m not in mints:
            mints.append(m)
    time.sleep(0.5)

    graduated = fetch_pump_fun_graduated(50)
    for m in graduated:
        if m not in mints:
            mints.append(m)

    mints = mints[:args.tokens]
    print(f"\nTotal unique moonshot tokens to mine: {len(mints)}\n")

    if not mints:
        print("No moonshot tokens found. Try lowering --min-gain.")
        raise SystemExit(1)

    mine_and_merge(mints, args.top)


if __name__ == "__main__":
    main()
