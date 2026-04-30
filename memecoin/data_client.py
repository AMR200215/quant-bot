"""
API wrappers for DexScreener, GMGN (free tier), Rugcheck, Honeypot.is,
Solana RPC, and BscScan.  All functions return plain dicts / None on failure.
"""

import logging
import time
from typing import Optional

import requests

from memecoin.config import (
    DEXSCREENER_BASE, GMGN_BASE, RUGCHECK_BASE, HONEYPOT_BASE,
    CHAINS,
)

log = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "quant-bot/1.0"})

_TIMEOUT = 10  # seconds


def _get(url: str, params: dict = None) -> Optional[dict]:
    try:
        r = SESSION.get(url, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.debug("GET %s failed: %s", url, e)
        return None


# ---------------------------------------------------------------------------
# DexScreener
# ---------------------------------------------------------------------------

def dex_get_token(chain: str, address: str) -> Optional[dict]:
    """Return the most liquid pair for a token on a given chain."""
    data = _get(f"{DEXSCREENER_BASE}/latest/dex/tokens/{address}")
    if not data or not data.get("pairs"):
        return None
    # pick the pair on the right chain with highest liquidity
    pairs = [p for p in data["pairs"] if p.get("chainId") == chain]
    if not pairs:
        pairs = data["pairs"]
    pairs.sort(key=lambda p: p.get("liquidity", {}).get("usd", 0) or 0, reverse=True)
    return pairs[0]


def dex_get_new_pairs(chain: str, limit: int = 50) -> list[dict]:
    """Return recently created pairs on a chain (DexScreener token-profiles endpoint)."""
    data = _get(f"{DEXSCREENER_BASE}/token-profiles/latest/v1")
    if not data:
        return []
    if isinstance(data, list):
        return [p for p in data if p.get("chainId") == chain][:limit]
    return []


def dex_get_boosted() -> list[dict]:
    """Return currently boosted/trending tokens across all chains."""
    data = _get(f"{DEXSCREENER_BASE}/token-boosts/latest/v1")
    if not data:
        return []
    if isinstance(data, list):
        return data
    return data.get("pairs", [])


def dex_search(query: str) -> list[dict]:
    data = _get(f"{DEXSCREENER_BASE}/latest/dex/search", params={"q": query})
    if not data:
        return []
    return data.get("pairs", [])


def dex_get_pairs_by_address(chain: str, pair_address: str) -> Optional[dict]:
    data = _get(f"{DEXSCREENER_BASE}/latest/dex/pairs/{chain}/{pair_address}")
    if not data or not data.get("pairs"):
        return None
    return data["pairs"][0]


# ---------------------------------------------------------------------------
# GMGN (free tier)  — best-effort, skip gracefully if endpoints change
# ---------------------------------------------------------------------------

def gmgn_trending_sol(limit: int = 20) -> list[dict]:
    """Return trending Solana tokens from GMGN."""
    data = _get(f"{GMGN_BASE}/tokens/sol/trending",
                params={"orderby": "volume", "direction": "desc", "limit": limit})
    if not data:
        return []
    return data.get("tokens", data if isinstance(data, list) else [])


def gmgn_new_sol(limit: int = 30) -> list[dict]:
    """Return newest Solana tokens from GMGN."""
    data = _get(f"{GMGN_BASE}/tokens/sol/trending",
                params={"orderby": "open_timestamp", "direction": "desc", "limit": limit})
    if not data:
        return []
    return data.get("tokens", data if isinstance(data, list) else [])


def gmgn_wallet_activity_sol(wallet: str, limit: int = 10) -> list[dict]:
    """Return recent swaps for a Solana wallet from GMGN."""
    data = _get(f"{GMGN_BASE}/wallet/sol/{wallet}/trades", params={"limit": limit})
    if not data:
        return []
    return data.get("trades", data if isinstance(data, list) else [])


def gmgn_wallet_stats_sol(wallet: str) -> Optional[dict]:
    """Return win-rate / PnL stats for a Solana wallet."""
    return _get(f"{GMGN_BASE}/wallet/sol/{wallet}/info")


# ---------------------------------------------------------------------------
# Solana RPC — wallet transaction polling
# ---------------------------------------------------------------------------

SOL_RPC = CHAINS["solana"]["rpc"]


def sol_get_recent_signatures(wallet: str, limit: int = 15,
                               before: str = None) -> list[dict]:
    """Return recent confirmed tx signatures for a Solana wallet."""
    params = {"limit": limit}
    if before:
        params["before"] = before
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getSignaturesForAddress",
        "params": [wallet, params],
    }
    try:
        r = SESSION.post(SOL_RPC, json=payload, timeout=_TIMEOUT)
        r.raise_for_status()
        result = r.json().get("result", [])
        return result or []
    except Exception as e:
        log.debug("sol_get_recent_signatures(%s) failed: %s", wallet, e)
        return []


def sol_get_transaction(signature: str) -> Optional[dict]:
    """Fetch a full Solana transaction by signature."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTransaction",
        "params": [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
    }
    try:
        r = SESSION.post(SOL_RPC, json=payload, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json().get("result")
    except Exception as e:
        log.debug("sol_get_transaction(%s) failed: %s", signature, e)
        return None


def sol_parse_swap(tx: dict) -> Optional[dict]:
    """
    Extract token swap info from a parsed Solana transaction.
    Returns dict with keys: action (buy/sell), token_address, token_symbol,
    amount_sol, or None if not a swap.
    """
    if not tx:
        return None
    try:
        meta = tx.get("meta", {})
        if meta.get("err"):
            return None

        pre_token  = {b["accountIndex"]: b for b in meta.get("preTokenBalances", [])}
        post_token = {b["accountIndex"]: b for b in meta.get("postTokenBalances", [])}

        # Find the token account that changed most
        changes = []
        for idx, post in post_token.items():
            pre = pre_token.get(idx, {})
            pre_amt  = float(pre.get("uiTokenAmount", {}).get("uiAmountString") or 0)
            post_amt = float(post.get("uiTokenAmount", {}).get("uiAmountString") or 0)
            delta = post_amt - pre_amt
            if abs(delta) > 0:
                changes.append({
                    "mint": post.get("mint"),
                    "delta": delta,
                    "symbol": post.get("uiTokenAmount", {}).get("uiAmountString"),
                })

        if not changes:
            return None

        # largest absolute delta = the token being swapped
        main = max(changes, key=lambda x: abs(x["delta"]))
        mint = main["mint"]
        if not mint:
            return None

        # SOL balance change for the fee payer
        pre_sol  = (meta.get("preBalances") or [0])[0] / 1e9
        post_sol = (meta.get("postBalances") or [0])[0] / 1e9
        sol_delta = post_sol - pre_sol

        action = "buy" if main["delta"] > 0 else "sell"
        return {
            "action": action,
            "token_address": mint,
            "token_symbol": "",   # filled later from DexScreener
            "amount_sol": abs(sol_delta),
        }
    except Exception as e:
        log.debug("sol_parse_swap failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# BscScan — wallet transaction polling
# ---------------------------------------------------------------------------

def bscscan_get_token_txs(wallet: str, api_key: str = "",
                           start_block: int = 0) -> list[dict]:
    """Return BEP-20 token transfers for a BSC wallet."""
    params = {
        "module": "account",
        "action": "tokentx",
        "address": wallet,
        "startblock": start_block,
        "endblock": 99999999,
        "sort": "desc",
        "offset": 20,
        "page": 1,
    }
    if api_key:
        params["apikey"] = api_key
    data = _get(CHAINS["bsc"]["bscscan_api"], params=params)
    if not data or data.get("status") != "1":
        return []
    return data.get("result", [])


def bscscan_parse_swap(txs: list[dict], wallet: str) -> list[dict]:
    """
    Group BEP-20 transfers by txHash to identify buy/sell swaps.
    Returns list of swap dicts: action, token_address, token_symbol, tx_hash.
    """
    from collections import defaultdict
    by_hash = defaultdict(list)
    for tx in txs:
        by_hash[tx["hash"]].append(tx)

    swaps = []
    wallet_lower = wallet.lower()
    for tx_hash, transfers in by_hash.items():
        for t in transfers:
            if t["to"].lower() == wallet_lower:
                swaps.append({
                    "action": "buy",
                    "token_address": t["contractAddress"],
                    "token_symbol": t.get("tokenSymbol", ""),
                    "token_name": t.get("tokenName", ""),
                    "tx_hash": tx_hash,
                    "block": int(t.get("blockNumber", 0)),
                })
            elif t["from"].lower() == wallet_lower:
                swaps.append({
                    "action": "sell",
                    "token_address": t["contractAddress"],
                    "token_symbol": t.get("tokenSymbol", ""),
                    "token_name": t.get("tokenName", ""),
                    "tx_hash": tx_hash,
                    "block": int(t.get("blockNumber", 0)),
                })
    return swaps


# ---------------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------------

def rugcheck_sol(token_address: str) -> Optional[dict]:
    """
    Rugcheck.xyz summary for a Solana token.
    Returns dict with 'score', 'risks', 'is_safe' keys.
    """
    url = CHAINS["solana"]["rugcheck_url"].format(address=token_address)
    data = _get(url)
    if not data:
        return None
    score = data.get("score", 500)
    risks = data.get("risks", [])
    # rugcheck score: lower = safer (0-1000)
    return {
        "score": score,
        "risks": [r.get("name", "") for r in risks],
        "is_safe": score < 300 and not any(
            r.get("level") == "danger" for r in risks
        ),
    }


def honeypot_bsc(token_address: str) -> Optional[dict]:
    """Check if a BSC token is a honeypot."""
    url = CHAINS["bsc"]["honeypot_url"].format(address=token_address)
    data = _get(url)
    if not data:
        return None
    hp = data.get("honeypotResult", {})
    sim = data.get("simulationResult", {})
    return {
        "is_honeypot": hp.get("isHoneypot", True),
        "buy_tax":  sim.get("buyTax", 100),
        "sell_tax": sim.get("sellTax", 100),
        "is_safe": not hp.get("isHoneypot", True)
                   and sim.get("sellTax", 100) < 10,
    }
