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
    risk_names  = [r.get("name", "") for r in risks]
    danger_names = [r.get("name", "") for r in risks if r.get("level") == "danger"]

    # Mint / freeze authority flags
    mint_disabled   = not any("Mint"   in n for n in risk_names)
    freeze_disabled = not any("Freeze" in n for n in risk_names)

    # rugcheck score: lower = safer (0-1000)
    return {
        "score":          score,
        "risks":          risk_names,
        "danger_risks":   danger_names,
        "mint_disabled":  mint_disabled,    # True = mint authority revoked (safer)
        "freeze_disabled": freeze_disabled, # True = freeze authority revoked (safer)
        "is_safe": score < 300 and not any(
            r.get("level") == "danger" for r in risks
        ),
    }


def sol_get_token_creator(token_address: str) -> Optional[str]:
    """
    Find the wallet that deployed a Solana token by tracing to its oldest transaction.
    For pump.fun tokens this is the dev who called the create instruction.
    """
    before = None
    last_page: list[dict] = []
    for _ in range(10):  # max 1000 txs lookback
        page = sol_get_recent_signatures(token_address, limit=100, before=before)
        if not page:
            break
        last_page = page
        if len(page) < 100:
            break
        before = page[-1]["signature"]

    if not last_page:
        return None

    tx = sol_get_transaction(last_page[-1]["signature"])
    if not tx:
        return None

    try:
        keys = tx["transaction"]["message"]["accountKeys"]
        for key in keys:
            if isinstance(key, dict) and key.get("signer") and key.get("writable"):
                return key["pubkey"]
        # fallback: first key is always the fee payer
        if keys:
            k = keys[0]
            return k["pubkey"] if isinstance(k, dict) else k
    except Exception as e:
        log.debug("sol_get_token_creator parse failed: %s", e)
    return None


PUMPFUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"


def sol_detect_new_tokens(dev_wallet: str,
                           after_sig: str = "") -> tuple[list[str], str]:
    """
    Detect new pump.fun token launches by a Solana dev wallet.
    Returns (list_of_new_mint_addresses, latest_sig_seen).
    """
    sigs = sol_get_recent_signatures(dev_wallet, limit=20)
    if not sigs:
        return [], after_sig

    latest_sig = sigs[0]["signature"]
    new_mints: list[str] = []

    for sig_info in sigs:
        sig = sig_info["signature"]
        if sig == after_sig:
            break
        tx = sol_get_transaction(sig)
        if not tx:
            continue
        try:
            keys = tx["transaction"]["message"].get("accountKeys", [])
            prog_ids = {k["pubkey"] if isinstance(k, dict) else k for k in keys}
            if PUMPFUN_PROGRAM not in prog_ids:
                continue
            meta = tx.get("meta", {})
            pre_mints = {b["mint"] for b in meta.get("preTokenBalances", [])}
            for tb in meta.get("postTokenBalances", []):
                mint = tb.get("mint")
                if mint and mint not in pre_mints and mint not in new_mints:
                    new_mints.append(mint)
        except Exception:
            continue

    return new_mints, latest_sig


def bscscan_get_contract_creator(token_address: str,
                                  api_key: str = "") -> Optional[str]:
    """Get the deployer wallet of a BSC token contract."""
    params = {
        "module": "contract",
        "action": "getcontractcreation",
        "contractaddresses": token_address,
    }
    if api_key:
        params["apikey"] = api_key
    data = _get(CHAINS["bsc"]["bscscan_api"], params=params)
    if not data or data.get("status") != "1":
        return None
    results = data.get("result", [])
    return results[0].get("contractCreator") if results else None


def bscscan_detect_new_contracts(dev_wallet: str, api_key: str = "",
                                  start_block: int = 0) -> tuple[list[dict], int]:
    """
    Detect new token contract deployments by a BSC dev wallet.
    Returns (list of {contract_address, block}, latest_block_seen).
    """
    params = {
        "module": "account",
        "action": "txlist",
        "address": dev_wallet,
        "startblock": start_block + 1,
        "endblock": 99999999,
        "sort": "desc",
        "page": 1,
        "offset": 20,
    }
    if api_key:
        params["apikey"] = api_key
    data = _get(CHAINS["bsc"]["bscscan_api"], params=params)
    if not data or data.get("status") != "1":
        return [], start_block

    contracts: list[dict] = []
    latest_block = start_block
    for tx in data.get("result", []):
        block = int(tx.get("blockNumber", 0))
        latest_block = max(latest_block, block)
        if tx.get("to") == "" and tx.get("contractAddress"):
            contracts.append({
                "contract_address": tx["contractAddress"].lower(),
                "block": block,
            })
    return contracts, latest_block


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
