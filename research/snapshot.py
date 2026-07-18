"""
Token snapshot fetcher — DexScreener + rugcheck.
Standalone: no imports from memecoin/.

Returns a flat dict of raw fields.  No screener decisions here —
those are computed at analysis time from the raw fields + config thresholds.
"""

import logging
import time
from typing import Optional

import requests

from research.config import (
    DEXSCREENER_BASE, RUGCHECK_BASE,
    DEX_RETRY_COUNT, DEX_RETRY_DELAY_S, DEX_CALL_DELAY_S,
)

log = logging.getLogger(__name__)

_last_dex_call: float = 0.0   # rate-limit guard (shared within process)


def _dex_request(path: str, timeout: int = 8) -> Optional[dict]:
    """Single DexScreener GET with rate-limit courtesy delay."""
    global _last_dex_call
    gap = time.time() - _last_dex_call
    if gap < DEX_CALL_DELAY_S:
        time.sleep(DEX_CALL_DELAY_S - gap)
    try:
        r = requests.get(f"{DEXSCREENER_BASE}{path}", timeout=timeout)
        _last_dex_call = time.time()
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log.debug("DexScreener request failed: %s", e)
    _last_dex_call = time.time()
    return None


def _rugcheck_request(token_address: str, timeout: int = 8) -> Optional[dict]:
    try:
        url = f"{RUGCHECK_BASE}/tokens/{token_address}/report/summary"
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        log.debug("Rugcheck request failed for %s: %s", token_address[:8], e)
    return None


def _rugcheck_holders(token_address: str, timeout: int = 8) -> dict:
    """
    Fetch full rugcheck report for holder concentration data.
    Returns dict with top10_holder_pct and creator_holds_pct (both may be None).
    Uses /report (not /report/summary) which includes topHolders array.
    """
    result: dict = {"top10_holder_pct": None, "creator_holds_pct": None}
    try:
        url = f"{RUGCHECK_BASE}/tokens/{token_address}/report"
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return result
        data = r.json()

        # topHolders: [{address, pct, insider, owner, ...}, ...]
        holders = data.get("topHolders") or []
        if holders:
            pcts = [float(h.get("pct") or 0) for h in holders[:10]]
            result["top10_holder_pct"] = round(sum(pcts), 2) if pcts else None

        # Creator / dev wallet — look in multiple possible locations
        creator_pct = None
        for key in ("creator", "creatorInfo", "dev"):
            c = data.get(key) or {}
            if isinstance(c, dict) and c.get("pct") is not None:
                creator_pct = float(c["pct"])
                break
        # Also check if any topHolder is flagged as insider (creator)
        if creator_pct is None:
            insider = next((h for h in holders if h.get("insider")), None)
            if insider:
                creator_pct = float(insider.get("pct") or 0)
        result["creator_holds_pct"] = round(creator_pct, 2) if creator_pct is not None else None
    except Exception as e:
        log.debug("Rugcheck holders failed for %s: %s", token_address[:8], e)
    return result


def _best_pair(data: dict) -> Optional[dict]:
    """Pick the pair with highest liquidity from DexScreener response."""
    pairs = data.get("pairs") or []
    if not pairs:
        return None
    return max(pairs, key=lambda p: float((p.get("liquidity") or {}).get("usd") or 0))


def fetch_snapshot(token_address: str, chain: str = "solana") -> dict:
    """
    Fetch DexScreener + rugcheck for a token.
    Returns a flat dict of all raw fields.
    'snapshot_ok' is True only when DexScreener returned usable price data.

    Does NOT retry — call fetch_snapshot_with_retry for the retry wrapper.
    """
    result: dict = {
        "snapshot_ok": False,
        "symbol": None,
        "price_usd": None,
        "mcap_usd": None,
        "liquidity_usd": None,
        "fdv": None,
        "age_minutes": None,
        "volume_5m": None,
        "volume_1h": None,
        "buys_5m": None,
        "sells_5m": None,
        "buy_sell_ratio_5m": None,
        "buys_1h": None,
        "sells_1h": None,
        "buy_sell_ratio_1h": None,
        "price_change_5m": None,
        "price_change_1h": None,
        "price_change_6h": None,
        "dex_id": None,
        "has_twitter": None,
        "has_telegram": None,
        "has_website": None,
        "rugcheck_score": None,
        "mint_disabled": None,
        "freeze_disabled": None,
    }

    # ── DexScreener ──────────────────────────────────────────────────────────
    data = _dex_request(f"/latest/dex/tokens/{token_address}")
    pair = _best_pair(data) if data else None

    if not pair:
        # DexScreener hasn't indexed this token yet (30-90s lag for pump.fun).
        # Try Jupiter for at least a price_usd — covers ~60% more tokens.
        # snapshot_ok stays False (partial data: no volume/liquidity/BSR).
        jup = _jupiter_price(token_address)
        if jup:
            result["price_usd"] = jup
            # Pump.fun supply = 1e9 tokens; gives approximate mcap
            result["mcap_usd"] = jup * 1_000_000_000
        return result

    price = float(pair.get("priceUsd") or 0)
    if price <= 0:
        return result   # pair exists but no price yet

    result["snapshot_ok"] = True
    result["symbol"]      = (pair.get("baseToken") or {}).get("symbol") or None
    result["price_usd"]   = price

    pc = pair.get("priceChange") or {}
    result["price_change_5m"] = float(pc.get("m5") or 0)
    result["price_change_1h"] = float(pc.get("h1") or 0)
    result["price_change_6h"] = float(pc.get("h6") or 0)

    vol = pair.get("volume") or {}
    result["volume_5m"] = float(vol.get("m5") or 0)
    result["volume_1h"] = float(vol.get("h1") or 0)

    txns = pair.get("txns") or {}
    m5   = txns.get("m5") or {}
    h1   = txns.get("h1") or {}
    b5   = int(m5.get("buys") or 0)
    s5   = int(m5.get("sells") or 0)
    b1h  = int(h1.get("buys") or 0)
    s1h  = int(h1.get("sells") or 0)
    result["buys_5m"]          = b5
    result["sells_5m"]         = s5
    result["buys_1h"]          = b1h
    result["sells_1h"]         = s1h
    result["buy_sell_ratio_5m"]  = round(b5  / (b5  + s5)  , 3) if (b5  + s5)  else None
    result["buy_sell_ratio_1h"]  = round(b1h / (b1h + s1h), 3) if (b1h + s1h) else None

    liq  = float((pair.get("liquidity") or {}).get("usd") or 0)
    mcap = float(pair.get("marketCap") or pair.get("fdv") or 0)
    result["liquidity_usd"] = liq if liq > 0 else None
    result["mcap_usd"]      = mcap if mcap > 0 else None
    result["fdv"]           = float(pair.get("fdv") or mcap) or None
    result["dex_id"]        = pair.get("dexId") or None

    created_ms = pair.get("pairCreatedAt")
    if created_ms:
        result["age_minutes"] = (time.time() - created_ms / 1000) / 60

    info     = pair.get("info") or {}
    socials  = info.get("socials") or []
    websites = info.get("websites") or []
    types    = {s.get("type", "").lower() for s in socials}
    result["has_twitter"]  = "twitter"  in types
    result["has_telegram"] = "telegram" in types
    result["has_website"]  = len(websites) > 0

    # ── Rugcheck (Solana only) ────────────────────────────────────────────────
    if chain == "solana":
        rc = _rugcheck_request(token_address)
        if rc:
            result["rugcheck_score"]  = rc.get("score")
            result["mint_disabled"]   = rc.get("mint_disabled")
            result["freeze_disabled"] = rc.get("freeze_disabled")
        # Full report for holder concentration (separate endpoint, best-effort)
        holder_data = _rugcheck_holders(token_address)
        result["top10_holder_pct"]  = holder_data.get("top10_holder_pct")
        result["creator_holds_pct"] = holder_data.get("creator_holds_pct")

    return result


def fetch_snapshot_with_retry(
    token_address: str,
    chain: str = "solana",
    retries: int = DEX_RETRY_COUNT,
    delay: float = DEX_RETRY_DELAY_S,
) -> tuple[dict, int]:
    """
    Fetch snapshot with up to `retries` attempts spaced `delay` seconds apart.
    Returns (snapshot_dict, attempts_used).
    If all retries fail, returns the empty snapshot with snapshot_ok=False.
    """
    for attempt in range(1, retries + 1):
        snap = fetch_snapshot(token_address, chain)
        if snap["snapshot_ok"]:
            return snap, attempt
        if attempt < retries:
            log.debug("Snapshot miss attempt %d/%d for %s — retrying in %ds",
                      attempt, retries, token_address[:8], delay)
            time.sleep(delay)
    log.info("Snapshot: no DexScreener data after %d attempts for %s",
             retries, token_address[:8])
    return fetch_snapshot(token_address, chain), retries


def fetch_first_buyers(
    token_address: str,
    helius_key: str,
    n: int = 30,
) -> list:
    """
    Return up to n unique early-buyer wallet addresses for a pump.fun token.

    Algorithm:
      1. getSignaturesForAddress(mint, limit=200) via Helius RPC → DESC order
      2. Reverse to get oldest-first; filter successful txns only
      3. Batch-parse via Helius enhanced API → extract feePayer for each SWAP
         where the feePayer receives the token (= buy, not sell)
      4. Return deduplicated wallet list (first-seen order = earliest buyers first)

    Works best for fresh tokens (<200 total txns at call time).
    For backfill of older tokens with many txns, captures the first 200
    transactions only — sufficient if the token was a quick pump-and-dump.

    Returns [] on any error or if helius_key is empty.
    """
    if not helius_key or not token_address:
        return []

    rpc_url   = f"https://mainnet.helius-rpc.com/?api-key={helius_key}"
    parse_url = f"https://api.helius.xyz/v0/transactions/?api-key={helius_key}"

    # Step 1: fetch signatures (DESC = newest first)
    try:
        r1 = requests.post(
            rpc_url,
            json={
                "jsonrpc": "2.0", "id": 1,
                "method":  "getSignaturesForAddress",
                "params":  [token_address, {"limit": 200, "commitment": "confirmed"}],
            },
            timeout=15,
        )
        sigs_result = r1.json().get("result") or []
    except Exception as _e:
        log.debug("fetch_first_buyers: getSignaturesForAddress failed %s: %s",
                  token_address[:8], _e)
        return []

    if not sigs_result:
        return []

    # Oldest first; skip failed transactions
    sigs_result.reverse()
    clean_sigs = [s["signature"] for s in sigs_result if not s.get("err")]
    if not clean_sigs:
        return []

    # Step 2: parse oldest min(n*3, 100) sigs
    batch = clean_sigs[:min(n * 3, 100)]
    try:
        r2 = requests.post(
            parse_url,
            json={"transactions": batch},
            timeout=25,
        )
        parsed = r2.json()
        if not isinstance(parsed, list):
            return []
    except Exception as _e:
        log.debug("fetch_first_buyers: Helius parse failed %s: %s",
                  token_address[:8], _e)
        return []

    # Step 3: extract buyers — feePayer who RECEIVES the token (not sells)
    buyers:  list = []
    seen:    set  = set()
    for tx in parsed:
        if not isinstance(tx, dict) or tx.get("type") not in ("SWAP", "UNKNOWN"):
            continue
        fee_payer = tx.get("feePayer", "")
        if not fee_payer:
            continue
        token_transfers = tx.get("tokenTransfers") or []
        for tt in token_transfers:
            if tt.get("mint") == token_address and tt.get("toUserAccount") == fee_payer:
                if fee_payer not in seen:
                    seen.add(fee_payer)
                    buyers.append(fee_payer)
                break
        if len(buyers) >= n:
            break

    log.debug("fetch_first_buyers: %s → %d buyers", token_address[:8], len(buyers))
    return buyers[:n]


def _jupiter_price(token_address: str) -> Optional[float]:
    """
    Jupiter Price API v2 — fast fallback for tokens not yet indexed by DexScreener.
    Solana only.  Returns price_usd or None.
    """
    try:
        r = requests.get(
            f"https://api.jup.ag/price/v2?ids={token_address}",
            timeout=5,
        )
        if r.status_code == 200:
            entry = (r.json().get("data") or {}).get(token_address)
            if entry:
                price = float(entry.get("price") or 0)
                return price if price > 0 else None
    except Exception as e:
        log.debug("Jupiter price fallback failed for %s: %s", token_address[:8], e)
    return None


def fetch_price(token_address: str) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Lightweight price-only fetch for outcome polling.
    Returns (price_usd, mcap_usd, liquidity_usd) or (None, None, None).
    Tries DexScreener first; falls back to Jupiter for Solana tokens that
    DexScreener hasn't indexed yet.
    """
    data = _dex_request(f"/latest/dex/tokens/{token_address}")
    pair = _best_pair(data) if data else None
    if pair:
        price = float(pair.get("priceUsd") or 0) or None
        mcap  = float(pair.get("marketCap") or 0) or None
        liq   = float((pair.get("liquidity") or {}).get("usd") or 0) or None
        if price:
            return price, mcap, liq

    # Jupiter fallback — no mcap/liq available but better than returning None
    jup_price = _jupiter_price(token_address)
    if jup_price:
        return jup_price, None, None

    return None, None, None
