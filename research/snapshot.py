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
        return result   # snapshot_ok stays False

    price = float(pair.get("priceUsd") or 0)
    if price <= 0:
        return result   # pair exists but no price yet

    result["snapshot_ok"] = True
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


def fetch_price(token_address: str) -> Optional[float]:
    """
    Lightweight price-only fetch for outcome polling.
    Returns (price_usd, mcap_usd, liquidity_usd) or (None, None, None).
    """
    data = _dex_request(f"/latest/dex/tokens/{token_address}")
    pair = _best_pair(data) if data else None
    if not pair:
        return None, None, None
    price = float(pair.get("priceUsd") or 0) or None
    mcap  = float(pair.get("marketCap") or 0) or None
    liq   = float((pair.get("liquidity") or {}).get("usd") or 0) or None
    return price, mcap, liq
