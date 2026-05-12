"""
Phase 2b — Token outcome tracker.

Runs every 4h via GitHub Actions.
- Fetches new/trending Solana tokens from DexScreener
- Inserts new tokens with launch_price = current_price
- Updates existing tokens: current_price, peak_price, peak_multiple
- Marks winners (peak_multiple >= 5) and rugged tokens

Run:
    python3 -m wallet_db.outcome_tracker
"""

import logging
import os
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from wallet_db.db import get_conn, init_db

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEX_BASE   = "https://api.dexscreener.com"
CALL_DELAY = 0.3

# Thresholds
MIN_PEAK_MULTIPLE  = 5.0     # 5x to qualify as winner
MIN_LIQUIDITY      = 20_000  # $20K
MIN_VOLUME_24H     = 100_000 # $100K
MAX_AGE_DAYS       = 14
RUG_PRICE_RATIO    = 0.10    # current < 10% of peak = rugged
RUG_LIQUIDITY      = 5_000   # < $5K liq = rugged


def _get(url: str, params: dict = None) -> dict | list | None:
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.debug("GET %s failed: %s", url, e)
        return None


def _fetch_sources() -> list[dict]:
    """Gather token pairs from multiple DexScreener sources."""
    pairs = []
    now   = time.time()
    cutoff = now - MAX_AGE_DAYS * 86400

    # Source 1: new token profiles (latest launches)
    data = _get(f"{DEX_BASE}/token-profiles/latest/v1")
    if isinstance(data, list):
        for item in data:
            if item.get("chainId") == "solana":
                pairs.append({"tokenAddress": item.get("tokenAddress"), "_source": "profile"})
    time.sleep(CALL_DELAY)

    # Source 2: boosted tokens
    data = _get(f"{DEX_BASE}/token-boosts/latest/v1")
    if isinstance(data, list):
        for item in data:
            if item.get("chainId") == "solana":
                pairs.append({"tokenAddress": item.get("tokenAddress"), "_source": "boosted"})
    time.sleep(CALL_DELAY)

    # Source 3: top boosted
    data = _get(f"{DEX_BASE}/token-boosts/top/v1")
    if isinstance(data, list):
        for item in data:
            if item.get("chainId") == "solana":
                pairs.append({"tokenAddress": item.get("tokenAddress"), "_source": "top_boosted"})
    time.sleep(CALL_DELAY)

    return pairs


def _get_pair_data(token_address: str) -> dict | None:
    """Fetch best pair data for a token from DexScreener."""
    data = _get(f"{DEX_BASE}/latest/dex/tokens/{token_address}")
    if not data or not data.get("pairs"):
        return None
    pairs = [p for p in data["pairs"] if p.get("chainId") == "solana"]
    if not pairs:
        return None
    pairs.sort(key=lambda p: (p.get("liquidity") or {}).get("usd", 0) or 0, reverse=True)
    return pairs[0]


def _upsert_token(token_address: str, pair: dict, now: int):
    """Insert new token or update existing one."""
    liq     = (pair.get("liquidity") or {}).get("usd", 0) or 0
    vol24   = (pair.get("volume") or {}).get("h24", 0) or 0
    price   = float(pair.get("priceUsd") or 0)
    created = pair.get("pairCreatedAt")
    launch_ts = int(created / 1000) if created else now

    if price <= 0:
        return

    conn = get_conn()

    # Check if already tracked
    existing = conn.execute(
        "SELECT launch_price, peak_price, peak_multiple FROM token_outcomes WHERE token_address = ? AND chain = 'solana'",
        (token_address,),
    ).fetchone()

    if not existing:
        # New token — use current price as launch price
        conn.execute(
            """
            INSERT INTO token_outcomes
                (token_address, chain, launch_time, launch_price, peak_price,
                 peak_time, peak_multiple, current_price, status, last_updated)
            VALUES (?, 'solana', ?, ?, ?, ?, 1.0, ?, 'active', ?)
            ON CONFLICT (token_address, chain) DO NOTHING
            """,
            (token_address, launch_ts, price, price, now, price, now),
        )
        log.debug("New token tracked: %s @ $%.8g", token_address[:8], price)
    else:
        launch_price = existing["launch_price"] or price
        old_peak     = existing["peak_price"] or price
        new_peak     = max(old_peak, price)
        peak_multiple = round(new_peak / launch_price, 2) if launch_price > 0 else 1.0
        peak_time    = now if price >= old_peak else None

        # Determine status
        if (price < old_peak * RUG_PRICE_RATIO and liq < RUG_LIQUIDITY):
            status = "rugged"
        elif peak_multiple >= MIN_PEAK_MULTIPLE:
            status = "winner"
        else:
            status = "active"

        update_sql = """
            UPDATE token_outcomes SET
                current_price = ?,
                peak_price    = ?,
                peak_multiple = ?,
                status        = ?,
                last_updated  = ?
        """
        params = [price, new_peak, peak_multiple, status, now]

        if peak_time:
            update_sql += ", peak_time = ?"
            params.append(peak_time)

        update_sql += " WHERE token_address = ? AND chain = 'solana'"
        params += [token_address]

        conn.execute(update_sql, params)

        if status == "winner" and peak_multiple >= MIN_PEAK_MULTIPLE:
            log.info("WINNER: %s  %.1fx  liq=$%.0f  vol24=$%.0f",
                     token_address[:8], peak_multiple, liq, vol24)

    conn.commit()
    conn.close()


def run():
    init_db()
    now    = int(time.time())
    tokens = _fetch_sources()

    # Deduplicate addresses
    seen    = set()
    unique  = []
    for t in tokens:
        addr = t.get("tokenAddress", "")
        if addr and addr not in seen:
            seen.add(addr)
            unique.append(addr)

    log.info("Phase 2b — tracking %d unique tokens", len(unique))
    updated = winners = 0

    for addr in unique:
        pair = _get_pair_data(addr)
        if not pair:
            time.sleep(CALL_DELAY)
            continue

        liq   = (pair.get("liquidity") or {}).get("usd", 0) or 0
        vol24 = (pair.get("volume") or {}).get("h24", 0) or 0

        # Also add tokens already pumped a lot on first sight
        # (priceChange.h24 >= 400% means ~5x in last day)
        pc24 = float((pair.get("priceChange") or {}).get("h24") or 0)
        if liq < MIN_LIQUIDITY and pc24 < 300:
            time.sleep(CALL_DELAY)
            continue

        _upsert_token(addr, pair, now)
        updated += 1
        time.sleep(CALL_DELAY)

    # Count winners
    conn = get_conn()
    winners = conn.execute(
        "SELECT COUNT(*) as n FROM token_outcomes WHERE status='winner' AND chain='solana'"
    ).fetchone()["n"]
    total = conn.execute(
        "SELECT COUNT(*) as n FROM token_outcomes WHERE chain='solana'"
    ).fetchone()["n"]
    conn.close()

    log.info("Phase 2b complete — %d tokens updated, %d/%d winners in DB",
             updated, winners, total)


if __name__ == "__main__":
    run()
