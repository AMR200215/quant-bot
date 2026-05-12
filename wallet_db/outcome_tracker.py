"""
Phase 2b — Token outcome tracker.

Sources tokens exclusively from whale wallet trades (Source A).
DexScreener boosted/profile lists removed — they are marketing-driven and
unconnected to whale activity.

Design:
  1. Query wallet_trades for tokens bought by S/A/B-tier wallets in last 30d.
  2. For any not yet in token_outcomes, fetch current DexScreener price and
     insert with that price as launch_price (proxy for whale entry — happens
     within one ingest cycle, i.e. ≤10 min after the buy).
  3. Update current_price / peak_price / peak_multiple for all active tokens.
  4. Flag winners (peak_multiple >= 5) and rugged tokens.

Sources B (pump.fun launch feed) and C (PancakeSwap PairCreated events) are
deferred — they require a persistent daemon on Hetzner. Revisit after Source A
is confirmed producing discovery candidates.

Run:
    python3 -m wallet_db.outcome_tracker
"""

import logging
import os
import time

import requests
from dotenv import load_dotenv

from wallet_db.db import get_conn, init_db, _USE_POSTGRES

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEX_BASE   = "https://api.dexscreener.com"
CALL_DELAY = 0.3

MIN_PEAK_MULTIPLE = 5.0
RUG_PRICE_RATIO   = 0.10
RUG_LIQUIDITY     = 5_000
MAX_AGE_DAYS      = 14
LOOKBACK_DAYS     = 30

_PH = "%s" if _USE_POSTGRES else "?"


def _get(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.debug("GET %s failed: %s", url, e)
        return None


def _get_pair_data(token_address: str) -> dict | None:
    """Fetch best liquidity Solana pair from DexScreener."""
    data = _get(f"{DEX_BASE}/latest/dex/tokens/{token_address}")
    if not data or not data.get("pairs"):
        return None
    pairs = [p for p in data["pairs"] if p.get("chainId") == "solana"]
    if not pairs:
        return None
    pairs.sort(key=lambda p: (p.get("liquidity") or {}).get("usd", 0) or 0, reverse=True)
    return pairs[0]


# ---------------------------------------------------------------------------
# Source A — whale-bought tokens
# ---------------------------------------------------------------------------

def _get_whale_bought_tokens() -> list[dict]:
    """
    Return distinct tokens bought by S/A/B-tier wallets in the last LOOKBACK_DAYS.
    Each row: {token_address, chain, first_buy_ts}
    """
    since = int(time.time()) - LOOKBACK_DAYS * 86400
    conn  = get_conn()
    rows  = conn.execute(
        f"""
        SELECT wt.token_address, wt.chain, MIN(wt.block_time) AS first_buy_ts
        FROM wallet_trades wt
        JOIN wallets w ON wt.wallet_address = w.address AND wt.chain = w.chain
        WHERE wt.side = 'buy'
          AND wt.block_time >= {_PH}
          AND w.current_tier IN ({_PH}, {_PH}, {_PH})
        GROUP BY wt.token_address, wt.chain
        """,
        (since, "S", "A", "B"),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _add_new_token(token_address: str, chain: str, first_buy_ts: int, now: int) -> bool:
    """
    Insert a token not yet tracked. launch_price = DexScreener price at time of
    first tracking (within one ingest cycle of the actual whale buy).
    """
    pair = _get_pair_data(token_address)
    time.sleep(CALL_DELAY)
    if not pair:
        log.debug("No DexScreener data for %s — skipping", token_address[:8])
        return False

    price = float(pair.get("priceUsd") or 0)
    if price <= 0:
        return False

    conn = get_conn()
    conn.execute(
        f"""
        INSERT INTO token_outcomes
            (token_address, chain, launch_time, launch_price, peak_price,
             peak_time, peak_multiple, current_price, status, last_updated)
        VALUES ({_PH}, {_PH}, {_PH}, {_PH}, {_PH}, {_PH}, 1.0, {_PH}, 'active', {_PH})
        ON CONFLICT (token_address, chain) DO NOTHING
        """,
        (token_address, chain, first_buy_ts, price, price, now, price, now),
    )
    conn.commit()
    conn.close()
    log.info("Tracking new token (whale buy): %s @ $%.8g", token_address[:8], price)
    return True


# ---------------------------------------------------------------------------
# Periodic update — all active tokens
# ---------------------------------------------------------------------------

def _update_active_tokens(now: int) -> tuple[int, int]:
    """
    Refresh current_price / peak / multiple for all active tokens younger than
    MAX_AGE_DAYS. Returns (updated_count, winner_count).
    """
    cutoff = now - MAX_AGE_DAYS * 86400
    conn   = get_conn()
    rows   = conn.execute(
        f"""
        SELECT token_address, chain, launch_price, peak_price, peak_time
        FROM token_outcomes
        WHERE status   = 'active'
          AND launch_time >= {_PH}
          AND chain    = 'solana'
        """,
        (cutoff,),
    ).fetchall()
    conn.close()

    updated = winners = 0

    for row in rows:
        token_address = row["token_address"]
        launch_price  = row["launch_price"] or 0
        old_peak      = row["peak_price"] or 0

        pair = _get_pair_data(token_address)
        time.sleep(CALL_DELAY)
        if not pair:
            continue

        price = float(pair.get("priceUsd") or 0)
        liq   = (pair.get("liquidity") or {}).get("usd", 0) or 0
        vol24 = (pair.get("volume") or {}).get("h24", 0) or 0

        if price <= 0 or launch_price <= 0:
            continue

        new_peak      = max(old_peak, price)
        peak_multiple = round(new_peak / launch_price, 2)
        new_peak_time = now if price >= old_peak else None

        if price < old_peak * RUG_PRICE_RATIO and liq < RUG_LIQUIDITY:
            status = "rugged"
        elif peak_multiple >= MIN_PEAK_MULTIPLE:
            status = "winner"
        else:
            status = "active"

        sql    = f"""
            UPDATE token_outcomes SET
                current_price = {_PH},
                peak_price    = {_PH},
                peak_multiple = {_PH},
                status        = {_PH},
                last_updated  = {_PH}
        """
        params = [price, new_peak, peak_multiple, status, now]

        if new_peak_time:
            sql   += f", peak_time = {_PH}"
            params.append(new_peak_time)

        sql    += f" WHERE token_address = {_PH} AND chain = {_PH}"
        params += [token_address, row["chain"]]

        conn = get_conn()
        conn.execute(sql, params)
        conn.commit()
        conn.close()

        updated += 1
        if status == "winner":
            winners += 1
            log.info("WINNER: %s  %.1fx  liq=$%.0f  vol24=$%.0f",
                     token_address[:8], peak_multiple, liq, vol24)

    return updated, winners


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    init_db()
    now = int(time.time())

    # Step 1: add newly whale-bought tokens not yet tracked
    whale_tokens = _get_whale_bought_tokens()
    log.info("Phase 2b — %d whale-bought tokens in last %dd", len(whale_tokens), LOOKBACK_DAYS)

    new_added = 0
    for t in whale_tokens:
        conn     = get_conn()
        existing = conn.execute(
            f"SELECT 1 FROM token_outcomes WHERE token_address = {_PH} AND chain = {_PH}",
            (t["token_address"], t["chain"]),
        ).fetchone()
        conn.close()

        if not existing:
            if _add_new_token(t["token_address"], t["chain"], t["first_buy_ts"], now):
                new_added += 1

    log.info("New tokens added: %d", new_added)

    # Step 2: update active tokens
    updated, winners = _update_active_tokens(now)

    conn  = get_conn()
    total = conn.execute(
        f"SELECT COUNT(*) AS n FROM token_outcomes WHERE chain = 'solana'"
    ).fetchone()["n"]
    conn.close()

    log.info("Phase 2b complete — %d tokens updated, %d/%d winners in DB",
             updated, winners, total)


if __name__ == "__main__":
    run()
