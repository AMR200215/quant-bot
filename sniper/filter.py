"""
Fast pre-filter for snipe candidates.

Designed to run in < 1 second — no rugcheck.xyz, no honeypot.is
(those take 2-5s each and would kill snipe timing).

Checks:
  1. Name keyword blacklist (obvious rugs)
  2. Initial buy size (whale setup to dump)
  3. Deployer token count via on-chain history (serial rugger detection)
  4. Migration: minimum mcap proof
"""

import logging
import time
from dataclasses import dataclass

import requests

from sniper.config import (
    RUG_NAME_KEYWORDS, MAX_INITIAL_BUY_PCT, MAX_DEPLOYER_TOKENS,
    MIN_INITIAL_SOL, MIN_MIGRATION_MCAP_SOL,
)
from sniper.listener import PumpEvent

log = logging.getLogger(__name__)

# In-memory deployer token count cache { wallet: (count, timestamp) }
_deployer_cache: dict[str, tuple[int, float]] = {}
_CACHE_TTL = 3600   # 1 hour


@dataclass
class FilterResult:
    passed: bool
    reason: str = ""


def fast_filter(event: PumpEvent) -> FilterResult:
    """
    Run all fast pre-filters on a PumpEvent.
    Returns FilterResult(passed=True) if the token should be considered.
    """

    # 1. Name blacklist
    name_lower = (event.name + " " + event.symbol).lower()
    for kw in RUG_NAME_KEYWORDS:
        if kw in name_lower:
            return FilterResult(False, f"name_keyword:{kw}")

    # 2. Minimum initial SOL (deployer put real money in)
    if event.initial_buy_sol < MIN_INITIAL_SOL and event.event_type == "new_token":
        return FilterResult(False, f"low_initial_sol:{event.initial_buy_sol:.3f}")

    # 3. Initial buy % of bonding curve (whale dump setup)
    #    Pump.fun bonding curve starts with ~1B tokens
    #    If initial buy > 40% of supply, deployer is positioned to dump on buyers
    BONDING_CURVE_SUPPLY = 1_000_000_000
    initial_token_pct = 0.0
    if event.initial_buy_sol > 0 and BONDING_CURVE_SUPPLY > 0:
        # Rough estimate: 1 SOL ≈ 28.4M tokens at launch price on Pump.fun
        approx_tokens_bought = event.initial_buy_sol * 28_400_000
        initial_token_pct = approx_tokens_bought / BONDING_CURVE_SUPPLY
    if initial_token_pct > MAX_INITIAL_BUY_PCT:
        return FilterResult(False, f"large_initial_buy:{initial_token_pct:.0%}")

    # 4. Migration: mcap must meet minimum proof of demand
    if event.event_type == "migration":
        if event.market_cap_sol < MIN_MIGRATION_MCAP_SOL:
            return FilterResult(False, f"low_migration_mcap:{event.market_cap_sol:.0f}_SOL")

    # 5. Deployer serial rugger check (cached)
    deployer_count = _get_deployer_token_count(event.trader)
    if deployer_count > MAX_DEPLOYER_TOKENS:
        return FilterResult(False, f"serial_deployer:{deployer_count}_tokens")

    return FilterResult(True)


def _get_deployer_token_count(wallet: str) -> int:
    """
    Estimate how many tokens this wallet has deployed recently.
    Uses DexScreener search as a cheap proxy — no auth needed.
    Returns 0 on error (fail open, let screener catch it).
    """
    if not wallet:
        return 0

    cached = _deployer_cache.get(wallet)
    if cached and (time.time() - cached[1]) < _CACHE_TTL:
        return cached[0]

    try:
        url = f"https://api.dexscreener.com/latest/dex/search?q={wallet}"
        resp = requests.get(url, timeout=1.5)
        if resp.status_code == 200:
            data = resp.json()
            pairs = data.get("pairs") or []
            # count unique token addresses where this wallet is referenced
            count = len({p.get("baseToken", {}).get("address", "") for p in pairs
                        if p.get("baseToken", {}).get("address")})
            _deployer_cache[wallet] = (count, time.time())
            return count
    except Exception as e:
        log.debug("Deployer check failed for %s: %s", wallet[:8], e)

    return 0
