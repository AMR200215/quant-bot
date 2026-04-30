"""
Safety screening for tokens.
Every candidate token must pass all checks before a signal is generated.
Captures all DexScreener fields needed for model training.
"""

import logging
import time
from typing import Optional

from memecoin.config import (
    MIN_LIQUIDITY_USD, MAX_MCAP_USD, MIN_HOLDERS, CHAINS,
)
from memecoin.data_client import (
    dex_get_token, rugcheck_sol, honeypot_bsc,
)
from memecoin.rug_detector import run_rug_checks, RugReport

log = logging.getLogger(__name__)


def screen_token(chain: str, token_address: str) -> dict:
    """
    Run full safety screening on a token.
    Returns a rich dict capturing everything from DexScreener + safety APIs
    that will later feed the model.
    """
    result = {
        # meta
        "passed": False,
        "reason": "",
        "pair": None,
        "safety": None,
        # price
        "price_usd": 0.0,
        "price_change_5m": 0.0,
        "price_change_1h": 0.0,
        "price_change_6h": 0.0,
        "price_change_24h": 0.0,
        # liquidity / market
        "liquidity_usd": 0.0,
        "mcap_usd": 0.0,
        "fdv": 0.0,
        # volume
        "volume_5m": 0.0,
        "volume_h1": 0.0,
        "volume_h6": 0.0,
        "volume_h24": 0.0,
        # transactions (buy/sell pressure)
        "buys_5m": 0,
        "sells_5m": 0,
        "buys_h1": 0,
        "sells_h1": 0,
        "buys_h6": 0,
        "sells_h6": 0,
        "buy_sell_ratio_5m": 0.0,   # buys / (buys + sells)
        "buy_sell_ratio_h1": 0.0,
        # token info
        "age_minutes": 9999.0,
        "holders": 0,
        "dex_id": "",               # pumpswap / raydium / pancakeswap
        "pair_address": "",
        "dexscreener_url": "",
        # socials (legitimacy signals)
        "has_twitter": False,
        "has_telegram": False,
        "has_website": False,
        # safety
        "rugcheck_score": None,     # Solana: 0-1000, lower = safer
        "buy_tax": None,            # BSC
        "sell_tax": None,           # BSC
        "is_honeypot": None,        # BSC
    }

    # ---- DexScreener pair data ----
    pair = dex_get_token(chain, token_address)
    if not pair:
        result["reason"] = "no_dex_data"
        return result

    result["pair"] = pair

    # Price
    result["price_usd"] = float(pair.get("priceUsd") or 0)

    # Price changes
    pc = pair.get("priceChange") or {}
    result["price_change_5m"]  = float(pc.get("m5")  or 0)
    result["price_change_1h"]  = float(pc.get("h1")  or 0)
    result["price_change_6h"]  = float(pc.get("h6")  or 0)
    result["price_change_24h"] = float(pc.get("h24") or 0)

    # Volume
    vol = pair.get("volume") or {}
    result["volume_5m"]  = float(vol.get("m5")  or 0)
    result["volume_h1"]  = float(vol.get("h1")  or 0)
    result["volume_h6"]  = float(vol.get("h6")  or 0)
    result["volume_h24"] = float(vol.get("h24") or 0)

    # Transactions
    txns = pair.get("txns") or {}
    m5  = txns.get("m5")  or {}
    h1  = txns.get("h1")  or {}
    h6  = txns.get("h6")  or {}
    result["buys_5m"]   = int(m5.get("buys")  or 0)
    result["sells_5m"]  = int(m5.get("sells") or 0)
    result["buys_h1"]   = int(h1.get("buys")  or 0)
    result["sells_h1"]  = int(h1.get("sells") or 0)
    result["buys_h6"]   = int(h6.get("buys")  or 0)
    result["sells_h6"]  = int(h6.get("sells") or 0)

    total_5m = result["buys_5m"] + result["sells_5m"]
    total_h1 = result["buys_h1"] + result["sells_h1"]
    result["buy_sell_ratio_5m"] = round(result["buys_5m"] / total_5m, 3) if total_5m else 0.0
    result["buy_sell_ratio_h1"] = round(result["buys_h1"] / total_h1, 3) if total_h1 else 0.0

    # Liquidity / market cap
    liq  = (pair.get("liquidity") or {}).get("usd") or 0
    mcap = pair.get("marketCap") or pair.get("fdv") or 0
    result["liquidity_usd"] = float(liq)
    result["mcap_usd"]      = float(mcap)
    result["fdv"]           = float(pair.get("fdv") or mcap)

    # DEX info & links
    result["dex_id"]          = pair.get("dexId", "")
    result["pair_address"]    = pair.get("pairAddress", "")
    result["dexscreener_url"] = pair.get("url", "")

    # Age
    created_ms = pair.get("pairCreatedAt")
    if created_ms:
        result["age_minutes"] = (time.time() - created_ms / 1000) / 60

    # Socials
    info     = pair.get("info") or {}
    socials  = info.get("socials") or []
    websites = info.get("websites") or []
    social_types = {s.get("type", "").lower() for s in socials}
    result["has_twitter"]  = "twitter" in social_types
    result["has_telegram"] = "telegram" in social_types
    result["has_website"]  = len(websites) > 0

    # ---- Basic numeric filters ----
    if result["liquidity_usd"] < MIN_LIQUIDITY_USD:
        result["reason"] = f"low_liquidity:{result['liquidity_usd']:.0f}"
        return result

    if 0 < result["mcap_usd"] > MAX_MCAP_USD:
        result["reason"] = f"mcap_too_high:{result['mcap_usd']:.0f}"
        return result

    # ---- Chain-specific safety ----
    if chain == "solana":
        safety = rugcheck_sol(token_address)
        result["safety"] = safety
        if safety is not None:
            result["rugcheck_score"] = safety.get("score")
            if not safety["is_safe"]:
                risky = ", ".join(safety["risks"][:3])
                result["reason"] = f"rugcheck_fail:{risky}"
                return result

    elif chain == "bsc":
        safety = honeypot_bsc(token_address)
        result["safety"] = safety
        if safety is not None:
            result["is_honeypot"] = safety.get("is_honeypot")
            result["buy_tax"]     = safety.get("buy_tax")
            result["sell_tax"]    = safety.get("sell_tax")
            if not safety["is_safe"]:
                result["reason"] = (
                    f"honeypot:{safety.get('is_honeypot')} "
                    f"sell_tax:{safety.get('sell_tax')}%"
                )
                return result

    # ---- Advanced rug detection ----
    rpc_url = CHAINS.get(chain, {}).get("rpc", "https://api.mainnet-beta.solana.com")
    rug_report: RugReport = run_rug_checks(
        screen=result,
        chain=chain,
        token_address=token_address,
        rpc_url=rpc_url,
        check_holders=(chain == "solana"),
    )
    result["rug_report"] = rug_report
    result["rug_flags"] = [f.code for f in rug_report.flags]
    result["rug_summary"] = rug_report.summary()

    if not rug_report.safe_to_trade:
        result["reason"] = f"rug_detector:{rug_report.summary()}"
        return result

    result["passed"] = True
    return result


def compute_safety_score(screen: dict) -> float:
    """0.0 – 1.0 composite safety score. Higher = safer."""
    if not screen["passed"]:
        return 0.0

    score = 0.5  # base

    liq = screen["liquidity_usd"]
    if liq >= 50_000:
        score += 0.2
    elif liq >= 20_000:
        score += 0.1

    safety = screen.get("safety")
    if safety:
        if safety.get("is_safe"):
            score += 0.15
        if not safety.get("risks"):
            score += 0.05

    # buy pressure bonus
    if screen["buy_sell_ratio_h1"] >= 0.6:
        score += 0.1

    # social legitimacy
    if screen["has_twitter"] or screen["has_telegram"]:
        score += 0.05
    if screen["has_website"]:
        score += 0.05

    # penalise very new tokens slightly
    age = screen["age_minutes"]
    if age < 5:
        score -= 0.1
    elif age > 30:
        score += 0.05

    # penalise advanced rug flags
    from memecoin.rug_detector import CRITICAL, HIGH, MEDIUM
    rug_report = screen.get("rug_report")
    if rug_report:
        for flag in rug_report.flags:
            if flag.severity == CRITICAL:
                score -= 0.40
            elif flag.severity == HIGH:
                score -= 0.20
            elif flag.severity == MEDIUM:
                score -= 0.08

    return max(0.0, min(1.0, score))
