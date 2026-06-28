"""
Safety screening for tokens.
Every candidate token must pass all checks before a signal is generated.
Captures all DexScreener fields needed for model training.
"""

import logging
import struct
import time
from typing import Optional

import requests

from memecoin.config import (
    MIN_LIQUIDITY_USD, MAX_MCAP_USD, MIN_HOLDERS, CHAINS,
)
from memecoin.data_client import (
    dex_get_token, rugcheck_sol, honeypot_bsc,
)
from memecoin.rug_detector import run_rug_checks, RugReport

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# T22 Extension type IDs (u16 LE in mint data)
# ---------------------------------------------------------------------------
_TOKEN_PROGRAM_T22 = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"

_T22_EXTENSION_NAMES = {
    0:  "Uninitialized",
    1:  "TransferFeeConfig",
    2:  "TransferFeeAmount",
    3:  "MintCloseAuthority",
    4:  "ConfidentialTransferMint",
    5:  "ConfidentialTransferAccount",
    6:  "DefaultAccountState",
    7:  "ImmutableOwner",
    8:  "MemoTransfer",
    9:  "NonTransferable",
    10: "InterestBearingConfig",
    11: "CpiGuard",
    12: "PermanentDelegate",
    13: "NonTransferableAccount",
    14: "TransferHook",
    15: "TransferHookAccount",
    16: "ConfidentialTransferFeeConfig",
    17: "ConfidentialTransferFeeAmount",
    18: "MetadataPointer",
    19: "TokenMetadata",
    20: "GroupPointer",
    21: "TokenGroup",
    22: "GroupMemberPointer",
    23: "TokenGroupMember",
}

# Safe extensions — live buys allowed when only these are present
_T22_SAFE_EXTENSIONS = {1, 3, 7, 8, 12, 18, 19, 20, 21, 22, 23}

# Transfer hook type ID
_T22_TRANSFER_HOOK_ID = 14


def check_token_program(mint: str, rpc_url: str = "https://api.mainnet-beta.solana.com") -> dict:
    """
    Check if a token mint uses Token-2022 and classify its extensions.

    Returns dict with:
      is_token2022        bool
      has_transfer_hook   bool
      has_unknown_extensions bool
      extensions_list     list[str]   — extension names found
      live_blocked        bool        — True = paper-only (hard block)
      canary_only         bool        — True = canary-size buys only
      token_program       str         — program ID string
    """
    result = {
        "is_token2022":           False,
        "has_transfer_hook":      False,
        "has_unknown_extensions": False,
        "extensions_list":        [],
        "live_blocked":           False,
        "canary_only":            False,
        "token_program":          "",
    }
    try:
        resp = requests.post(rpc_url, json={
            "jsonrpc": "2.0", "id": 1,
            "method": "getAccountInfo",
            "params": [mint, {"encoding": "base64"}],
        }, timeout=5)
        data = resp.json().get("result", {}).get("value") or {}
        owner = data.get("owner", "")
        result["token_program"] = owner

        if owner != _TOKEN_PROGRAM_T22:
            return result  # SPL token — no T22 checks needed

        result["is_token2022"] = True

        # Parse extensions from mint account data (base64-encoded)
        import base64
        raw_b64 = (data.get("data") or [""])[0]
        mint_data = base64.b64decode(raw_b64) if raw_b64 else b""

        # T22 mint layout: 82 bytes fixed header, then extensions (type u16 LE + length u16 LE + data)
        extensions_found = []
        has_hook = False
        has_unknown = False
        offset = 82  # skip standard mint header
        while offset + 4 <= len(mint_data):
            ext_type = struct.unpack_from("<H", mint_data, offset)[0]
            ext_len  = struct.unpack_from("<H", mint_data, offset + 2)[0]
            if ext_type == 0:
                break  # Uninitialized — end of extension list
            name = _T22_EXTENSION_NAMES.get(ext_type, f"Unknown({ext_type})")
            extensions_found.append(name)
            if ext_type == _T22_TRANSFER_HOOK_ID:
                has_hook = True
            if ext_type not in _T22_SAFE_EXTENSIONS and ext_type != _T22_TRANSFER_HOOK_ID:
                has_unknown = True
            offset += 4 + ext_len

        result["extensions_list"]        = extensions_found
        result["has_transfer_hook"]      = has_hook
        result["has_unknown_extensions"] = has_unknown

        # Apply policy
        try:
            from memecoin.config import (
                BLOCK_T22_TRANSFER_HOOK, BLOCK_T22_UNKNOWN_EXTENSIONS,
                ALLOW_T22_LIVE_NORMAL, ALLOW_T22_CANARY,
            )
        except ImportError:
            BLOCK_T22_TRANSFER_HOOK      = True
            BLOCK_T22_UNKNOWN_EXTENSIONS = True
            ALLOW_T22_LIVE_NORMAL        = False
            ALLOW_T22_CANARY             = True

        if has_hook and BLOCK_T22_TRANSFER_HOOK:
            result["live_blocked"] = True
        elif has_unknown and BLOCK_T22_UNKNOWN_EXTENSIONS:
            result["live_blocked"] = True
        elif not ALLOW_T22_LIVE_NORMAL and ALLOW_T22_CANARY:
            result["canary_only"] = True

    except Exception as _e:
        log.debug("check_token_program error for %s: %s", mint[:8], _e)

    return result


def screen_token(
    chain: str,
    token_address: str,
    *,
    pair: Optional[dict] = None,
    safety: Optional[dict] = None,
) -> dict:
    """
    Run full safety screening on a token.
    Returns a rich dict capturing everything from DexScreener + safety APIs
    that will later feed the model.

    pair:   pre-fetched DexScreener pair dict (from parallel prefetch).
            When provided the internal dex_get_token() call is skipped.
            Pass None to fetch internally (default / fallback behavior).
    safety: pre-fetched rugcheck/honeypot dict.
            When provided the internal rugcheck_sol()/honeypot_bsc() call is skipped.
            Pass None to fetch internally (default / fallback behavior).
    """
    result = {
        # meta
        "passed": False,
        "reason": "",
        "pair": None,
        "safety": None,
        # T22 token program fields
        "is_token2022":      False,
        "has_transfer_hook": False,
        "token_extensions":  [],
        "live_blocked_t22":  False,
        "canary_only_t22":   False,
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
        "mint_disabled": None,      # Solana: True = mint authority revoked
        "freeze_disabled": None,    # Solana: True = freeze authority revoked
        "buy_tax": None,            # BSC
        "sell_tax": None,           # BSC
        "is_honeypot": None,        # BSC
    }

    # ---- DexScreener pair data ----
    # Use prefetched result when available; fall back to synchronous fetch.
    if pair is None:
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
    # Use prefetched result when available; fall back to synchronous fetch.
    if chain == "solana":
        if safety is None:
            safety = rugcheck_sol(token_address)
        result["safety"] = safety
        if safety is not None:
            result["rugcheck_score"]  = safety.get("score")
            result["mint_disabled"]   = safety.get("mint_disabled")
            result["freeze_disabled"] = safety.get("freeze_disabled")
            if not safety["is_safe"]:
                risky = ", ".join(safety["risks"][:3])
                result["reason"] = f"rugcheck_fail:{risky}"
                return result

    elif chain == "bsc":
        if safety is None:
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

    # ---- Creator / dev history check (Solana only) ----
    # Fetch creator wallet and look up in dev_wallets.json.
    # Serial ruggers (rug_count ≥ 2) are blocked outright.
    # Known winner devs get their score logged for downstream use.
    result["creator_wallet"] = ""
    result["dev_score"]      = 0.0
    if chain == "solana":
        try:
            from memecoin.data_client import sol_get_token_creator
            from memecoin.dev_tracker import is_known_dev, is_serial_rugger
            creator = sol_get_token_creator(token_address) or ""
            result["creator_wallet"] = creator
            if creator:
                if is_serial_rugger(chain, creator):
                    result["reason"] = f"serial_rugger:{creator[:8]}"
                    return result
                dev_entry = is_known_dev(chain, creator)
                if dev_entry:
                    ds = dev_entry.get("score", 0.0)
                    result["dev_score"] = ds
                    log.info("Known dev %s  token=%s  score=%.2f  wins=%d  rugs=%d",
                             creator[:8], token_address[:8], ds,
                             dev_entry.get("win_count", 0),
                             dev_entry.get("rug_count", 0))
        except Exception as _de:
            log.debug("Dev history check failed for %s: %s", token_address[:8], _de)

    # ---- Token-2022 extension screening (Solana only) ----
    if chain == "solana":
        try:
            rpc_url_t22 = CHAINS.get(chain, {}).get("rpc", "https://api.mainnet-beta.solana.com")
            t22_info = check_token_program(token_address, rpc_url=rpc_url_t22)
            result["is_token2022"]      = t22_info.get("is_token2022", False)
            result["has_transfer_hook"] = t22_info.get("has_transfer_hook", False)
            result["token_extensions"]  = t22_info.get("extensions_list", [])
            result["live_blocked_t22"]  = t22_info.get("live_blocked", False)
            result["canary_only_t22"]   = t22_info.get("canary_only", False)
            if t22_info.get("live_blocked"):
                log.info(
                    "T22 live blocked  token=%s  hook=%s  unknown_ext=%s  exts=%s",
                    token_address[:8],
                    t22_info.get("has_transfer_hook"),
                    t22_info.get("has_unknown_extensions"),
                    t22_info.get("extensions_list"),
                )
        except Exception as _t22e:
            log.debug("T22 check failed for %s: %s", token_address[:8], _t22e)

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
