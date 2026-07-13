"""
Authoritative mint token-program and extension classifier.
Part 3 of G-batch spec.

Determines token program from mint account owner via RPC — never from:
  - mint address suffix ("pump" suffix does NOT mean SPL)
  - token symbol, dex_id, PP visibility, migration state, notes string

Policy categories:
  1. SPL — classic token program owner
  2. T22_CLEAN — Token-2022, no blocking extensions
  3. T22_TRANSFER_FEE — Token-2022 with TransferFeeConfig
  4. T22_TRANSFER_HOOK — Token-2022 with TransferHook (unsupported unless required accounts handled)
  5. T22_UNKNOWN_EXT — Token-2022 with unrecognized extensions
  6. UNKNOWN — owner unrecognized or RPC unavailable

Thread-safe, cached per mint. Extensions classified separately from token program.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

# ── Program IDs ──────────────────────────────────────────────────────────────

SPL_TOKEN_PROGRAM_ID    = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM_ID   = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
CLASSIFICATION_VERSION  = "1"

# ── Known extension type names ────────────────────────────────────────────────
_EXT_TRANSFER_FEE      = "TransferFeeConfig"
_EXT_TRANSFER_HOOK     = "TransferHook"
_EXT_INTEREST_BEARING  = "InterestBearingConfig"
_EXT_METADATA          = "TokenMetadata"
_EXT_METADATA_POINTER  = "MetadataPointer"
_EXT_MINT_CLOSE_AUTH   = "MintCloseAuthority"
_EXT_PERMANENT_DELEGATE = "PermanentDelegate"
_EXT_CONFIDENTIAL      = "ConfidentialTransferMint"

# Known-safe extensions (explicit allowlist — anything else → UNKNOWN_EXTENSION policy)
_KNOWN_EXTENSIONS = frozenset({
    _EXT_TRANSFER_FEE,
    _EXT_METADATA,
    _EXT_METADATA_POINTER,
    _EXT_MINT_CLOSE_AUTH,
    _EXT_INTEREST_BEARING,
    _EXT_PERMANENT_DELEGATE,
    _EXT_TRANSFER_HOOK,      # known but blocking
    _EXT_CONFIDENTIAL,       # known but blocking
})

# Extensions that block live trading (unsupported execution paths)
_BLOCKING_EXTENSIONS = frozenset({
    _EXT_TRANSFER_HOOK,
    _EXT_CONFIDENTIAL,
})

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class MintClassification:
    mint: str
    mint_owner_program: str           # raw owner pubkey from RPC
    token_program: str                 # "SPL" | "T22" | "UNKNOWN"
    token_extensions: list[str]        # list of extension type names
    unsupported_extensions: list[str]  # subset that block trading
    transfer_hook_present: bool
    transfer_fee_present: bool
    policy_category: str               # see module docstring
    detection_source: str              # "rpc_live" | "cache" | "fallback"
    detection_timestamp_wall: float
    detection_timestamp_monotonic: float
    rpc_commitment: str
    classification_version: str = CLASSIFICATION_VERSION
    error: Optional[str] = None        # set when classification failed

    @property
    def is_spl(self) -> bool:
        return self.token_program == "SPL"

    @property
    def is_t22(self) -> bool:
        return self.token_program == "T22"

    @property
    def is_unknown(self) -> bool:
        return self.token_program == "UNKNOWN"

    @property
    def has_blocking_extension(self) -> bool:
        return bool(self.unsupported_extensions)

    @property
    def is_tradeable(self) -> bool:
        """True if token program and extensions are supported for live trading."""
        return self.token_program in ("SPL", "T22") and not self.has_blocking_extension

    def to_dict(self) -> dict:
        return {
            "mint": self.mint,
            "mint_owner_program": self.mint_owner_program,
            "token_program": self.token_program,
            "token_extensions": self.token_extensions,
            "unsupported_extensions": self.unsupported_extensions,
            "transfer_hook_present": self.transfer_hook_present,
            "transfer_fee_present": self.transfer_fee_present,
            "policy_category": self.policy_category,
            "detection_source": self.detection_source,
            "detection_timestamp_wall": self.detection_timestamp_wall,
            "detection_timestamp_monotonic": self.detection_timestamp_monotonic,
            "rpc_commitment": self.rpc_commitment,
            "classification_version": self.classification_version,
            "error": self.error,
        }


# ── Cache ─────────────────────────────────────────────────────────────────────

_cache: dict[str, tuple[MintClassification, float]] = {}  # (result, cache_until)
_cache_lock = threading.Lock()

# B6(a): Definitive SPL/T22 results cached permanently (token program never changes).
# UNKNOWN or error results get 60s TTL so transient RPC failures are retried.


def _get_cached(mint: str) -> Optional[MintClassification]:
    with _cache_lock:
        entry = _cache.get(mint)
        if entry is None:
            return None
        result, cache_until = entry
        if cache_until == float("inf") or time.time() < cache_until:
            return result
        # TTL expired (non-definitive result)
        del _cache[mint]
        return None


def _put_cached(mint: str, result: MintClassification) -> None:
    with _cache_lock:
        # Definitive SPL/T22 results: permanent cache (token program never changes).
        # UNKNOWN or error: 60s TTL (retry later — may be RPC transient).
        if result.token_program in ("SPL", "T22") and not result.error:
            cache_until = float("inf")
        else:
            cache_until = time.time() + 60.0
        _cache[mint] = (result, cache_until)


def clear_cache() -> None:
    """Clear the classification cache. For testing only."""
    with _cache_lock:
        _cache.clear()


# ── Extension parsing ─────────────────────────────────────────────────────────

def _parse_extensions(account_data: dict) -> list[str]:
    """
    Extract extension type names from a Token-2022 mint account data structure.
    Handles both 'parsed' and 'base64' account data formats from Solana RPC.
    Returns list of extension type name strings.
    """
    extensions: list[str] = []
    try:
        # Parsed format: data.parsed.info.extensions
        parsed = account_data.get("parsed", {})
        info = parsed.get("info", {})
        raw_exts = info.get("extensions", [])
        if isinstance(raw_exts, list):
            for ext in raw_exts:
                if isinstance(ext, dict):
                    ext_type = ext.get("extension") or ext.get("type") or ext.get("extensionType")
                    if ext_type:
                        extensions.append(str(ext_type))
    except Exception as exc:
        log.debug("_parse_extensions: failed to parse extensions: %s", exc)
    return extensions


def _classify_policy(token_program: str, extensions: list[str]) -> str:
    """Map token program + extensions to policy category string."""
    if token_program == "UNKNOWN":
        return "6_UNKNOWN_token_program"
    if token_program == "SPL":
        return "1_SPL_supported"
    # T22
    if _EXT_TRANSFER_HOOK in extensions:
        return "4_T22_transfer_hook_unsupported"
    if _EXT_CONFIDENTIAL in extensions:
        return "5_T22_unknown_extension"
    if _EXT_TRANSFER_FEE in extensions:
        return "3_T22_transfer_fee"
    # Clean T22 (no blocking extensions)
    return "2_T22_clean"


# ── RPC fetch ─────────────────────────────────────────────────────────────────

_CLASSIFY_FALLBACK_RPCS = [
    "https://api.mainnet-beta.solana.com",
    "https://rpc.ankr.com/solana",
]


def _fetch_mint_owner(mint: str) -> dict:
    """
    Fetch mint account info from Solana RPC with fallback chain.

    Tries Helius first; on 429/5xx/timeout falls through to public RPC
    endpoints so a rate-limit never blocks the entry gate.
    Returns dict with keys: owner (str), data (dict), error (str|None).
    """
    import os, requests

    helius_key = os.environ.get("HELIUS_API_KEY", "")
    primary_url = (
        f"https://mainnet.helius-rpc.com/?api-key={helius_key}"
        if helius_key
        else _CLASSIFY_FALLBACK_RPCS[0]
    )
    urls_to_try = [primary_url] + [u for u in _CLASSIFY_FALLBACK_RPCS if u != primary_url]

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getAccountInfo",
        "params": [mint, {"encoding": "jsonParsed", "commitment": "confirmed"}],
    }

    last_error = None
    for url in urls_to_try:
        try:
            resp = requests.post(url, json=payload, timeout=5)
            if resp.status_code in (429, 503):
                last_error = f"rpc_429_or_503 url={url}"
                log.debug("MINT CLASSIFY 429/503 from %s — trying next RPC", url)
                continue
            resp.raise_for_status()
            body  = resp.json()
            value = (body.get("result") or {}).get("value")
            if value is None:
                return {"owner": None, "data": {}, "error": "account_not_found"}
            owner = value.get("owner", "")
            data  = value.get("data", {})
            if isinstance(data, list):
                data = {}
            return {"owner": owner, "data": data, "error": None, "commitment": "confirmed"}
        except Exception as exc:
            last_error = str(exc)
            log.debug("MINT CLASSIFY RPC error (%s): %s — trying next", url, exc)

    return {"owner": None, "data": {}, "error": last_error, "commitment": "confirmed"}


# ── Public API ────────────────────────────────────────────────────────────────

def classify_mint(mint: str, force_refresh: bool = False) -> MintClassification:
    """
    Authoritatively classify a mint's token program and extensions.

    Token program is determined solely from the mint account owner on-chain.
    Suffix heuristics, notes strings, and dex_id are never used.

    Results are cached permanently per mint (token program never changes for a mint).
    Pass force_refresh=True only in tests or when the cache entry is known stale.
    """
    if not force_refresh:
        cached = _get_cached(mint)
        if cached is not None:
            log.debug(
                "MINT CLASSIFY cache=%s  mint=%s  program=%s",
                "hit", mint[:8], cached.token_program,
            )
            return MintClassification(
                mint=cached.mint,
                mint_owner_program=cached.mint_owner_program,
                token_program=cached.token_program,
                token_extensions=cached.token_extensions,
                unsupported_extensions=cached.unsupported_extensions,
                transfer_hook_present=cached.transfer_hook_present,
                transfer_fee_present=cached.transfer_fee_present,
                policy_category=cached.policy_category,
                detection_source="cache",
                detection_timestamp_wall=time.time(),
                detection_timestamp_monotonic=time.monotonic(),
                rpc_commitment="confirmed",
                error=cached.error,
            )

    t_wall = time.time()
    t_mono = time.monotonic()

    rpc_result = _fetch_mint_owner(mint)
    owner = rpc_result.get("owner") or ""
    data  = rpc_result.get("data") or {}
    error = rpc_result.get("error")

    # Determine token program from owner
    if owner == SPL_TOKEN_PROGRAM_ID:
        token_program = "SPL"
    elif owner == TOKEN_2022_PROGRAM_ID:
        token_program = "T22"
    elif error:
        token_program = "UNKNOWN"
    else:
        token_program = "UNKNOWN"

    # Parse extensions (only meaningful for T22)
    extensions: list[str] = []
    if token_program == "T22" and isinstance(data, dict):
        extensions = _parse_extensions(data)

    # B6(b): unknown extension check — unrecognized extensions are not tradeable
    unknown_exts = [e for e in extensions if e not in _KNOWN_EXTENSIONS]
    if unknown_exts:
        # Unrecognized extension → policy 5 (UNKNOWN_EXTENSION), not tradeable
        policy = "5_T22_unknown_extension"
        unsupported_unk = unknown_exts
        result = MintClassification(
            mint=mint,
            mint_owner_program=owner,
            token_program=token_program,
            token_extensions=extensions,
            unsupported_extensions=unsupported_unk,
            transfer_hook_present=_EXT_TRANSFER_HOOK in extensions,
            transfer_fee_present=_EXT_TRANSFER_FEE in extensions,
            policy_category=policy,
            detection_source="rpc_live",
            detection_timestamp_wall=t_wall,
            detection_timestamp_monotonic=t_mono,
            rpc_commitment="confirmed",
            error=f"unknown_extension:{','.join(unknown_exts)}",
        )
        log.warning(
            "MINT CLASSIFY unrecognized extension  mint=%s  exts=%s  policy=%s  tradeable=False",
            mint[:8], unknown_exts, policy,
        )
        _put_cached(mint, result)
        return result

    unsupported = [e for e in extensions if e in _BLOCKING_EXTENSIONS]
    policy = _classify_policy(token_program, extensions)

    result = MintClassification(
        mint=mint,
        mint_owner_program=owner,
        token_program=token_program,
        token_extensions=extensions,
        unsupported_extensions=unsupported,
        transfer_hook_present=_EXT_TRANSFER_HOOK in extensions,
        transfer_fee_present=_EXT_TRANSFER_FEE in extensions,
        policy_category=policy,
        detection_source="rpc_live",
        detection_timestamp_wall=t_wall,
        detection_timestamp_monotonic=t_mono,
        rpc_commitment="confirmed",
        error=error,
    )

    log.info(
        "ENTRY PROGRAM GATE  mint=%s  program=%s  extensions=%s  "
        "policy=%s  tradeable=%s  error=%s",
        mint[:8], token_program, extensions or "[]", policy,
        result.is_tradeable, error or "none",
    )

    _put_cached(mint, result)
    return result


def get_token_program(mint: str) -> str:
    """Convenience: return 'SPL', 'T22', or 'UNKNOWN'. Cached."""
    return classify_mint(mint).token_program


def is_tradeable(mint: str) -> bool:
    """Convenience: True if SPL or clean T22. Cached."""
    return classify_mint(mint).is_tradeable
