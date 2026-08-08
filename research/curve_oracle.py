"""
curve_oracle.py — Pump.fun bonding-curve price oracle via Helius getMultipleAccounts.

Fetches bonding-curve account data directly from Solana RPC for tokens that are
still on the bonding curve (CURVE_ACTIVE).  Avoids the 30-90s DexScreener indexing
lag that causes massive NULL rates at T1m/T3m for BC tokens.

Bonding-curve account layout (base64 decoded, little-endian, pump.fun layout):
  Offset  8: virtualTokenReserves (u64)
  Offset 16: virtualSolReserves   (u64)
  Offset 24: realTokenReserves    (u64)
  Offset 32: realSolReserves      (u64)
  Offset 40: tokenTotalSupply     (u64)
  Offset 48: complete             (bool, 1 byte)

Price formula (FIXED 2026-08-07 — see PROGRESS-FIX PF0):
  vsol_sol            = virtualSolReserves / 1e9                 (lamports -> SOL)
  vtoken_ui            = virtualTokenReserves / PUMPFUN_TOKEN_DECIMALS_DIVISOR
                                                                  (raw base units -> UI tokens)
  price_sol_per_token  = vsol_sol / vtoken_ui
  price_usd            = price_sol_per_token * sol_price_usd
  vsol_ui              = virtualSolReserves / 1e9                (unchanged)

  Previous formula (price_sol = virtualSolReserves / virtualTokenReserves;
  price_usd = price_sol * sol_price_usd / 1e9) never converted
  virtualTokenReserves from raw base units to UI units (pump.fun tokens use
  6 decimals) — it silently underpriced every curve_account-sourced price by
  exactly 1,000,000x. Fixture check (docs/PUMPFUN_COMPATIBILITY_REPORT.md
  sample): virtual_token_reserves=1,063,494,656,015,142,
  virtual_sol_reserves=3,107,652,233 -> old formula ~$4.38e-13/token,
  correct formula ~$4.38e-7/token. See RECEIPTS.md PROGRESS-FIX PF0 for the
  historical-row contamination audit; ratio-based metrics computed entirely
  from curve_account-to-curve_account comparisons are unaffected (the
  constant scaling factor cancels), but any absolute price/mcap value, or a
  ratio mixing curve_account with another price source, is not.

Never treat RPC errors or parse failures as graduation.
complete=True in the curve data is the ONLY valid graduation signal here.
"""

import base64
import logging
import struct
import threading
import time
from typing import Optional

import requests

from research.config import (
    SOL_USD_MAX_CACHE_AGE_S,
    CURVE_BATCH_SIZE,
)

log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

PUMP_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

# PROGRESS-FIX PF3 follow-up (2026-08-08): Helius is on the free/rate-limited
# plan (see CLAUDE.md) and getMultipleAccounts was 429ing on nearly every
# call, silently starving progress_capture's curve_account source (observed
# in production: 61/62 fresh captures fell through to pp_post_alert/timeout,
# zero succeeded via curve_account). Same two-tier public fallback pattern
# memecoin/executor.py already uses for its own RPC calls — read-only
# getMultipleAccounts, no wallet/signing involved, so reusing public RPCs
# here carries no execution risk.
_RPC_FALLBACK_1 = "https://api.mainnet-beta.solana.com"
_RPC_FALLBACK_2 = "https://rpc.ankr.com/solana"

# Bonding-curve account layout offsets (after 8-byte discriminator)
_OFFSET_VIRTUAL_TOKEN  =  8
_OFFSET_VIRTUAL_SOL    = 16
_OFFSET_REAL_TOKEN     = 24
_OFFSET_REAL_SOL       = 32
_OFFSET_TOTAL_SUPPLY   = 40
_OFFSET_COMPLETE       = 48
_MIN_ACCOUNT_DATA_LEN  = 49   # must have at least 49 bytes (offsets 0-48)

# Pump.fun token decimals are always 6
PUMP_DECIMALS = 6

# ── SOL/USD cache (module-level, thread-safe) ─────────────────────────────────

_sol_usd_lock   = threading.Lock()
_sol_usd_price  = 0.0
_sol_usd_fetched_at = 0.0   # epoch seconds

_JUPITER_SOL_MINT = "So11111111111111111111111111111111111111112"
_JUP_PRICE_URL    = f"https://api.jup.ag/price/v2?ids={_JUPITER_SOL_MINT}"
_JUP_TIMEOUT_S    = 5


def get_sol_usd_cached() -> tuple[float, float]:
    """
    Returns (price_usd, age_seconds).  Thread-safe.
    Fetches from Jupiter Price API if cache is stale or empty.
    Returns (0.0, very_large) if fetch fails.
    """
    global _sol_usd_price, _sol_usd_fetched_at

    with _sol_usd_lock:
        age = time.time() - _sol_usd_fetched_at
        if _sol_usd_price > 0 and age < 60:
            return _sol_usd_price, age
        # Need refresh — release lock during network call
        need_fetch = True

    if need_fetch:
        fetched = _fetch_sol_usd_from_jupiter()
        with _sol_usd_lock:
            if fetched and fetched > 0:
                _sol_usd_price = fetched
                _sol_usd_fetched_at = time.time()
                age = 0.0
            else:
                age = time.time() - _sol_usd_fetched_at
            return _sol_usd_price, age


def _fetch_sol_usd_from_jupiter() -> Optional[float]:
    """Fetch SOL/USD price from Jupiter Price API v2. Returns None on error."""
    try:
        r = requests.get(_JUP_PRICE_URL, timeout=_JUP_TIMEOUT_S)
        if r.status_code == 200:
            entry = (r.json().get("data") or {}).get(_JUPITER_SOL_MINT)
            if entry:
                price = float(entry.get("price") or 0)
                return price if price > 0 else None
    except Exception as e:
        log.debug("get_sol_usd_cached: Jupiter fetch failed: %s", e)
    return None


# ── PDA derivation ────────────────────────────────────────────────────────────

def derive_curve_address(mint: str) -> Optional[str]:
    """
    Derive the pump.fun bonding-curve PDA for a mint.
    Seeds: [b"bonding-curve", bytes(mint_pubkey)]
    Returns base58 address string or None if derivation fails.
    """
    try:
        from solders.pubkey import Pubkey
        mint_pubkey = Pubkey.from_string(mint)
        program_id  = Pubkey.from_string(PUMP_PROGRAM)
        seeds = [b"bonding-curve", bytes(mint_pubkey)]
        curve_addr, _ = Pubkey.find_program_address(seeds, program_id)
        return str(curve_addr)
    except ImportError:
        log.debug("derive_curve_address: solders not available, falling back to pure-python")
        return _derive_curve_address_pure(mint)
    except Exception as e:
        log.debug("derive_curve_address: failed for %s: %s", mint[:8], e)
        return None


def _derive_curve_address_pure(mint: str) -> Optional[str]:
    """
    Pure-Python PDA derivation fallback (no solders dependency).
    Uses hashlib + base58 encoding per Solana PDA spec.
    """
    try:
        import hashlib

        def _b58decode(s: str) -> bytes:
            ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
            n = 0
            for c in s.encode():
                n = n * 58 + ALPHABET.index(c)
            result = n.to_bytes(32, "big")
            # count leading '1's → leading zero bytes
            pad = len(s) - len(s.lstrip("1"))
            return b"\x00" * pad + result.lstrip(b"\x00")

        def _b58encode(b: bytes) -> str:
            ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
            n = int.from_bytes(b, "big")
            s = ""
            while n:
                n, r = divmod(n, 58)
                s = ALPHABET[r] + s
            pad = len(b) - len(b.lstrip(b"\x00"))
            return "1" * pad + s

        mint_bytes    = _b58decode(mint)
        program_bytes = _b58decode(PUMP_PROGRAM)

        for nonce in range(255, -1, -1):
            seeds_with_nonce = (
                b"bonding-curve"
                + mint_bytes
                + bytes([nonce])
                + program_bytes
                + b"ProgramDerivedAddress"
            )
            h = hashlib.sha256(seeds_with_nonce).digest()
            # Valid PDA must NOT be on the ed25519 curve
            # Quick check: if the point is off-curve (common for PDAs), use it
            try:
                # Attempt to import and check — if nacl unavailable, skip check
                import nacl.signing  # type: ignore
                nacl.signing.VerifyKey(h)
                continue  # on-curve → not a valid PDA
            except Exception:
                pass
            return _b58encode(h)
    except Exception as e:
        log.debug("_derive_curve_address_pure: failed for %s: %s", mint[:8], e)
    return None


# ── Account data parsing ──────────────────────────────────────────────────────
#
# PROGRESS-FIX PF3: state parsing (this account's raw reserves/graduation
# flag) is now separate from USD price calculation, so callers that only
# need vsol_ui/progress (e.g. progress capture) can succeed even when
# SOL/USD is stale or unavailable — the old _parse_curve_account rejected
# everything up front on a stale SOL/USD price, which was correct for its
# own price_usd-producing callers but wrong to force on state-only callers.

def _decode_curve_account_state(data_b64: str, mint: str) -> dict:
    """
    Decode a base64-encoded bonding-curve account into its raw reserves and
    graduation flag. No SOL/USD dependency at all — succeeds independent of
    price-feed availability.

    Returns a dict with:
      virtual_token_reserves (int|None), virtual_sol_reserves (int|None),
      vsol_ui (float|None), complete (bool|None), failure_reason (str|None)
    failure_reason is one of: curve_parse_error, curve_layout_unknown, or None.
    """
    result = {
        "virtual_token_reserves": None,
        "virtual_sol_reserves":   None,
        "vsol_ui":                None,
        "complete":               None,
        "failure_reason":         "curve_parse_error",
    }

    try:
        raw = base64.b64decode(data_b64)
    except Exception as e:
        log.debug("_decode_curve_account_state: base64 decode failed for %s: %s", mint[:8], e)
        return result

    if len(raw) < _MIN_ACCOUNT_DATA_LEN:
        log.debug("_decode_curve_account_state: data too short (%d bytes) for %s", len(raw), mint[:8])
        result["failure_reason"] = "curve_layout_unknown"
        return result

    try:
        virtual_token_reserves, = struct.unpack_from("<Q", raw, _OFFSET_VIRTUAL_TOKEN)
        virtual_sol_reserves,   = struct.unpack_from("<Q", raw, _OFFSET_VIRTUAL_SOL)
        complete_byte,          = struct.unpack_from("<?", raw, _OFFSET_COMPLETE)
    except struct.error as e:
        log.debug("_decode_curve_account_state: struct unpack failed for %s: %s", mint[:8], e)
        result["failure_reason"] = "curve_layout_unknown"
        return result

    result["virtual_token_reserves"] = virtual_token_reserves
    result["virtual_sol_reserves"]   = virtual_sol_reserves
    result["vsol_ui"]                = virtual_sol_reserves / 1e9
    result["complete"]               = bool(complete_byte)
    result["failure_reason"]         = None
    return result


def _parse_curve_account(
    data_b64: str,
    mint: str,
    sol_price_usd: float,
    sol_price_age_s: float,
) -> dict:
    """
    Parse a base64-encoded bonding-curve account into a price_usd result.
    Returns a result dict with venue_state, price_usd, vsol_ui, complete,
    failure_reason populated.
    """
    result = {
        "price_usd":      None,
        "vsol_ui":        None,
        "complete":       None,
        "venue_state":    "PARSE_ERROR",
        "failure_reason": "curve_parse_error",
    }

    # Reject stale SOL/USD price (only relevant here — this function must
    # produce a USD price. State-only callers use _decode_curve_account_state
    # directly and don't need SOL/USD at all.)
    if sol_price_age_s > SOL_USD_MAX_CACHE_AGE_S or sol_price_usd <= 0:
        result["venue_state"]    = "PARSE_ERROR"
        result["failure_reason"] = "sol_usd_stale"
        log.debug("_parse_curve_account: SOL/USD stale (age=%.1fs) for %s",
                  sol_price_age_s, mint[:8])
        return result

    state = _decode_curve_account_state(data_b64, mint)
    if state["failure_reason"] is not None:
        result["failure_reason"] = state["failure_reason"]
        return result

    virtual_token_reserves = state["virtual_token_reserves"]
    virtual_sol_reserves   = state["virtual_sol_reserves"]
    complete                = state["complete"]
    result["complete"] = complete

    if complete:
        # Token has graduated — no price from curve, switch to DEX path
        result["venue_state"]    = "GRADUATED"
        result["failure_reason"] = None
        return result

    # Guard against division by zero
    if virtual_token_reserves == 0:
        log.debug("_parse_curve_account: zero virtualTokenReserves for %s", mint[:8])
        result["failure_reason"] = "curve_parse_error"
        return result

    # Price calculation (fixed 2026-08-07, PROGRESS-FIX PF0 — see module
    # docstring; previously missing the token-decimal conversion, silently
    # underpricing every curve_account read by exactly 1,000,000x)
    vsol_sol            = virtual_sol_reserves / 1e9
    vtoken_ui            = virtual_token_reserves / (10 ** PUMP_DECIMALS)
    price_sol_per_token  = vsol_sol / vtoken_ui
    price_usd            = price_sol_per_token * sol_price_usd
    vsol_ui              = virtual_sol_reserves / 1e9             # unchanged

    result["price_usd"]      = price_usd
    result["vsol_ui"]        = vsol_ui
    result["venue_state"]    = "CURVE_ACTIVE"
    result["failure_reason"] = None
    return result


# ── Batch price fetch ─────────────────────────────────────────────────────────

def get_curve_prices_batch(
    mints: list,
    helius_key: str,
    sol_price_usd: float,
    sol_price_age_s: float,
) -> dict:
    """
    Fetch bonding-curve prices for a batch of mints via Helius getMultipleAccounts.

    Returns:
        {
            mint: {
                price_usd:      float | None,
                vsol_ui:        float | None,
                complete:       bool | None,
                venue_state:    "CURVE_ACTIVE" | "GRADUATED" | "CURVE_MISSING" |
                                "PARSE_ERROR" | "RPC_ERROR",
                failure_reason: None | "curve_account_missing" | "curve_parse_error" |
                                "curve_layout_unknown" | "curve_rpc_error" | "sol_usd_stale",
                curve_address:  str | None,
                rpc_latency_ms: float | None,
            }
        }

    Processes up to CURVE_BATCH_SIZE mints per RPC call.
    Partial failures (some mints null, others succeed) are handled per-mint.
    RPC errors are never treated as graduation.
    """
    if not mints:
        return {}

    results: dict = {}

    # Process in batches of CURVE_BATCH_SIZE
    for batch_start in range(0, len(mints), CURVE_BATCH_SIZE):
        batch_mints = mints[batch_start: batch_start + CURVE_BATCH_SIZE]
        _fetch_batch(batch_mints, helius_key, sol_price_usd, sol_price_age_s, results)

    return results


def get_curve_state_batch(mints: list, helius_key: str) -> dict:
    """
    PROGRESS-FIX PF3: fetch bonding-curve STATE (vsol_ui, complete) for a
    batch of mints via Helius getMultipleAccounts — no SOL/USD price
    dependency at all. Used by progress capture, which only needs
    vsol_ui/GRAD_SOL_UI, not a USD price, and must succeed even when the
    SOL/USD cache is stale or unavailable.

    Returns the same shape as get_curve_prices_batch but price_usd is
    always None (never computed in this path).
    """
    if not mints:
        return {}

    results: dict = {}
    for batch_start in range(0, len(mints), CURVE_BATCH_SIZE):
        batch_mints = mints[batch_start: batch_start + CURVE_BATCH_SIZE]
        _fetch_batch(batch_mints, helius_key, None, None, results)

    return results


def _fetch_batch(
    mints: list,
    helius_key: str,
    sol_price_usd: Optional[float],
    sol_price_age_s: Optional[float],
    results: dict,
) -> None:
    """
    Internal: derive curve addresses for `mints`, call getMultipleAccounts,
    parse results, write into `results` dict.

    sol_price_usd/sol_price_age_s = None -> state-only mode (used by
    get_curve_state_batch): parses via _decode_curve_account_state instead
    of _parse_curve_account, price_usd always None, no SOL/USD requirement.
    """
    state_only = sol_price_usd is None
    # Step 1: derive curve addresses for all mints in batch
    mint_to_curve: dict[str, Optional[str]] = {}
    for mint in mints:
        curve_addr = derive_curve_address(mint)
        mint_to_curve[mint] = curve_addr

    # Mints where derivation failed — mark immediately
    ok_mints = [m for m in mints if mint_to_curve[m] is not None]
    for mint in mints:
        if mint_to_curve[mint] is None:
            results[mint] = {
                "price_usd":      None,
                "vsol_ui":        None,
                "complete":       None,
                "venue_state":    "CURVE_MISSING",
                "failure_reason": "curve_account_missing",
                "curve_address":  None,
                "rpc_latency_ms": None,
            }

    if not ok_mints:
        return

    # Step 2: build pubkeys list in same order as ok_mints
    pubkeys = [mint_to_curve[m] for m in ok_mints]

    # Step 3: call getMultipleAccounts, with public-RPC fallback on
    # Helius failure/429 (see _RPC_FALLBACK_1/_2 comment above).
    payload = {
        "jsonrpc": "2.0",
        "id":      1,
        "method":  "getMultipleAccounts",
        "params":  [pubkeys, {"encoding": "base64", "commitment": "confirmed"}],
    }
    rpc_tiers = [f"https://mainnet.helius-rpc.com/?api-key={helius_key}",
                 _RPC_FALLBACK_1, _RPC_FALLBACK_2]

    t0 = time.time()
    rpc_data = None
    last_err: Optional[Exception] = None
    for tier_i, rpc_url in enumerate(rpc_tiers):
        try:
            resp = requests.post(rpc_url, json=payload, timeout=10)
            resp.raise_for_status()
            rpc_data = resp.json()
            if tier_i > 0:
                log.info("curve_oracle: recovered via fallback RPC tier %d for batch of %d mints",
                          tier_i, len(ok_mints))
            break
        except Exception as e:
            last_err = e
            if tier_i < len(rpc_tiers) - 1:
                log.debug("curve_oracle: RPC tier %d failed (%s), trying next tier", tier_i, e)
    rpc_latency_ms = (time.time() - t0) * 1000

    if rpc_data is None:
        log.warning("curve_oracle: RPC error for batch of %d mints (all tiers): %s",
                     len(ok_mints), last_err)
        # Mark ALL ok_mints as RPC_ERROR — NOT graduation
        for mint in ok_mints:
            results[mint] = {
                "price_usd":      None,
                "vsol_ui":        None,
                "complete":       None,
                "venue_state":    "RPC_ERROR",
                "failure_reason": "curve_rpc_error",
                "curve_address":  mint_to_curve[mint],
                "rpc_latency_ms": rpc_latency_ms,
            }
        return

    rpc_error = rpc_data.get("error")
    if rpc_error:
        log.warning("curve_oracle: RPC returned error for batch: %s", rpc_error)
        for mint in ok_mints:
            results[mint] = {
                "price_usd":      None,
                "vsol_ui":        None,
                "complete":       None,
                "venue_state":    "RPC_ERROR",
                "failure_reason": "curve_rpc_error",
                "curve_address":  mint_to_curve[mint],
                "rpc_latency_ms": rpc_latency_ms,
            }
        return

    # Step 4: parse per-account results
    accounts = (rpc_data.get("result") or {}).get("value") or []

    for i, mint in enumerate(ok_mints):
        curve_addr = mint_to_curve[mint]
        account = accounts[i] if i < len(accounts) else None

        if account is None:
            # Account doesn't exist — token never existed or already graduated
            # NOT treated as graduation — only complete=True in data is graduation
            results[mint] = {
                "price_usd":      None,
                "vsol_ui":        None,
                "complete":       None,
                "venue_state":    "CURVE_MISSING",
                "failure_reason": "curve_account_missing",
                "curve_address":  curve_addr,
                "rpc_latency_ms": rpc_latency_ms,
            }
            log.debug("curve_oracle: null account for %s (curve %s)", mint[:8],
                      str(curve_addr)[:8] if curve_addr else "none")
            continue

        # Account data is [base64_string, encoding]
        data_field = account.get("data")
        if not data_field or not isinstance(data_field, list) or len(data_field) < 1:
            results[mint] = {
                "price_usd":      None,
                "vsol_ui":        None,
                "complete":       None,
                "venue_state":    "PARSE_ERROR",
                "failure_reason": "curve_parse_error",
                "curve_address":  curve_addr,
                "rpc_latency_ms": rpc_latency_ms,
            }
            continue

        data_b64 = data_field[0]

        if state_only:
            state = _decode_curve_account_state(data_b64, mint)
            if state["failure_reason"] is not None:
                parsed = {
                    "price_usd":      None,
                    "vsol_ui":        None,
                    "complete":       None,
                    "venue_state":    "PARSE_ERROR",
                    "failure_reason": state["failure_reason"],
                }
            elif state["complete"]:
                parsed = {
                    "price_usd":      None,
                    "vsol_ui":        state["vsol_ui"],
                    "complete":       True,
                    "venue_state":    "GRADUATED",
                    "failure_reason": None,
                }
            else:
                parsed = {
                    "price_usd":      None,
                    "vsol_ui":        state["vsol_ui"],
                    "complete":       False,
                    "venue_state":    "CURVE_ACTIVE",
                    "failure_reason": None,
                }
        else:
            parsed = _parse_curve_account(data_b64, mint, sol_price_usd, sol_price_age_s)

        results[mint] = {
            **parsed,
            "curve_address":  curve_addr,
            "rpc_latency_ms": rpc_latency_ms,
        }

        log.debug(
            "curve_oracle: %s → venue=%s price=%s vsol=%.2f",
            mint[:8],
            parsed["venue_state"],
            f"${parsed['price_usd']:.8f}" if parsed["price_usd"] else "None",
            parsed["vsol_ui"] or 0.0,
        )
