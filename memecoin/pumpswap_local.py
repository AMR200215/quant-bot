"""
PumpSwap local sell TX builder — Level 3 exit layer.

Builds and optionally sends PumpSwap AMM sell transactions locally, bypassing
PumpPortal's trade-local API which does not support Token-2022 ATA derivation
(returns HTTP 400 on T22 tokens with pool="pump-amm").

Public API
----------
fetch_pool(token_mint, rpc_url)            → dict  (raises PumpSwapPoolError)
build_pumpswap_sell_tx(*, ...)             → bytes (signed TX, not sent)
simulate_sell(tx_bytes, rpc_url)           → (ok, error_class, log_lines)
PumpSwapPoolError(error_class, detail)

Internal flow
-------------
1. fetch_pool()  — getProgramAccounts on PUMP_AMM_PROGRAM filtered to base_mint
2. Parse pool state (binary Anchor layout, MIN_SIZE=211 / FULL_SIZE=243)
3. build_pumpswap_sell_tx()  — assemble 21-account sell instruction + 3 remainingAccounts + 5-ix TX
4. simulate_sell()  — simulateTransaction RPC → parse logs → classify error
"""

import base64
import logging
import random
import struct
import threading
import time

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Program constants (verified mainnet addresses — do not change without IDL bump)
# ---------------------------------------------------------------------------
PUMP_AMM_PROGRAM      = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
FEE_PROGRAM           = "pfeeUxB6jkeY1Hxd7CsFCAjcbHA9rWtchMGdZ6VojVZ"
GLOBAL_CONFIG         = "ADyA8hdefvWN2dbGGWFotbzWxrAvLW83WG6QCVXvJKqw"
EVENT_AUTHORITY       = "GS4CU59F31iL7aR2Q8zVS8DRrcRnXX1yjQ66TqNVQnaR"
PROTOCOL_FEE_RECIPIENT = "62qc2CNXwrYqQScmEdiZFFAnJR262PxWEuNQtxfafNgV"

# Buyback fee recipients — one of these 8 is appended (readonly) + its WSOL ATA (writable)
# as remainingAccounts on every sell TX, per the April 28 2025 breaking upgrade.
# The SDK picks randomly; we do the same.
BUYBACK_FEE_RECIPIENTS = [
    "5YxQFdt3Tr9zJLvkFccqXVUwhdTWJQc1fFg2YPbxvxeD",
    "9M4giFFMxmFGXtc3feFzRai56WbBqehoSeRE5GK7gf7",
    "GXPFM2caqTtQYC2cJ5yJRi9VDkpsYZXzYdwYpGnLmtDL",
    "3BpXnfJaUTiwXnJNe7Ej1rcbzqTTQUvLShZaWazebsVR",
    "5cjcW9wExnJJiqgLjq7DEG75Pm6JBgE1hNv4B2vHXUW6",
    "EHAAiTxcdDwQ3U4bU6YcMsQGaekdzLS3B5SmYo46kJtL",
    "5eHhjP8JaYkz83CWwvGU2uMUXefd3AazWGx4gpcuEEYD",
    "A7hAgCzFw14fejgCp387JUJRMNyz4j89JKnhtKU8piqW",
]

TOKEN_PROGRAM_SPL     = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_PROGRAM_T22     = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
ATA_PROGRAM           = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
WSOL_MINT             = "So11111111111111111111111111111111111111112"
SYSTEM_PROGRAM        = "11111111111111111111111111111111"
COMPUTE_BUDGET        = "ComputeBudget111111111111111111111111111111"

# Sell instruction discriminator: sha256("global:sell")[:8] for PumpSwap AMM.
# Pre-computed from the official IDL — do not recalculate from a different namespace.
_SELL_DISCRIMINATOR = bytes.fromhex("33e685a4017f83ad")

# Null pubkey — used to detect absent coin_creator in old pool layouts
_NULL_PUBKEY_BYTES = bytes(32)
_NULL_PUBKEY_STR   = "11111111111111111111111111111111"

# Pool state layout (Anchor binary — see module docstring for field offsets)
_POOL_MIN_SIZE  = 211   # through lp_supply (8+1+2+32+32+32+32+32+8 + 32 = 211, first 211 cover creator field offset)
_POOL_FULL_SIZE = 243   # includes coin_creator (32 bytes at offset 211)

# Pool discovery result cache: mint → (pool_dict, cached_at)
POOL_CACHE_TTL = 120   # seconds
_pool_cache: dict[str, tuple[dict, float]] = {}
_pool_cache_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PumpSwapPoolError(Exception):
    """Raised when pool discovery or layout parsing fails."""
    def __init__(self, error_class: str, detail: str = ""):
        self.error_class = error_class
        super().__init__(f"{error_class}: {detail}")


# ---------------------------------------------------------------------------
# RPC helper (mirrors executor._rpc_post but takes rpc_url as arg)
# ---------------------------------------------------------------------------

def _rpc(rpc_url: str, payload: dict, timeout: int = 10) -> dict:
    """POST JSON-RPC to rpc_url, return parsed JSON dict. Raises on HTTP error."""
    resp = requests.post(rpc_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# ATA derivation (8-line re-implementation — mirrors executor._derive_ata)
# ---------------------------------------------------------------------------

def _derive_ata(owner_str: str, mint_str: str, token_program_id: str = TOKEN_PROGRAM_SPL) -> str:
    """Derive the Associated Token Account (ATA) address for owner + mint.

    token_program_id: TOKEN_PROGRAM_T22 for Token-2022 mints, else TOKEN_PROGRAM_SPL.
    Uses solders Pubkey.find_program_address — same derivation as on-chain ATA program.
    """
    from solders.pubkey import Pubkey
    ata_prog = Pubkey.from_string(ATA_PROGRAM)
    owner    = Pubkey.from_string(owner_str)
    mint     = Pubkey.from_string(mint_str)
    tok_prog = Pubkey.from_string(token_program_id)
    ata, _   = Pubkey.find_program_address(
        [bytes(owner), bytes(tok_prog), bytes(mint)],
        ata_prog,
    )
    return str(ata)


# ---------------------------------------------------------------------------
# PDA derivations (PumpSwap-specific)
# ---------------------------------------------------------------------------

def _derive_coin_creator_vault_authority(coin_creator_str: str) -> str:
    """PDA: seeds=[b"creator_vault", coin_creator_bytes], program=PUMP_AMM_PROGRAM."""
    from solders.pubkey import Pubkey
    prog  = Pubkey.from_string(PUMP_AMM_PROGRAM)
    cc_pk = Pubkey.from_string(coin_creator_str)
    pda, _ = Pubkey.find_program_address([b"creator_vault", bytes(cc_pk)], prog)
    return str(pda)


def _derive_pool_v2_pda(base_mint_str: str) -> str:
    """PDA: seeds=[b"pool-v2", base_mint_bytes], program=PUMP_AMM_PROGRAM.

    Added in the April 2025 pool-v2 upgrade. Appended as a readonly remaining
    account on sell TXs when coin_creator is non-null.
    """
    from solders.pubkey import Pubkey
    prog     = Pubkey.from_string(PUMP_AMM_PROGRAM)
    mint_pk  = Pubkey.from_string(base_mint_str)
    pda, _   = Pubkey.find_program_address([b"pool-v2", bytes(mint_pk)], prog)
    return str(pda)


def _derive_fee_config() -> str:
    """PDA: seeds=[b"fee_config", PUMP_AMM_PROGRAM_bytes], program=FEE_PROGRAM."""
    from solders.pubkey import Pubkey
    fee_prog = Pubkey.from_string(FEE_PROGRAM)
    amm_pk   = Pubkey.from_string(PUMP_AMM_PROGRAM)
    pda, _   = Pubkey.find_program_address([b"fee_config", bytes(amm_pk)], fee_prog)
    return str(pda)


# ---------------------------------------------------------------------------
# Pool discovery
# ---------------------------------------------------------------------------

def fetch_pool(token_mint: str, rpc_url: str) -> dict:
    """
    Discover the PumpSwap AMM pool for this mint via RPC (source of truth).

    Uses getProgramAccounts with memcmp filter at offset=43 (base_mint field)
    to find the pool account for this mint. Parses Anchor binary layout.

    Returns pool data dict with keys:
        pool_address, pool_base_token_account, pool_quote_token_account,
        coin_creator, base_mint, quote_mint

    Caches result for POOL_CACHE_TTL seconds.
    Raises PumpSwapPoolError on any failure.
    """
    # Cache check
    with _pool_cache_lock:
        if token_mint in _pool_cache:
            cached_pool, cached_at = _pool_cache[token_mint]
            if time.time() - cached_at < POOL_CACHE_TTL:
                log.debug("fetch_pool cache hit  mint=%s", token_mint[:8])
                return cached_pool

    log.debug("fetch_pool RPC lookup  mint=%s", token_mint[:8])

    # Validate mint address before the RPC call.
    # memcmp filter uses token_mint directly (base58 — correct format for Solana RPC).
    from solders.pubkey import Pubkey
    try:
        Pubkey.from_string(token_mint)
    except Exception as e:
        raise PumpSwapPoolError("pumpswap_bad_pool_layout", f"Invalid mint pubkey: {e}")

    try:
        data = _rpc(rpc_url, {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getProgramAccounts",
            "params": [
                PUMP_AMM_PROGRAM,
                {
                    "encoding": "base64",
                    "commitment": "confirmed",
                    "filters": [
                        # base_mint field starts at offset 43 in pool state layout
                        {
                            "memcmp": {
                                "offset": 43,
                                "bytes": token_mint,  # base58 encoded mint address
                            }
                        }
                    ],
                }
            ],
        }, timeout=15)
    except Exception as e:
        raise PumpSwapPoolError("pumpswap_no_pool", f"getProgramAccounts RPC error: {e}")

    accounts = data.get("result", [])
    if not accounts:
        raise PumpSwapPoolError("pumpswap_no_pool",
                                f"No PumpSwap pool found for mint {token_mint[:8]}")

    # Use first result (there should be exactly one pool per mint)
    acct      = accounts[0]
    pool_addr = acct["pubkey"]
    raw_b64   = acct["account"]["data"][0]
    try:
        raw = base64.b64decode(raw_b64)
    except Exception as e:
        raise PumpSwapPoolError("pumpswap_bad_pool_layout", f"base64 decode failed: {e}")

    if len(raw) < _POOL_MIN_SIZE:
        raise PumpSwapPoolError(
            "pumpswap_bad_pool_layout",
            f"Pool data too short: {len(raw)} < {_POOL_MIN_SIZE} bytes  pool={pool_addr[:8]}",
        )

    # Parse pool state fields (Anchor binary layout, all offsets relative to byte 0)
    #
    # Offset  Size  Field
    # 0       8     anchor discriminator
    # 8       1     pool_bump
    # 9       2     index (u16 LE)
    # 11      32    creator
    # 43      32    base_mint
    # 75      32    quote_mint
    # 107     32    lp_mint
    # 139     32    pool_base_token_account
    # 171     32    pool_quote_token_account
    # 203     8     lp_supply (u64 LE)
    # 211     32    coin_creator  ← only present in new pools (len >= 243)
    try:
        base_mint_bytes  = raw[43:75]
        quote_mint_bytes = raw[75:107]
        pool_base_ta     = raw[139:171]
        pool_quote_ta    = raw[171:203]

        base_mint_parsed  = str(Pubkey.from_bytes(base_mint_bytes))
        quote_mint_parsed = str(Pubkey.from_bytes(quote_mint_bytes))
        pool_base_ta_str  = str(Pubkey.from_bytes(pool_base_ta))
        pool_quote_ta_str = str(Pubkey.from_bytes(pool_quote_ta))
    except Exception as e:
        raise PumpSwapPoolError("pumpswap_bad_pool_layout", f"Field parse error: {e}")

    # Validate mints match what we searched for
    if base_mint_parsed != token_mint:
        raise PumpSwapPoolError(
            "pumpswap_bad_pool_layout",
            f"base_mint mismatch: expected {token_mint[:8]} got {base_mint_parsed[:8]}",
        )

    # coin_creator: present only when data >= FULL_SIZE (243 bytes)
    # Old pools: absent → treat as null pubkey (PDA derivation still happens, but is a no-op)
    if len(raw) >= _POOL_FULL_SIZE:
        coin_creator_bytes = raw[211:243]
        coin_creator_str   = str(Pubkey.from_bytes(coin_creator_bytes))
        # Also treat null pubkey as absent
        if coin_creator_bytes == _NULL_PUBKEY_BYTES:
            coin_creator_str = _NULL_PUBKEY_STR
    else:
        coin_creator_str = _NULL_PUBKEY_STR

    pool = {
        "pool_address":             pool_addr,
        "base_mint":                base_mint_parsed,
        "quote_mint":               quote_mint_parsed,
        "pool_base_token_account":  pool_base_ta_str,
        "pool_quote_token_account": pool_quote_ta_str,
        "coin_creator":             coin_creator_str,
    }

    log.info(
        "fetch_pool OK  mint=%s  pool=%s  cc=%s  pool_base_ta=%s",
        token_mint[:8], pool_addr[:8],
        coin_creator_str[:8] if coin_creator_str != _NULL_PUBKEY_STR else "null",
        pool_base_ta_str[:8],
    )

    with _pool_cache_lock:
        _pool_cache[token_mint] = (pool, time.time())

    return pool


# ---------------------------------------------------------------------------
# Reserve reading and min_sol_out computation
# ---------------------------------------------------------------------------

def fetch_pool_sol_reserves(pool: dict, rpc_url: str) -> int | None:
    """
    Read the pool's WSOL reserve via getTokenAccountBalance on pool_quote_token_account.
    Returns lamports (int) or None on any error.
    Used to compute min_sol_out for live sells.
    """
    quote_ta = pool.get("pool_quote_token_account")
    if not quote_ta:
        return None
    try:
        data = _rpc(rpc_url, {
            "jsonrpc": "2.0", "id": 1,
            "method": "getTokenAccountBalance",
            "params": [quote_ta],
        }, timeout=10)
        amount_str = (
            data.get("result", {})
                .get("value", {})
                .get("amount")
        )
        if amount_str:
            return int(amount_str)
    except Exception as e:
        log.warning("fetch_pool_sol_reserves: RPC error  pool_quote_ta=%s  err=%s",
                    quote_ta[:8], e)
    return None


def compute_min_sol_out(
    token_amount_raw: int,
    pool: dict,
    rpc_url: str,
    slippage_pct: float = 35.0,
) -> tuple[int, int | None]:
    """
    Estimate expected SOL out using constant-product AMM formula, then apply slippage.

    We need:
      - pool_base_token_account balance (token reserves)
      - pool_quote_token_account balance (SOL reserves in lamports)

    Returns (min_sol_out_lamports, sol_reserves_lam).
      min_sol_out_lamports = 0 if reserves cannot be read (caller must decide to abort or use 0).
    """
    sol_reserves_lam = fetch_pool_sol_reserves(pool, rpc_url)
    if sol_reserves_lam is None:
        return 0, None

    # Fetch token reserves via getTokenAccountBalance on pool_base_token_account
    base_ta = pool.get("pool_base_token_account")
    token_reserves_raw = None
    if base_ta:
        try:
            data = _rpc(rpc_url, {
                "jsonrpc": "2.0", "id": 1,
                "method": "getTokenAccountBalance",
                "params": [base_ta],
            }, timeout=10)
            amt = data.get("result", {}).get("value", {}).get("amount")
            if amt:
                token_reserves_raw = int(amt)
        except Exception as e:
            log.warning("compute_min_sol_out: token reserve RPC error  base_ta=%s  err=%s",
                        base_ta[:8], e)

    if token_reserves_raw is None or token_reserves_raw == 0:
        return 0, sol_reserves_lam

    # Constant-product AMM: x*y=k
    # Expected SOL out = sol_reserves * token_in / (token_reserves + token_in)
    k = sol_reserves_lam * token_reserves_raw
    sol_out_expected = sol_reserves_lam - k // (token_reserves_raw + token_amount_raw)
    # Apply slippage floor
    min_sol_out = int(sol_out_expected * (1.0 - slippage_pct / 100.0))
    min_sol_out = max(0, min_sol_out)
    log.info(
        "compute_min_sol_out  token_in=%d  sol_reserves=%d lam  token_reserves=%d"
        "  expected_out=%d lam  min_out=%d lam (%.0f%% slippage)",
        token_amount_raw, sol_reserves_lam, token_reserves_raw,
        sol_out_expected, min_sol_out, slippage_pct,
    )
    return min_sol_out, sol_reserves_lam


# ---------------------------------------------------------------------------
# TX builder
# ---------------------------------------------------------------------------

def build_pumpswap_sell_tx(
    *,
    wallet_pubkey: str,
    keypair,                    # solders Keypair (already loaded by caller)
    token_mint: str,
    token_amount_raw: int,      # exact raw token count to sell (u64)
    min_sol_out_lamports: int,  # minimum SOL out after slippage (u64)
    priority_fee_sol: float,    # SOL value for SetComputeUnitPrice
    token_program_id: str,      # TOKEN_PROGRAM_SPL or TOKEN_PROGRAM_T22
    pool: dict,                 # from fetch_pool()
    rpc_url: str,
) -> bytes:
    """
    Build and sign a PumpSwap AMM sell transaction (21 accounts, official IDL order).
    Does NOT send the transaction.

    TX structure:
      0. SetComputeUnitPrice
      1. SetComputeUnitLimit (200_000 CU)
      2. CreateAssociatedTokenAccountIdempotent for user WSOL ATA  (data=bytes([1]))
      3. PumpSwap sell instruction (21 IDL accounts + 2-3 remainingAccounts):
           remaining[0] = poolV2Pda(base_mint)  ← only when coin_creator is non-null
           remaining[-2] = buyback_fee_recipient (readonly, random from 8)
           remaining[-1] = ATA(buyback_fee_recipient, WSOL, SPL)  (writable)
      4. CloseAccount on user WSOL ATA → unwrap SOL to wallet

    Returns fully-signed VersionedTransaction bytes ready for _send_transaction().
    """
    from solders.pubkey import Pubkey
    from solders.instruction import Instruction, AccountMeta
    from solders.message import MessageV0
    from solders.hash import Hash
    from solders.transaction import VersionedTransaction

    CU_BUDGET = 200_000  # compute units reserved for this TX

    # --- Pubkey objects ---
    amm_prog_pk     = Pubkey.from_string(PUMP_AMM_PROGRAM)
    fee_prog_pk     = Pubkey.from_string(FEE_PROGRAM)
    global_cfg_pk   = Pubkey.from_string(GLOBAL_CONFIG)
    event_auth_pk   = Pubkey.from_string(EVENT_AUTHORITY)
    fee_recip_pk    = Pubkey.from_string(PROTOCOL_FEE_RECIPIENT)
    tok_prog_pk     = Pubkey.from_string(token_program_id)
    spl_tok_pk      = Pubkey.from_string(TOKEN_PROGRAM_SPL)  # WSOL is always SPL
    ata_prog_pk     = Pubkey.from_string(ATA_PROGRAM)
    wsol_pk         = Pubkey.from_string(WSOL_MINT)
    system_pk       = Pubkey.from_string(SYSTEM_PROGRAM)
    compute_pk      = Pubkey.from_string(COMPUTE_BUDGET)
    wallet_pk       = Pubkey.from_string(wallet_pubkey)
    base_mint_pk    = Pubkey.from_string(token_mint)
    pool_pk         = Pubkey.from_string(pool["pool_address"])
    pool_base_ta_pk = Pubkey.from_string(pool["pool_base_token_account"])
    pool_quote_ta_pk = Pubkey.from_string(pool["pool_quote_token_account"])

    # --- ATA derivations ---
    # user_base_token_account: Token-2022 ATA if T22 mint, else SPL ATA
    user_base_ata_str  = _derive_ata(wallet_pubkey, token_mint, token_program_id)
    user_base_ta_pk    = Pubkey.from_string(user_base_ata_str)

    # user_quote_token_account: WSOL ATA (always SPL — WSOL is on SPL token program)
    user_wsol_ata_str  = _derive_ata(wallet_pubkey, WSOL_MINT, TOKEN_PROGRAM_SPL)
    user_wsol_ta_pk    = Pubkey.from_string(user_wsol_ata_str)

    # protocol_fee_recipient_token_account: ATA(fee_recip, WSOL, SPL)
    fee_recip_wsol_ata_str = _derive_ata(PROTOCOL_FEE_RECIPIENT, WSOL_MINT, TOKEN_PROGRAM_SPL)
    fee_recip_wsol_ta_pk   = Pubkey.from_string(fee_recip_wsol_ata_str)

    # --- PDA derivations ---
    # fee_config PDA: seeds=[b"fee_config", PUMP_AMM_PROGRAM_bytes], program=FEE_PROGRAM
    fee_config_str = _derive_fee_config()
    fee_config_pk  = Pubkey.from_string(fee_config_str)

    # coin_creator_vault_authority PDA (even for old pools — the on-chain program expects it)
    coin_creator_str = pool["coin_creator"]
    cc_vault_auth_str = _derive_coin_creator_vault_authority(
        coin_creator_str if coin_creator_str != _NULL_PUBKEY_STR else _NULL_PUBKEY_STR
    )
    cc_vault_auth_pk = Pubkey.from_string(cc_vault_auth_str)

    # coin_creator_vault_ata: ATA(vault_authority, WSOL, SPL)
    cc_vault_ata_str = _derive_ata(cc_vault_auth_str, WSOL_MINT, TOKEN_PROGRAM_SPL)
    cc_vault_ata_pk  = Pubkey.from_string(cc_vault_ata_str)

    # --- Instruction 0: SetComputeUnitPrice ---
    # microlamports_per_cu = priority_fee_sol * 1e15 / CU_BUDGET
    micro_lam = max(1, int(priority_fee_sol * 1e15 / CU_BUDGET))
    ix_cu_price = Instruction(
        program_id=compute_pk,
        data=bytes([0x03]) + struct.pack("<Q", micro_lam),
        accounts=[],
    )

    # --- Instruction 1: SetComputeUnitLimit ---
    ix_cu_limit = Instruction(
        program_id=compute_pk,
        data=bytes([0x02]) + struct.pack("<I", CU_BUDGET),
        accounts=[],
    )

    # --- Instruction 2: CreateAssociatedTokenAccountIdempotent for WSOL ATA ---
    # data=bytes([1]) = idempotent variant (instruction index 1 in ATA program)
    ix_create_wsol_ata = Instruction(
        program_id=ata_prog_pk,
        data=bytes([1]),  # CreateAssociatedTokenAccountIdempotent
        accounts=[
            AccountMeta(pubkey=wallet_pk,      is_signer=True,  is_writable=True),  # payer
            AccountMeta(pubkey=user_wsol_ta_pk, is_signer=False, is_writable=True),  # ata
            AccountMeta(pubkey=wallet_pk,      is_signer=False, is_writable=False), # owner
            AccountMeta(pubkey=wsol_pk,        is_signer=False, is_writable=False), # mint
            AccountMeta(pubkey=system_pk,      is_signer=False, is_writable=False), # system program
            AccountMeta(pubkey=spl_tok_pk,     is_signer=False, is_writable=False), # token program
        ],
    )

    # --- Instruction 3: PumpSwap sell (21 IDL accounts + remainingAccounts) ---
    sell_data = _SELL_DISCRIMINATOR + struct.pack("<QQ", token_amount_raw, min_sol_out_lamports)

    # --- remainingAccounts (April 2025 breaking upgrade) ---
    # When coin_creator is non-null: prepend poolV2Pda (readonly)
    # Always: append one random buyback fee recipient (readonly) + its WSOL ATA (writable)
    remaining_accounts: list = []

    has_coin_creator = (coin_creator_str != _NULL_PUBKEY_STR)
    if has_coin_creator:
        pool_v2_pda_str = _derive_pool_v2_pda(token_mint)
        pool_v2_pda_pk  = Pubkey.from_string(pool_v2_pda_str)
        remaining_accounts.append(
            AccountMeta(pubkey=pool_v2_pda_pk, is_signer=False, is_writable=False),
        )

    buyback_recip_str     = random.choice(BUYBACK_FEE_RECIPIENTS)
    buyback_recip_pk      = Pubkey.from_string(buyback_recip_str)
    buyback_recip_wsol_str = _derive_ata(buyback_recip_str, WSOL_MINT, TOKEN_PROGRAM_SPL)
    buyback_recip_wsol_pk  = Pubkey.from_string(buyback_recip_wsol_str)
    remaining_accounts.extend([
        AccountMeta(pubkey=buyback_recip_pk,      is_signer=False, is_writable=False),
        AccountMeta(pubkey=buyback_recip_wsol_pk, is_signer=False, is_writable=True),
    ])

    ix_sell = Instruction(
        program_id=amm_prog_pk,
        data=sell_data,
        accounts=[
            # 0  pool                          writable
            AccountMeta(pubkey=pool_pk,            is_signer=False, is_writable=True),
            # 1  user                          writable, signer
            AccountMeta(pubkey=wallet_pk,          is_signer=True,  is_writable=True),
            # 2  global_config                 readonly
            AccountMeta(pubkey=global_cfg_pk,      is_signer=False, is_writable=False),
            # 3  base_mint                     readonly
            AccountMeta(pubkey=base_mint_pk,       is_signer=False, is_writable=False),
            # 4  quote_mint (WSOL)             readonly
            AccountMeta(pubkey=wsol_pk,            is_signer=False, is_writable=False),
            # 5  user_base_token_account       writable  ← T22 ATA if T22 mint
            AccountMeta(pubkey=user_base_ta_pk,    is_signer=False, is_writable=True),
            # 6  user_quote_token_account      writable  ← user WSOL ATA
            AccountMeta(pubkey=user_wsol_ta_pk,    is_signer=False, is_writable=True),
            # 7  pool_base_token_account       writable  ← from pool state
            AccountMeta(pubkey=pool_base_ta_pk,    is_signer=False, is_writable=True),
            # 8  pool_quote_token_account      writable  ← from pool state
            AccountMeta(pubkey=pool_quote_ta_pk,   is_signer=False, is_writable=True),
            # 9  protocol_fee_recipient        readonly
            AccountMeta(pubkey=fee_recip_pk,       is_signer=False, is_writable=False),
            # 10 protocol_fee_recipient_token_account  writable  ← ATA(fee_recip, WSOL, SPL)
            AccountMeta(pubkey=fee_recip_wsol_ta_pk, is_signer=False, is_writable=True),
            # 11 base_token_program            readonly  ← T22 if T22 token, else SPL
            AccountMeta(pubkey=tok_prog_pk,        is_signer=False, is_writable=False),
            # 12 quote_token_program           readonly  ← SPL always (WSOL is SPL)
            AccountMeta(pubkey=spl_tok_pk,         is_signer=False, is_writable=False),
            # 13 system_program                readonly
            AccountMeta(pubkey=system_pk,          is_signer=False, is_writable=False),
            # 14 associated_token_program      readonly
            AccountMeta(pubkey=ata_prog_pk,        is_signer=False, is_writable=False),
            # 15 event_authority               readonly
            AccountMeta(pubkey=event_auth_pk,      is_signer=False, is_writable=False),
            # 16 program (self-ref)            readonly
            AccountMeta(pubkey=amm_prog_pk,        is_signer=False, is_writable=False),
            # 17 coin_creator_vault_ata        writable  ← ATA(vault_authority, WSOL, SPL)
            AccountMeta(pubkey=cc_vault_ata_pk,    is_signer=False, is_writable=True),
            # 18 coin_creator_vault_authority  readonly  ← PDA
            AccountMeta(pubkey=cc_vault_auth_pk,   is_signer=False, is_writable=False),
            # 19 fee_config                    readonly  ← PDA
            AccountMeta(pubkey=fee_config_pk,      is_signer=False, is_writable=False),
            # 20 fee_program                   readonly
            AccountMeta(pubkey=fee_prog_pk,        is_signer=False, is_writable=False),
            # 21+ remainingAccounts (see above):
            #   [poolV2Pda readonly]  ← only when coin_creator non-null
            #   buyback_fee_recipient  readonly
            #   ATA(buyback_fee_recipient, WSOL) writable
            *remaining_accounts,
        ],
    )

    # --- Instruction 4: CloseAccount on user WSOL ATA → unwrap SOL ---
    # Discriminator for CloseAccount in SPL Token program = [9]
    ix_close_wsol = Instruction(
        program_id=spl_tok_pk,
        data=bytes([9]),  # CloseAccount instruction
        accounts=[
            AccountMeta(pubkey=user_wsol_ta_pk, is_signer=False, is_writable=True),  # account to close
            AccountMeta(pubkey=wallet_pk,        is_signer=False, is_writable=True),  # destination (wallet receives lamports)
            AccountMeta(pubkey=wallet_pk,        is_signer=True,  is_writable=False), # authority
        ],
    )

    # --- Fetch recent blockhash ---
    try:
        bh_data = _rpc(rpc_url, {
            "jsonrpc": "2.0", "id": 1,
            "method": "getLatestBlockhash",
            "params": [{"commitment": "confirmed"}],
        }, timeout=8)
        blockhash_str = bh_data["result"]["value"]["blockhash"]
    except Exception as e:
        raise RuntimeError(f"getLatestBlockhash failed: {e}")

    # --- Build and sign VersionedTransaction ---
    msg = MessageV0.try_compile(
        payer=wallet_pk,
        instructions=[ix_cu_price, ix_cu_limit, ix_create_wsol_ata, ix_sell, ix_close_wsol],
        address_lookup_table_accounts=[],
        recent_blockhash=Hash.from_string(blockhash_str),
    )

    n_accounts = 21 + len(remaining_accounts)
    log.debug(
        "build_pumpswap_sell_tx  mint=%s  pool=%s  amount=%d  min_sol_out=%d  is_t22=%s  accounts=%d  buyback=%s",
        token_mint[:8], pool["pool_address"][:8],
        token_amount_raw, min_sol_out_lamports,
        token_program_id == TOKEN_PROGRAM_T22,
        n_accounts, buyback_recip_str[:8],
    )

    return bytes(VersionedTransaction(msg, [keypair]))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

# Error string patterns → error_class mapping (evaluated in order, first match wins)
_SIM_ERROR_PATTERNS = [
    # pump.fun bonding-curve Custom:6001 — T22 ATA wrong token program or BC sell on graduated
    ("custom program error: 0x1771", "pumpfun_t22_custom_6001"),
    ("Custom:6001",                   "pumpfun_t22_custom_6001"),
    # pump.fun Custom:6005 — sell on bonding-curve of already-graduated token
    ("custom program error: 0x177d", "pumpfun_custom_6005_graduated"),
    ("Custom:6005",                   "pumpfun_custom_6005_graduated"),
    # No pool / account not found
    ("AccountNotFound",              "pumpswap_no_pool"),
    ("invalid account data",         "pumpswap_no_pool"),
    # Pool state layout mismatch
    ("InvalidAccountData",           "pumpswap_bad_pool_layout"),
    ("custom program error: 0x",     "pumpswap_bad_pool_layout"),
    # Token-2022 ATA derivation errors
    ("invalid program id",           "pumpswap_token2022_ata_error"),
    ("IncorrectProgramId",           "pumpswap_token2022_ata_error"),
    # Transfer hook errors (T22 extension)
    ("transfer hook",                "pumpswap_transfer_hook_error"),
    ("TransferHook",                 "pumpswap_transfer_hook_error"),
    # Honeypot / sell restricted
    ("insufficient funds",           "pumpswap_honeypot_or_sell_restricted"),
    ("TokenAccountNotFound",         "pumpswap_honeypot_or_sell_restricted"),
    # Generic simulation failure
    ("Error",                        "pumpswap_simulation_failed"),
]


def simulate_sell(tx_bytes: bytes, rpc_url: str) -> tuple[bool, str, list[str]]:
    """
    Simulate the sell TX via simulateTransaction RPC.

    Returns (ok, error_class, log_lines) where:
      ok           — True if simulation succeeded with no error
      error_class  — "" if ok, else one of:
                     "pumpswap_no_pool"
                     "pumpswap_bad_pool_layout"
                     "pumpswap_token2022_ata_error"
                     "pumpswap_transfer_hook_error"
                     "pumpswap_honeypot_or_sell_restricted"
                     "pumpswap_simulation_failed"
      log_lines    — list of program log strings from simulation
    """
    tx_b64 = base64.b64encode(tx_bytes).decode()
    try:
        data = _rpc(rpc_url, {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "simulateTransaction",
            "params": [
                tx_b64,
                {
                    "encoding":               "base64",
                    "commitment":             "confirmed",
                    "sigVerify":              False,
                    "replaceRecentBlockhash": True,
                },
            ],
        }, timeout=15)
    except Exception as e:
        log.warning("simulateTransaction RPC error: %s", e)
        return False, "pumpswap_simulation_failed", [str(e)]

    result = data.get("result", {})
    value  = result.get("value", {})
    err    = value.get("err")
    logs   = value.get("logs") or []

    if err is None:
        # No error → simulation succeeded
        log.info("simulate_sell OK  logs=%d lines", len(logs))
        return True, "", list(logs)

    # Classify the error from logs and error object
    err_str  = str(err)
    logs_str = "\n".join(logs)
    combined = err_str + "\n" + logs_str

    error_class = "pumpswap_simulation_failed"  # default
    for pattern, cls in _SIM_ERROR_PATTERNS:
        if pattern.lower() in combined.lower():
            error_class = cls
            break

    log.warning(
        "simulate_sell FAILED  error_class=%s  err=%s  first_log=%s",
        error_class, err_str, logs[0] if logs else "(none)",
    )
    return False, error_class, list(logs)
