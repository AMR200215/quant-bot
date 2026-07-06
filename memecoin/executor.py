"""
On-chain execution — PumpPortal-native path (primary) + Jupiter fallback.

Controlled by EXECUTOR_BACKEND in config.py:
  "pumpportal" — POST pumpportal.fun/api/trade-local → sign → Helius RPC
  "jupiter"    — Jupiter quote + swap → Helius RPC (fallback / legacy)

Bugs fixed vs prior version:
  1. _confirm_tx now checks status["err"] — reverted txs are failures, not phantom
     successes. Prior: any confirmed tx (even reverting on slippage) returned True.
  2. Fill price from on-chain balance delta, not from pre-swap quote outAmount.
     Prior: quote outAmount used even on reverted trades → fake fill recorded.
  3. Asymmetric slippage: 25% buys, 35% sells (pump.fun tokens move 20-100%
     between signal and execution; 5% was constant revert source).
  4. PumpPortal path does not send to Jito — sends to Helius RPC directly.
     Prior: tipless tx to Jito = no inclusion incentive. Jito only for $50+ trades
     where Jito tip cost is small relative to position and top-of-block matters.

Environment variables:
  SOLANA_PRIVATE_KEY — base58-encoded wallet private key
  SOLANA_RPC_URL     — Helius (or other) RPC endpoint
"""

import base64
import concurrent.futures
import hashlib as _hashlib
import logging
import os
import struct as _struct
import threading
import time

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SOL_MINT     = "So11111111111111111111111111111111111111112"
USDC_MINT    = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
SOL_DECIMALS = 9

JUPITER_QUOTE_URL = "https://lite-api.jup.ag/swap/v1/quote"
JUPITER_SWAP_URL  = "https://lite-api.jup.ag/swap/v1/swap"

PUMPPORTAL_TRADE_URL = "https://pumpportal.fun/api/trade-local"

# Buy: 30% — aligns with abort_tripwire threshold (fill > signal*1.30 → abort) AND
#             entry gate (SLIPPAGE_GATE_RT_PCT=0.30). All three are intentionally identical:
#             tokens 0-30% above signal → held normally; tokens >30% → revert (no fill-abort
#             round trip). Graduated tokens fill within 2% of Jupiter quote → no regression.
# Sell: 35% — stop-loss that fails to execute is the worst outcome in the system.
SLIPPAGE_BUY_PCT        = 30
SLIPPAGE_BUY_PCT_RETRY  = 30   # retry at same cap — 6063 at 30% means token moved >30%, revert again is correct
SLIPPAGE_SELL_PCT       = 35

PRIORITY_FEE_SOL  = 0.0005   # floor fallback (~$0.085 at $170 SOL)

# Sell ladder: on revert, escalate slippage and fee immediately.
# fee_sol values are FLOOR minimums — dynamic Helius estimate is used when higher.
#   Step 1: 35% / "High"       — covers routine price movement
#   Step 2: 60% / "VeryHigh"   — covers fast dump during stop-loss window
#   Step 3: 98% / "UnsafeMax"  — last resort; get out at any cost
SELL_LADDER = [
    (35,  0.0005, "High"),       # (slippage_pct, fee_floor_sol, helius_level)
    (60,  0.0015, "VeryHigh"),
    (98,  0.005,  "UnsafeMax"),
]

# Urgent sell ladder: stop-losses, rugs, and forced exits (hard_stop, trailing_stop,
# feed_blind, graduated_exit, dev_dump). Price is already moving adversely, so the
# 35%/60% rungs will revert — skip straight to 98% slippage on the first attempt.
# Fee escalation (High→VeryHigh→UnsafeMax) only fires when the previous attempt was
# unconfirmed (didn't land). Reverts at 98% = genuine no-liquidity, not slippage miss.
URGENT_SELL_LADDER = [
    (98, 0.0005, "High"),       # (slippage_pct, fee_floor_sol, helius_level)
    (98, 0.0015, "VeryHigh"),   # escalates only if previous step was unconfirmed
    (98, 0.005,  "UnsafeMax"),
]

# pump.fun program — used as anchor account for priority fee estimation
_PUMPFUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
# Estimated compute units per pump.fun buy/sell tx (used to convert microlamports/CU → SOL)
_CU_BUDGET_EST   = 200_000

# ---------------------------------------------------------------------------
# pump.fun local build — eliminates PumpPortal HTTP round-trip (~500-1000ms)
# ---------------------------------------------------------------------------
# Discriminators = sha256("global:<method>")[:8] (Anchor IDL convention).
# Verified: hashlib.sha256(b"global:buy").digest()[:8]  == [102,6,61,18,1,218,235,234]
#           hashlib.sha256(b"global:sell").digest()[:8] == [51,230,133,164,1,127,131,173]
_PUMP_BUY_DISCRIMINATOR  = _hashlib.sha256(b"global:buy").digest()[:8]
_PUMP_SELL_DISCRIMINATOR = _hashlib.sha256(b"global:sell").digest()[:8]

# Fixed accounts (derived once from mainnet; pump.fun program does not rotate these)
_PUMPFUN_GLOBAL_PDA  = "4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf"
_PUMPFUN_EVENT_AUTH  = "Ce6TQqeHC9p8KetsN6JsjHK7UTZk7nasjjnr7XxXp9F1"
_TOKEN_PROGRAM_ID    = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
_TOKEN22_PROGRAM_ID  = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"   # Token-2022
_ATA_PROGRAM_ID      = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJe1bRS"
_RENT_SYSVAR_ID      = "SysvarRent111111111111111111111111111111111"
_SYSTEM_PROGRAM_ID   = "11111111111111111111111111111111"
_COMPUTE_BUDGET_ID   = "ComputeBudget111111111111111111111111111111"

# Per-mint token program cache (Token SPL vs Token-2022).
# pump.fun migrated all new tokens to Token-2022 in 2025; old tokens stay SPL.
# Populated lazily on first local build per mint; never evicted (mints don't change).
_mint_token_program_cache: dict[str, str] = {}
_mint_token_program_lock = threading.Lock()

# Fee recipient lives in the pump.fun global account (offset 41:73).
# Read once from chain and cached; falls back to hardcoded if RPC unavailable.
_pumpfun_fee_recipient_cache: str = ""
_pumpfun_fee_recipient_lock        = threading.Lock()

# 8 Jito tip accounts (only used when EXECUTOR_BACKEND == "jupiter" + JITO_TIPS = True)
_JITO_TIP_ACCOUNTS = [
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
    "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
    "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
    "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
    "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
]
JITO_TIP_LAMPORTS = 1_000_000   # 0.001 SOL (~$0.17) — only applied when Jito path active
JITO_RPC_URL      = "https://mainnet.block-engine.jito.wtf/api/v1/transactions"

SOLANA_RPC           = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
SOLANA_RPC_FALLBACK  = "https://api.mainnet-beta.solana.com"   # Solana public (free)
SOLANA_RPC_FALLBACK2 = "https://rpc.ankr.com/solana"           # Ankr free tier (no auth)

# Which execution backend to use.  "pumpportal" is the default; set to
# "jupiter" to fall back to the old path (useful for A/B testing).
try:
    from memecoin.config import EXECUTOR_BACKEND as _BACKEND
except ImportError:
    _BACKEND = "pumpportal"
EXECUTOR_BACKEND = _BACKEND

try:
    from memecoin.config import LIVE_DRY_RUN as _LIVE_DRY_RUN
except ImportError:
    _LIVE_DRY_RUN = False
LIVE_DRY_RUN = _LIVE_DRY_RUN

try:
    from memecoin.config import SLIPPAGE_GATE_DEX_PCT as _SLIPPAGE_GATE_DEX_PCT
    from memecoin.config import SLIPPAGE_GATE_RT_PCT  as _SLIPPAGE_GATE_RT_PCT
except ImportError:
    _SLIPPAGE_GATE_DEX_PCT = 0.50
    _SLIPPAGE_GATE_RT_PCT  = 0.30
SLIPPAGE_GATE_DEX_PCT = _SLIPPAGE_GATE_DEX_PCT
SLIPPAGE_GATE_RT_PCT  = _SLIPPAGE_GATE_RT_PCT

# Pre-buy reserve constants (lamports).
# Trade lamports + all three reserves must fit within the wallet's free balance.
_RENT_RESERVE      = 2_039_280   # token-account rent exemption
_FEE_RESERVE       = 100_000    # tx priority-fee budget
_PRESIGNED_RESERVE = 50_000     # headroom for presigned emergency exit


# ---------------------------------------------------------------------------
# Blockhash cache — refreshed every 2s by background thread.
# Eliminates one getLatestBlockhash RPC call from the local-build path.
# PumpPortal path fetches its own blockhash server-side; cache used by
# Jupiter fallback and future local build.
# ---------------------------------------------------------------------------
_blockhash_cache: dict = {"blockhash": None, "ts": 0.0}
_blockhash_lock  = threading.Lock()


def _blockhash_refresher():
    while True:
        try:
            resp = requests.post(
                SOLANA_RPC,
                json={"jsonrpc": "2.0", "id": 1,
                      "method": "getLatestBlockhash",
                      "params": [{"commitment": "confirmed"}]},
                timeout=5,
            )
            bh = resp.json().get("result", {}).get("value", {}).get("blockhash")
            if bh:
                with _blockhash_lock:
                    _blockhash_cache["blockhash"] = bh
                    _blockhash_cache["ts"]        = time.time()
        except Exception:
            pass
        time.sleep(2)


threading.Thread(target=_blockhash_refresher, daemon=True,
                 name="blockhash-cache").start()


def _get_cached_blockhash() -> str | None:
    """Return cached blockhash if < 5s old, else None."""
    with _blockhash_lock:
        if time.time() - _blockhash_cache["ts"] < 5.0:
            return _blockhash_cache["blockhash"]
    return None


# ---------------------------------------------------------------------------
# pump.fun local build helpers
# ---------------------------------------------------------------------------

def _pumpfun_fee_recipient() -> str:
    """Return pump.fun fee_recipient pubkey, reading from global account once then caching."""
    global _pumpfun_fee_recipient_cache
    with _pumpfun_fee_recipient_lock:
        if _pumpfun_fee_recipient_cache:
            return _pumpfun_fee_recipient_cache
    try:
        resp = _rpc_post({
            "jsonrpc": "2.0", "id": 1,
            "method": "getAccountInfo",
            "params": [_PUMPFUN_GLOBAL_PDA, {"encoding": "base64"}],
        }, timeout=5)
        raw  = base64.b64decode(resp.json()["result"]["value"]["data"][0])
        # Layout: 8 discriminator + 1 initialized + 32 authority + 32 feeRecipient
        from solders.pubkey import Pubkey
        fee_pk = str(Pubkey.from_bytes(raw[41:73]))
        with _pumpfun_fee_recipient_lock:
            _pumpfun_fee_recipient_cache = fee_pk
        log.info("pump.fun fee_recipient cached: %s", fee_pk)
        return fee_pk
    except Exception as e:
        log.warning("pump.fun fee_recipient read failed: %s — using hardcoded fallback", e)
        # Fallback: known mainnet fee recipient (valid as of 2026-06)
        return "62qc2CNXwrYqQScmEdiZFFAnJR262PxWEuNQtxfafNgV"


def _derive_ata(owner_str: str, mint_str: str, token_program_id: str = _TOKEN_PROGRAM_ID) -> str:
    """Derive Associated Token Account address for owner+mint.

    token_program_id: pass _TOKEN22_PROGRAM_ID for Token-2022 mints.
    """
    from solders.pubkey import Pubkey
    ata_prog = Pubkey.from_string(_ATA_PROGRAM_ID)
    owner    = Pubkey.from_string(owner_str)
    mint     = Pubkey.from_string(mint_str)
    tok_prog = Pubkey.from_string(token_program_id)
    ata, _   = Pubkey.find_program_address(
        [bytes(owner), bytes(tok_prog), bytes(mint)],
        ata_prog,
    )
    return str(ata)


def _pumpfun_mint_token_program(mint_str: str) -> str:
    """Return the token program ID for a given mint (cached after first call).

    pump.fun switched all new tokens to Token-2022 in 2025. The owner field
    of the mint account identifies the token program.
    Returns _TOKEN22_PROGRAM_ID or _TOKEN_PROGRAM_ID.
    """
    with _mint_token_program_lock:
        if mint_str in _mint_token_program_cache:
            return _mint_token_program_cache[mint_str]
    try:
        resp = _rpc_post({
            "jsonrpc": "2.0", "id": 1,
            "method":  "getAccountInfo",
            "params":  [mint_str, {"encoding": "base64"}],
        }, timeout=5)
        owner = (resp.json().get("result") or {}).get("value", {}).get("owner", "")
    except Exception:
        owner = ""
    prog = _TOKEN22_PROGRAM_ID if owner == _TOKEN22_PROGRAM_ID else _TOKEN_PROGRAM_ID
    with _mint_token_program_lock:
        _mint_token_program_cache[mint_str] = prog
    log.debug("mint_token_program  mint=%s  prog=%s", mint_str[:8], "T22" if prog == _TOKEN22_PROGRAM_ID else "SPL")
    return prog


def _pumpfun_read_bc(bc_pubkey: str) -> tuple[int, int]:
    """
    Fetch pump.fun bonding curve account → (v_token_reserves, v_sol_reserves).
    Account data layout (after 8-byte Anchor discriminator):
      offset  8: virtual_token_reserves  u64 LE
      offset 16: virtual_sol_reserves    u64 LE
    """
    resp = _rpc_post({
        "jsonrpc": "2.0", "id": 1,
        "method": "getAccountInfo",
        "params": [bc_pubkey, {"encoding": "base64"}],
    }, timeout=5)
    val = resp.json()["result"]["value"]
    if val is None:
        raise ValueError(f"Bonding curve not found: {bc_pubkey[:8]} — token may have graduated")
    raw = base64.b64decode(val["data"][0])
    if len(raw) < 24:
        raise ValueError(f"BC account data too short ({len(raw)} bytes) for {bc_pubkey[:8]}")
    v_token, v_sol = _struct.unpack_from("<QQ", raw, 8)
    return v_token, v_sol


def _pumpfun_local_build_tx(
    action: str,            # "buy" | "sell"
    wallet_pubkey: str,
    token_mint: str,
    keypair,
    sol_amount: float = 0.0,    # buy: SOL to spend (float)
    token_amount: int = 0,       # sell: exact raw token count
    slippage_pct: int = 30,
    priority_fee_sol: float = PRIORITY_FEE_SOL,
) -> bytes:
    """
    Build and sign a pump.fun bonding-curve buy/sell instruction locally.
    Eliminates the PumpPortal HTTP round-trip (~500-1000ms per buy or sell).

    Returns fully-signed VersionedTransaction bytes ready for _send_transaction().
    Raises on any failure so callers can fall back to PumpPortal.

    Do NOT call for graduated tokens (bonding curve complete → Raydium):
    those always route through PumpPortal pool="auto" → Jupiter.
    """
    from solders.pubkey import Pubkey
    from solders.instruction import Instruction, AccountMeta
    from solders.message import MessageV0
    from solders.hash import Hash
    from solders.transaction import VersionedTransaction as _VT

    # Detect token program (SPL vs Token-2022) — cached after first call per mint
    tok_prog_id = _pumpfun_mint_token_program(token_mint)

    prog         = Pubkey.from_string(_PUMPFUN_PROGRAM)
    mint_pk      = Pubkey.from_string(token_mint)
    wallet_pk    = Pubkey.from_string(wallet_pubkey)
    system_pk    = Pubkey.from_string(_SYSTEM_PROGRAM_ID)
    tok_prog_pk  = Pubkey.from_string(tok_prog_id)
    rent_pk      = Pubkey.from_string(_RENT_SYSVAR_ID)
    ata_prog_pk  = Pubkey.from_string(_ATA_PROGRAM_ID)
    compute_pk   = Pubkey.from_string(_COMPUTE_BUDGET_ID)
    global_pk    = Pubkey.from_string(_PUMPFUN_GLOBAL_PDA)
    event_pk     = Pubkey.from_string(_PUMPFUN_EVENT_AUTH)
    fee_recv_pk  = Pubkey.from_string(_pumpfun_fee_recipient())

    # Derive bonding curve PDA and its ATA (using detected token program for seeds)
    bc_pk_str    = str(Pubkey.find_program_address([b"bonding-curve", bytes(mint_pk)], prog)[0])
    bc_pk        = Pubkey.from_string(bc_pk_str)
    assoc_bc_pk  = Pubkey.from_string(_derive_ata(bc_pk_str, token_mint, tok_prog_id))
    assoc_usr_pk = Pubkey.from_string(_derive_ata(wallet_pubkey, token_mint, tok_prog_id))

    # Read bonding curve reserves (1 RPC call — ~30ms on Helius)
    v_token, v_sol = _pumpfun_read_bc(bc_pk_str)

    if action == "buy":
        sol_in_lam = int(sol_amount * 1e9)
        # Constant-product AMM: tokens_out = v_token * sol_in / (v_sol + sol_in)
        # pump.fun charges 1% buy fee on sol_in; effective sol = sol_in * 0.99
        eff_sol    = sol_in_lam * 99 // 100
        tokens_out = (v_token * eff_sol) // (v_sol + eff_sol) if (v_sol + eff_sol) > 0 else 0
        if tokens_out <= 0:
            raise ValueError(f"local buy: tokens_out=0 (v_token={v_token} v_sol={v_sol} sol={sol_in_lam})")
        min_tokens  = int(tokens_out * (1 - slippage_pct / 100))
        max_sol_lam = int(sol_in_lam * (1 + slippage_pct / 100))
        ix_data  = bytes(_PUMP_BUY_DISCRIMINATOR) + _struct.pack("<QQ", min_tokens, max_sol_lam)
        accounts = [
            AccountMeta(pubkey=global_pk,    is_signer=False, is_writable=False),
            AccountMeta(pubkey=fee_recv_pk,  is_signer=False, is_writable=True),
            AccountMeta(pubkey=mint_pk,      is_signer=False, is_writable=False),
            AccountMeta(pubkey=bc_pk,        is_signer=False, is_writable=True),
            AccountMeta(pubkey=assoc_bc_pk,  is_signer=False, is_writable=True),
            AccountMeta(pubkey=assoc_usr_pk, is_signer=False, is_writable=True),
            AccountMeta(pubkey=wallet_pk,    is_signer=True,  is_writable=True),
            AccountMeta(pubkey=system_pk,    is_signer=False, is_writable=False),
            AccountMeta(pubkey=tok_prog_pk,  is_signer=False, is_writable=False),
            AccountMeta(pubkey=rent_pk,      is_signer=False, is_writable=False),
            AccountMeta(pubkey=event_pk,     is_signer=False, is_writable=False),
            AccountMeta(pubkey=prog,         is_signer=False, is_writable=False),
        ]
        log.debug("local buy  token=%s  v_tok=%d  v_sol=%d  tokens_out=%d  min=%d  max_sol=%d",
                  token_mint[:8], v_token, v_sol, tokens_out, min_tokens, max_sol_lam)
    elif action == "sell":
        if token_amount <= 0:
            raise ValueError(f"local sell: token_amount={token_amount} — exact count required")
        # sol_out ≈ v_sol * token_amount / (v_token + token_amount)
        # Apply 1% pump.fun sell fee + slippage tolerance
        sol_out     = (v_sol * token_amount) // (v_token + token_amount) if (v_token + token_amount) > 0 else 0
        min_sol_lam = int(sol_out * 0.99 * (1 - slippage_pct / 100))
        ix_data  = bytes(_PUMP_SELL_DISCRIMINATOR) + _struct.pack("<QQ", token_amount, min_sol_lam)
        accounts = [
            AccountMeta(pubkey=global_pk,    is_signer=False, is_writable=False),
            AccountMeta(pubkey=fee_recv_pk,  is_signer=False, is_writable=True),
            AccountMeta(pubkey=mint_pk,      is_signer=False, is_writable=False),
            AccountMeta(pubkey=bc_pk,        is_signer=False, is_writable=True),
            AccountMeta(pubkey=assoc_bc_pk,  is_signer=False, is_writable=True),
            AccountMeta(pubkey=assoc_usr_pk, is_signer=False, is_writable=True),
            AccountMeta(pubkey=wallet_pk,    is_signer=True,  is_writable=True),
            AccountMeta(pubkey=system_pk,    is_signer=False, is_writable=False),
            AccountMeta(pubkey=ata_prog_pk,  is_signer=False, is_writable=False),
            AccountMeta(pubkey=tok_prog_pk,  is_signer=False, is_writable=False),
            AccountMeta(pubkey=event_pk,     is_signer=False, is_writable=False),
            AccountMeta(pubkey=prog,         is_signer=False, is_writable=False),
        ]
        log.debug("local sell  token=%s  tokens=%d  min_sol=%d", token_mint[:8], token_amount, min_sol_lam)
    else:
        raise ValueError(f"_pumpfun_local_build_tx: unknown action {action!r}")

    # ComputeBudget instructions
    micro_lam = max(1, int(priority_fee_sol * 1e15 / _CU_BUDGET_EST))   # microlamports per CU
    cb_limit  = Instruction(
        program_id=compute_pk,
        data=bytes([0x02]) + _struct.pack("<I", _CU_BUDGET_EST),
        accounts=[],
    )
    cb_price  = Instruction(
        program_id=compute_pk,
        data=bytes([0x03]) + _struct.pack("<Q", micro_lam),
        accounts=[],
    )
    pump_ix = Instruction(program_id=prog, data=ix_data, accounts=accounts)

    blockhash = _get_cached_blockhash()
    if not blockhash:
        raise RuntimeError("Blockhash cache empty — local build blocked (refreshes every 2s)")

    msg = MessageV0.try_compile(
        payer=keypair.pubkey(),
        instructions=[cb_limit, cb_price, pump_ix],
        address_lookup_table_accounts=[],
        recent_blockhash=Hash.from_string(blockhash),
    )
    return bytes(_VT(msg, [keypair]))


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
def _load_solders():
    try:
        from solders.keypair import Keypair
        from solders.transaction import VersionedTransaction
        import base58 as _base58
        return Keypair, VersionedTransaction, _base58
    except ImportError as e:
        raise RuntimeError(
            "solders and base58 are required for live trading: "
            "pip install solders base58"
        ) from e


_keypair_cache = None
_keypair_lock  = threading.Lock()

def _get_keypair():
    """Return the wallet Keypair, constructing it once and caching as a singleton."""
    global _keypair_cache
    if _keypair_cache is not None:
        return _keypair_cache
    with _keypair_lock:
        if _keypair_cache is not None:
            return _keypair_cache
        Keypair, _, base58 = _load_solders()
        raw = os.getenv("SOLANA_PRIVATE_KEY", "")
        if not raw:
            raise RuntimeError("SOLANA_PRIVATE_KEY env var not set")
        _keypair_cache = Keypair.from_bytes(base58.b58decode(raw))
        return _keypair_cache


# ---------------------------------------------------------------------------
# Shared RPC helpers
# ---------------------------------------------------------------------------
def _rpc_post(payload: dict, timeout: int = 10) -> requests.Response:
    """POST to SOLANA_RPC with a three-tier fallback chain.

    Tier 1 — Helius (primary, paid):
      429 burst → retry once after 1.5s before escalating.
      Timeout or persistent 429 → Tier 2.

    Tier 2 — Solana public RPC (api.mainnet-beta.solana.com):
      Timeout or 429 → Tier 3.

    Tier 3 — Ankr free RPC (rpc.ankr.com/solana):
      Last resort; response returned regardless of status.
    """
    # ── Tier 1: Helius ──────────────────────────────────────────────────────
    _primary_failed = False
    try:
        resp = requests.post(SOLANA_RPC, json=payload, timeout=timeout)
        if resp.status_code == 429 and SOLANA_RPC != SOLANA_RPC_FALLBACK:
            log.warning("Primary RPC 429 — retrying Helius in 1.5s")
            time.sleep(1.5)
            resp = requests.post(SOLANA_RPC, json=payload, timeout=timeout)
        if resp.status_code not in (429, 503) or SOLANA_RPC == SOLANA_RPC_FALLBACK:
            return resp
        _primary_failed = True
    except requests.exceptions.Timeout:
        log.warning("Primary RPC unavailable (timeout) — falling back to mainnet-beta")
        _primary_failed = True
    except requests.exceptions.RequestException as e:
        log.warning("Primary RPC unavailable (%s) — falling back to mainnet-beta", e)
        _primary_failed = True

    if not _primary_failed:
        return resp  # type: ignore[return-value]  # 503 path falls through

    # ── Tier 2: Solana public RPC ───────────────────────────────────────────
    log.warning("Primary RPC still failing — falling back to mainnet-beta")
    try:
        resp2 = requests.post(SOLANA_RPC_FALLBACK, json=payload, timeout=timeout)
        if resp2.status_code not in (429, 503):
            return resp2
        log.warning("mainnet-beta also failing (%d) — trying Ankr", resp2.status_code)
    except requests.exceptions.Timeout:
        log.warning("mainnet-beta timeout — trying Ankr")
    except requests.exceptions.RequestException as e:
        log.warning("mainnet-beta unavailable (%s) — trying Ankr", e)

    # ── Tier 3: Ankr free RPC ───────────────────────────────────────────────
    log.warning("Primary RPC unavailable (429) — falling back to publicnode")
    return requests.post(SOLANA_RPC_FALLBACK2, json=payload, timeout=timeout)


def _token_balance(wallet_pubkey: str, token_mint: str, retries: int = 1) -> int:
    """Return raw token balance (smallest unit) for a wallet.

    Checks both SPL Token and Token-2022 accounts. pump.fun migrated all new
    tokens to Token-2022 in 2025; standard getTokenAccountsByOwner{mint:...}
    does not cross program boundaries, so we must query each program explicitly
    when {mint:...} returns nothing.

    Raises RuntimeError on RPC error so callers can decide how to handle it
    (e.g. fall back to quote price) rather than silently returning 0.
    """
    last_err = None
    for attempt in range(retries):
        if attempt > 0:
            time.sleep(2)
        try:
            # Use cached token program if known, otherwise try SPL first
            cached_prog = _mint_token_program_cache.get(token_mint)
            if cached_prog:
                # Known program — query directly with programId filter for efficiency
                resp = _rpc_post({
                    "jsonrpc": "2.0", "id": 1,
                    "method":  "getTokenAccountsByOwner",
                    "params":  [
                        wallet_pubkey,
                        {"programId": cached_prog},
                        {"encoding": "jsonParsed"},
                    ],
                })
                data = resp.json()
                if "result" not in data:
                    last_err = data.get("error", resp.status_code)
                    continue
                # Filter to our mint (programId returns all accounts for that program)
                accounts = [
                    a for a in data["result"].get("value", [])
                    if a["account"]["data"]["parsed"]["info"].get("mint") == token_mint
                ]
                if accounts:
                    return int(accounts[0]["account"]["data"]["parsed"]["info"]["tokenAmount"]["amount"])
                return 0
            else:
                # Unknown program — try SPL Token first (legacy + most non-pump tokens)
                resp = _rpc_post({
                    "jsonrpc": "2.0", "id": 1,
                    "method":  "getTokenAccountsByOwner",
                    "params":  [
                        wallet_pubkey,
                        {"mint": token_mint},
                        {"encoding": "jsonParsed"},
                    ],
                })
                data = resp.json()
                if "result" not in data:
                    last_err = data.get("error", resp.status_code)
                    continue
                accounts = data["result"].get("value", [])
                if accounts:
                    return int(accounts[0]["account"]["data"]["parsed"]["info"]["tokenAmount"]["amount"])
                # SPL returned nothing — check Token-2022
                resp22 = _rpc_post({
                    "jsonrpc": "2.0", "id": 1,
                    "method":  "getTokenAccountsByOwner",
                    "params":  [
                        wallet_pubkey,
                        {"programId": _TOKEN22_PROGRAM_ID},
                        {"encoding": "jsonParsed"},
                    ],
                })
                data22 = resp22.json()
                if "result" in data22:
                    accounts22 = [
                        a for a in data22["result"].get("value", [])
                        if a["account"]["data"]["parsed"]["info"].get("mint") == token_mint
                    ]
                    if accounts22:
                        # Cache for future calls
                        with _mint_token_program_lock:
                            _mint_token_program_cache[token_mint] = _TOKEN22_PROGRAM_ID
                        return int(accounts22[0]["account"]["data"]["parsed"]["info"]["tokenAmount"]["amount"])
                return 0
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Token balance unavailable: {last_err}")


def _token_decimals_from_rpc(wallet_pubkey: str, token_mint: str) -> int:
    """Return token decimals from the wallet's token account. Defaults to 6."""
    try:
        # Try SPL first, then Token-2022 (matching _token_balance logic)
        resp = _rpc_post({
            "jsonrpc": "2.0", "id": 1,
            "method":  "getTokenAccountsByOwner",
            "params":  [wallet_pubkey, {"mint": token_mint}, {"encoding": "jsonParsed"}],
        })
        accounts = resp.json().get("result", {}).get("value", [])
        if not accounts:
            prog = _mint_token_program_cache.get(token_mint, _TOKEN22_PROGRAM_ID)
            resp22 = _rpc_post({
                "jsonrpc": "2.0", "id": 1,
                "method":  "getTokenAccountsByOwner",
                "params":  [wallet_pubkey, {"programId": prog}, {"encoding": "jsonParsed"}],
            })
            accounts = [
                a for a in resp22.json().get("result", {}).get("value", [])
                if a["account"]["data"]["parsed"]["info"].get("mint") == token_mint
            ]
        if accounts:
            return int(
                accounts[0]["account"]["data"]["parsed"]["info"]["tokenAmount"]["decimals"]
            )
    except Exception:
        pass
    return 6  # pump.fun default


def _sol_balance(wallet_pubkey: str) -> int:
    """Return wallet SOL balance in lamports. Raises on RPC error.

    Uses 'confirmed' commitment so balance reflects txs within ~2s,
    not ~14s (finalized default). This is the key fix for sol_recv=0.
    """
    resp = _rpc_post({
        "jsonrpc": "2.0", "id": 1,
        "method":  "getBalance",
        "params":  [wallet_pubkey, {"commitment": "confirmed"}],
    })
    data = resp.json()
    if "result" not in data:
        raise RuntimeError(f"SOL balance RPC error: {data.get('error', resp.status_code)}")
    return int(data["result"]["value"])


def _confirm_tx(sig: str, max_wait: int = 40,
                t_sent: float = 0.0) -> tuple[bool, object]:
    """
    Poll getSignatureStatuses until confirmed/finalized or timeout.

    Returns (success: bool, err: object)
      success=True,  err=None   — tx landed and succeeded
      success=False, err=dict   — tx landed but REVERTED (slippage exceeded, etc.)
      success=False, err=None   — timeout, tx not seen within max_wait seconds

    IMPORTANT: callers must check err even when success=False. A reverted tx
    is on-chain; do not record it as a success.

    t_sent: time.time() when sendTransaction was called.  When provided, logs
    the confirmed slot so we can measure whether the tx is landing in 1-3 slots
    (fast — detection lag is the bottleneck) or many more (slow — landing is
    the bottleneck, and maxRetries / fee changes help).
    Detection-lag notes:
      searchTransactionHistory=False — sig is always fresh (<40s), recent-cache
        is sufficient and the call is ~3× faster.  Falls back gracefully if cache
        misses (None result → next poll).
      Poll interval: 0.5s for the first 6 polls (~3s window), then 2s.
        The first slot usually confirms in 1-3 blocks (0.4-1.2s); tight polling
        catches it early.  Switching to 2s after 3s avoids burning Helius quota
        while waiting for slower-landing txs.
      429 backoff: capped at 2s for the first 15s of waiting, then allows 4s.
        Old escalation to 10s was ballooning detection time under rate-limit.
    """
    _t0      = t_sent or time.time()
    deadline = time.time() + max_wait
    _poll    = 0
    backoff  = 2
    while time.time() < deadline:
        try:
            resp = _rpc_post({
                "jsonrpc": "2.0", "id": 1,
                "method":  "getSignatureStatuses",
                "params":  [[sig], {"searchTransactionHistory": False}],
            })
            if resp.status_code == 429:
                _wait_429 = 2 if (time.time() - _t0) < 15 else 4
                log.debug("getSignatureStatuses 429 — backoff %ds  sig=%s", _wait_429, sig[:16])
                time.sleep(_wait_429)
                continue
            status = (resp.json().get("result", {}).get("value") or [None])[0]
            if status:
                conf = status.get("confirmationStatus")
                err  = status.get("err")
                if conf in ("confirmed", "finalized"):
                    if err is not None:
                        log.warning("Tx REVERTED on-chain  sig=%s  err=%s", sig[:16], err)
                        return False, err
                    # ── Step 0 instrumentation ────────────────────────────────
                    # Logs confirmed slot so we can compare to send-slot and
                    # determine if latency is tx-landing (slot delta large) or
                    # detection-lag (slot delta small, t_detected large).
                    _t_detected = time.time()
                    if t_sent:
                        log.warning(
                            "TX LANDED  sig=%s  slot=%s  detected=+%.1fs_after_send",
                            sig[:16], status.get("slot", "?"),
                            _t_detected - _t0,
                        )
                    return True, None
        except Exception:
            pass
        _poll += 1
        time.sleep(0.5 if _poll <= 6 else 2)
    return False, None   # timeout


def _fill_from_transaction(sig: str, wallet: str, token_address: str,
                           size_usd: float, retries: int = 3) -> tuple[float | None, bool]:
    """
    Fetch a confirmed tx and compute fill price from on-chain token balance deltas.

    Returns (fill_price, entry_estimated, tokens_received_raw):
      fill_price          — real fill derived from postTokenBalances; None if unavailable
      entry_estimated     — True if fill_price is None (caller should use Jupiter quote fallback
                            and tag the position entry_estimated=True)
      tokens_received_raw — exact raw integer token count received (before decimal division).
                            Used for known-balance TP sells — eliminates the RPC query lag
                            that causes zero_balance_partial on fast-pumping tokens.
                            0 when the delta cannot be read.

    Uses one getTransaction call instead of getTokenAccounts + getTokenDecimals.
    Replaces three separate RPC calls and eliminates the stale-Jupiter-quote entry bug.
    """
    for attempt in range(retries):
        try:
            resp = _rpc_post({
                "jsonrpc": "2.0", "id": 1,
                "method":  "getTransaction",
                "params":  [sig, {
                    "encoding":                       "jsonParsed",
                    "commitment":                     "confirmed",
                    "maxSupportedTransactionVersion": 0,
                }],
            }, timeout=10)
            data = resp.json().get("result")
            if data is None:
                # Not indexed yet — retry
                time.sleep(2)
                continue

            meta = data.get("meta") or {}

            # Find the wallet's token balance delta for this mint
            pre_by_idx = {
                b["accountIndex"]: int(b["uiTokenAmount"]["amount"])
                for b in (meta.get("preTokenBalances") or [])
                if b.get("mint") == token_address
            }
            for b in (meta.get("postTokenBalances") or []):
                if b.get("mint") != token_address:
                    continue
                if b.get("owner") != wallet:
                    continue
                idx      = b["accountIndex"]
                post_amt = int(b["uiTokenAmount"]["amount"])
                decimals = int(b["uiTokenAmount"].get("decimals") or 6)
                pre_amt  = pre_by_idx.get(idx, 0)
                tokens_received = post_amt - pre_amt
                if tokens_received > 0:
                    tokens_human = tokens_received / 10 ** decimals
                    # Prefer actual SOL spent (tx preBalances/postBalances) over intended
                    # size_usd — includes priority fee and pump.fun 0.5% platform fee.
                    _sol_spent_lam = 0
                    try:
                        _pre_b  = meta.get("preBalances") or []
                        _post_b = meta.get("postBalances") or []
                        _akeys  = (data.get("transaction", {})
                                       .get("message", {})
                                       .get("accountKeys", []))
                        for _ai, _ak in enumerate(_akeys):
                            _apub = _ak.get("pubkey") if isinstance(_ak, dict) else str(_ak)
                            if _apub == wallet and _ai < len(_pre_b) and _ai < len(_post_b):
                                _sol_spent_lam = max(0, _pre_b[_ai] - _post_b[_ai])
                                break
                    except Exception:
                        pass
                    if _sol_spent_lam > 0:
                        fill_price = (_sol_spent_lam / 1e9 * _sol_price_usd()) / tokens_human
                    else:
                        fill_price = size_usd / tokens_human   # fallback: intended spend
                    log.info("fill_from_tx  sig=%s  tokens=%d  decimals=%d  sol_lam=%d  fill=$%.10f",
                             sig[:16], tokens_received, decimals, _sol_spent_lam, fill_price)
                    return fill_price, False, tokens_received

            log.warning("fill_from_tx: no matching token delta for %s in tx %s",
                        token_address[:8], sig[:16])
            return None, True, 0

        except Exception as e:
            log.debug("fill_from_tx attempt %d/%d failed: %s", attempt + 1, retries, e)
            time.sleep(2)

    log.warning("fill_from_tx: all %d attempts failed for sig=%s", retries, sig[:16])
    return None, True, 0


def _sol_delta_from_tx(sig: str, wallet: str, action: str = "sell") -> int:
    """
    Return wallet SOL delta (lamports) from a confirmed tx's preBalances/postBalances.

    action="sell" → SOL received  (postBalance − preBalance, clamped ≥ 0).
    action="buy"  → SOL spent     (preBalance  − postBalance, clamped ≥ 0).

    Fires a single getTransaction call. Returns 0 on any failure — never raises.
    Use when getBalance settle-lag gives sol_received=0 after 5 retries.
    """
    try:
        resp = _rpc_post({
            "jsonrpc": "2.0", "id": 1,
            "method":  "getTransaction",
            "params":  [sig, {
                "encoding":                       "jsonParsed",
                "commitment":                     "confirmed",
                "maxSupportedTransactionVersion": 0,
            }],
        }, timeout=10)
        data = resp.json().get("result")
        if not data:
            return 0
        meta      = data.get("meta") or {}
        pre_bals  = meta.get("preBalances") or []
        post_bals = meta.get("postBalances") or []
        acct_keys = (data.get("transaction", {})
                         .get("message", {})
                         .get("accountKeys", []))
        for _i, _k in enumerate(acct_keys):
            _pub = _k.get("pubkey") if isinstance(_k, dict) else str(_k)
            if _pub == wallet and _i < len(pre_bals) and _i < len(post_bals):
                pre, post = pre_bals[_i], post_bals[_i]
                return max(0, post - pre) if action == "sell" else max(0, pre - post)
    except Exception:
        pass
    return 0


_sol_price_cache: dict = {"price": 170.0, "ts": 0.0}
_sol_price_lock  = threading.Lock()
_SOL_PRICE_TTL   = 60.0   # seconds before refreshing from Jupiter

def _sol_price_usd() -> float:
    """Return SOL/USD price, cached for 60s. On fetch failure returns last cached value."""
    now = time.time()
    with _sol_price_lock:
        if now - _sol_price_cache["ts"] < _SOL_PRICE_TTL:
            return _sol_price_cache["price"]
    # Stale — fetch outside lock to avoid blocking callers during the network round-trip.
    try:
        resp = requests.get(
            JUPITER_QUOTE_URL,
            params={
                "inputMint":  SOL_MINT,
                "outputMint": USDC_MINT,
                "amount":     1_000_000_000,
            },
            timeout=5,
        )
        resp.raise_for_status()
        price = round(float(resp.json()["outAmount"]) / 1e6, 4)
        with _sol_price_lock:
            _sol_price_cache["price"] = price
            _sol_price_cache["ts"]    = time.time()
        return price
    except Exception as e:
        log.warning("SOL price fetch failed: %s — returning cached $%.2f", e, _sol_price_cache["price"])
        # Bump ts so we don't hammer Jupiter on repeated failures inside the same cycle.
        with _sol_price_lock:
            _sol_price_cache["ts"] = time.time()
        return _sol_price_cache["price"]


def _helius_priority_fee(token_mint: str, level: str = "High") -> float:
    """
    Query Helius getPriorityFeeEstimate for a pump.fun trade on token_mint.
    Returns recommended priority fee in SOL.
    Falls back to PRIORITY_FEE_SOL on any error or missing key.

    level: "High" (buys + sell step 1) | "VeryHigh" (sell step 2) | "UnsafeMax" (sell step 3)
    """
    api_key = os.getenv("HELIUS_API_KEY", "")
    if not api_key:
        return PRIORITY_FEE_SOL
    try:
        resp = requests.post(
            f"https://mainnet.helius-rpc.com/?api-key={api_key}",
            json={
                "jsonrpc": "2.0",
                "id":      1,
                "method":  "getPriorityFeeEstimate",
                "params":  [{
                    "accountKeys": [_PUMPFUN_PROGRAM, token_mint],
                    "options":     {"priorityLevel": level},
                }],
            },
            timeout=3,
        )
        micro_lamports = resp.json()["result"]["priorityFeeEstimate"]
        fee_lamports   = (micro_lamports * _CU_BUDGET_EST) / 1_000_000
        fee_sol        = fee_lamports / 1e9
        # Floor at PRIORITY_FEE_SOL, cap at 0.01 SOL (~$1.70) to avoid runaway fees
        fee_sol = max(PRIORITY_FEE_SOL, min(fee_sol, 0.01))
        log.debug("Helius priority fee (%s): %.0f µlam/CU → %.6f SOL", level, micro_lamports, fee_sol)
        return fee_sol
    except Exception as e:
        log.debug("getPriorityFeeEstimate failed (%s) — using floor %.4f SOL", e, PRIORITY_FEE_SOL)
        return PRIORITY_FEE_SOL


def _send_transaction(signed_bytes: bytes) -> str:
    """
    Send a signed tx via Helius RPC.
    Returns the transaction signature string.
    Does NOT use Jito — Helius priority fee in the tx is sufficient for $1-5 trades.
    """
    payload = {
        "jsonrpc": "2.0",
        "id":      1,
        "method":  "sendTransaction",
        "params":  [
            base64.b64encode(signed_bytes).decode(),
            {"encoding": "base64", "skipPreflight": True, "maxRetries": 0},
        ],
    }
    resp = _rpc_post(payload, timeout=20)
    resp.raise_for_status()
    result = resp.json()
    if "error" in result:
        raise RuntimeError(f"sendTransaction RPC error: {result['error']}")
    return result["result"]


# ---------------------------------------------------------------------------
# PumpPortal execution path
# ---------------------------------------------------------------------------

def _pumpportal_build_tx(
    wallet_pubkey: str,
    action: str,          # "buy" | "sell"
    token_mint: str,
    amount,               # SOL float (buy) | "100%" or token count str (sell)
    denominated_in_sol: bool,
    slippage_pct: int,
    priority_fee_sol: float = PRIORITY_FEE_SOL,
    pool: str = "auto",   # "auto" (default), "pump-amm" (PumpSwap direct for graduated tokens)
) -> bytes:
    """
    Call PumpPortal trade-local API to get a pre-built serialized transaction.
    Returns raw transaction bytes ready for signing.

    Docs: https://pumpportal.fun/integrate/trading-api

    pool="auto":     routes bonding-curve OR graduated tokens automatically.
                     WARNING: returns Custom:6005 on graduated tokens when called
                     via trade-local (bonding curve lookup fails). Use "pump-amm".
    pool="pump-amm": PumpSwap direct — use for tokens that have graduated from
                     the bonding curve. Bypasses bonding curve lookup entirely.
    """
    payload = {
        "publicKey":       wallet_pubkey,
        "action":          action,
        "mint":            token_mint,
        "amount":          amount,
        "denominatedInSol": "true" if denominated_in_sol else "false",
        "slippage":        slippage_pct,
        "priorityFee":     priority_fee_sol,
        "pool":            pool,
    }
    resp = requests.post(PUMPPORTAL_TRADE_URL, data=payload, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(
            f"PumpPortal trade-local returned {resp.status_code}: {resp.text[:200]}"
        )
    if not resp.content:
        raise RuntimeError("PumpPortal trade-local returned empty response")
    return resp.content


# ---------------------------------------------------------------------------
# Jupiter execution path (fallback)
# ---------------------------------------------------------------------------

# Error classes emitted by the Jupiter retry helpers.
# These surface in logs and exit_route_attempts.csv.
_JUP_EC_RETRYING          = "jupiter_429_retrying"
_JUP_EC_EXHAUSTED         = "jupiter_429_exhausted"
_JUP_EC_QUOTE_FAILED      = "jupiter_quote_failed"
_JUP_EC_SWAP_BUILD_FAILED = "jupiter_swap_build_failed"
_JUP_EC_SIM_FAILED        = "jupiter_sim_failed"
_JUP_EC_READY_NO_SEND     = "jupiter_ready_but_send_disabled"


def _jup_backoff_ms(attempt: int) -> float:
    """Exponential backoff with jitter in milliseconds, capped at JUPITER_BACKOFF_MAX_MS."""
    import random
    try:
        from memecoin.config import (
            JUPITER_BACKOFF_BASE_MS   as _base,
            JUPITER_BACKOFF_MAX_MS    as _ceil,
            JUPITER_BACKOFF_JITTER_MS as _jitter,
        )
    except ImportError:
        _base, _ceil, _jitter = 250, 3000, 150
    raw = _base * (2 ** (attempt - 1))  # exponential: 250, 500, 1000, 2000, …
    capped = min(raw, _ceil)
    return capped + random.uniform(0, _jitter)


def _jup_get_quote(input_mint: str, output_mint: str, amount: int) -> dict:
    """
    Fetch Jupiter quote with retry on HTTP 429 or transient timeout.

    Config: JUPITER_MAX_RETRIES, JUPITER_BACKOFF_BASE_MS,
            JUPITER_BACKOFF_MAX_MS, JUPITER_BACKOFF_JITTER_MS
    """
    try:
        from memecoin.config import JUPITER_MAX_RETRIES as _max_retries
    except ImportError:
        _max_retries = 4

    params = {
        "inputMint":       input_mint,
        "outputMint":      output_mint,
        "amount":          amount,
        "dynamicSlippage": "true",
        "maxSlippageBps":  SLIPPAGE_BUY_PCT * 100,
    }
    last_exc: Exception | None = None
    for attempt in range(1, _max_retries + 1):
        try:
            resp = requests.get(JUPITER_QUOTE_URL, params=params, timeout=10)
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            wait_ms = _jup_backoff_ms(attempt)
            log.warning(
                "Jupiter quote timeout  endpoint=quote  attempt=%d/%d  backoff_ms=%.0f  mint=%s",
                attempt, _max_retries, wait_ms, input_mint[:8],
            )
            if attempt < _max_retries:
                time.sleep(wait_ms / 1000)
            continue

        if resp.status_code == 429:
            wait_ms = _jup_backoff_ms(attempt)
            if attempt < _max_retries:
                log.warning(
                    "Jupiter quote 429  error_class=%s  endpoint=quote  attempt=%d/%d"
                    "  backoff_ms=%.0f  mint=%s",
                    _JUP_EC_RETRYING, attempt, _max_retries, wait_ms, input_mint[:8],
                )
                time.sleep(wait_ms / 1000)
                continue
            log.error(
                "Jupiter quote 429  error_class=%s  endpoint=quote  attempt=%d/%d  mint=%s",
                _JUP_EC_EXHAUSTED, attempt, _max_retries, input_mint[:8],
            )
            resp.raise_for_status()  # raises HTTPError(429) to caller

        resp.raise_for_status()
        return resp.json()

    # All retries exhausted via timeout path
    log.error(
        "Jupiter quote  error_class=%s  attempt=%d/%d  mint=%s  last_exc=%s",
        _JUP_EC_EXHAUSTED, _max_retries, _max_retries, input_mint[:8], last_exc,
    )
    raise last_exc or RuntimeError(_JUP_EC_EXHAUSTED)


def _jup_build_swap_tx(quote: dict, wallet_pubkey: str,
                       slippage_bps: int = SLIPPAGE_SELL_PCT * 100,
                       priority_fee_lamports: int = None) -> bytes:
    """
    Build swap transaction via Jupiter swap API with retry on HTTP 429.
    Returns raw tx bytes. Does NOT sign or send.

    Config: JUPITER_MAX_RETRIES, JUPITER_BACKOFF_BASE_MS,
            JUPITER_BACKOFF_MAX_MS, JUPITER_BACKOFF_JITTER_MS
    """
    if priority_fee_lamports is None:
        priority_fee_lamports = int(PRIORITY_FEE_SOL * 1e9)
    try:
        from memecoin.config import JUPITER_MAX_RETRIES as _max_retries
    except ImportError:
        _max_retries = 4

    payload = {
        "quoteResponse":             quote,
        "userPublicKey":             wallet_pubkey,
        "wrapAndUnwrapSol":          True,
        "dynamicSlippage":           {"maxBps": slippage_bps},
        "prioritizationFeeLamports": priority_fee_lamports,
    }
    last_exc: Exception | None = None
    for attempt in range(1, _max_retries + 1):
        try:
            swap_resp = requests.post(JUPITER_SWAP_URL, json=payload, timeout=15)
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            wait_ms = _jup_backoff_ms(attempt)
            log.warning(
                "Jupiter swap timeout  endpoint=swap  attempt=%d/%d  backoff_ms=%.0f  wallet=%s",
                attempt, _max_retries, wait_ms, wallet_pubkey[:8],
            )
            if attempt < _max_retries:
                time.sleep(wait_ms / 1000)
            continue

        if swap_resp.status_code == 429:
            wait_ms = _jup_backoff_ms(attempt)
            if attempt < _max_retries:
                log.warning(
                    "Jupiter swap 429  error_class=%s  endpoint=swap  attempt=%d/%d"
                    "  backoff_ms=%.0f  wallet=%s",
                    _JUP_EC_RETRYING, attempt, _max_retries, wait_ms, wallet_pubkey[:8],
                )
                time.sleep(wait_ms / 1000)
                continue
            log.error(
                "Jupiter swap 429  error_class=%s  endpoint=swap  attempt=%d/%d  wallet=%s",
                _JUP_EC_EXHAUSTED, attempt, _max_retries, wallet_pubkey[:8],
            )
            swap_resp.raise_for_status()

        swap_resp.raise_for_status()
        return base64.b64decode(swap_resp.json()["swapTransaction"])

    log.error(
        "Jupiter swap  error_class=%s  attempt=%d/%d  wallet=%s  last_exc=%s",
        _JUP_EC_EXHAUSTED, _max_retries, _max_retries, wallet_pubkey[:8], last_exc,
    )
    raise last_exc or RuntimeError(_JUP_EC_EXHAUSTED)


# ---------------------------------------------------------------------------
# Buy gate daily counters (PART 5)
# ---------------------------------------------------------------------------

_BUY_GATE_LOCK   = threading.Lock()
_BUY_GATE_DATE   = ""
_BUY_GATE_COUNTS: dict = {}  # {reason: count} + "candidates_seen", "buys_allowed"


def _record_buy_gate(action: str, reason: str):
    """Record buy gate decision for daily summary."""
    global _BUY_GATE_DATE, _BUY_GATE_COUNTS
    from datetime import date as _dt
    today = str(_dt.today())
    with _BUY_GATE_LOCK:
        if today != _BUY_GATE_DATE:
            if _BUY_GATE_COUNTS:
                log.info("BUY GATE DAILY SUMMARY %s: %s", _BUY_GATE_DATE,
                         {k: v for k, v in sorted(_BUY_GATE_COUNTS.items())})
            _BUY_GATE_DATE   = today
            _BUY_GATE_COUNTS = {}
        _BUY_GATE_COUNTS["candidates_seen"] = _BUY_GATE_COUNTS.get("candidates_seen", 0) + 1
        if action == "allow":
            _BUY_GATE_COUNTS["buys_allowed"] = _BUY_GATE_COUNTS.get("buys_allowed", 0) + 1
        else:
            key = f"blocked_{reason}"
            _BUY_GATE_COUNTS[key] = _BUY_GATE_COUNTS.get(key, 0) + 1


# ---------------------------------------------------------------------------
# Pump.fun graduation oracle — single source of truth for buy-side
# ---------------------------------------------------------------------------

_GRAD_ORACLE_CACHE: dict = {}   # mint → (timestamp, result_dict)
_GRAD_ORACLE_TTL = 5.0          # seconds


def get_pumpfun_curve_complete(mint: str) -> dict:
    """
    Query the pump.fun bonding curve account to determine if the token has graduated.

    Returns:
      {
        "ok":      bool,               # False on RPC/parse errors
        "complete": bool | None,       # True=graduated, False=still on curve, None=unknown
        "reason":  str,                # complete_false|complete_true|account_missing|rpc_error|parse_error
        "bc_pda":  str,                # base58 bonding curve PDA
        "rpc_ms":  int,
      }

    BUY INVARIANT: MIGRATION_UNCERTAIN is not graduation proof. This oracle is the
    single source of truth for buy-side graduation detection. Do not re-add ExitRouter
    states (MIGRATION_UNCERTAIN etc.) as buy blockers. They are sell-side routing only.
    """
    # Check cache
    now = time.monotonic()
    cached = _GRAD_ORACLE_CACHE.get(mint)
    if cached:
        ts, result = cached
        if now - ts < _GRAD_ORACLE_TTL:
            return result

    t0 = time.monotonic()
    result = _get_pumpfun_curve_complete_uncached(mint)
    result["rpc_ms"] = int((time.monotonic() - t0) * 1000)

    # Cache successful reads and account_missing; do not cache transient RPC errors
    if result["reason"] not in ("rpc_error",):
        _GRAD_ORACLE_CACHE[mint] = (time.monotonic(), result)

    action = "block" if (result["complete"] is not False) else "allow"
    log.info(
        "GRAD ORACLE %s complete=%s reason=%s action=%s rpc_ms=%d",
        mint[:8], result["complete"], result["reason"], action, result["rpc_ms"],
    )
    return result


def _get_pumpfun_curve_complete_uncached(mint: str) -> dict:
    """Inner uncached implementation. See get_pumpfun_curve_complete()."""
    import base64 as _b64

    # Derive bonding curve PDA: seeds=[b"bonding-curve", mint_pubkey_bytes]
    # Same pattern as _pumpfun_local_build_tx (line 344)
    try:
        from solders.pubkey import Pubkey
        mint_pubkey = Pubkey.from_string(mint)
        prog        = Pubkey.from_string(_PUMPFUN_PROGRAM)
        bc_pda_pubkey, _ = Pubkey.find_program_address(
            [b"bonding-curve", bytes(mint_pubkey)], prog,
        )
        bc_pda = str(bc_pda_pubkey)
    except Exception as e:
        log.error("GRAD ORACLE PDA derivation failed for %s: %s", mint[:8], e)
        return {"ok": False, "complete": None, "reason": "rpc_error", "bc_pda": "", "rpc_ms": 0}

    # Fetch account via getAccountInfo using existing _rpc_post fallback chain
    try:
        resp = _rpc_post({
            "jsonrpc": "2.0", "id": 1,
            "method": "getAccountInfo",
            "params": [bc_pda, {"encoding": "base64", "commitment": "confirmed"}],
        })
        result_field = resp.json().get("result", {})
        value = result_field.get("value") if isinstance(result_field, dict) else None
    except Exception as e:
        log.warning("GRAD ORACLE RPC error for %s: %s", mint[:8], e)
        return {"ok": False, "complete": None, "reason": "rpc_error", "bc_pda": bc_pda, "rpc_ms": 0}

    # Account missing / null = curve closed or migrated
    if value is None:
        return {"ok": True, "complete": None, "reason": "account_missing", "bc_pda": bc_pda, "rpc_ms": 0}

    # Parse account data
    try:
        data_list = value.get("data", [])
        if not data_list:
            return {"ok": False, "complete": None, "reason": "parse_error", "bc_pda": bc_pda, "rpc_ms": 0}
        data_b64  = data_list[0]
        raw       = _b64.b64decode(data_b64)
        # Layout:
        #   [0:8]  8-byte Anchor discriminator
        #   [8:16] u64 virtual_token_reserves
        #   [16:24] u64 virtual_sol_reserves
        #   [24:32] u64 real_token_reserves
        #   [32:40] u64 real_sol_reserves
        #   [40:48] u64 token_total_supply
        #   [48]   bool complete
        if len(raw) < 49:
            log.error("GRAD ORACLE short account data len=%d for %s", len(raw), mint[:8])
            return {"ok": False, "complete": None, "reason": "parse_error", "bc_pda": bc_pda, "rpc_ms": 0}
        complete = bool(raw[48])
        reason   = "complete_true" if complete else "complete_false"
        import struct as _struct
        vtr = _struct.unpack_from("<Q", raw, 8)[0]   # virtual_token_reserves
        vsr = _struct.unpack_from("<Q", raw, 16)[0]  # virtual_sol_reserves
        return {
            "ok": True, "complete": complete, "reason": reason, "bc_pda": bc_pda, "rpc_ms": 0,
            "virtual_token_reserves": vtr,
            "virtual_sol_reserves":   vsr,
        }
    except Exception as e:
        log.error("GRAD ORACLE parse error for %s: %s", mint[:8], e)
        return {"ok": False, "complete": None, "reason": "parse_error", "bc_pda": bc_pda, "rpc_ms": 0}


def get_pumpfun_curve_price(mint: str) -> dict:
    """
    Fetch the pump.fun bonding curve account and compute a live price.

    Used by Fix 4 (curve feed) as a price source for PP-silent positions.
    Bypasses the 5s TTL cache intentionally — this IS the fallback feed.

    Returns
    -------
    {
        "ok":        bool,
        "price_usd": float | None,
        "price_sol": float | None,
        "complete":  bool | None,
        "reason":    str,
    }
    """
    try:
        result = _get_pumpfun_curve_complete_uncached(mint)
        if not result.get("ok"):
            return {"ok": False, "price_usd": None, "price_sol": None,
                    "complete": result.get("complete"), "reason": result.get("reason", "oracle_error")}
        if result.get("complete") is None:
            return {"ok": True, "price_usd": None, "price_sol": None,
                    "complete": None, "reason": "account_missing"}
        vtr = result.get("virtual_token_reserves", 0)
        vsr = result.get("virtual_sol_reserves", 0)
        if vtr == 0:
            return {"ok": False, "price_usd": None, "price_sol": None,
                    "complete": result.get("complete"), "reason": "zero_token_reserves"}
        price_sol = (vsr / 1e9) / (vtr / 1e6)   # SOL per token (6-decimal pump.fun tokens)
        price_usd = price_sol * _sol_price_usd()
        return {
            "ok":        True,
            "price_usd": price_usd,
            "price_sol": price_sol,
            "complete":  result.get("complete"),
            "reason":    result.get("reason", "ok"),
        }
    except Exception as e:
        log.debug("get_pumpfun_curve_price error for %s: %s", mint[:8], e)
        return {"ok": False, "price_usd": None, "price_sol": None, "complete": None, "reason": str(e)}


def get_pumpfun_curve_snapshot(mint: str) -> dict:
    """
    Query the pump.fun bonding curve: completion status + live price in one call.

    Shares the same 5-second _GRAD_ORACLE_CACHE as get_pumpfun_curve_complete(),
    so portfolio.py preflight and the executor buy gate never duplicate the RPC —
    whichever fires first populates the cache; the other gets a cache hit.

    Returns
    -------
    {
        "ok":                     bool,
        "complete":               bool | None,   # False=on curve, True=graduated, None=missing
        "price_usd":              float | None,  # set only when complete=False and reserves ok
        "price_sol":              float | None,
        "virtual_token_reserves": int | None,
        "virtual_sol_reserves":   int | None,
        "reason":                 str,           # complete_false|complete_true|account_missing|rpc_error|parse_error
        "bc_pda":                 str,
        "rpc_ms":                 int,
    }
    """
    result = get_pumpfun_curve_complete(mint)   # uses / populates _GRAD_ORACLE_CACHE
    out = dict(result)
    out.setdefault("price_usd", None)
    out.setdefault("price_sol", None)
    vtr = result.get("virtual_token_reserves") or 0
    vsr = result.get("virtual_sol_reserves") or 0
    if result.get("ok") and result.get("complete") is False and vtr > 0:
        try:
            price_sol = (vsr / 1e9) / (vtr / 1e6)
            out["price_sol"] = price_sol
            out["price_usd"] = price_sol * _sol_price_usd()
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# MemeExecutor — public interface used by portfolio.py
# ---------------------------------------------------------------------------

class MemeExecutor:

    def buy(
        self,
        token_address: str,
        size_usd: float,
        chain: str = "solana",
        signal_price: float = 0.0,
        max_slippage_pct: float = 0.50,
        dex_id: str = "",
    ) -> dict:
        """
        Swap SOL → token_address worth size_usd USD.

        Returns dict:
          {"success": True,  "fill_price": float, "tx_sig": str, "timing": {...}}
          {"success": False, "reason": str, ...}

        fill_price is derived from on-chain token balance delta, not from
        any pre-swap quote estimate.
        """
        _t0 = time.time()
        try:
            # ── Kill switch: check before live buy ───────────────────────────────
            try:
                import memecoin.kill_switch as _kill_switch
                if not _kill_switch.live_buys_enabled():
                    log.warning("BUY blocked — kill switch active (live buys disabled)  token=%s", token_address[:8])
                    return {"success": False, "reason": "kill_switch_active"}
            except Exception:
                pass

            # ── Canary T22-probe-only gate ────────────────────────────────────────
            # When CANARY_T22_PROBE_ONLY=True, block all non-T22 live buys so only
            # Token-2022 tokens are purchased during the probe validation window.
            try:
                from memecoin.config import CANARY_T22_PROBE_ONLY as _t22_only
                if _t22_only:
                    _prog = _pumpfun_mint_token_program(token_address)
                    if _prog != _TOKEN22_PROGRAM_ID:
                        log.info(
                            "BUY blocked — CANARY_T22_PROBE_ONLY: SPL token skipped  token=%s",
                            token_address[:8],
                        )
                        return {"success": False, "reason": "canary_t22_probe_only_spl_blocked"}
            except Exception:
                pass

            # ── T22 policy: enforce before live buy ──────────────────────────────
            # Checks token program and T22 extensions. Forces paper-only or canary
            # based on config policy. Note tags are appended to journal.
            _t22_buy_notes = ""
            try:
                from memecoin.screener import check_token_program as _check_t22
                from memecoin.config import (
                    BLOCK_T22_TRANSFER_HOOK as _blk_hook,
                    BLOCK_T22_UNKNOWN_EXTENSIONS as _blk_unk,
                    ALLOW_T22_LIVE_NORMAL as _allow_t22_live,
                    ALLOW_T22_CANARY as _allow_t22_canary,
                    EXIT_SYSTEM_VALIDATED as _exit_validated,
                    LIVE_DRY_RUN as _ldr,
                )
                _t22_info = _check_t22(token_address)
                if _t22_info.get("is_token2022"):
                    _t22_buy_notes = f"|token_program:TOKEN_2022|t22_extensions:{','.join(_t22_info.get('extensions_list', []))}"
                    if _t22_info.get("has_transfer_hook") and _blk_hook:
                        log.warning("BUY blocked — T22 transfer hook  token=%s", token_address[:8])
                        return {"success": False, "reason": "t22_transfer_hook_blocked",
                                "notes": _t22_buy_notes + "|t22_policy:paper_only_transfer_hook"}
                    if _t22_info.get("has_unknown_extensions") and _blk_unk:
                        log.warning("BUY blocked — T22 unknown extension  token=%s", token_address[:8])
                        return {"success": False, "reason": "t22_unknown_extension_blocked",
                                "notes": _t22_buy_notes + "|t22_policy:paper_only_unknown_extension"}
                    # Normal T22: live only if validated; canary only if canary allowed
                    if not _allow_t22_live or not _exit_validated:
                        if _allow_t22_canary:
                            _t22_buy_notes += "|t22_policy:canary"
                            log.info("T22 normal — canary mode  token=%s", token_address[:8])
                            # Size is already capped by canary mode in portfolio.py;
                            # if called directly, enforce cap here too.
                            if not _ldr and size_usd > 3:
                                log.info("T22 canary size cap: %.2f→3  token=%s", size_usd, token_address[:8])
                                size_usd = 3
                        else:
                            log.warning("BUY blocked — T22 live not allowed  token=%s", token_address[:8])
                            return {"success": False, "reason": "t22_live_not_allowed",
                                    "notes": _t22_buy_notes + "|t22_policy:paper_only"}
                elif _t22_info.get("token_program") and _t22_info["token_program"] not in (
                    "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
                    "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",
                ):
                    log.warning("BUY blocked — unknown token program %s  token=%s",
                                _t22_info.get("token_program", "?")[:8], token_address[:8])
                    return {"success": False, "reason": "unknown_token_program",
                            "notes": f"|token_program:{_t22_info.get('token_program', 'unknown')}|t22_policy:paper_only"}
            except Exception as _t22_check_err:
                log.debug("T22 pre-buy check failed (non-blocking): %s", _t22_check_err)

            _, VersionedTransaction, _ = _load_solders()
            keypair    = _get_keypair()
            wallet     = str(keypair.pubkey())
            sol_price  = _sol_price_usd()
            sol_amount = size_usd / sol_price          # SOL to spend (float)
            lamports   = int(sol_amount * 10 ** SOL_DECIMALS)

            # ── Parallel pre-flight: SOL balance + Jupiter quote + priority fee ──
            # Three independent network calls fired simultaneously.
            # Critical path = max(~150ms, ~350ms, ~150ms) ≈ 350ms
            # vs old sequential order (~150ms + ~350ms = ~500ms).
            # SOL balance is collected first since RuntimeError there is fatal for buy.
            _quote_gate         = SLIPPAGE_GATE_DEX_PCT
            jupiter_quote_price = 0.0
            _quote_fetch_err    = None
            _buy_fee            = PRIORITY_FEE_SOL
            _free_sol_lam       = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as _pool:
                _sol_fut   = None if LIVE_DRY_RUN else _pool.submit(_sol_balance, wallet)
                _quote_fut = _pool.submit(_jup_get_quote, SOL_MINT, token_address, lamports)
                _fee_fut   = _pool.submit(_helius_priority_fee, token_address, "High")

                if _sol_fut is not None:
                    _free_sol_lam = _sol_fut.result(timeout=5)   # RuntimeError propagates → buy blocked
                try:
                    quote            = _quote_fut.result(timeout=10)
                    token_decimals_q = int(quote.get("outputDecimals") or 6)
                    tokens_out_q     = int(quote["outAmount"]) / (10 ** token_decimals_q)
                    jupiter_quote_price = size_usd / tokens_out_q if tokens_out_q > 0 else 0
                except Exception as e:
                    _quote_fetch_err = e
                    log.warning("Jupiter pre-flight quote failed: %s", e)
                try:
                    _buy_fee = _fee_fut.result(timeout=3)
                except Exception:
                    pass

            _t_quoted = time.time()

            # ── Pre-buy free-SOL check (balance now from parallel fetch) ──────────
            _required_lam = lamports + _RENT_RESERVE + _FEE_RESERVE + _PRESIGNED_RESERVE
            if not LIVE_DRY_RUN and _free_sol_lam < _required_lam:
                _free_sol_val = _free_sol_lam / 1e9
                _required_sol = _required_lam / 1e9
                log.warning(
                    "BUY blocked — insufficient_free_sol  free=%.6f SOL  "
                    "needed=%.6f SOL  token=%s",
                    _free_sol_val, _required_sol, token_address[:8],
                )
                try:
                    from app.alerts import _send as _alert_send
                    _alert_send(
                        f"⚠️ LOW SOL BALANCE — {_free_sol_val:.4f} SOL free, "
                        f"need {_required_sol:.4f} SOL for {token_address[:8]}. "
                        f"Refill wallet."
                    )
                except Exception:
                    pass
                return {
                    "success":    False,
                    "reason":     "insufficient_free_sol",
                    "free_sol":   round(_free_sol_val, 6),
                    "needed_sol": round(_required_sol, 6),
                }
            elif not LIVE_DRY_RUN and _free_sol_lam < 60_000_000:
                try:
                    from app.alerts import _send as _alert_send
                    _alert_send(
                        f"⚠️ LOW SOL WARNING — wallet only {_free_sol_lam/1e9:.4f} SOL free. "
                        f"Consider topping up."
                    )
                except Exception:
                    pass

            # Gate 1: no quote → unquotable token, block before spending.
            # Exception: PumpPortal backend routes directly via pump.fun bonding curve
            # and does NOT use Jupiter for the actual swap. Jupiter HTTP 400 for these
            # tokens means they are still on the bonding curve — exactly what PP handles.
            # Blocking on no_quote for PP backend creates a systematic selection bias:
            # it blocks fresh bonding-curve tokens (the fast pumpers / best performers)
            # and only allows through older tokens that Jupiter can already route
            # (later in their cycle, past peak). Skip Gate 1 for PP backend.
            if signal_price > 0 and jupiter_quote_price == 0:
                if EXECUTOR_BACKEND != "pumpportal":
                    log.warning(
                        "BUY blocked — no_quote  token=%s  err=%s",
                        token_address[:8], _quote_fetch_err,
                    )
                    return {
                        "success":             False,
                        "reason":              "no_quote",
                        "jupiter_quote_price": 0,
                        "error":               str(_quote_fetch_err),
                    }
                else:
                    log.info(
                        "Jupiter no_quote (PP backend) — bonding-curve token, proceeding via PumpPortal  "
                        "token=%s  err=%s",
                        token_address[:8], _quote_fetch_err,
                    )

            # Gate 2: drift gate — only fires when PP live price is available.
            #
            # When PP has a price: same-venue comparison (PP vs Jupiter, both real-time).
            # Blocks if Jupiter quote is >SLIPPAGE_GATE_RT_PCT above PP. Legitimate
            # protection — catches tokens that spiked between PP tick and quote fetch.
            #
            # When PP is silent (pp=0): skip drift gate entirely.
            # Comparing Jupiter (real-time) to signal_price (stale DexScreener, possibly
            # 30s+ old) measures indexer lag, not real drift. The most common case is a
            # token that graduated from the pump.fun bonding curve to Raydium before the
            # alert fired — PP has no data, DexScreener has pre-graduation price, Jupiter
            # routes through Raydium at post-graduation price. The "drift" is the bonding
            # curve graduation jump, which is real price movement but not a reason to block.
            # Signal filters (pc5m, vol_5m, bs) already validated this token; Gate 1
            # (no_quote) ensures it's tradeable. That is sufficient.
            # Gate 2 baseline priority:
            # 1. PP live price — same-venue, sub-second fresh (best)
            # 2. Jupiter quote — real current market price (when PP silent)
            # 3. DexScreener signal_price — NEVER used as baseline: 10-30s stale minimum,
            #    makes drift measurement meaningless and blocks valid entries.
            _gate_baseline = jupiter_quote_price   # floor: always live
            _pp_active     = False
            try:
                from memecoin.pumpportal_monitor import monitor as _pp_exec
                _pp_now = _pp_exec.get_prices().get(token_address, 0)
                if _pp_now > 0:
                    _gate_baseline = _pp_now
                    _quote_gate    = SLIPPAGE_GATE_RT_PCT
                    _pp_active     = True
                    log.debug("Gate 2 baseline: PP live $%.10f (same-venue gate %.0f%%)",
                              _pp_now, _quote_gate * 100)
                else:
                    log.debug("Gate 2 baseline: Jupiter quote $%.10f (PP silent, DexScreener skipped)",
                              jupiter_quote_price)
            except Exception:
                pass

            # ── Graduated-entry block ─────────────────────────────────────
            # BUY INVARIANT: ExitRouter / MIGRATION_UNCERTAIN is a sell-side routing state only.
            # It is NOT graduation proof. Do not re-add it as a buy blocker.
            # Graduation proof comes exclusively from: dex_id=pumpswap/raydium/orca (CAT-3),
            # or get_pumpfun_curve_complete() returning complete=True / account_missing.
            #
            # Graduated-cohort exclusion: complete=True means the bonding curve is sealed.
            # account_missing means the curve account was closed (token migrated).
            # Both block buy. The preflight in portfolio.py runs get_pumpfun_curve_snapshot()
            # first; this oracle re-uses the same 5s cache so no duplicate RPC is issued.
            _is_graduated   = False
            _grad_evidence  = []
            _curve          = {}          # populated by oracle in elif branch
            _dex_lower      = (dex_id or "").lower()

            if _dex_lower == "pumpswap":
                _is_graduated  = True
                _grad_evidence = ["dex_id=pumpswap"]
            elif not _pp_active and jupiter_quote_price > 0:
                # PP is silent and Jupiter can quote this token.
                # Use pump.fun bonding curve oracle as the single source of truth.
                # BUY INVARIANT: MIGRATION_UNCERTAIN is not graduation proof.
                # It is a sell-side routing state only. Do not re-add it as a buy blocker.
                # Graduation proof comes exclusively from curve.complete or account_missing.
                _curve = get_pumpfun_curve_complete(token_address)
                if _curve["complete"] is False:
                    # Still on bonding curve — not graduated. Allow buy through normal gates.
                    _grad_evidence = []
                    _is_graduated  = False
                else:
                    # complete=True (graduated), None/account_missing, or oracle error
                    _is_graduated  = True
                    _grad_evidence = [f"grad_oracle:{_curve['reason']}"]

            _curve_reason = _curve.get("reason", "n/a")
            log.info(
                "BUY GATE token=%s live=True gate=graduated_entry action=%s reason=%s details=%s",
                token_address[:8], "block" if _is_graduated else "allow",
                _grad_evidence[0] if _grad_evidence else "none",
                {"oracle": _curve_reason},
            )
            _record_buy_gate("block" if _is_graduated else "allow",
                             _grad_evidence[0] if _grad_evidence else "graduated_entry")

            if _is_graduated:
                log.warning(
                    "BUY blocked — graduated token (dex_id=%s PP-active=%s jup=$%.10f "
                    "evidence=%s)  token=%s",
                    dex_id or "n/a", _pp_active, jupiter_quote_price,
                    _grad_evidence, token_address[:8],
                )
                return {
                    "success":            False,
                    "reason":             "blocked_graduated_entry",
                    "dex_id":             dex_id,
                    "pp_active":          _pp_active,
                    "jupiter_quote_price": jupiter_quote_price,
                    "grad_evidence":      _grad_evidence,
                }

            if _pp_active and signal_price > 0 and jupiter_quote_price > 0:
                # Same-venue gate: PP live vs Jupiter quote (measures real movement only)
                slippage = (jupiter_quote_price / _gate_baseline - 1)
                if slippage > _quote_gate:
                    log.warning(
                        "BUY blocked — quote drift %.1f%% > %.0f%%  "
                        "token=%s  pp=$%.10f  quote=$%.10f",
                        slippage * 100, _quote_gate * 100,
                        token_address[:8], _gate_baseline, jupiter_quote_price,
                    )
                    return {
                        "success":             False,
                        "reason":              "blocked_quote_drift",
                        "slippage_pct":        round(slippage * 100, 1),
                        "jupiter_quote_price": jupiter_quote_price,
                        "gate_baseline":       _gate_baseline,
                        "pp_used":             True,
                    }

            # ── Shadow-live / dry-run mode ────────────────────────────────────
            # LIVE_DRY_RUN = True → full live path traversal (pre-flight + quote)
            # but tx is NOT sent.  Returns synthetic fill at Jupiter quote price.
            # Every gate decision was already logged above with real values.
            if LIVE_DRY_RUN:
                _dry_fill = jupiter_quote_price if jupiter_quote_price > 0 else (signal_price or 0)
                log.warning(
                    "DRY_RUN BUY (not sent) — token=%s  size=$%.2f  "
                    "jup_quote=$%.10f  backend=%s",
                    token_address[:8], size_usd, jupiter_quote_price, EXECUTOR_BACKEND,
                )
                return {
                    "success":             True,
                    "fill_price":          _dry_fill,
                    "tx_sig":              f"DRY_RUN_{int(time.time())}",
                    "dry_run":             True,
                    "jupiter_quote_price": jupiter_quote_price,
                    "timing": {
                        "t_quote":   round(_t_quoted    - _t0, 3),
                        "t_submit":  0,
                        "t_confirm": 0,
                    },
                }

            # ── Snapshot SOL + token balance before swap ──────────────────────
            # sol_bal_before_lam: used post-confirm to compute exact SOL spent
            # (includes priority fee + PumpPortal 0.5% + swap fee — all in one delta).
            sol_bal_before_lam = None
            if not LIVE_DRY_RUN:
                try:
                    sol_bal_before_lam = _sol_balance(wallet)
                except RuntimeError as _e:
                    log.debug("BUY pre-swap SOL balance failed (non-blocking): %s", _e)
            bal_before = _token_balance(wallet, token_address)

            # ── Build and sign transaction ────────────────────────────────────
            # PRIMARY (pumpportal backend): local pump.fun instruction build.
            #   Saves ~500-1000ms PumpPortal HTTP round-trip for SPL-Token mints.
            #   Skipped for Token-2022 mints: pump.fun migrated all new tokens to
            #   Token-2022 in 2025 and uses a new program+instruction format
            #   (FAdo9NCw...) that local build does not yet implement.
            #   Falls back to PumpPortal on any exception (graduated token,
            #   RPC error, bonding-curve not found, Token-2022, etc.).
            # FALLBACK: PumpPortal trade-local (handles all cases via pool="auto").
            # ALTERNATE backend: Jupiter (unchanged).
            #
            # T22 venue harness (Token-2022): buy = inventory acquisition only
            #   (local build skipped → PumpPortal fallback is the real buy).
            #   Sell legs for T22 positions are experimental: the exit router tests
            #   Jupiter sell then pump-amm sell and logs which venue clears. This
            #   buy block acquires real inventory; sell routing is data collection.
            if EXECUTOR_BACKEND == "pumpportal":
                _sig_sent = False
                # Check token program — local build only works for classic SPL Token mints
                _tok_prog = _mint_token_program_cache.get(token_address)
                if _tok_prog is None:
                    _tok_prog = _pumpfun_mint_token_program(token_address)
                _use_local_build = (_tok_prog == _TOKEN_PROGRAM_ID)
                if _use_local_build:
                    try:
                        _t_lb     = time.time()
                        _lb_bytes = _pumpfun_local_build_tx(
                            action="buy",
                            wallet_pubkey=wallet,
                            token_mint=token_address,
                            keypair=keypair,
                            sol_amount=sol_amount,
                            slippage_pct=SLIPPAGE_BUY_PCT,
                            priority_fee_sol=_buy_fee,
                        )
                        log.info("LOCAL BUILD buy  token=%s  build_ms=%.0f",
                                 token_address[:8], (time.time() - _t_lb) * 1000)
                        sig       = _send_transaction(_lb_bytes)
                        _sig_sent = True
                    except Exception as _lb_err:
                        log.warning("LOCAL BUILD buy failed (%s) — PumpPortal fallback  token=%s",
                                    _lb_err, token_address[:8])
                else:
                    log.info("LOCAL BUILD buy SKIPPED (Token-2022) — PumpPortal  token=%s",
                             token_address[:8])
                if not _sig_sent:
                    tx_bytes  = _pumpportal_build_tx(
                        wallet_pubkey=wallet, action="buy", token_mint=token_address,
                        amount=sol_amount, denominated_in_sol=True,
                        slippage_pct=SLIPPAGE_BUY_PCT, priority_fee_sol=_buy_fee,
                    )
                    tx        = VersionedTransaction.from_bytes(tx_bytes)
                    signed_tx = VersionedTransaction(tx.message, [keypair])
                    sig       = _send_transaction(bytes(signed_tx))
            else:
                # Jupiter fallback backend
                if not jupiter_quote_price:
                    quote = _jup_get_quote(SOL_MINT, token_address, lamports)
                tx_bytes  = _jup_build_swap_tx(quote, wallet)
                tx        = VersionedTransaction.from_bytes(tx_bytes)
                signed_tx = VersionedTransaction(tx.message, [keypair])
                sig       = _send_transaction(bytes(signed_tx))
            _t_submitted = time.time()
            log.info("BUY tx sent  sig=%s  token=%s  size=$%.2f  backend=%s",
                     sig[:16], token_address[:8], size_usd, EXECUTOR_BACKEND)

            # ── Confirm — check meta.err ──────────────────────────────────────
            confirmed, err = _confirm_tx(sig)
            _t_confirmed   = time.time()

            if not confirmed:
                if err is not None:
                    # Tx landed but reverted (slippage exceeded, etc.)
                    # Retry once with higher slippage before giving up.
                    log.warning("BUY reverted on-chain  sig=%s  err=%s  token=%s — retrying with %d%% slippage",
                                sig[:16], err, token_address[:8], SLIPPAGE_BUY_PCT_RETRY)
                    try:
                        tx_bytes_r = _pumpportal_build_tx(
                            wallet_pubkey=wallet,
                            action="buy",
                            token_mint=token_address,
                            amount=sol_amount,
                            denominated_in_sol=True,
                            slippage_pct=SLIPPAGE_BUY_PCT_RETRY,
                            priority_fee_sol=_buy_fee,
                        )
                        tx_r        = VersionedTransaction.from_bytes(tx_bytes_r)
                        signed_tx_r = VersionedTransaction(tx_r.message, [keypair])
                        sig_r       = _send_transaction(bytes(signed_tx_r))
                        log.info("BUY retry tx sent  sig=%s  token=%s  slippage=%d%%",
                                 sig_r[:16], token_address[:8], SLIPPAGE_BUY_PCT_RETRY)
                        confirmed_r, err_r = _confirm_tx(sig_r)
                        if confirmed_r:
                            sig = sig_r   # use retry sig for fill price lookup below
                            err = None
                            log.info("BUY retry confirmed  sig=%s  token=%s", sig_r[:16], token_address[:8])
                        else:
                            log.warning("BUY retry also failed  sig=%s  err=%s  token=%s",
                                        sig_r[:16], err_r, token_address[:8])
                            return {"success": False, "reason": "tx_reverted",
                                    "tx_sig": sig_r, "on_chain_err": str(err_r),
                                    "jupiter_quote_price": jupiter_quote_price}
                    except Exception as _retry_err:
                        log.warning("BUY retry exception  token=%s  err=%s", token_address[:8], _retry_err)
                        return {"success": False, "reason": "tx_reverted",
                                "tx_sig": sig, "on_chain_err": str(err),
                                "jupiter_quote_price": jupiter_quote_price}

                if err is not None:
                    return {"success": False, "reason": "tx_reverted",
                            "tx_sig": sig, "on_chain_err": str(err),
                            "jupiter_quote_price": jupiter_quote_price}
                # Timeout — check if tokens arrived anyway (sometimes confirm poll lags)
                bal_after = _token_balance(wallet, token_address)
                if bal_after > bal_before:
                    tokens_received = bal_after - bal_before
                    decimals        = _token_decimals_from_rpc(wallet, token_address)
                    _tx_fill, _, _  = _fill_from_transaction(sig, wallet, token_address, size_usd, retries=1)
                    fill_price      = _tx_fill if _tx_fill else (size_usd / (tokens_received / 10 ** decimals))
                    log.warning(
                        "BUY confirm-poll timed out but tokens found — treating as success  "
                        "sig=%s  delta=%d  fill=$%.10f",
                        sig[:16], tokens_received, fill_price,
                    )
                    return {
                        "success":             True,
                        "fill_price":          fill_price,
                        "tokens_received_raw": tokens_received,
                        "tx_sig":              sig,
                        "confirm_fallback":    True,
                        "jupiter_quote_price": jupiter_quote_price,
                        "timing": {
                            "t_quote":   round(_t_quoted    - _t0, 3),
                            "t_submit":  round(_t_submitted - _t0, 3),
                            "t_confirm": round(_t_confirmed - _t0, 3),
                        },
                    }
                log.warning("BUY tx unconfirmed and zero balance  sig=%s", sig[:16])
                return {"success": False, "reason": "unconfirmed", "tx_sig": sig,
                        "jupiter_quote_price": jupiter_quote_price}

            # ── Fill from on-chain tx data (single getTransaction call) ──────
            # Returns the real fill from postTokenBalances — same token, same tx,
            # correct venue (PumpPortal curve), correct time.
            # Falls back to Jupiter quote only if getTransaction can't parse the delta.
            fill_price, entry_estimated, tokens_received_raw = _fill_from_transaction(
                sig, wallet, token_address, size_usd
            )

            # ── Sol spent (buy-side fee gap) ──────────────────────────────────
            # Compute exact SOL deducted from wallet: includes priority fee +
            # PumpPortal's 0.5% platform fee + any swap fee.
            # fill_price is then (sol_spent * sol_price) / tokens_received,
            # not the intended size_usd — so it's the real cost-basis per token.
            sol_spent_usd = None
            if sol_bal_before_lam is not None and not entry_estimated:
                try:
                    _sol_bal_after_lam = _sol_balance(wallet)
                    # Retry up to 3s so confirmed commitment has time to settle
                    for _ in range(3):
                        if _sol_bal_after_lam != sol_bal_before_lam:
                            break
                        time.sleep(1)
                        _sol_bal_after_lam = _sol_balance(wallet)
                    _sol_spent_lam = max(0, sol_bal_before_lam - _sol_bal_after_lam)
                    if _sol_spent_lam > 0:
                        sol_spent_usd = round(_sol_spent_lam / 1e9 * sol_price, 4)
                        # Recompute fill_price using actual SOL cost (includes all fees)
                        # Use tokens_received from _fill_from_transaction via fill_price
                        # inversion: tokens = size_usd / fill_price, then scale.
                        if fill_price and fill_price > 0:
                            tokens_received = size_usd / fill_price
                            if tokens_received > 0:
                                fill_price = sol_spent_usd / tokens_received
                        log.info(
                            "BUY sol_spent=%.6f SOL ($%.4f)  fill_adj=$%.10f  token=%s",
                            _sol_spent_lam / 1e9, sol_spent_usd, fill_price, token_address[:8],
                        )
                except Exception as _sfee_e:
                    log.debug("BUY sol_spent calc failed (non-blocking): %s", _sfee_e)

            if fill_price is None:
                fill_price = jupiter_quote_price
                log.warning(
                    "BUY confirmed but tx parse failed — "
                    "opening at Jupiter quote $%.10f (entry_estimated)  sig=%s",
                    fill_price, sig[:16],
                )

            # ── Entry slippage: fill vs signal_price ─────────────────────────
            entry_slippage_pct = None
            if signal_price > 0 and fill_price and not entry_estimated:
                entry_slippage_pct = round((fill_price / signal_price - 1) * 100, 1)
                log.info(
                    "ENTRY SLIPPAGE  sig=$%.10f  fill=$%.10f  slip=%+.1f%%  token=%s",
                    signal_price, fill_price, entry_slippage_pct, token_address[:8],
                )

            return {
                "success":             True,
                "fill_price":          fill_price,
                "tokens_received_raw": tokens_received_raw,   # raw int — for known-balance TP sells
                "entry_estimated":     entry_estimated,
                "entry_slippage_pct":  entry_slippage_pct,
                "sol_spent_usd":       sol_spent_usd,
                "tx_sig":              sig,
                "jupiter_quote_price": jupiter_quote_price,
                "pp_silent":           not _pp_active,
                # True when bonding curve oracle confirmed complete=False at buy time.
                # Used by portfolio.py to tag cohort:bonding_curve even when PP is silent
                # (T22 tokens are always PP-silent but may still be on bonding curve).
                "oracle_bonding_curve": _curve.get("complete") is False,
                "timing": {
                    "t_quote":   round(_t_quoted    - _t0, 3),
                    "t_submit":  round(_t_submitted - _t0, 3),
                    "t_confirm": round(_t_confirmed - _t0, 3),
                },
            }

        except Exception as e:
            log.error("BUY failed  token=%s  err=%s", token_address[:8], e)
            return {"success": False, "error": str(e)}

    def sell(
        self,
        token_address: str,
        size_usd: float,
        entry_price: float,
        chain: str = "solana",
        fraction: float = 1.0,
        escalate: bool = False,
        known_token_count: int = 0,
        urgent: bool = False,
    ) -> dict:
        """
        Swap all held token_address → SOL.

        urgent=False (TP, time-stop): SELL_LADDER      — 35→60→98% slippage, fee escalates.
        urgent=True  (stops, rugs):   URGENT_SELL_LADDER — 98% slippage from step 1;
                                       fee escalates (High→VeryHigh→UnsafeMax) only if
                                       the previous attempt came back unconfirmed.

        Returns: {success, fill_price, tx_sig, all_sigs, ladder_step}
                 all_sigs contains every submitted sig for journaling.

        fill_price derived from SOL balance delta (not quote outAmount).
        """
        try:
            _, VersionedTransaction, _ = _load_solders()
            keypair = _get_keypair()
            wallet  = str(keypair.pubkey())

            # ── Shadow-live / dry-run mode ────────────────────────────────────
            if LIVE_DRY_RUN:
                log.warning(
                    "DRY_RUN SELL (not sent) — token=%s  entry_price=$%.10f",
                    token_address[:8], entry_price,
                )
                return {
                    "success":     True,
                    "fill_price":  entry_price,
                    "tx_sig":      f"DRY_RUN_{int(time.time())}",
                    "dry_run":     True,
                    "all_sigs":    [],
                    "ladder_step": 0,
                }

            # _token_balance / _sol_balance can raise RuntimeError on RPC rate-limit.
            # For PumpPortal sells we use amount="100%" so balance is only needed for
            # fill price calculation — not for the tx itself.  Never block a sell on RPC.
            #
            # known_token_count: caller passes the exact raw count received at buy time
            # (read from the buy tx's postTokenBalances delta). For partial TP sells this
            # bypasses the RPC query entirely — zero latency, no settle-lag, no zero_balance.
            # Only used for partial sells (fraction < 1.0); full exits use "100%" anyway.
            #
            # When escalate=True (graduated token fast path), both calls are needed
            # before the Jupiter quote — fire them in parallel to save ~300ms.
            balance        = None
            sol_bal_before = None
            if known_token_count > 0:
                # Use caller-supplied count — avoids RPC settle lag.
                # Works for both partial TP sells (fraction < 1.0) and full exits
                # (fraction == 1.0, e.g. close_position passing pos.tokens_held).
                # PumpPortal full exits still use amount="100%" in the build call;
                # balance is used here for fill-price calculation and local build.
                balance = known_token_count
                log.info("SELL using known_token_count=%d (no RPC query)  token=%s",
                         known_token_count, token_address[:8])
            elif escalate:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _pool:
                    _bal_f = _pool.submit(_token_balance, wallet, token_address)
                    _sol_f = _pool.submit(_sol_balance, wallet)
                    try:
                        balance = _bal_f.result(timeout=5)
                    except RuntimeError as _rpc_err:
                        log.warning("SELL _token_balance RPC error (escalate): %s", _rpc_err)
                    try:
                        sol_bal_before = _sol_f.result(timeout=5)
                    except RuntimeError as _rpc_err:
                        log.warning("SELL _sol_balance RPC error (escalate): %s", _rpc_err)
            else:
                try:
                    balance = _token_balance(wallet, token_address)
                except RuntimeError as _rpc_err:
                    if EXECUTOR_BACKEND == "pumpportal":
                        log.warning("SELL _token_balance RPC error — proceeding with 100%% sell: %s", _rpc_err)
                    else:
                        raise   # Jupiter path needs exact balance for quote lamports

            if balance is not None and balance == 0:
                # RPC nodes can lag 1-3s after a recent buy confirmation — retry before giving up.
                # UK bug: abort_tripwire fired 1s after BUY confirmed; RPC returned stale 0.
                for _bal_retry in range(3):
                    time.sleep(1.0)
                    try:
                        balance = _token_balance(wallet, token_address)
                    except RuntimeError:
                        pass
                    if balance and balance > 0:
                        log.info("SELL balance retry %d — found %d tokens  token=%s",
                                 _bal_retry + 1, balance, token_address[:8])
                        break
                if balance == 0:
                    if EXECUTOR_BACKEND == "pumpportal":
                        # PumpPortal amount="100%" doesn't need the balance number.
                        # Proceed anyway — better to attempt a sell than leave tokens stuck.
                        log.warning("SELL zero balance after retries — proceeding with 100%% PumpPortal sell  token=%s",
                                    token_address[:8])
                    else:
                        log.warning("SELL skipped — zero balance  token=%s", token_address[:8])
                        return {"success": False, "reason": "zero_balance"}

            # Determine sell amount: "100%" for full exits, token count for partial TP sells.
            # fraction < 1.0 = TP partial sell — must pass exact token count to PumpPortal.
            # _jup_sell_tokens: used by Jupiter fallback so it also sells the correct fraction
            # (not always 100% of balance). Full exits use full balance; partial use the slice.
            if fraction < 1.0:
                if balance is None or balance == 0:
                    log.warning(
                        "SELL partial %.0f%% — cannot get balance, aborting  token=%s",
                        fraction * 100, token_address[:8],
                    )
                    return {"success": False, "reason": "zero_balance_partial"}
                tokens_to_sell = int(balance * fraction)
                if tokens_to_sell <= 0:
                    log.warning(
                        "SELL partial tokens_to_sell=0  balance=%d  fraction=%.2f  token=%s",
                        balance, fraction, token_address[:8],
                    )
                    return {"success": False, "reason": "zero_tokens_partial"}
                _sell_amount      = str(tokens_to_sell)
                _jup_sell_tokens  = tokens_to_sell  # Jupiter fallback respects the fraction
                _fill_token_count = tokens_to_sell  # fill price denominator (partial, not full)
                log.info(
                    "SELL partial %.0f%%  tokens=%d/%d  token=%s",
                    fraction * 100, tokens_to_sell, balance, token_address[:8],
                )
            else:
                _sell_amount      = "100%"
                _jup_sell_tokens  = balance  # may be None — Jupiter fallback checks before use
                _fill_token_count = balance   # full exit — use full balance for fill price

            # Local build needs exact token count (can't use "100%").
            # fraction < 1.0: use tokens_to_sell (computed above).
            # fraction == 1.0: use balance (may be None if RPC failed → local build skipped).
            _tokens_to_sell_local = tokens_to_sell if fraction < 1.0 else (
                balance if (balance and balance > 0) else 0
            )

            # sol_bal_before: needed for fill price (sol_received = after − before).
            # escalate path: already fetched in parallel above.
            # normal path: fetch now (serial, before ladder starts).
            if not escalate:
                try:
                    sol_bal_before = _sol_balance(wallet)
                except RuntimeError as _rpc_err:
                    log.warning("SELL _sol_balance RPC error — fill price will use entry_price fallback: %s", _rpc_err)

            all_sigs: list[str] = []

            # ── Graduated token fast-path (escalate=True) ────────────────────────
            # Token has migrated to PumpSwap (bonding curve exhausted).
            # pool="auto" returns Custom:6005 — it tries the bonding curve first.
            # PRIMARY:  PumpPortal pool="pump-amm" (PumpSwap direct, single RPC call).
            # FALLBACK: Jupiter (in case pump-amm also reverts, e.g. T22 account ordering).
            # If both fail: alert "graduated unsellable" and return — do NOT loop-retry
            # burning fees; caller (portfolio) will keep position open for manual exit.
            if escalate:
                # Non-urgent graduated sells start at High/0.0005 (same as urgent first
                # attempt). Escalation to VeryHigh/UnsafeMax only on non-landing.
                # Prior UnsafeMax/0.005 floor was applied on the first attempt even when
                # the pool had ample liquidity — over-paying on routine graduated exits.
                _grad_fee_level = "High"
                _grad_fee_floor = 0.0005
                _grad_fee = max(_helius_priority_fee(token_address, _grad_fee_level), _grad_fee_floor)

                # ── T22 check: pump-amm cannot handle Token-2022 ATAs ────────────
                # pump-amm uses SPL ATA derivation → always reverts with Custom:6001
                # for T22 tokens. Skip directly to Jupiter for T22 graduated tokens.
                _skip_pamm_t22 = (
                    _mint_token_program_cache.get(token_address) == _TOKEN22_PROGRAM_ID
                    or "TokenzQ" in _mint_token_program_cache.get(token_address, "")
                )
                if _skip_pamm_t22:
                    log.warning(
                        "SELL escalate (graduated T22) — pump-amm SKIPPED, going direct to Jupiter  token=%s",
                        token_address[:8],
                    )
                else:
                    log.warning("SELL escalate (graduated) — pump-amm PRIMARY  token=%s", token_address[:8])

                # ── PRIMARY: PumpPortal pool="pump-amm" (SPL tokens only) ─────────
                _pamm_sig  = None
                _pamm_conf = False
                _pamm_err  = None
                try:
                    if _skip_pamm_t22:
                        # pump-amm always reverts Custom:6001 for T22 (wrong ATA program).
                        # Set err so the fallback condition below evaluates naturally.
                        _pamm_err = "pump_amm_skipped_t22"
                    else:
                        _t_pamm = time.time()
                        _pamm_bytes = _pumpportal_build_tx(
                            wallet_pubkey=wallet, action="sell", token_mint=token_address,
                            amount=_sell_amount, denominated_in_sol=False,
                            slippage_pct=98, priority_fee_sol=_grad_fee,
                            pool="pump-amm",
                        )
                        _pamm_tx     = VersionedTransaction.from_bytes(_pamm_bytes)
                        _pamm_signed = VersionedTransaction(_pamm_tx.message, [keypair])
                        _pamm_sig    = _send_transaction(bytes(_pamm_signed))
                        all_sigs.append(_pamm_sig)
                        log.warning("SELL pump-amm sent  sig=%s  token=%s  build_ms=%.0f",
                                    _pamm_sig[:16], token_address[:8], (time.time() - _t_pamm) * 1000)
                        _pamm_conf, _pamm_err = _confirm_tx(_pamm_sig, t_sent=_t_pamm)
                except Exception as _pamm_ex:
                    log.warning("SELL pump-amm build/send failed: %s  token=%s", _pamm_ex, token_address[:8])

                if _pamm_conf:
                    fill_price   = None
                    sol_received = 0.0
                    try:
                        sol_bal_after = _sol_balance(wallet)
                        if sol_bal_before is not None:
                            for _sr in range(5):
                                if sol_bal_after != sol_bal_before:
                                    break
                                time.sleep(1.0)
                                try:
                                    sol_bal_after = _sol_balance(wallet)
                                except RuntimeError:
                                    break
                            sol_recv_lam = max(0, sol_bal_after - sol_bal_before)
                            sol_received = sol_recv_lam / 1e9
                            sol_price    = _sol_price_usd()
                            if _fill_token_count is not None:
                                decimals   = _token_decimals_from_rpc(wallet, token_address)
                                tokens_sold = _fill_token_count / (10 ** decimals)
                                fill_price = (sol_received * sol_price) / tokens_sold if tokens_sold > 0 else None
                    except RuntimeError as _rpc_err:
                        log.warning("SELL pump-amm fill price RPC error: %s", _rpc_err)
                    if not fill_price and _pamm_sig:
                        _lam = _sol_delta_from_tx(_pamm_sig, wallet, action="sell")
                        if _lam > 0 and _fill_token_count:
                            try:
                                _dec  = _token_decimals_from_rpc(wallet, token_address)
                                _tsld = _fill_token_count / (10 ** _dec)
                                if _tsld > 0:
                                    sol_received = _lam / 1e9
                                    fill_price   = (sol_received * _sol_price_usd()) / _tsld
                                    log.info("SELL fill from tx meta (pump-amm)  sig=%s  fill=$%.10f",
                                             _pamm_sig[:16], fill_price)
                            except Exception:
                                pass
                    log.info("SELL pump-amm confirmed  sig=%s  sol_recv=%.6f  fill=$%.10f",
                             _pamm_sig[:16], sol_received, fill_price or 0)
                    return {
                        "success":      True,
                        "fill_price":   fill_price,
                        "sol_received": sol_received,
                        "tx_sig":       _pamm_sig,
                        "all_sigs":     all_sigs,
                        "ladder_step":  1,
                        "pump_amm":     True,
                    }
                if _pamm_sig:
                    log.warning("SELL pump-amm reverted  sig=%s  err=%s  token=%s",
                                _pamm_sig[:16], _pamm_err, token_address[:8])

                # ── FALLBACK: Jupiter ─────────────────────────────────────────────
                if _jup_sell_tokens is not None and _jup_sell_tokens > 0:
                    log.warning("SELL pump-amm failed — trying Jupiter fallback  token=%s", token_address[:8])
                    try:
                        _jup_quote    = _jup_get_quote(token_address, SOL_MINT, _jup_sell_tokens)
                        # ── Price impact guard ────────────────────────────────────────────────
                        try:
                            from memecoin.config import MAX_JUPITER_EXIT_PRICE_IMPACT_PCT as _max_impact
                            from memecoin.config import ALLOW_JUPITER_PANIC_EXIT as _panic_ok
                            _impact_pct = float(_jup_quote.get("priceImpactPct", 0)) * 100
                            if _impact_pct > _max_impact and not _panic_ok:
                                log.error(
                                    "SELL Jupiter blocked — price impact %.1f%% > %.0f%% limit  token=%s",
                                    _impact_pct, _max_impact, token_address[:8],
                                )
                                try:
                                    from app.alerts import _send
                                    _send(f"⚠️ JUPITER BLOCKED: {token_address[:8]} — impact {_impact_pct:.1f}% > {_max_impact}% limit. Manual rescue needed.")
                                except Exception:
                                    pass
                                return {
                                    "success": False,
                                    "reason":  "jupiter_impact_too_high",
                                    "error_class": "jupiter_price_impact_too_high",
                                    "jupiter_price_impact_pct": _impact_pct,
                                }
                            if _impact_pct > 0:
                                log.warning("SELL Jupiter price impact: %.1f%%  token=%s", _impact_pct, token_address[:8])
                        except (ImportError, KeyError, TypeError, ValueError):
                            pass
                        _jup_fee_lvl  = "High"    # start High for both urgent and non-urgent
                        _jup_fee_flr  = 0.0005   # 0.0005 SOL floor; escalate only on non-landing
                        _jup_fee      = max(_helius_priority_fee(token_address, _jup_fee_lvl), _jup_fee_flr)
                        _jup_tx    = _jup_build_swap_tx(
                            _jup_quote, wallet,
                            slippage_bps=9900,
                            priority_fee_lamports=int(_jup_fee * 1e9),
                        )
                        _, VersionedTransaction, _ = _load_solders()
                        _jup_signed = VersionedTransaction(
                            VersionedTransaction.from_bytes(_jup_tx).message, [keypair]
                        )
                        _jup_sig    = _send_transaction(bytes(_jup_signed))
                        _t_jup      = time.time()
                        all_sigs.append(_jup_sig)
                        log.warning("SELL Jupiter fallback sent  sig=%s  token=%s", _jup_sig[:16], token_address[:8])
                        _jup_conf, _jup_ferr = _confirm_tx(_jup_sig, t_sent=_t_jup)
                        if _jup_conf:
                            sol_bal_after = None
                            try:
                                sol_bal_after = _sol_balance(wallet)
                                if sol_bal_before is not None:
                                    for _sr in range(5):
                                        if sol_bal_after != sol_bal_before:
                                            break
                                        time.sleep(1.0)
                                        try:
                                            sol_bal_after = _sol_balance(wallet)
                                        except RuntimeError:
                                            break
                            except RuntimeError:
                                pass
                            sol_received = max(0, sol_bal_after - sol_bal_before) / 1e9 if (sol_bal_after and sol_bal_before) else 0
                            sol_price    = _sol_price_usd()
                            decimals     = _token_decimals_from_rpc(wallet, token_address) if _jup_sell_tokens else 6
                            tokens_sold  = _jup_sell_tokens / (10 ** decimals) if _jup_sell_tokens else 0
                            if sol_received > 0 and tokens_sold > 0:
                                fill_price = (sol_received * sol_price) / tokens_sold
                            else:
                                _jlam = _sol_delta_from_tx(_jup_sig, wallet, action="sell")
                                if _jlam > 0 and tokens_sold > 0:
                                    sol_received = _jlam / 1e9
                                    fill_price   = (sol_received * sol_price) / tokens_sold
                                    log.info("SELL Jupiter fill from tx meta (escalate)  sig=%s  fill=$%.10f",
                                             _jup_sig[:16], fill_price)
                                else:
                                    fill_price = entry_price
                            log.info("SELL Jupiter fallback confirmed  sig=%s  fill=$%.10f", _jup_sig[:16], fill_price)
                            return {
                                "success":       True,
                                "fill_price":    fill_price,
                                "tx_sig":        _jup_sig,
                                "all_sigs":      all_sigs,
                                "ladder_step":   2,
                                "jup_fallback":  True,
                            }
                        log.warning("SELL Jupiter fallback reverted  sig=%s  err=%s", _jup_sig[:16], _jup_ferr)
                    except Exception as _jup_ex:
                        log.warning("SELL Jupiter fallback error: %s  token=%s", _jup_ex, token_address[:8])
                else:
                    log.warning("SELL Jupiter fallback skipped — _jup_sell_tokens=0  token=%s", token_address[:8])

                # Both paths failed — alert and return without retrying (caller keeps position open)
                log.error(
                    "SELL graduated unsellable — pump-amm + Jupiter both failed  token=%s  mint=%s  sigs=%s",
                    token_address[:8], token_address, all_sigs,
                )
                try:
                    from app.alerts import _send
                    _send(
                        f"\U0001f6a8 GRADUATED UNSELLABLE {token_address[:8]} — "
                        f"pump-amm + Jupiter both failed — Jupiter rescue will be attempted. "
                        f"mint={token_address}"
                    )
                except Exception:
                    pass
                return {
                    "success":    False,
                    "reason":     "graduated_unsellable",
                    "all_sigs":   all_sigs,
                    "ladder_step": 0,
                }

            # escalate=False: bonding-curve path below (unchanged)
            # _graduated_detected → first 6005 seen mid-ladder, skip remaining PP steps
            _graduated_detected = False

            _active_ladder = URGENT_SELL_LADDER if urgent else SELL_LADDER
            for step, (slip_pct, fee_floor_sol, fee_level) in enumerate(_active_ladder, 1):
                try:
                    # If first PumpPortal step reverted with 6005, this token has graduated
                    # (bonding curve exhausted → now on Raydium). Skip remaining PumpPortal
                    # steps immediately — they will all fail identically and each burns fees.
                    if _graduated_detected:
                        log.warning(
                            "SELL graduated fast-path: skipping PumpPortal step %d  token=%s",
                            step, token_address[:8],
                        )
                        break

                    # Dynamic fee: Helius estimate at escalating levels; floor at ladder minimum
                    fee_sol = max(_helius_priority_fee(token_address, level=fee_level), fee_floor_sol)
                    if EXECUTOR_BACKEND == "pumpportal":
                        # PRIMARY: local build for classic SPL Token mints.
                        # SKIPPED for Token-2022 (new pump.fun program, not yet implemented).
                        # FALLBACK: PumpPortal trade-local (all cases).
                        _sig_sent_local = False
                        _sell_tok_prog  = _mint_token_program_cache.get(token_address, _TOKEN22_PROGRAM_ID)
                        _use_lb_sell    = (_tokens_to_sell_local > 0 and
                                           _sell_tok_prog == _TOKEN_PROGRAM_ID)
                        if _use_lb_sell:
                            try:
                                _t_lb     = time.time()
                                _lb_bytes = _pumpfun_local_build_tx(
                                    action="sell",
                                    wallet_pubkey=wallet,
                                    token_mint=token_address,
                                    keypair=keypair,
                                    token_amount=_tokens_to_sell_local,
                                    slippage_pct=slip_pct,
                                    priority_fee_sol=fee_sol,
                                )
                                log.info("LOCAL BUILD sell step %d  token=%s  build_ms=%.0f",
                                         step, token_address[:8], (time.time() - _t_lb) * 1000)
                                sig             = _send_transaction(_lb_bytes)
                                _sig_sent_local = True
                            except Exception as _lb_err:
                                log.warning("LOCAL BUILD sell step %d failed (%s) — PumpPortal  token=%s",
                                            step, _lb_err, token_address[:8])
                        elif _tokens_to_sell_local > 0:
                            # T22 bonding-curve sell: use T22-native local path first.
                            # bonding_curve_t22.run_bc_t22_sell() derives user ATA with Token-2022
                            # token program — fixes Custom:6001 from wrong ATA derivation.
                            _bc_t22_result = None
                            try:
                                from memecoin.bonding_curve_t22 import run_bc_t22_sell as _run_bc_t22
                                from memecoin.exit_router import TokenExitState as _TES, _log_route_attempt as _lra
                                # executor.sell() has no `pos` param — use token_address as fallback
                                _fake_pos = type("P", (), {
                                    "token_address": token_address, "token_symbol": token_address[:8],
                                    "id": token_address, "tokens_held": _tokens_to_sell_local,
                                    "notes": "",
                                })()
                                _bc_t22_result = _run_bc_t22(_fake_pos, "auto_sell", rpc_url=SOLANA_RPC)
                                _lra(_bc_t22_result, _fake_pos, _TES.BONDING_CURVE_T22)
                            except Exception as _t22_err:
                                log.warning("BC T22 sell call failed (%s) — PumpPortal fallback  token=%s",
                                            _t22_err, token_address[:8])
                            # If T22 local path succeeded, skip PumpPortal
                            if _bc_t22_result and _bc_t22_result.get("success"):
                                log.info("BC T22 LOCAL SELL success  token=%s", token_address[:8])
                                _sig_sent_local = True
                                sig = _bc_t22_result.get("tx_sig", "")
                        if not _sig_sent_local:
                            tx_bytes  = _pumpportal_build_tx(
                                wallet_pubkey=wallet, action="sell", token_mint=token_address,
                                amount=_sell_amount, denominated_in_sol=False,
                                slippage_pct=slip_pct, priority_fee_sol=fee_sol,
                            )
                            tx        = VersionedTransaction.from_bytes(tx_bytes)
                            signed_tx = VersionedTransaction(tx.message, [keypair])
                            sig       = _send_transaction(bytes(signed_tx))
                    else:
                        # Jupiter fallback — slippage via dynamicSlippage, fee via
                        # prioritizationFeeLamports
                        quote    = _jup_get_quote(token_address, SOL_MINT, balance)
                        tx_bytes = _jup_build_swap_tx(
                            quote, wallet,
                            slippage_bps=slip_pct * 100,
                            priority_fee_lamports=int(fee_sol * 1e9),
                        )
                        tx        = VersionedTransaction.from_bytes(tx_bytes)
                        signed_tx = VersionedTransaction(tx.message, [keypair])
                        sig       = _send_transaction(bytes(signed_tx))
                    _t_sig_sent = time.time()
                    all_sigs.append(sig)

                    log.info(
                        "SELL ladder step %d/%d  sig=%s  slip=%d%%  fee=%.4f  token=%s",
                        step, len(_active_ladder), sig[:16], slip_pct, fee_sol, token_address[:8],
                    )

                    confirmed, err = _confirm_tx(sig, t_sent=_t_sig_sent)

                    if confirmed:
                        # Success — compute fill from SOL balance delta.
                        # RPC nodes lag 1-3s after tx confirmation before reflecting the
                        # updated balance. Retry until the balance changes (up to 5s).
                        fill_price   = None
                        sol_received = 0.0
                        try:
                            sol_bal_after = _sol_balance(wallet)
                            if sol_bal_before is not None:
                                for _sol_retry in range(5):
                                    if sol_bal_after != sol_bal_before:
                                        break
                                    time.sleep(1.0)
                                    try:
                                        sol_bal_after = _sol_balance(wallet)
                                    except RuntimeError:
                                        break
                                if sol_bal_after == sol_bal_before:
                                    log.warning(
                                        "SELL sol_balance unchanged after 5 retries — "
                                        "fill price will be unavailable  token=%s",
                                        token_address[:8],
                                    )
                                sol_recv_lam  = max(0, sol_bal_after - sol_bal_before)
                                sol_received  = sol_recv_lam / 1e9
                                sol_price     = _sol_price_usd()
                                if _fill_token_count is not None:
                                    decimals      = _token_decimals_from_rpc(wallet, token_address)
                                    tokens_sold   = _fill_token_count / (10 ** decimals)
                                    fill_price    = (sol_received * sol_price) / tokens_sold if tokens_sold > 0 else None
                        except RuntimeError as _rpc_err:
                            log.warning("SELL fill price RPC error — using entry_price fallback: %s", _rpc_err)

                        # Tx-meta fallback: getBalance settle-lag (sol_bal unchanged after retries).
                        # Read SOL received directly from the confirmed tx preBalances/postBalances.
                        if not fill_price and sig:
                            _lam = _sol_delta_from_tx(sig, wallet, action="sell")
                            if _lam > 0 and _fill_token_count:
                                try:
                                    _dec  = _token_decimals_from_rpc(wallet, token_address)
                                    _tsld = _fill_token_count / (10 ** _dec)
                                    if _tsld > 0:
                                        sol_received = _lam / 1e9
                                        fill_price   = (sol_received * _sol_price_usd()) / _tsld
                                        log.info("SELL fill from tx meta (ladder)  sig=%s  fill=$%.10f",
                                                 sig[:16], fill_price)
                                except Exception:
                                    pass

                        if not fill_price:
                            # Catches both None (balance query failed) and 0.0 (sol_recv=0).
                            # Return None so portfolio knows fill is unknown — it will use
                            # pos.exit_price (stop trigger price) rather than the stale peak.
                            fill_price = None
                            log.warning("SELL fill_price unavailable (sol_recv=0) — portfolio will use trigger price  token=%s", token_address[:8])

                        log.info(
                            "SELL confirmed step %d  sig=%s  sol_recv=%.6f  fill=$%.10f",
                            step, sig[:16], sol_received, fill_price or 0,
                        )
                        return {
                            "success":     True,
                            "fill_price":  fill_price,
                            "sol_received": sol_received,
                            "tx_sig":      sig,
                            "all_sigs":    all_sigs,
                            "ladder_step": step,
                        }

                    if err is not None:
                        # On-chain revert → escalate immediately.
                        # Custom 6005 = slippage exceeded on pump.fun program.  When this
                        # fires on step 1 it almost always means the token graduated from
                        # the bonding curve — pump.fun rejects it at any slippage.  Flag it
                        # so the loop skips steps 2 and 3 and jumps to Jupiter.
                        _is_6005 = (
                            isinstance(err, dict)
                            and err.get("InstructionError", [None, None])[1] == {"Custom": 6005}
                        )
                        if _is_6005 and step == 1:
                            _graduated_detected = True
                            log.warning(
                                "SELL 6005 on step 1 — token likely graduated, skipping PumpPortal  token=%s",
                                token_address[:8],
                            )
                        log.warning(
                            "SELL reverted step %d/%d  sig=%s  err=%s — escalating",
                            step, len(_active_ladder), sig[:16], err,
                        )
                        continue   # next ladder rung

                    else:
                        # Unconfirmed (timeout) — tx may be in-flight; don't double-sell
                        log.warning(
                            "SELL unconfirmed step %d/%d  sig=%s — stopping ladder",
                            step, len(_active_ladder), sig[:16],
                        )
                        return {
                            "success":     False,
                            "reason":      "unconfirmed",
                            "tx_sig":      sig,
                            "all_sigs":    all_sigs,
                            "ladder_step": step,
                            "unconfirmed": True,
                        }

                except Exception as _step_err:
                    log.warning("SELL ladder step %d build/send error: %s — escalating",
                                step, _step_err)
                    continue

            # ── Post-ladder routing ──────────────────────────────────────────────
            # _graduated_detected=True: 6005 at step 1 → token is on PumpSwap.
            #   Route: pump-amm PRIMARY → Jupiter FALLBACK.
            #   Skips the remaining bonding-curve ladder steps (already broken out).
            # _graduated_detected=False: all 3 PP steps failed (non-6005 reverts).
            #   Route: Jupiter directly (existing behaviour for dead/rugged tokens).
            if _graduated_detected:
                # ── Try local PumpSwap first (correct account structure) ──────────────
                log.warning("SELL 6005-detected → trying local PumpSwap first  token=%s", token_address[:8])
                try:
                    from memecoin.exit_router import run_pumpswap_local_path as _er_local
                    from memecoin.portfolio import _get_open_position_by_token as _get_pos
                    _pos_6005 = _get_pos(token_address)
                    if _pos_6005 is not None:
                        _local_result = _er_local(_pos_6005, "6005_graduated", rpc_url=SOLANA_RPC)
                        if _local_result.get("success"):
                            log.info("SELL local PumpSwap (6005 path) confirmed  sig=%s  fill=$%.10f",
                                     _local_result.get("tx_sig", "")[:16], _local_result.get("fill_price", 0))
                            return {
                                "success":    True,
                                "fill_price": _local_result.get("fill_price", entry_price),
                                "tx_sig":     _local_result.get("tx_sig", ""),
                                "all_sigs":   all_sigs,
                                "ladder_step": len(_active_ladder) + 1,
                                "jup_fallback": False,
                            }
                        log.warning("SELL local PumpSwap (6005 path) failed  error_class=%s — falling to pump-amm",
                                    _local_result.get("error_class", "unknown"))
                except Exception as _local_6005_err:
                    log.warning("SELL local PumpSwap (6005 path) exception: %s — falling to pump-amm", _local_6005_err)
                # ── Fall through to PumpPortal pump-amm ──────────────────────────────
                log.warning(
                    "SELL 6005-detected — routing pump-amm (local PS failed)  token=%s",
                    token_address[:8],
                )
                _grad2_fee_level = "High" if urgent else "UnsafeMax"
                _grad2_fee_floor = 0.0005 if urgent else 0.005
                _grad2_fee   = max(_helius_priority_fee(token_address, _grad2_fee_level), _grad2_fee_floor)
                _pamm2_sig   = None
                _pamm2_conf  = False
                _pamm2_err   = None
                try:
                    _t_p2        = time.time()
                    _pamm2_bytes = _pumpportal_build_tx(
                        wallet_pubkey=wallet, action="sell", token_mint=token_address,
                        amount=_sell_amount, denominated_in_sol=False,
                        slippage_pct=98, priority_fee_sol=_grad2_fee,
                        pool="pump-amm",
                    )
                    _pamm2_tx     = VersionedTransaction.from_bytes(_pamm2_bytes)
                    _pamm2_signed = VersionedTransaction(_pamm2_tx.message, [keypair])
                    _pamm2_sig    = _send_transaction(bytes(_pamm2_signed))
                    all_sigs.append(_pamm2_sig)
                    log.warning("SELL pump-amm (6005 path) sent  sig=%s  token=%s",
                                _pamm2_sig[:16], token_address[:8])
                    _pamm2_conf, _pamm2_err = _confirm_tx(_pamm2_sig, t_sent=_t_p2)
                except Exception as _p2ex:
                    log.warning("SELL pump-amm (6005 path) build/send failed: %s  token=%s",
                                _p2ex, token_address[:8])
                if _pamm2_conf:
                    fill_price = None
                    sol_received = 0.0
                    try:
                        sol_bal_after = _sol_balance(wallet)
                        if sol_bal_before is not None:
                            for _sr in range(5):
                                if sol_bal_after != sol_bal_before:
                                    break
                                time.sleep(1.0)
                                try:
                                    sol_bal_after = _sol_balance(wallet)
                                except RuntimeError:
                                    break
                            sol_recv_lam = max(0, sol_bal_after - sol_bal_before)
                            sol_received = sol_recv_lam / 1e9
                            sol_price    = _sol_price_usd()
                            if _fill_token_count is not None:
                                decimals    = _token_decimals_from_rpc(wallet, token_address)
                                tokens_sold = _fill_token_count / (10 ** decimals)
                                fill_price  = (sol_received * sol_price) / tokens_sold if tokens_sold > 0 else None
                    except RuntimeError as _p2re:
                        log.warning("SELL pump-amm (6005 path) fill RPC error: %s", _p2re)
                    if not fill_price and _pamm2_sig:
                        _lam2 = _sol_delta_from_tx(_pamm2_sig, wallet, action="sell")
                        if _lam2 > 0 and _fill_token_count:
                            try:
                                _dec2  = _token_decimals_from_rpc(wallet, token_address)
                                _tsld2 = _fill_token_count / (10 ** _dec2)
                                if _tsld2 > 0:
                                    sol_received = _lam2 / 1e9
                                    fill_price   = (sol_received * _sol_price_usd()) / _tsld2
                                    log.info("SELL fill from tx meta (6005 path)  sig=%s  fill=$%.10f",
                                             _pamm2_sig[:16], fill_price)
                            except Exception:
                                pass
                    log.info("SELL pump-amm (6005 path) confirmed  sig=%s  sol_recv=%.6f  fill=$%.10f",
                             _pamm2_sig[:16], sol_received, fill_price or 0)
                    return {
                        "success":      True,
                        "fill_price":   fill_price,
                        "sol_received": sol_received,
                        "tx_sig":       _pamm2_sig,
                        "all_sigs":     all_sigs,
                        "ladder_step":  1,
                        "pump_amm":     True,
                    }
                if _pamm2_sig:
                    log.warning("SELL pump-amm (6005 path) reverted  sig=%s  err=%s  token=%s",
                                _pamm2_sig[:16], _pamm2_err, token_address[:8])
                # pump-amm failed → fall through to Jupiter below

            # Jupiter fallback: fires when ladder exhausted (non-6005) OR pump-amm failed.
            log.warning(
                "SELL %s — trying Jupiter fallback  token=%s",
                "pump-amm failed (6005 path)" if _graduated_detected else "ladder EXHAUSTED",
                token_address[:8],
            )
            try:
                if _jup_sell_tokens is not None and _jup_sell_tokens > 0:
                    _jup_quote    = _jup_get_quote(token_address, SOL_MINT, _jup_sell_tokens)
                    # ── Price impact guard ────────────────────────────────────────────────
                    try:
                        from memecoin.config import MAX_JUPITER_EXIT_PRICE_IMPACT_PCT as _max_impact
                        from memecoin.config import ALLOW_JUPITER_PANIC_EXIT as _panic_ok
                        _impact_pct = float(_jup_quote.get("priceImpactPct", 0)) * 100
                        if _impact_pct > _max_impact and not _panic_ok:
                            log.error(
                                "SELL Jupiter blocked — price impact %.1f%% > %.0f%% limit  token=%s",
                                _impact_pct, _max_impact, token_address[:8],
                            )
                            try:
                                from app.alerts import _send
                                _send(f"⚠️ JUPITER BLOCKED: {token_address[:8]} — impact {_impact_pct:.1f}% > {_max_impact}% limit. Manual rescue needed.")
                            except Exception:
                                pass
                            return {
                                "success": False,
                                "reason":  "jupiter_impact_too_high",
                                "error_class": "jupiter_price_impact_too_high",
                                "jupiter_price_impact_pct": _impact_pct,
                            }
                        if _impact_pct > 0:
                            log.warning("SELL Jupiter price impact: %.1f%%  token=%s", _impact_pct, token_address[:8])
                    except (ImportError, KeyError, TypeError, ValueError):
                        pass
                    _jup2_fee_lvl = "High" if urgent else "UnsafeMax"
                    _jup2_fee_flr = 0.0005 if urgent else 0.005
                    _jup_fee      = max(_helius_priority_fee(token_address, _jup2_fee_lvl), _jup2_fee_flr)
                    _jup_tx    = _jup_build_swap_tx(
                        _jup_quote, wallet,
                        slippage_bps=9900,          # 99% — last resort, get anything out
                        priority_fee_lamports=int(_jup_fee * 1e9),
                    )
                    _, VersionedTransaction, _ = _load_solders()
                    _jup_signed = VersionedTransaction(
                        VersionedTransaction.from_bytes(_jup_tx).message, [keypair]
                    )
                    _jup_sig      = _send_transaction(bytes(_jup_signed))
                    _t_jup_sent   = time.time()
                    all_sigs.append(_jup_sig)
                    log.warning("SELL Jupiter fallback sent  sig=%s  token=%s",
                                _jup_sig[:16], token_address[:8])
                    _jup_conf, _jup_err = _confirm_tx(_jup_sig, t_sent=_t_jup_sent)
                    if _jup_conf:
                        sol_bal_after = None
                        try:
                            sol_bal_after = _sol_balance(wallet)
                            if sol_bal_before is not None:
                                for _sol_retry in range(5):
                                    if sol_bal_after != sol_bal_before:
                                        break
                                    time.sleep(1.0)
                                    try:
                                        sol_bal_after = _sol_balance(wallet)
                                    except RuntimeError:
                                        break
                        except RuntimeError:
                            pass
                        sol_received = max(0, sol_bal_after - sol_bal_before) / 1e9 if (sol_bal_after and sol_bal_before) else 0
                        sol_price    = _sol_price_usd()
                        decimals     = _token_decimals_from_rpc(wallet, token_address) if _jup_sell_tokens else 6
                        tokens_sold  = _jup_sell_tokens / (10 ** decimals) if _jup_sell_tokens else 0
                        if sol_received > 0 and tokens_sold > 0:
                            fill_price = (sol_received * sol_price) / tokens_sold
                        else:
                            _jlam2 = _sol_delta_from_tx(_jup_sig, wallet, action="sell")
                            if _jlam2 > 0 and tokens_sold > 0:
                                sol_received = _jlam2 / 1e9
                                fill_price   = (sol_received * sol_price) / tokens_sold
                                log.info("SELL Jupiter fill from tx meta (ladder)  sig=%s  fill=$%.10f",
                                         _jup_sig[:16], fill_price)
                            else:
                                fill_price = entry_price
                        log.info("SELL Jupiter fallback confirmed  sig=%s  fill=$%.10f",
                                 _jup_sig[:16], fill_price)
                        return {
                            "success":       True,
                            "fill_price":    fill_price,
                            "tx_sig":        _jup_sig,
                            "all_sigs":      all_sigs,
                            "ladder_step":   len(_active_ladder) + 1,
                            "jup_fallback":  True,
                        }
                    log.warning("SELL Jupiter fallback unconfirmed  sig=%s  err=%s",
                                _jup_sig[:16], _jup_err)
            except Exception as _jup_err:
                log.warning("SELL Jupiter fallback error: %s", _jup_err)

            log.error(
                "SELL ladder EXHAUSTED (incl. Jupiter fallback)  token=%s  sigs=%s",
                token_address[:8], all_sigs,
            )
            # ── Kill switch: auto-disable live buys on unknown sell failure ───────────
            try:
                from memecoin.config import AUTO_DISABLE_ON_UNKNOWN_SELL_FAILURE as _auto_disable
                if _auto_disable:
                    import memecoin.kill_switch as _ks
                    _ks.disable_live_buys(f"unknown sell failure: token={token_address[:8]}")
            except Exception:
                pass
            return {
                "success":     False,
                "reason":      "all_steps_reverted",
                "all_sigs":    all_sigs,
                "ladder_step": len(_active_ladder),
            }

        except Exception as e:
            log.error("SELL failed  token=%s  err=%s", token_address[:8], e)
            return {"success": False, "error": str(e)}


def test_jupiter_dry_run(token_address: str, size_usd: float = 1.0) -> dict:
    """
    Item 6: Verify the Jupiter fallback executor end-to-end in dry-run.

    Gets a Jupiter quote for token_address, builds the swap transaction,
    and returns the result WITHOUT sending anything.  Confirms that:
      - Jupiter quote API is reachable
      - Jito tip accounts list is accessible (jitoTipLamports = JITO_TIP_LAMPORTS)
      - Swap tx builds without error

    Returns:
      {"success": True,  "quote_price": float, "jito_tip_lamports": int, "tx_size_bytes": int}
      {"success": False, "reason": str, "error": str}

    Usage:
        from memecoin.executor import test_jupiter_dry_run
        print(test_jupiter_dry_run("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", 1.0))
    """
    try:
        sol_price  = _sol_price_usd()
        sol_amount = size_usd / sol_price
        lamports   = int(sol_amount * 1e9)

        quote = _jup_get_quote(SOL_MINT, token_address, lamports)
        token_decimals = int(quote.get("outputDecimals") or 6)
        tokens_out     = int(quote["outAmount"]) / (10 ** token_decimals)
        quote_price    = size_usd / tokens_out if tokens_out > 0 else 0

        # Try building the swap tx (requires a wallet pubkey — use a dummy if no key)
        wallet_pubkey = ""
        try:
            keypair, _, _ = _load_solders()
            kp = _get_keypair()
            wallet_pubkey = str(kp.pubkey())
        except Exception:
            wallet_pubkey = "11111111111111111111111111111111"   # system program (dummy)

        tx_bytes = _jup_build_swap_tx(quote, wallet_pubkey)

        return {
            "success":            True,
            "quote_price":        quote_price,
            "tokens_out":         tokens_out,
            "jito_tip_lamports":  JITO_TIP_LAMPORTS,
            "jito_accounts":      len(_JITO_TIP_ACCOUNTS),
            "tx_size_bytes":      len(tx_bytes),
            "backend":            EXECUTOR_BACKEND,
        }
    except Exception as e:
        return {"success": False, "reason": "jupiter_dry_run_failed", "error": str(e)}
