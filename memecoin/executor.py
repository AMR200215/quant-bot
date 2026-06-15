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
import logging
import os
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

# Buy: 50% — tokens move 20-50% between screen and confirm; 25% was causing 6063 reverts
#             on tokens that moved 40-50% from DexScreener baseline to on-chain confirm.
# Sell: 35% — stop-loss that fails to execute is the worst outcome in the system.
SLIPPAGE_BUY_PCT        = 50
SLIPPAGE_BUY_PCT_RETRY  = 60   # one-time retry on tx_reverted with higher slippage
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

# pump.fun program — used as anchor account for priority fee estimation
_PUMPFUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
# Estimated compute units per pump.fun buy/sell tx (used to convert microlamports/CU → SOL)
_CU_BUDGET_EST   = 200_000

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

SOLANA_RPC          = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
SOLANA_RPC_FALLBACK = "https://solana-rpc.publicnode.com"

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
    _SLIPPAGE_GATE_RT_PCT  = 0.20
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


def _get_keypair():
    Keypair, _, base58 = _load_solders()
    raw = os.getenv("SOLANA_PRIVATE_KEY", "")
    if not raw:
        raise RuntimeError("SOLANA_PRIVATE_KEY env var not set")
    return Keypair.from_bytes(base58.b58decode(raw))


# ---------------------------------------------------------------------------
# Shared RPC helpers
# ---------------------------------------------------------------------------
def _rpc_post(payload: dict, timeout: int = 10) -> requests.Response:
    """POST to SOLANA_RPC; on 429 or quota error fall back to publicnode."""
    resp = requests.post(SOLANA_RPC, json=payload, timeout=timeout)
    _should_fallback = (
        SOLANA_RPC != SOLANA_RPC_FALLBACK
        and (resp.status_code == 429 or "result" not in resp.json())
    )
    if _should_fallback:
        log.warning("Primary RPC unavailable (%s) — falling back to publicnode", resp.status_code)
        resp = requests.post(SOLANA_RPC_FALLBACK, json=payload, timeout=timeout)
    return resp


def _token_balance(wallet_pubkey: str, token_mint: str, retries: int = 1) -> int:
    """Return raw token balance (smallest unit) for a wallet.
    Raises RuntimeError on RPC error so callers can decide how to handle it
    (e.g. fall back to quote price) rather than silently returning 0.
    """
    last_err = None
    for attempt in range(retries):
        if attempt > 0:
            time.sleep(2)
        try:
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
            if not accounts:
                return 0
            return int(accounts[0]["account"]["data"]["parsed"]["info"]["tokenAmount"]["amount"])
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Token balance unavailable: {last_err}")


def _token_decimals_from_rpc(wallet_pubkey: str, token_mint: str) -> int:
    """Return token decimals from the wallet's token account. Defaults to 6."""
    try:
        resp = _rpc_post({
            "jsonrpc": "2.0", "id": 1,
            "method":  "getTokenAccountsByOwner",
            "params":  [
                wallet_pubkey,
                {"mint": token_mint},
                {"encoding": "jsonParsed"},
            ],
        })
        accounts = resp.json().get("result", {}).get("value", [])
        if accounts:
            return int(
                accounts[0]["account"]["data"]["parsed"]["info"]["tokenAmount"]["decimals"]
            )
    except Exception:
        pass
    return 6  # pump.fun default


def _sol_balance(wallet_pubkey: str) -> int:
    """Return wallet SOL balance in lamports. Raises on RPC error."""
    resp = _rpc_post({
        "jsonrpc": "2.0", "id": 1,
        "method":  "getBalance",
        "params":  [wallet_pubkey],
    })
    data = resp.json()
    if "result" not in data:
        raise RuntimeError(f"SOL balance RPC error: {data.get('error', resp.status_code)}")
    return int(data["result"]["value"])


def _confirm_tx(sig: str, max_wait: int = 40) -> tuple[bool, object]:
    """
    Poll getSignatureStatuses until confirmed/finalized or timeout.

    Returns (success: bool, err: object)
      success=True,  err=None   — tx landed and succeeded
      success=False, err=dict   — tx landed but REVERTED (slippage exceeded, etc.)
      success=False, err=None   — timeout, tx not seen within max_wait seconds

    IMPORTANT: callers must check err even when success=False. A reverted tx
    is on-chain; do not record it as a success.
    """
    deadline = time.time() + max_wait
    backoff  = 2
    while time.time() < deadline:
        try:
            resp = _rpc_post({
                "jsonrpc": "2.0", "id": 1,
                "method":  "getSignatureStatuses",
                "params":  [[sig], {"searchTransactionHistory": True}],
            })
            if resp.status_code == 429:
                time.sleep(backoff)
                backoff = min(backoff * 2, 10)
                continue
            status = (resp.json().get("result", {}).get("value") or [None])[0]
            if status:
                conf = status.get("confirmationStatus")
                err  = status.get("err")
                if conf in ("confirmed", "finalized"):
                    if err is not None:
                        log.warning("Tx REVERTED on-chain  sig=%s  err=%s", sig[:16], err)
                        return False, err
                    return True, None
        except Exception:
            pass
        time.sleep(2)
    return False, None   # timeout


def _fill_from_transaction(sig: str, wallet: str, token_address: str,
                           size_usd: float, retries: int = 3) -> tuple[float | None, bool]:
    """
    Fetch a confirmed tx and compute fill price from on-chain token balance deltas.

    Returns (fill_price, entry_estimated):
      fill_price      — real fill derived from postTokenBalances; None if unavailable
      entry_estimated — True if fill_price is None (caller should use Jupiter quote fallback
                        and tag the position entry_estimated=True)

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
                    fill_price = size_usd / (tokens_received / 10 ** decimals)
                    log.info("fill_from_tx  sig=%s  tokens=%d  decimals=%d  fill=$%.10f",
                             sig[:16], tokens_received, decimals, fill_price)
                    return fill_price, False

            log.warning("fill_from_tx: no matching token delta for %s in tx %s",
                        token_address[:8], sig[:16])
            return None, True

        except Exception as e:
            log.debug("fill_from_tx attempt %d/%d failed: %s", attempt + 1, retries, e)
            time.sleep(2)

    log.warning("fill_from_tx: all %d attempts failed for sig=%s", retries, sig[:16])
    return None, True


def _sol_price_usd() -> float:
    """Fetch current SOL/USD price by quoting 1 SOL → USDC via Jupiter."""
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
        return round(float(resp.json()["outAmount"]) / 1e6, 4)
    except Exception as e:
        log.warning("SOL price fetch failed: %s — using fallback $170", e)
        return 170.0


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
) -> bytes:
    """
    Call PumpPortal trade-local API to get a pre-built serialized transaction.
    Returns raw transaction bytes ready for signing.

    Docs: https://pumpportal.fun/integrate/trading-api
    """
    payload = {
        "publicKey":       wallet_pubkey,
        "action":          action,
        "mint":            token_mint,
        "amount":          amount,
        "denominatedInSol": "true" if denominated_in_sol else "false",
        "slippage":        slippage_pct,
        "priorityFee":     priority_fee_sol,
        "pool":            "auto",   # routes bonding-curve and graduated tokens; handles migration
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

def _jup_get_quote(input_mint: str, output_mint: str, amount: int) -> dict:
    resp = requests.get(
        JUPITER_QUOTE_URL,
        params={
            "inputMint":       input_mint,
            "outputMint":      output_mint,
            "amount":          amount,
            "dynamicSlippage": "true",
            "maxSlippageBps":  SLIPPAGE_BUY_PCT * 100,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _jup_build_swap_tx(quote: dict, wallet_pubkey: str,
                       slippage_bps: int = SLIPPAGE_SELL_PCT * 100,
                       priority_fee_lamports: int = None) -> bytes:
    """Build swap transaction via Jupiter swap API. Returns raw tx bytes."""
    if priority_fee_lamports is None:
        priority_fee_lamports = int(PRIORITY_FEE_SOL * 1e9)
    swap_resp = requests.post(
        JUPITER_SWAP_URL,
        json={
            "quoteResponse":             quote,
            "userPublicKey":             wallet_pubkey,
            "wrapAndUnwrapSol":          True,
            "dynamicSlippage":           {"maxBps": slippage_bps},
            "prioritizationFeeLamports": priority_fee_lamports,
        },
        timeout=15,
    )
    swap_resp.raise_for_status()
    return base64.b64decode(swap_resp.json()["swapTransaction"])


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

            # Gate 1: no quote → unquotable token, block before spending
            if signal_price > 0 and jupiter_quote_price == 0:
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
            _gate_baseline = signal_price
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
            except Exception:
                pass

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
            elif not _pp_active and signal_price > 0 and jupiter_quote_price > 0:
                # Cross-venue fallback gate: DexScreener signal vs Jupiter quote.
                # Measures how far the token has moved since the DexScreener snapshot.
                # UK case: 273% above signal → blocked (bought near blown-off top).
                # Graduated tokens with stale DexScreener: also blocked (correct —
                #   PumpPortal can't buy graduated tokens anyway).
                # Tokens within 50%: allowed through (normal indexer lag / small move).
                _xv_slippage = (jupiter_quote_price / signal_price - 1)
                if _xv_slippage > SLIPPAGE_GATE_DEX_PCT:
                    log.warning(
                        "BUY blocked — cross-venue drift %.1f%% > %.0f%% (PP silent, DEX fallback gate)  "
                        "token=%s  dex_signal=$%.10f  jup=$%.10f",
                        _xv_slippage * 100, SLIPPAGE_GATE_DEX_PCT * 100,
                        token_address[:8], signal_price, jupiter_quote_price,
                    )
                    return {
                        "success":             False,
                        "reason":              "blocked_quote_drift",
                        "slippage_pct":        round(_xv_slippage * 100, 1),
                        "jupiter_quote_price": jupiter_quote_price,
                        "gate_baseline":       signal_price,
                        "pp_used":             False,
                    }
                else:
                    log.info(
                        "Gate 2 DEX fallback OK — %.1f%% < %.0f%% (PP silent)  token=%s  "
                        "dex_signal=$%.10f  jup=$%.10f",
                        _xv_slippage * 100, SLIPPAGE_GATE_DEX_PCT * 100,
                        token_address[:8], signal_price, jupiter_quote_price,
                    )

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

            # ── Snapshot balance before swap ──────────────────────────────────
            bal_before = _token_balance(wallet, token_address)

            # ── Build and sign transaction ────────────────────────────────────
            if EXECUTOR_BACKEND == "pumpportal":
                tx_bytes = _pumpportal_build_tx(
                    wallet_pubkey=wallet,
                    action="buy",
                    token_mint=token_address,
                    amount=sol_amount,
                    denominated_in_sol=True,
                    slippage_pct=SLIPPAGE_BUY_PCT,
                    priority_fee_sol=_buy_fee,
                )
            else:
                # Jupiter fallback
                if not jupiter_quote_price:
                    quote = _jup_get_quote(SOL_MINT, token_address, lamports)
                tx_bytes = _jup_build_swap_tx(quote, wallet)

            tx        = VersionedTransaction.from_bytes(tx_bytes)
            signed_tx = VersionedTransaction(tx.message, [keypair])

            sig          = _send_transaction(bytes(signed_tx))
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
                    fill_price      = size_usd / (tokens_received / 10 ** decimals)
                    log.warning(
                        "BUY confirm-poll timed out but tokens found — treating as success  "
                        "sig=%s  delta=%d  fill=$%.10f",
                        sig[:16], tokens_received, fill_price,
                    )
                    return {
                        "success":             True,
                        "fill_price":          fill_price,
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
            # Replaces _token_balance + _token_decimals (2 extra RPC calls).
            # Returns the real fill from postTokenBalances — same token, same tx,
            # correct venue (PumpPortal curve), correct time.
            # Falls back to Jupiter quote only if getTransaction can't parse the
            # delta, and tags those positions entry_estimated=True so they're
            # excluded from live PnL comparisons.
            fill_price, entry_estimated = _fill_from_transaction(
                sig, wallet, token_address, size_usd
            )
            if fill_price is None:
                fill_price = jupiter_quote_price
                log.warning(
                    "BUY confirmed but tx parse failed — "
                    "opening at Jupiter quote $%.10f (entry_estimated)  sig=%s",
                    fill_price, sig[:16],
                )

            # ── Entry slippage: fill vs signal_price (Step 0 measurement) ────────
            # Pins which row of the PnL-vs-slippage grid we're on.
            # entry_estimated positions have unreliable fill_price — excluded.
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
                "entry_estimated":     entry_estimated,
                "entry_slippage_pct":  entry_slippage_pct,
                "tx_sig":              sig,
                "jupiter_quote_price": jupiter_quote_price,
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
    ) -> dict:
        """
        Swap all held token_address → SOL using a 3-step escalating sell ladder.

        Ladder: on revert, immediately escalate slippage + fee (no delay).
          Step 1: 35% slippage, normal fee  (~$0.085)
          Step 2: 60% slippage, 3× fee     (~$0.255)
          Step 3: 98% slippage, max fee    (~$0.85)

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
            balance = None
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

            # Snapshot SOL balance before first attempt (for fill price only — not blocking)
            sol_bal_before = None
            try:
                sol_bal_before = _sol_balance(wallet)
            except RuntimeError as _rpc_err:
                log.warning("SELL _sol_balance RPC error — fill price will use entry_price fallback: %s", _rpc_err)

            all_sigs: list[str] = []

            for step, (slip_pct, fee_floor_sol, fee_level) in enumerate(SELL_LADDER, 1):
                try:
                    # Dynamic fee: Helius estimate at escalating levels; floor at ladder minimum
                    fee_sol = max(_helius_priority_fee(token_address, level=fee_level), fee_floor_sol)
                    if EXECUTOR_BACKEND == "pumpportal":
                        tx_bytes = _pumpportal_build_tx(
                            wallet_pubkey=wallet,
                            action="sell",
                            token_mint=token_address,
                            amount="100%",
                            denominated_in_sol=False,
                            slippage_pct=slip_pct,
                            priority_fee_sol=fee_sol,
                        )
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
                    all_sigs.append(sig)

                    log.info(
                        "SELL ladder step %d/%d  sig=%s  slip=%d%%  fee=%.4f  token=%s",
                        step, len(SELL_LADDER), sig[:16], slip_pct, fee_sol, token_address[:8],
                    )

                    confirmed, err = _confirm_tx(sig)

                    if confirmed:
                        # Success — compute fill from SOL balance delta
                        fill_price   = None
                        sol_received = 0.0
                        try:
                            sol_bal_after = _sol_balance(wallet)
                            if sol_bal_before is not None:
                                sol_recv_lam  = max(0, sol_bal_after - sol_bal_before)
                                sol_received  = sol_recv_lam / 1e9
                                sol_price     = _sol_price_usd()
                                if balance is not None:
                                    decimals      = _token_decimals_from_rpc(wallet, token_address)
                                    tokens_sold   = balance / (10 ** decimals)
                                    fill_price    = (sol_received * sol_price) / tokens_sold if tokens_sold > 0 else None
                        except RuntimeError as _rpc_err:
                            log.warning("SELL fill price RPC error — using entry_price fallback: %s", _rpc_err)

                        if fill_price is None:
                            fill_price = entry_price   # fallback: at least journal closes correctly
                            log.warning("SELL fill_price unavailable — journaling at entry_price $%.10f", fill_price)

                        log.info(
                            "SELL confirmed step %d  sig=%s  sol_recv=%.6f  fill=$%.10f",
                            step, sig[:16], sol_received, fill_price or 0,
                        )
                        return {
                            "success":     True,
                            "fill_price":  fill_price,
                            "tx_sig":      sig,
                            "all_sigs":    all_sigs,
                            "ladder_step": step,
                        }

                    if err is not None:
                        # On-chain revert → escalate immediately
                        log.warning(
                            "SELL reverted step %d/%d  sig=%s  err=%s — escalating",
                            step, len(SELL_LADDER), sig[:16], err,
                        )
                        continue   # next ladder rung

                    else:
                        # Unconfirmed (timeout) — tx may be in-flight; don't double-sell
                        log.warning(
                            "SELL unconfirmed step %d/%d  sig=%s — stopping ladder",
                            step, len(SELL_LADDER), sig[:16],
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

            # All 3 PumpPortal steps reverted — token may have graduated to Raydium.
            # Graduated tokens (bonding curve complete, now on Raydium AMM) revert
            # PumpPortal trade-local. Jupiter routes through Raydium directly.
            # Only fires on the all-reverted path — cannot harm sells that work.
            log.warning(
                "SELL ladder EXHAUSTED — trying Jupiter fallback (graduated token?)  token=%s",
                token_address[:8],
            )
            try:
                if balance is not None and balance > 0:
                    _jup_quote = _jup_get_quote(token_address, SOL_MINT, balance)
                    _jup_fee   = max(_helius_priority_fee(token_address, "UnsafeMax"), 0.005)
                    _jup_tx    = _jup_build_swap_tx(
                        _jup_quote, wallet,
                        slippage_bps=5000,          # 50% — graduated token, take the exit
                        priority_fee_lamports=int(_jup_fee * 1e9),
                    )
                    _, VersionedTransaction, _ = _load_solders()
                    _jup_signed = VersionedTransaction(
                        VersionedTransaction.from_bytes(_jup_tx).message, [keypair]
                    )
                    _jup_sig  = _send_transaction(bytes(_jup_signed))
                    all_sigs.append(_jup_sig)
                    log.warning("SELL Jupiter fallback sent  sig=%s  token=%s",
                                _jup_sig[:16], token_address[:8])
                    _jup_conf, _jup_err = _confirm_tx(_jup_sig)
                    if _jup_conf:
                        sol_bal_after = None
                        try:
                            sol_bal_after = _sol_balance(wallet)
                        except RuntimeError:
                            pass
                        sol_received = max(0, sol_bal_after - sol_bal_before) / 1e9 if (sol_bal_after and sol_bal_before) else 0
                        sol_price    = _sol_price_usd()
                        decimals     = _token_decimals_from_rpc(wallet, token_address) if balance else 6
                        tokens_sold  = balance / (10 ** decimals) if balance else 0
                        fill_price   = (sol_received * sol_price) / tokens_sold if tokens_sold > 0 else entry_price
                        log.info("SELL Jupiter fallback confirmed  sig=%s  fill=$%.10f",
                                 _jup_sig[:16], fill_price)
                        return {
                            "success":       True,
                            "fill_price":    fill_price,
                            "tx_sig":        _jup_sig,
                            "all_sigs":      all_sigs,
                            "ladder_step":   len(SELL_LADDER) + 1,
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
            return {
                "success":     False,
                "reason":      "all_steps_reverted",
                "all_sigs":    all_sigs,
                "ladder_step": len(SELL_LADDER),
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
