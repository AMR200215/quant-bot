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
import logging
import os
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

# Buy: 25% — tokens move 20-50% between screen and confirm; 5% was constant revert source.
# Sell: 35% — stop-loss that fails to execute is the worst outcome in the system.
SLIPPAGE_BUY_PCT  = 25
SLIPPAGE_SELL_PCT = 35

PRIORITY_FEE_SOL  = 0.0005   # ~$0.085 at $170 SOL — enough for Helius landing

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
SOLANA_RPC_FALLBACK = "https://api.mainnet-beta.solana.com"

# Which execution backend to use.  "pumpportal" is the default; set to
# "jupiter" to fall back to the old path (useful for A/B testing).
try:
    from memecoin.config import EXECUTOR_BACKEND as _BACKEND
except ImportError:
    _BACKEND = "pumpportal"
EXECUTOR_BACKEND = _BACKEND


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
    """POST to SOLANA_RPC; on 429 fall back to public mainnet automatically."""
    resp = requests.post(SOLANA_RPC, json=payload, timeout=timeout)
    if resp.status_code == 429 and SOLANA_RPC != SOLANA_RPC_FALLBACK:
        log.warning("Primary RPC 429 — falling back to public mainnet")
        resp = requests.post(SOLANA_RPC_FALLBACK, json=payload, timeout=timeout)
    return resp


def _token_balance(wallet_pubkey: str, token_mint: str) -> int:
    """Return raw token balance (smallest unit) for a wallet."""
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
        if not accounts:
            return 0
        return int(accounts[0]["account"]["data"]["parsed"]["info"]["tokenAmount"]["amount"])
    except Exception as e:
        log.warning("Token balance fetch failed: %s", e)
        return 0


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
    """Return wallet SOL balance in lamports."""
    try:
        resp = _rpc_post({
            "jsonrpc": "2.0", "id": 1,
            "method":  "getBalance",
            "params":  [wallet_pubkey],
        })
        return int(resp.json().get("result", {}).get("value", 0))
    except Exception as e:
        log.warning("SOL balance fetch failed: %s", e)
        return 0


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
            {"encoding": "base64", "skipPreflight": True, "maxRetries": 2},
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
        "priorityFee":     PRIORITY_FEE_SOL,
        "pool":            "auto",   # routes bonding-curve and graduated tokens
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


def _jup_build_swap_tx(quote: dict, wallet_pubkey: str) -> bytes:
    """Build swap transaction via Jupiter swap API. Returns raw tx bytes."""
    swap_resp = requests.post(
        JUPITER_SWAP_URL,
        json={
            "quoteResponse":             quote,
            "userPublicKey":             wallet_pubkey,
            "wrapAndUnwrapSol":          True,
            "dynamicSlippage":           {"maxBps": SLIPPAGE_SELL_PCT * 100},
            "prioritizationFeeLamports": int(PRIORITY_FEE_SOL * 1e9),
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

            # ── Pre-flight slippage check (Jupiter quote) ─────────────────────
            # Even on the PumpPortal path we take a Jupiter quote as a clean
            # price baseline to guard against extreme signal staleness.
            jupiter_quote_price = 0.0
            try:
                quote = _jup_get_quote(SOL_MINT, token_address, lamports)
                _t_quoted = time.time()
                token_decimals_q  = int(quote.get("outputDecimals") or 6)
                tokens_out_q      = int(quote["outAmount"]) / (10 ** token_decimals_q)
                jupiter_quote_price = size_usd / tokens_out_q if tokens_out_q > 0 else 0
            except Exception as e:
                _t_quoted = time.time()
                log.debug("Jupiter pre-flight quote failed: %s — skipping slippage gate", e)

            if signal_price > 0 and jupiter_quote_price > 0:
                slippage = (jupiter_quote_price / signal_price - 1)
                if slippage > max_slippage_pct:
                    log.warning(
                        "BUY skipped — pre-flight slippage %.1f%% > max %.0f%%  "
                        "token=%s  signal=$%.10f  quote=$%.10f",
                        slippage * 100, max_slippage_pct * 100,
                        token_address[:8], signal_price, jupiter_quote_price,
                    )
                    return {
                        "success":             False,
                        "reason":              "slippage",
                        "slippage_pct":        round(slippage * 100, 1),
                        "jupiter_quote_price": jupiter_quote_price,
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
                    # Tx landed but reverted (slippage exceeded, etc.) — do NOT record a position.
                    log.warning("BUY reverted on-chain  sig=%s  err=%s  token=%s",
                                sig[:16], err, token_address[:8])
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

            # ── Fill from on-chain balance delta (not from quote) ─────────────
            bal_after       = _token_balance(wallet, token_address)
            tokens_received = bal_after - bal_before
            if tokens_received <= 0:
                # Confirmed but zero delta — treat as failure (shouldn't happen often)
                log.error(
                    "BUY confirmed but zero token delta  sig=%s  before=%d  after=%d",
                    sig[:16], bal_before, bal_after,
                )
                return {"success": False, "reason": "zero_token_delta",
                        "tx_sig": sig, "jupiter_quote_price": jupiter_quote_price}

            decimals   = _token_decimals_from_rpc(wallet, token_address)
            fill_price = size_usd / (tokens_received / 10 ** decimals)
            log.info("BUY confirmed  sig=%s  tokens=%d  fill=$%.10f",
                     sig[:16], tokens_received, fill_price)

            return {
                "success":             True,
                "fill_price":          fill_price,
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
        Swap all held token_address → SOL.
        Returns dict: {success, fill_price, tx_sig} or {success: False, reason, ...}

        fill_price is derived from SOL balance delta, not from quote outAmount.
        """
        try:
            _, VersionedTransaction, _ = _load_solders()
            keypair = _get_keypair()
            wallet  = str(keypair.pubkey())

            balance = _token_balance(wallet, token_address)
            if balance == 0:
                log.warning("SELL skipped — zero balance  token=%s", token_address[:8])
                return {"success": False, "reason": "zero_balance"}

            # Snapshot SOL balance before swap for fill calculation
            sol_bal_before = _sol_balance(wallet)

            if EXECUTOR_BACKEND == "pumpportal":
                tx_bytes = _pumpportal_build_tx(
                    wallet_pubkey=wallet,
                    action="sell",
                    token_mint=token_address,
                    amount="100%",
                    denominated_in_sol=False,
                    slippage_pct=SLIPPAGE_SELL_PCT,
                )
            else:
                # Jupiter fallback
                quote    = _jup_get_quote(token_address, SOL_MINT, balance)
                tx_bytes = _jup_build_swap_tx(quote, wallet)

            tx        = VersionedTransaction.from_bytes(tx_bytes)
            signed_tx = VersionedTransaction(tx.message, [keypair])

            sig = _send_transaction(bytes(signed_tx))
            log.info("SELL tx sent  sig=%s  token=%s  backend=%s",
                     sig[:16], token_address[:8], EXECUTOR_BACKEND)

            confirmed, err = _confirm_tx(sig)
            if not confirmed:
                if err is not None:
                    log.warning("SELL reverted on-chain  sig=%s  err=%s", sig[:16], err)
                    return {"success": False, "reason": "tx_reverted",
                            "tx_sig": sig, "on_chain_err": str(err)}
                log.warning("SELL tx unconfirmed  sig=%s", sig[:16])
                return {"success": False, "reason": "unconfirmed", "tx_sig": sig}

            # Fill from SOL balance delta
            sol_bal_after     = _sol_balance(wallet)
            sol_received_lam  = sol_bal_after - sol_bal_before   # may be negative due to fees
            # If fee > sol received (shouldn't happen but be safe):
            if sol_received_lam < 0:
                sol_received_lam = 0
            sol_price      = _sol_price_usd()
            decimals       = _token_decimals_from_rpc(wallet, token_address)
            tokens_sold    = balance / (10 ** decimals)
            sol_received   = sol_received_lam / 1e9
            fill_price     = (sol_received * sol_price) / tokens_sold if tokens_sold > 0 else None

            log.info("SELL confirmed  sig=%s  sol_received=%.6f  fill=$%.10f",
                     sig[:16], sol_received, fill_price or 0)
            return {"success": True, "fill_price": fill_price, "tx_sig": sig}

        except Exception as e:
            log.error("SELL failed  token=%s  err=%s", token_address[:8], e)
            return {"success": False, "error": str(e)}
