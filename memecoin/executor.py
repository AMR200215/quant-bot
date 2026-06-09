"""
Real on-chain execution via Jupiter (swap) + Jito (priority inclusion).

Only active when LIVE_TRADING = True in config.
Falls back to paper price on any error so the rest of the system is unaffected.

Environment variables required when LIVE_TRADING = True:
  SOLANA_PRIVATE_KEY  — base58-encoded wallet private key
  SOLANA_RPC_URL      — optional, defaults to mainnet-beta

Flow:
  buy(token_address, size_usd)  → swap SOL→token, returns real fill price
  sell(token_address, size_usd) → swap token→SOL, returns real fill price
  Both return None on failure (caller falls back to paper price).
"""

import base64
import logging
import os
import random
import time

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
JUPITER_QUOTE_URL = "https://lite-api.jup.ag/swap/v1/quote"
JUPITER_SWAP_URL  = "https://lite-api.jup.ag/swap/v1/swap"

SOL_MINT     = "So11111111111111111111111111111111111111112"
SOL_DECIMALS = 9

# 8 official Jito tip accounts — one picked randomly per tx
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
JITO_TIP_LAMPORTS = 1_000_000        # 0.001 SOL (~$0.17)
JITO_RPC_URL      = "https://mainnet.block-engine.jito.wtf/api/v1/transactions"

SOLANA_RPC = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")


# ---------------------------------------------------------------------------
# Lazy imports — solders + base58 only needed when LIVE_TRADING = True
# ---------------------------------------------------------------------------
def _load_solders():
    try:
        from solders.keypair import Keypair
        from solders.pubkey import Pubkey
        from solders.transaction import VersionedTransaction
        from solders.system_program import transfer, TransferParams
        import base58 as _base58
        return Keypair, Pubkey, VersionedTransaction, transfer, TransferParams, _base58
    except ImportError as e:
        raise RuntimeError(
            "solders and base58 are required for live trading: "
            "pip install solders base58"
        ) from e


def _get_keypair():
    """Load wallet keypair from SOLANA_PRIVATE_KEY env var."""
    Keypair, _, _, _, _, base58 = _load_solders()
    raw = os.getenv("SOLANA_PRIVATE_KEY", "")
    if not raw:
        raise RuntimeError("SOLANA_PRIVATE_KEY env var not set")
    return Keypair.from_bytes(base58.b58decode(raw))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

def _sol_price_usd() -> float:
    """Fetch current SOL/USD price by quoting 1 SOL → USDC via Jupiter."""
    try:
        resp = requests.get(
            JUPITER_QUOTE_URL,
            params={
                "inputMint":  SOL_MINT,
                "outputMint": USDC_MINT,
                "amount":     1_000_000_000,  # 1 SOL in lamports
            },
            timeout=5,
        )
        resp.raise_for_status()
        usdc_out = float(resp.json()["outAmount"])  # USDC has 6 decimals
        return round(usdc_out / 1e6, 4)
    except Exception as e:
        log.warning("SOL price fetch failed: %s — using fallback $170", e)
        return 170.0


def _get_quote(input_mint: str, output_mint: str, amount: int) -> dict:
    """Get Jupiter swap quote. amount is in smallest unit of input token."""
    resp = requests.get(
        JUPITER_QUOTE_URL,
        params={
            "inputMint":       input_mint,
            "outputMint":      output_mint,
            "amount":          amount,
            "dynamicSlippage": "true",
            "maxSlippageBps":  500,   # 5% hard cap
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _execute_swap(quote: dict, wallet_pubkey: str) -> str:
    """
    Build swap tx from Jupiter, inject Jito tip instruction, sign, send.
    Returns transaction signature.
    """
    Keypair, Pubkey, VersionedTransaction, transfer_fn, TransferParams, _ = _load_solders()
    keypair = _get_keypair()

    # Get swap transaction from Jupiter
    swap_resp = requests.post(
        JUPITER_SWAP_URL,
        json={
            "quoteResponse":             quote,
            "userPublicKey":             wallet_pubkey,
            "wrapAndUnwrapSol":          True,
            "dynamicSlippage":           {"maxBps": 500},
            "prioritizationFeeLamports": "auto",
        },
        timeout=15,
    )
    swap_resp.raise_for_status()
    tx_bytes = base64.b64decode(swap_resp.json()["swapTransaction"])

    # Deserialize, inject Jito tip, re-sign
    tx = VersionedTransaction.from_bytes(tx_bytes)
    tip_ix = transfer_fn(TransferParams(
        from_pubkey=keypair.pubkey(),
        to_pubkey=Pubkey.from_string(random.choice(_JITO_TIP_ACCOUNTS)),
        lamports=JITO_TIP_LAMPORTS,
    ))
    tx.message.instructions.append(tip_ix)
    signed_tx = VersionedTransaction(tx.message, [keypair])

    # Send via Jito block engine
    jito_resp = requests.post(
        JITO_RPC_URL,
        json={
            "jsonrpc": "2.0",
            "id":      1,
            "method":  "sendTransaction",
            "params":  [base64.b64encode(bytes(signed_tx)).decode(), {"encoding": "base64"}],
        },
        timeout=20,
    )
    jito_resp.raise_for_status()
    result = jito_resp.json()
    if "error" in result:
        raise RuntimeError(f"Jito RPC error: {result['error']}")
    return result["result"]  # tx signature


def _confirm_tx(sig: str, max_wait: int = 30) -> bool:
    """Poll until tx is confirmed or timeout."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            resp = requests.post(
                SOLANA_RPC,
                json={
                    "jsonrpc": "2.0", "id": 1,
                    "method":  "getSignatureStatuses",
                    "params":  [[sig], {"searchTransactionHistory": True}],
                },
                timeout=10,
            )
            status = (resp.json().get("result", {}).get("value") or [None])[0]
            if status and status.get("confirmationStatus") in ("confirmed", "finalized"):
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def _token_balance(wallet_pubkey: str, token_mint: str) -> int:
    """Return raw token balance (smallest unit) for a wallet."""
    try:
        resp = requests.post(
            SOLANA_RPC,
            json={
                "jsonrpc": "2.0", "id": 1,
                "method":  "getTokenAccountsByOwner",
                "params":  [
                    wallet_pubkey,
                    {"mint": token_mint},
                    {"encoding": "jsonParsed"},
                ],
            },
            timeout=10,
        )
        accounts = resp.json().get("result", {}).get("value", [])
        if not accounts:
            return 0
        return int(accounts[0]["account"]["data"]["parsed"]["info"]["tokenAmount"]["amount"])
    except Exception as e:
        log.warning("Token balance fetch failed: %s", e)
        return 0


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def buy(token_address: str, size_usd: float) -> float | None:
    """
    Swap SOL → token_address worth size_usd USD.
    Returns real fill price (USD per token), or None on any failure.
    """
    try:
        sol_price  = _sol_price_usd()
        lamports   = int((size_usd / sol_price) * 10 ** SOL_DECIMALS)

        quote      = _get_quote(SOL_MINT, token_address, lamports)
        keypair    = _get_keypair()
        wallet     = str(keypair.pubkey())

        sig = _execute_swap(quote, wallet)
        log.info("BUY tx sent  sig=%s  token=%s  size=$%.2f", sig[:12], token_address[:8], size_usd)

        if not _confirm_tx(sig):
            log.warning("BUY tx unconfirmed within timeout: %s", sig)
            return None

        # Real fill price = USD spent / tokens received
        token_decimals = int(quote.get("outputDecimals") or 6)
        tokens_out     = int(quote["outAmount"]) / (10 ** token_decimals)
        fill_price     = size_usd / tokens_out if tokens_out > 0 else None

        log.info("BUY confirmed  sig=%s  fill=$%.10f", sig[:12], fill_price or 0)
        return fill_price

    except Exception as e:
        log.error("BUY failed  token=%s  err=%s — paper fallback", token_address[:8], e)
        return None


def sell(token_address: str, size_usd: float, entry_price: float) -> float | None:
    """
    Swap all held token_address → SOL.
    Returns real fill price (USD per token), or None on any failure.
    """
    try:
        keypair = _get_keypair()
        wallet  = str(keypair.pubkey())

        balance = _token_balance(wallet, token_address)
        if balance == 0:
            log.warning("SELL skipped — zero balance  token=%s", token_address[:8])
            return None

        quote = _get_quote(token_address, SOL_MINT, balance)
        sig   = _execute_swap(quote, wallet)
        log.info("SELL tx sent  sig=%s  token=%s", sig[:12], token_address[:8])

        if not _confirm_tx(sig):
            log.warning("SELL tx unconfirmed within timeout: %s", sig)
            return None

        # Real fill price = SOL received × SOL price / tokens sold
        sol_price      = _sol_price_usd()
        sol_out        = int(quote["outAmount"]) / (10 ** SOL_DECIMALS)
        token_decimals = int(quote.get("inputDecimals") or 6)
        tokens_in      = balance / (10 ** token_decimals)
        fill_price     = (sol_out * sol_price) / tokens_in if tokens_in > 0 else None

        log.info("SELL confirmed  sig=%s  fill=$%.10f", sig[:12], fill_price or 0)
        return fill_price

    except Exception as e:
        log.error("SELL failed  token=%s  err=%s — paper fallback", token_address[:8], e)
        return None


# ---------------------------------------------------------------------------
# MemeExecutor — dict-based interface used by portfolio.py
# Pre-flight slippage check: get quote → check price → only submit if within
# max_slippage_pct of signal_price. Saves Jito tip on high-slippage skips.
# ---------------------------------------------------------------------------

class MemeExecutor:

    def buy(self, token_address: str, size_usd: float, chain: str = "solana",
            signal_price: float = 0.0, max_slippage_pct: float = 0.10) -> dict:
        """
        Swap SOL → token. Returns dict: {success, fill_price, tx_sig} or
        {success: False, reason, ...} on abort/failure.
        Pre-flight: if Jupiter quote implies price > signal_price * (1 + max_slippage_pct),
        abort before paying any Jito tip.
        """
        try:
            sol_price = _sol_price_usd()
            lamports  = int((size_usd / sol_price) * 10 ** SOL_DECIMALS)
            quote     = _get_quote(SOL_MINT, token_address, lamports)

            # ── Pre-flight slippage check (free — no tip spent) ──────────────
            if signal_price > 0:
                token_decimals = int(quote.get("outputDecimals") or 6)
                tokens_out     = int(quote["outAmount"]) / (10 ** token_decimals)
                quote_price    = size_usd / tokens_out if tokens_out > 0 else 0
                slippage       = (quote_price / signal_price - 1) if signal_price > 0 else 0
                if slippage > max_slippage_pct:
                    log.warning(
                        "BUY skipped — slippage %.1f%% > max %.0f%%  token=%s  "
                        "signal=$%.10f  quote=$%.10f",
                        slippage * 100, max_slippage_pct * 100,
                        token_address[:8], signal_price, quote_price,
                    )
                    return {"success": False, "reason": "slippage",
                            "slippage_pct": round(slippage * 100, 1),
                            "quote_price": quote_price}

            keypair = _get_keypair()
            wallet  = str(keypair.pubkey())
            sig     = _execute_swap(quote, wallet)
            log.info("BUY tx sent  sig=%s  token=%s  size=$%.2f", sig[:12], token_address[:8], size_usd)

            if not _confirm_tx(sig):
                log.warning("BUY tx unconfirmed  sig=%s", sig[:12])
                return {"success": False, "unconfirmed": True, "tx_sig": sig}

            token_decimals = int(quote.get("outputDecimals") or 6)
            tokens_out     = int(quote["outAmount"]) / (10 ** token_decimals)
            fill_price     = size_usd / tokens_out if tokens_out > 0 else None
            log.info("BUY confirmed  sig=%s  fill=$%.10f", sig[:12], fill_price or 0)
            return {"success": True, "fill_price": fill_price, "tx_sig": sig}

        except Exception as e:
            log.error("BUY failed  token=%s  err=%s", token_address[:8], e)
            return {"success": False, "error": str(e)}

    def sell(self, token_address: str, size_usd: float, entry_price: float,
             chain: str = "solana") -> dict:
        """Swap all held token → SOL. Returns dict: {success, fill_price, tx_sig}."""
        try:
            keypair = _get_keypair()
            wallet  = str(keypair.pubkey())
            balance = _token_balance(wallet, token_address)
            if balance == 0:
                log.warning("SELL skipped — zero balance  token=%s", token_address[:8])
                return {"success": False, "reason": "zero_balance"}

            quote = _get_quote(token_address, SOL_MINT, balance)
            sig   = _execute_swap(quote, wallet)
            log.info("SELL tx sent  sig=%s  token=%s", sig[:12], token_address[:8])

            if not _confirm_tx(sig):
                log.warning("SELL tx unconfirmed  sig=%s", sig[:12])
                return {"success": False, "unconfirmed": True, "tx_sig": sig}

            sol_price      = _sol_price_usd()
            sol_out        = int(quote["outAmount"]) / (10 ** SOL_DECIMALS)
            token_decimals = int(quote.get("inputDecimals") or 6)
            tokens_in      = balance / (10 ** token_decimals)
            fill_price     = (sol_out * sol_price) / tokens_in if tokens_in > 0 else None
            log.info("SELL confirmed  sig=%s  fill=$%.10f", sig[:12], fill_price or 0)
            return {"success": True, "fill_price": fill_price, "tx_sig": sig}

        except Exception as e:
            log.error("SELL failed  token=%s  err=%s", token_address[:8], e)
            return {"success": False, "error": str(e)}
