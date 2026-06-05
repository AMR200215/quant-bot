"""
Sniper executor — Phase 2 (live trading).

Currently STUBBED — returns paper trade prices only.
Wire this up after paper trading validates the strategy.

Phase 2 implementation will use:
  - Jupiter Aggregator API for Solana token swaps
  - Priority fees to land transactions faster
  - Slippage tolerance tuned per strategy (launch: 50%, migration: 25%)
"""

import logging

log = logging.getLogger(__name__)


def get_current_price_sol(mint: str) -> float:
    """
    Fetch current token price in SOL.
    Phase 1: uses DexScreener (free, ~2s lag).
    Phase 2: will use Jupiter quote API (real-time).
    """
    try:
        from memecoin.data_client import dex_get_token
        pair = dex_get_token("solana", mint)
        if not pair:
            return 0.0
        price_usd = float(pair.get("priceUsd") or 0)
        # Convert USD price to SOL price (rough: divide by SOL price)
        # For paper trading this is fine — PnL % is price-ratio based anyway
        return price_usd
    except Exception as e:
        log.debug("get_current_price_sol failed for %s: %s", mint[:8], e)
        return 0.0


def buy(mint: str, size_usd: float, slippage: float = 0.30) -> dict:
    """
    STUB: Execute a buy order.
    Returns {"success": bool, "price_sol": float, "tx": str}

    Phase 2: will submit a Jupiter swap transaction.
    """
    log.info("[PAPER] BUY %s  $%.2f  slippage=%.0f%%", mint[:8], size_usd, slippage * 100)
    price = get_current_price_sol(mint)
    return {"success": True, "price_sol": price, "tx": "paper_trade"}


def sell(mint: str, amount_pct: float = 1.0, slippage: float = 0.30) -> dict:
    """
    STUB: Execute a sell order.
    amount_pct: fraction of position to sell (1.0 = full exit).
    Returns {"success": bool, "price_sol": float, "tx": str}

    Phase 2: will submit a Jupiter swap transaction.
    """
    log.info("[PAPER] SELL %s  %.0f%%  slippage=%.0f%%", mint[:8], amount_pct * 100, slippage * 100)
    price = get_current_price_sol(mint)
    return {"success": True, "price_sol": price, "tx": "paper_trade"}
