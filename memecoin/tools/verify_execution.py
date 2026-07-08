"""
verify_execution.py — sell-matrix receipt harness.

Runs one cell from the sell-matrix. Buys ~$1, sells via intended route,
prints a receipt with: buy sig, sell sig, route, token program, SOL received.

Usage:
    python -m memecoin.tools.verify_execution --cell <CELL> --mint <MINT>

CELL options:
    spl_bc_full      SPL bonding-curve full sell
    spl_bc_partial   SPL bonding-curve 30% partial sell
    t22_bc_full      T22 bonding-curve full sell  (native path, PP fallback DISABLED)
    t22_bc_partial   T22 bonding-curve 30% partial sell (native path, PP fallback DISABLED)

Requirements:
  - MINT must be on bonding curve and match the token program (SPL or T22)
  - SOLANA_PRIVATE_KEY env var set
  - ~0.012 SOL in wallet (buy + fees)
"""

import argparse
import os
import sys
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("verify_execution")

# ── Cells ────────────────────────────────────────────────────────────────────

CELLS = ("spl_bc_full", "spl_bc_partial", "t22_bc_full", "t22_bc_partial")

CELL_IS_T22 = {
    "spl_bc_full":    False,
    "spl_bc_partial": False,
    "t22_bc_full":    True,
    "t22_bc_partial": True,
}

CELL_PARTIAL_PCT = {
    "spl_bc_full":    1.0,
    "spl_bc_partial": 0.30,
    "t22_bc_full":    1.0,
    "t22_bc_partial": 0.30,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_program(mint: str) -> str:
    from memecoin.executor import _pumpfun_mint_token_program, _TOKEN22_PROGRAM_ID
    prog = _pumpfun_mint_token_program(mint)
    return "T22" if prog == _TOKEN22_PROGRAM_ID else "SPL"


def _check_bc(mint: str) -> dict:
    from memecoin.executor import get_pumpfun_curve_complete
    return get_pumpfun_curve_complete(mint)


def _buy(mint: str, size_usd: float = 1.0) -> dict:
    from memecoin.executor import MemeExecutor
    return MemeExecutor().buy(mint, size_usd=size_usd, chain="solana")


def _spl_sell(mint: str, tokens_raw: int, partial: float = 1.0) -> dict:
    """Sell SPL bonding-curve token via PumpPortal."""
    tokens_to_sell = int(tokens_raw * partial)
    from memecoin.executor import MemeExecutor
    # Pass pre-computed token count as known_token_count with fraction=1.0
    # (tokens_to_sell already encodes the partial percentage)
    return MemeExecutor().sell(
        mint,
        size_usd=1.0,        # approximate — not used for BC route selection
        entry_price=0.0,     # not used for exit routing
        chain="solana",
        fraction=1.0,
        known_token_count=tokens_to_sell,
        skip_pumpswap=True,  # bonding-curve only
    )


def _t22_native_sell(mint: str, tokens_raw: int, partial: float = 1.0) -> dict:
    """Sell T22 bonding-curve token via native path. NO PumpPortal fallback."""
    import base58
    from solders.keypair import Keypair
    from memecoin.bonding_curve_t22 import run_bc_t22_sell
    from memecoin.executor import SOLANA_RPC, SOLANA_RPC_FALLBACK

    _pk = os.getenv("SOLANA_PRIVATE_KEY", "")
    if not _pk:
        raise RuntimeError("SOLANA_PRIVATE_KEY not set")

    tokens_to_sell = int(tokens_raw * partial)

    fake_pos = type("P", (), {
        "token_address":    mint,
        "token_symbol":     mint[:8],
        "id":               mint,
        "tokens_held":      tokens_to_sell,   # native sell uses tokens_held as sell amount
        "remaining_fraction": partial,
        "notes":            "",
    })()

    result = run_bc_t22_sell(fake_pos, "verify_harness", rpc_url=SOLANA_RPC)
    # If simulation fails (e.g. primary RPC rate-limited), retry on public fallback
    if not result.get("success"):
        log.info("T22 sell attempt 1 failed (%s), retrying on fallback RPC",
                 result.get("error_class", ""))
        result = run_bc_t22_sell(fake_pos, "verify_harness", rpc_url=SOLANA_RPC_FALLBACK)
    return result


def _print_receipt(cell: str, mint: str, token_prog: str, buy_sig: str,
                   sell_sig: str, route: str, sol_received: float,
                   confirmed: bool, partial_pct: float) -> None:
    print()
    print("=" * 60)
    print("SELL MATRIX RECEIPT")
    print("=" * 60)
    print(f"Cell:        {cell}")
    print(f"Mint:        {mint}")
    print(f"Token prog:  {token_prog}")
    print(f"Type:        {'partial ' + str(int(partial_pct*100)) + '%' if partial_pct < 1 else 'full'}")
    print(f"Route:       {route}")
    print(f"Buy sig:     {buy_sig}")
    print(f"Sell sig:    {sell_sig}")
    print(f"SOL received:{sol_received:.6f}")
    print(f"Confirmed:   {confirmed}")
    print("=" * 60)
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sell matrix receipt harness")
    parser.add_argument("--cell", required=True, choices=CELLS,
                        help="Matrix cell to test")
    parser.add_argument("--mint", required=True,
                        help="Mint address (must be on bonding curve)")
    parser.add_argument("--buy-size", type=float, default=1.0,
                        help="Buy size in USD (default: 1.0)")
    args = parser.parse_args()

    cell      = args.cell
    mint      = args.mint
    buy_size  = args.buy_size
    expect_t22 = CELL_IS_T22[cell]
    partial    = CELL_PARTIAL_PCT[cell]

    log.info("=== VERIFY EXECUTION: %s ===", cell.upper())
    log.info("Mint:     %s", mint)
    log.info("Partial:  %.0f%%", partial * 100)

    # Step 1: detect token program
    log.info("Detecting token program...")
    tok_prog = _detect_program(mint)
    log.info("Token program: %s", tok_prog)
    if expect_t22 and tok_prog != "T22":
        log.error("ABORT: cell %s requires T22 token, got %s", cell, tok_prog)
        sys.exit(1)
    if not expect_t22 and tok_prog != "SPL":
        log.error("ABORT: cell %s requires SPL token, got %s", cell, tok_prog)
        sys.exit(1)

    # Step 2: confirm bonding curve
    log.info("Checking bonding curve status...")
    curve = _check_bc(mint)
    log.info("Curve: complete=%s reason=%s", curve.get("complete"), curve.get("reason"))
    if curve.get("complete") is not False:
        log.error("ABORT: token not on bonding curve (complete=%s). Find a fresh bonding-curve token.", curve.get("complete"))
        sys.exit(1)

    # Step 3: buy
    log.info("Buying $%.2f of %s...", buy_size, mint[:8])
    buy_result = _buy(mint, size_usd=buy_size)
    if not buy_result.get("success"):
        log.error("BUY FAILED: %s", buy_result)
        sys.exit(1)
    buy_sig      = buy_result.get("tx_sig", "")
    tokens_bought = int(buy_result.get("tokens_received_raw") or buy_result.get("token_amount") or buy_result.get("tokens_received") or 0)
    log.info("BUY SIG:      %s", buy_sig)
    log.info("TOKENS BOUGHT:%s", tokens_bought)

    if tokens_bought == 0:
        log.error("ABORT: buy returned 0 tokens — check buy_result: %s", buy_result)
        sys.exit(1)

    log.info("Waiting 4s for TX to land...")
    time.sleep(4)

    # Step 4: sell via correct route
    sell_sig     = ""
    route        = ""
    sol_received = 0.0
    confirmed    = False

    if expect_t22:
        log.info("=== T22 NATIVE SELL (PumpPortal fallback DISABLED) ===")
        try:
            result = _t22_native_sell(mint, tokens_bought, partial=partial)
        except Exception as e:
            log.error("T22 native sell EXCEPTION: %s: %s", type(e).__name__, e)
            import traceback
            traceback.print_exc()
            sys.exit(1)

        if result.get("success"):
            sell_sig     = result.get("tx_sig", "")
            sol_received = float(result.get("sol_received") or 0.0)
            confirmed    = bool(result.get("confirmed", True))
            route        = "bonding_curve_t22 (native)"
            log.info("T22 native sell SUCCESS")
        else:
            log.error("T22 native sell FAILED: %s", result.get("error") or result)
            sys.exit(1)
    else:
        log.info("=== SPL BONDING-CURVE SELL (PumpPortal) ===")
        try:
            result = _spl_sell(mint, tokens_bought, partial=partial)
        except Exception as e:
            log.error("SPL sell EXCEPTION: %s: %s", type(e).__name__, e)
            import traceback
            traceback.print_exc()
            sys.exit(1)

        if result.get("success") or result.get("sol_received"):
            sell_sig     = result.get("tx_sig", "")
            sol_received = float(result.get("sol_received") or 0.0)
            confirmed    = bool(result.get("confirmed", True))
            route        = "pumpportal_bc (SPL bonding-curve)"
            log.info("SPL sell SUCCESS")
        else:
            log.error("SPL sell FAILED: %s", result)
            sys.exit(1)

    # Step 5: print receipt
    _print_receipt(
        cell=cell,
        mint=mint,
        token_prog=tok_prog,
        buy_sig=buy_sig,
        sell_sig=sell_sig,
        route=route,
        sol_received=sol_received,
        confirmed=confirmed,
        partial_pct=partial,
    )

    log.info("=== DONE ===")


if __name__ == "__main__":
    main()
