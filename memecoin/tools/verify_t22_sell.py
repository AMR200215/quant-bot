"""
T22 native sell verification harness.

Usage:
    python -m memecoin.tools.verify_t22_sell <MINT_ADDRESS>

Buys ~$1 of the given bonding-curve T22 mint, then sells through the
NATIVE T22 path (bonding_curve_t22.run_bc_t22_sell) with PumpPortal
fallback DISABLED. Prints exact error if native path fails.

Requirements:
  - MINT must be a pump.fun Token-2022 token still on bonding curve
  - SOLANA_PRIVATE_KEY env var set
  - ~0.01 SOL in wallet
"""

import os, sys, time, logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("verify_t22_sell")

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m memecoin.tools.verify_t22_sell <MINT_ADDRESS>")
        sys.exit(1)

    mint = sys.argv[1]
    log.info("=== T22 NATIVE SELL VERIFICATION ===")
    log.info("Mint: %s", mint)

    # 1. Confirm T22
    from memecoin.executor import _mint_token_program_cache, _TOKEN22_PROGRAM_ID, _detect_token_program
    tok_prog = _detect_token_program(mint)
    log.info("Token program: %s (T22=%s)", tok_prog, tok_prog == _TOKEN22_PROGRAM_ID)
    if tok_prog != _TOKEN22_PROGRAM_ID:
        log.error("ABORT: mint is not Token-2022")
        sys.exit(1)

    # 2. Confirm bonding curve
    from memecoin.executor import get_pumpfun_curve_complete
    curve = get_pumpfun_curve_complete(mint)
    log.info("Curve: complete=%s reason=%s", curve.get("complete"), curve.get("reason"))
    if curve.get("complete") is not False:
        log.error("ABORT: token not on bonding curve (complete=%s)", curve.get("complete"))
        sys.exit(1)

    # 3. Buy ~$1
    from memecoin.executor import Executor
    ex = Executor()
    buy_result = ex.buy(mint, size_usd=1.0, chain="solana")
    if not buy_result.get("success"):
        log.error("BUY FAILED: %s", buy_result)
        sys.exit(1)
    buy_sig = buy_result.get("tx_sig", "")
    tokens_bought = buy_result.get("token_amount", 0)
    log.info("BUY SIG: %s", buy_sig)
    log.info("TOKENS BOUGHT: %s", tokens_bought)

    time.sleep(3)

    # 4. Native T22 sell — PumpPortal fallback DISABLED
    log.info("=== NATIVE T22 SELL (no PumpPortal fallback) ===")
    try:
        from memecoin.bonding_curve_t22 import run_bc_t22_sell
        from memecoin.executor import SOLANA_RPC
        import base58
        from solders.keypair import Keypair

        _pk = os.getenv("SOLANA_PRIVATE_KEY", "")
        keypair = Keypair.from_bytes(base58.b58decode(_pk))
        wallet = str(keypair.pubkey())

        # Build fake pos
        fake_pos = type("P", (), {
            "token_address": mint,
            "token_symbol": mint[:8],
            "id": mint,
            "tokens_held": tokens_bought,
            "notes": "",
        })()

        result = run_bc_t22_sell(fake_pos, "verify_harness", rpc_url=SOLANA_RPC)

        if result.get("success"):
            log.info("=== NATIVE T22 SELL SUCCESS ===")
            log.info("SELL SIG:     %s", result.get("tx_sig"))
            log.info("SOL RECEIVED: %s", result.get("sol_received"))
            log.info("ROUTE:        bonding_curve_t22 (native)")
            log.info("CONFIRMED:    %s", result.get("confirmed"))
            log.info("NO PUMPPORTAL FALLBACK USED")
        else:
            log.error("=== NATIVE T22 SELL FAILED ===")
            log.error("Error: %s", result.get("error") or result)
            log.error("Fix this error, then rerun the harness.")
            sys.exit(1)

    except Exception as e:
        log.error("=== NATIVE T22 SELL EXCEPTION ===")
        log.error("%s: %s", type(e).__name__, e)
        import traceback
        traceback.print_exc()
        log.error("Fix this error, then rerun the harness.")
        sys.exit(1)

if __name__ == "__main__":
    main()
