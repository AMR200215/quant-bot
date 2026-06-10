#!/usr/bin/env python3
"""
Execution verification harness.

Usage:
  python tools/verify_execution.py [MINT_ADDRESS]

If no mint is given, fetches a live pump.fun token from DexScreener (sorted by
5m volume) and uses the first result with >$500 liquidity.

Steps:
  1. BUY $1 via current MemeExecutor
  2. On-chain verify via getTransaction: meta.err == null, token balance delta,
     actual fill price from balance delta
  3. Compare executor's fill_price vs chain-derived fill_price (must be <1% diff)
  4. SELL 100% via current MemeExecutor
  5. Same chain verification, check SOL balance delta
  6. Print PASS/FAIL report with Solscan links

Run this against the CURRENT executor before making any executor changes.
Paste the output as proof of baseline behaviour.
"""

import os
import sys
import time
import json

import requests

# Add project root to path so we can import memecoin.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memecoin.executor import MemeExecutor, _token_balance, _get_keypair, _rpc_post, SOLANA_RPC

BUY_SIZE_USD = 1.0
FILL_MATCH_TOLERANCE = 0.01   # 1% — executor fill vs chain fill must be within this
SOL_MINT   = "So11111111111111111111111111111111111111112"
SOL_DECIMALS = 9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pick_mint() -> str:
    """Pick a currently-active pump.fun token from DexScreener by 5m volume."""
    print("No mint specified — picking from DexScreener pump.fun tokens...")
    try:
        resp = requests.get(
            "https://api.dexscreener.com/latest/dex/tokens/solana",
            params={"dex": "pumpfun"},
            timeout=10,
        )
        pairs = resp.json().get("pairs") or []
        # Sort by 5m volume descending, take first with real liquidity
        pairs = sorted(pairs, key=lambda p: float((p.get("volume") or {}).get("m5") or 0), reverse=True)
        for p in pairs[:20]:
            liq = float((p.get("liquidity") or {}).get("usd") or 0)
            mint = (p.get("baseToken") or {}).get("address", "")
            if liq > 500 and mint:
                symbol = (p.get("baseToken") or {}).get("symbol", "?")
                v5m = float((p.get("volume") or {}).get("m5") or 0)
                print(f"  Selected: {symbol} ({mint[:16]}...)  liq=${liq:.0f}  vol5m=${v5m:.0f}")
                return mint
    except Exception as e:
        pass
    raise SystemExit("Could not find a suitable pump.fun token. Pass MINT_ADDRESS explicitly.")


def _get_transaction(sig: str) -> dict:
    """Fetch full transaction details via getTransaction (jsonParsed)."""
    deadline = time.time() + 60
    while time.time() < deadline:
        resp = _rpc_post({
            "jsonrpc": "2.0", "id": 1,
            "method":  "getTransaction",
            "params":  [sig, {
                "encoding":                     "jsonParsed",
                "maxSupportedTransactionVersion": 0,
                "commitment":                   "confirmed",
            }],
        }, timeout=15)
        result = resp.json().get("result")
        if result:
            return result
        time.sleep(3)
    return {}


def _verify_buy(sig: str, mint: str, wallet: str, size_usd: float,
                executor_fill: float) -> dict:
    """
    Verify a buy tx on-chain.
    Returns {"ok": bool, "err": str|None, "chain_fill": float, "tokens_received": int}
    """
    print(f"\n  Fetching tx {sig[:16]}... from chain...")
    tx = _get_transaction(sig)
    if not tx:
        return {"ok": False, "err": "getTransaction returned empty — tx may not be confirmed yet"}

    meta = tx.get("meta") or {}

    # 1. Did the tx land without error?
    on_chain_err = meta.get("err")
    if on_chain_err is not None:
        return {
            "ok":    False,
            "err":   f"tx reverted on-chain: {on_chain_err}",
            "chain_fill": 0,
            "tokens_received": 0,
        }

    # 2. Token balance delta — find our wallet's token account for this mint
    pre_bal  = {b["accountIndex"]: b for b in (meta.get("preTokenBalances")  or [])}
    post_bal = {b["accountIndex"]: b for b in (meta.get("postTokenBalances") or [])}

    tokens_received = 0
    token_decimals  = 6

    for idx, pb in post_bal.items():
        if pb.get("mint") == mint and pb.get("owner") == wallet:
            token_decimals = int((pb.get("uiTokenAmount") or {}).get("decimals") or 6)
            pre_raw  = int(((pre_bal.get(idx) or {}).get("uiTokenAmount") or {}).get("amount") or 0)
            post_raw = int((pb.get("uiTokenAmount") or {}).get("amount") or 0)
            tokens_received = post_raw - pre_raw
            break

    if tokens_received <= 0:
        return {
            "ok":  False,
            "err": f"token balance delta is {tokens_received} — swap may have silently failed",
            "chain_fill": 0,
            "tokens_received": tokens_received,
        }

    tokens_human = tokens_received / (10 ** token_decimals)
    chain_fill   = size_usd / tokens_human

    # 3. Compare executor fill to chain fill
    if executor_fill and executor_fill > 0:
        pct_diff = abs(chain_fill - executor_fill) / chain_fill
        if pct_diff > FILL_MATCH_TOLERANCE:
            return {
                "ok":              False,
                "err":             (f"fill price mismatch: executor=${executor_fill:.10f} "
                                    f"chain=${chain_fill:.10f} diff={pct_diff*100:.1f}%"),
                "chain_fill":      chain_fill,
                "tokens_received": tokens_received,
            }

    return {
        "ok":              True,
        "err":             None,
        "chain_fill":      chain_fill,
        "tokens_received": tokens_received,
    }


def _verify_sell(sig: str, mint: str, wallet: str, tokens_sold: int,
                 executor_fill: float, sol_price: float) -> dict:
    """
    Verify a sell tx on-chain.
    Returns {"ok": bool, "err": str|None, "chain_fill": float}
    """
    print(f"\n  Fetching sell tx {sig[:16]}... from chain...")
    tx = _get_transaction(sig)
    if not tx:
        return {"ok": False, "err": "getTransaction returned empty"}

    meta = tx.get("meta") or {}

    on_chain_err = meta.get("err")
    if on_chain_err is not None:
        return {"ok": False, "err": f"sell tx reverted: {on_chain_err}", "chain_fill": 0}

    # SOL balance delta for our wallet
    accounts = (tx.get("transaction") or {}).get("message", {}).get("accountKeys") or []
    pre_sol  = meta.get("preBalances",  [])
    post_sol = meta.get("postBalances", [])

    sol_received_lamports = 0
    for i, acc in enumerate(accounts):
        acc_key = acc if isinstance(acc, str) else (acc.get("pubkey") or "")
        if acc_key == wallet and i < len(pre_sol) and i < len(post_sol):
            delta = post_sol[i] - pre_sol[i]
            if delta > 0:
                sol_received_lamports = delta
                break

    # Subtract fee if we couldn't isolate receiver vs fee payer
    fee = meta.get("fee", 0)
    if sol_received_lamports == 0:
        # Fee payer's balance went down by fee — find the net SOL change
        for i, acc in enumerate(accounts):
            acc_key = acc if isinstance(acc, str) else (acc.get("pubkey") or "")
            if acc_key == wallet and i < len(pre_sol) and i < len(post_sol):
                delta = post_sol[i] - pre_sol[i] + fee
                sol_received_lamports = delta
                break

    if sol_received_lamports <= 0:
        return {
            "ok":         False,
            "err":        f"SOL balance delta is {sol_received_lamports} lamports — sell may have failed",
            "chain_fill": 0,
        }

    # Token balance delta (should be negative — we sold)
    pre_bal  = {b["accountIndex"]: b for b in (meta.get("preTokenBalances")  or [])}
    post_bal = {b["accountIndex"]: b for b in (meta.get("postTokenBalances") or [])}
    token_decimals = 6
    tokens_sold_chain = 0
    for idx, pb in pre_bal.items():
        if pb.get("mint") == mint and pb.get("owner") == wallet:
            token_decimals  = int((pb.get("uiTokenAmount") or {}).get("decimals") or 6)
            pre_raw  = int((pb.get("uiTokenAmount") or {}).get("amount") or 0)
            post_raw = int(((post_bal.get(idx) or {}).get("uiTokenAmount") or {}).get("amount") or 0)
            tokens_sold_chain = pre_raw - post_raw
            break

    sol_received  = sol_received_lamports / 1e9
    tokens_human  = (tokens_sold_chain or tokens_sold) / (10 ** token_decimals)
    chain_fill    = (sol_received * sol_price) / tokens_human if tokens_human > 0 else 0

    if executor_fill and executor_fill > 0 and chain_fill > 0:
        pct_diff = abs(chain_fill - executor_fill) / chain_fill
        if pct_diff > FILL_MATCH_TOLERANCE:
            return {
                "ok":         False,
                "err":        (f"sell fill mismatch: executor=${executor_fill:.10f} "
                               f"chain=${chain_fill:.10f} diff={pct_diff*100:.1f}%"),
                "chain_fill": chain_fill,
            }

    return {"ok": True, "err": None, "chain_fill": chain_fill}


def _sol_price() -> float:
    try:
        resp = requests.get(
            "https://lite-api.jup.ag/swap/v1/quote",
            params={
                "inputMint":  SOL_MINT,
                "outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                "amount":     1_000_000_000,
            },
            timeout=5,
        )
        return round(float(resp.json()["outAmount"]) / 1e6, 4)
    except Exception:
        return 170.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mint = sys.argv[1] if len(sys.argv) > 1 else None
    if not mint:
        mint = _pick_mint()

    keypair    = _get_keypair()
    wallet     = str(keypair.pubkey())
    sol_price  = _sol_price()
    ex         = MemeExecutor()

    print(f"\n{'='*60}")
    print(f"  EXECUTION VERIFICATION HARNESS")
    print(f"  Mint:   {mint}")
    print(f"  Wallet: {wallet}")
    print(f"  SOL:    ${sol_price}")
    print(f"  Buy:    ${BUY_SIZE_USD}")
    print(f"{'='*60}")

    results = []

    # ── Step 1: BUY ──────────────────────────────────────────────────────────
    print(f"\n[1] BUY ${BUY_SIZE_USD} of {mint[:16]}...")
    tokens_before = _token_balance(wallet, mint)

    buy_result = ex.buy(
        token_address=mint,
        size_usd=BUY_SIZE_USD,
        chain="solana",
        signal_price=0,         # no slippage gate for this test
        max_slippage_pct=1.0,   # 100% — let it go through no matter what
    )

    print(f"  executor returned: {json.dumps({k: v for k, v in buy_result.items() if k != 'timing'}, default=str)}")
    if buy_result.get("timing"):
        t = buy_result["timing"]
        print(f"  timing: quote={t.get('t_quote'):.2f}s  submit={t.get('t_submit'):.2f}s  confirm={t.get('t_confirm'):.2f}s")

    if not buy_result.get("success"):
        print(f"\n  FAIL: executor reported failure — {buy_result.get('reason') or buy_result.get('error')}")
        results.append(("BUY executor", False, buy_result.get("reason") or buy_result.get("error")))
    else:
        buy_sig          = buy_result.get("tx_sig", "")
        executor_fill    = buy_result.get("fill_price") or 0
        tokens_after     = _token_balance(wallet, mint)
        tokens_received  = tokens_after - tokens_before

        print(f"\n  Solscan: https://solscan.io/tx/{buy_sig}")
        print(f"  Executor fill:  ${executor_fill:.10f}")
        print(f"  Balance before: {tokens_before}")
        print(f"  Balance after:  {tokens_after}")
        print(f"  Tokens received (balance delta): {tokens_received}")

        # Wait for finalization before getTransaction
        print("  Waiting 5s for finalization...")
        time.sleep(5)

        v = _verify_buy(buy_sig, mint, wallet, BUY_SIZE_USD, executor_fill)
        if v["ok"]:
            print(f"\n  ✓ BUY PASS")
            print(f"    meta.err:    null")
            print(f"    chain fill:  ${v['chain_fill']:.10f}")
            print(f"    match:       within tolerance")
            results.append(("BUY on-chain verify", True, None))
        else:
            print(f"\n  ✗ BUY FAIL: {v['err']}")
            results.append(("BUY on-chain verify", False, v["err"]))

    # ── Step 2: SELL ─────────────────────────────────────────────────────────
    print(f"\n[2] SELL all {mint[:16]}...")
    tokens_to_sell = _token_balance(wallet, mint)
    print(f"  Balance to sell: {tokens_to_sell}")

    if tokens_to_sell == 0:
        print("  SKIP: zero token balance (buy may have failed)")
        results.append(("SELL", False, "skipped — zero balance from buy failure"))
    else:
        sell_result = ex.sell(
            token_address=mint,
            size_usd=BUY_SIZE_USD,
            entry_price=buy_result.get("fill_price") or 0,
            chain="solana",
        )
        print(f"  executor returned: {json.dumps(sell_result, default=str)}")

        if not sell_result.get("success"):
            print(f"\n  FAIL: executor reported failure — {sell_result.get('reason') or sell_result.get('error')}")
            results.append(("SELL executor", False, sell_result.get("reason") or sell_result.get("error")))
        else:
            sell_sig      = sell_result.get("tx_sig", "")
            executor_fill = sell_result.get("fill_price") or 0

            print(f"\n  Solscan: https://solscan.io/tx/{sell_sig}")
            print(f"  Executor fill:  ${executor_fill:.10f}")

            print("  Waiting 5s for finalization...")
            time.sleep(5)

            v = _verify_sell(sell_sig, mint, wallet, tokens_to_sell, executor_fill, sol_price)
            if v["ok"]:
                print(f"\n  ✓ SELL PASS")
                print(f"    meta.err:    null")
                print(f"    chain fill:  ${v.get('chain_fill', 0):.10f}")
                results.append(("SELL on-chain verify", True, None))
            else:
                print(f"\n  ✗ SELL FAIL: {v['err']}")
                results.append(("SELL on-chain verify", False, v["err"]))

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  REPORT")
    print(f"{'='*60}")
    all_pass = True
    for check, ok, reason in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        line   = f"  {status}  {check}"
        if reason:
            line += f"  [{reason}]"
        print(line)
        if not ok:
            all_pass = False

    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")
    print(f"{'='*60}\n")

    # Known bugs to check for — annotate failure modes
    if not all_pass:
        print("Known bugs to look for in the output above:")
        print("  BUG-1: meta.err != null but executor returned success")
        print("         → _confirm_tx doesn't check err field (reverted tx = phantom position)")
        print("  BUG-2: chain fill vs executor fill >1% mismatch")
        print("         → executor uses quote outAmount not balance delta")
        print("  BUG-3: tx never confirms / unconfirmed timeout")
        print("         → tipless Jito tx not landing")
        print("  BUG-4: executor returned success=False reason=slippage")
        print("         → maxSlippageBps=500 (5%) too tight, tx reverted before reaching executor")
        print()

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
