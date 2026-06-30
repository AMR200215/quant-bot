"""
jupiter_t22_probe.py — Spike: can Jupiter quote, build, and simulate a sell
for a real graduated Token-2022 PumpSwap token?

NO transactions are sent. NO live behavior is enabled.

Usage:
    python -m memecoin.tools.jupiter_t22_probe \\
        --mint <T22_GRADUATED_MINT> \\
        --wallet <BOT_WALLET> \\
        --token-amount-mode percent \\
        --token-amount 100 \\
        [--slippage-bps 3000] \\
        [--rpc <RPC_URL>]
"""

import argparse
import base64
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ── Verdict constants ─────────────────────────────────────────────────────────
PASS_JUPITER_T22_SELL_SIM_OK  = "PASS_JUPITER_T22_SELL_SIM_OK"
FAIL_JUPITER_NO_ROUTE         = "FAIL_JUPITER_NO_ROUTE"
FAIL_JUPITER_429              = "FAIL_JUPITER_429"
FAIL_JUPITER_SIM_FAILED       = "FAIL_JUPITER_SIM_FAILED"
FAIL_NO_TOKEN_BALANCE         = "FAIL_NO_TOKEN_BALANCE"
FAIL_UNKNOWN                  = "FAIL_UNKNOWN"

# ── Jupiter error classes ─────────────────────────────────────────────────────
JUPITER_QUOTE_UNAVAILABLE      = "jupiter_quote_unavailable"
JUPITER_SWAP_BUILD_FAILED      = "jupiter_swap_build_failed"
JUPITER_429                    = "jupiter_429"
JUPITER_ROUTE_NO_LIQUIDITY     = "jupiter_route_no_liquidity"
JUPITER_PRICE_IMPACT_TOO_HIGH  = "jupiter_price_impact_too_high"
JUPITER_SIM_OK                 = "jupiter_sim_ok"
JUPITER_SIM_FAILED             = "jupiter_sim_failed"
TOKEN_BALANCE_MISSING          = "token_balance_missing"
TOKEN2022_UNSUPPORTED          = "token2022_unsupported"
RPC_SIM_FAILED                 = "rpc_sim_failed"

TOKEN_PROGRAM_SPL = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_PROGRAM_T22 = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
WSOL_MINT         = "So11111111111111111111111111111111111111112"

JUPITER_QUOTE_URL = "https://lite-api.jup.ag/swap/v1/quote"
JUPITER_SWAP_URL  = "https://lite-api.jup.ag/swap/v1/swap"

LOG_PATH = Path(__file__).parent.parent.parent / "logs" / "jupiter_t22_probe.csv"

_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _head(title: str):
    print(f"\n{_BOLD}{'─'*60}{_RESET}")
    print(f"{_BOLD}{title}{_RESET}")
    print(f"{'─'*60}")


def _ok(label: str, detail: str = ""):
    print(f"  {_GREEN}✓{_RESET}  {label}")
    if detail:
        for line in detail.splitlines():
            print(f"      {line}")


def _warn(label: str, detail: str = ""):
    print(f"  {_YELLOW}⚠{_RESET}  {label}")
    if detail:
        for line in detail.splitlines():
            print(f"      {line}")


def _err(label: str, detail: str = ""):
    print(f"  {_RED}✗{_RESET}  {label}")
    if detail:
        for line in detail.splitlines():
            print(f"      {line}")


# ── RPC helper ────────────────────────────────────────────────────────────────

def _rpc(rpc_url: str, payload: dict, timeout: int = 15) -> dict:
    resp = requests.post(rpc_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"RPC error: {data['error']}")
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Confirm token state
# ═══════════════════════════════════════════════════════════════════════════════

def step_token_state(mint: str, wallet: str, rpc_url: str) -> dict:
    _head("1. TOKEN STATE")
    result = {
        "token_program":    None,
        "is_t22":           False,
        "is_graduated":     False,
        "dex_id":           None,
        "token_balance_raw": None,
        "ata_exists":       False,
        "error":            None,
    }

    # ── Token program ─────────────────────────────────────────────────────────
    try:
        from memecoin.screener import check_token_program
        info = check_token_program(mint, rpc_url=rpc_url)
        result["token_program"] = info.get("token_program", "unknown")
        result["is_t22"] = info.get("is_token2022", False)
        if result["is_t22"]:
            exts = info.get("extensions_list", [])
            _ok(f"Token program: TOKEN_2022", f"extensions={exts or 'none'}")
        else:
            _warn(f"Token program: {result['token_program']} (not T22)",
                  "Warning: this mint may not be Token-2022")
    except Exception as e:
        result["error"] = f"check_token_program failed: {e}"
        _err("Token program check failed", str(e))

    # ── PumpSwap pool / graduation status ─────────────────────────────────────
    try:
        from memecoin.pumpswap_local import fetch_pool, PumpSwapPoolError
        pool = fetch_pool(mint, rpc_url)
        result["is_graduated"] = True
        result["dex_id"] = "pumpswap"
        _ok("PumpSwap pool found (graduated)",
            f"pool={pool['pool_address'][:16]}…  "
            f"coin_creator={'set' if pool.get('coin_creator') not in ('', '11111111111111111111111111111111') else 'null'}")
    except Exception as e:
        _warn("PumpSwap pool not found (not graduated, or pool lookup failed)", str(e))
        result["dex_id"] = "unknown"

    # ── Wallet ATA balance ────────────────────────────────────────────────────
    try:
        tok_prog = TOKEN_PROGRAM_T22 if result["is_t22"] else TOKEN_PROGRAM_SPL
        data = _rpc(rpc_url, {
            "jsonrpc": "2.0", "id": 1,
            "method": "getTokenAccountsByOwner",
            "params": [
                wallet,
                {"mint": mint},
                {"encoding": "jsonParsed", "programId": tok_prog},
            ],
        })
        accounts = data.get("result", {}).get("value", [])
        if accounts:
            ui = accounts[0]["account"]["data"]["parsed"]["info"]["tokenAmount"]
            raw = int(ui.get("amount", 0))
            result["token_balance_raw"] = raw
            result["ata_exists"] = True
            if raw > 0:
                _ok(f"Wallet ATA balance: {raw:,} raw  ({ui.get('uiAmountString', '?')} tokens)")
            else:
                _warn("Wallet ATA exists but balance = 0",
                      "Probe will still attempt Jupiter quote for any non-zero amount")
        else:
            _warn("No ATA found for this wallet/mint",
                  "Will use a synthetic amount for quote/build/sim tests")
            result["token_balance_raw"] = 0
    except Exception as e:
        _warn(f"ATA lookup failed: {e}")
        result["token_balance_raw"] = 0

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Jupiter quote
# ═══════════════════════════════════════════════════════════════════════════════

def step_jupiter_quote(
    mint: str, amount_raw: int, slippage_bps: int, max_retries: int = 4
) -> tuple[dict | None, str, float]:
    """Returns (quote_dict, error_class, latency_ms). Retries on 429."""
    _head("2. JUPITER QUOTE")
    t0 = time.time()
    error_class = ""
    params = {
        "inputMint":       mint,
        "outputMint":      WSOL_MINT,
        "amount":          str(amount_raw),
        "dynamicSlippage": "true",
        "maxSlippageBps":  str(slippage_bps),
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(JUPITER_QUOTE_URL, params=params, timeout=12)
            latency_ms = (time.time() - t0) * 1000

            if resp.status_code == 429:
                wait = 3 * attempt
                if attempt < max_retries:
                    _warn(f"Jupiter 429 — rate limited (attempt {attempt}/{max_retries}), retrying in {wait}s…",
                          f"latency={latency_ms:.0f}ms")
                    time.sleep(wait)
                    continue
                _err(f"Jupiter 429 — rate limited after {max_retries} attempts", f"latency={latency_ms:.0f}ms")
                return None, JUPITER_429, latency_ms

            if resp.status_code != 200:
                body = resp.text[:300]
                if "no route" in body.lower() or "could not find" in body.lower():
                    _err("Jupiter — no route found", body)
                    return None, JUPITER_ROUTE_NO_LIQUIDITY, latency_ms
                if "token2022" in body.lower() or "unsupported" in body.lower():
                    _err("Jupiter — T22 unsupported", body)
                    return None, TOKEN2022_UNSUPPORTED, latency_ms
                _err(f"Jupiter HTTP {resp.status_code}", body)
                return None, JUPITER_QUOTE_UNAVAILABLE, latency_ms

            quote = resp.json()
            if "error" in quote:
                msg = quote["error"]
                if "no route" in msg.lower() or "could not find" in msg.lower():
                    _err(f"Jupiter no route: {msg}", f"latency={latency_ms:.0f}ms")
                    return None, JUPITER_ROUTE_NO_LIQUIDITY, latency_ms
                _err(f"Jupiter quote error: {msg}", f"latency={latency_ms:.0f}ms")
                return None, JUPITER_QUOTE_UNAVAILABLE, latency_ms

            # Success
            out_amount   = int(quote.get("outAmount", 0))
            price_impact = float(quote.get("priceImpactPct", 0))
            route_labels = [
                s.get("label", s.get("ammKey", "?"))
                for plan in quote.get("routePlan", [])
                for s in [plan.get("swapInfo", {})]
            ]

            suffix = f"  (attempt {attempt})" if attempt > 1 else ""
            _ok(f"Quote OK  latency={latency_ms:.0f}ms{suffix}",
                f"outAmount={out_amount:,} lamports ({out_amount/1e9:.6f} SOL)\n"
                f"priceImpact={price_impact:.4f}%\n"
                f"route={route_labels}")

            if price_impact > 50:
                _warn(f"Price impact very high: {price_impact:.2f}%")
                error_class = JUPITER_PRICE_IMPACT_TOO_HIGH

            return quote, error_class, latency_ms

        except requests.exceptions.Timeout:
            latency_ms = (time.time() - t0) * 1000
            if attempt < max_retries:
                _warn(f"Jupiter quote timeout (attempt {attempt}/{max_retries}), retrying…")
                time.sleep(2)
                continue
            _err("Jupiter quote timed out", f"latency={latency_ms:.0f}ms")
            return None, JUPITER_QUOTE_UNAVAILABLE, latency_ms
        except Exception as e:
            latency_ms = (time.time() - t0) * 1000
            _err(f"Jupiter quote exception: {e}", f"latency={latency_ms:.0f}ms")
            return None, JUPITER_QUOTE_UNAVAILABLE, latency_ms

    latency_ms = (time.time() - t0) * 1000
    return None, JUPITER_429, latency_ms


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Jupiter swap TX build
# ═══════════════════════════════════════════════════════════════════════════════

def step_jupiter_swap_build(
    quote: dict, wallet: str, slippage_bps: int
) -> tuple[bytes | None, str, float, int, int]:
    """Returns (tx_bytes, error_class, latency_ms, tx_size, account_count)."""
    _head("3. JUPITER SWAP TX BUILD (no sign/send)")
    t0 = time.time()
    try:
        payload = {
            "quoteResponse":    quote,
            "userPublicKey":    wallet,
            "wrapAndUnwrapSol": True,
            "dynamicSlippage":  {"maxBps": slippage_bps},
            "prioritizationFeeLamports": 100_000,
        }
        resp = requests.post(JUPITER_SWAP_URL, json=payload, timeout=15)
        latency_ms = (time.time() - t0) * 1000

        if resp.status_code == 429:
            _err("Jupiter swap build 429 — rate limited", f"latency={latency_ms:.0f}ms")
            return None, JUPITER_429, latency_ms, 0, 0

        if resp.status_code != 200:
            body = resp.text[:400]
            _err(f"Jupiter swap build HTTP {resp.status_code}", body)
            return None, JUPITER_SWAP_BUILD_FAILED, latency_ms, 0, 0

        data = resp.json()
        if "error" in data:
            _err(f"Jupiter swap build error: {data['error']}", f"latency={latency_ms:.0f}ms")
            return None, JUPITER_SWAP_BUILD_FAILED, latency_ms, 0, 0

        swap_tx_b64 = data.get("swapTransaction", "")
        if not swap_tx_b64:
            _err("Jupiter swap build: no swapTransaction in response")
            return None, JUPITER_SWAP_BUILD_FAILED, latency_ms, 0, 0

        tx_bytes = base64.b64decode(swap_tx_b64)

        # Parse account count
        account_count = 0
        try:
            from solders.transaction import VersionedTransaction
            tx = VersionedTransaction.from_bytes(tx_bytes)
            account_count = len(tx.message.account_keys)
        except Exception:
            pass

        _ok(f"Swap TX built  latency={latency_ms:.0f}ms",
            f"tx_size={len(tx_bytes)} bytes  accounts={account_count}")
        return tx_bytes, "", latency_ms, len(tx_bytes), account_count

    except requests.exceptions.Timeout:
        latency_ms = (time.time() - t0) * 1000
        _err("Jupiter swap build timed out")
        return None, JUPITER_SWAP_BUILD_FAILED, latency_ms, 0, 0
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        _err(f"Jupiter swap build exception: {e}")
        return None, JUPITER_SWAP_BUILD_FAILED, latency_ms, 0, 0


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Simulate TX
# ═══════════════════════════════════════════════════════════════════════════════

def step_simulate(tx_bytes: bytes, rpc_url: str) -> tuple[bool, str, list[str], float]:
    """Returns (sim_ok, error_class, logs, latency_ms)."""
    _head("4. SIMULATE TRANSACTION (sigVerify=False, no broadcast)")
    t0 = time.time()
    tx_b64 = base64.b64encode(tx_bytes).decode()
    try:
        data = _rpc(rpc_url, {
            "jsonrpc": "2.0", "id": 1,
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
        }, timeout=20)
        latency_ms = (time.time() - t0) * 1000
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000
        _err(f"simulateTransaction RPC error: {e}", f"latency={latency_ms:.0f}ms")
        return False, RPC_SIM_FAILED, [str(e)], latency_ms

    value = data.get("result", {}).get("value", {})
    err   = value.get("err")
    logs  = value.get("logs") or []

    if err is None:
        _ok(f"Simulation OK  latency={latency_ms:.0f}ms",
            f"logs={len(logs)} lines")
        for line in logs[:6]:
            print(f"      {line[:120]}")
        if len(logs) > 6:
            print(f"      … +{len(logs)-6} more lines")
        return True, JUPITER_SIM_OK, logs, latency_ms

    # Classify error
    err_str  = str(err)
    combined = err_str + "\n" + "\n".join(logs)
    combined_lower = combined.lower()

    if "429" in combined or "rate limit" in combined_lower:
        cls = JUPITER_429
    elif "no route" in combined_lower or "could not find" in combined_lower:
        cls = JUPITER_ROUTE_NO_LIQUIDITY
    elif "token2022" in combined_lower and "unsupported" in combined_lower:
        cls = TOKEN2022_UNSUPPORTED
    elif "invalid program id" in combined_lower or "incorrectprogramid" in combined_lower:
        cls = TOKEN2022_UNSUPPORTED
    else:
        cls = JUPITER_SIM_FAILED

    _err(f"Simulation FAILED  error_class={cls}  latency={latency_ms:.0f}ms",
         f"err={err_str}")
    for line in logs[:8]:
        print(f"      {line[:120]}")
    if len(logs) > 8:
        print(f"      … +{len(logs)-8} more lines")

    return False, cls, logs, latency_ms


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Local route comparison
# ═══════════════════════════════════════════════════════════════════════════════

def step_local_comparison(
    mint: str, wallet: str, amount_raw: int, rpc_url: str
) -> tuple[bool, bool, str, list[str]]:
    """Returns (local_build_ok, local_sim_ok, local_error_class, local_logs)."""
    _head("5. LOCAL PUMPSWAP_LOCAL ROUTE COMPARISON")

    local_build_ok   = False
    local_sim_ok     = False
    local_error_class = "not_attempted"
    local_logs: list[str] = []

    try:
        from memecoin.pumpswap_local import (
            fetch_pool, build_pumpswap_sell_tx, simulate_sell,
            TOKEN_PROGRAM_T22, PumpSwapPoolError,
        )
        from solders.keypair import Keypair

        dummy_kp = Keypair()
        # wallet_pubkey must match keypair.pubkey() — use dummy's pubkey for dry-run build
        dummy_wallet = str(dummy_kp.pubkey())

        # Build
        try:
            pool = fetch_pool(mint, rpc_url)
            tx_bytes = build_pumpswap_sell_tx(
                wallet_pubkey=dummy_wallet,
                keypair=dummy_kp,
                token_mint=mint,
                token_amount_raw=amount_raw,
                min_sol_out_lamports=0,
                priority_fee_sol=0.0001,
                token_program_id=TOKEN_PROGRAM_T22,
                pool=pool,
                rpc_url=rpc_url,
            )
            local_build_ok = True
            _ok(f"Local build OK  tx={len(tx_bytes)} bytes")
        except PumpSwapPoolError as e:
            local_error_class = e.args[0] if e.args else "pumpswap_pool_error"
            _err(f"Local build FAILED (PumpSwapPoolError): {local_error_class}", str(e))
            return local_build_ok, local_sim_ok, local_error_class, local_logs
        except Exception as e:
            local_error_class = "local_build_exception"
            _err(f"Local build exception: {e}")
            return local_build_ok, local_sim_ok, local_error_class, local_logs

        # Simulate
        local_sim_ok, local_error_class, local_logs = simulate_sell(tx_bytes, rpc_url)
        if local_sim_ok:
            _ok("Local sim OK")
        else:
            _err(f"Local sim FAILED: {local_error_class}")
            for line in local_logs[:5]:
                print(f"      {line[:120]}")

    except ImportError as e:
        local_error_class = "import_error"
        _err(f"Import error: {e}")

    return local_build_ok, local_sim_ok, local_error_class, local_logs


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Verdict + comparison table
# ═══════════════════════════════════════════════════════════════════════════════

def step_verdict(
    token_state: dict,
    quote_ok: bool,
    quote_error_class: str,
    swap_build_ok: bool,
    swap_error_class: str,
    sim_ok: bool,
    sim_error_class: str,
    local_build_ok: bool,
    local_sim_ok: bool,
    local_error_class: str,
) -> str:
    _head("6. VERDICT")

    # Comparison table
    print(f"  {'Route':<26}  {'Build':<6}  {'Sim':<6}  {'Error class'}")
    print(f"  {'─'*26}  {'─'*6}  {'─'*6}  {'─'*30}")

    def _tick(v): return f"{_GREEN}PASS{_RESET}" if v else f"{_RED}FAIL{_RESET}"

    jup_build = quote_ok and swap_build_ok
    print(f"  {'Jupiter':<26}  {_tick(jup_build):<15}  {_tick(sim_ok):<15}  {sim_error_class or swap_error_class or quote_error_class or '—'}")
    print(f"  {'PUMPSWAP_LOCAL (T22)':<26}  {_tick(local_build_ok):<15}  {_tick(local_sim_ok):<15}  {local_error_class or '—'}")

    # Verdict logic
    if not token_state.get("is_t22"):
        verdict = FAIL_UNKNOWN
        _warn("Mint is not Token-2022 — results may not be representative")
    elif not quote_ok:
        if quote_error_class == JUPITER_429:
            verdict = FAIL_JUPITER_429
        elif quote_error_class == JUPITER_ROUTE_NO_LIQUIDITY:
            verdict = FAIL_JUPITER_NO_ROUTE
        else:
            verdict = FAIL_UNKNOWN
    elif not swap_build_ok:
        if swap_error_class == JUPITER_429:
            verdict = FAIL_JUPITER_429
        else:
            verdict = FAIL_UNKNOWN
    elif sim_ok:
        verdict = PASS_JUPITER_T22_SELL_SIM_OK
    elif sim_error_class == JUPITER_429:
        verdict = FAIL_JUPITER_429
    else:
        verdict = FAIL_JUPITER_SIM_FAILED

    color = _GREEN if verdict.startswith("PASS") else _RED
    print(f"\n  {color}{_BOLD}OVERALL VERDICT: {verdict}{_RESET}")

    return verdict


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Write CSV log
# ═══════════════════════════════════════════════════════════════════════════════

def write_log(row: dict):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    fieldnames = [
        "timestamp", "mint", "wallet", "token_program", "state",
        "token_balance_raw",
        "jupiter_quote_ok", "jupiter_swap_build_ok", "jupiter_sim_ok",
        "jupiter_error_class", "price_impact", "expected_out", "route_labels",
        "quote_latency_ms", "swap_latency_ms", "sim_latency_ms",
        "local_build_ok", "local_sim_ok", "local_error_class",
        "verdict",
    ]
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"\n  Log written → {LOG_PATH}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Probe whether Jupiter can sell a T22 PumpSwap token — simulation only"
    )
    parser.add_argument("--mint",               required=True,  help="Token-2022 graduated mint address")
    parser.add_argument("--wallet",             required=False, help="Bot wallet pubkey (default: from config)")
    parser.add_argument("--token-amount-mode",  default="percent",
                        choices=["percent", "raw"],
                        help="How to interpret --token-amount (default: percent)")
    parser.add_argument("--token-amount",       type=float, default=100.0,
                        help="Token amount to quote: percentage of balance (default: 100) or raw amount")
    parser.add_argument("--slippage-bps",       type=int, default=3000,
                        help="Max slippage in bps (default: 3000 = 30%%)")
    parser.add_argument("--rpc",                help="RPC URL (default: from config)")
    args = parser.parse_args()

    # ── Resolve wallet ────────────────────────────────────────────────────────
    wallet = args.wallet
    if not wallet:
        try:
            from memecoin.executor import _get_keypair
            wallet = str(_get_keypair().pubkey())
            print(f"Using bot wallet: {wallet[:16]}…")
        except Exception as e:
            print(f"{_RED}ERROR: --wallet not provided and could not load from config: {e}{_RESET}")
            sys.exit(1)

    # ── Resolve RPC ───────────────────────────────────────────────────────────
    rpc_url = args.rpc
    if not rpc_url:
        # Prefer the env var (set by .env / systemd EnvironmentFile)
        rpc_url = os.environ.get("SOLANA_RPC_URL", "")
    if not rpc_url:
        try:
            import memecoin.config as cfg
            rpc_url = cfg.CHAINS.get("solana", {}).get("rpc", "")
        except Exception:
            pass
    if not rpc_url:
        rpc_url = "https://api.mainnet-beta.solana.com"
        print(f"Using public RPC (rate limits apply): {rpc_url}")

    mint = args.mint

    print(f"\n{'═'*60}")
    print(f"{_BOLD}JUPITER T22 SELL PROBE{_RESET}")
    print(f"{'═'*60}")
    print(f"  mint:           {mint}")
    print(f"  wallet:         {wallet[:16]}…")
    print(f"  slippage_bps:   {args.slippage_bps}")
    print(f"  amount_mode:    {args.token_amount_mode} ({args.token_amount})")
    print(f"  rpc:            {rpc_url[:50]}…" if len(rpc_url) > 50 else f"  rpc:            {rpc_url}")
    print(f"\n  {_YELLOW}NO TRANSACTIONS WILL BE SENT{_RESET}")

    # ── Step 1: token state ───────────────────────────────────────────────────
    token_state = step_token_state(mint, wallet, rpc_url)

    # ── Resolve amount ────────────────────────────────────────────────────────
    balance_raw = token_state.get("token_balance_raw") or 0
    if args.token_amount_mode == "percent":
        if balance_raw > 0:
            amount_raw = max(1, int(balance_raw * args.token_amount / 100))
        else:
            # No balance — use synthetic 1M for quote/build/sim probe
            amount_raw = 1_000_000
            _warn("No balance found — using synthetic 1,000,000 raw tokens for probe")
    else:
        amount_raw = int(args.token_amount)

    print(f"\n  Probe amount: {amount_raw:,} raw tokens")

    # Accumulators
    quote             = None
    quote_ok          = False
    quote_error_class = ""
    price_impact      = 0.0
    expected_out      = 0
    route_labels      = []
    quote_latency_ms  = 0.0

    swap_build_ok    = False
    swap_error_class = ""
    swap_latency_ms  = 0.0
    tx_size          = 0
    account_count    = 0

    sim_ok            = False
    sim_error_class   = ""
    sim_logs: list[str] = []
    sim_latency_ms    = 0.0

    local_build_ok    = False
    local_sim_ok      = False
    local_error_class = "not_attempted"

    # ── Step 2: Jupiter quote ─────────────────────────────────────────────────
    quote, quote_ec, quote_latency_ms = step_jupiter_quote(
        mint, amount_raw, args.slippage_bps
    )
    quote_ok = (quote is not None)
    quote_error_class = quote_ec

    if quote is not None:
        expected_out  = int(quote.get("outAmount", 0))
        price_impact  = float(quote.get("priceImpactPct", 0))
        route_labels  = [
            s.get("label", s.get("ammKey", "?"))
            for plan in quote.get("routePlan", [])
            for s in [plan.get("swapInfo", {})]
        ]

        # ── Step 3: swap TX build ─────────────────────────────────────────────
        tx_bytes, swap_ec, swap_latency_ms, tx_size, account_count = step_jupiter_swap_build(
            quote, wallet, args.slippage_bps
        )
        swap_build_ok    = (tx_bytes is not None)
        swap_error_class = swap_ec

        if tx_bytes is not None:
            # ── Step 4: simulate ──────────────────────────────────────────────
            sim_ok, sim_error_class, sim_logs, sim_latency_ms = step_simulate(
                tx_bytes, rpc_url
            )
    else:
        _warn("Skipping swap build and simulation — no quote")

    # ── Step 5: local comparison ──────────────────────────────────────────────
    local_build_ok, local_sim_ok, local_error_class, _ = step_local_comparison(
        mint, wallet, amount_raw, rpc_url
    )

    # ── Step 6: verdict ───────────────────────────────────────────────────────
    verdict = step_verdict(
        token_state=token_state,
        quote_ok=quote_ok,
        quote_error_class=quote_error_class,
        swap_build_ok=swap_build_ok,
        swap_error_class=swap_error_class,
        sim_ok=sim_ok,
        sim_error_class=sim_error_class,
        local_build_ok=local_build_ok,
        local_sim_ok=local_sim_ok,
        local_error_class=local_error_class,
    )

    # ── Step 7: log ───────────────────────────────────────────────────────────
    write_log({
        "timestamp":            datetime.now(timezone.utc).isoformat(),
        "mint":                 mint,
        "wallet":               wallet,
        "token_program":        token_state.get("token_program", ""),
        "state":                "graduated_pumpswap" if token_state.get("is_graduated") else "unknown",
        "token_balance_raw":    balance_raw,
        "jupiter_quote_ok":     quote_ok,
        "jupiter_swap_build_ok": swap_build_ok,
        "jupiter_sim_ok":       sim_ok,
        "jupiter_error_class":  sim_error_class or swap_error_class or quote_error_class,
        "price_impact":         price_impact,
        "expected_out":         expected_out,
        "route_labels":         "|".join(route_labels),
        "quote_latency_ms":     round(quote_latency_ms, 1),
        "swap_latency_ms":      round(swap_latency_ms, 1),
        "sim_latency_ms":       round(sim_latency_ms, 1),
        "local_build_ok":       local_build_ok,
        "local_sim_ok":         local_sim_ok,
        "local_error_class":    local_error_class,
        "verdict":              verdict,
    })

    print(f"\n{'═'*60}\n")
    sys.exit(0 if verdict == PASS_JUPITER_T22_SELL_SIM_OK else 1)


if __name__ == "__main__":
    main()
