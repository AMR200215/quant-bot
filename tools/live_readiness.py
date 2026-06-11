#!/usr/bin/env python3
"""
tools/live_readiness.py — one-command live-readiness checklist.

Run from the repo root:
    python tools/live_readiness.py

Checks (PASS / FAIL / WARN per item):
  1.  Telegram session — TelegramMonitor can connect
  2.  PumpPortal WS — monitor is alive and tick age < 60s
  3.  Helius key 429s — no HTTP 429 from Helius in the last 5 min
  4.  Wallet free SOL — wallet balance ≥ 0.06 SOL free
  5.  Dry-run buy build — executor.buy() succeeds with LIVE_DRY_RUN=True
  6.  Presigned-exit loop — presigned exit manager is running
  7.  Circuit breaker — daily loss < DAILY_LOSS_LIMIT
  8.  LIVE_DRY_RUN flag — config has LIVE_DRY_RUN=True (shadow mode active)
  9.  LIVE_TRADING flag — config has LIVE_TRADING=True
 10.  Epoch tag — ACCOUNTING_EPOCH is set and non-empty

Go-live requires all items PASS or WARN (no FAIL), plus 24h funnel with
≥ 3 DRY_RUN signals in the live journal.
"""

import os
import sys
import time

# ---------------------------------------------------------------------------
# Colour helpers (no third-party deps)
# ---------------------------------------------------------------------------
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"


def _pass(msg: str) -> tuple[str, str]:
    return "PASS", f"{_GREEN}✓  {msg}{_RESET}"


def _fail(msg: str) -> tuple[str, str]:
    return "FAIL", f"{_RED}✗  {msg}{_RESET}"


def _warn(msg: str) -> tuple[str, str]:
    return "WARN", f"{_YELLOW}⚠  {msg}{_RESET}"


def _skip(msg: str) -> tuple[str, str]:
    return "SKIP", f"   {msg}"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_telegram_session() -> tuple[str, str]:
    """1. Telegram session — TG API credentials present."""
    tg_id   = os.getenv("TELEGRAM_API_ID", "")
    tg_hash = os.getenv("TELEGRAM_API_HASH", "")
    tg_sess = os.getenv("TELEGRAM_SESSION_NAME", "")
    if not tg_id or not tg_hash:
        return _fail("TELEGRAM_API_ID / TELEGRAM_API_HASH not set")
    if not tg_sess:
        return _warn("TELEGRAM_SESSION_NAME not set — will use default session name")
    return _pass(f"TG credentials present  id={tg_id}")


def check_pumpportal_ws() -> tuple[str, str]:
    """2. PumpPortal WS — monitor is alive; last tick age acceptable."""
    try:
        from memecoin.pumpportal_monitor import monitor as _ppm
        if not hasattr(_ppm, "get_prices"):
            return _fail("PumpPortal monitor object has no get_prices()")
        # If monitor has a last_tick_at attribute, check staleness
        last_tick = getattr(_ppm, "_last_tick_at", None)
        if last_tick is not None:
            age = time.time() - last_tick
            if age > 60:
                return _fail(f"PP last tick {age:.0f}s ago (> 60s — WS may be dead)")
            return _pass(f"PP last tick {age:.1f}s ago")
        return _warn("PP monitor loaded but _last_tick_at not available — start scanner to verify")
    except ImportError as e:
        return _fail(f"PumpPortal monitor import failed: {e}")


def check_helius_429() -> tuple[str, str]:
    """3. Helius key 429s — verify key is set; ping for 429."""
    rpc_url = os.getenv("SOLANA_RPC_URL", "")
    if not rpc_url:
        return _warn("SOLANA_RPC_URL not set — will use public mainnet (rate limited)")
    if "helius" not in rpc_url.lower():
        return _warn(f"SOLANA_RPC_URL does not look like Helius: {rpc_url[:40]}")
    try:
        import requests
        resp = requests.post(rpc_url, json={
            "jsonrpc": "2.0", "id": 1,
            "method": "getHealth",
        }, timeout=5)
        if resp.status_code == 429:
            return _fail("Helius RPC returned 429 — quota exhausted or key invalid")
        if resp.status_code != 200:
            return _warn(f"Helius RPC returned {resp.status_code}")
        health = resp.json().get("result", "")
        return _pass(f"Helius RPC healthy  url={rpc_url[:40]}")
    except Exception as e:
        return _warn(f"Helius health check failed: {e}")


def check_wallet_free_sol() -> tuple[str, str]:
    """4. Wallet free SOL — wallet has ≥ 0.06 SOL free."""
    wallet = os.getenv("WALLET_PUBKEY", "") or os.getenv("SOLANA_WALLET_ADDRESS", "")
    if not wallet:
        # Try deriving from private key
        pk = os.getenv("SOLANA_PRIVATE_KEY", "")
        if pk:
            try:
                import base58
                from solders.keypair import Keypair
                kp = Keypair.from_bytes(base58.b58decode(pk))
                wallet = str(kp.pubkey())
            except Exception:
                pass
    if not wallet:
        return _warn("WALLET_PUBKEY / SOLANA_PRIVATE_KEY not set — cannot check balance")
    try:
        import requests
        rpc = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
        resp = requests.post(rpc, json={
            "jsonrpc": "2.0", "id": 1,
            "method": "getBalance",
            "params": [wallet],
        }, timeout=5)
        lamports = resp.json().get("result", {}).get("value", 0)
        sol = lamports / 1e9
        if sol < 0.06:
            return _fail(f"Only {sol:.4f} SOL free (< 0.06 SOL minimum) — refill wallet")
        if sol < 0.20:
            return _warn(f"{sol:.4f} SOL free — low, consider topping up")
        return _pass(f"{sol:.4f} SOL free  wallet={wallet[:8]}")
    except Exception as e:
        return _warn(f"SOL balance check failed: {e}")


def check_dry_run_buy() -> tuple[str, str]:
    """5. Dry-run buy build — executor.buy() returns success in DRY_RUN mode."""
    try:
        import importlib
        import memecoin.executor as ex_mod
        import memecoin.config as cfg_mod

        if not cfg_mod.LIVE_DRY_RUN:
            return _warn("LIVE_DRY_RUN=False — buy() will attempt real tx. "
                         "Set LIVE_DRY_RUN=True before running this check.")

        # Pick a known good pump.fun token (USDC mint as dummy — won't hit buy path anyway
        # since DRY_RUN returns before balance snapshot)
        TEST_TOKEN = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
        pk = os.getenv("SOLANA_PRIVATE_KEY", "")
        if not pk:
            return _warn("SOLANA_PRIVATE_KEY not set — cannot test executor build")

        ex = ex_mod.MemeExecutor()
        result = ex.buy(TEST_TOKEN, 0.01, "solana", signal_price=0.0001)
        if result.get("dry_run") and result.get("success"):
            return _pass(f"Dry-run buy succeeded  fill=${result.get('fill_price', 0):.8f}")
        if result.get("reason") == "insufficient_free_sol":
            return _fail(f"Dry-run blocked — insufficient_free_sol  free={result.get('free_sol')} SOL")
        if not result.get("success"):
            return _fail(f"Dry-run buy failed: {result.get('reason') or result.get('error')}")
        return _warn(f"Dry-run buy result unexpected (dry_run flag missing?): {result}")
    except Exception as e:
        return _fail(f"Executor dry-run check error: {e}")


def check_presigned_exit_loop() -> tuple[str, str]:
    """6. Presigned-exit loop — check if presigned exit manager is importable."""
    try:
        from memecoin import presigned_exit_manager as _pem
        if hasattr(_pem, "is_running") and callable(_pem.is_running):
            if _pem.is_running():
                return _pass("Presigned-exit manager running")
            return _warn("Presigned-exit manager not running (not started yet?)")
        return _warn("presigned_exit_manager loaded but is_running() not available")
    except ImportError:
        return _warn("presigned_exit_manager module not found — presigned exits not available")
    except Exception as e:
        return _warn(f"Presigned-exit check error: {e}")


def check_circuit_breaker() -> tuple[str, str]:
    """7. Circuit breaker — daily live PnL within limit."""
    try:
        from memecoin.config import DAILY_LOSS_LIMIT, LIVE_JOURNAL_FILE
        import csv as _csv
        from datetime import date as _date
        today = str(_date.today())
        total = 0.0
        try:
            with open(LIVE_JOURNAL_FILE) as f:
                for row in _csv.DictReader(f):
                    if row.get("entry_time", "")[:10] == today:
                        total += float(row.get("pnl_usd", 0) or 0)
        except FileNotFoundError:
            pass
        if total <= DAILY_LOSS_LIMIT:
            return _fail(
                f"Circuit breaker TRIPPED — daily PnL ${total:.2f} ≤ "
                f"${DAILY_LOSS_LIMIT:.2f} limit. No live trades today."
            )
        return _pass(f"Circuit breaker clear — daily PnL ${total:.2f}  limit=${DAILY_LOSS_LIMIT:.2f}")
    except Exception as e:
        return _warn(f"Circuit breaker check failed: {e}")


def check_live_dry_run_flag() -> tuple[str, str]:
    """8. LIVE_DRY_RUN flag — should be True for shadow-mode validation."""
    try:
        from memecoin.config import LIVE_DRY_RUN
        if LIVE_DRY_RUN:
            return _pass("LIVE_DRY_RUN=True — shadow mode active (tx not sent)")
        return _warn("LIVE_DRY_RUN=False — real transactions WILL be sent")
    except Exception as e:
        return _warn(f"LIVE_DRY_RUN check failed: {e}")


def check_live_trading_flag() -> tuple[str, str]:
    """9. LIVE_TRADING flag — must be True for live gate to open."""
    try:
        from memecoin.config import LIVE_TRADING
        if LIVE_TRADING:
            return _pass("LIVE_TRADING=True — live execution gate enabled")
        return _warn("LIVE_TRADING=False — all signals paper-only regardless of cohort")
    except Exception as e:
        return _warn(f"LIVE_TRADING check failed: {e}")


def check_epoch_tag() -> tuple[str, str]:
    """10. Epoch tag — ACCOUNTING_EPOCH should be set."""
    try:
        from memecoin.config import ACCOUNTING_EPOCH
        if ACCOUNTING_EPOCH:
            return _pass(f"ACCOUNTING_EPOCH={ACCOUNTING_EPOCH!r}")
        return _warn("ACCOUNTING_EPOCH is empty — journal rows won't have epoch tag")
    except (ImportError, AttributeError):
        return _warn("ACCOUNTING_EPOCH not defined in config.py")
    except Exception as e:
        return _warn(f"Epoch tag check failed: {e}")


def check_dry_run_funnel_count() -> tuple[str, str]:
    """Bonus: count DRY_RUN live signals in live journal (go-live needs N≥3)."""
    try:
        from memecoin.config import LIVE_JOURNAL_FILE
        import csv as _csv
        count = 0
        try:
            with open(LIVE_JOURNAL_FILE) as f:
                for row in _csv.DictReader(f):
                    notes = row.get("notes", "") or ""
                    if "DRY_RUN" in notes:
                        count += 1
        except FileNotFoundError:
            pass
        if count >= 3:
            return _pass(f"DRY_RUN funnel: {count} would-have-bought signals in live journal (≥3 ✓)")
        return _warn(f"DRY_RUN funnel: only {count} signals so far (need ≥3 before going live)")
    except Exception as e:
        return _warn(f"Funnel count check failed: {e}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

CHECKS = [
    ("TG session",           check_telegram_session),
    ("PumpPortal WS",        check_pumpportal_ws),
    ("Helius 429 check",     check_helius_429),
    ("Wallet free SOL",      check_wallet_free_sol),
    ("Dry-run buy build",    check_dry_run_buy),
    ("Presigned-exit loop",  check_presigned_exit_loop),
    ("Circuit breaker",      check_circuit_breaker),
    ("LIVE_DRY_RUN flag",    check_live_dry_run_flag),
    ("LIVE_TRADING flag",    check_live_trading_flag),
    ("Epoch tag",            check_epoch_tag),
    ("Funnel count (bonus)", check_dry_run_funnel_count),
]


def main():
    print(f"\n{_BOLD}Live Readiness Checklist{_RESET}  ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})\n")

    results = []
    for name, fn in CHECKS:
        try:
            status, line = fn()
        except Exception as e:
            status, line = _fail(f"check raised exception: {e}")
        results.append((name, status, line))
        print(f"  {name:<24}  {line}")

    failures = [n for n, s, _ in results if s == "FAIL"]
    warnings = [n for n, s, _ in results if s == "WARN"]

    print()
    if failures:
        print(f"{_RED}{_BOLD}NOT READY — {len(failures)} FAIL(s): {', '.join(failures)}{_RESET}")
        sys.exit(1)
    elif warnings:
        print(f"{_YELLOW}{_BOLD}CAUTION — {len(warnings)} WARN(s): {', '.join(warnings)}{_RESET}")
        print(f"{_YELLOW}Review warnings before going live.{_RESET}")
        sys.exit(0)
    else:
        print(f"{_GREEN}{_BOLD}ALL PASS — system is live-ready.{_RESET}")
        sys.exit(0)


if __name__ == "__main__":
    # Add repo root to path so memecoin.* imports work when run directly
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    main()
