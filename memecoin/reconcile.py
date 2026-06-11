"""
memecoin/reconcile.py — daily wallet vs journal reconciliation.

Compares the wallet's SOL balance delta (since last reconciliation)
against the sum of live trade PnL from the live journal over the same window.
Fires a Telegram alert if the discrepancy exceeds $0.50.

The wallet is the arbiter — if it disagrees with the journal, the journal is wrong.

State stored in: memecoin/data/reconcile_state.json
  {
    "last_ts":      <unix float>,   # timestamp of last reconciliation
    "last_sol_bal": <float>,        # SOL balance at last reconciliation
    "last_sol_price": <float>       # SOL/USD at last reconciliation
  }

Called by:
  - scanner.py at startup and every 24h via _reconcile_loop thread
  - Standalone:  python -m memecoin.reconcile
"""

import json
import logging
import time
from datetime import datetime, timezone, date
from pathlib import Path

log = logging.getLogger(__name__)

_STATE_FILE = Path(__file__).parent / "data" / "reconcile_state.json"
DISCREPANCY_THRESHOLD_USD = 0.50


def _get_sol_balance() -> float:
    """Return current SOL balance of the bot wallet in SOL.

    Requires WALLET_PUBKEY env var (base58 public key string).
    In paper-trading mode this env var is typically not set — function
    returns 0.0 silently and reconciliation is skipped.
    """
    import os
    import requests as _req
    pubkey_str = os.getenv("WALLET_PUBKEY", "").strip()
    if not pubkey_str:
        # Paper-trading mode: no real wallet to check — expected, not an error.
        return 0.0
    try:
        resp = _req.post(
            "https://api.mainnet-beta.solana.com",
            json={"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [pubkey_str]},
            timeout=10,
        )
        lamports = resp.json()["result"]["value"]
        return lamports / 1e9
    except Exception as e:
        log.warning("reconcile: get_sol_balance failed: %s", e)
        return 0.0


def _get_sol_price() -> float:
    """Return current SOL/USD price."""
    try:
        from memecoin.executor import _sol_price_usd
        return _sol_price_usd()
    except Exception as e:
        log.warning("reconcile: get_sol_price failed: %s", e)
        return 0.0


def _load_state() -> dict:
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_state(state: dict):
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state, indent=2))


def _live_journal_pnl_since(since_ts: float) -> float:
    """Sum realized pnl_usd from live journal for trades closed after since_ts."""
    import csv as _csv
    from memecoin.config import LIVE_JOURNAL_FILE
    total = 0.0
    try:
        with open(LIVE_JOURNAL_FILE, newline="") as f:
            for row in _csv.DictReader(f):
                et = row.get("exit_time", "")
                if not et:
                    continue
                try:
                    dt = datetime.strptime(et.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    if dt.timestamp() > since_ts:
                        total += float(row.get("pnl_usd") or 0)
                except (ValueError, TypeError):
                    continue
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning("reconcile: live journal read error: %s", e)
    return total


def reconcile(force: bool = False) -> dict:
    """
    Run one reconciliation cycle.

    Returns a result dict with keys: ok, discrepancy_usd, wallet_delta_usd,
    journal_delta_usd, sol_balance, sol_price, message.
    """
    state = _load_state()
    now   = time.time()

    # On first run there's no baseline — just establish it.
    if not state or not state.get("last_sol_bal"):
        sol_bal   = _get_sol_balance()
        sol_price = _get_sol_price()
        if sol_bal == 0:
            log.info("reconcile: WALLET_PUBKEY not set or balance unavailable — skipping (paper mode)")
            return {"ok": True, "message": "baseline skipped (paper mode / WALLET_PUBKEY not set)"}
        _save_state({
            "last_ts": now,
            "last_sol_bal": sol_bal,
            "last_sol_price": sol_price,
        })
        log.info(
            "reconcile: baseline established  sol=%.4f  price=%.2f  usd=%.2f",
            sol_bal, sol_price, sol_bal * sol_price,
        )
        return {"ok": True, "message": f"baseline set: {sol_bal:.4f} SOL @ ${sol_price:.2f}"}

    last_ts        = float(state["last_ts"])
    last_sol_bal   = float(state["last_sol_bal"])
    last_sol_price = float(state.get("last_sol_price") or 0) or 150.0

    # Fetch current state
    sol_bal   = _get_sol_balance()
    sol_price = _get_sol_price()
    if sol_bal == 0 or sol_price == 0:
        log.warning("reconcile: skipping — could not read balance or price")
        return {"ok": True, "message": "skipped (balance/price unavailable)"}

    # Wallet delta in USD (use current SOL price for both — close enough for daily check)
    wallet_delta_usd  = (sol_bal - last_sol_bal) * sol_price
    journal_delta_usd = _live_journal_pnl_since(last_ts)
    discrepancy       = abs(wallet_delta_usd - journal_delta_usd)

    since_str = datetime.fromtimestamp(last_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    log.info(
        "reconcile: since=%s  wallet_delta=$%.2f  journal_delta=$%.2f  discrepancy=$%.2f",
        since_str, wallet_delta_usd, journal_delta_usd, discrepancy,
    )

    ok = discrepancy <= DISCREPANCY_THRESHOLD_USD
    msg = (
        f"Reconcile OK: wallet ${wallet_delta_usd:+.2f}  journal ${journal_delta_usd:+.2f}  "
        f"diff ${discrepancy:.2f}"
        if ok else
        f"RECONCILE MISMATCH: wallet ${wallet_delta_usd:+.2f}  journal ${journal_delta_usd:+.2f}  "
        f"diff ${discrepancy:.2f} > ${DISCREPANCY_THRESHOLD_USD}  since {since_str}"
    )

    if not ok:
        log.error("RECONCILE MISMATCH  wallet_delta=$%.2f  journal_delta=$%.2f  diff=$%.2f",
                  wallet_delta_usd, journal_delta_usd, discrepancy)
        try:
            from app.alerts import _send
            _send(
                f"⚠️ WALLET RECONCILE MISMATCH\n"
                f"Wallet: ${wallet_delta_usd:+.2f}  Journal: ${journal_delta_usd:+.2f}\n"
                f"Discrepancy: ${discrepancy:.2f} (threshold ${DISCREPANCY_THRESHOLD_USD})\n"
                f"Since: {since_str}\n"
                f"Action: review live journal vs on-chain txs"
            )
        except Exception as _ae:
            log.debug("reconcile: alert send failed: %s", _ae)

    # Update state for next run
    _save_state({
        "last_ts": now,
        "last_sol_bal": sol_bal,
        "last_sol_price": sol_price,
    })

    return {
        "ok": ok,
        "discrepancy_usd": round(discrepancy, 4),
        "wallet_delta_usd": round(wallet_delta_usd, 4),
        "journal_delta_usd": round(journal_delta_usd, 4),
        "sol_balance": sol_bal,
        "sol_price": sol_price,
        "message": msg,
    }


def _reconcile_loop():
    """Background thread: run reconciliation every 24h."""
    import threading
    # Delay first run by 60s so the scanner has fully initialized
    time.sleep(60)
    reconcile()
    while True:
        time.sleep(24 * 3600)
        reconcile()


def start_background():
    """Start the reconciliation background thread. Called from scanner.py."""
    import threading
    t = threading.Thread(target=_reconcile_loop, daemon=True, name="reconcile")
    t.start()
    log.info("reconcile: background thread started (24h interval)")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    result = reconcile(force=True)
    print()
    print(result["message"])
    sys.exit(0 if result["ok"] else 1)
