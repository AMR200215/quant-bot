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


def _self_heal_missing_journal_rows() -> int:
    """
    P2'(b) — Reconciler self-heal: scan closed positions (last 24h with live|tx: in notes)
    and synthesize missing live-journal rows for any that have no matching entry.

    Returns count of backfilled rows.
    """
    import csv as _csv
    import json as _json
    import time as _time
    from pathlib import Path as _Path

    try:
        from memecoin.config import POSITIONS_FILE, LIVE_JOURNAL_FILE, JOURNAL_FIELDS
    except ImportError:
        log.debug("self_heal: config imports unavailable")
        return 0

    # Load positions file to find recently-closed live positions
    positions_path = _Path(POSITIONS_FILE)
    if not positions_path.exists():
        return 0

    try:
        raw_positions = _json.loads(positions_path.read_text())
    except Exception as e:
        log.warning("self_heal: could not load positions: %s", e)
        return 0

    cutoff = _time.time() - 86400  # 24 hours

    # Load existing live journal row IDs and tx sigs for dedup
    existing_ids: set = set()
    existing_notes_sigs: set = set()
    try:
        if _Path(LIVE_JOURNAL_FILE).exists():
            with open(LIVE_JOURNAL_FILE, newline="") as f:
                for row in _csv.DictReader(f):
                    existing_ids.add(row.get("id", ""))
                    notes = row.get("notes", "") or ""
                    # extract tx sigs from notes
                    import re as _re
                    for m in _re.findall(r'(?:tx:|sell_tx:)([A-Za-z0-9]{32,})', notes):
                        existing_notes_sigs.add(m)
    except Exception as e:
        log.warning("self_heal: could not read live journal: %s", e)

    backfilled = 0
    for pos_data in raw_positions:
        notes = pos_data.get("notes", "") or ""
        status = pos_data.get("status", "")
        exit_time_raw = pos_data.get("exit_time") or 0

        # Only closed live positions within last 24h
        if status != "closed":
            continue
        if "live|tx:" not in notes:
            continue
        if exit_time_raw and float(exit_time_raw) < cutoff:
            continue

        pos_id = pos_data.get("id", "")
        if pos_id in existing_ids:
            continue

        # Check if any tx sig from notes is already in the journal
        import re as _re
        tx_sigs = _re.findall(r'(?:tx:|sell_tx:|jupiter_rescue_pending:)([A-Za-z0-9]{32,})', notes)
        if any(sig in existing_notes_sigs for sig in tx_sigs):
            log.debug("self_heal: pos %s already journaled by tx sig — skip", pos_id)
            continue

        # Synthesize a journal row from position data
        log.warning(
            "self_heal: pos %s (%s) has live|tx: in notes but no journal row — backfilling",
            pos_id, pos_data.get("token_symbol", "?"),
        )
        try:
            # Build synthetic journal row from available position fields
            from memecoin.portfolio import CONFIG_TAG, ACCOUNTING_EPOCH
            _et = pos_data.get("exit_time") or 0
            _ent = pos_data.get("entry_time") or 0
            _st = pos_data.get("signal_time") or 0

            def _fmt_ts(ts):
                if not ts:
                    return ""
                try:
                    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    return ""

            _ep = float(pos_data.get("entry_price") or 0)
            _xp = float(pos_data.get("exit_price") or 0)
            _sz = float(pos_data.get("size_usd") or 0)
            _rf = float(pos_data.get("remaining_fraction") or 1.0)
            _rpnl = float(pos_data.get("realized_pnl_usd") or 0)
            _pnl_pct = (_xp / _ep - 1) * 100 if _ep > 0 and _xp > 0 else 0
            _pnl_usd = _rpnl + _pnl_pct / 100 * _sz * _rf

            row = {f: "" for f in JOURNAL_FIELDS}
            row.update({
                "id": pos_id,
                "signal_id": pos_data.get("signal_id", ""),
                "chain": pos_data.get("chain", "solana"),
                "token_address": pos_data.get("token_address", ""),
                "token_symbol": pos_data.get("token_symbol", ""),
                "signal_type": pos_data.get("signal_type", ""),
                "strength": pos_data.get("strength", ""),
                "signal_price": pos_data.get("signal_price", ""),
                "signal_time": _fmt_ts(_st),
                "entry_price": _ep,
                "entry_time": _fmt_ts(_ent),
                "size_usd": _sz,
                "exit_price": _xp,
                "exit_time": _fmt_ts(_et),
                "exit_reason": pos_data.get("exit_reason", ""),
                "pnl_usd": round(_pnl_usd, 4),
                "pnl_pct": round(_pnl_pct, 2),
                "peak_price": pos_data.get("peak_price", ""),
                "hard_stop_pct": pos_data.get("hard_stop_pct", ""),
                "whale_count": pos_data.get("whale_count", 0),
                "whale_tiers": ",".join(str(t) for t in (pos_data.get("whale_tiers") or [])),
                "safety_score": pos_data.get("safety_score", ""),
                "momentum_score": pos_data.get("momentum_score", ""),
                "composite_score": pos_data.get("composite_score", ""),
                "notes": (notes or "") + "|journal_backfilled=True",
                "config_tag": pos_data.get("config_tag") or CONFIG_TAG,
                "tp_levels_hit": ",".join(pos_data.get("tp_levels_hit") or []),
                "realized_partial_usd": round(_rpnl, 4),
                "remaining_fraction": round(_rf, 4),
                "sol_received": pos_data.get("sol_received") or "",
                "accounting_epoch": pos_data.get("accounting_epoch") or ACCOUNTING_EPOCH,
            })

            from memecoin.journal_io import JOURNAL_LOCK
            from memecoin.portfolio import _ensure_journal_header
            _Path(LIVE_JOURNAL_FILE).parent.mkdir(parents=True, exist_ok=True)
            with JOURNAL_LOCK:
                _ensure_journal_header(_Path(LIVE_JOURNAL_FILE))
                _wh = not _Path(LIVE_JOURNAL_FILE).exists() or _Path(LIVE_JOURNAL_FILE).stat().st_size == 0
                with open(LIVE_JOURNAL_FILE, "a", newline="") as f:
                    writer = _csv.DictWriter(f, fieldnames=JOURNAL_FIELDS, extrasaction="ignore")
                    if _wh:
                        writer.writeheader()
                    writer.writerow(row)
            backfilled += 1
            log.warning("self_heal: backfilled journal row for %s (%s)",
                        pos_id, pos_data.get("token_symbol", "?"))
            try:
                from app.alerts import _send
                _send(
                    f"🔧 JOURNAL BACKFILL: {pos_data.get('token_symbol','?')} (pos={pos_id}) "
                    f"was missing from live journal — row synthesized from positions file. "
                    f"exit_reason={pos_data.get('exit_reason','')} pnl={round(_pnl_usd,4)}"
                )
            except Exception:
                pass
        except Exception as e:
            log.warning("self_heal: failed to backfill pos %s: %s", pos_id, e)

    if backfilled:
        log.warning("self_heal: backfilled %d missing live journal row(s)", backfilled)
    return backfilled


def reconcile(force: bool = False) -> dict:
    """
    Run one reconciliation cycle.

    Returns a result dict with keys: ok, discrepancy_usd, wallet_delta_usd,
    journal_delta_usd, sol_balance, sol_price, message.
    """
    # P2'(b): self-heal missing journal rows before doing SOL reconciliation
    try:
        _self_heal_missing_journal_rows()
    except Exception as _sh_err:
        log.warning("reconcile: self_heal failed: %s", _sh_err)

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
