"""
journal_reconciler.py — Periodic journal accounting corrector.

Scans LIVE_JOURNAL_FILE + SOCIAL_JOURNAL_FILE for rows closed with
exit_price=0 / pnl_pct<=-99 / fill_estimated / sell_unconf tags that
contain at least one on-chain transaction signature, then reads the
real SOL delta from the chain and corrects the row in place.

Public API
----------
run_reconciler_pass(wallet: str) -> dict
    Synchronous single pass. Returns {rows_checked, rows_corrected, usd_adjustment}.

start_reconciler_thread(wallet: str)
    Starts a daemon thread that runs one pass at startup then loops every 60s.
"""

import csv
import io
import logging
import os
import re
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

# Lazy import — pulled in at first call so that the module can be imported
# without a live memecoin.tx_meta dependency (useful for testing stubs).
# Tests patch this name directly: patch("memecoin.journal_reconciler.read_sol_delta", ...)
try:
    from memecoin.tx_meta import read_sol_delta
except ImportError:
    read_sol_delta = None  # type: ignore[assignment]

# Regex to extract base58 signatures from notes tags.
# Captures everything after the colon until '|' or end of string.
_SIG_RE = re.compile(
    r'(?:sell_tx|sell_unconf|jupiter_rescue_pending|sell_pending):([A-Za-z0-9]+)'
)

# Tags that mean "this row already needs no further work"
_ALREADY_DONE_TAGS = ("journal_reconciled:", "reconciler_checked_no_recovery")


def _get_journal_paths() -> tuple[Path, Path]:
    from memecoin.config import LIVE_JOURNAL_FILE, SOCIAL_JOURNAL_FILE
    return Path(LIVE_JOURNAL_FILE), Path(SOCIAL_JOURNAL_FILE)


def _get_sol_price() -> float:
    """Best-effort SOL/USD price. Returns 0.0 on failure."""
    try:
        from memecoin.executor import _sol_price_usd
        return _sol_price_usd()
    except Exception:
        pass
    try:
        import requests
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price?ids=solana&vs_currencies=usd",
            timeout=5,
        )
        return float(r.json()["solana"]["usd"])
    except Exception:
        return 0.0


def _query_token_balance(wallet: str, mint: str) -> int:
    """Return raw on-chain token balance for (wallet, mint). Returns -1 on error."""
    try:
        from memecoin.executor import _token_balance
        return _token_balance(wallet, mint)
    except Exception:
        pass
    # Fallback: raw RPC call
    try:
        import requests as _req
        rpc_url = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "getTokenAccountsByOwner",
            "params": [
                wallet,
                {"mint": mint},
                {"encoding": "jsonParsed"},
            ],
        }
        resp = _req.post(rpc_url, json=payload, timeout=10)
        data = resp.json()
        accounts = (data.get("result") or {}).get("value") or []
        total = 0
        for acct in accounts:
            amt = (
                acct.get("account", {})
                .get("data", {})
                .get("parsed", {})
                .get("info", {})
                .get("tokenAmount", {})
                .get("amount", "0")
            )
            total += int(amt)
        return total
    except Exception as exc:
        log.debug("_query_token_balance error mint=%s: %s", mint[:8], exc)
        return -1


def _is_target_row(row: dict) -> bool:
    """Return True if this row is a candidate for reconciliation."""
    notes = row.get("notes", "") or ""

    # Skip already-processed rows
    for tag in _ALREADY_DONE_TAGS:
        if tag in notes:
            return False

    # Must have at least one sig tag
    if not _SIG_RE.search(notes):
        return False

    # Must match at least one of the loss/unknown conditions
    exit_price_raw = str(row.get("exit_price", "")).strip()
    pnl_pct_raw = row.get("pnl_pct", "")
    try:
        pnl_pct = float(pnl_pct_raw)
    except (TypeError, ValueError):
        pnl_pct = 0.0

    exit_price_zero = exit_price_raw in ("0", "0.0", "")
    pnl_very_negative = pnl_pct <= -99
    has_fill_estimated = "fill_estimated" in notes
    has_sell_unconf = "sell_unconf:" in notes
    # graduated_recovered sets exit_price in SOL (wrong unit) — reconciler recomputes
    has_grad_recovered = "graduated_recovered:" in notes

    return (exit_price_zero or pnl_very_negative or has_fill_estimated
            or has_sell_unconf or has_grad_recovered)


def _extract_sigs(notes: str) -> list[str]:
    """Extract all sigs from notes, newest-first (notes are chronological so reverse)."""
    sigs = _SIG_RE.findall(notes or "")
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in sigs:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return list(reversed(unique))


def _process_file(path: Path, wallet: str, sol_price: float) -> tuple[int, int, float]:
    """
    Process one journal file. Returns (rows_checked, rows_corrected, usd_adjustment).
    Writes atomically via .reconciler_tmp then os.replace.
    """
    if not path.exists():
        return 0, 0, 0.0

    text = path.read_text()
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return 0, 0, 0.0

    fieldnames = list(rows[0].keys())

    # Work on last 200 rows only for target detection, but rewrite all rows
    candidate_indices = []
    start_idx = max(0, len(rows) - 200)
    for i in range(start_idx, len(rows)):
        if _is_target_row(rows[i]):
            candidate_indices.append(i)

    rows_checked = len(candidate_indices)
    rows_corrected = 0
    usd_adjustment = 0.0

    _rsd = read_sol_delta  # module-level; tests patch this name

    for idx in candidate_indices:
        row = rows[idx]
        notes = row.get("notes", "") or ""
        sigs = _extract_sigs(notes)

        confirmed_sig = None
        confirmed_delta = None

        for sig in sigs:
            result = _rsd(sig, wallet)
            if result.get("ok") and result.get("sol_delta") is not None:
                delta = result["sol_delta"]
                if delta > 0:
                    confirmed_sig = sig
                    confirmed_delta = delta
                    break

        if confirmed_sig is not None and confirmed_delta is not None:
            # We have a confirmed positive SOL delta — correct the row
            sol_received = confirmed_delta

            # Entry cost
            try:
                entry_cost_usd = float(row.get("size_usd") or 0)
            except (TypeError, ValueError):
                entry_cost_usd = 0.0

            # Compute pnl
            if sol_price > 0 and entry_cost_usd > 0:
                pnl_usd_new = sol_received * sol_price - entry_cost_usd
                pnl_pct_new = (pnl_usd_new / entry_cost_usd) * 100
            else:
                pnl_usd_new = None
                pnl_pct_new = None

            # Compute exit_price if possible
            tokens_held = 0
            try:
                tokens_held = int(float(row.get("tokens_held") or 0))
            except (TypeError, ValueError):
                tokens_held = 0

            # Also try to find tokens_held from notes or pos
            if tokens_held == 0:
                # Try to extract from notes tag tokens_held:NNN
                m = re.search(r'tokens_held:(\d+)', notes)
                if m:
                    tokens_held = int(m.group(1))

            exit_price_new = None
            if tokens_held > 0 and sol_price > 0:
                # sol_received SOL / (tokens_held in human units) * sol_price
                # tokens are stored as raw lamports (1e6 decimals for pump.fun)
                tokens_human = tokens_held / 1e6
                if tokens_human > 0:
                    exit_price_new = (sol_received / tokens_human) * sol_price

            # Capture old pnl BEFORE overwriting
            try:
                old_pnl_usd = float(row.get("pnl_usd") or 0)
            except (TypeError, ValueError):
                old_pnl_usd = 0.0

            # Update fields
            if exit_price_new is not None:
                row["exit_price"] = exit_price_new
            if pnl_usd_new is not None:
                row["pnl_usd"] = round(pnl_usd_new, 4)
            if pnl_pct_new is not None:
                row["pnl_pct"] = round(pnl_pct_new, 2)

            # Append reconciled tag to notes
            row["notes"] = notes + f"|journal_reconciled:{confirmed_sig[:8]}"

            rows[idx] = row
            rows_corrected += 1
            if pnl_usd_new is not None:
                usd_adjustment += pnl_usd_new - old_pnl_usd

            log.info(
                "journal_reconciler: corrected  id=%s  sig=%s  sol=%.6f  "
                "pnl_usd=%s  pnl_pct=%s",
                row.get("id", "?"),
                confirmed_sig[:16],
                sol_received,
                round(pnl_usd_new, 4) if pnl_usd_new is not None else "n/a",
                round(pnl_pct_new, 2) if pnl_pct_new is not None else "n/a",
            )

        else:
            # No confirmed sig — check on-chain balance
            mint = row.get("token_address", "")
            if mint and wallet:
                balance = _query_token_balance(wallet, mint)
                if balance == 0:
                    # Tokens gone but no sig — mark as checked
                    tag = "|reconciler_checked_no_recovery"
                    if tag not in notes:
                        row["notes"] = notes + tag
                        rows[idx] = row
                        log.info(
                            "journal_reconciler: no_recovery  id=%s  mint=%s  "
                            "(zero balance, no confirmed sig)",
                            row.get("id", "?"), mint[:8],
                        )
                # If balance > 0 or error (-1): leave row alone this pass

    # Only write if something changed
    changed = rows_corrected > 0 or any(
        "|reconciler_checked_no_recovery" in (rows[i].get("notes", "") or "")
        and "|reconciler_checked_no_recovery" not in (
            list(csv.DictReader(io.StringIO(text)))[i].get("notes", "") or ""
        )
        for i in candidate_indices
    )

    # Simpler approach: always write if we touched any candidate
    if candidate_indices:
        # Check if any row was actually modified by comparing notes
        original_rows = list(csv.DictReader(io.StringIO(text)))
        actually_changed = False
        for i in candidate_indices:
            if rows[i].get("notes") != original_rows[i].get("notes"):
                actually_changed = True
                break
            for f in ("exit_price", "pnl_usd", "pnl_pct"):
                if str(rows[i].get(f)) != str(original_rows[i].get(f)):
                    actually_changed = True
                    break
            if actually_changed:
                break

        if actually_changed:
            # Atomic write
            out = io.StringIO()
            writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            tmp = path.with_suffix(path.suffix + ".reconciler_tmp")
            tmp.write_text(out.getvalue())
            os.replace(str(tmp), str(path))

    return rows_checked, rows_corrected, usd_adjustment


def run_reconciler_pass(wallet: str, _live_path: Path = None, _social_path: Path = None) -> dict:
    """
    Single synchronous reconciler pass over LIVE_JOURNAL_FILE + SOCIAL_JOURNAL_FILE.

    Parameters
    ----------
    wallet : str
        Wallet pubkey to check SOL deltas against.
    _live_path, _social_path : Path | None
        Override journal file paths (used in tests).

    Returns
    -------
    dict with keys: rows_checked, rows_corrected, usd_adjustment
    """
    if not wallet:
        log.debug("journal_reconciler: no wallet configured — skipping pass")
        return {"rows_checked": 0, "rows_corrected": 0, "usd_adjustment": 0.0}

    if _live_path is None or _social_path is None:
        live_path, social_path = _get_journal_paths()
    else:
        live_path, social_path = _live_path, _social_path

    sol_price = _get_sol_price()
    if sol_price == 0.0:
        log.warning("journal_reconciler: could not fetch SOL price — pnl recompute skipped")

    total_checked = 0
    total_corrected = 0
    total_usd = 0.0

    for path in (live_path, social_path):
        chk, cor, usd = _process_file(path, wallet, sol_price)
        total_checked += chk
        total_corrected += cor
        total_usd += usd

    log.info(
        "journal_reconciler: pass done  checked=%d  corrected=%d  usd_adj=%.4f",
        total_checked, total_corrected, total_usd,
    )
    return {
        "rows_checked": total_checked,
        "rows_corrected": total_corrected,
        "usd_adjustment": round(total_usd, 4),
    }


def start_reconciler_thread(wallet: str) -> None:
    """
    Start a daemon thread that runs run_reconciler_pass(wallet) every 60s.
    Runs once immediately at startup before entering the loop.
    """
    def _loop():
        log.info("journal_reconciler thread started  wallet=%s", wallet[:8] if wallet else "none")
        # Startup pass
        try:
            run_reconciler_pass(wallet)
        except Exception as exc:
            log.warning("journal_reconciler startup pass error: %s", exc)
        while True:
            time.sleep(60)
            try:
                run_reconciler_pass(wallet)
            except Exception as exc:
                log.warning("journal_reconciler pass error: %s", exc)

    t = threading.Thread(target=_loop, name="journal_reconciler", daemon=True)
    t.start()
