"""
journal_reconciler.py — Periodic journal accounting corrector.

Scans LIVE_JOURNAL_FILE + SOCIAL_JOURNAL_FILE + JOURNAL_FILE for rows
closed with exit_price=0 / pnl_pct<=-99 / fill_estimated / sell_unconf
tags that contain at least one on-chain transaction signature, then reads
the real SOL delta from the chain and corrects the row in place.

Two-phase rewrite to prevent lost-update race with _append_journal
-------------------------------------------------------------------
Phase 1 (no lock):
  - snapshot-read the journal file
  - identify candidate rows
  - run all read_sol_delta + token-balance RPC calls
  - build corrections dict: {stable_row_key -> correction_delta}

Phase 2 (under JOURNAL_LOCK):
  - re-read the file fresh
  - apply correction_delta onto matching rows by stable key
  - rows appended after Phase 1 are preserved (not in corrections dict)
  - write tmp → os.replace → release lock

Stable row key: id + first sell-tx sig  (fallback: id + mint + exit_time)
Never match on symbol alone.

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


def _get_journal_paths() -> tuple[Path, Path, Path]:
    from memecoin.config import LIVE_JOURNAL_FILE, SOCIAL_JOURNAL_FILE, JOURNAL_FILE
    return Path(LIVE_JOURNAL_FILE), Path(SOCIAL_JOURNAL_FILE), Path(JOURNAL_FILE)


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
    # graduated_recovered used to store raw SOL in exit_price (wrong unit) — reconciler recomputes
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


def _stable_key(row: dict) -> str:
    """
    Stable identity key for a journal row, used to match Phase-1 candidates
    against the fresh re-read in Phase 2.

    Preferred:  id + first sell-tx sig
    Fallback:   id + token_address + exit_time
    """
    row_id = row.get("id", "") or ""
    notes = row.get("notes", "") or ""
    sigs = _SIG_RE.findall(notes)
    if sigs:
        return f"{row_id}|{sigs[0]}"
    return f"{row_id}|{row.get('token_address', '')}|{row.get('exit_time', '')}"


def _compute_correction(
    row: dict,
    wallet: str,
    sol_price: float,
    already_corrected_sigs: set[str],
) -> dict | None:
    """
    Run RPC calls for one candidate row (Phase 1 — no lock held).

    Returns a correction dict:
      {
        "exit_price":     float | None,
        "pnl_usd":        float | None,
        "pnl_pct":        float | None,
        "sol_received":   float | None,
        "notes_suffix":   str,          # tag to append to fresh notes in Phase 2
        "usd_delta":      float,        # for stats: new_pnl_usd - old_pnl_usd
        "confirmed_sig":  str | None,   # sig that confirmed
      }

    Returns None if no correction is needed (no confirmed sig, balance unknown/positive).
    """
    notes = row.get("notes", "") or ""
    sigs = _extract_sigs(notes)

    _rsd = read_sol_delta  # module-level; tests patch this name

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

        # Compute exit_price (USD/token)
        tokens_held = 0
        try:
            tokens_held = int(float(row.get("tokens_held") or 0))
        except (TypeError, ValueError):
            tokens_held = 0

        if tokens_held == 0:
            m = re.search(r'tokens_held:(\d+)', notes)
            if m:
                tokens_held = int(m.group(1))

        remaining_fraction = 1.0
        try:
            remaining_fraction = float(row.get("remaining_fraction") or 1.0)
        except (TypeError, ValueError):
            remaining_fraction = 1.0

        exit_price_new = None
        if tokens_held > 0 and sol_price > 0:
            from memecoin.tx_meta import compute_fill_price
            _raw_tokens = int(tokens_held * remaining_fraction)
            if _raw_tokens > 0:
                exit_price_new = compute_fill_price(sol_received, _raw_tokens, sol_price)

        # Old pnl for delta stats (only count delta once per sig across files)
        usd_delta = 0.0
        if pnl_usd_new is not None and confirmed_sig not in already_corrected_sigs:
            try:
                old_pnl_usd = float(row.get("pnl_usd") or 0)
            except (TypeError, ValueError):
                old_pnl_usd = 0.0
            usd_delta = pnl_usd_new - old_pnl_usd

        log.info(
            "journal_reconciler: will_correct  id=%s  sig=%s  sol=%.6f  "
            "pnl_usd=%s  pnl_pct=%s",
            row.get("id", "?"),
            confirmed_sig[:16],
            sol_received,
            round(pnl_usd_new, 4) if pnl_usd_new is not None else "n/a",
            round(pnl_pct_new, 2) if pnl_pct_new is not None else "n/a",
        )

        # ── Telemetry: journal corrected ──
        try:
            from memecoin import telemetry as _tel
            _pos_id = row.get("id", "")
            _rt = _tel.get_trace_id_for_pos(_pos_id)
            if _rt:
                _old_pnl = 0.0
                try:
                    _old_pnl = float(row.get("pnl_usd") or 0)
                except (TypeError, ValueError):
                    pass
                _tel.event(_rt, "journal_corrected",
                    row_id=_pos_id,
                    old_pnl=round(_old_pnl, 4),
                    new_pnl=round(pnl_usd_new, 4) if pnl_usd_new is not None else None,
                    sol_delta=round(sol_received, 8),
                )
        except Exception:
            pass

        return {
            "exit_price":    exit_price_new,
            "pnl_usd":       round(pnl_usd_new, 4) if pnl_usd_new is not None else None,
            "pnl_pct":       round(pnl_pct_new, 2) if pnl_pct_new is not None else None,
            "sol_received":  round(sol_received, 8),
            "notes_suffix":  f"|journal_reconciled:{confirmed_sig[:8]}",
            "usd_delta":     usd_delta,
            "confirmed_sig": confirmed_sig,
        }

    # No confirmed sig — check on-chain balance
    mint = row.get("token_address", "")
    if mint and wallet:
        balance = _query_token_balance(wallet, mint)
        if balance == 0:
            log.info(
                "journal_reconciler: no_recovery  id=%s  mint=%s  (zero balance, no confirmed sig)",
                row.get("id", "?"), mint[:8],
            )
            return {
                "exit_price":    None,
                "pnl_usd":       None,
                "pnl_pct":       None,
                "sol_received":  None,
                "notes_suffix":  "|reconciler_checked_no_recovery",
                "usd_delta":     0.0,
                "confirmed_sig": None,
            }
        # balance > 0 or unknown (-1): leave row for next pass
        return None

    return None


def _process_file(
    path: Path,
    wallet: str,
    sol_price: float,
    already_corrected_sigs: set[str],
) -> tuple[int, int, float]:
    """
    Process one journal file. Returns (rows_checked, rows_corrected, usd_adjustment).

    Two-phase:
      Phase 1 — no lock: snapshot-read, identify candidates, run all RPCs.
      Phase 2 — under JOURNAL_LOCK: re-read fresh, apply corrections, atomic write.

    Rows appended by portfolio._append_journal() between Phase 1 and Phase 2
    are preserved because Phase 2 re-reads the file fresh before writing.
    """
    if not path.exists():
        return 0, 0, 0.0

    # ── Phase 1: snapshot-read + all RPC calls (no lock) ─────────────────────
    text = path.read_text()
    snapshot_rows = list(csv.DictReader(io.StringIO(text)))
    if not snapshot_rows:
        return 0, 0, 0.0

    start_idx = max(0, len(snapshot_rows) - 200)
    candidate_indices = [
        i for i in range(start_idx, len(snapshot_rows))
        if _is_target_row(snapshot_rows[i])
    ]

    rows_checked = len(candidate_indices)
    if rows_checked == 0:
        return 0, 0, 0.0

    # Build corrections dict: stable_key -> correction (may be None = skip)
    corrections: dict[str, dict] = {}
    for idx in candidate_indices:
        row = snapshot_rows[idx]
        key = _stable_key(row)
        corr = _compute_correction(row, wallet, sol_price, already_corrected_sigs)
        if corr is not None:
            corrections[key] = corr

    if not corrections:
        return rows_checked, 0, 0.0

    # ── Phase 2: under JOURNAL_LOCK — re-read fresh, apply, write ────────────
    from memecoin.journal_io import JOURNAL_LOCK

    rows_corrected = 0
    usd_adjustment = 0.0

    with JOURNAL_LOCK:
        fresh_text = path.read_text()
        fresh_rows = list(csv.DictReader(io.StringIO(fresh_text)))
        if not fresh_rows:
            return rows_checked, 0, 0.0

        fieldnames = list(fresh_rows[0].keys())

        # Add sol_received column if not present (header migration)
        if "sol_received" not in fieldnames:
            fieldnames.append("sol_received")

        actually_changed = False
        for i, row in enumerate(fresh_rows):
            key = _stable_key(row)
            corr = corrections.get(key)
            if corr is None:
                continue

            notes_now = row.get("notes", "") or ""

            # Skip if row was already reconciled between Phase 1 and Phase 2
            if any(tag in notes_now for tag in _ALREADY_DONE_TAGS):
                continue

            if corr.get("exit_price") is not None:
                row["exit_price"] = corr["exit_price"]
            if corr.get("pnl_usd") is not None:
                row["pnl_usd"] = corr["pnl_usd"]
            if corr.get("pnl_pct") is not None:
                row["pnl_pct"] = corr["pnl_pct"]
            if corr.get("sol_received") is not None:
                row["sol_received"] = corr["sol_received"]

            suffix = corr.get("notes_suffix", "")
            if suffix and suffix not in notes_now:
                row["notes"] = notes_now + suffix

            fresh_rows[i] = row
            rows_corrected += 1
            usd_adjustment += corr.get("usd_delta", 0.0)

            # Track confirmed sig so sibling files don't double-count USD delta
            if corr.get("confirmed_sig"):
                already_corrected_sigs.add(corr["confirmed_sig"])

            actually_changed = True
            log.info(
                "journal_reconciler: corrected  id=%s  sig=%s  sol=%s  "
                "pnl_usd=%s  pnl_pct=%s",
                row.get("id", "?"),
                (corr.get("confirmed_sig") or "")[:16],
                corr.get("sol_received", "n/a"),
                corr.get("pnl_usd", "n/a"),
                corr.get("pnl_pct", "n/a"),
            )

        if actually_changed:
            out = io.StringIO()
            writer = csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(fresh_rows)
            tmp = path.with_suffix(path.suffix + ".reconciler_tmp")
            tmp.write_text(out.getvalue())
            os.replace(str(tmp), str(path))

    return rows_checked, rows_corrected, usd_adjustment


def run_reconciler_pass(
    wallet: str,
    _live_path: Path = None,
    _social_path: Path = None,
    _main_path: Path = None,
) -> dict:
    """
    Single synchronous reconciler pass over:
      1. LIVE_JOURNAL_FILE
      2. SOCIAL_JOURNAL_FILE
      3. JOURNAL_FILE  (main memecoin journal)

    Parameters
    ----------
    wallet : str
        Wallet pubkey to check SOL deltas against.
    _live_path, _social_path, _main_path : Path | None
        Override journal file paths (used in tests).

    Returns
    -------
    dict with keys: rows_checked, rows_corrected, usd_adjustment
    """
    if not wallet:
        log.debug("journal_reconciler: no wallet configured — skipping pass")
        return {"rows_checked": 0, "rows_corrected": 0, "usd_adjustment": 0.0}

    if _live_path is None or _social_path is None or _main_path is None:
        live_path, social_path, main_path = _get_journal_paths()
    else:
        live_path, social_path, main_path = _live_path, _social_path, _main_path

    sol_price = _get_sol_price()
    if sol_price == 0.0:
        log.warning("journal_reconciler: could not fetch SOL price — pnl recompute skipped")

    total_checked = 0
    total_corrected = 0
    total_usd = 0.0

    # Shared set: confirmed sigs seen across files so USD delta is not double-counted
    # when the same live trade appears in both live and social journals.
    already_corrected_sigs: set[str] = set()

    for path in (live_path, social_path, main_path):
        chk, cor, usd = _process_file(path, wallet, sol_price, already_corrected_sigs)
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
