"""
jupiter_rescue.py — Universal Jupiter rescue sell for graduated/stuck positions.

Fires when PumpSwap local path fails (pumpswap_no_pool / pumpswap_bad_pool_layout /
etc.) and before the graduated_loss write-off.  Uses the jupiter_governor
(Purpose.EXIT / EMERGENCY) so the rescue call is properly rate-limited and does
not collide with the executor's own Jupiter calls.

Hardening features
------------------
* Multi-RPC failover via execution_rpc.ExecutionRpcClient (Part 6)
* TTL-based double-sell prevention — |jupiter_rescue_pending:<sig>:<ts>| (Part 8)
* Same-signed-tx rebroadcast during confirmation window (Part 9)
* Extended CSV logging (Part 12)
* No-route diagnostics CSV (Part 13)

Public API
----------
force_jupiter_rescue_sell(pos, reason, purpose="EXIT") -> dict

Return dict shape
-----------------
{
    "success":                  bool,
    "route":                    "JUPITER_RESCUE",
    "tx_sig":                   str,
    "fill_price":               float | None,
    "sol_received":             float,
    "token_balance_raw":        int,
    "used_pos_tokens_held":     bool,
    "amount_source":            str,   # "pos_tokens_held" | "rpc_balance" | "rpc_balance_failed"
    "balance_rpc_failed":       bool,
    "jupiter_quote_ok":         bool,
    "jupiter_swap_build_ok":    bool,
    "jupiter_send_attempted":   bool,
    "jupiter_confirmed":        bool,
    "jupiter_price_impact_pct": float | None,
    "jupiter_retry_count":      int,
    "jupiter_429_count":        int,
    "rebroadcast_count":        int,
    "panic_price_impact_allowed": bool,
    "rpc_provider_used":        str,
    "pending_tag_age_sec":      float,
    "error_class":              str,
    "reason":                   str,
}
"""

import base64
import csv
import logging
import os
import re as _re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests as _requests

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_WSOL_MINT  = "So11111111111111111111111111111111111111112"
_QUOTE_URL  = "https://lite-api.jup.ag/swap/v1/quote"
_SWAP_URL   = "https://lite-api.jup.ag/swap/v1/swap"

_RESCUE_LOG  = Path(__file__).parent.parent / "logs" / "jupiter_rescue_attempts.csv"
_NOROUTE_LOG = Path(__file__).parent.parent / "logs" / "jupiter_no_route.csv"

_RESCUE_LOG_FIELDS = [
    "timestamp", "position_id", "symbol", "mint", "reason", "rescue_trigger",
    "exit_state", "dex_id", "token_program_if_known",
    "token_balance_raw", "used_pos_tokens_held",
    "amount_source", "balance_rpc_failed",
    "jupiter_rescue_attempted", "jupiter_quote_ok", "jupiter_swap_build_ok",
    "jupiter_send_attempted", "jupiter_confirmed", "jupiter_signature",
    "jupiter_error_class", "jupiter_429_count", "jupiter_retry_count",
    "jupiter_price_impact_pct", "panic_price_impact_allowed",
    "rebroadcast_count", "rpc_provider_used", "pending_tag_age_sec",
    "sol_received", "fill_price", "final_exit_status",
]

_NOROUTE_FIELDS = [
    "timestamp", "mint", "symbol", "pos_id", "amount_raw", "reason",
    "quote_error_field", "quote_error_msg",
]

_URGENT_REASONS = frozenset({
    "hard_stop", "hard_stop_pp",
    "trailing_stop", "trailing_stop_pp",
    "feed_blind",
    "migration_uncertain", "migration_uncertain_no_pool", "migration_uncertain_retry",
    "graduated_exit", "sell_stuck",
    "dev_dump", "rug_lp", "velocity",
    "abort_tripwire", "pre_graduation_exit",
})

_MIN_VALID_SIG_LEN = 80   # Solana base58 sigs are 87-88 chars; < 80 = legacy truncated

# One-time startup warning for missing fallback RPC config
_REBROADCAST_WARN_LOGGED = False


# ── Result classifier ─────────────────────────────────────────────────────────

def classify_rescue_result(rescue_result: dict) -> str:
    """
    Classify a force_jupiter_rescue_sell() return dict into a disposal class.

    Called by portfolio.close_position() and scanner.py after rescue to decide
    whether to call executor.sell, arm retry, or finalize.

    Returns
    -------
    "sold"             — tx confirmed; position should be finalized normally
    "already_sold"     — position was already closed (stale confirmed sig or zero bal)
    "pending"          — tx sent but not yet confirmed; keep pending tag, arm retry
    "no_route"         — no Jupiter route; no tx sent; arm retry, zero SOL burned
    "retry_no_send"    — transient failure before send (429, price impact, etc.); retry safe
    "fatal_no_send"    — structural failure (keypair/sign); no tx; alert operator
    "fallback_allowed" — executor fallback is safe (pre-send, non-rescue-critical state)
    """
    success        = rescue_result.get("success", False)
    reason         = rescue_result.get("reason", "")
    error_class    = rescue_result.get("error_class", "")
    send_attempted = rescue_result.get("jupiter_send_attempted", False)
    tx_sig         = rescue_result.get("tx_sig", "")
    confirmed      = rescue_result.get("jupiter_confirmed", False)

    if success:
        return "sold"

    if error_class == "already_sold" or reason == "rescue_stale_sig_confirmed":
        return "already_sold"

    # Any result where a tx was sent but not confirmed (includes TTL-pending)
    if reason == "rescue_already_pending":
        return "pending"
    if reason == "tx_not_confirmed" or (send_attempted and tx_sig and not confirmed):
        return "pending"

    if error_class == "jupiter_no_route" or reason == "no_route":
        return "no_route"

    _RETRY_NO_SEND_EC = frozenset({
        "jupiter_429_exhausted", "jupiter_quote_error", "jupiter_http_error",
        "price_impact_exceeded",
    })
    _RETRY_NO_SEND_REASONS = frozenset({
        "jupiter_429_exhausted", "quote_failed", "quote_http_error",
        "price_impact_too_high", "zero_balance",
        "token_balance_unavailable", "rescue_disabled", "position_not_open",
    })
    if error_class in _RETRY_NO_SEND_EC or reason in _RETRY_NO_SEND_REASONS:
        return "retry_no_send"

    _FATAL_EC      = frozenset({"keypair_error", "tx_sign_error"})
    _FATAL_REASONS = frozenset({"keypair_load_failed", "sign_failed"})
    if error_class in _FATAL_EC or reason in _FATAL_REASONS:
        return "fatal_no_send"

    # If send was never attempted, executor fallback is safe for non-rescue-critical states
    if not send_attempted:
        return "fallback_allowed"

    # Catch-all: tx was sent but something unexpected happened → treat as pending
    return "pending"


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _log_rescue(row: dict) -> None:
    try:
        _RESCUE_LOG.parent.mkdir(parents=True, exist_ok=True)
        write_header = not _RESCUE_LOG.exists()
        with open(_RESCUE_LOG, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_RESCUE_LOG_FIELDS, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(row)
    except Exception as e:
        log.debug("jupiter_rescue: CSV log failed: %s", e)


def _log_no_route(mint: str, symbol: str, pos_id: str, amount: int,
                  reason: str, quote: dict) -> None:
    try:
        _NOROUTE_LOG.parent.mkdir(parents=True, exist_ok=True)
        write_header = not _NOROUTE_LOG.exists()
        error_field = quote.get("error", "") if isinstance(quote, dict) else str(quote)
        error_msg   = quote.get("errorCode", "") if isinstance(quote, dict) else ""
        with open(_NOROUTE_LOG, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_NOROUTE_FIELDS, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow({
                "timestamp":        datetime.now(timezone.utc).isoformat(),
                "mint":             mint,
                "symbol":           symbol,
                "pos_id":           pos_id,
                "amount_raw":       amount,
                "reason":           reason,
                "quote_error_field": error_field,
                "quote_error_msg":  error_msg,
            })
    except Exception as e:
        log.debug("jupiter_rescue: no-route log failed: %s", e)


# ── RPC helper (via execution_rpc — patchable at module level for tests) ──────

def _rpc_url() -> str:
    try:
        from memecoin.config import CHAINS as _CHAINS
        return _CHAINS.get("solana", {}).get("rpc", "https://api.mainnet-beta.solana.com")
    except Exception:
        return os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")


def _rpc_post(payload: dict, timeout: int = 15) -> dict:
    """Single-entry-point for all RPC calls.  Tests can monkey-patch this."""
    try:
        from memecoin.execution_rpc import rpc_post as _exec_rpc_post
        return _exec_rpc_post(payload, timeout_override_sec=float(timeout))
    except ImportError:
        url  = _rpc_url()
        resp = _requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()


def _rpc_post_to_url(url: str, payload: dict, timeout: float = 5.0) -> None:
    """Fire-and-forget RPC post to a specific URL (used for rebroadcast)."""
    try:
        _requests.post(url, json=payload, timeout=timeout)
    except Exception:
        pass


def _get_rebroadcast_urls() -> list:
    """Return all execution RPC URLs except the first primary (already used for send)."""
    try:
        from memecoin.config import EXECUTION_RPC_URLS, EXECUTION_RPC_FALLBACK_URLS
        all_urls = list(EXECUTION_RPC_URLS) + list(EXECUTION_RPC_FALLBACK_URLS)
        return all_urls[1:] if len(all_urls) > 1 else []
    except ImportError:
        return []


# ── Pending-tag helpers (TTL + sig-check) ────────────────────────────────────

def _parse_pending_tag(notes: str) -> tuple:
    """
    Parse |jupiter_rescue_pending:<val>| from notes.

    Returns (sig, ts) where:
      ("", None)    — no pending tag
      (sig, None)   — old-style tag (no timestamp or non-parseable)
      (sig, int)    — new-style tag with unix timestamp
      ("sending", ts) — mid-send placeholder
    """
    m = _re.search(r'\|jupiter_rescue_pending:([^|]+)', notes)
    if not m:
        return "", None
    val       = m.group(1)
    # rfind lets sig contain ":" characters safely (base58 sigs don't, but be safe)
    colon_idx = val.rfind(":")
    if colon_idx == -1:
        return val, None
    sig_part = val[:colon_idx]
    ts_part  = val[colon_idx + 1:]
    try:
        return sig_part, int(ts_part)
    except ValueError:
        return val, None


def _clear_pending_tag(notes: str) -> str:
    return _re.sub(r'\|jupiter_rescue_pending:[^|]*', '', notes or '')


# ── Balance lookup ─────────────────────────────────────────────────────────────

def _get_token_balance_with_status(wallet: str, mint: str) -> tuple:
    """Return (raw_balance: int, rpc_failed: bool)."""
    try:
        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "getTokenAccountsByOwner",
            "params": [wallet, {"mint": mint}, {"encoding": "jsonParsed"}],
        }
        data    = _rpc_post(payload, timeout=12)
        accounts = data.get("result", {}).get("value", [])
        if not accounts:
            return 0, False
        amt = (
            accounts[0]
            .get("account", {})
            .get("data", {})
            .get("parsed", {})
            .get("info", {})
            .get("tokenAmount", {})
            .get("amount", "0")
        )
        return int(amt), False
    except Exception as exc:
        log.warning("jupiter_rescue: getTokenAccountsByOwner failed: %s", exc)
        return 0, True


def _get_token_balance(wallet: str, mint: str) -> int:
    bal, _ = _get_token_balance_with_status(wallet, mint)
    return bal


# ── Sol received from confirmed tx ─────────────────────────────────────────────

def _sol_received_from_tx(sig: str, wallet: str) -> float:
    try:
        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "getTransaction",
            "params": [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
        }
        data = _rpc_post(payload, timeout=15)
        tx   = data.get("result")
        if not tx:
            return 0.0
        meta          = tx.get("meta", {})
        pre_balances  = meta.get("preBalances", [])
        post_balances = meta.get("postBalances", [])
        account_keys  = tx.get("transaction", {}).get("message", {}).get("accountKeys", [])
        for i, ak in enumerate(account_keys):
            addr = ak if isinstance(ak, str) else ak.get("pubkey", "")
            if addr == wallet:
                pre   = pre_balances[i]  if i < len(pre_balances)  else 0
                post  = post_balances[i] if i < len(post_balances) else 0
                delta = (post - pre) / 1e9
                return delta if delta > 0 else 0.0
        return 0.0
    except Exception as exc:
        log.debug("jupiter_rescue: getTransaction parse failed: %s", exc)
        return 0.0


# ── Note field extractor ──────────────────────────────────────────────────────

def _note_field(notes: str, key: str) -> str:
    m = _re.search(rf'\|{_re.escape(key)}:([^|]*)', notes)
    return m.group(1) if m else ""


# ── Log row builder ───────────────────────────────────────────────────────────

def _build_log_row(pos_id: str, symbol: str, mint: str, trigger_reason: str,
                   result: dict, pos_notes: str = "") -> dict:
    exit_state = _note_field(pos_notes, "exit_state")
    dex_id     = _note_field(pos_notes, "dex_id")
    tp_if_known = (
        "TOKEN_2022" if "token_program:TOKEN_2022" in pos_notes else "SPL"
    )
    return {
        "timestamp":                  datetime.now(timezone.utc).isoformat(),
        "position_id":                pos_id,
        "symbol":                     symbol,
        "mint":                       mint,
        "reason":                     result.get("reason", ""),
        "rescue_trigger":             trigger_reason,
        "exit_state":                 exit_state,
        "dex_id":                     dex_id,
        "token_program_if_known":     tp_if_known,
        "token_balance_raw":          result.get("token_balance_raw", 0),
        "used_pos_tokens_held":       result.get("used_pos_tokens_held", False),
        "amount_source":              result.get("amount_source", ""),
        "balance_rpc_failed":         result.get("balance_rpc_failed", False),
        "jupiter_rescue_attempted":   True,
        "jupiter_quote_ok":           result.get("jupiter_quote_ok", False),
        "jupiter_swap_build_ok":      result.get("jupiter_swap_build_ok", False),
        "jupiter_send_attempted":     result.get("jupiter_send_attempted", False),
        "jupiter_confirmed":          result.get("jupiter_confirmed", False),
        "jupiter_signature":          result.get("tx_sig", ""),
        "jupiter_error_class":        result.get("error_class", ""),
        "jupiter_429_count":          result.get("jupiter_429_count", 0),
        "jupiter_retry_count":        result.get("jupiter_retry_count", 0),
        "jupiter_price_impact_pct":   result.get("jupiter_price_impact_pct"),
        "panic_price_impact_allowed": result.get("panic_price_impact_allowed", False),
        "rebroadcast_count":          result.get("rebroadcast_count", 0),
        "rpc_provider_used":          result.get("rpc_provider_used", ""),
        "pending_tag_age_sec":        result.get("pending_tag_age_sec", 0.0),
        "sol_received":               result.get("sol_received", 0.0),
        "fill_price":                 result.get("fill_price"),
        "final_exit_status": (
            "success" if result.get("success") else result.get("reason", "failed")
        ),
    }


# ── Config startup warning ────────────────────────────────────────────────────

def _warn_rebroadcast_config() -> None:
    """Log once if rebroadcast is enabled but no fallback RPCs are configured."""
    global _REBROADCAST_WARN_LOGGED
    if _REBROADCAST_WARN_LOGGED:
        return
    _REBROADCAST_WARN_LOGGED = True
    try:
        from memecoin.config import JUPITER_RESCUE_REBROADCAST_ENABLED, EXECUTION_RPC_FALLBACK_URLS
        if JUPITER_RESCUE_REBROADCAST_ENABLED and not EXECUTION_RPC_FALLBACK_URLS:
            log.warning(
                "jupiter_rescue: rebroadcast enabled but EXECUTION_RPC_FALLBACK_URLS is empty"
                " — rescue tx will only be sent to the primary RPC"
            )
    except Exception:
        pass


# ── Main rescue function ──────────────────────────────────────────────────────

def force_jupiter_rescue_sell(pos, reason: str, purpose: str = "EXIT") -> dict:  # noqa: C901
    """
    Attempt a Jupiter rescue sell for a position whose PumpSwap / executor path failed.

    Always returns a dict — never raises.  Caller must check result["success"].
    """
    # ── Stub result ───────────────────────────────────────────────────────────
    result: dict = {
        "success":                  False,
        "route":                    "JUPITER_RESCUE",
        "tx_sig":                   "",
        "fill_price":               None,
        "sol_received":             0.0,
        "token_balance_raw":        0,
        "used_pos_tokens_held":     False,
        "amount_source":            "",
        "balance_rpc_failed":       False,
        "jupiter_quote_ok":         False,
        "jupiter_swap_build_ok":    False,
        "jupiter_send_attempted":   False,
        "jupiter_confirmed":        False,
        "jupiter_price_impact_pct": None,
        "jupiter_retry_count":      0,
        "jupiter_429_count":        0,
        "rebroadcast_count":        0,
        "panic_price_impact_allowed": False,
        "rpc_provider_used":        "",
        "pending_tag_age_sec":      0.0,
        "error_class":              "",
        "reason":                   "",
    }

    pos_notes = ""  # captured early for log row; updated inside try block

    _warn_rebroadcast_config()

    try:
        # ── (a) Config guard ──────────────────────────────────────────────────
        try:
            from memecoin.config import (
                JUPITER_RESCUE_ENABLED,
                JUPITER_RESCUE_MAX_PRICE_IMPACT_PCT,
                ALLOW_JUPITER_RESCUE_PANIC_EXIT,
                JUPITER_RESCUE_PENDING_TTL_SEC,
                JUPITER_RESCUE_REBROADCAST_ENABLED,
                JUPITER_RESCUE_REBROADCAST_INTERVAL_MS,
                JUPITER_RESCUE_REBROADCAST_MAX_RPC,
            )
        except ImportError:
            JUPITER_RESCUE_ENABLED               = True
            JUPITER_RESCUE_MAX_PRICE_IMPACT_PCT  = 50
            ALLOW_JUPITER_RESCUE_PANIC_EXIT      = True
            JUPITER_RESCUE_PENDING_TTL_SEC       = 30
            JUPITER_RESCUE_REBROADCAST_ENABLED   = True
            JUPITER_RESCUE_REBROADCAST_INTERVAL_MS = 500
            JUPITER_RESCUE_REBROADCAST_MAX_RPC   = 3

        if not JUPITER_RESCUE_ENABLED:
            result["reason"] = "rescue_disabled"
            return result

        # ── (b) TTL-based double-sell guard ───────────────────────────────────
        pos_notes       = getattr(pos, "notes", "") or ""
        pending_sig, pending_ts = _parse_pending_tag(pos_notes)

        if pending_sig:
            if pending_ts is None:
                # Old-style tag (no timestamp) — conservative: treat as within TTL
                result["reason"] = "rescue_already_pending"
                log.info(
                    "jupiter_rescue: old-style pending tag — skipping  pos=%s",
                    getattr(pos, "id", "?"),
                )
                return result

            tag_age = time.time() - pending_ts
            result["pending_tag_age_sec"] = tag_age

            if tag_age < JUPITER_RESCUE_PENDING_TTL_SEC:
                result["reason"] = "rescue_already_pending"
                log.info(
                    "jupiter_rescue: pending tag fresh (age=%.0fs < TTL=%ds) — skipping  pos=%s",
                    tag_age, JUPITER_RESCUE_PENDING_TTL_SEC, getattr(pos, "id", "?"),
                )
                return result

            # Tag expired — check sig status if we have a full-length valid sig
            _sig_is_full = (
                pending_sig
                and pending_sig != "sending"
                and len(pending_sig) >= _MIN_VALID_SIG_LEN
            )
            _sig_is_truncated = (
                pending_sig
                and pending_sig != "sending"
                and len(pending_sig) < _MIN_VALID_SIG_LEN
            )

            if _sig_is_truncated:
                # Legacy 16-char truncated sig — never call getSignatureStatuses with it.
                # Clear the tag and allow a fresh rescue (balance check inside build path
                # will confirm whether tokens remain before sending any tx).
                log.info(
                    "jupiter_rescue: legacy truncated pending sig (len=%d) cleared"
                    " — allowing fresh rescue  pos=%s",
                    len(pending_sig), getattr(pos, "id", "?"),
                )
                pos.notes = _clear_pending_tag(pos.notes or "")
                pos_notes = pos.notes
                # Fall through to fresh rescue below

            elif _sig_is_full:
                log.info(
                    "jupiter_rescue: pending tag expired (age=%.0fs) — checking sig %s…",
                    tag_age, pending_sig[:16],
                )
                try:
                    chk = _rpc_post({
                        "jsonrpc": "2.0", "id": 1,
                        "method": "getSignatureStatuses",
                        "params": [[pending_sig], {"searchTransactionHistory": True}],
                    }, timeout=10)
                    statuses = chk.get("result", {}).get("value", [None])
                    st       = statuses[0] if statuses else None
                    if (st
                            and st.get("confirmationStatus") in ("confirmed", "finalized")
                            and not st.get("err")):
                        log.warning(
                            "jupiter_rescue: stale sig IS confirmed — position already sold"
                            "  sig=%s  pos=%s",
                            pending_sig[:16], getattr(pos, "id", "?"),
                        )
                        result["success"]           = True
                        result["reason"]            = "rescue_stale_sig_confirmed"
                        result["error_class"]       = "already_sold"
                        result["tx_sig"]            = pending_sig
                        result["jupiter_confirmed"] = True
                        return result
                except Exception as chk_err:
                    log.warning(
                        "jupiter_rescue: stale sig check failed: %s — proceeding with fresh rescue",
                        chk_err,
                    )

                # Stale tag with unconfirmed sig — clear it and allow fresh rescue
                pos.notes = _clear_pending_tag(pos.notes or "")
                pos_notes = pos.notes
                log.info(
                    "jupiter_rescue: stale pending tag cleared — allowing fresh rescue  pos=%s",
                    getattr(pos, "id", "?"),
                )

            else:
                # "sending" placeholder — stale but no sig to check; clear and allow fresh
                pos.notes = _clear_pending_tag(pos.notes or "")
                pos_notes = pos.notes
                log.info(
                    "jupiter_rescue: stale 'sending' placeholder cleared  pos=%s",
                    getattr(pos, "id", "?"),
                )

        # ── (c) Status guard ──────────────────────────────────────────────────
        pos_status = getattr(pos, "status", "open")
        if pos_status not in ("open", "sell_stuck"):
            result["reason"] = "position_not_open"
            return result

        mint   = getattr(pos, "token_address", "")
        symbol = getattr(pos, "token_symbol", "")
        pos_id = getattr(pos, "id", "")

        # ── (d) Load keypair ──────────────────────────────────────────────────
        try:
            from memecoin.executor import _get_keypair
            kp = _get_keypair()
        except Exception:
            try:
                import base58 as _base58
                from solders.keypair import Keypair
                raw = os.environ.get("SOLANA_PRIVATE_KEY", "")
                if not raw:
                    raise RuntimeError("SOLANA_PRIVATE_KEY not set")
                kp = Keypair.from_bytes(_base58.b58decode(raw))
            except Exception as kp_err:
                result["reason"]      = "keypair_load_failed"
                result["error_class"] = "keypair_error"
                log.error("jupiter_rescue: keypair load failed: %s", kp_err)
                _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result, pos_notes))
                return result

        wallet = str(kp.pubkey())

        # ── (e/f) Token balance ───────────────────────────────────────────────
        tokens_held = int(getattr(pos, "tokens_held", 0) or 0)
        if tokens_held > 0:
            balance              = tokens_held
            used_pos_tokens_held = True
            balance_rpc_failed   = False
            amount_source        = "pos_tokens_held"
        else:
            balance, balance_rpc_failed = _get_token_balance_with_status(wallet, mint)
            used_pos_tokens_held        = False
            amount_source = "rpc_balance_failed" if balance_rpc_failed else "rpc_balance"

        result["token_balance_raw"]    = balance
        result["used_pos_tokens_held"] = used_pos_tokens_held
        result["balance_rpc_failed"]   = balance_rpc_failed
        result["amount_source"]        = amount_source

        if balance == 0:
            result["reason"] = "zero_balance"
            log.warning("jupiter_rescue: zero balance  pos=%s  mint=%s", pos_id, mint[:8])
            _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result, pos_notes))
            return result

        # ── (g) Governor quote call ───────────────────────────────────────────
        from memecoin.jupiter_governor import governor, Purpose

        purpose_enum = Purpose.EXIT if purpose == "EXIT" else Purpose.EMERGENCY
        quote_params = {
            "inputMint":        mint,
            "outputMint":       _WSOL_MINT,
            "amount":           str(balance),
            "slippageBps":      "5000",
            "onlyDirectRoutes": "false",
        }

        quote       = None
        _429_count  = 0
        _retry_count = 0

        try:
            resp = governor.request(
                purpose=purpose_enum, endpoint="quote",
                fn=_requests.get, mint=mint,
                url=_QUOTE_URL, params=quote_params, timeout=20,
            )
            _retry_count += 1
            resp.raise_for_status()
            quote = resp.json()
        except _requests.exceptions.HTTPError as http_err:
            if http_err.response is not None and http_err.response.status_code == 429:
                _429_count += 1
                log.warning(
                    "jupiter_rescue: EXIT bucket 429 — escalating to EMERGENCY  mint=%s", mint[:8],
                )
                try:
                    resp2 = governor.request(
                        purpose=Purpose.EMERGENCY, endpoint="quote",
                        fn=_requests.get, mint=mint,
                        url=_QUOTE_URL, params=quote_params, timeout=20,
                    )
                    _retry_count += 1
                    resp2.raise_for_status()
                    quote = resp2.json()
                except _requests.exceptions.HTTPError as http_err2:
                    if http_err2.response is not None and http_err2.response.status_code == 429:
                        _429_count += 1
                    result.update({
                        "jupiter_429_count": _429_count,
                        "jupiter_retry_count": _retry_count,
                        "reason": "jupiter_429_exhausted",
                        "error_class": "jupiter_429_exhausted",
                    })
                    log.error("jupiter_rescue: EMERGENCY bucket also 429  mint=%s", mint[:8])
                    _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result, pos_notes))
                    return result
                except Exception as em_err:
                    result.update({
                        "jupiter_429_count": _429_count,
                        "jupiter_retry_count": _retry_count,
                        "reason": "quote_failed",
                        "error_class": "jupiter_quote_error",
                    })
                    log.error("jupiter_rescue: EMERGENCY quote error: %s  mint=%s", em_err, mint[:8])
                    _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result, pos_notes))
                    return result
            else:
                result.update({
                    "jupiter_429_count": _429_count,
                    "jupiter_retry_count": _retry_count,
                    "reason": "quote_http_error",
                    "error_class": "jupiter_http_error",
                })
                log.error("jupiter_rescue: quote HTTP error: %s  mint=%s", http_err, mint[:8])
                _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result, pos_notes))
                return result
        except Exception as q_err:
            result.update({
                "jupiter_429_count": _429_count,
                "jupiter_retry_count": _retry_count,
                "reason": "quote_failed",
                "error_class": "jupiter_quote_error",
            })
            log.error("jupiter_rescue: quote error: %s  mint=%s", q_err, mint[:8])
            _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result, pos_notes))
            return result

        result["jupiter_retry_count"] = _retry_count
        result["jupiter_429_count"]   = _429_count

        if not quote or "error" in quote:
            result["reason"]      = "no_route"
            result["error_class"] = "jupiter_no_route"
            log.warning("jupiter_rescue: no route  mint=%s  quote=%s", mint[:8], quote)
            _log_no_route(mint, symbol, pos_id, balance, reason, quote or {})
            _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result, pos_notes))
            return result

        result["jupiter_quote_ok"] = True

        # ── (h) Price impact check ────────────────────────────────────────────
        impact_pct = float(quote.get("priceImpactPct", 0)) * 100
        result["jupiter_price_impact_pct"] = impact_pct

        is_urgent = (
            reason.lower() in _URGENT_REASONS
            or "|graduated_unsellable" in pos_notes
        )
        panic_allowed = (
            ALLOW_JUPITER_RESCUE_PANIC_EXIT
            and is_urgent
            and impact_pct > JUPITER_RESCUE_MAX_PRICE_IMPACT_PCT
        )
        result["panic_price_impact_allowed"] = panic_allowed

        if impact_pct > JUPITER_RESCUE_MAX_PRICE_IMPACT_PCT:
            if panic_allowed:
                log.warning(
                    "jupiter_rescue: HIGH impact %.1f%% — panic exit allowed  mint=%s  reason=%s",
                    impact_pct, mint[:8], reason,
                )
            else:
                result["reason"]      = "price_impact_too_high"
                result["error_class"] = "price_impact_exceeded"
                log.warning(
                    "jupiter_rescue: impact %.1f%% > %.0f%% — blocked  mint=%s",
                    impact_pct, JUPITER_RESCUE_MAX_PRICE_IMPACT_PCT, mint[:8],
                )
                _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result, pos_notes))
                return result

        # ── (i) Governor swap build call ──────────────────────────────────────
        swap_body = {
            "quoteResponse":             quote,
            "userPublicKey":             wallet,
            "wrapAndUnwrapSol":          True,
            "dynamicComputeUnitLimit":   True,
            "prioritizationFeeLamports": "auto",
        }

        swap_resp_json = None
        try:
            swap_resp = governor.request(
                purpose=purpose_enum, endpoint="swap",
                fn=_requests.post, mint=mint,
                url=_SWAP_URL, json=swap_body, timeout=20,
            )
            swap_resp.raise_for_status()
            swap_resp_json = swap_resp.json()
        except _requests.exceptions.HTTPError as swap_http_err:
            if swap_http_err.response is not None and swap_http_err.response.status_code == 429:
                result["jupiter_429_count"] += 1
                log.warning("jupiter_rescue: swap build 429 — trying EMERGENCY  mint=%s", mint[:8])
                try:
                    swap_resp2 = governor.request(
                        purpose=Purpose.EMERGENCY, endpoint="swap",
                        fn=_requests.post, mint=mint,
                        url=_SWAP_URL, json=swap_body, timeout=20,
                    )
                    swap_resp2.raise_for_status()
                    swap_resp_json = swap_resp2.json()
                except Exception as sw2_err:
                    if (hasattr(sw2_err, "response") and sw2_err.response is not None
                            and sw2_err.response.status_code == 429):
                        result["jupiter_429_count"] += 1
                    result["reason"]      = "swap_build_failed"
                    result["error_class"] = "jupiter_swap_build_error"
                    log.error("jupiter_rescue: swap EMERGENCY also failed: %s  mint=%s", sw2_err, mint[:8])
                    _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result, pos_notes))
                    return result
            else:
                result["reason"]      = "swap_build_http_error"
                result["error_class"] = "jupiter_swap_build_error"
                log.error("jupiter_rescue: swap build HTTP error: %s  mint=%s", swap_http_err, mint[:8])
                _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result, pos_notes))
                return result
        except Exception as sw_err:
            result["reason"]      = "swap_build_failed"
            result["error_class"] = "jupiter_swap_build_error"
            log.error("jupiter_rescue: swap build error: %s  mint=%s", sw_err, mint[:8])
            _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result, pos_notes))
            return result

        tx_b64 = swap_resp_json.get("swapTransaction", "")
        if not tx_b64:
            result["reason"]      = "swap_build_no_tx"
            result["error_class"] = "jupiter_swap_build_error"
            log.error("jupiter_rescue: swap build returned no swapTransaction  mint=%s", mint[:8])
            _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result, pos_notes))
            return result

        result["jupiter_swap_build_ok"] = True

        # ── (j) Sign transaction ONCE ─────────────────────────────────────────
        try:
            from solders.transaction import VersionedTransaction  # type: ignore
            raw_tx   = base64.b64decode(tx_b64)
            vtx      = VersionedTransaction.from_bytes(raw_tx)
            signed   = VersionedTransaction(vtx.message, [kp])
            tx_bytes = bytes(signed)
        except Exception as sign_err:
            result["reason"]      = "sign_failed"
            result["error_class"] = "tx_sign_error"
            log.error("jupiter_rescue: sign failed: %s  mint=%s", sign_err, mint[:8])
            _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result, pos_notes))
            return result

        tx_b64_signed = base64.b64encode(tx_bytes).decode()
        send_payload  = {
            "jsonrpc": "2.0", "id": 1,
            "method": "sendTransaction",
            "params": [
                tx_b64_signed,
                {"encoding": "base64", "skipPreflight": True, "maxRetries": 3},
            ],
        }

        # ── (k) Tag notes BEFORE send (with timestamp) ────────────────────────
        _rescue_ts  = int(time.time())
        rescue_tag  = f"|jupiter_rescue_pending:sending:{_rescue_ts}"
        pos.notes   = (pos.notes or "") + rescue_tag
        pos_notes   = pos.notes

        # ── (l) Send via RPC ──────────────────────────────────────────────────
        result["jupiter_send_attempted"] = True
        sig = ""
        try:
            send_data = _rpc_post(send_payload, timeout=20)
            sig       = send_data.get("result", "")

            # Track which RPC was used
            try:
                from memecoin.execution_rpc import get_client as _get_rpc_client
                result["rpc_provider_used"] = _get_rpc_client().last_used_url
            except Exception:
                result["rpc_provider_used"] = _rpc_url()

            if not sig:
                err = send_data.get("error", {})
                result["reason"]      = "send_rpc_error"
                result["error_class"] = "rpc_send_error"
                log.error("jupiter_rescue: sendTransaction no sig  err=%s  mint=%s", err, mint[:8])
                pos.notes = _re.sub(
                    r'\|jupiter_rescue_pending:sending:\d+',
                    "|jupiter_rescue_send_failed",
                    pos.notes or "",
                )
                _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result, pos_notes))
                return result
        except Exception as send_err:
            result["reason"]      = "send_failed"
            result["error_class"] = "rpc_send_error"
            log.error("jupiter_rescue: send failed: %s  mint=%s", send_err, mint[:8])
            pos.notes = _re.sub(
                r'\|jupiter_rescue_pending:sending:\d+',
                "|jupiter_rescue_send_failed",
                pos.notes or "",
            )
            _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result, pos_notes))
            return result

        result["tx_sig"] = sig

        # Update pending tag with FULL sig and timestamp (never truncate — needed for
        # getSignatureStatuses confirmation on retry).  Use sig[:16] only in logs.
        pos.notes = _re.sub(
            r'\|jupiter_rescue_pending:sending:\d+',
            f"|jupiter_rescue_pending:{sig}:{_rescue_ts}",
            pos.notes or "",
        )
        pos_notes = pos.notes
        log.info("jupiter_rescue: tx sent  sig=%s…  mint=%s", sig[:16], mint[:8])

        # ── (m) Confirm + rebroadcast ─────────────────────────────────────────
        confirmed      = False
        deadline       = time.time() + 45
        rb_urls        = _get_rebroadcast_urls()[:JUPITER_RESCUE_REBROADCAST_MAX_RPC] \
                         if JUPITER_RESCUE_REBROADCAST_ENABLED else []
        rb_idx         = 0
        rb_count       = 0
        last_rb_time   = time.time()
        rb_interval    = JUPITER_RESCUE_REBROADCAST_INTERVAL_MS / 1000.0

        while time.time() < deadline:
            # Rebroadcast same signed bytes to fallback RPCs
            if (rb_urls and rb_idx < len(rb_urls)
                    and rb_count < JUPITER_RESCUE_REBROADCAST_MAX_RPC
                    and time.time() - last_rb_time >= rb_interval):
                _rpc_post_to_url(rb_urls[rb_idx], send_payload)
                log.debug(
                    "jupiter_rescue: rebroadcast #%d to %s  sig=%s",
                    rb_count + 1, rb_urls[rb_idx], sig[:16],
                )
                rb_count     += 1
                rb_idx       += 1
                last_rb_time  = time.time()

            time.sleep(2)

            try:
                confirm_data = _rpc_post({
                    "jsonrpc": "2.0", "id": 1,
                    "method": "getSignatureStatuses",
                    "params": [[sig], {"searchTransactionHistory": True}],
                }, timeout=10)
                statuses = confirm_data.get("result", {}).get("value", [None])
                st = statuses[0] if statuses else None
                if st is None:
                    continue
                if st.get("err") is not None:
                    result["reason"]      = "tx_reverted"
                    result["error_class"] = "tx_reverted"
                    result["rebroadcast_count"] = rb_count
                    log.warning(
                        "jupiter_rescue: tx reverted  sig=%s  err=%s  mint=%s",
                        sig[:16], st.get("err"), mint[:8],
                    )
                    _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result, pos_notes))
                    return result
                if st.get("confirmationStatus") in ("confirmed", "finalized"):
                    confirmed = True
                    break
            except Exception:
                pass

        result["rebroadcast_count"] = rb_count
        result["jupiter_confirmed"] = confirmed

        if not confirmed:
            result["reason"]      = "tx_not_confirmed"
            result["error_class"] = "tx_timeout"
            log.warning(
                "jupiter_rescue: tx not confirmed within 45s  sig=%s  mint=%s", sig[:16], mint[:8],
            )
            _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result, pos_notes))
            return result

        # ── (n) Sol received ──────────────────────────────────────────────────
        sol_recv = _sol_received_from_tx(sig, wallet)
        result["sol_received"] = sol_recv
        result["fill_price"]   = None   # portfolio caller uses pos.exit_price if None

        # ── (o) Success ───────────────────────────────────────────────────────
        result["success"] = True
        result["reason"]  = "jupiter_rescue_ok"
        log.info(
            "jupiter_rescue: SUCCESS  mint=%s  sig=%s  sol=%.6f  impact=%.1f%%  rb=%d",
            mint[:8], sig[:16], sol_recv, impact_pct, rb_count,
        )

    except Exception as outer_exc:
        result["reason"]      = "rescue_exception"
        result["error_class"] = "unexpected_error"
        log.error("jupiter_rescue: unexpected exception (non-fatal): %s", outer_exc, exc_info=True)

    # ── (p) Always log ────────────────────────────────────────────────────────
    try:
        _log_rescue(_build_log_row(
            getattr(pos, "id", ""),
            getattr(pos, "token_symbol", ""),
            getattr(pos, "token_address", ""),
            reason,
            result,
            pos_notes,
        ))
    except Exception:
        pass

    return result
