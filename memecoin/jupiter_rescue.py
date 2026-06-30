"""
jupiter_rescue.py — Universal Jupiter rescue sell for graduated/stuck positions.

Fires when PumpSwap local path fails (pumpswap_no_pool / pumpswap_bad_pool_layout)
and before the graduated_loss write-off.  Uses the jupiter_governor (Purpose.EXIT /
EMERGENCY) so the rescue call is properly rate-limited and doesn't collide with the
executor's own Jupiter calls.

Public API
----------
force_jupiter_rescue_sell(pos, reason, purpose="EXIT") -> dict

Return dict shape
-----------------
{
    "success": bool,
    "route": "JUPITER_RESCUE",
    "tx_sig": str,
    "fill_price": float or None,
    "sol_received": float,
    "token_balance_raw": int,
    "used_pos_tokens_held": bool,
    "jupiter_quote_ok": bool,
    "jupiter_swap_build_ok": bool,
    "jupiter_send_attempted": bool,
    "jupiter_confirmed": bool,
    "jupiter_price_impact_pct": float or None,
    "jupiter_retry_count": int,
    "jupiter_429_count": int,
    "error_class": str,
    "reason": str,
}
"""

import base64
import csv
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests as _requests

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_WSOL_MINT   = "So11111111111111111111111111111111111111112"
_QUOTE_URL   = "https://lite-api.jup.ag/swap/v1/quote"
_SWAP_URL    = "https://lite-api.jup.ag/swap/v1/swap"
_RESCUE_LOG  = Path(__file__).parent.parent / "logs" / "jupiter_rescue_attempts.csv"

_RESCUE_LOG_FIELDS = [
    "timestamp", "position_id", "symbol", "mint", "reason", "rescue_trigger",
    "token_program_if_known", "token_balance_raw", "used_pos_tokens_held",
    "jupiter_rescue_attempted", "jupiter_quote_ok", "jupiter_swap_build_ok",
    "jupiter_send_attempted", "jupiter_confirmed", "jupiter_signature",
    "jupiter_error_class", "jupiter_429_count", "jupiter_retry_count",
    "jupiter_price_impact_pct", "sol_received", "fill_price", "final_exit_status",
]

_URGENT_REASONS = frozenset({
    "hard_stop", "trailing_stop", "feed_blind", "migration_uncertain",
    "graduated_exit", "sell_stuck", "dev_dump", "rug_lp", "velocity",
    "abort_tripwire",
})

# ── CSV log ───────────────────────────────────────────────────────────────────

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


# ── RPC helper (direct — not a Jupiter call) ──────────────────────────────────

def _rpc_url() -> str:
    try:
        from memecoin.config import CHAINS as _CHAINS
        return _CHAINS.get("solana", {}).get("rpc", "https://api.mainnet-beta.solana.com")
    except Exception:
        return os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")


def _rpc_post(payload: dict, timeout: int = 15) -> dict:
    url = _rpc_url()
    resp = _requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ── Balance lookup ────────────────────────────────────────────────────────────

def _get_token_balance(wallet: str, mint: str) -> int:
    """Return raw token balance (u64) via getTokenAccountsByOwner.  Returns 0 on error."""
    try:
        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "getTokenAccountsByOwner",
            "params": [
                wallet,
                {"mint": mint},
                {"encoding": "jsonParsed"},
            ],
        }
        data = _rpc_post(payload, timeout=12)
        accounts = data.get("result", {}).get("value", [])
        if not accounts:
            return 0
        amt = (
            accounts[0]
            .get("account", {})
            .get("data", {})
            .get("parsed", {})
            .get("info", {})
            .get("tokenAmount", {})
            .get("amount", "0")
        )
        return int(amt)
    except Exception as exc:
        log.warning("jupiter_rescue: getTokenAccountsByOwner failed: %s", exc)
        return 0


# ── Sol received from tx ──────────────────────────────────────────────────────

def _sol_received_from_tx(sig: str, wallet: str) -> float:
    """Parse SOL balance delta for wallet from a confirmed transaction."""
    try:
        payload = {
            "jsonrpc": "2.0", "id": 1,
            "method": "getTransaction",
            "params": [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
        }
        data = _rpc_post(payload, timeout=15)
        tx = data.get("result")
        if not tx:
            return 0.0
        meta         = tx.get("meta", {})
        pre_balances = meta.get("preBalances", [])
        post_balances= meta.get("postBalances", [])
        account_keys = tx.get("transaction", {}).get("message", {}).get("accountKeys", [])
        for i, ak in enumerate(account_keys):
            addr = ak if isinstance(ak, str) else ak.get("pubkey", "")
            if addr == wallet:
                pre  = pre_balances[i]  if i < len(pre_balances)  else 0
                post = post_balances[i] if i < len(post_balances) else 0
                delta = (post - pre) / 1e9
                return delta if delta > 0 else 0.0
        return 0.0
    except Exception as exc:
        log.debug("jupiter_rescue: getTransaction parse failed: %s", exc)
        return 0.0


# ── Main rescue function ──────────────────────────────────────────────────────

def force_jupiter_rescue_sell(pos, reason: str, purpose: str = "EXIT") -> dict:  # noqa: C901
    """
    Attempt a Jupiter rescue sell for a position whose PumpSwap / executor path failed.

    Always returns a dict — never raises.  Caller must check result["success"].
    """
    # ── Stub result ───────────────────────────────────────────────────────────
    result: dict = {
        "success":                False,
        "route":                  "JUPITER_RESCUE",
        "tx_sig":                 "",
        "fill_price":             None,
        "sol_received":           0.0,
        "token_balance_raw":      0,
        "used_pos_tokens_held":   False,
        "jupiter_quote_ok":       False,
        "jupiter_swap_build_ok":  False,
        "jupiter_send_attempted": False,
        "jupiter_confirmed":      False,
        "jupiter_price_impact_pct": None,
        "jupiter_retry_count":    0,
        "jupiter_429_count":      0,
        "error_class":            "",
        "reason":                 "",
    }

    try:
        # ── (a) Config guard ──────────────────────────────────────────────────
        try:
            from memecoin.config import (
                JUPITER_RESCUE_ENABLED,
                JUPITER_RESCUE_MAX_PRICE_IMPACT_PCT,
                ALLOW_JUPITER_RESCUE_PANIC_EXIT,
            )
        except ImportError:
            JUPITER_RESCUE_ENABLED              = True
            JUPITER_RESCUE_MAX_PRICE_IMPACT_PCT = 50
            ALLOW_JUPITER_RESCUE_PANIC_EXIT     = True

        if not JUPITER_RESCUE_ENABLED:
            result["reason"] = "rescue_disabled"
            return result

        # ── (b) Double-sell guard ─────────────────────────────────────────────
        pos_notes = getattr(pos, "notes", "") or ""
        if "|jupiter_rescue_pending:" in pos_notes:
            result["reason"] = "rescue_already_pending"
            log.info("jupiter_rescue: skipping — rescue already pending  pos=%s", getattr(pos, "id", "?"))
            return result

        # ── (c) Status guard ──────────────────────────────────────────────────
        pos_status = getattr(pos, "status", "open")
        if pos_status not in ("open", "sell_stuck"):
            result["reason"] = "position_not_open"
            return result

        mint       = getattr(pos, "token_address", "")
        symbol     = getattr(pos, "token_symbol", "")
        pos_id     = getattr(pos, "id", "")

        # ── (d) Load keypair ──────────────────────────────────────────────────
        try:
            from memecoin.executor import _get_keypair
            kp = _get_keypair()
        except Exception:
            # Fallback: reconstruct directly from env
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
                _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result))
                return result

        wallet = str(kp.pubkey())

        # ── (e/f) Token balance ───────────────────────────────────────────────
        tokens_held = int(getattr(pos, "tokens_held", 0) or 0)
        used_pos_tokens_held = False
        if tokens_held > 0:
            balance = tokens_held
            used_pos_tokens_held = True
        else:
            balance = _get_token_balance(wallet, mint)

        result["token_balance_raw"]    = balance
        result["used_pos_tokens_held"] = used_pos_tokens_held

        if balance == 0:
            result["reason"] = "zero_balance"
            log.warning("jupiter_rescue: zero balance  pos=%s  mint=%s", pos_id, mint[:8])
            _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result))
            return result

        # ── (g) Governor quote call ───────────────────────────────────────────
        from memecoin.jupiter_governor import governor, Purpose

        purpose_enum = Purpose.EXIT if purpose == "EXIT" else Purpose.EMERGENCY
        quote_params = {
            "inputMint":        mint,
            "outputMint":       _WSOL_MINT,
            "amount":           str(balance),
            "slippageBps":      "5000",   # 50% slippage tolerance for rescue
            "onlyDirectRoutes": "false",
        }

        quote = None
        _429_count = 0
        _retry_count = 0

        try:
            resp = governor.request(
                purpose=purpose_enum,
                endpoint="quote",
                fn=_requests.get,
                mint=mint,
                url=_QUOTE_URL,
                params=quote_params,
                timeout=20,
            )
            _retry_count += 1
            resp.raise_for_status()
            quote = resp.json()
        except _requests.exceptions.HTTPError as http_err:
            if http_err.response is not None and http_err.response.status_code == 429:
                _429_count += 1
                log.warning("jupiter_rescue: EXIT bucket 429-exhausted, escalating to EMERGENCY  mint=%s", mint[:8])
                try:
                    resp2 = governor.request(
                        purpose=Purpose.EMERGENCY,
                        endpoint="quote",
                        fn=_requests.get,
                        mint=mint,
                        url=_QUOTE_URL,
                        params=quote_params,
                        timeout=20,
                    )
                    _retry_count += 1
                    resp2.raise_for_status()
                    quote = resp2.json()
                except _requests.exceptions.HTTPError as http_err2:
                    if http_err2.response is not None and http_err2.response.status_code == 429:
                        _429_count += 1
                    result["jupiter_429_count"]   = _429_count
                    result["jupiter_retry_count"] = _retry_count
                    result["reason"]      = "jupiter_429_exhausted"
                    result["error_class"] = "jupiter_429_exhausted"
                    log.error("jupiter_rescue: EMERGENCY bucket also 429-exhausted  mint=%s", mint[:8])
                    _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result))
                    return result
                except Exception as em_err:
                    result["jupiter_429_count"]   = _429_count
                    result["jupiter_retry_count"] = _retry_count
                    result["reason"]      = "quote_failed"
                    result["error_class"] = "jupiter_quote_error"
                    log.error("jupiter_rescue: EMERGENCY quote error: %s  mint=%s", em_err, mint[:8])
                    _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result))
                    return result
            else:
                result["jupiter_429_count"]   = _429_count
                result["jupiter_retry_count"] = _retry_count
                result["reason"]      = "quote_http_error"
                result["error_class"] = "jupiter_http_error"
                log.error("jupiter_rescue: quote HTTP error: %s  mint=%s", http_err, mint[:8])
                _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result))
                return result
        except Exception as q_err:
            result["jupiter_429_count"]   = _429_count
            result["jupiter_retry_count"] = _retry_count
            result["reason"]      = "quote_failed"
            result["error_class"] = "jupiter_quote_error"
            log.error("jupiter_rescue: quote error: %s  mint=%s", q_err, mint[:8])
            _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result))
            return result

        result["jupiter_retry_count"] = _retry_count
        result["jupiter_429_count"]   = _429_count

        if not quote or "error" in quote:
            result["reason"]      = "no_route"
            result["error_class"] = "jupiter_no_route"
            log.warning("jupiter_rescue: no route  mint=%s  quote=%s", mint[:8], quote)
            _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result))
            return result

        result["jupiter_quote_ok"] = True

        # ── (h) Price impact check ────────────────────────────────────────────
        impact_pct = float(quote.get("priceImpactPct", 0)) * 100
        result["jupiter_price_impact_pct"] = impact_pct

        is_urgent = (
            reason.lower() in _URGENT_REASONS
            or "|graduated_unsellable" in pos_notes
        )

        if impact_pct > JUPITER_RESCUE_MAX_PRICE_IMPACT_PCT:
            if ALLOW_JUPITER_RESCUE_PANIC_EXIT and is_urgent:
                log.warning(
                    "jupiter_rescue: HIGH price impact %.1f%% — panic exit allowed  mint=%s  reason=%s",
                    impact_pct, mint[:8], reason,
                )
            else:
                result["reason"]      = "price_impact_too_high"
                result["error_class"] = "price_impact_exceeded"
                log.warning(
                    "jupiter_rescue: price impact %.1f%% > %.0f%% — blocked  mint=%s",
                    impact_pct, JUPITER_RESCUE_MAX_PRICE_IMPACT_PCT, mint[:8],
                )
                _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result))
                return result

        # ── (i) Governor swap build call ──────────────────────────────────────
        swap_body = {
            "quoteResponse":              quote,
            "userPublicKey":              wallet,
            "wrapAndUnwrapSol":           True,
            "dynamicComputeUnitLimit":    True,
            "prioritizationFeeLamports":  "auto",
        }

        swap_resp_json = None
        try:
            swap_resp = governor.request(
                purpose=purpose_enum,
                endpoint="swap",
                fn=_requests.post,
                mint=mint,
                url=_SWAP_URL,
                json=swap_body,
                timeout=20,
            )
            swap_resp.raise_for_status()
            swap_resp_json = swap_resp.json()
        except _requests.exceptions.HTTPError as swap_http_err:
            if swap_http_err.response is not None and swap_http_err.response.status_code == 429:
                result["jupiter_429_count"] += 1
                log.warning("jupiter_rescue: swap build 429 (EXIT), trying EMERGENCY  mint=%s", mint[:8])
                try:
                    swap_resp2 = governor.request(
                        purpose=Purpose.EMERGENCY,
                        endpoint="swap",
                        fn=_requests.post,
                        mint=mint,
                        url=_SWAP_URL,
                        json=swap_body,
                        timeout=20,
                    )
                    swap_resp2.raise_for_status()
                    swap_resp_json = swap_resp2.json()
                except Exception as sw2_err:
                    if hasattr(sw2_err, "response") and sw2_err.response is not None and sw2_err.response.status_code == 429:
                        result["jupiter_429_count"] += 1
                    result["reason"]      = "swap_build_failed"
                    result["error_class"] = "jupiter_swap_build_error"
                    log.error("jupiter_rescue: swap build EMERGENCY also failed: %s  mint=%s", sw2_err, mint[:8])
                    _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result))
                    return result
            else:
                result["reason"]      = "swap_build_http_error"
                result["error_class"] = "jupiter_swap_build_error"
                log.error("jupiter_rescue: swap build HTTP error: %s  mint=%s", swap_http_err, mint[:8])
                _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result))
                return result
        except Exception as sw_err:
            result["reason"]      = "swap_build_failed"
            result["error_class"] = "jupiter_swap_build_error"
            log.error("jupiter_rescue: swap build error: %s  mint=%s", sw_err, mint[:8])
            _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result))
            return result

        tx_b64 = swap_resp_json.get("swapTransaction", "")
        if not tx_b64:
            result["reason"]      = "swap_build_no_tx"
            result["error_class"] = "jupiter_swap_build_error"
            log.error("jupiter_rescue: swap build returned no swapTransaction  mint=%s", mint[:8])
            _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result))
            return result

        result["jupiter_swap_build_ok"] = True

        # ── (j) Sign transaction ──────────────────────────────────────────────
        try:
            from solders.transaction import VersionedTransaction  # type: ignore
            raw_tx  = base64.b64decode(tx_b64)
            vtx     = VersionedTransaction.from_bytes(raw_tx)
            signed  = VersionedTransaction(vtx.message, [kp])
            tx_bytes = bytes(signed)
        except Exception as sign_err:
            result["reason"]      = "sign_failed"
            result["error_class"] = "tx_sign_error"
            log.error("jupiter_rescue: sign failed: %s  mint=%s", sign_err, mint[:8])
            _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result))
            return result

        # ── (k) Tag notes BEFORE send ─────────────────────────────────────────
        pos.notes = (pos.notes or "") + "|jupiter_rescue_pending:sending"

        # ── (l) Send via RPC ──────────────────────────────────────────────────
        result["jupiter_send_attempted"] = True
        sig = ""
        try:
            tx_b64_signed = base64.b64encode(tx_bytes).decode()
            send_payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "sendTransaction",
                "params": [
                    tx_b64_signed,
                    {
                        "encoding":        "base64",
                        "skipPreflight":   True,
                        "maxRetries":      3,
                    },
                ],
            }
            send_data = _rpc_post(send_payload, timeout=20)
            sig = send_data.get("result", "")
            if not sig:
                err = send_data.get("error", {})
                result["reason"]      = "send_rpc_error"
                result["error_class"] = "rpc_send_error"
                log.error("jupiter_rescue: sendTransaction returned no sig  err=%s  mint=%s", err, mint[:8])
                # revert notes tag
                pos.notes = pos.notes.replace("|jupiter_rescue_pending:sending", "|jupiter_rescue_send_failed")
                _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result))
                return result
        except Exception as send_err:
            result["reason"]      = "send_failed"
            result["error_class"] = "rpc_send_error"
            log.error("jupiter_rescue: send failed: %s  mint=%s", send_err, mint[:8])
            pos.notes = pos.notes.replace("|jupiter_rescue_pending:sending", "|jupiter_rescue_send_failed")
            _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result))
            return result

        result["tx_sig"] = sig
        # Update notes with sig prefix
        pos.notes = pos.notes.replace("|jupiter_rescue_pending:sending", f"|jupiter_rescue_pending:{sig[:16]}")
        log.info("jupiter_rescue: tx sent  sig=%s  mint=%s", sig[:16], mint[:8])

        # ── (m) Confirm ───────────────────────────────────────────────────────
        confirmed = False
        deadline  = time.time() + 45
        while time.time() < deadline:
            time.sleep(2)
            try:
                confirm_payload = {
                    "jsonrpc": "2.0", "id": 1,
                    "method": "getSignatureStatuses",
                    "params": [[sig], {"searchTransactionHistory": True}],
                }
                confirm_data = _rpc_post(confirm_payload, timeout=10)
                statuses = confirm_data.get("result", {}).get("value", [None])
                st = statuses[0] if statuses else None
                if st is None:
                    continue
                err = st.get("err")
                if err is not None:
                    result["reason"]      = "tx_reverted"
                    result["error_class"] = "tx_reverted"
                    log.warning("jupiter_rescue: tx reverted  sig=%s  err=%s  mint=%s", sig[:16], err, mint[:8])
                    _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result))
                    return result
                cf = st.get("confirmationStatus", "")
                if cf in ("confirmed", "finalized"):
                    confirmed = True
                    break
            except Exception:
                pass

        result["jupiter_confirmed"] = confirmed

        if not confirmed:
            result["reason"]      = "tx_not_confirmed"
            result["error_class"] = "tx_timeout"
            log.warning("jupiter_rescue: tx not confirmed within 45s  sig=%s  mint=%s", sig[:16], mint[:8])
            _log_rescue(_build_log_row(pos_id, symbol, mint, reason, result))
            return result

        # ── (n) Sol received ──────────────────────────────────────────────────
        sol_recv = _sol_received_from_tx(sig, wallet)
        result["sol_received"] = sol_recv

        # ── fill_price: outAmount lamports → SOL → USD requires entry_price_sol
        # Best we can compute here: if sol_recv > 0 and entry_price_sol known:
        # fill_price = sol_recv / (balance / 1e{decimals}) — skip for simplicity,
        # portfolio will use pos.exit_price (stop trigger) as fill if fill_price=None.
        result["fill_price"] = None   # portfolio caller sets from pos.exit_price if needed

        # ── (o/q) Success ─────────────────────────────────────────────────────
        result["success"] = True
        result["reason"]  = "jupiter_rescue_ok"
        log.info(
            "jupiter_rescue: SUCCESS  mint=%s  sig=%s  sol=%.6f  impact=%.1f%%",
            mint[:8], sig[:16], sol_recv, impact_pct,
        )

    except Exception as outer_exc:
        # Never crash the caller
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
        ))
    except Exception:
        pass

    return result


# ── Log row builder ───────────────────────────────────────────────────────────

def _build_log_row(pos_id: str, symbol: str, mint: str, reason: str, result: dict) -> dict:
    return {
        "timestamp":                  datetime.now(timezone.utc).isoformat(),
        "position_id":                pos_id,
        "symbol":                     symbol,
        "mint":                       mint,
        "reason":                     reason,
        "rescue_trigger":             reason,
        "token_program_if_known":     "",
        "token_balance_raw":          result.get("token_balance_raw", 0),
        "used_pos_tokens_held":       result.get("used_pos_tokens_held", False),
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
        "sol_received":               result.get("sol_received", 0.0),
        "fill_price":                 result.get("fill_price"),
        "final_exit_status":          "success" if result.get("success") else result.get("reason", "failed"),
    }
