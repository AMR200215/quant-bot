"""
tx_meta.py — Shared transaction metadata reader with retry.

Provides `read_sol_delta(sig, wallet)` which reads the SOL balance delta
for a given wallet from a confirmed transaction, with a multi-attempt retry
schedule to handle RPC indexing lag.

Public API
----------
read_sol_delta(sig, wallet) -> dict
    {
        "ok":       bool,
        "sol_delta": float | None,   # None = unindexed, NEVER 0.0 as sentinel
        "source":   str,             # "native_lamports" | "wsol_delta" | "unindexed" | "parse_failed"
        "attempts": int,
        "reason":   str,
    }
"""

import logging
import os
import time

import requests as _requests

log = logging.getLogger(__name__)

# Retry schedule: cumulative seconds from start before each attempt fires.
# attempt 1: immediately (0s), attempt 2: 1.5s, attempt 3: 3.5s, attempt 4: 6s
_RETRY_DELAYS = [0.0, 1.5, 3.5, 6.0]


def _rpc_post(payload: dict, timeout: int = 15) -> dict:
    """Single-entry-point for all RPC calls. Tests can monkey-patch this."""
    try:
        from memecoin.execution_rpc import rpc_post as _exec_rpc_post
        return _exec_rpc_post(payload, timeout_override_sec=float(timeout))
    except ImportError:
        url = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
        resp = _requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()


def read_sol_delta(sig: str, wallet: str) -> dict:
    """
    Read the SOL balance delta for `wallet` from transaction `sig`.

    Retries up to 4 times with increasing delays to handle RPC indexing lag.
    Never returns 0.0 as a sentinel — if the parse finds a real 0 delta,
    that's returned as a genuine value with ok=True.

    Returns
    -------
    dict with keys: ok, sol_delta, source, attempts, reason
    """
    t0 = time.monotonic()
    last_source = "unindexed"
    last_reason = "all_attempts_unindexed"

    for attempt_idx, delay in enumerate(_RETRY_DELAYS):
        # Wait until the scheduled time
        elapsed = time.monotonic() - t0
        wait = delay - elapsed
        if wait > 0:
            time.sleep(wait)

        attempt_num = attempt_idx + 1

        try:
            payload = {
                "jsonrpc": "2.0", "id": 1,
                "method": "getTransaction",
                "params": [sig, {
                    "encoding": "jsonParsed",
                    "commitment": "confirmed",
                    "maxSupportedTransactionVersion": 0,
                }],
            }
            data = _rpc_post(payload, timeout=15)
            tx = data.get("result") if isinstance(data, dict) else None

            # Handle Response objects (executor.py _rpc_post returns Response)
            if hasattr(data, 'json'):
                tx = data.json().get("result")

            if tx is None:
                # Not indexed yet — retry
                log.debug(
                    "read_sol_delta: attempt %d/%d unindexed  sig=%s",
                    attempt_num, len(_RETRY_DELAYS), sig[:16],
                )
                last_source = "unindexed"
                last_reason = "tx_not_indexed"
                continue

            # Transaction found — parse balance delta
            meta = tx.get("meta") or {}
            pre_balances = meta.get("preBalances") or []
            post_balances = meta.get("postBalances") or []
            account_keys = (
                tx.get("transaction", {})
                .get("message", {})
                .get("accountKeys", [])
            )

            for i, ak in enumerate(account_keys):
                addr = ak if isinstance(ak, str) else ak.get("pubkey", "")
                if addr == wallet and i < len(pre_balances) and i < len(post_balances):
                    pre = pre_balances[i]
                    post = post_balances[i]
                    sol_delta = (post - pre) / 1e9
                    log.info(
                        "read_sol_delta: OK  sig=%s  delta=%.6f SOL  attempt=%d",
                        sig[:16], sol_delta, attempt_num,
                    )
                    return {
                        "ok": True,
                        "sol_delta": sol_delta,
                        "source": "native_lamports",
                        "attempts": attempt_num,
                        "reason": "success",
                    }

            # Wallet not found in account keys — check wSOL token balance delta
            wsol_delta = _parse_wsol_delta(meta, wallet)
            if wsol_delta is not None:
                log.info(
                    "read_sol_delta: OK (wSOL)  sig=%s  delta=%.6f  attempt=%d",
                    sig[:16], wsol_delta, attempt_num,
                )
                return {
                    "ok": True,
                    "sol_delta": wsol_delta,
                    "source": "wsol_delta",
                    "attempts": attempt_num,
                    "reason": "success_wsol",
                }

            # Account found but wallet not in keys
            log.warning(
                "read_sol_delta: wallet not in accountKeys  sig=%s  wallet=%s",
                sig[:16], wallet[:8],
            )
            return {
                "ok": False,
                "sol_delta": None,
                "source": "parse_failed",
                "attempts": attempt_num,
                "reason": "wallet_not_in_account_keys",
            }

        except Exception as exc:
            log.debug(
                "read_sol_delta: attempt %d/%d error: %s  sig=%s",
                attempt_num, len(_RETRY_DELAYS), exc, sig[:16],
            )
            last_source = "parse_failed"
            last_reason = f"exception: {exc}"
            continue

    # All attempts exhausted
    log.warning(
        "read_sol_delta: all %d attempts exhausted  sig=%s  reason=%s",
        len(_RETRY_DELAYS), sig[:16], last_reason,
    )
    return {
        "ok": False,
        "sol_delta": None,
        "source": last_source,
        "attempts": len(_RETRY_DELAYS),
        "reason": last_reason,
    }


def _parse_wsol_delta(meta: dict, wallet: str) -> float | None:
    """Check for wSOL (wrapped SOL) token balance changes."""
    WSOL_MINT = "So11111111111111111111111111111111111111112"
    pre_tokens = {
        (b.get("accountIndex"), b.get("mint")): int(b.get("uiTokenAmount", {}).get("amount", 0))
        for b in (meta.get("preTokenBalances") or [])
        if b.get("owner") == wallet and b.get("mint") == WSOL_MINT
    }
    for b in (meta.get("postTokenBalances") or []):
        if b.get("owner") != wallet or b.get("mint") != WSOL_MINT:
            continue
        idx = b.get("accountIndex")
        post_amt = int(b.get("uiTokenAmount", {}).get("amount", 0))
        pre_amt = pre_tokens.get((idx, WSOL_MINT), 0)
        delta = (post_amt - pre_amt) / 1e9
        return delta
    return None
