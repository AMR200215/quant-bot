"""
bonding_curve_t22.py — T22 pump.fun bonding-curve sell path.

Builds and sends a pump.fun bonding-curve sell TX for Token-2022 mints.
Key difference from SPL path: user_associated_token_account is derived
with TOKEN_PROGRAM_T22 (not TOKEN_PROGRAM_SPL).

Route name: BONDING_CURVE_T22_LOCAL
"""

import logging
import struct as _struct
import time

log = logging.getLogger(__name__)

# Token program IDs
TOKEN_PROGRAM_SPL = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_PROGRAM_T22 = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"


def build_bc_t22_sell_tx(
    token_mint: str,
    wallet_pubkey: str,
    token_amount: int,
    rpc_url: str,
    slippage_bps: int = 500,
    keypair=None,
    priority_fee_sol: float = 0.0005,
) -> dict:
    """
    Build a pump.fun bonding-curve sell TX for a Token-2022 mint.

    Returns:
      {"tx_bytes": bytes, "min_sol_lam": int, "token_amount": int, "error": str}
    """
    result = {"tx_bytes": None, "min_sol_lam": 0, "token_amount": token_amount, "error": ""}
    try:
        # Import executor helpers — reuse all the existing machinery
        from memecoin.executor import (
            _pumpfun_local_build_tx, _get_keypair, PRIORITY_FEE_SOL,
        )
        if keypair is None:
            keypair = _get_keypair()

        slippage_pct = slippage_bps // 100
        tx_bytes = _pumpfun_local_build_tx(
            action="sell",
            wallet_pubkey=wallet_pubkey,
            token_mint=token_mint,
            keypair=keypair,
            token_amount=token_amount,
            slippage_pct=slippage_pct,
            priority_fee_sol=priority_fee_sol or PRIORITY_FEE_SOL,
        )
        # Note: _pumpfun_local_build_tx already calls _pumpfun_mint_token_program()
        # internally and sets the correct token program (T22) for ATA derivation.
        result["tx_bytes"] = tx_bytes
        log.debug("build_bc_t22_sell_tx: built OK  mint=%s  amount=%d", token_mint[:8], token_amount)
    except Exception as e:
        result["error"] = str(e)
        log.warning("build_bc_t22_sell_tx: failed  mint=%s  err=%s", token_mint[:8], e)
    return result


def simulate_bc_t22_sell(
    token_mint: str,
    wallet_pubkey: str,
    token_amount: int,
    rpc_url: str,
) -> dict:
    """
    Simulate the BC T22 sell TX.

    Returns:
      {"sim_ok": bool, "error_class": str, "logs": list}
    """
    result = {"sim_ok": False, "error_class": "", "logs": []}
    try:
        build_result = build_bc_t22_sell_tx(
            token_mint=token_mint,
            wallet_pubkey=wallet_pubkey,
            token_amount=token_amount,
            rpc_url=rpc_url,
        )
        if build_result.get("error") or not build_result.get("tx_bytes"):
            result["error_class"] = "bc_t22_build_failed"
            return result

        from memecoin.pumpswap_local import simulate_sell
        sim_ok, sim_err_cls, sim_logs = simulate_sell(build_result["tx_bytes"], rpc_url)
        result["sim_ok"]      = sim_ok
        result["error_class"] = sim_err_cls
        result["logs"]        = sim_logs
    except Exception as e:
        result["error_class"] = "bc_t22_simulate_error"
        log.warning("simulate_bc_t22_sell: error  mint=%s  err=%s", token_mint[:8], e)
    return result


def run_bc_t22_sell(pos, reason: str, rpc_url: str) -> dict:
    """
    Full BONDING_CURVE_T22_LOCAL path: build → simulate → send.

    Behaviour controlled by PUMPSWAP_LOCAL_SELL_ENABLED config flag
    (reused to gate BC T22 sends as well).

    Returns result dict matching run_pumpswap_local_path() format.
    """
    result = {
        "success":      False,
        "fill_price":   0.0,
        "tx_sig":       "",
        "sim_ok":       False,
        "sim_error":    "",
        "error_class":  "",
        "route":        "BONDING_CURVE_T22_LOCAL",
        "is_token2022": True,
        "token_program": TOKEN_PROGRAM_T22,
        "notes":        "",
    }

    token_mint = pos.token_address

    try:
        from memecoin.executor import _get_keypair, _token_balance, _send_transaction, _confirm_tx
        from memecoin.config import PUMPSWAP_LOCAL_SELL_ENABLED, PUMPSWAP_LOCAL_SIM_ONLY

        keypair      = _get_keypair()
        wallet_pubkey = str(keypair.pubkey())

        # ── Step 1: get token balance ────────────────────────────────────────
        token_amount_raw = int(pos.tokens_held or 0)
        if token_amount_raw <= 0:
            try:
                token_amount_raw = _token_balance(wallet_pubkey, token_mint)
            except Exception as be:
                log.warning("run_bc_t22_sell: balance query failed: %s", be)
                result["error_class"] = "bc_t22_zero_balance"
                result["notes"]       = f"balance_err:{be}"
                return result

        if token_amount_raw <= 0:
            log.warning("run_bc_t22_sell: zero balance  mint=%s", token_mint[:8])
            result["error_class"] = "bc_t22_zero_balance"
            result["notes"]       = "zero_balance"
            return result

        # ── Step 2: build TX ─────────────────────────────────────────────────
        build_result = build_bc_t22_sell_tx(
            token_mint=token_mint,
            wallet_pubkey=wallet_pubkey,
            token_amount=token_amount_raw,
            rpc_url=rpc_url,
            keypair=keypair,
        )
        if build_result.get("error") or not build_result.get("tx_bytes"):
            result["error_class"] = "bc_t22_build_failed"
            result["notes"]       = f"build_err:{build_result.get('error', 'unknown')}"
            return result

        tx_bytes = build_result["tx_bytes"]

        # ── Step 3: simulate ─────────────────────────────────────────────────
        try:
            from memecoin.pumpswap_local import simulate_sell
            sim_ok, sim_err_cls, sim_logs = simulate_sell(tx_bytes, rpc_url)
        except Exception as sim_ex:
            sim_ok      = False
            sim_err_cls = "bc_t22_simulate_error"
            sim_logs    = []
            log.warning("run_bc_t22_sell: sim exception  mint=%s  err=%s", token_mint[:8], sim_ex)

        result["sim_ok"]    = sim_ok
        result["sim_error"] = sim_err_cls

        if not sim_ok:
            log.warning(
                "run_bc_t22_sell: sim FAILED  mint=%s  error_class=%s  t22=True",
                token_mint[:8], sim_err_cls,
            )
            result["error_class"] = sim_err_cls
            result["notes"]       = f"sim_failed:{sim_err_cls}"
            for line in sim_logs[:5]:
                log.debug("  sim_log: %s", line)
            return result

        log.info("run_bc_t22_sell: sim OK  mint=%s  amount=%d", token_mint[:8], token_amount_raw)
        result["notes"] = "sim_ok"

        # ── Step 4: send (gated by config) ───────────────────────────────────
        if PUMPSWAP_LOCAL_SIM_ONLY:
            log.info("run_bc_t22_sell: SIM_ONLY — not sending  mint=%s  reason=%s",
                     token_mint[:8], reason)
            result["notes"] += "|sim_only:fallthrough"
            return result  # success=False so caller falls through to executor

        if not PUMPSWAP_LOCAL_SELL_ENABLED:
            log.info("run_bc_t22_sell: SELL_ENABLED=False — not sending  mint=%s", token_mint[:8])
            result["notes"] += "|sell_disabled:fallthrough"
            return result

        # Send
        try:
            sig = _send_transaction(tx_bytes)
            log.info("run_bc_t22_sell: TX sent  mint=%s  sig=%s", token_mint[:8], sig[:16])
            result["tx_sig"] = sig
            result["notes"] += f"|tx_sent:{sig[:16]}"

            t_sent = time.time()
            conf_ok, conf_err = _confirm_tx(sig, max_wait=40, t_sent=t_sent)
            if conf_ok:
                log.info("run_bc_t22_sell: TX CONFIRMED  mint=%s  sig=%s", token_mint[:8], sig[:16])
                result["success"] = True
                result["notes"]  += "|confirmed"
            else:
                log.warning("run_bc_t22_sell: TX unconfirmed  mint=%s  sig=%s  err=%s",
                            token_mint[:8], sig[:16], conf_err)
                result["error_class"] = "bc_t22_unconfirmed"
                result["notes"]      += f"|unconf:{conf_err}"
        except Exception as send_err:
            log.warning("run_bc_t22_sell: send failed  mint=%s  err=%s", token_mint[:8], send_err)
            result["error_class"] = "bc_t22_send_failed"
            result["notes"]      += f"|send_err:{send_err}"

    except Exception as outer_err:
        log.exception("run_bc_t22_sell: unexpected error  mint=%s  err=%s", token_mint[:8], outer_err)
        result["error_class"] = "bc_t22_unexpected_error"
        result["notes"]       = f"unexpected_err:{outer_err}"

    return result
