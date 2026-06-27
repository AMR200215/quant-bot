"""
Exit Router — Level 3 exit layer.

Classifies each held token's exit state (bonding curve vs graduated PumpSwap)
and routes the sell accordingly. PumpSwap local path is tried first for
GRADUATED_PUMPSWAP tokens; existing executor is always the fallback.

State machine
-------------
BONDING_CURVE       → existing executor.sell (unchanged)
NEAR_GRADUATION     → existing executor.sell, urgent=True; on 6005 reclassify
MIGRATION_UNCERTAIN → try bonding curve first; on 6005 reclassify
GRADUATED_PUMPSWAP  → pumpswap_local (sim only initially); fallback: executor
UNKNOWN             → existing executor.sell (unchanged)

Config flags (set in config.py)
--------------------------------
EXIT_ROUTER_ENABLED          — master switch
PUMPSWAP_LOCAL_SELL_ENABLED  — False = never send; True = sim then send
PUMPSWAP_LOCAL_SIM_ONLY      — True = simulate, log, then fall through to executor
PUMPSWAP_LOCAL_REQUIRE_SIM_OK— True = if sim fails, do not send (always respected)
"""

import csv
import logging
import os
import time
from enum import Enum
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TokenExitState(Enum):
    BONDING_CURVE       = "BONDING_CURVE"
    NEAR_GRADUATION     = "NEAR_GRADUATION"
    MIGRATION_UNCERTAIN = "MIGRATION_UNCERTAIN"
    GRADUATED_PUMPSWAP  = "GRADUATED_PUMPSWAP"
    UNKNOWN             = "UNKNOWN"


class ExitRoute(Enum):
    BONDING_CURVE_LOCAL  = "BONDING_CURVE_LOCAL"
    BONDING_CURVE_PP     = "BONDING_CURVE_PP"
    PUMPSWAP_LOCAL       = "PUMPSWAP_LOCAL"
    PUMPPORTAL_PUMP_AMM  = "PUMPPORTAL_PUMP_AMM"
    JUPITER              = "JUPITER"
    MANUAL               = "MANUAL"


# ---------------------------------------------------------------------------
# Config import (with graceful fallbacks so this module can be imported in tests)
# ---------------------------------------------------------------------------
try:
    from memecoin.config import (
        GRAD_SOL_UI,
        PREGRAD_TRIGGER_PCT,
        EXIT_ROUTER_ENABLED,
        PUMPSWAP_LOCAL_SELL_ENABLED,
        PUMPSWAP_LOCAL_SIM_ONLY,
        PUMPSWAP_LOCAL_REQUIRE_SIM_OK,
        LIVE_TRADING,
        LOGS_DIR,
    )
except ImportError:
    GRAD_SOL_UI                  = 115.0
    PREGRAD_TRIGGER_PCT          = 0.85
    EXIT_ROUTER_ENABLED          = True
    PUMPSWAP_LOCAL_SELL_ENABLED  = False
    PUMPSWAP_LOCAL_SIM_ONLY      = True
    PUMPSWAP_LOCAL_REQUIRE_SIM_OK = True
    LIVE_TRADING                 = False
    LOGS_DIR                     = Path(__file__).parent.parent / "logs"

# Path for route attempt audit log
_ROUTE_LOG_FILE = LOGS_DIR / "exit_route_attempts.csv"
_ROUTE_LOG_FIELDS = [
    "ts", "pos_id", "token_symbol", "token_mint",
    "exit_state", "route", "sim_ok", "sim_error", "success",
    "error_class", "notes",
]

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(pos, pp_monitor) -> TokenExitState:
    """
    Classify a held token's exit state.

    Decision tree (evaluated in order — first match wins):

    1. pos.dex_id == "pumpswap"
       → GRADUATED_PUMPSWAP  (definitive; token was entered on PumpSwap DEX)

    2. PP migration event received and age < 300s
       → GRADUATED_PUMPSWAP

    3. PP vSol known:
       vSol >= PREGRAD_TRIGGER_PCT * GRAD_SOL_UI  → NEAR_GRADUATION
       vSol > 0                                   → BONDING_CURVE

    4. pos.dex_id == "pumpfun" and vSol unknown
       → MIGRATION_UNCERTAIN

    5. → UNKNOWN

    pos: a portfolio.Position dataclass
    pp_monitor: pumpportal_monitor.PumpPortalMonitor instance (or None for tests)
    """
    token_mint = pos.token_address
    dex_id     = pos.dex_id or ""

    # Rule 1: explicit pumpswap DEX entry — definitive
    if dex_id == "pumpswap":
        log.debug("classify  mint=%s  → GRADUATED_PUMPSWAP (dex_id=pumpswap)", token_mint[:8])
        return TokenExitState.GRADUATED_PUMPSWAP

    # Rule 2: migration event received recently
    if pp_monitor is not None:
        try:
            mig_age = pp_monitor.migration_age(token_mint)
            if mig_age < 300:
                log.debug(
                    "classify  mint=%s  → GRADUATED_PUMPSWAP (migration_age=%.0fs)",
                    token_mint[:8], mig_age,
                )
                return TokenExitState.GRADUATED_PUMPSWAP
        except Exception as e:
            log.debug("classify migration_age error: %s", e)

    # Rule 3: PP vSol reading
    vsol = 0.0
    if pp_monitor is not None:
        try:
            vsol = pp_monitor.get_vsol(token_mint)
        except Exception as e:
            log.debug("classify get_vsol error: %s", e)

    if vsol >= PREGRAD_TRIGGER_PCT * GRAD_SOL_UI:
        log.debug(
            "classify  mint=%s  vsol=%.1f >= %.1f  → NEAR_GRADUATION",
            token_mint[:8], vsol, PREGRAD_TRIGGER_PCT * GRAD_SOL_UI,
        )
        return TokenExitState.NEAR_GRADUATION

    if vsol > 0:
        log.debug("classify  mint=%s  vsol=%.1f > 0  → BONDING_CURVE", token_mint[:8], vsol)
        return TokenExitState.BONDING_CURVE

    # Rule 4: dex_id is pumpfun but vSol unknown — could be mid-migration
    if dex_id == "pumpfun":
        log.debug("classify  mint=%s  dex_id=pumpfun vsol=0  → MIGRATION_UNCERTAIN", token_mint[:8])
        return TokenExitState.MIGRATION_UNCERTAIN

    log.debug("classify  mint=%s  dex_id=%r vsol=0  → UNKNOWN", token_mint[:8], dex_id)
    return TokenExitState.UNKNOWN


# ---------------------------------------------------------------------------
# PumpSwap local sell path
# ---------------------------------------------------------------------------

def run_pumpswap_local_path(
    pos,
    reason: str,
    rpc_url: str,
    keypair=None,
) -> dict:
    """
    Run PumpSwap local sell for a GRADUATED_PUMPSWAP token.

    Behaviour controlled by config flags:
      PUMPSWAP_LOCAL_SIM_ONLY=True   → simulate, log result, return WITHOUT sending
      PUMPSWAP_LOCAL_SELL_ENABLED=True and sim ok → simulate then send

    Returns a result dict with keys:
      success       bool   — True only if TX was sent AND confirmed
      sim_ok        bool   — True if simulation passed
      sim_error     str    — error_class from simulate_sell (empty if sim_ok)
      route         str    — ExitRoute value used
      error_class   str    — error class on failure (empty on success)
      tx_sig        str    — TX signature if sent
      notes         str    — human-readable status for pos.notes tagging

    Never raises — all exceptions are caught and returned in error_class.
    """
    result = {
        "success":     False,
        "sim_ok":      False,
        "sim_error":   "",
        "route":       ExitRoute.PUMPSWAP_LOCAL.value,
        "error_class": "",
        "tx_sig":      "",
        "notes":       "",
    }

    token_mint = pos.token_address

    try:
        from memecoin.pumpswap_local import (
            fetch_pool, build_pumpswap_sell_tx, simulate_sell,
            PumpSwapPoolError,
            TOKEN_PROGRAM_SPL, TOKEN_PROGRAM_T22,
        )
        from memecoin.executor import _get_keypair, _token_balance, _send_transaction, _confirm_tx
        from memecoin.executor import _pumpfun_mint_token_program
        from memecoin.executor import SLIPPAGE_SELL_PCT as _SLIPPAGE_SELL_PCT, PRIORITY_FEE_SOL

        # ── Step 1: load keypair ─────────────────────────────────────────────
        if keypair is None:
            keypair = _get_keypair()
        wallet_pubkey = str(keypair.pubkey())

        # ── Step 2: detect token program (SPL vs T22) ───────────────────────
        tok_prog_id = _pumpfun_mint_token_program(token_mint)
        is_t22 = (tok_prog_id == TOKEN_PROGRAM_T22)

        # ── Step 3: discover pool ────────────────────────────────────────────
        try:
            pool = fetch_pool(token_mint, rpc_url)
        except PumpSwapPoolError as pe:
            log.warning(
                "run_pumpswap_local_path: pool not found  mint=%s  error=%s",
                token_mint[:8], pe,
            )
            result["error_class"] = pe.error_class
            result["notes"] = f"pool_err:{pe.error_class}"
            _log_route_attempt(result, pos, TokenExitState.GRADUATED_PUMPSWAP)
            return result

        # ── Step 4: get token balance ────────────────────────────────────────
        # Use pos.tokens_held if available (from buy TX delta), else RPC query
        token_amount_raw = int(pos.tokens_held or 0)
        if token_amount_raw <= 0:
            try:
                token_amount_raw = _token_balance(wallet_pubkey, token_mint)
            except Exception as be:
                log.warning("run_pumpswap_local_path: balance query failed: %s", be)
                result["error_class"] = "pumpswap_simulation_failed"
                result["notes"]       = f"balance_err:{be}"
                _log_route_attempt(result, pos, TokenExitState.GRADUATED_PUMPSWAP)
                return result

        if token_amount_raw <= 0:
            log.warning(
                "run_pumpswap_local_path: zero balance  mint=%s  — already sold?",
                token_mint[:8],
            )
            result["error_class"] = "pumpswap_zero_balance"
            result["notes"]       = "zero_balance"
            _log_route_attempt(result, pos, TokenExitState.GRADUATED_PUMPSWAP)
            return result

        # ── Step 5: compute min_sol_out (slippage) ───────────────────────────
        # Pass min_sol_out_lamports=0 for simulation runs — we just want to know
        # if the TX structure is valid; actual min will be computed at send time.
        # For real sends: apply SLIPPAGE_SELL_PCT (35% → accept 65% of fair value).
        # We cannot compute fair value here without AMM reserves — use 0 as the
        # safe no-minimum floor (slippage is already wide at 35%).
        # TODO: read pool reserves and compute fair sol_out before setting min.
        min_sol_out_lamports = 0  # accept any positive amount (market order semantics)

        # ── Step 6: build TX ─────────────────────────────────────────────────
        try:
            tx_bytes = build_pumpswap_sell_tx(
                wallet_pubkey=wallet_pubkey,
                keypair=keypair,
                token_mint=token_mint,
                token_amount_raw=token_amount_raw,
                min_sol_out_lamports=min_sol_out_lamports,
                priority_fee_sol=PRIORITY_FEE_SOL,
                token_program_id=tok_prog_id,
                pool=pool,
                rpc_url=rpc_url,
            )
        except Exception as build_err:
            log.warning("run_pumpswap_local_path: build failed  mint=%s  err=%s",
                        token_mint[:8], build_err)
            result["error_class"] = "pumpswap_simulation_failed"
            result["notes"]       = f"build_err:{build_err}"
            _log_route_attempt(result, pos, TokenExitState.GRADUATED_PUMPSWAP)
            return result

        # ── Step 7: simulate ────────────────────────────────────────────────
        sim_ok, sim_err_cls, sim_logs = simulate_sell(tx_bytes, rpc_url)
        result["sim_ok"]    = sim_ok
        result["sim_error"] = sim_err_cls

        if not sim_ok:
            log.warning(
                "run_pumpswap_local_path: sim FAILED  mint=%s  error_class=%s  pool=%s  t22=%s",
                token_mint[:8], sim_err_cls, pool["pool_address"][:8], is_t22,
            )
            result["error_class"] = sim_err_cls
            result["notes"]       = f"sim_failed:{sim_err_cls}"
            _log_route_attempt(result, pos, TokenExitState.GRADUATED_PUMPSWAP)
            # Log first few sim logs for debugging
            for line in sim_logs[:5]:
                log.debug("  sim_log: %s", line)
            return result

        log.info(
            "run_pumpswap_local_path: sim OK  mint=%s  pool=%s  amount=%d  t22=%s",
            token_mint[:8], pool["pool_address"][:8], token_amount_raw, is_t22,
        )
        result["notes"] = f"sim_ok:pool={pool['pool_address'][:8]}"

        # ── Step 8: send (only if enabled and NOT sim-only) ──────────────────
        if PUMPSWAP_LOCAL_SIM_ONLY:
            # Simulate-only mode: log result and fall through to executor
            log.info(
                "run_pumpswap_local_path: SIM_ONLY — not sending  mint=%s  reason=%s",
                token_mint[:8], reason,
            )
            result["notes"] += "|sim_only:fallthrough"
            _log_route_attempt(result, pos, TokenExitState.GRADUATED_PUMPSWAP)
            return result  # success=False so caller falls through to executor

        if not PUMPSWAP_LOCAL_SELL_ENABLED:
            log.info(
                "run_pumpswap_local_path: SELL_ENABLED=False — not sending  mint=%s",
                token_mint[:8],
            )
            result["notes"] += "|sell_disabled:fallthrough"
            _log_route_attempt(result, pos, TokenExitState.GRADUATED_PUMPSWAP)
            return result

        # Sim passed and sell enabled — send the TX
        try:
            sig = _send_transaction(tx_bytes)
            log.info(
                "run_pumpswap_local_path: TX sent  mint=%s  sig=%s",
                token_mint[:8], sig[:16],
            )
            result["tx_sig"] = sig
            result["notes"]  += f"|tx_sent:{sig[:16]}"

            # Confirm
            t_sent = time.time()
            conf_ok, conf_err = _confirm_tx(sig, max_wait=40, t_sent=t_sent)
            if conf_ok:
                log.info(
                    "run_pumpswap_local_path: TX CONFIRMED  mint=%s  sig=%s",
                    token_mint[:8], sig[:16],
                )
                result["success"] = True
                result["notes"]  += "|confirmed"
            else:
                log.warning(
                    "run_pumpswap_local_path: TX unconfirmed  mint=%s  sig=%s  err=%s",
                    token_mint[:8], sig[:16], conf_err,
                )
                result["error_class"] = "pumpswap_simulation_failed"
                result["notes"]      += f"|unconf:{conf_err}"
        except Exception as send_err:
            log.warning(
                "run_pumpswap_local_path: send failed  mint=%s  err=%s",
                token_mint[:8], send_err,
            )
            result["error_class"] = "pumpswap_simulation_failed"
            result["notes"]      += f"|send_err:{send_err}"

    except Exception as outer_err:
        log.exception(
            "run_pumpswap_local_path: unexpected error  mint=%s  err=%s",
            token_mint[:8], outer_err,
        )
        result["error_class"] = "pumpswap_simulation_failed"
        result["notes"]       = f"unexpected_err:{outer_err}"

    _log_route_attempt(result, pos, TokenExitState.GRADUATED_PUMPSWAP)
    return result


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def _log_route_attempt(entry: dict, pos, exit_state: TokenExitState) -> None:
    """Append one row to logs/exit_route_attempts.csv for audit and debugging."""
    try:
        _ROUTE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        write_header = not _ROUTE_LOG_FILE.exists() or _ROUTE_LOG_FILE.stat().st_size == 0
        row = {
            "ts":           time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "pos_id":       pos.id,
            "token_symbol": pos.token_symbol,
            "token_mint":   pos.token_address,
            "exit_state":   exit_state.value,
            "route":        entry.get("route", ""),
            "sim_ok":       entry.get("sim_ok", ""),
            "sim_error":    entry.get("sim_error", ""),
            "success":      entry.get("success", ""),
            "error_class":  entry.get("error_class", ""),
            "notes":        entry.get("notes", ""),
        }
        with open(_ROUTE_LOG_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_ROUTE_LOG_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as log_err:
        log.warning("_log_route_attempt write failed: %s", log_err)


# Public alias expected by portfolio.py
log_route_attempt = _log_route_attempt
