"""
Dev wallet tracker.

Automatically learns the deployer wallets of profitable copy-trade tokens
and monitors them for new launches. This builds a second layer of alpha on
top of whale copy trading: instead of waiting for a whale to buy, you can
get in the moment a known successful dev deploys.

Flow:
  profitable close → get_token_dev() → register_winner_dev() → dev_wallets.json
  background poll  → poll_all_devs() → new token found → dev_launch signal
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

from memecoin.config import DEV_WALLETS_FILE, DEV_LAST_SEEN_FILE
from memecoin.data_client import (
    sol_get_token_creator, sol_detect_new_tokens,
    bscscan_get_contract_creator, bscscan_detect_new_contracts,
)

log = logging.getLogger(__name__)
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_dev_wallets() -> list[dict]:
    if not DEV_WALLETS_FILE.exists():
        return []
    try:
        return json.loads(DEV_WALLETS_FILE.read_text())
    except Exception:
        return []


def save_dev_wallets(devs: list[dict]):
    DEV_WALLETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        DEV_WALLETS_FILE.write_text(json.dumps(devs, indent=2))


def _load_last_seen() -> dict:
    if not DEV_LAST_SEEN_FILE.exists():
        return {}
    try:
        return json.loads(DEV_LAST_SEEN_FILE.read_text())
    except Exception:
        return {}


def _save_last_seen(state: dict):
    DEV_LAST_SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEV_LAST_SEEN_FILE.write_text(json.dumps(state))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def dev_score(entry: dict) -> float:
    """
    0–1 reliability score for a dev wallet.
    win_count drives base (up to 0.6), avg PnL adds bonus (up to 0.4).
    A dev with 3 wins averaging 200% PnL scores ~0.76.
    """
    wins    = entry.get("win_count", 0)
    avg_pnl = entry.get("avg_pnl_pct", 0.0)
    base      = min(wins / 5.0, 0.6)
    pnl_bonus = min(avg_pnl / 500.0, 0.4)
    return round(min(base + pnl_bonus, 1.0), 3)


def dev_signal_strength(entry: dict) -> str:
    wins    = entry.get("win_count", 0)
    avg_pnl = entry.get("avg_pnl_pct", 0.0)
    if wins >= 3 or avg_pnl >= 200:
        return "strong"
    if wins >= 2 or avg_pnl >= 100:
        return "medium"
    return "weak"


# ---------------------------------------------------------------------------
# Dev lookup
# ---------------------------------------------------------------------------

def get_token_dev(chain: str, token_address: str) -> Optional[str]:
    """Return the deployer wallet of a token. Best-effort, may return None."""
    try:
        if chain == "solana":
            return sol_get_token_creator(token_address)
        elif chain == "bsc":
            return bscscan_get_contract_creator(
                token_address, api_key=os.getenv("BSCSCAN_API_KEY", "")
            )
    except Exception as e:
        log.debug("get_token_dev(%s) failed: %s", token_address[:8], e)
    return None


def is_known_dev(chain: str, dev_address: str) -> Optional[dict]:
    """Return the dev entry if this wallet is a tracked winner dev, else None."""
    return next(
        (d for d in load_dev_wallets()
         if d["address"] == dev_address and d["chain"] == chain),
        None,
    )


# ---------------------------------------------------------------------------
# Registration — called after every profitable close
# ---------------------------------------------------------------------------

def register_winner_dev(chain: str, token_address: str, token_symbol: str,
                         pnl_pct: float, pnl_usd: float, signal_id: str):
    """
    Find and persist the deployer of a profitable token.
    Intended to be called in a background thread — involves RPC calls.
    """
    dev_addr = get_token_dev(chain, token_address)
    if not dev_addr:
        log.debug("Could not resolve dev for %s/%s", chain, token_symbol)
        return

    devs  = load_dev_wallets()
    entry = next(
        (d for d in devs if d["address"] == dev_addr and d["chain"] == chain),
        None,
    )
    win_record = {
        "token_address": token_address,
        "token_symbol":  token_symbol,
        "pnl_pct":       round(pnl_pct, 2),
        "pnl_usd":       round(pnl_usd, 2),
        "signal_id":     signal_id,
        "date":          time.strftime("%Y-%m-%d"),
    }

    if entry is None:
        entry = {
            "address":     dev_addr,
            "chain":       chain,
            "name":        "auto",
            "wins":        [win_record],
            "win_count":   1,
            "avg_pnl_pct": round(pnl_pct, 2),
            "score":       0.0,
            "added_at":    time.strftime("%Y-%m-%d"),
        }
        devs.append(entry)
        log.info("New dev tracked: %s  token=%s  pnl=+%.1f%%",
                 dev_addr[:8], token_symbol, pnl_pct)
    else:
        entry["wins"].append(win_record)
        entry["win_count"]   = len(entry["wins"])
        entry["avg_pnl_pct"] = round(
            sum(w["pnl_pct"] for w in entry["wins"]) / entry["win_count"], 2
        )
        log.info("Dev updated: %s  wins=%d  avg_pnl=%.1f%%",
                 dev_addr[:8], entry["win_count"], entry["avg_pnl_pct"])

    entry["score"] = dev_score(entry)
    save_dev_wallets(devs)


# ---------------------------------------------------------------------------
# Polling — detect new launches by tracked devs
# ---------------------------------------------------------------------------

def poll_all_devs() -> list[dict]:
    """
    Check every tracked dev wallet for new token deployments since last poll.
    Returns list of: {chain, token_address, dev_address, dev_entry}
    """
    devs = load_dev_wallets()
    if not devs:
        return []

    state      = _load_last_seen()
    bscscan_key = os.getenv("BSCSCAN_API_KEY", "")
    findings: list[dict] = []

    for entry in devs:
        chain    = entry["chain"]
        dev_addr = entry["address"]
        key      = f"{chain}:{dev_addr}"

        try:
            if chain == "solana":
                last_sig = state.get(key, "")
                new_mints, latest_sig = sol_detect_new_tokens(dev_addr, after_sig=last_sig)
                if latest_sig:
                    state[key] = latest_sig
                for mint in new_mints:
                    findings.append({
                        "chain":         chain,
                        "token_address": mint,
                        "dev_address":   dev_addr,
                        "dev_entry":     entry,
                    })

            elif chain == "bsc":
                last_block = int(state.get(key, 0))
                contracts, latest_block = bscscan_detect_new_contracts(
                    dev_addr, api_key=bscscan_key, start_block=last_block
                )
                if latest_block:
                    state[key] = latest_block
                for c in contracts:
                    findings.append({
                        "chain":         chain,
                        "token_address": c["contract_address"],
                        "dev_address":   dev_addr,
                        "dev_entry":     entry,
                    })

        except Exception as e:
            log.debug("poll dev %s/%s error: %s", chain, dev_addr[:8], e)

        time.sleep(0.5)  # rate-limit between wallets

    _save_last_seen(state)
    return findings
