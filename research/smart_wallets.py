"""
Smart-money wallet registry.

smart_wallets.json is built by research/backfill_smart_wallets.py.
This module provides a lightweight, cached loader + lookup function
used by tracker.py at ingest time.

A "smart wallet" is one that appeared in the first N buyers of ≥2 tokens
that went on to peak ≥+100%.
"""

import json
import logging
import threading
from pathlib import Path

from research.config import SMART_WALLETS_PATH

log = logging.getLogger(__name__)

_lock    = threading.Lock()
_wallets: set = set()   # set of wallet address strings
_loaded  = False


def _ensure_loaded() -> None:
    global _wallets, _loaded
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        if SMART_WALLETS_PATH.exists():
            try:
                data = json.loads(SMART_WALLETS_PATH.read_text())
                _wallets = set(data.get("wallets", {}).keys())
                log.info("smart_wallets loaded: %d wallets", len(_wallets))
            except Exception as e:
                log.warning("smart_wallets load failed: %s", e)
                _wallets = set()
        else:
            log.info("smart_wallets.json not found — smart_money_hit will be NULL until backfill runs")
            _wallets = set()
        _loaded = True


def reload() -> None:
    """Force reload from disk (call after backfill script writes a new file)."""
    global _loaded
    with _lock:
        _loaded = False
    _ensure_loaded()


def check_smart_money(buyers: list) -> tuple:
    """
    Given a list of wallet addresses (early buyers), return (hit, count) where:
      hit   bool  — True if any buyer is in the smart-wallet registry
      count int   — number of smart wallets found in the list

    Returns (False, 0) if the registry is empty (backfill not yet run).
    """
    _ensure_loaded()
    if not _wallets or not buyers:
        return False, 0
    hits = sum(1 for b in buyers if b in _wallets)
    return hits > 0, hits
