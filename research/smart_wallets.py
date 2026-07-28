"""
Smart-money wallet registry.

RF6: versioned loading.

smart_wallets_latest.json (or smart_wallets_vN.json if SMART_MONEY_PINNED_VERSION is set)
is built by research/backfill_smart_wallets.py.

This module provides:
  check_smart_money(buyers)          — primary scorer (uses pinned or latest version)
  check_smart_money_versioned(buyers, version)  — score against any specific version
  get_loaded_version() -> int|None   — version number of the loaded registry
  get_metadata() -> dict|None        — metadata for the loaded version
  reload()                           — force reload from disk

A "smart wallet" is one that appeared in the first N buyers of ≥2 tokens
that went on to peak ≥+100%.

Thread-safety: per-version cache keyed by version number; all writes under _lock.
"""

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from research.config import SMART_WALLETS_PATH

log = logging.getLogger(__name__)

# ── Module-level state ─────────────────────────────────────────────────────────

_lock           = threading.Lock()
_wallets: set   = set()    # wallet address strings for the primary (pinned or latest) load
_loaded         = False
_loaded_version: Optional[int] = None   # version number of the primary loaded registry
_metadata: Optional[dict]       = None  # metadata for the primary loaded registry

# Per-version cache for check_smart_money_versioned()
# dict[int, set]  — keyed by version number
_version_cache: dict = {}
_version_cache_lock  = threading.Lock()


# ── Internal loaders ───────────────────────────────────────────────────────────

def _load_pinned_version() -> tuple[set, Optional[int], Optional[dict]]:
    """
    Load the configured version (SMART_MONEY_PINNED_VERSION) or latest.
    Returns (wallet_set, version_int_or_None, metadata_or_None).
    Never raises.
    """
    try:
        from research.config import SMART_MONEY_PINNED_VERSION, SMART_WALLETS_BASE_DIR
    except ImportError:
        from research.config import SMART_WALLETS_PATH as _p
        SMART_MONEY_PINNED_VERSION = None
        SMART_WALLETS_BASE_DIR = _p.parent

    if SMART_MONEY_PINNED_VERSION:
        path = SMART_WALLETS_BASE_DIR / f"smart_wallets_v{SMART_MONEY_PINNED_VERSION}.json"
        version = SMART_MONEY_PINNED_VERSION
    else:
        # Try latest copy first; fall back to legacy smart_wallets.json
        latest = SMART_WALLETS_BASE_DIR / "smart_wallets_latest.json"
        path = latest if latest.exists() else SMART_WALLETS_PATH
        version = None  # will be read from file

    return _load_wallet_file(path, version)


def _load_wallet_file(path: Path, hint_version: Optional[int]) -> tuple[set, Optional[int], Optional[dict]]:
    """
    Load a single wallet JSON file.
    Returns (wallet_set, version, metadata).
    Never raises.
    """
    if not path.exists():
        log.info("smart_wallets file not found at %s — registry empty until backfill runs", path)
        return set(), None, None

    try:
        data = json.loads(path.read_text())
        wallets = set(data.get("wallets", {}).keys())
        version = data.get("version") or hint_version

        # Load paired metadata file if it exists
        metadata = None
        meta_path = path.parent / f"smart_wallets_v{version}.metadata.json" if version else None
        if meta_path and meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text())
            except Exception as me:
                log.debug("metadata load failed for v%s: %s", version, me)

        log.info("smart_wallets loaded: %d wallets (v%s) from %s", len(wallets), version, path.name)
        return wallets, version, metadata
    except Exception as e:
        log.warning("smart_wallets load failed from %s: %s", path, e)
        return set(), None, None


def _ensure_loaded() -> None:
    global _wallets, _loaded, _loaded_version, _metadata
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        _wallets, _loaded_version, _metadata = _load_pinned_version()
        _loaded = True


# ── Public API ─────────────────────────────────────────────────────────────────

def reload() -> None:
    """Force reload from disk (call after backfill script writes a new file)."""
    global _loaded, _wallets, _loaded_version, _metadata
    with _lock:
        _loaded = False
        _wallets = set()
        _loaded_version = None
        _metadata = None
    with _version_cache_lock:
        _version_cache.clear()
    _ensure_loaded()


def get_loaded_version() -> Optional[int]:
    """Return the version number of the primary loaded registry, or None if unknown."""
    _ensure_loaded()
    return _loaded_version


def get_metadata() -> Optional[dict]:
    """Return the metadata dict for the primary loaded registry, or None if unavailable."""
    _ensure_loaded()
    return _metadata


def check_smart_money(buyers: list) -> tuple:
    """
    Given a list of wallet addresses (early buyers), return (hit, count) where:
      hit   bool  — True if any buyer is in the smart-wallet registry
      count int   — number of smart wallets found in the list

    Uses the pinned version (SMART_MONEY_PINNED_VERSION) or smart_wallets_latest.json.
    Returns (False, 0) if the registry is empty (backfill not yet run).
    """
    _ensure_loaded()
    if not _wallets or not buyers:
        return False, 0
    hits = sum(1 for b in buyers if b in _wallets)
    return hits > 0, hits


def check_smart_money_versioned(buyers: list, version: int) -> tuple:
    """
    Score buyers against a specific version of the smart-money registry.
    Loads from file if not already cached (thread-safe per-version cache).
    Returns (hit, count).

    Useful for shadow scoring: run V2 alongside pinned V1 without affecting
    the primary scoring path.
    """
    if not buyers:
        return False, 0

    with _version_cache_lock:
        if version not in _version_cache:
            try:
                from research.config import SMART_WALLETS_BASE_DIR
            except ImportError:
                SMART_WALLETS_BASE_DIR = Path(__file__).parent
            path = SMART_WALLETS_BASE_DIR / f"smart_wallets_v{version}.json"
            wallets, _, _ = _load_wallet_file(path, version)
            _version_cache[version] = wallets
        wallet_set = _version_cache[version]

    if not wallet_set:
        return False, 0
    hits = sum(1 for b in buyers if b in wallet_set)
    return hits > 0, hits
