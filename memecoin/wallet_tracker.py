"""
Whale wallet tracker.

Loads 300 Solana + 100 BNB whale wallets, polls them periodically,
detects new buy/sell swaps, and yields WalletEvent objects.

Supported wallet file formats (auto-detected):
  - JSON list of strings:          ["addr1", "addr2", ...]
  - JSON list of objects:          [{"address": "addr1", "label": "..."}, ...]
  - Plain text, one address/line
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from memecoin.config import (
    SOL_WALLETS_FILE, BNB_WALLETS_FILE, WHALE_STATS_FILE,
    TIER1_TOP_N, TIER2_TOP_N,
    SOL_WALLET_POLL_SEC, BNB_WALLET_POLL_SEC,
    TIER1_COPY_MULTIPLIER,
)
from memecoin.data_client import (
    sol_get_recent_signatures, sol_get_transaction, sol_parse_swap,
    bscscan_get_token_txs, bscscan_parse_swap,
    gmgn_wallet_stats_sol,
    dex_get_token,
)

log = logging.getLogger(__name__)


@dataclass
class WalletEvent:
    chain: str           # "solana" | "bsc"
    wallet: str
    wallet_rank: int     # 0-based rank across all wallets on that chain
    action: str          # "buy" | "sell"
    token_address: str
    token_symbol: str
    token_name: str
    amount_native: float  # SOL or BNB spent/received
    tx_id: str            # signature (SOL) or tx_hash (BSC)
    timestamp: float      # unix seconds


# ---------------------------------------------------------------------------
# Wallet loading
# ---------------------------------------------------------------------------

def _load_wallet_file(path: Path) -> list[str]:
    """Load a wallet file, return list of address strings."""
    if not path.exists():
        log.warning("Wallet file not found: %s", path)
        return []
    raw = path.read_text().strip()
    if not raw:
        return []

    # Try JSON first
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            addrs = []
            for item in data:
                if isinstance(item, str):
                    addrs.append(item.strip())
                elif isinstance(item, dict):
                    addr = (
                        item.get("trackedWalletAddress")
                        or item.get("address")
                        or item.get("wallet")
                        or item.get("addr")
                        or ""
                    )
                    if addr:
                        addrs.append(addr.strip())
            return [a for a in addrs if a]
        if isinstance(data, dict):
            # {"wallets": [...]}
            items = data.get("wallets") or data.get("addresses") or list(data.values())
            return [str(i).strip() for i in items if i]
    except json.JSONDecodeError:
        pass

    # Plain text
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _load_db_wallets() -> dict[str, dict]:
    """
    Load active S/A/B wallets from wallet_db (Phase 7 integration point).
    Returns {address: {"tier": "S"|"A"|"B", "score": float}} or {} on failure.
    Silently falls back to empty dict if DB is unavailable or has < 3 wallets.
    """
    try:
        from wallet_db.db import get_active_wallets
        rows = get_active_wallets("solana", tiers=("S", "A", "B"))
        if len(rows) < 3:
            return {}
        return {
            r["address"]: {
                "tier":  r.get("current_tier") or "",
                "score": float(r.get("current_score") or 0.0),
            }
            for r in rows
        }
    except Exception as e:
        log.warning("wallet_db unavailable — Phase 7 falling back to JSON-only: %s", e)
        return {}


def load_all_wallets() -> dict:
    """
    Returns wallet lists for each chain.
    Solana: JSON file wallets + any DB-tiered wallets not already in the file.
    BSC:    JSON file only (DB does not yet track BSC).
    """
    json_sol = _load_wallet_file(SOL_WALLETS_FILE)
    db_wallets = _load_db_wallets()

    # Add DB wallets that aren't already in the JSON list
    json_sol_set = set(json_sol)
    extra = [addr for addr in db_wallets if addr not in json_sol_set]
    if extra:
        log.info("Phase 7: adding %d DB-tiered wallets not in JSON file", len(extra))

    return {
        "solana": json_sol + extra,
        "bsc":    _load_wallet_file(BNB_WALLETS_FILE),
    }


# ---------------------------------------------------------------------------
# Wallet ranking (by win-rate, fetched from GMGN for Solana)
# ---------------------------------------------------------------------------

def build_wallet_ranks(wallets: dict) -> dict[str, int]:
    """
    Assign ranks to wallet addresses.

    DB-tiered wallets (Phase 7) get reserved negative ranks that encode their
    Phase 6 tier directly — tier_for_wallet() reads these as DB overrides:
        S-tier → rank -3  (maps to copy-trade tier 1)
        A-tier → rank -2  (maps to copy-trade tier 2)
        B-tier → rank -1  (maps to copy-trade tier 3)

    JSON-only wallets get 0-based positive ranks sorted by whale_stats.json
    score (or file-order if no stats available).

    Returns: { wallet_address: rank }
    """
    db_wallets = _load_db_wallets()

    # whale_stats.json — fallback scoring for JSON-only wallets
    cached: dict[str, dict] = {}
    if WHALE_STATS_FILE.exists():
        try:
            entries = json.loads(WHALE_STATS_FILE.read_text())
            if isinstance(entries, list):
                for entry in entries:
                    addr = entry.get("wallet", "")
                    if addr:
                        cached[addr] = entry
            elif isinstance(entries, dict):
                cached = entries
        except Exception:
            pass

    _DB_TIER_RANK = {"S": -3, "A": -2, "B": -1}

    ranks: dict[str, int] = {}

    # Solana: DB wallets get tier-encoded negative ranks; others get score-sorted positive ranks
    sol_wallets = wallets.get("solana", [])
    json_only: list[tuple[str, float]] = []

    for addr in sol_wallets:
        db = db_wallets.get(addr)
        if db and db["tier"] in _DB_TIER_RANK:
            ranks[addr] = _DB_TIER_RANK[db["tier"]]
        else:
            # Fall back to whale_stats.json score for rank ordering
            score = cached.get(addr, {}).get("score", None)
            json_only.append((addr, float(score) if score is not None else -1.0))

    json_only.sort(key=lambda x: x[1], reverse=True)
    for i, (addr, _) in enumerate(json_only):
        ranks[addr] = i

    # BSC: no DB data yet — assign tier 2 by default
    bnb_wallets = wallets.get("bsc", [])
    for addr in bnb_wallets:
        ranks[addr] = TIER1_TOP_N  # first slot of Tier 2

    db_count = sum(1 for r in ranks.values() if r < 0)
    if db_count:
        log.debug("Phase 7: %d wallets using DB tier, %d using JSON rank",
                  db_count, len(json_only))

    return ranks


def tier_for_wallet(wallet: str, ranks: dict[str, int]) -> int:
    rank = ranks.get(wallet, 9999)
    # Negative ranks = DB-tiered wallets (Phase 7): -3=S→tier1, -2=A→tier2, -1=B→tier3
    if rank == -3:
        return 1
    if rank == -2:
        return 2
    if rank == -1:
        return 3
    # Positive ranks = JSON-only wallets (legacy rank-based tier)
    if rank < TIER1_TOP_N:
        return 1
    if rank < TIER1_TOP_N + TIER2_TOP_N:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Polling state — tracks last-seen tx per wallet
# ---------------------------------------------------------------------------

class WalletTrackerState:
    def __init__(self):
        # sol_last_sig[wallet] = last signature string seen
        self.sol_last_sig: dict[str, str] = {}
        # bnb_last_block[wallet] = last block number seen
        self.bnb_last_block: dict[str, int] = {}

    def update_sol(self, wallet: str, sig: str):
        self.sol_last_sig[wallet] = sig

    def update_bnb(self, wallet: str, block: int):
        self.bnb_last_block[wallet] = max(
            self.bnb_last_block.get(wallet, 0), block
        )


_state = WalletTrackerState()


# ---------------------------------------------------------------------------
# Solana polling
# ---------------------------------------------------------------------------

def poll_sol_wallet(wallet: str, rank: int) -> list[WalletEvent]:
    """Poll a single Solana wallet. Returns list of new WalletEvents."""
    events: list[WalletEvent] = []
    sigs = sol_get_recent_signatures(wallet, limit=10)
    if not sigs:
        return events

    last_known = _state.sol_last_sig.get(wallet)
    new_sigs = []
    for s in sigs:
        sig_str = s.get("signature", "")
        if sig_str == last_known:
            break
        new_sigs.append(sig_str)

    if new_sigs:
        _state.update_sol(wallet, sigs[0]["signature"])

    for sig_str in new_sigs[:5]:  # limit to 5 new txs per poll
        tx = sol_get_transaction(sig_str)
        swap = sol_parse_swap(tx)
        if not swap:
            continue

        # enrich symbol from DexScreener if possible
        token_sym = swap.get("token_symbol", "")
        token_name = ""
        if swap["token_address"]:
            pair = dex_get_token("solana", swap["token_address"])
            if pair:
                base = pair.get("baseToken") or {}
                token_sym  = base.get("symbol", token_sym)
                token_name = base.get("name", "")

        events.append(WalletEvent(
            chain="solana",
            wallet=wallet,
            wallet_rank=rank,
            action=swap["action"],
            token_address=swap["token_address"],
            token_symbol=token_sym,
            token_name=token_name,
            amount_native=swap.get("amount_sol", 0.0),
            tx_id=sig_str,
            timestamp=time.time(),
        ))
        time.sleep(0.1)  # gentle rate-limit on RPC

    return events


def poll_sol_wallets_batch(
    wallets: list[str],
    ranks: dict[str, int],
    batch_size: int = 10,
) -> list[WalletEvent]:
    """Poll all Solana wallets in batches. Returns aggregated events."""
    all_events: list[WalletEvent] = []
    for i in range(0, len(wallets), batch_size):
        batch = wallets[i: i + batch_size]
        for w in batch:
            try:
                evts = poll_sol_wallet(w, ranks.get(w, 9999))
                all_events.extend(evts)
            except Exception as e:
                log.debug("poll_sol_wallet(%s) error: %s", w, e)
        time.sleep(0.3)  # pause between batches
    return all_events


# ---------------------------------------------------------------------------
# BSC polling
# ---------------------------------------------------------------------------

def poll_bnb_wallet(wallet: str, rank: int,
                    bscscan_api_key: str = "") -> list[WalletEvent]:
    """Poll a single BSC wallet. Returns new WalletEvents."""
    start_block = _state.bnb_last_block.get(wallet, 0)
    # use start_block+1 so we don't re-process the last seen block
    txs = bscscan_get_token_txs(wallet, api_key=bscscan_api_key,
                                  start_block=start_block + 1)
    if not txs:
        return []

    swaps = bscscan_parse_swap(txs, wallet)
    events: list[WalletEvent] = []
    for s in swaps:
        block = s.get("block", 0)
        _state.update_bnb(wallet, block)

        # enrich via DexScreener
        token_sym  = s.get("token_symbol", "")
        token_name = s.get("token_name", "")
        if s["token_address"]:
            pair = dex_get_token("bsc", s["token_address"])
            if pair:
                base = pair.get("baseToken") or {}
                token_sym  = base.get("symbol", token_sym)
                token_name = base.get("name", token_name)

        events.append(WalletEvent(
            chain="bsc",
            wallet=wallet,
            wallet_rank=rank,
            action=s["action"],
            token_address=s["token_address"],
            token_symbol=token_sym,
            token_name=token_name,
            amount_native=0.0,
            tx_id=s.get("tx_hash", ""),
            timestamp=time.time(),
        ))
    return events


def poll_bnb_wallets_batch(
    wallets: list[str],
    ranks: dict[str, int],
    bscscan_api_key: str = "",
    batch_size: int = 5,
) -> list[WalletEvent]:
    all_events: list[WalletEvent] = []
    for i in range(0, len(wallets), batch_size):
        batch = wallets[i: i + batch_size]
        for w in batch:
            try:
                evts = poll_bnb_wallet(w, ranks.get(w, 9999), bscscan_api_key)
                all_events.extend(evts)
            except Exception as e:
                log.debug("poll_bnb_wallet(%s) error: %s", w, e)
        time.sleep(0.5)  # BscScan rate limit
    return all_events
