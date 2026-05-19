"""
Polymarket position state machine — persisted to data/pm_positions.json.

Each position lives in one of three states:
  open      → order placed, awaiting resolution
  resolved  → market settled, outcome known
  cancelled → order was rejected or never filled

The file is read/written atomically (write-to-tmp, then rename) so a
mid-write crash never corrupts the state.
"""

from __future__ import annotations

import json
import os
import time
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

_STATE_FILE = Path(__file__).parent.parent / "data" / "pm_positions.json"
_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class PMPosition:
    # ── identity ──────────────────────────────────────────────────────────────
    market_id: str
    question: str
    side: str                # "YES" or "NO"
    token_id: str            # CLOB token ID we bought

    # ── order details ─────────────────────────────────────────────────────────
    order_id: str            # Polymarket CLOB order ID (empty string for dry-run)
    entry_price: float       # price paid per contract (0–1)
    size_usdc: float         # $ amount placed
    contracts: float         # contracts = size_usdc / entry_price

    # ── timestamps ────────────────────────────────────────────────────────────
    opened_at: float = field(default_factory=time.time)  # unix ts
    resolved_at: Optional[float] = None

    # ── resolution ────────────────────────────────────────────────────────────
    status: str = "open"     # "open" | "resolved" | "cancelled"
    outcome: Optional[str] = None   # "win" | "loss" after resolution
    exit_price: Optional[float] = None
    pnl_usdc: Optional[float] = None

    # ── metadata ──────────────────────────────────────────────────────────────
    dry_run: bool = True
    adjusted_edge: Optional[float] = None
    end_date: Optional[str] = None


# ── persistence ───────────────────────────────────────────────────────────────

def _load() -> Dict[str, dict]:
    if not _STATE_FILE.exists():
        return {}
    try:
        return json.loads(_STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save(state: Dict[str, dict]) -> None:
    tmp = _STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, _STATE_FILE)


# ── public API ────────────────────────────────────────────────────────────────

def open_position(pos: PMPosition) -> None:
    """Persist a newly opened position."""
    state = _load()
    state[pos.market_id] = asdict(pos)
    _save(state)


def get_position(market_id: str) -> Optional[PMPosition]:
    state = _load()
    raw = state.get(market_id)
    if raw is None:
        return None
    return PMPosition(**raw)


def list_open() -> List[PMPosition]:
    state = _load()
    return [PMPosition(**v) for v in state.values() if v.get("status") == "open"]


def resolve_position(
    market_id: str,
    outcome: str,          # "win" or "loss"
    exit_price: float,
) -> Optional[PMPosition]:
    """Mark a position as resolved and compute PnL."""
    state = _load()
    raw = state.get(market_id)
    if raw is None:
        return None

    pos = PMPosition(**raw)
    pos.status = "resolved"
    pos.outcome = outcome
    pos.exit_price = exit_price
    pos.resolved_at = time.time()

    if outcome == "win":
        # Contracts resolve at $1 each
        pos.pnl_usdc = round(pos.contracts - pos.size_usdc, 4)
    else:
        pos.pnl_usdc = round(-pos.size_usdc, 4)

    state[market_id] = asdict(pos)
    _save(state)
    return pos


def cancel_position(market_id: str, reason: str = "") -> None:
    state = _load()
    if market_id in state:
        state[market_id]["status"] = "cancelled"
        if reason:
            state[market_id]["cancel_reason"] = reason
        _save(state)


def daily_pnl() -> float:
    """Sum of PnL for positions opened today (UTC)."""
    import datetime
    today = datetime.date.today().isoformat()
    state = _load()
    total = 0.0
    for raw in state.values():
        opened = raw.get("opened_at", 0)
        opened_date = time.strftime("%Y-%m-%d", time.gmtime(opened))
        pnl = raw.get("pnl_usdc")
        if opened_date == today and pnl is not None:
            total += pnl
    return round(total, 4)


def open_count() -> int:
    return len(list_open())
