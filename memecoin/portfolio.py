"""
Paper trade portfolio for the memecoin module.

Tracks open positions, evaluates exit conditions on every price update,
and writes closed trades to the journal CSV.
"""

import csv
import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from memecoin.config import (
    POSITIONS_FILE, JOURNAL_FILE,
    HARD_STOP_PCT, TRAILING_STOP_PCT, TRAIL_ACTIVATES_PCT,
    TIME_STOP_MINUTES, TIME_STOP_MIN_GAIN,
    TP_LEVELS, TRADE_SIZE_USD,
    get_signal_settings,
)
from memecoin.data_client import dex_get_token
from memecoin.candidate_log import promote_to_winners

log = logging.getLogger(__name__)

JOURNAL_FIELDS = [
    # identity
    "id", "signal_id", "chain", "token_address", "token_symbol",
    "signal_type", "strength",
    # trade
    "entry_price", "entry_time", "size_usd",
    "exit_price", "exit_time", "exit_reason",
    "pnl_usd", "pnl_pct", "peak_price",
    # whale info
    "whale_count", "whale_tiers",
    # scores
    "safety_score", "momentum_score", "composite_score",
    # price action at entry (model features)
    "price_change_5m", "price_change_1h", "price_change_6h",
    # buy pressure at entry (model features)
    "buys_5m", "sells_5m", "buys_h1", "sells_h1",
    "buy_sell_ratio_5m", "buy_sell_ratio_h1",
    # volume at entry
    "volume_5m", "volume_h1", "volume_h6",
    # market at entry
    "liquidity_usd", "mcap_usd", "fdv", "age_minutes",
    # token info
    "dex_id", "dexscreener_url",
    "has_twitter", "has_telegram", "has_website",
    "rugcheck_score", "buy_tax", "sell_tax",
    "notes",
]


@dataclass
class Position:
    id: str
    signal_id: str
    chain: str
    token_address: str
    token_symbol: str
    signal_type: str
    strength: str
    whale_count: int
    whale_tiers: list           # e.g. [1, 2]
    whales_involved: list       # wallet addresses
    entry_price: float
    entry_time: float           # unix seconds
    size_usd: float             # TRADE_SIZE_USD
    # live / updated fields
    current_price: float = 0.0
    peak_price: float = 0.0
    status: str = "open"        # open | closed
    exit_price: float = 0.0
    exit_time: float = 0.0
    exit_reason: str = ""
    # per-position exit settings (editable live)
    hard_stop_pct: float = -0.35
    trailing_stop_pct: float = -0.40
    trail_activates_pct: float = 1.00
    time_stop_minutes: int = 45
    # partial TP tracking
    tp_levels_hit: list = field(default_factory=list)
    remaining_fraction: float = 1.0  # 1.0 = full position still open
    notes: str = ""
    # --- model training features (captured at entry) ---
    price_change_5m: float = 0.0
    price_change_1h: float = 0.0
    price_change_6h: float = 0.0
    buys_5m: int = 0
    sells_5m: int = 0
    buys_h1: int = 0
    sells_h1: int = 0
    buy_sell_ratio_5m: float = 0.0
    buy_sell_ratio_h1: float = 0.0
    volume_5m: float = 0.0
    volume_h1: float = 0.0
    volume_h6: float = 0.0
    liquidity_usd: float = 0.0
    mcap_usd: float = 0.0
    fdv: float = 0.0
    age_minutes: float = 0.0
    safety_score: float = 0.0
    momentum_score: float = 0.0
    composite_score: float = 0.0
    dex_id: str = ""
    dexscreener_url: str = ""
    has_twitter: bool = False
    has_telegram: bool = False
    has_website: bool = False
    rugcheck_score: float = 0.0
    buy_tax: float = 0.0
    sell_tax: float = 0.0

    @property
    def pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        ref = self.exit_price if self.status == "closed" else self.current_price
        return (ref - self.entry_price) / self.entry_price

    @property
    def pnl_usd(self) -> float:
        return self.pnl_pct * self.size_usd


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _load_positions() -> dict[str, Position]:
    if not POSITIONS_FILE.exists():
        return {}
    try:
        raw = json.loads(POSITIONS_FILE.read_text())
        out = {}
        for d in raw:
            p = Position(**d)
            out[p.id] = p
        return out
    except Exception as e:
        log.warning("Could not load positions: %s", e)
        return {}


def _save_positions(positions: dict[str, Position]):
    POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(p) for p in positions.values()]
    POSITIONS_FILE.write_text(json.dumps(data, indent=2))


def _append_journal(pos: Position):
    JOURNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_header = not JOURNAL_FILE.exists()
    with open(JOURNAL_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "id": pos.id,
            "signal_id": pos.signal_id,
            "chain": pos.chain,
            "token_address": pos.token_address,
            "token_symbol": pos.token_symbol,
            "signal_type": pos.signal_type,
            "strength": pos.strength,
            "entry_price": pos.entry_price,
            "entry_time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(pos.entry_time)),
            "size_usd": pos.size_usd,
            "exit_price": pos.exit_price,
            "exit_time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(pos.exit_time)) if pos.exit_time else "",
            "exit_reason": pos.exit_reason,
            "pnl_usd": round(pos.pnl_usd, 4),
            "pnl_pct": round(pos.pnl_pct * 100, 2),
            "peak_price": pos.peak_price,
            "whale_count": pos.whale_count,
            "whale_tiers": ",".join(str(t) for t in pos.whale_tiers),
            "safety_score": pos.safety_score,
            "momentum_score": pos.momentum_score,
            "composite_score": pos.composite_score,
            "price_change_5m": pos.price_change_5m,
            "price_change_1h": pos.price_change_1h,
            "price_change_6h": pos.price_change_6h,
            "buys_5m": pos.buys_5m,
            "sells_5m": pos.sells_5m,
            "buys_h1": pos.buys_h1,
            "sells_h1": pos.sells_h1,
            "buy_sell_ratio_5m": pos.buy_sell_ratio_5m,
            "buy_sell_ratio_h1": pos.buy_sell_ratio_h1,
            "volume_5m": pos.volume_5m,
            "volume_h1": pos.volume_h1,
            "volume_h6": pos.volume_h6,
            "liquidity_usd": pos.liquidity_usd,
            "mcap_usd": pos.mcap_usd,
            "fdv": pos.fdv,
            "age_minutes": round(pos.age_minutes, 1),
            "dex_id": pos.dex_id,
            "dexscreener_url": pos.dexscreener_url,
            "has_twitter": pos.has_twitter,
            "has_telegram": pos.has_telegram,
            "has_website": pos.has_website,
            "rugcheck_score": pos.rugcheck_score,
            "buy_tax": pos.buy_tax,
            "sell_tax": pos.sell_tax,
            "notes": pos.notes,
        })


# ---------------------------------------------------------------------------
# Portfolio manager
# ---------------------------------------------------------------------------

class Portfolio:
    def __init__(self):
        self._positions: dict[str, Position] = _load_positions()

    # ---- open ----

    def open_position(self, signal) -> Position:
        """Open a paper position from a Signal object."""
        pos = Position(
            id=str(uuid.uuid4())[:8],
            signal_id=signal.id,
            chain=signal.chain,
            token_address=signal.token_address,
            token_symbol=signal.token_symbol,
            signal_type=signal.signal_type,
            strength=signal.strength,
            whale_count=getattr(signal, "whale_count", 0),
            whale_tiers=list(getattr(signal, "whale_tiers", [])),
            whales_involved=list(getattr(signal, "whales_involved", [])),
            entry_price=signal.price_usd,
            entry_time=time.time(),
            size_usd=get_signal_settings(signal.signal_type)["trade_size_usd"],
            hard_stop_pct=get_signal_settings(signal.signal_type)["hard_stop_pct"],
            trailing_stop_pct=get_signal_settings(signal.signal_type)["trailing_stop_pct"],
            trail_activates_pct=get_signal_settings(signal.signal_type)["trail_activates_pct"],
            time_stop_minutes=get_signal_settings(signal.signal_type)["time_stop_minutes"],
            current_price=signal.price_usd,
            peak_price=signal.price_usd,
            # enriched model features
            price_change_5m=getattr(signal, "price_change_5m", 0.0),
            price_change_1h=getattr(signal, "price_change_1h", 0.0),
            price_change_6h=getattr(signal, "price_change_6h", 0.0),
            buys_5m=getattr(signal, "buys_5m", 0),
            sells_5m=getattr(signal, "sells_5m", 0),
            buys_h1=getattr(signal, "buys_h1", 0),
            sells_h1=getattr(signal, "sells_h1", 0),
            buy_sell_ratio_5m=getattr(signal, "buy_sell_ratio_5m", 0.0),
            buy_sell_ratio_h1=getattr(signal, "buy_sell_ratio_h1", 0.0),
            volume_5m=getattr(signal, "volume_5m", 0.0),
            volume_h1=getattr(signal, "volume_h1", 0.0),
            volume_h6=getattr(signal, "volume_h6", 0.0),
            liquidity_usd=getattr(signal, "liquidity_usd", 0.0),
            mcap_usd=getattr(signal, "mcap_usd", 0.0),
            fdv=getattr(signal, "fdv", 0.0),
            age_minutes=getattr(signal, "age_minutes", 0.0),
            safety_score=getattr(signal, "safety_score", 0.0),
            momentum_score=getattr(signal, "momentum_score", 0.0),
            composite_score=getattr(signal, "composite_score", 0.0),
            dex_id=getattr(signal, "dex_id", ""),
            dexscreener_url=getattr(signal, "dexscreener_url", ""),
            has_twitter=getattr(signal, "has_twitter", False),
            has_telegram=getattr(signal, "has_telegram", False),
            has_website=getattr(signal, "has_website", False),
            rugcheck_score=getattr(signal, "rugcheck_score", 0.0),
            buy_tax=getattr(signal, "buy_tax", 0.0),
            sell_tax=getattr(signal, "sell_tax", 0.0),
        )
        self._positions[pos.id] = pos
        _save_positions(self._positions)
        log.info("Opened paper position %s  %s/%s @ $%.8f",
                 pos.id, pos.chain, pos.token_symbol, pos.entry_price)
        return pos

    # ---- close ----

    def close_position(self, pos_id: str, reason: str,
                       price: float = 0.0) -> Optional[Position]:
        pos = self._positions.get(pos_id)
        if not pos or pos.status == "closed":
            return None
        pos.exit_price = price or pos.current_price
        pos.exit_time  = time.time()
        pos.exit_reason = reason
        pos.status = "closed"
        _append_journal(pos)
        promote_to_winners(pos)
        del self._positions[pos_id]
        _save_positions(self._positions)
        log.info("Closed position %s  reason=%s  pnl=%.1f%%",
                 pos_id, reason, pos.pnl_pct * 100)
        return pos

    # ---- update prices & evaluate exit conditions ----

    def update_prices(self, whale_sells: dict[str, list[str]] = None) -> list[dict]:
        """
        Fetch current prices for all open positions, evaluate exit conditions.

        whale_sells: { token_address: [wallet1, wallet2] } — wallets that just sold

        Returns list of exit events: [{"pos_id", "reason", "pnl_pct"}]
        """
        if whale_sells is None:
            whale_sells = {}

        exits = []
        for pos in list(self._positions.values()):
            if pos.status != "open":
                continue

            # fetch latest price
            pair = dex_get_token(pos.chain, pos.token_address)
            if pair:
                price = float(pair.get("priceUsd") or 0)
                if price > 0:
                    pos.current_price = price
                    pos.peak_price = max(pos.peak_price, price)

            gain = pos.pnl_pct
            reason = None

            # 1. Hard stop
            if gain <= pos.hard_stop_pct:
                reason = "hard_stop"

            # 2. Trailing stop
            elif pos.peak_price > 0 and gain >= pos.trail_activates_pct:
                drawdown_from_peak = (pos.current_price - pos.peak_price) / pos.peak_price
                if drawdown_from_peak <= pos.trailing_stop_pct:
                    reason = "trailing_stop"

            # 3. Whale exit (primary exit signal)
            elif pos.token_address in whale_sells:
                sellers = whale_sells[pos.token_address]
                involved = [w for w in sellers if w in pos.whales_involved]
                if involved:
                    n_whales = pos.whale_count or 1
                    # single whale entry → exit immediately when they sell
                    if n_whales == 1:
                        reason = f"whale_exit:{involved[0][:8]}"
                    # multi-whale: exit when majority sell
                    elif len(involved) >= max(1, n_whales // 2):
                        reason = f"whale_exit:{len(involved)}_of_{n_whales}"

            # 4. Time stop
            elif (time.time() - pos.entry_time) / 60 > pos.time_stop_minutes:
                if gain < TIME_STOP_MIN_GAIN:
                    reason = "time_stop"

            if reason:
                closed = self.close_position(pos.id, reason)
                if closed:
                    exits.append({
                        "pos_id": pos.id,
                        "token_symbol": pos.token_symbol,
                        "reason": reason,
                        "pnl_pct": round(pos.pnl_pct * 100, 2),
                    })
            else:
                # check take-profit ladder
                for tp_pct, tp_fraction in TP_LEVELS:
                    level_key = f"tp_{int(tp_pct*100)}"
                    if gain >= tp_pct and level_key not in pos.tp_levels_hit:
                        pos.tp_levels_hit.append(level_key)
                        sell_frac = tp_fraction * pos.remaining_fraction
                        pos.remaining_fraction -= sell_frac
                        partial_usd = sell_frac * pos.size_usd
                        log.info(
                            "TP hit %s  %s +%.0f%%  sold %.0f%% ($%.2f)",
                            pos.id, pos.token_symbol, gain * 100,
                            tp_fraction * 100, partial_usd,
                        )

        _save_positions(self._positions)
        return exits

    # ---- manual ----

    def manual_close(self, pos_id: str) -> Optional[Position]:
        pos = self._positions.get(pos_id)
        if not pos:
            return None
        return self.close_position(pos_id, "manual", pos.current_price)

    # ---- queries ----

    def open_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.status == "open"]

    def get_position(self, pos_id: str) -> Optional[Position]:
        return self._positions.get(pos_id)

    def summary(self) -> dict:
        open_pos = self.open_positions()
        return {
            "open_count": len(open_pos),
            "total_deployed_usd": sum(p.size_usd for p in open_pos),
            "unrealised_pnl_usd": round(sum(p.pnl_usd for p in open_pos), 2),
        }

    def load_journal(self) -> list[dict]:
        if not JOURNAL_FILE.exists():
            return []
        with open(JOURNAL_FILE, newline="") as f:
            return list(csv.DictReader(f))
