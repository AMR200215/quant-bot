"""
Paper-trade portfolio for the sniper module.
Tracks open snipe positions, evaluates exits, writes journal CSV.
"""

import csv
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from sniper.config import (
    POSITIONS_FILE, JOURNAL_FILE,
    HARD_STOP_PCT, TRAILING_STOP_PCT, TRAIL_ACTIVATES_PCT,
    TIME_STOP_MINUTES, TIME_STOP_MIN_GAIN, TP_LEVELS,
    SNIPE_SIZE_LAUNCH, SNIPE_SIZE_MIGRATION,
)

log = logging.getLogger(__name__)

JOURNAL_FIELDS = [
    "id", "mint", "name", "symbol", "strategy",
    "entry_price_sol", "entry_time", "size_usd",
    "exit_price_sol", "exit_time", "exit_reason",
    "pnl_usd", "pnl_pct", "peak_price_sol",
    "market_cap_sol_entry", "initial_buy_sol",
    "has_twitter", "has_telegram", "has_website",
    "deployer",
    "notes",
    "config_tag",
]

CONFIG_TAG = "v1_2026-05-05"   # sniper launched today


@dataclass
class SnipePosition:
    id: str
    mint: str
    name: str
    symbol: str
    strategy: str                # "launch" | "migration"
    deployer: str
    entry_price_sol: float
    entry_time: float
    size_usd: float
    market_cap_sol_entry: float
    initial_buy_sol: float
    has_twitter: bool
    has_telegram: bool
    has_website: bool
    # live fields
    current_price_sol: float = 0.0
    peak_price_sol: float = 0.0
    status: str = "open"
    exit_price_sol: float = 0.0
    exit_time: float = 0.0
    exit_reason: str = ""
    # exit settings
    hard_stop_pct: float = HARD_STOP_PCT
    trailing_stop_pct: float = TRAILING_STOP_PCT
    trail_activates_pct: float = TRAIL_ACTIVATES_PCT
    time_stop_minutes: int = TIME_STOP_MINUTES
    # partial TP tracking
    tp_levels_hit: list = field(default_factory=list)
    remaining_fraction: float = 1.0
    notes: str = ""

    @property
    def pnl_pct(self) -> float:
        if self.entry_price_sol <= 0:
            return 0.0
        ref = self.exit_price_sol if self.status == "closed" else self.current_price_sol
        return (ref - self.entry_price_sol) / self.entry_price_sol

    @property
    def pnl_usd(self) -> float:
        return self.pnl_pct * self.size_usd


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _load_positions() -> dict[str, SnipePosition]:
    if not POSITIONS_FILE.exists():
        return {}
    try:
        raw = json.loads(POSITIONS_FILE.read_text())
        return {d["id"]: SnipePosition(**d) for d in raw}
    except Exception as e:
        log.warning("Could not load sniper positions: %s", e)
        return {}


def _save_positions(positions: dict[str, SnipePosition]):
    POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    POSITIONS_FILE.write_text(json.dumps([asdict(p) for p in positions.values()], indent=2))


def _append_journal(pos: SnipePosition):
    JOURNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Auto-fix stale header
    if JOURNAL_FILE.exists() and JOURNAL_FILE.stat().st_size > 0:
        with open(JOURNAL_FILE, newline="") as f:
            header = next(csv.reader(f), [])
        if header != JOURNAL_FIELDS:
            with open(JOURNAL_FILE, newline="") as f:
                rows = list(csv.DictReader(f))
            with open(JOURNAL_FILE, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)

    write_header = not JOURNAL_FILE.exists() or JOURNAL_FILE.stat().st_size == 0
    with open(JOURNAL_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({
            "id": pos.id,
            "mint": pos.mint,
            "name": pos.name,
            "symbol": pos.symbol,
            "strategy": pos.strategy,
            "entry_price_sol": pos.entry_price_sol,
            "entry_time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(pos.entry_time)),
            "size_usd": pos.size_usd,
            "exit_price_sol": pos.exit_price_sol,
            "exit_time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(pos.exit_time)) if pos.exit_time else "",
            "exit_reason": pos.exit_reason,
            "pnl_usd": round(pos.pnl_usd, 4),
            "pnl_pct": round(pos.pnl_pct * 100, 2),
            "peak_price_sol": pos.peak_price_sol,
            "market_cap_sol_entry": pos.market_cap_sol_entry,
            "initial_buy_sol": pos.initial_buy_sol,
            "has_twitter": pos.has_twitter,
            "has_telegram": pos.has_telegram,
            "has_website": pos.has_website,
            "deployer": pos.deployer,
            "notes": pos.notes,
            "config_tag": CONFIG_TAG,
        })


# ---------------------------------------------------------------------------
# Portfolio manager
# ---------------------------------------------------------------------------

class SniperPortfolio:

    def __init__(self):
        self._positions: dict[str, SnipePosition] = _load_positions()

    def open_position(self, event, price_sol: float) -> SnipePosition:
        size = SNIPE_SIZE_MIGRATION if event.event_type == "migration" else SNIPE_SIZE_LAUNCH
        pos = SnipePosition(
            id=str(uuid.uuid4())[:8],
            mint=event.mint,
            name=event.name,
            symbol=event.symbol,
            strategy=event.event_type,
            deployer=event.trader,
            entry_price_sol=price_sol,
            entry_time=time.time(),
            size_usd=size,
            market_cap_sol_entry=event.market_cap_sol,
            initial_buy_sol=event.initial_buy_sol,
            has_twitter=event.has_twitter,
            has_telegram=event.has_telegram,
            has_website=event.has_website,
            current_price_sol=price_sol,
            peak_price_sol=price_sol,
        )
        self._positions[pos.id] = pos
        _save_positions(self._positions)
        log.info("SNIPE opened  %s/%s  strategy=%s  price=%.8f SOL  $%.2f",
                 pos.symbol, pos.mint[:8], pos.strategy, price_sol, size)
        return pos

    def close_position(self, pos_id: str, reason: str, price_sol: float = 0.0) -> Optional[SnipePosition]:
        pos = self._positions.get(pos_id)
        if not pos or pos.status == "closed":
            return None
        pos.exit_price_sol = price_sol or pos.current_price_sol
        pos.exit_time = time.time()
        pos.exit_reason = reason
        pos.status = "closed"
        _append_journal(pos)
        del self._positions[pos_id]
        _save_positions(self._positions)
        log.info("SNIPE closed  %s  reason=%s  pnl=%.1f%%",
                 pos.symbol, reason, pos.pnl_pct * 100)
        return pos

    def update_prices(self, prices: dict[str, float]) -> list[dict]:
        """
        prices: { mint → current_price_sol }
        Returns list of exit events.
        """
        exits = []
        for pos in list(self._positions.values()):
            if pos.status != "open":
                continue

            price = prices.get(pos.mint)
            if price and price > 0:
                pos.current_price_sol = price
                pos.peak_price_sol = max(pos.peak_price_sol, price)

            gain = pos.pnl_pct
            reason = None

            # 1. Hard stop
            if gain <= pos.hard_stop_pct:
                reason = "hard_stop"

            # 2. Trailing stop
            elif pos.peak_price_sol > 0 and gain >= pos.trail_activates_pct:
                drawdown = (pos.current_price_sol - pos.peak_price_sol) / pos.peak_price_sol
                if drawdown <= pos.trailing_stop_pct:
                    reason = "trailing_stop"

            # 3. Time stop
            elif (time.time() - pos.entry_time) / 60 > pos.time_stop_minutes:
                if gain < TIME_STOP_MIN_GAIN:
                    reason = "time_stop"

            if reason:
                closed = self.close_position(pos.id, reason)
                if closed:
                    exits.append({
                        "pos_id": pos.id,
                        "symbol": pos.symbol,
                        "reason": reason,
                        "pnl_pct": round(pos.pnl_pct * 100, 2),
                        "pnl_usd": round(pos.pnl_usd, 2),
                    })
            else:
                # Take-profit ladder
                for tp_pct, tp_fraction in TP_LEVELS:
                    level_key = f"tp_{int(tp_pct * 100)}"
                    if gain >= tp_pct and level_key not in pos.tp_levels_hit:
                        pos.tp_levels_hit.append(level_key)
                        pos.remaining_fraction -= tp_fraction * pos.remaining_fraction
                        log.info("TP hit %s %s +%.0f%%  sold %.0f%%",
                                 pos.id, pos.symbol, gain * 100, tp_fraction * 100)

        _save_positions(self._positions)
        return exits

    def manual_close(self, pos_id: str) -> Optional[SnipePosition]:
        pos = self._positions.get(pos_id)
        if not pos:
            return None
        return self.close_position(pos_id, "manual", pos.current_price_sol)

    def open_positions(self) -> list[SnipePosition]:
        return [p for p in self._positions.values() if p.status == "open"]

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
