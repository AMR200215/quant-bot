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

T10_VELOCITY_FILE = Path(__file__).parent.parent / "logs" / "t10_velocity.csv"
_T10_FIELDS = [
    "position_id", "signal_id", "token_symbol", "chain", "dex_id",
    "signal_type", "strength", "entry_price", "entry_time",
    "t10_time", "age_at_t10_min",
    "buys_5m_t10", "sells_5m_t10", "buy_sell_ratio_5m_t10",
    "volume_5m_t10", "price_usd_t10", "price_change_5m_t10",
]

from memecoin.config import (
    POSITIONS_FILE, JOURNAL_FILE, LIVE_JOURNAL_FILE, TRAJECTORY_FILE,
    HARD_STOP_PCT, TRAILING_STOP_PCT, TRAIL_ACTIVATES_PCT,
    TIME_STOP_MINUTES, TIME_STOP_MIN_GAIN,
    TP_LEVELS, TRADE_SIZE_USD,
    get_signal_settings,
    LIVE_TRADING,
)
from memecoin.data_client import dex_get_token
from memecoin.candidate_log import promote_to_winners
from app import alerts

log = logging.getLogger(__name__)

JOURNAL_FIELDS = [
    # identity
    "id", "signal_id", "chain", "token_address", "token_symbol",
    "signal_type", "strength",
    # trade — diagnostic timestamps + signal price for measuring execution cost
    "signal_price", "signal_time",   # price/time when signal fired (before any execution)
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
    # session tag — blank = legacy (pre-2026-05-05), otherwise "v2_YYYY-MM-DD"
    "config_tag",
]

# Stamp applied to every trade written from this session onward
CONFIG_TAG = "v7_entry_filters_2026-06-06"


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
    signal_price: float = 0.0    # DexScreener price when signal fired — never overwritten
    signal_time: float = 0.0    # unix seconds when signal fired
    entry_price: float = 0.0
    entry_time: float = 0.0     # unix seconds
    size_usd: float = 0.0       # TRADE_SIZE_USD
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
    sell_attempts: int = 0    # retry counter — if > MAX_SELL_RETRIES give up
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
    t10_logged: bool = False   # True once T+10 buy-velocity snapshot is written
    t30_logged: bool = False   # True once T+30s post-signal price is recorded
    t60_logged: bool = False   # True once T+60s post-signal price is recorded

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


def _ensure_journal_header():
    """Rewrite the journal header if it is stale (schema changed since file was created)."""
    if not JOURNAL_FILE.exists() or JOURNAL_FILE.stat().st_size == 0:
        return
    with open(JOURNAL_FILE, newline="") as f:
        current_header = next(csv.reader(f), [])
    if current_header == JOURNAL_FIELDS:
        return
    # Header is stale — read all rows, rewrite with correct header
    log.warning("Journal header mismatch — migrating %s", JOURNAL_FILE)
    with open(JOURNAL_FILE, newline="") as f:
        rows = list(csv.DictReader(f))
    with open(JOURNAL_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info("Journal header migrated (%d rows preserved)", len(rows))


def _append_journal(pos: Position):
    JOURNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ensure_journal_header()
    write_header = not JOURNAL_FILE.exists() or JOURNAL_FILE.stat().st_size == 0
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
            "signal_price": pos.signal_price,
            "signal_time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(pos.signal_time)) if pos.signal_time else "",
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
            "config_tag": CONFIG_TAG,
        })

    # If this was a live trade, also write to the live journal
    if pos.notes and "live|tx:" in pos.notes:
        write_header = not LIVE_JOURNAL_FILE.exists() or LIVE_JOURNAL_FILE.stat().st_size == 0
        with open(LIVE_JOURNAL_FILE, "a", newline="") as f:
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
                "signal_price": pos.signal_price,
                "signal_time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(pos.signal_time)) if pos.signal_time else "",
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
                "config_tag": CONFIG_TAG,
            })


# ---------------------------------------------------------------------------
# Portfolio manager
# ---------------------------------------------------------------------------

class Portfolio:
    def __init__(self):
        self._positions: dict[str, Position] = _load_positions()
        # In-memory stall tracker: pos_id → {"last_peak": float, "stall_since": float}
        # Not persisted — resets on restart. Fine for short pumpfun trades.
        self._stall_tracker: dict[str, dict] = {}

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
            signal_price=signal.price_usd,   # snapshot at signal fire — never changes
            signal_time=time.time(),
            entry_price=signal.price_usd,
            entry_time=time.time(),
            size_usd=getattr(signal, "_tier1_size", None)
                     or get_signal_settings(signal.signal_type)["trade_size_usd"],
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
        # Pre-graduation tokens on pumpswap experience more jitter before breakout.
        # Widen hard stop to -40% so normal price oscillation doesn't stop us out early.
        if pos.dex_id.lower() == "pumpswap":
            pos.hard_stop_pct = -0.40

        # Live execution gate — only fire for social_alert pumpfun signals
        # ── Paper position (always opened, independent of live) ──────────────
        self._positions[pos.id] = pos
        _save_positions(self._positions)
        log.info("Opened paper position %s  %s/%s @ $%.8f  dex=%s",
                 pos.id, pos.chain, pos.token_symbol, pos.entry_price, pos.dex_id)

        # ── Live position (parallel, independent — social_alert+pump only) ──
        _is_live_signal = (
            LIVE_TRADING
            and signal.signal_type == "social_alert"
            and "pump" in getattr(signal, "dex_id", "").lower()
        )
        if not _is_live_signal and LIVE_TRADING:
            _why = []
            if signal.signal_type != "social_alert":
                _why.append(f"type={signal.signal_type}")
            if "pump" not in getattr(signal, "dex_id", "").lower():
                _why.append(f"dex={getattr(signal, 'dex_id', 'n/a')}")
            if _why:
                log.info("LIVE GATE BLOCKED %s — paper only: %s", signal.token_symbol, ", ".join(_why))
        if _is_live_signal:
            # ── Circuit breaker 1: daily loss limit ──────────────────────────
            daily_loss = self._live_daily_pnl()
            if daily_loss <= -15.0:
                log.warning(
                    "CIRCUIT BREAKER: daily live PnL=$%.2f — skipping live trade for %s",
                    daily_loss, signal.token_symbol,
                )
            # ── Circuit breaker 2: max concurrent live positions ─────────────
            elif self._count_open_live() >= 2:
                log.warning(
                    "CIRCUIT BREAKER: %d live positions already open — skipping %s",
                    self._count_open_live(), signal.token_symbol,
                )
            else:
                self._open_live_position(signal, pos)

        return pos

    def _live_daily_pnl(self) -> float:
        """Sum of today's closed live position PnL from the journal."""
        import csv as _csv
        from datetime import date as _date
        today = str(_date.today())
        total = 0.0
        try:
            with open(LIVE_JOURNAL_FILE) as f:
                for row in _csv.DictReader(f):
                    if row.get("entry_time", "")[:10] == today:
                        total += float(row.get("pnl_usd", 0) or 0)
        except Exception:
            pass
        return total

    def _count_open_live(self) -> int:
        """Count active live positions. Excludes sell_pending (stuck retries)."""
        return sum(
            1 for p in self._positions.values()
            if p.id.startswith("L")
            and p.status == "open"
            and "|sell_pending" not in (p.notes or "")
        )

    def _open_live_position(self, signal, paper_pos: "Position") -> None:
        """
        Fire a real on-chain buy and store a parallel live position.
        Completely independent from the paper position — different ID,
        different size, different journal. They do not know each other exist.
        """
        import uuid as _uuid
        from memecoin.executor import MemeExecutor
        from memecoin.config import get_signal_settings as _gss

        _live_size = _gss(signal.signal_type).get("live_trade_size_usd", paper_pos.size_usd)

        # Build the live position as a separate object
        live_pos = Position(
            id=f"L{str(_uuid.uuid4())[:7]}",   # L-prefix = live, visually distinct
            signal_id=signal.id,
            chain=paper_pos.chain,
            token_address=paper_pos.token_address,
            token_symbol=paper_pos.token_symbol,
            signal_type=paper_pos.signal_type,
            strength=paper_pos.strength,
            whale_count=paper_pos.whale_count,
            whale_tiers=list(paper_pos.whale_tiers),
            whales_involved=list(paper_pos.whales_involved),
            signal_price=paper_pos.signal_price,   # original signal price — never overwritten
            signal_time=paper_pos.signal_time,
            entry_price=paper_pos.entry_price,
            entry_time=paper_pos.entry_time,
            size_usd=_live_size,
            hard_stop_pct=paper_pos.hard_stop_pct,
            trailing_stop_pct=paper_pos.trailing_stop_pct,
            trail_activates_pct=paper_pos.trail_activates_pct,
            time_stop_minutes=paper_pos.time_stop_minutes,
            current_price=paper_pos.current_price,
            peak_price=paper_pos.peak_price,
            price_change_5m=paper_pos.price_change_5m,
            price_change_1h=paper_pos.price_change_1h,
            price_change_6h=paper_pos.price_change_6h,
            buys_5m=paper_pos.buys_5m, sells_5m=paper_pos.sells_5m,
            buys_h1=paper_pos.buys_h1, sells_h1=paper_pos.sells_h1,
            buy_sell_ratio_5m=paper_pos.buy_sell_ratio_5m,
            buy_sell_ratio_h1=paper_pos.buy_sell_ratio_h1,
            volume_5m=paper_pos.volume_5m, volume_h1=paper_pos.volume_h1,
            volume_h6=paper_pos.volume_h6,
            liquidity_usd=paper_pos.liquidity_usd,
            mcap_usd=paper_pos.mcap_usd, fdv=paper_pos.fdv,
            age_minutes=paper_pos.age_minutes,
            safety_score=paper_pos.safety_score,
            momentum_score=paper_pos.momentum_score,
            composite_score=paper_pos.composite_score,
            dex_id=paper_pos.dex_id,
            dexscreener_url=paper_pos.dexscreener_url,
            has_twitter=paper_pos.has_twitter,
            has_telegram=paper_pos.has_telegram,
            has_website=paper_pos.has_website,
            rugcheck_score=paper_pos.rugcheck_score,
            buy_tax=paper_pos.buy_tax, sell_tax=paper_pos.sell_tax,
        )

        try:
            ex = MemeExecutor()
            result = ex.buy(signal.token_address, _live_size, signal.chain,
                            signal_price=paper_pos.signal_price,
                            max_slippage_pct=0.50)
            if result.get("success"):
                fill_price = result.get("fill_price") or live_pos.entry_price
                signal_price = live_pos.entry_price
                # Abort if entry slippage > 40% — we'd need a massive recovery just to
                # break even, and the hard stop would fire at -60%+ from actual fill.
                if signal_price > 0 and fill_price > signal_price * 1.40:
                    log.warning(
                        "LIVE BUY ABORTED %s — fill slippage %.1f%% > 40%% limit  "
                        "fill=%.10f  signal=%.10f",
                        live_pos.token_symbol, (fill_price / signal_price - 1) * 100,
                        fill_price, signal_price,
                    )
                    try:
                        ex2 = MemeExecutor()
                        ex2.sell(live_pos.token_address, _live_size, fill_price, live_pos.chain)
                    except Exception as _e:
                        log.error("Abort-sell failed %s: %s", live_pos.token_symbol, _e)
                    return
                # Anchor stops to actual fill price, not signal detection price.
                # Without this the hard stop fires at -48% from fill even though it
                # is set to -35%, because the fill averaged +24% above signal.
                live_pos.entry_price   = fill_price
                live_pos.current_price = fill_price
                live_pos.peak_price    = fill_price
                live_pos.notes = f"live|tx:{result.get('tx_sig', '')[:12]}|fill:{fill_price:.10f}"
                log.info("LIVE BUY confirmed %s  tx=%s  fill=%.10f",
                         live_pos.token_symbol, result.get("tx_sig","")[:16], result.get("fill_price",0))
                try:
                    from app.alerts import alert_live_buy
                    alert_live_buy(live_pos, result.get("tx_sig",""), result.get("sol_spent", _live_size / 70))
                except Exception:
                    pass
                # Store live position — it will be monitored independently
                self._positions[live_pos.id] = live_pos
                _save_positions(self._positions)
                log.info("Opened LIVE position %s  size=$%.2f  entry=$%.8f",
                         live_pos.id, live_pos.size_usd, live_pos.entry_price)
            elif result.get("unconfirmed"):
                live_pos.notes = f"unconfirmed|tx:{result.get('tx_sig', '')[:12]}"
                self._positions[live_pos.id] = live_pos
                _save_positions(self._positions)
                log.error("LIVE BUY UNCONFIRMED %s — check on-chain: %s",
                          live_pos.token_symbol, result.get("tx_sig", ""))
            else:
                log.error("LIVE BUY failed for %s: %s — paper trade continues independently",
                          live_pos.token_symbol, result.get("error"))
        except RuntimeError as e:
            log.error("LIVE executor error for %s: %s — paper trade continues independently",
                      live_pos.token_symbol, e)

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

        # Live execution gate — only sell on-chain if this position was a live buy
        _was_live_buy = bool(pos.notes and "live|tx:" in pos.notes)
        MAX_SELL_RETRIES = 5
        if LIVE_TRADING and _was_live_buy:
            from memecoin.executor import MemeExecutor
            try:
                ex = MemeExecutor()
                result = ex.sell(pos.token_address, pos.size_usd, pos.entry_price, pos.chain)
                if result.get("success"):
                    fill = result.get("fill_price") or pos.exit_price
                    pos.notes = (pos.notes or "") + f"|sell_tx:{result.get('tx_sig','')[:12]}|sell_fill:{fill:.10f}"
                    log.info("Live sell confirmed %s  tx=%s  fill=%.10f",
                             pos.token_symbol, result.get("tx_sig","")[:16], fill)
                    try:
                        from app.alerts import alert_live_sell
                        alert_live_sell(pos, result.get("sol_received", 0), result.get("tx_sig", ""))
                    except Exception:
                        pass
                elif result.get("reason") == "zero_balance":
                    # Tokens already gone (prev unconfirmed sell went through) — close cleanly
                    log.warning("Live sell %s — zero balance, tokens already sold. Closing.",
                                pos.token_symbol)
                    pos.notes = (pos.notes or "") + "|sell_already_gone"
                else:
                    # Sell failed or unconfirmed — retry up to MAX_SELL_RETRIES
                    pos.sell_attempts = getattr(pos, "sell_attempts", 0) + 1
                    reason_tag = "sell_unconf" if result.get("unconfirmed") else "sell_failed"
                    tx_tag = f":{result.get('tx_sig','')[:12]}" if result.get("tx_sig") else ""
                    pos.notes = (pos.notes or "") + f"|{reason_tag}{tx_tag}(attempt {pos.sell_attempts})"
                    if pos.sell_attempts < MAX_SELL_RETRIES:
                        pos.status = "open"
                        pos.exit_price  = 0.0
                        pos.exit_time   = 0.0
                        pos.exit_reason = ""
                        # Mark as sell_pending so _count_open_live excludes it from the cap
                        if "|sell_pending" not in (pos.notes or ""):
                            pos.notes = (pos.notes or "") + "|sell_pending"
                        self._positions[pos_id] = pos
                        _save_positions(self._positions)
                        log.error("Live sell %s for %s (attempt %d/%d) — retrying next cycle. err=%s",
                                  reason_tag, pos.token_symbol, pos.sell_attempts, MAX_SELL_RETRIES,
                                  result.get("error") or result.get("tx_sig",""))
                        return pos
                    else:
                        log.error("Live sell GAVE UP after %d attempts for %s — TOKENS MAY REMAIN IN WALLET",
                                  MAX_SELL_RETRIES, pos.token_symbol)
                        try:
                            from app.alerts import _send
                            _send(f"ALERT: sell gave up after {MAX_SELL_RETRIES} retries for {pos.token_symbol} — check wallet manually")
                        except Exception:
                            pass
            except Exception as e:
                pos.sell_attempts = getattr(pos, "sell_attempts", 0) + 1
                pos.notes = (pos.notes or "") + f"|sell_error(attempt {pos.sell_attempts})"
                if pos.sell_attempts < MAX_SELL_RETRIES:
                    pos.status = "open"
                    pos.exit_price  = 0.0
                    pos.exit_time   = 0.0
                    pos.exit_reason = ""
                    if "|sell_pending" not in (pos.notes or ""):
                        pos.notes = (pos.notes or "") + "|sell_pending"
                    self._positions[pos_id] = pos
                    _save_positions(self._positions)
                    log.error("Executor error during sell for %s (attempt %d/%d): %s — retrying",
                              pos.token_symbol, pos.sell_attempts, MAX_SELL_RETRIES, e)
                    return pos
                log.error("Live sell GAVE UP after %d attempts for %s: %s",
                          MAX_SELL_RETRIES, pos.token_symbol, e)

        _append_journal(pos)
        promote_to_winners(pos)
        del self._positions[pos_id]
        _save_positions(self._positions)
        log.info("Closed position %s  reason=%s  pnl=%.1f%%",
                 pos_id, reason, pos.pnl_pct * 100)
        try:
            alerts.alert_position_close(pos)
        except Exception:
            pass
        return pos

    # ---- update prices & evaluate exit conditions ----

    def update_prices(self, whale_sells: dict[str, list[str]] = None,
                      price_overrides: dict[str, float] = None) -> list[dict]:
        """
        Fetch current prices for all open positions, evaluate exit conditions.

        whale_sells:     { token_address: [wallet1, wallet2] } — wallets that just sold
        price_overrides: { token_address: price_usd } — real-time prices from PumpPortal.
                         When present and fresh, these replace DexScreener for that token.

        Returns list of exit events: [{"pos_id", "reason", "pnl_pct"}]
        """
        if whale_sells is None:
            whale_sells = {}
        if price_overrides is None:
            price_overrides = {}

        exits = []
        for pos in list(self._positions.values()):
            if pos.status != "open":
                continue

            # ── Price source priority ─────────────────────────────────────────
            # 1. PumpPortal real-time (sub-second, from bonding curve reserves)
            # 2. DexScreener poll (5-30s lag — fallback heartbeat)
            # 3. Jupiter quote (last resort when DexScreener is down)
            pp_price = price_overrides.get(pos.token_address)
            if pp_price and pp_price > 0:
                pos.current_price = pp_price
                pos.peak_price = max(pos.peak_price, pp_price)
            else:
                # DexScreener fallback — used when PumpPortal has no fresh data
                # (graduated tokens, or token not yet subscribed)
                pair = dex_get_token(pos.chain, pos.token_address)
                if pair:
                    price = float(pair.get("priceUsd") or 0)
                    if price > 0:
                        pos.current_price = price
                        pos.peak_price = max(pos.peak_price, price)
                if (not pair or pos.current_price == 0) and pos.chain == "solana":
                    try:
                        from memecoin.executor import _get_quote, _sol_price_usd, SOL_MINT, SOL_DECIMALS
                        _sol = _sol_price_usd()
                        _q   = _get_quote(SOL_MINT, pos.token_address, int(pos.size_usd / _sol * 10**SOL_DECIMALS))
                        _decimals = int(_q.get("outputDecimals") or 6)
                        _tokens   = int(_q["outAmount"]) / (10 ** _decimals)
                        _price    = pos.size_usd / _tokens if _tokens > 0 else 0
                        if _price > 0:
                            pos.current_price = _price
                            pos.peak_price = max(pos.peak_price, _price)
                            log.warning("DexScreener unavailable — using Jupiter quote price for %s: $%.10f",
                                        pos.token_symbol, _price)
                    except Exception:
                        pass   # both sources failed — price stays stale, time stop will still fire

            # T+10 buy-velocity snapshot (8–12 min window, logged once per position)
            age_min = (time.time() - pos.entry_time) / 60
            if not pos.t10_logged and 8 <= age_min <= 12 and pair:
                txns = pair.get("txns") or {}
                m5   = txns.get("m5") or {}
                vol  = pair.get("volume") or {}
                pc   = pair.get("priceChange") or {}
                b5   = int(m5.get("buys") or 0)
                s5   = int(m5.get("sells") or 0)
                bsr5 = round(b5 / (b5 + s5), 3) if (b5 + s5) else 0.0
                T10_VELOCITY_FILE.parent.mkdir(parents=True, exist_ok=True)
                write_hdr = not T10_VELOCITY_FILE.exists()
                with open(T10_VELOCITY_FILE, "a", newline="") as fv:
                    wv = csv.DictWriter(fv, fieldnames=_T10_FIELDS)
                    if write_hdr:
                        wv.writeheader()
                    wv.writerow({
                        "position_id":          pos.id,
                        "signal_id":            pos.signal_id,
                        "token_symbol":         pos.token_symbol,
                        "chain":                pos.chain,
                        "dex_id":               pos.dex_id,
                        "signal_type":          pos.signal_type,
                        "strength":             pos.strength,
                        "entry_price":          pos.entry_price,
                        "entry_time":           time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(pos.entry_time)),
                        "t10_time":             time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                        "age_at_t10_min":       round(age_min, 1),
                        "buys_5m_t10":          b5,
                        "sells_5m_t10":         s5,
                        "buy_sell_ratio_5m_t10": bsr5,
                        "volume_5m_t10":        float(vol.get("m5") or 0),
                        "price_usd_t10":        pos.current_price,
                        "price_change_5m_t10":  float(pc.get("m5") or 0),
                    })
                pos.t10_logged = True
                log.debug("T+10 velocity logged for %s  bsr5m=%.2f  buys=%d", pos.id, bsr5, b5)

            # T+30s / T+60s post-signal price trajectory
            # Answers Opus's diagnostic question: does price continue after signal?
            # Applies to ALL positions (paper + live) so we get clean data on winners too.
            if pos.signal_time > 0:
                elapsed_signal = time.time() - pos.signal_time
                for label, threshold, already_logged, set_flag in [
                    ("t30",  30,  pos.t30_logged, "t30_logged"),
                    ("t60",  60,  pos.t60_logged, "t60_logged"),
                ]:
                    if not already_logged and elapsed_signal >= threshold:
                        gain_from_signal = (
                            (pos.current_price - pos.signal_price) / pos.signal_price
                            if pos.signal_price > 0 else 0.0
                        )
                        TRAJECTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
                        write_hdr = not TRAJECTORY_FILE.exists() or TRAJECTORY_FILE.stat().st_size == 0
                        with open(TRAJECTORY_FILE, "a", newline="") as _tf:
                            _tw = csv.DictWriter(_tf, fieldnames=[
                                "pos_id", "signal_id", "token_symbol", "chain",
                                "signal_type", "strength", "is_live",
                                "signal_price", "signal_time",
                                "snapshot_label", "snapshot_time", "snapshot_price",
                                "gain_from_signal_pct",
                            ])
                            if write_hdr:
                                _tw.writeheader()
                            _tw.writerow({
                                "pos_id":               pos.id,
                                "signal_id":            pos.signal_id,
                                "token_symbol":         pos.token_symbol,
                                "chain":                pos.chain,
                                "signal_type":          pos.signal_type,
                                "strength":             pos.strength,
                                "is_live":              "live|tx:" in (pos.notes or ""),
                                "signal_price":         pos.signal_price,
                                "signal_time":          time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(pos.signal_time)),
                                "snapshot_label":       label,
                                "snapshot_time":        time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                                "snapshot_price":       pos.current_price,
                                "gain_from_signal_pct": round(gain_from_signal * 100, 2),
                            })
                        setattr(pos, set_flag, True)
                        log.debug("%s trajectory %s  gain_from_signal=%.1f%%",
                                  pos.token_symbol, label, gain_from_signal * 100)

            gain = pos.pnl_pct
            reason = None

            # Update stall tracker — reset timer whenever peak_price improves
            stall = self._stall_tracker.setdefault(
                pos.id, {"last_peak": pos.peak_price, "stall_since": time.time()}
            )
            if pos.peak_price > stall["last_peak"]:
                stall["last_peak"] = pos.peak_price
                stall["stall_since"] = time.time()

            # 0. Profit-lock on stall: if we're in small profit and the peak
            #    hasn't moved for N seconds, exit before momentum fully dies.
            #    Skip for big runners (gain > max_gain) — let trail handle those.
            from memecoin.config import get_signal_settings as _gss_pl
            _pl = _gss_pl(pos.signal_type)
            _pl_min  = _pl.get("profit_lock_min_gain",  0.05)
            _pl_max  = _pl.get("profit_lock_max_gain",  0.30)
            _pl_sec  = _pl.get("profit_lock_stall_sec", 999999)  # default: disabled
            if (_pl_min <= gain <= _pl_max and
                    (time.time() - stall["stall_since"]) >= _pl_sec):
                reason = "profit_lock"

            # 1. Hard stop
            if not reason and gain <= pos.hard_stop_pct:
                reason = "hard_stop"

            # 2. Trailing stop
            elif not reason and pos.peak_price > 0 and gain >= pos.trail_activates_pct:
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
                        try:
                            alerts.alert_tp_hit(pos, tp_pct, partial_usd)
                        except Exception:
                            pass

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
