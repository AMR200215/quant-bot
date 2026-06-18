"""
Paper trade portfolio for the memecoin module.

Tracks open positions, evaluates exit conditions on every price update,
and writes closed trades to the journal CSV.
"""

import csv
import json
import logging
import threading
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
    POSITIONS_FILE, JOURNAL_FILE, SOCIAL_JOURNAL_FILE, LIVE_JOURNAL_FILE, TRAJECTORY_FILE,
    HARD_STOP_PCT, TRAILING_STOP_PCT, TRAIL_ACTIVATES_PCT,
    TIME_STOP_MINUTES, TIME_STOP_MIN_GAIN,
    TP_LEVELS, TRADE_SIZE_USD,
    get_signal_settings,
    LIVE_TRADING,
    DAILY_LOSS_LIMIT,
    LIVE_DRY_RUN,
    REALTIME_PRICE_FEED,
    SLIPPAGE_GATE_RT_PCT, SLIPPAGE_GATE_DEX_PCT,
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
    "pnl_usd", "pnl_pct", "peak_price", "hard_stop_pct",
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
    # accounting v2 fields (added 2026-06-11)
    "tp_levels_hit",         # comma-joined list of TP keys hit before close, e.g. "tp_100,tp_300"
    "realized_partial_usd",  # USD locked in from partial TP sells (before final close)
    "remaining_fraction",    # fraction of original position closed at exit_price
    "accounting_epoch",      # which accounting logic produced this row
]

# Stamp applied to every trade written from this session onward
CONFIG_TAG = "v7_entry_filters_2026-06-06"

# Accounting epoch — bump when position accounting logic changes so we can
# split reports cleanly.  Past rows are backfilled by tools/v7_journal_corrected.py.
# e1_baseline             : pre-2026-06-11 01:20 UTC (no PP exits, simple pnl)
# e2_pp_exits             : commit 81de8da — PP real-time exits wired to paper
# e3_pp_entries_anchored_stops: commit 9a2a332 — signal-anchored stops + this accounting fix
ACCOUNTING_EPOCH = "e4_rt_feed_quote_gate"


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
    realized_pnl_usd: float = 0.0   # locked-in USD from partial TP sells
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
    creator_wallet: str = ""   # token deployer — triggers dev_dump exit if they sell
    tokens_held: int = 0       # raw token count from buy tx delta — used for known-balance TP sells

    @property
    def pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        ref = self.exit_price if self.status == "closed" else self.current_price
        return (ref - self.entry_price) / self.entry_price

    @property
    def pnl_usd(self) -> float:
        # realized_pnl_usd: locked-in profit from partial TP sells
        # remaining portion: pnl_pct on whatever fraction is still open/being closed
        return self.realized_pnl_usd + self.pnl_pct * self.size_usd * self.remaining_fraction


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


def _ensure_journal_header(path=None):
    """Rewrite the journal header if it is stale (schema changed since file was created)."""
    target = path or JOURNAL_FILE
    if not target.exists() or target.stat().st_size == 0:
        return
    with open(target, newline="") as f:
        current_header = next(csv.reader(f), [])
    if current_header == JOURNAL_FIELDS:
        return
    # Header is stale — read all rows, rewrite with correct header
    log.warning("Journal header mismatch — migrating %s", target)
    with open(target, newline="") as f:
        rows = list(csv.DictReader(f))
    with open(target, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info("Journal header migrated (%d rows preserved)", len(rows))


def _build_journal_row(pos: Position) -> dict:
    """Build the canonical journal dict for a closed position."""
    return {
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
        "hard_stop_pct": pos.hard_stop_pct,
        "notes": pos.notes,
        "config_tag": CONFIG_TAG,
        # accounting v2
        "tp_levels_hit": ",".join(pos.tp_levels_hit),
        "realized_partial_usd": round(pos.realized_pnl_usd, 4),
        "remaining_fraction": round(pos.remaining_fraction, 4),
        "accounting_epoch": ACCOUNTING_EPOCH,
    }


def _append_journal(pos: Position):
    JOURNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    row = _build_journal_row(pos)

    # Route to split journals: social_alert → social journal, everything else → main journal
    _is_social = pos.signal_type == "social_alert"
    target = SOCIAL_JOURNAL_FILE if _is_social else JOURNAL_FILE

    _ensure_journal_header(target)
    write_header = not target.exists() or target.stat().st_size == 0
    with open(target, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    # If this was a live trade, also write to the live journal
    if pos.notes and "live|tx:" in pos.notes:
        _ensure_journal_header(LIVE_JOURNAL_FILE)
        write_header = not LIVE_JOURNAL_FILE.exists() or LIVE_JOURNAL_FILE.stat().st_size == 0
        with open(LIVE_JOURNAL_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        # DRY_RUN funnel counter — notify when threshold crossed
        if "DRY_RUN" in (pos.notes or ""):
            try:
                count = 0
                with open(LIVE_JOURNAL_FILE) as _f:
                    for _r in csv.DictReader(_f):
                        if "DRY_RUN" in (_r.get("notes") or ""):
                            count += 1
                if count >= 3:
                    from app.alerts import _send
                    _send(
                        f"✅ DRY_RUN funnel: {count} signals through live path\n"
                        f"Latest: {pos.token_symbol} ({pos.signal_type})\n"
                        f"Ready for your go-live decision."
                    )
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Portfolio manager
# ---------------------------------------------------------------------------

class Portfolio:
    def __init__(self):
        self._positions: dict[str, Position] = _load_positions()
        # In-memory stall tracker: pos_id → {"last_peak": float, "stall_since": float}
        # Not persisted — resets on restart. Fine for short pumpfun trades.
        self._stall_tracker: dict[str, dict] = {}

        # ── Pre-signed emergency exits ─────────────────────────────────────
        # mint → signed tx bytes (step-3 ladder: 98% slippage, 0.005 SOL fee)
        # Refreshed every 45s by _presigned_refresh_loop.
        # Only used for rug-path exits (dev_dump, rug_lp, velocity) where
        # the 300-500ms build step matters most.  Orderly exits (time_stop,
        # trailing_stop) still build on demand via the normal ladder.
        self._presigned_exits: dict = {}   # mint → bytes
        self._presigned_ts:    dict = {}   # mint → last-sign time (float)
        self._presigned_lock        = threading.Lock()
        self._graduated_mints: set  = set()  # mints that traded on Raydium (not bonding curve)
        # sell_stuck throttle: pos_id → earliest time to retry sell
        # In-memory only — resets on restart (which itself gives a fresh attempt).
        self._sell_stuck_until: dict[str, float] = {}
        # dex_pair_loss tracking: pos_id → timestamp when Jupiter fallback started
        self._jup_fallback_since: dict[str, float] = {}
        # how many positions had PP or DexScreener data last cycle (>0 = feeds healthy)
        self._last_cycle_n_with_dex: int = 0
        threading.Thread(
            target=self._presigned_refresh_loop,
            daemon=True,
            name="presign-refresh",
        ).start()

    # ---- pre-signed emergency exit management ----

    def _build_presigned_exit(self, mint: str, is_graduated: bool = False) -> None:
        """
        Build and sign the emergency sell tx for a held mint and store it.

        Bonding-curve tokens (is_graduated=False):
          PumpPortal 98% slippage, 0.005 SOL fee.  PumpPortal re-fetches the
          latest blockhash server-side so the tx stays valid through 45s refresh.

        Graduated tokens (is_graduated=True):
          PumpPortal rejects these with Custom:6005.  Build Jupiter 99% slippage
          tx instead.  Jupiter embeds a blockhash that expires in ~60s, so these
          are refreshed every 30s (vs 45s for PumpPortal).
        """
        try:
            from memecoin.executor import (
                _get_keypair, _load_solders, _pumpportal_build_tx,
                _jup_get_quote, _jup_build_swap_tx, _helius_priority_fee,
                _token_balance, SOL_MINT, EXECUTOR_BACKEND,
            )
            if EXECUTOR_BACKEND != "pumpportal":
                return
            _t0 = time.time()
            _, VersionedTransaction, _ = _load_solders()
            keypair = _get_keypair()
            wallet  = str(keypair.pubkey())

            if is_graduated:
                # Graduated token — PumpPortal will reject with 6005.
                # Build a Jupiter presigned exit at 99% slippage instead.
                try:
                    balance = _token_balance(wallet, mint)
                    if not balance:
                        log.debug("Presigned Jupiter exit skipped — zero balance  mint=%s", mint[:8])
                        return
                    _jup_fee = max(_helius_priority_fee(mint, "UnsafeMax"), 0.005)
                    quote    = _jup_get_quote(mint, SOL_MINT, balance)
                    tx_bytes = _jup_build_swap_tx(
                        quote, wallet,
                        slippage_bps=9900,
                        priority_fee_lamports=int(_jup_fee * 1e9),
                    )
                    _path = "Jupiter"
                except Exception as _jup_e:
                    log.warning("Presigned Jupiter exit build failed for %s: %s", mint[:8], _jup_e)
                    return
            else:
                tx_bytes = _pumpportal_build_tx(
                    wallet_pubkey=wallet,
                    action="sell",
                    token_mint=mint,
                    amount="100%",
                    denominated_in_sol=False,
                    slippage_pct=98,
                    priority_fee_sol=0.005,
                )
                _path = "PumpPortal"

            tx        = VersionedTransaction.from_bytes(tx_bytes)
            signed    = VersionedTransaction(tx.message, [keypair])
            signed_b  = bytes(signed)
            with self._presigned_lock:
                self._presigned_exits[mint] = signed_b
                self._presigned_ts[mint]    = time.time()
            log.info(
                "Presigned exit built  mint=%s  path=%s  build_ms=%.0f",
                mint[:8], _path, (time.time() - _t0) * 1000,
            )
        except Exception as e:
            log.warning("Presigned exit build failed for %s: %s", mint[:8], e)

    def _schedule_presigned_exit(self, mint: str, is_graduated: bool = False) -> None:
        """Build presigned exit immediately (non-blocking) after live buy confirms."""
        if is_graduated:
            with self._presigned_lock:
                self._graduated_mints.add(mint)
        threading.Thread(
            target=self._build_presigned_exit,
            args=(mint, is_graduated),
            daemon=True,
            name=f"presign-build-{mint[:8]}",
        ).start()

    def _presigned_refresh_loop(self) -> None:
        """Refresh presigned exits periodically.

        PumpPortal exits: every 45s (server-side blockhash refresh, ~90s TTL).
        Jupiter exits:    every 30s (embedded blockhash expires in ~60s).
        """
        _last_refresh = 0.0
        while True:
            time.sleep(10)
            now = time.time()
            with self._presigned_lock:
                mints    = list(self._presigned_exits.keys())
                grad_set = set(self._graduated_mints)
            for mint in mints:
                is_grad  = mint in grad_set
                interval = 30 if is_grad else 45
                with self._presigned_lock:
                    last_ts = self._presigned_ts.get(mint, 0)
                if now - last_ts >= interval:
                    self._build_presigned_exit(mint, is_graduated=is_grad)

    # ---- open ----

    def open_position(self, signal) -> Position:
        """Open a paper position from a Signal object."""
        # Determine the signal price baseline.
        # REALTIME_PRICE_FEED=True: use PP live price captured after screening —
        # eliminates DexScreener indexer lag (~15-30%) from stop anchor and PnL.
        # Falls back to DexScreener price if PP had no tick yet (safe default).
        _sig_price_pp  = getattr(signal, "_price_pp", 0) or 0
        _use_pp        = REALTIME_PRICE_FEED and _sig_price_pp > 0
        _baseline_price = _sig_price_pp if _use_pp else signal.price_usd

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
            signal_price=_baseline_price,   # PP live price (or DexScreener fallback) — stop anchor
            signal_time=time.time(),
            entry_price=_baseline_price,
            entry_time=time.time(),
            size_usd=getattr(signal, "_tier1_size", None)
                     or get_signal_settings(signal.signal_type)["trade_size_usd"],
            hard_stop_pct=get_signal_settings(signal.signal_type)["hard_stop_pct"],
            trailing_stop_pct=get_signal_settings(signal.signal_type)["trailing_stop_pct"],
            trail_activates_pct=get_signal_settings(signal.signal_type)["trail_activates_pct"],
            time_stop_minutes=get_signal_settings(signal.signal_type)["time_stop_minutes"],
            current_price=_baseline_price,
            peak_price=_baseline_price,
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
            creator_wallet=getattr(signal, "creator_wallet", ""),
        )
        # Pre-graduation tokens on pumpswap experience more jitter before breakout.
        # Widen hard stop to -40% so normal price oscillation doesn't stop us out early.
        if pos.dex_id.lower() == "pumpswap":
            pos.hard_stop_pct = -0.40

        # Live execution gate — fire for pumpfun_stream + telegram_pump cohorts
        # ── Paper position (always opened, independent of live) ──────────────
        # For pump cohort signals: use live PP price as entry rather than stale
        # DexScreener snapshot.  If PP monitor has no price yet keep the signal
        # price — the paper twin will be rebased to fill price once live buys.
        _token_cohort = getattr(signal, "token_cohort", "")
        if _token_cohort in ("pumpfun_stream", "telegram_pump"):
            try:
                from memecoin import pumpportal_monitor as _ppm_entry
                _ppm_price = _ppm_entry.monitor.get_prices().get(signal.token_address, 0)
                if _ppm_price > 0:
                    pos.entry_price   = _ppm_price
                    pos.current_price = _ppm_price
                    pos.peak_price    = _ppm_price
            except Exception:
                pass

        self._positions[pos.id] = pos
        _save_positions(self._positions)
        log.info("Opened paper position %s  %s/%s @ $%.8f  dex=%s",
                 pos.id, pos.chain, pos.token_symbol, pos.entry_price, pos.dex_id)

        # ── Live position (parallel, independent — social_alert only) ──
        # Only social_alert (telegram_pump cohort) goes live.
        # pumpportal_screen, copy_trade, vol_breakout, new_launch, dev_launch → paper only.
        _is_live_signal = (
            LIVE_TRADING
            and signal.signal_type == "social_alert"
            and _token_cohort == "telegram_pump"
        )
        if not _is_live_signal and LIVE_TRADING:
            _why = []
            if signal.signal_type != "social_alert":
                _why.append(f"type={signal.signal_type}")
            elif _token_cohort != "telegram_pump":
                _why.append(f"cohort={_token_cohort}")
            if _why:
                log.info("LIVE GATE BLOCKED %s — paper only: %s", signal.token_symbol, ", ".join(_why))
        if _is_live_signal:
            # ── Cohort auto-gate ─────────────────────────────────────────────
            # After 50 live trades for this cohort: if net PnL < 0, close gate.
            _cohort_key  = _token_cohort or signal.signal_type
            _cohort_data = self.live_cohort_stats().get(_cohort_key, {})
            if _cohort_data.get("trade_count", 0) >= 50 and _cohort_data.get("net_pnl_usd", 0.0) < 0:
                log.warning(
                    "COHORT LIVE GATE CLOSED [%s] — %d trades net=$%.2f — paper only",
                    _cohort_key, _cohort_data["trade_count"], _cohort_data["net_pnl_usd"],
                )
                _is_live_signal = False
        if _is_live_signal:
            # ── Circuit breaker 1: daily loss limit ──────────────────────────
            daily_loss = self._live_daily_pnl()
            if daily_loss <= DAILY_LOSS_LIMIT:
                log.warning(
                    "CIRCUIT BREAKER: daily live PnL=$%.2f — skipping live trade for %s",
                    daily_loss, signal.token_symbol,
                )
                try:
                    from memecoin.gate_logger import log_gate_block as _lgb
                    _lgb("breaker", signal.chain, signal.token_address,
                         signal.token_symbol, pp_price=0.0,
                         signal_price=signal.price_usd or 0,
                         size_usd=get_signal_settings(signal.signal_type).get("live_trade_size_usd", pos.size_usd))
                except Exception:
                    pass
            # ── Circuit breaker 2: max concurrent live positions ─────────────
            elif self._count_open_live() >= 2:
                log.warning(
                    "CIRCUIT BREAKER: %d live positions already open — skipping %s",
                    self._count_open_live(), signal.token_symbol,
                )
                try:
                    from memecoin.gate_logger import log_gate_block as _lgb
                    _lgb("breaker", signal.chain, signal.token_address,
                         signal.token_symbol, pp_price=0.0,
                         signal_price=signal.price_usd or 0,
                         size_usd=get_signal_settings(signal.signal_type).get("live_trade_size_usd", pos.size_usd))
                except Exception:
                    pass
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

    def live_cohort_stats(self) -> dict:
        """
        Per-signal_type live trade stats from live_journal.csv.
        Returns {signal_type: {trade_count, gross_pnl_pct, net_pnl_usd}}.
        Used by the type-2 auto-gate and for reporting.
        """
        import csv as _csv
        stats: dict = {}
        try:
            with open(LIVE_JOURNAL_FILE) as f:
                for row in _csv.DictReader(f):
                    st = row.get("signal_type") or "unknown"
                    if st not in stats:
                        stats[st] = {"trade_count": 0, "gross_pnl_pct": 0.0, "net_pnl_usd": 0.0}
                    stats[st]["trade_count"]   += 1
                    stats[st]["gross_pnl_pct"] += float(row.get("pnl_pct", 0) or 0)
                    stats[st]["net_pnl_usd"]   += float(row.get("pnl_usd", 0) or 0)
        except Exception:
            pass
        return stats

    def screening_confirmation_rate(
        self,
        hours: float = 24.0,
        signal_type: str = "social_alert",
    ) -> dict:
        """
        Report the fraction of type-1/type-2 signals that turn into open positions.

        Reads the live journal (confirmed trades) and compares against the
        signal candidates log (every signal that was screened).  Returns a dict:

          {
            "window_hours":   24.0,
            "signal_type":    "social_alert",
            "screened":       N,   # signals that passed screening
            "confirmed":      M,   # live positions opened (from journal)
            "rate_pct":       R,   # M / N * 100
            "target_per_day": T,   # extrapolated confirms/day
          }

        A rate of 0% means nothing is reaching the executor.
        A typical healthy rate is 5-20% (most signals blocked by gates).
        """
        import csv as _csv
        from datetime import datetime as _dt, timezone as _tz

        cutoff = time.time() - hours * 3600

        # Count screened candidates from signal_candidates.csv
        screened = 0
        try:
            from memecoin.config import CANDIDATES_FILE
            with open(CANDIDATES_FILE) as f:
                for row in _csv.DictReader(f):
                    ts_str = row.get("timestamp", "") or row.get("signal_time", "")
                    try:
                        ts = _dt.fromisoformat(ts_str).timestamp()
                    except Exception:
                        continue
                    if ts >= cutoff and row.get("signal_type", "") == signal_type:
                        screened += 1
        except FileNotFoundError:
            pass

        # Count confirmed live positions from live_journal.csv
        confirmed = 0
        try:
            with open(LIVE_JOURNAL_FILE) as f:
                for row in _csv.DictReader(f):
                    ts_str = row.get("entry_time", "")
                    try:
                        ts = _dt.fromisoformat(ts_str).timestamp()
                    except Exception:
                        continue
                    if ts >= cutoff and row.get("signal_type", "") == signal_type:
                        confirmed += 1
        except FileNotFoundError:
            pass

        rate_pct     = (confirmed / screened * 100) if screened > 0 else 0.0
        day_scale    = 24.0 / hours if hours > 0 else 1.0
        target_per_day = round(confirmed * day_scale, 1)

        return {
            "window_hours":   hours,
            "signal_type":    signal_type,
            "screened":       screened,
            "confirmed":      confirmed,
            "rate_pct":       round(rate_pct, 1),
            "target_per_day": target_per_day,
        }

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
            # ── PumpPortal pre-flight ─────────────────────────────────────────────
            # Behaviour differs by signal price source:
            #
            # source == "pp"  (type-1, pumpfun_stream — seconds-old token):
            #   Poll PP 2s for a live price. Silence = dead token → fail-closed.
            #   Price present → PP-to-PP drift gate (SLIPPAGE_GATE_RT_PCT, 20%).
            #
            # source == "dex" (type-2, telegram_pump — hours-old token):
            #   PP won't accumulate ticks in 2s for an established token.
            #   Skip the poll and the no-price block entirely. PP subscription
            #   is still set so monitoring ticks begin arriving after fill.
            #   Drift enforcement is deferred to the Jupiter quote in executor:
            #   quote > signal_price × (1 + SLIPPAGE_GATE_DEX_PCT) → blocked_quote_drift.
            _price_source      = getattr(signal, "_price_source", "dex")
            _pf_blocked        = False
            _pp_price          = 0.0
            _pp_at_gate        = 0.0   # PP price captured at gate time (dex-source path)
            _exec_signal_price = paper_pos.signal_price  # baseline passed to executor
            try:
                from memecoin import pumpportal_monitor as _pp_monitor
                _pp        = _pp_monitor.monitor
                _mint      = signal.token_address
                _sig_price = paper_pos.signal_price

                # Always subscribe so monitoring ticks start arriving immediately
                _pp.subscribe({_mint})

                if _price_source == "pp":
                    # ── Type-1 path: poll for live PP price, fail-closed if silent ──
                    _pp_price = 0.0
                    _pf_deadline = time.time() + 2.0
                    while time.time() < _pf_deadline:
                        _p = _pp.get_prices().get(_mint, 0)
                        if _p > 0:
                            _pp_price = _p
                            break
                        time.sleep(0.1)

                    try:
                        from memecoin.health_monitor import bump_preflight_attempt as _bpa
                        _bpa()
                    except Exception:
                        pass

                    if _pp_price == 0:
                        try:
                            from memecoin.health_monitor import bump_preflight_no_price as _bpnp
                            _bpnp()
                        except Exception:
                            pass
                        log.warning(
                            "LIVE PREFLIGHT NO PRICE %s — PP returned no price in 2s, "
                            "blocking trade (fail-closed, type-1)",
                            live_pos.token_symbol,
                        )
                        try:
                            from app.alerts import _send
                            _send(
                                f"🚫 PREFLIGHT NO PRICE {live_pos.token_symbol} — "
                                f"PP silent 2s. Trade blocked, no SOL spent."
                            )
                        except Exception:
                            pass
                        try:
                            from memecoin.gate_logger import log_gate_block as _lgb
                            _lgb("preflight_no_price", live_pos.chain, live_pos.token_address,
                                 live_pos.token_symbol, pp_price=0.0,
                                 signal_price=_sig_price or 0,
                                 size_usd=_live_size)
                        except Exception:
                            pass
                        _pf_blocked = True

                    else:
                        # PP-to-PP drift gate (20% — measures genuine movement only)
                        _gate = SLIPPAGE_GATE_RT_PCT
                        if _sig_price and _pp_price > _sig_price * (1 + _gate):
                            _pf_slip = (_pp_price / _sig_price - 1) * 100
                            log.warning(
                                "LIVE PREFLIGHT BLOCKED %s — PP %.10f is %.1f%% above signal "
                                "%.10f (>%.0f%% pp gate)",
                                live_pos.token_symbol, _pp_price, _pf_slip, _sig_price,
                                _gate * 100,
                            )
                            try:
                                from app.alerts import _send
                                _send(
                                    f"🚫 PREFLIGHT BLOCKED {live_pos.token_symbol} — "
                                    f"PP ${_pp_price:.8f} already {_pf_slip:.1f}% above signal "
                                    f"${_sig_price:.8f} (>{_gate*100:.0f}% pp gate). "
                                    f"No SOL spent."
                                )
                            except Exception:
                                pass
                            try:
                                from memecoin.gate_logger import log_gate_block as _lgb
                                _lgb("preflight_price", live_pos.chain, live_pos.token_address,
                                     live_pos.token_symbol, pp_price=_pp_price,
                                     signal_price=_sig_price or 0,
                                     size_usd=_live_size)
                            except Exception:
                                pass
                            _pf_blocked = True

                else:
                    # ── Type-2 path (dex source): wait up to 2s for PP tick ──────
                    # PP was subscribed above. For a fresh pump.fun token the first
                    # trade event often arrives within 1-2s of subscribe. If we get
                    # one, upgrade to same-venue comparison (PP_at_gate ÷ Jupiter)
                    # which measures only real movement, not DexScreener indexer lag.
                    # If PP stays silent, fall back to cross-venue dex baseline.
                    try:
                        from memecoin.health_monitor import bump_preflight_attempt as _bpa
                        _bpa()
                    except Exception:
                        pass
                    _pp_at_gate    = 0.0
                    _pf_deadline2  = time.time() + 2.0
                    while time.time() < _pf_deadline2:
                        _p2 = _pp.get_prices().get(_mint, 0)
                        if _p2 > 0:
                            _pp_at_gate = _p2
                            break
                        time.sleep(0.1)
                    if _pp_at_gate > 0:
                        _exec_signal_price = _pp_at_gate
                        log.info(
                            "LIVE PREFLIGHT UPGRADED %s — PP ticked at gate: $%.10f "
                            "(dex was $%.10f) — using same-venue baseline",
                            live_pos.token_symbol, _pp_at_gate, _sig_price or 0,
                        )
                    else:
                        _exec_signal_price = paper_pos.signal_price
                        log.info(
                            "LIVE PREFLIGHT DEFERRED %s — PP silent after 2s, "
                            "using dex baseline $%.10f (%.0f%% cross-venue gate)",
                            live_pos.token_symbol, _sig_price or 0, SLIPPAGE_GATE_DEX_PCT * 100,
                        )

            except Exception as _pf_err:
                log.warning(
                    "PumpPortal pre-flight error for %s: %s — blocking (fail-closed)",
                    live_pos.token_symbol, _pf_err,
                )
                _pf_blocked = True

            if _pf_blocked:
                return

            # Creator is fetched in background (dev-dump detection wires when ready).
            # No longer blocking entry — v4 live data showed gate added no value.
            _sig_creator = getattr(signal, "creator_wallet", "")

            # ── Size normalisation: equalise $ at risk regardless of fill slip ──
            # stop_level is signal-anchored (same formula as the stop check above).
            # stop_dist_from_fill = (pp_price - stop_level) / pp_price
            # size_mult = base_stop_pct / stop_dist  →  floors at 0.5×, caps at 1.0×
            # Example: signal=1.00, pp=1.25, hard_stop=-0.35
            #   stop_level = 0.65; stop_dist = (1.25-0.65)/1.25 = 0.48
            #   size_mult = 0.35/0.48 = 0.73  →  73% of base size
            _base_stop_pct = abs(paper_pos.hard_stop_pct)
            if _sig_price and _pp_price > 0:
                _sa_stop_level = _sig_price * (1 + paper_pos.hard_stop_pct)
                if _pp_price > _sa_stop_level > 0:
                    _stop_dist_fill = (_pp_price - _sa_stop_level) / _pp_price
                    if _stop_dist_fill > 0:
                        _size_mult = _base_stop_pct / _stop_dist_fill
                        _size_mult = max(0.5, min(1.0, _size_mult))
                        _orig_size = _live_size
                        _live_size = round(_live_size * _size_mult, 2)
                        log.info(
                            "SIZE NORM %s: pp=%.8f stop=%.8f dist=%.1f%% "
                            "mult=%.2f  size $%.2f→$%.2f",
                            live_pos.token_symbol, _pp_price, _sa_stop_level,
                            _stop_dist_fill * 100, _size_mult, _orig_size, _live_size,
                        )

            ex = MemeExecutor()
            result = ex.buy(signal.token_address, _live_size, signal.chain,
                            signal_price=_exec_signal_price,
                            max_slippage_pct=0.30)
            if result.get("success"):
                fill_price = result.get("fill_price") or live_pos.entry_price
                signal_price = live_pos.entry_price
                buy_tx_sig = result.get("tx_sig", "")
                # Abort if fill is much worse than the Jupiter quote we used to size.
                # Compare fill vs jupiter_quote_price (fresh at tx-build time), NOT vs
                # signal_price (stale DexScreener snapshot — can be 200%+ below real for
                # graduated tokens). UK bug: fill=$0.0000236 vs stale signal=$0.0000064
                # triggered abort wrongly; fill vs Jupiter quote was actually -1.2% (fine).
                # Threshold 30%: if fill is >30% above what Jupiter quoted, something
                # went wrong (price spiked during the 5s confirmation window).
                _jup_ref = result.get("jupiter_quote_price") or 0
                _abort_ref = _jup_ref if _jup_ref > 0 else signal_price
                _abort_label = "jup_quote" if _jup_ref > 0 else "signal"
                if _abort_ref > 0 and fill_price > _abort_ref * 1.30:
                    _abort_slip = (fill_price / _abort_ref - 1) * 100
                    log.warning(
                        "LIVE BUY ABORTED %s — fill %.1f%% above %s ($%.10f)  "
                        "fill=%.10f  buy_tx=%s",
                        live_pos.token_symbol, _abort_slip, _abort_label, _abort_ref,
                        fill_price, buy_tx_sig,
                    )
                    sell_tx_sig = ""
                    try:
                        ex2 = MemeExecutor()
                        abort_sell = ex2.sell(live_pos.token_address, _live_size, fill_price, live_pos.chain)
                        sell_tx_sig = abort_sell.get("tx_sig", "") if abort_sell else ""
                    except Exception as _e:
                        log.error("Abort-sell failed %s: %s", live_pos.token_symbol, _e)
                    # Write abort row to live journal so the burn is visible and auditable
                    live_pos.entry_price = fill_price
                    live_pos.current_price = fill_price
                    live_pos.peak_price = fill_price
                    live_pos.exit_price = fill_price  # approximate — immediate sell
                    live_pos.exit_time = time.time()
                    live_pos.exit_reason = "abort_tripwire"
                    live_pos.status = "closed"
                    live_pos.notes = (
                        f"live|tx:{buy_tx_sig}|fill:{fill_price:.10f}"
                        f"|abort_slip:{_abort_slip:.1f}%vs{_abort_label}"
                        + (f"|sell_tx:{sell_tx_sig}" if sell_tx_sig else "")
                    )
                    _append_journal(live_pos)
                    try:
                        from app.alerts import _send
                        _send(
                            f"⚠️ LIVE ABORT {live_pos.token_symbol} — fill {_abort_slip:.1f}% above {_abort_label} "
                            f"buy_tx={buy_tx_sig} sell_tx={sell_tx_sig or 'pending'}"
                        )
                    except Exception:
                        pass
                    return
                # Anchor stops to actual fill price, not signal detection price.
                # Without this the hard stop fires at -48% from fill even though it
                # is set to -35%, because the fill averaged +24% above signal.
                live_pos.entry_price   = fill_price
                live_pos.current_price = fill_price
                live_pos.peak_price    = fill_price
                live_pos.tokens_held   = result.get("tokens_received_raw", 0)  # known-balance TP sells
                _dry_tag     = "DRY_RUN|" if result.get("dry_run") else ""
                _est_tag     = "|entry_estimated" if result.get("entry_estimated") else ""
                _slip_tag    = f"|slip:{result['entry_slippage_pct']:+.1f}%" if result.get("entry_slippage_pct") is not None else ""
                _cohort_tag  = "|cohort:graduated" if result.get("pp_silent") else "|cohort:bonding_curve"
                live_pos.notes = f"{_dry_tag}live|tx:{result.get('tx_sig', '')}|fill:{fill_price:.10f}{_est_tag}{_slip_tag}{_cohort_tag}"
                # ── Paper twin: mirror live fill price for honest P&L comparison ──
                # Rebase paper entry to actual fill so paper and live stops
                # trigger at the same token price regardless of price source.
                paper_pos.entry_price   = fill_price
                paper_pos.current_price = fill_price
                paper_pos.peak_price    = fill_price
                _dry_pfx = "DRY_RUN " if result.get("dry_run") else ""
                log.warning("%sLIVE BUY confirmed %s  tx=%s  fill=%.10f",
                            _dry_pfx, live_pos.token_symbol,
                            result.get("tx_sig","")[:16], result.get("fill_price",0))

                # ── Entry latency / slippage instrumentation ─────────────────
                # dex_price   = DexScreener at screening time (stale — indexer lag)
                # pp_sig      = PumpPortal at decision time (live — no lag)
                # pp_gate     = PumpPortal at gate time (2s after subscribe, dex-source only)
                # jup_price   = Jupiter quote at execution time
                # fill_price  = actual on-chain fill
                # artifact    = (pp_sig/dex - 1) — DexScreener indexer lag %
                # screen_slip = (jup/pp_sig - 1) — movement during screen window
                # real_slip   = (fill/jup - 1)   — movement during execution window
                # total_slip  = (fill/exec_signal - 1) — true cost vs gate baseline
                _t_now       = time.time()
                _dex_price   = getattr(signal, "_price_dex", 0) or 0
                _pp_sig      = getattr(signal, "_price_pp", 0) or 0
                _price_src   = getattr(signal, "_price_source", "dex")
                _jup_price   = result.get("jupiter_quote_price") or 0
                _timing      = result.get("timing") or {}
                _t_receive   = getattr(signal, "_t_tg_receive", 0) or 0
                _t_screen    = getattr(signal, "_t_screen_end", 0) or 0
                _artifact    = (_pp_sig / _dex_price - 1) * 100 if _dex_price and _pp_sig else None
                _screen_slip = (_jup_price / _pp_sig - 1) * 100 if _pp_sig and _jup_price else None
                _real_slip   = (fill_price / _jup_price - 1) * 100 if _jup_price and fill_price else None
                _total_slip  = (fill_price / _exec_signal_price - 1) * 100 if _exec_signal_price and fill_price else None
                _total       = (_t_now - _t_receive) if _t_receive else None
                _leg_screen  = (_t_screen - _t_receive) if _t_receive and _t_screen else None
                _leg_exec    = _timing.get("t_confirm")
                log.warning(
                    "ENTRY TIMING %s | src=%s | "
                    "dex=$%.8f  pp_sig=$%.8f  pp_gate=$%.8f  jup=$%.8f  fill=$%.8f | "
                    "artifact=%s%%  screen_slip=%s%%  real_slip=%s%%  total_slip=%s%% | "
                    "screen=%.1fs  quote=%.2fs  submit=%.2fs  confirm=%.2fs  total=%.1fs",
                    live_pos.token_symbol, _price_src,
                    _dex_price, _pp_sig or 0, _pp_at_gate or 0, _jup_price or 0, fill_price,
                    f"{_artifact:.1f}" if _artifact is not None else "?",
                    f"{_screen_slip:.1f}" if _screen_slip is not None else "?",
                    f"{_real_slip:.1f}" if _real_slip is not None else "?",
                    f"{_total_slip:.1f}" if _total_slip is not None else "?",
                    _leg_screen or 0,
                    _timing.get("t_quote", 0),
                    _timing.get("t_submit", 0),
                    _leg_exec or 0,
                    _total or 0,
                )
                try:
                    from app.alerts import alert_live_buy
                    alert_live_buy(live_pos, result.get("tx_sig",""), result.get("sol_spent", _live_size / 70))
                except Exception:
                    pass
                # ── Wire creator wallet ───────────────────────────────────────
                # If already resolved (type-1 or resolved type-2) → wire immediately.
                # Otherwise → background fetch (should only be non-social_alert paths
                # that passed the creator gate above without a pre-resolved creator).
                _sig_creator = getattr(signal, "creator_wallet", "")
                if _sig_creator:
                    live_pos.creator_wallet = _sig_creator
                    try:
                        from memecoin.pumpportal_monitor import monitor as _ppmon2
                        _ppmon2.set_creator(signal.token_address, _sig_creator)
                        log.info("Creator wired immediately %s: %s",
                                 live_pos.token_symbol, _sig_creator[:8])
                    except Exception as _cw_err:
                        log.debug("Creator immediate wire error: %s", _cw_err)
                else:
                    # Background fallback for paths that don't go through the
                    # social_alert gate (dev_launch, copy_trade, etc.)
                    def _fetch_and_store_creator(pos_id: str, mint: str, sym: str):
                        try:
                            from memecoin.data_client import sol_get_token_creator
                            creator = sol_get_token_creator(mint) or ""
                            if creator:
                                p = self._positions.get(pos_id)
                                if p:
                                    p.creator_wallet = creator
                                    _save_positions(self._positions)
                                from memecoin.pumpportal_monitor import monitor as _ppmon
                                _ppmon.set_creator(mint, creator)
                                log.info("Creator wallet resolved %s: %s", sym, creator[:8])
                        except Exception as _ce:
                            log.debug("Creator fetch failed for %s: %s", sym, _ce)

                    threading.Thread(
                        target=_fetch_and_store_creator,
                        args=(live_pos.id, signal.token_address, live_pos.token_symbol),
                        daemon=True,
                    ).start()

                # ── Pre-signed emergency exit ─────────────────────────────────
                # Build step-3 sell tx immediately; refresh periodically.
                # Dev-dump, rug_lp, and velocity exits use this directly, saving
                # the ~300-500ms PumpPortal build step at the worst possible moment.
                # For graduated tokens (pp_silent=True) build a Jupiter tx instead —
                # PumpPortal rejects graduated tokens with Custom:6005 at any slippage.
                _is_graduated_token = bool(result.get("pp_silent"))
                self._schedule_presigned_exit(signal.token_address, is_graduated=_is_graduated_token)

                # Store live position — it will be monitored independently
                self._positions[live_pos.id] = live_pos
                _save_positions(self._positions)
                log.info("Opened LIVE position %s  size=$%.2f  entry=$%.8f",
                         live_pos.id, live_pos.size_usd, live_pos.entry_price)
            elif result.get("unconfirmed"):
                live_pos.notes = f"unconfirmed|tx:{result.get('tx_sig', '')}"
                self._positions[live_pos.id] = live_pos
                _save_positions(self._positions)
                log.error("LIVE BUY UNCONFIRMED %s — check on-chain: %s",
                          live_pos.token_symbol, result.get("tx_sig", ""))
            elif result.get("reason") in ("blocked_quote_drift", "no_quote"):
                _reason = result.get("reason")
                _qprice = result.get("jupiter_quote_price", 0)
                _drift  = result.get("slippage_pct", 0)
                log.warning(
                    "LIVE BUY BLOCKED (executor: %s) %s — "
                    "quote=$%.10f signal=$%.10f drift=%.1f%%",
                    _reason, live_pos.token_symbol,
                    _qprice, paper_pos.signal_price or 0, _drift,
                )
                try:
                    from app.alerts import _send
                    if _reason == "no_quote":
                        _send(
                            f"🚫 NO QUOTE {live_pos.token_symbol} — "
                            f"Jupiter returned no executable quote. Trade blocked."
                        )
                    else:
                        _send(
                            f"🚫 QUOTE DRIFT {live_pos.token_symbol} — "
                            f"quote ${_qprice:.8f} is {_drift:.1f}% above signal. "
                            f"No SOL spent."
                        )
                except Exception:
                    pass
                try:
                    from memecoin.gate_logger import log_gate_block as _lgb
                    _lgb(_reason, live_pos.chain, live_pos.token_address,
                         live_pos.token_symbol, pp_price=_qprice,
                         signal_price=paper_pos.signal_price or 0,
                         size_usd=_live_size)
                except Exception:
                    pass
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

        # sell_stuck throttle: position is stuck waiting for a sell to confirm.
        # The monitor will re-trigger close_position every cycle — rate-limit retries
        # to 60s so we don't hammer the RPC with back-to-back submissions.
        if pos.status == "sell_stuck":
            _retry_at = self._sell_stuck_until.get(pos_id, 0)
            if time.time() < _retry_at:
                return pos   # too soon — skip this cycle
            # Reset for the next ladder attempt
            pos.status = "open"
            pos.sell_attempts = 0
            self._sell_stuck_until.pop(pos_id, None)
            log.info("SELL STUCK retry window open for %s", pos.token_symbol)

        pos.exit_price = price or pos.current_price
        pos.exit_time  = time.time()
        pos.exit_reason = reason
        pos.status = "closed"
        self._jup_fallback_since.pop(pos_id, None)  # clean up dex_pair_loss tracker

        # Live execution gate — only sell on-chain if this position was a live buy
        _was_live_buy = bool(pos.notes and "live|tx:" in pos.notes)
        MAX_SELL_RETRIES = 5
        if LIVE_TRADING and _was_live_buy:
            from memecoin.executor import MemeExecutor
            try:
                # ── Rug-path: pre-signed emergency exit ──────────────────────────
                # For dev_dump / rug_lp / velocity we skip the ~300-500ms
                # PumpPortal build step and send the pre-built tx directly.
                # Baseline: build-on-demand ~350ms detect→land.
                # Target:   presigned ~10ms detect→send.
                # Presigned exits are used for two categories:
                #   Rug-path  (dev_dump, rug_lp, velocity): always — no TP sell can
                #             have fired yet since rg detection is near-instant.
                #   Orderly stops (hard_stop, trailing_stop): ONLY when remaining_fraction
                #             == 1.0 (no partial TP sell has happened yet).  If a TP sold
                #             30%, the presigned tx was built for 100% of original balance —
                #             submitting it would fail (insufficient balance) and fall back
                #             to a fresh Jupiter quote harmlessly, but wastes ~0.5s.
                #             With the guard, presigned fires only when token count matches.
                _RUG_REASONS    = frozenset({"dev_dump", "rug_lp", "velocity"})
                _STOP_REASONS   = frozenset({"hard_stop", "trailing_stop"})
                _presigned_used = False
                _use_presigned  = (
                    reason in _RUG_REASONS
                    or (reason in _STOP_REASONS and pos.remaining_fraction >= 1.0)
                )
                if _use_presigned:
                    with self._presigned_lock:
                        _ps_bytes = self._presigned_exits.pop(pos.token_address, None)
                        self._presigned_ts.pop(pos.token_address, None)
                    if _ps_bytes:
                        from memecoin.executor import _send_transaction, _confirm_tx
                        _t_detect = time.time()
                        try:
                            _psig    = _send_transaction(_ps_bytes)
                            _t_send  = time.time()
                            log.warning(
                                "PRESIGNED EXIT %s (%s)  sig=%s  detect→send=%.0fms",
                                pos.token_symbol, reason, _psig[:16],
                                (_t_send - _t_detect) * 1000,
                            )
                            pos.notes = (pos.notes or "") + f"|presigned:{_psig}"
                            _pconf, _perr = _confirm_tx(_psig, t_sent=_t_send)
                            if _pconf:
                                log.info("Presigned exit confirmed %s  sig=%s",
                                         pos.token_symbol, _psig[:16])
                            else:
                                log.warning("Presigned exit unconfirmed %s  sig=%s  err=%s",
                                            pos.token_symbol, _psig[:16], _perr)
                                pos.notes += "|presigned_unconf"
                            _presigned_used = True
                        except Exception as _pe:
                            log.warning(
                                "Presigned exit send failed %s: %s — falling back to ladder",
                                pos.token_symbol, _pe,
                            )
                            # Restore for ladder attempt
                            if _ps_bytes:
                                with self._presigned_lock:
                                    self._presigned_exits[pos.token_address] = _ps_bytes

                if not _presigned_used:
                    ex     = MemeExecutor()
                    # escalate=True when:
                    #   (a) this is a retry — previous full ladder failed
                    #   (b) token is graduated (cohort:graduated in notes) — PumpPortal rejects
                    #       graduated tokens with Custom:6005 at any slippage; skip straight to Jupiter
                    _is_retry     = getattr(pos, "sell_attempts", 0) > 0
                    _is_graduated = "|cohort:graduated" in (pos.notes or "")
                    if _is_graduated and not _is_retry:
                        log.info("SELL graduated token — skipping PumpPortal ladder, going straight to Jupiter  token=%s",
                                 pos.token_address[:8])
                    result = ex.sell(pos.token_address, pos.size_usd, pos.entry_price, pos.chain,
                                     escalate=_is_retry or _is_graduated)
                    if result.get("success"):
                        _exec_fill = result.get("fill_price")
                        # Only overwrite trigger price if executor measured a real fill.
                        # fill_price=None means sol_recv=0 (balance lag) — keep pos.exit_price
                        # (the stop trigger price) which is the best estimate we have.
                        # fill_price=0.0 also indicates unknown — same treatment.
                        fill  = _exec_fill if _exec_fill else pos.exit_price
                        if _exec_fill:
                            pos.exit_price = fill   # real on-chain fill measured
                        _step = result.get("ladder_step", 1)
                        _all  = result.get("all_sigs", [])
                        _sigs_tag = f"|all_sigs:{','.join(_all)}" if len(_all) > 1 else ""
                        pos.notes = (
                            (pos.notes or "")
                            + f"|sell_tx:{result.get('tx_sig','')}|sell_fill:{fill:.10f}"
                            + (f"|sell_step:{_step}" if _step > 1 else "")
                            + _sigs_tag
                        )
                        log.info("Live sell confirmed %s  tx=%s  fill=%.10f",
                                 pos.token_symbol, result.get("tx_sig","")[:16], fill)
                        try:
                            from app.alerts import alert_live_sell
                            alert_live_sell(pos, result.get("sol_received", 0), result.get("tx_sig", ""))
                        except Exception:
                            pass
                    elif result.get("reason") == "zero_balance":
                        log.warning("Live sell %s — zero balance, tokens already sold. Closing.",
                                    pos.token_symbol)
                        pos.notes = (pos.notes or "") + "|sell_already_gone"
                    else:
                        # Sell failed or unconfirmed — retry up to MAX_SELL_RETRIES
                        pos.sell_attempts = getattr(pos, "sell_attempts", 0) + 1
                        reason_tag = "sell_unconf" if result.get("unconfirmed") else "sell_failed"
                        tx_tag = f":{result.get('tx_sig','')}" if result.get("tx_sig") else ""
                        pos.notes = (pos.notes or "") + f"|{reason_tag}{tx_tag}(attempt {pos.sell_attempts})"
                        if pos.sell_attempts < MAX_SELL_RETRIES:
                            pos.status = "open"
                            pos.exit_price  = 0.0
                            pos.exit_time   = 0.0
                            pos.exit_reason = ""
                            if "|sell_pending" not in (pos.notes or ""):
                                pos.notes = (pos.notes or "") + "|sell_pending"
                            self._positions[pos_id] = pos
                            _save_positions(self._positions)
                            log.error("Live sell %s for %s (attempt %d/%d) — retrying next cycle. err=%s",
                                      reason_tag, pos.token_symbol, pos.sell_attempts, MAX_SELL_RETRIES,
                                      result.get("error") or result.get("tx_sig",""))
                            return pos
                        else:
                            # Ladder exhausted — do NOT phantom-close.
                            # Position stays open as sell_stuck; monitoring continues.
                            # Retry ladder every 60s with fresh blockhash.
                            # Journal write only happens on confirmed sell or reconciler verdict.
                            pos.status = "sell_stuck"
                            pos.exit_price  = 0.0
                            pos.exit_time   = 0.0
                            pos.exit_reason = ""
                            pos.sell_attempts = 0   # reset counter for next 60s window
                            if "|sell_stuck" not in (pos.notes or ""):
                                pos.notes = (pos.notes or "") + "|sell_stuck"
                            self._positions[pos_id] = pos
                            self._sell_stuck_until[pos_id] = time.time() + 5
                            _save_positions(self._positions)
                            log.error(
                                "SELL STUCK %s — ladder exhausted, position stays open, "
                                "retry in 60s.  mint=%s",
                                pos.token_symbol, pos.token_address,
                            )
                            try:
                                from app.alerts import _send
                                _current_bal = ""
                                try:
                                    from memecoin.executor import _token_balance, _get_keypair
                                    _kp   = _get_keypair()
                                    _bal  = _token_balance(str(_kp.pubkey()), pos.token_address)
                                    _current_bal = f"  on-chain_balance={_bal}"
                                except Exception:
                                    pass
                                _sigs = pos.notes or ""
                                _send(
                                    f"🚨 SELL STUCK {pos.token_symbol} — "
                                    f"{MAX_SELL_RETRIES} retries failed, retrying every 60s. "
                                    f"mint={pos.token_address}{_current_bal} | sigs={_sigs[-80:]}"
                                )
                            except Exception:
                                pass
                            return pos
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
                # Ladder exhausted via exception — same sell_stuck endgame
                pos.status = "sell_stuck"
                pos.exit_price  = 0.0
                pos.exit_time   = 0.0
                pos.exit_reason = ""
                pos.sell_attempts = 0
                if "|sell_stuck" not in (pos.notes or ""):
                    pos.notes = (pos.notes or "") + f"|sell_stuck|err:{e}"
                self._positions[pos_id] = pos
                self._sell_stuck_until[pos_id] = time.time() + 5
                _save_positions(self._positions)
                log.error("SELL STUCK %s (exception path) — stays open, retry in 60s: %s",
                          pos.token_symbol, e)
                try:
                    from app.alerts import _send
                    _send(
                        f"🚨 SELL STUCK {pos.token_symbol} — exception after "
                        f"{MAX_SELL_RETRIES} retries: {e}  mint={pos.token_address}"
                    )
                except Exception:
                    pass
                return pos

        _append_journal(pos)
        promote_to_winners(pos)
        del self._positions[pos_id]
        _save_positions(self._positions)
        # Clean up creator mapping in PP monitor
        try:
            from memecoin.pumpportal_monitor import monitor as _ppmon
            _ppmon.clear_creator(pos.token_address)
        except Exception:
            pass
        # Clean up presigned exit (if not already consumed by rug-path)
        with self._presigned_lock:
            self._presigned_exits.pop(pos.token_address, None)
            self._presigned_ts.pop(pos.token_address, None)
            self._graduated_mints.discard(pos.token_address)
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
        _this_cycle_n_with_dex = 0   # positions with PP or DexScreener data this cycle
        for pos in list(self._positions.values()):
            if pos.status != "open":
                continue

            # ── Price source priority ─────────────────────────────────────────
            # 1. PumpPortal real-time (sub-second, from bonding curve reserves)
            # 2. DexScreener poll (5-30s lag — fallback heartbeat)
            # 3. Jupiter quote (last resort when DexScreener is down)
            pp_price = price_overrides.get(pos.token_address)
            _used_dex_source = False
            if pp_price and pp_price > 0:
                pos.current_price = pp_price
                pos.peak_price = max(pos.peak_price, pp_price)
                _used_dex_source = True
            else:
                # DexScreener fallback — used when PumpPortal has no fresh data
                # (graduated tokens, or token not yet subscribed)
                pair = dex_get_token(pos.chain, pos.token_address)
                if pair:
                    price = float(pair.get("priceUsd") or 0)
                    if price > 0:
                        pos.current_price = price
                        pos.peak_price = max(pos.peak_price, price)
                        _used_dex_source = True
                if (not pair or pos.current_price == 0) and pos.chain == "solana":
                    try:
                        from memecoin.executor import _jup_get_quote, _sol_price_usd, SOL_MINT, SOL_DECIMALS
                        _sol = _sol_price_usd()
                        _q   = _jup_get_quote(SOL_MINT, pos.token_address, int(pos.size_usd / _sol * 10**SOL_DECIMALS))
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

            # Track which price source was used for dex_pair_loss discriminator
            if _used_dex_source:
                _this_cycle_n_with_dex += 1
                self._jup_fallback_since.pop(pos.id, None)   # clear if previously in Jupiter-only mode
            else:
                # Jupiter-only (or stale) — start/continue the fallback timer
                if pos.id not in self._jup_fallback_since:
                    self._jup_fallback_since[pos.id] = time.time()

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

            # Guard: if all feeds failed current_price stays at last known value
            # (never set to 0 in the price-update block above). If it were 0,
            # the hard_stop check (price <= stop_level) would fire on a phantom
            # value — skip all stop checks and let the blind-exit timer (scanner)
            # handle the truly-dead-feed case after 20s of silence.
            if pos.current_price <= 0:
                log.debug(
                    "update_prices: skipping stop checks for %s — current_price=0 "
                    "(all feeds silent, blind-exit timer will handle)",
                    pos.token_symbol,
                )
                continue

            gain = pos.pnl_pct
            reason = None

            # 0a. DexScreener pair loss — token-specific feed loss while other positions
            #     still have real data. Fires when:
            #     • Jupiter-only pricing for ≥10s (DexScreener pair gone for this token)
            #     • Other positions still had PP/DexScreener data last cycle (not a global outage)
            #     • Price already -10% from entry (confirming downward move, not just lag)
            _jup_fb_ts = self._jup_fallback_since.get(pos.id, 0)
            if (
                not reason
                and _jup_fb_ts > 0
                and (time.time() - _jup_fb_ts) >= 10
                and self._last_cycle_n_with_dex > 0
                and pos.entry_price > 0
                and pos.current_price < pos.entry_price * 0.90
            ):
                reason = "dex_pair_loss"
                log.warning(
                    "DEX PAIR LOSS %s — Jupiter-only for %.0fs, price -%.1f%% from entry, "
                    "%d other position(s) have DexScreener data → early exit",
                    pos.token_symbol,
                    time.time() - _jup_fb_ts,
                    (1 - pos.current_price / pos.entry_price) * 100,
                    self._last_cycle_n_with_dex,
                )

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

            from memecoin.config import get_signal_settings as _gss_exit
            _exit_cfg   = _gss_exit(pos.signal_type)
            _trail_tiers = _exit_cfg.get("trail_tiers", None)
            _peak_gain   = ((pos.peak_price / pos.entry_price) - 1) if pos.entry_price > 0 else 0

            # 1. Hard stop — signal-anchored when fill > signal price.
            #    Only governs the sub-+30% region (before any trail tier activates).
            if not reason:
                _stop_lvl = pos.entry_price * (1 + pos.hard_stop_pct)
                if pos.signal_price > 0 and pos.entry_price > pos.signal_price:
                    _stop_lvl = pos.signal_price * (1 + pos.hard_stop_pct)
                if pos.current_price <= _stop_lvl:
                    reason = "hard_stop"

            # 2. Trailing stop (ATH-anchored tier system or legacy single-tier)
            if not reason and pos.peak_price > 0 and pos.entry_price > 0:
                if _trail_tiers:
                    # Find active tier: highest activates_at ≤ peak_gain
                    _active_tier = None
                    for _tier in sorted(_trail_tiers,
                                        key=lambda t: t["activates_at"], reverse=True):
                        if _peak_gain >= _tier["activates_at"]:
                            _active_tier = _tier
                            break

                    if _active_tier:
                        _trail_pct   = -abs(_active_tier["trail_pct"])
                        _trail_stop  = pos.peak_price * (1 + _trail_pct)
                        # Breakeven floor: peak ≥ +40% → trail_stop ≥ entry * 1.02.
                        # A trade that has shown +40% can never close as a loser.
                        if _peak_gain >= 0.40:
                            _floor = pos.entry_price * 1.02
                            if _trail_stop < _floor:
                                _trail_stop = _floor
                        if pos.current_price <= _trail_stop:
                            reason = "trailing_stop"
                else:
                    # Legacy single-tier (non-social_alert signal types)
                    if gain >= pos.trail_activates_pct:
                        drawdown_from_peak = (pos.current_price - pos.peak_price) / pos.peak_price
                        if drawdown_from_peak <= pos.trailing_stop_pct:
                            reason = "trailing_stop"

            # 3. Whale exit (primary exit signal)
            if not reason and pos.token_address in whale_sells:
                sellers = whale_sells[pos.token_address]
                involved = [w for w in sellers if w in pos.whales_involved]
                if involved:
                    n_whales = pos.whale_count or 1
                    if n_whales == 1:
                        reason = f"whale_exit:{involved[0][:8]}"
                    elif len(involved) >= max(1, n_whales // 2):
                        reason = f"whale_exit:{len(involved)}_of_{n_whales}"

            # 4. Time stop — only fires while peak_gain < 30%.
            #    Never interrupt a runner mid-leg.
            if not reason and (time.time() - pos.entry_time) / 60 > pos.time_stop_minutes:
                if _peak_gain < 0.30 and gain < TIME_STOP_MIN_GAIN:
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
                _was_live_buy = bool(pos.notes and "live|tx:" in pos.notes)
                for tp_pct, tp_fraction in TP_LEVELS:
                    level_key = f"tp_{int(tp_pct*100)}"
                    if gain >= tp_pct and level_key not in pos.tp_levels_hit:
                        pos.tp_levels_hit.append(level_key)
                        sell_frac = tp_fraction * pos.remaining_fraction
                        pos.remaining_fraction -= sell_frac
                        partial_usd = sell_frac * pos.size_usd

                        # ── Live TP sell (background thread) ───────────────────────
                        # Apply paper estimate immediately so the price-monitor loop
                        # continues without blocking.  A daemon thread does the real
                        # on-chain sell in the background and corrects realized_pnl_usd
                        # once the tx confirms — the trailing stop can fire at any time.
                        # Paper path or failed live sell: estimate stays.
                        pos.realized_pnl_usd += sell_frac * pos.size_usd * tp_pct  # paper estimate

                        if LIVE_TRADING and _was_live_buy:
                            _tp_thread = threading.Thread(
                                target=self._run_tp_sell_bg,
                                args=(pos.id, sell_frac, tp_pct, level_key),
                                daemon=True,
                            )
                            _tp_thread.start()

                        log.info(
                            "TP hit %s  %s +%.0f%%  sold %.0f%% ($%.2f)  realized=$%.2f",
                            pos.id, pos.token_symbol, gain * 100,
                            tp_fraction * 100, partial_usd, pos.realized_pnl_usd,
                        )
                        try:
                            alerts.alert_tp_hit(pos, tp_pct, partial_usd)
                        except Exception:
                            pass

        # Update cycle counter for next iteration's dex_pair_loss discriminator
        self._last_cycle_n_with_dex = _this_cycle_n_with_dex

        _save_positions(self._positions)
        return exits

    # ---- background TP sell ----

    def _run_tp_sell_bg(self, pos_id: str, sell_frac: float, tp_pct: float, level_key: str):
        """Execute a live partial TP sell in a daemon thread.

        The paper estimate was already applied to realized_pnl_usd before this
        thread started.  On success the estimate is replaced with the real fill.
        The price-monitor loop is never blocked — the trailing stop can fire
        during the ~10s tx confirmation window.
        """
        try:
            from memecoin.executor import MemeExecutor as _MEx
        except Exception as _e:
            log.warning("LIVE TP BG: executor import failed: %s", _e)
            return

        pos = self._positions.get(pos_id)
        if pos is None or pos.status != "open":
            # Position already closed by trailing stop while we were waiting — nothing to do
            log.info("LIVE TP BG: position %s already closed, skipping TP sell", pos_id)
            return

        # Use the exact token count received at buy time (from tx postTokenBalances delta).
        # This bypasses the _token_balance() RPC call entirely — no settle lag, no
        # zero_balance_partial failure on fast-pumping tokens that TP within 2-5s of entry.
        # tokens_held is set on the live position at buy confirm time.
        _known_count = int(pos.tokens_held * sell_frac) if pos.tokens_held > 0 else 0
        if _known_count > 0:
            log.info("LIVE TP BG: using known_token_count=%d for %.0f%% sell  %s",
                     _known_count, sell_frac * 100, pos.token_symbol)
        else:
            log.info("LIVE TP BG: tokens_held=0, falling back to RPC balance query  %s",
                     pos.token_symbol)

        try:
            _tp_ex      = _MEx()
            _tp_is_grad = "|cohort:graduated" in (pos.notes or "")
            _tp_r  = _tp_ex.sell(
                pos.token_address, pos.size_usd, pos.entry_price,
                pos.chain, fraction=sell_frac,
                escalate=_tp_is_grad,
                known_token_count=_known_count,
            )
        except Exception as _tp_err:
            log.warning("LIVE TP BG exception %s: %s — paper estimate kept", pos.token_symbol, _tp_err)
            return

        # Re-fetch position: it may have been closed during the tx confirmation
        pos = self._positions.get(pos_id)

        if _tp_r.get("success"):
            _tp_fill_raw = _tp_r.get("fill_price")
            _tp_fill     = _tp_fill_raw if _tp_fill_raw else (pos.current_price if pos else 0.0)
            _tp_sig     = _tp_r.get("tx_sig", "")
            _paper_est  = sell_frac * (pos.size_usd if pos else 0.0) * tp_pct
            _real_pnl   = (
                (_tp_fill / pos.entry_price - 1) * sell_frac * pos.size_usd
                if pos and pos.entry_price > 0 else _paper_est
            )

            if pos and pos.status == "open":
                # Position still open: swap paper estimate for real fill
                pos.realized_pnl_usd += _real_pnl - _paper_est
                pos.notes = (
                    (pos.notes or "")
                    + f"|{level_key}_tx:{_tp_sig}"
                    + f"|{level_key}_fill:{_tp_fill:.10f}"
                )
                _save_positions(self._positions)
                log.warning(
                    "LIVE TP SELL BG %s  tp=+%.0f%%  frac=%.0f%%  "
                    "fill=%.10f  real=$%.2f  paper_est=$%.2f  tx=%s",
                    pos.token_symbol, tp_pct * 100, sell_frac * 100,
                    _tp_fill, _real_pnl, _paper_est, _tp_sig[:16],
                )
            else:
                # Position was closed while tx was confirming — just log
                log.warning(
                    "LIVE TP SELL BG %s confirmed but position already closed  "
                    "fill=%.10f  realized=$%.2f  tx=%s",
                    pos_id, _tp_fill, _real_pnl, _tp_sig[:16],
                )

            try:
                _sym = pos.token_symbol if pos else pos_id
                from app.alerts import _send
                _send(
                    f"✅ LIVE TP {_sym} +{tp_pct*100:.0f}%\n"
                    f"Sold: {sell_frac*100:.0f}% of position\n"
                    f"Locked: ${_real_pnl:.2f}\n"
                    + (f"Remaining: {pos.remaining_fraction*100:.0f}%\n" if pos and pos.status == "open" else "")
                    + f"tx: {_tp_sig[:20]}"
                )
            except Exception:
                pass
        else:
            log.warning(
                "LIVE TP BG sell FAILED %s — paper estimate kept. reason=%s",
                pos.token_symbol if pos else pos_id,
                _tp_r.get("reason") or _tp_r.get("error", "unknown"),
            )

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

    def load_journal(self, signal_type: str = None) -> list[dict]:
        """
        Load closed trade records.
        signal_type="social_alert" → social journal only.
        signal_type=None → both journals merged, sorted by exit_time.
        """
        rows = []
        targets = []
        if signal_type == "social_alert":
            targets = [SOCIAL_JOURNAL_FILE]
        elif signal_type is not None:
            targets = [JOURNAL_FILE]
        else:
            targets = [JOURNAL_FILE, SOCIAL_JOURNAL_FILE]

        for path in targets:
            if path.exists():
                with open(path, newline="") as f:
                    rows.extend(csv.DictReader(f))

        # Sort merged result by exit_time ascending
        rows.sort(key=lambda r: float(r.get("exit_time") or 0))
        return rows
