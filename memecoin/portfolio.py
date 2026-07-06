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
    PRICE_PATHS_DIR,
    get_signal_settings,
    LIVE_TRADING,
    DAILY_LOSS_LIMIT,
    LIVE_DRY_RUN,
    REALTIME_PRICE_FEED,
    SLIPPAGE_GATE_RT_PCT, SLIPPAGE_GATE_DEX_PCT,
    SELL_STUCK_RETRY_SEC,
)
from memecoin.data_client import dex_get_token
from memecoin.candidate_log import promote_to_winners
from memecoin.journal_io import JOURNAL_LOCK
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
    # accounting v3 fields (added 2026-07-06)
    "sol_received",          # raw SOL received from chain at exit (from tx_meta / reconciler)
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
    sol_received: float = 0.0  # raw SOL received at exit (from on-chain delta — accounting only)

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
    # Atomic write: write to temp then rename so a mid-write kill never corrupts the file.
    # os.replace / Path.replace is atomic on POSIX (single filesystem).
    _tmp = POSITIONS_FILE.with_suffix(".json.tmp")
    _tmp.write_text(json.dumps(data, indent=2))
    _tmp.replace(POSITIONS_FILE)


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
        "sol_received": round(pos.sol_received, 8) if pos.sol_received else "",
        "accounting_epoch": ACCOUNTING_EPOCH,
    }


def _append_price_tick(pos: "Position", price: float) -> None:
    """Append one (epoch, price_usd) tick to logs/price_paths/<mint>.csv.

    Lightweight, append-only. Called on every monitoring loop iteration so
    the full entry-to-exit price path is available for offline trail/TP replay.
    Any exception is swallowed — this must never interrupt the hot path.
    """
    try:
        PRICE_PATHS_DIR.mkdir(parents=True, exist_ok=True)
        path = PRICE_PATHS_DIR / f"{pos.token_address}.csv"
        write_hdr = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_hdr:
                w.writerow(["epoch", "price_usd"])
            w.writerow([round(time.time(), 3), price])
    except Exception:
        pass


def _append_journal(pos: Position):
    JOURNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    row = _build_journal_row(pos)

    # Route to split journals: social_alert → social journal, everything else → main journal
    _is_social = pos.signal_type == "social_alert"
    target = SOCIAL_JOURNAL_FILE if _is_social else JOURNAL_FILE

    # Hold JOURNAL_LOCK for all file writes so reconciler rewrites cannot erase new rows.
    # Never hold this lock during RPC calls — build the row before acquiring.
    with JOURNAL_LOCK:
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

    # DRY_RUN funnel counter — read + alert outside lock (non-critical, can be eventually consistent)
    if pos.notes and "live|tx:" in pos.notes and "DRY_RUN" in (pos.notes or ""):
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


def is_rescue_eligible_error(
    error_class: str = "",
    exit_state: str = "",
    reason: str = "",
    oracle_bonding_curve: bool = False,
) -> bool:
    """
    Return True when the current sell context warrants a Jupiter rescue attempt.
    Replaces the fragile ``"_er" in dir()`` pattern in close_position().

    Parameters
    ----------
    error_class          : str   error_class from the PumpSwap local path result
    exit_state           : str   TokenExitState.value string from ExitRouter classification
    reason               : str   reason string passed to close_position()
    oracle_bonding_curve : bool  True when bonding curve oracle confirmed complete=False at
                                 buy time. MIGRATION_UNCERTAIN is PP-silence-based and fires
                                 for T22 tokens that are still on the bonding curve — Jupiter
                                 rescue has no route for them. Route via PumpPortal instead.
    """
    # Oracle-confirmed bonding curve + MIGRATION_UNCERTAIN = T22 token still on BC.
    # Jupiter cannot route these. Skip rescue and let executor.sell use PumpPortal.
    if oracle_bonding_curve and exit_state == "MIGRATION_UNCERTAIN":
        return False
    _RESCUE_EXIT_STATES = frozenset({
        "GRADUATED_PUMPSWAP",
        "GRADUATED_PUMPSWAP_SPL",
        "GRADUATED_PUMPSWAP_T22",
        "MIGRATION_UNCERTAIN",
        "MIGRATION_UNCERTAIN_SPL",
        "MIGRATION_UNCERTAIN_T22",
    })
    _RESCUE_ERROR_CLASSES = frozenset({
        "pumpswap_no_pool",
        "pumpswap_bad_pool_layout",
        "pool_not_indexed",
        "local_build_failed",
        "local_sim_failed",
        "pumpswap_simulation_failed",
        "jupiter_no_route",          # retry after no-route (route may appear later)
        "graduated_unsellable",      # pump-amm + Jupiter in executor both failed
        "Custom:6005",               # BC graduation detected during sell
        "Custom:6001",
    })
    _RESCUE_REASONS = frozenset({
        "migration_uncertain_no_pool",
        "migration_uncertain_retry",
        "sell_stuck",
        "graduated_exit",
        "feed_blind",
    })
    if exit_state in _RESCUE_EXIT_STATES:
        return True
    if error_class in _RESCUE_ERROR_CLASSES:
        return True
    if reason in _RESCUE_REASONS:
        return True
    return False


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
        # Per-position close lock: prevents double-sell when the 0.5s poll loop and
        # the event-driven exit queue both fire for the same position simultaneously.
        # Key: pos_id → Lock.  Acquired at close_position entry, released after
        # pos.status = "closed" is set.  Second caller blocks then sees "closed" → no-op.
        self._close_locks: dict[str, threading.Lock] = {}
        self._close_locks_meta = threading.Lock()  # guards _close_locks dict itself
        # sell_stuck throttle: pos_id → earliest time to retry sell
        # In-memory only — resets on restart (which itself gives a fresh attempt).
        self._sell_stuck_until: dict[str, float] = {}
        # graduated_unsellable retry counter: pos_id → attempts
        # After MAX_GRADUATED_RETRIES the position is written off as a total loss.
        self._graduated_retry_count: dict[str, int] = {}
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
          PumpPortal pool="auto", 98% slippage, 0.005 SOL fee.  PumpPortal
          re-fetches the latest blockhash server-side → valid through 45s refresh.

        Graduated tokens (is_graduated=True):
          pool="auto" returns Custom:6005 (tries bonding curve, fails).
          Use pool="pump-amm" (PumpSwap direct) instead — same PP trade-local
          API, same 45s blockhash TTL, no Jupiter needed.
        """
        try:
            from memecoin.executor import (
                _get_keypair, _load_solders, _pumpportal_build_tx,
                _helius_priority_fee, EXECUTOR_BACKEND,
            )
            if EXECUTOR_BACKEND != "pumpportal":
                return
            _t0 = time.time()
            _, VersionedTransaction, _ = _load_solders()
            keypair = _get_keypair()
            wallet  = str(keypair.pubkey())

            if is_graduated:
                # Graduated token — pool="auto" returns Custom:6005 (bonding curve path).
                # Use pool="pump-amm" (PumpSwap direct) for the presigned exit.
                # PumpPortal re-fetches blockhash server-side → stays valid ~90s.
                try:
                    _grad_fee = max(_helius_priority_fee(mint, "UnsafeMax"), 0.005)
                    tx_bytes  = _pumpportal_build_tx(
                        wallet_pubkey=wallet,
                        action="sell",
                        token_mint=mint,
                        amount="100%",
                        denominated_in_sol=False,
                        slippage_pct=98,
                        priority_fee_sol=_grad_fee,
                        pool="pump-amm",
                    )
                    _path = "PumpPortal/pump-amm"
                except Exception as _pamm_e:
                    log.warning("Presigned pump-amm exit build failed for %s: %s", mint[:8], _pamm_e)
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

        All exits now use PumpPortal trade-local (server-side blockhash refresh, ~90s TTL).
        Bonding-curve tokens: pool="auto",      refresh every 45s.
        Graduated tokens:     pool="pump-amm",  refresh every 45s (same TTL).
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
                interval = 45  # same for both; PP re-fetches blockhash server-side
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

        Only rows with entry_time >= LIVE_GATE_EPOCH are counted.
        Pre-epoch trades were executed by buggy builds and are not evidence
        about the current system.
        """
        import csv as _csv
        try:
            from memecoin.config import LIVE_GATE_EPOCH as _epoch
        except ImportError:
            _epoch = "1970-01-01"
        stats: dict = {}
        try:
            with open(LIVE_JOURNAL_FILE) as f:
                for row in _csv.DictReader(f):
                    if (row.get("entry_time") or "")[:10] < _epoch:
                        continue
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

        # ── Canary mode size enforcement ─────────────────────────────────────
        _canary_capped = False
        try:
            from memecoin.config import LIVE_CANARY_MODE as _canary_mode
            from memecoin.config import EXIT_SYSTEM_VALIDATED as _validated
            from memecoin.config import MAX_CANARY_TRADE_USD as _canary_max
            import memecoin.kill_switch as _ks
            if not _ks.live_buys_enabled():
                # Will be caught by kill switch check in executor, but also gate here
                _live_size = 0
            elif _canary_mode and not _validated:
                _live_size = min(_live_size, _canary_max)
                _canary_capped = True
                log.info("CANARY MODE: capping trade size to $%.2f (EXIT_SYSTEM_VALIDATED=False)", _canary_max)
        except Exception:
            pass

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

                # ── Category gate (before subscribing / spending any SOL) ────
                # Three categories of social-alert tokens; each has a different fix:
                #   Cat-1 (new launch, pre-screened, PP source): PP live price baseline ✓
                #   Cat-2 (bonding curve, TG-alerted, 5-30 min): age gate — stale = skip
                #   Cat-3 (graduated / PumpSwap): skip entirely — no real-time PumpSwap
                #          price source yet; pump-amm sell path unreliable for Token-2022
                _token_dex    = (paper_pos.dex_id or "").lower()
                _token_age    = paper_pos.age_minutes or 0.0
                # Graduated-cohort exclusion: CAT-3 tokens (dex_id=pumpswap/raydium/orca)
                # are already migrated. No bonding-curve buy possible; no real-time feed.
                # get_pumpfun_curve_snapshot() also blocks complete=True / account_missing
                # tokens further down in the type-1 and type-2 paths.
                _is_graduated = _token_dex in ("pumpswap", "raydium", "orca")
                if _is_graduated:
                    log.info(
                        "LIVE PREFLIGHT CAT-3 SKIP %s — graduated (%s); no PumpSwap "
                        "price feed yet. Skipping live, paper continues.",
                        live_pos.token_symbol, _token_dex,
                    )
                    return

                # Cat-2 age gate: TG-alerted bonding-curve tokens signal at 5-30 min age.
                # Beyond MAX_AGE_SOCIAL_LIVE the signal is stale (pump likely resolved).
                # Cat-1 (PP source) is exempt — pre-screened at creation, always fresh.
                from memecoin.config import MAX_AGE_SOCIAL_LIVE as _MAX_AGE
                if _price_source != "pp" and _token_age > _MAX_AGE:
                    log.info(
                        "LIVE PREFLIGHT CAT-2 STALE %s — age=%.1f min > %d min gate; "
                        "skipping live, paper continues.",
                        live_pos.token_symbol, _token_age, _MAX_AGE,
                    )
                    return

                # Always subscribe so monitoring ticks start arriving immediately
                _pp.subscribe({_mint})

                if _price_source == "pp":
                    # ── Type-1 path: bonding-curve baseline, fail-closed if no price ─
                    # New preflight order (replaces 2s flat PP wait):
                    #   1. PP cache hit (instant)    → baseline_source='pp_tick'
                    #   2. Curve snapshot (one RPC)  → baseline_source='curve' if complete=False
                    #      complete=True / account_missing → block as graduated/migrated
                    #   3. Curve RPC error           → wait 0.5s for PP tick (reduced from 2s)
                    #   4. signal._price_pp fallback → bonding-curve confirmed at screen time
                    #   5. All silent               → fail-closed
                    _pp_price = 0.0
                    _baseline_source = ""
                    _t_preflight_start = time.time()

                    # 1. Check PP cache immediately (no wait)
                    _pp_price = _pp.get_prices().get(_mint, 0)
                    if _pp_price > 0:
                        _baseline_source = "pp_tick"
                    else:
                        # 2. Fetch curve snapshot (shares 5s cache with executor buy gate)
                        try:
                            from memecoin.executor import get_pumpfun_curve_snapshot as _gcs
                            _curve_snap = _gcs(_mint)
                            _curve_complete = _curve_snap.get("complete")
                            _curve_reason   = _curve_snap.get("reason", "")
                            if _curve_complete is False and (_curve_snap.get("price_usd") or 0) > 0:
                                # Still on bonding curve, got live price → use as baseline
                                _pp_price = _curve_snap["price_usd"]
                                _baseline_source = "curve"
                            elif _curve_complete is True or _curve_reason == "account_missing":
                                # Token graduated or curve account closed → block
                                log.info(
                                    "LIVE PREFLIGHT GRADUATED %s — curve complete=%s reason=%s; "
                                    "blocking live buy (graduated/migrated)",
                                    live_pos.token_symbol, _curve_complete, _curve_reason,
                                )
                                _pf_blocked = True
                            elif not _curve_snap.get("ok"):
                                # RPC/parse error → short PP wait (0.5s, not 2s)
                                _pf_deadline = time.time() + 0.5
                                while time.time() < _pf_deadline:
                                    _p = _pp.get_prices().get(_mint, 0)
                                    if _p > 0:
                                        _pp_price = _p
                                        _baseline_source = "pp_tick"
                                        break
                                    time.sleep(0.05)
                        except Exception as _snap_e:
                            log.debug("curve snapshot error type-1 %s: %s", _mint[:8], _snap_e)
                            # Curve unavailable → short PP wait (0.5s)
                            _pf_deadline = time.time() + 0.5
                            while time.time() < _pf_deadline:
                                _p = _pp.get_prices().get(_mint, 0)
                                if _p > 0:
                                    _pp_price = _p
                                    _baseline_source = "pp_tick"
                                    break
                                time.sleep(0.05)

                    _elapsed_ms = int((time.time() - _t_preflight_start) * 1000)

                    try:
                        from memecoin.health_monitor import bump_preflight_attempt as _bpa
                        _bpa()
                    except Exception:
                        pass

                    if not _pf_blocked:
                        log.info(
                            "LIVE PREFLIGHT BASELINE token=%s symbol=%s baseline=%s "
                            "price=%.10f elapsed_ms=%d",
                            _mint[:8], live_pos.token_symbol, _baseline_source or "none",
                            _pp_price, _elapsed_ms,
                        )

                    if _pp_price == 0 and not _pf_blocked:
                        # 4. signal._price_pp fallback: bonding-curve confirmed at screen time.
                        # A 1-3 min old token can have a 2-20s quiet period between the
                        # screening accumulation and the TG alert.
                        _sig_pp_fallback = getattr(signal, "_price_pp", 0.0) or 0.0
                        if _sig_pp_fallback > 0:
                            _pp_price = _sig_pp_fallback
                            log.info(
                                "LIVE PREFLIGHT PP-QUIET %s — using screening price "
                                "$%.10f (PP silent, bonding-curve confirmed at screen)",
                                live_pos.token_symbol, _pp_price,
                            )
                        else:
                            # 5. All silent → fail-closed
                            try:
                                from memecoin.health_monitor import bump_preflight_no_price as _bpnp
                                _bpnp()
                            except Exception:
                                pass
                            log.warning(
                                "LIVE PREFLIGHT NO PRICE %s — PP and curve both silent, "
                                "blocking trade (fail-closed, type-1)",
                                live_pos.token_symbol,
                            )
                            try:
                                from app.alerts import _send
                                _send(
                                    f"🚫 PREFLIGHT NO PRICE {live_pos.token_symbol} — "
                                    f"PP and curve silent. Trade blocked, no SOL spent."
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

                    if _pp_price > 0 and not _pf_blocked:
                        # Drift gate: live price vs signal price (measures real movement)
                        _gate = SLIPPAGE_GATE_RT_PCT
                        if _sig_price and _pp_price > _sig_price * (1 + _gate):
                            _pf_slip = (_pp_price / _sig_price - 1) * 100
                            log.warning(
                                "LIVE PREFLIGHT BLOCKED %s — %s price %.10f is %.1f%% above "
                                "signal %.10f (>%.0f%% gate)",
                                live_pos.token_symbol, _baseline_source, _pp_price, _pf_slip,
                                _sig_price, _gate * 100,
                            )
                            try:
                                from app.alerts import _send
                                _send(
                                    f"🚫 PREFLIGHT BLOCKED {live_pos.token_symbol} — "
                                    f"{_baseline_source} ${_pp_price:.8f} already {_pf_slip:.1f}% "
                                    f"above signal ${_sig_price:.8f} (>{_gate*100:.0f}% gate). "
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
                    # ── Type-2 path (dex source): curve baseline, PP upgrade ────────
                    # New preflight order (replaces 2s flat PP wait):
                    #   1. PP cache hit (instant)    → upgrade to same-venue baseline
                    #   2. Curve snapshot (one RPC)  → baseline_source='curve' if complete=False
                    #      complete=True / account_missing → block as graduated/migrated (NEW)
                    #   3. Curve RPC error           → wait 0.5s for PP tick (reduced from 2s)
                    #   4. All silent               → use dex signal_price (existing fallback)
                    try:
                        from memecoin.health_monitor import bump_preflight_attempt as _bpa
                        _bpa()
                    except Exception:
                        pass
                    _pp_at_gate       = 0.0
                    _baseline_source2 = ""
                    _t_preflight2     = time.time()

                    # 1. Check PP cache immediately (no wait)
                    _p2 = _pp.get_prices().get(_mint, 0)
                    if _p2 > 0:
                        _pp_at_gate = _p2
                        _baseline_source2 = "pp_tick"
                    else:
                        # 2. Fetch curve snapshot
                        try:
                            from memecoin.executor import get_pumpfun_curve_snapshot as _gcs2
                            _curve_snap2 = _gcs2(_mint)
                            _curve_complete2 = _curve_snap2.get("complete")
                            _curve_reason2   = _curve_snap2.get("reason", "")
                            if _curve_complete2 is False and (_curve_snap2.get("price_usd") or 0) > 0:
                                _pp_at_gate = _curve_snap2["price_usd"]
                                _baseline_source2 = "curve"
                            elif _curve_complete2 is True or _curve_reason2 == "account_missing":
                                # Token graduated or migrated → block (NEW: previously fell through)
                                log.info(
                                    "LIVE PREFLIGHT GRADUATED %s — curve complete=%s reason=%s; "
                                    "blocking live buy (type-2, graduated/migrated)",
                                    live_pos.token_symbol, _curve_complete2, _curve_reason2,
                                )
                                _pf_blocked = True
                            elif not _curve_snap2.get("ok"):
                                # RPC/parse error → short PP wait (0.5s, not 2s)
                                _pf_deadline2 = time.time() + 0.5
                                while time.time() < _pf_deadline2:
                                    _p2b = _pp.get_prices().get(_mint, 0)
                                    if _p2b > 0:
                                        _pp_at_gate = _p2b
                                        _baseline_source2 = "pp_tick"
                                        break
                                    time.sleep(0.05)
                        except Exception as _snap2_e:
                            log.debug("curve snapshot error type-2 %s: %s", _mint[:8], _snap2_e)
                            _pf_deadline2 = time.time() + 0.5
                            while time.time() < _pf_deadline2:
                                _p2b = _pp.get_prices().get(_mint, 0)
                                if _p2b > 0:
                                    _pp_at_gate = _p2b
                                    _baseline_source2 = "pp_tick"
                                    break
                                time.sleep(0.05)

                    _elapsed_ms2 = int((time.time() - _t_preflight2) * 1000)

                    if _pp_at_gate > 0 and not _pf_blocked:
                        _exec_signal_price = _pp_at_gate
                        # Also set _pp_price so size normalisation fires for curve-baseline buys
                        if _baseline_source2 == "curve":
                            _pp_price = _pp_at_gate
                        log.info(
                            "LIVE PREFLIGHT BASELINE token=%s symbol=%s baseline=%s "
                            "price=%.10f elapsed_ms=%d",
                            _mint[:8], live_pos.token_symbol, _baseline_source2,
                            _pp_at_gate, _elapsed_ms2,
                        )
                    elif not _pf_blocked:
                        # 4. All silent → use dex signal_price as anchor.
                        # Never enter with signal_price=0: stop_level=0*0.65=0 fires
                        # on any retracement and size norm is skipped entirely.
                        _exec_signal_price = _sig_price or paper_pos.signal_price
                        if not _exec_signal_price:
                            log.warning(
                                "LIVE PREFLIGHT NO PRICE %s — PP and curve silent and no "
                                "DexScreener price; blocking (fail-closed, type-2)",
                                live_pos.token_symbol,
                            )
                            _pf_blocked = True
                        else:
                            log.info(
                                "LIVE PREFLIGHT BASELINE token=%s symbol=%s baseline=dex "
                                "price=%.10f elapsed_ms=%d",
                                _mint[:8], live_pos.token_symbol, _exec_signal_price, _elapsed_ms2,
                            )
                            log.info(
                                "LIVE PREFLIGHT DEFERRED %s — PP and curve silent, "
                                "using dex signal price $%.10f as stop/size anchor; "
                                "Jupiter quote will be live baseline",
                                live_pos.token_symbol, _exec_signal_price,
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
                            max_slippage_pct=0.30,
                            dex_id=live_pos.dex_id)
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
                # Abort reference priority:
                # 1. Jupiter quote — live AMM price (best)
                # 2. PP-fresh signal_price — only if signal came from PP cache (sub-second fresh)
                # 3. Missing — skip abort entirely, tag note, log warning
                # Never use DexScreener-derived price (stale 10-30s, makes abort meaningless)
                if _jup_ref > 0:
                    _abort_ref = _jup_ref
                    _abort_ref_label = "jup_quote"
                elif _pp_at_gate > 0 and signal_price > 0:
                    _abort_ref = signal_price
                    _abort_ref_label = "pp_fresh"
                else:
                    _abort_ref = 0
                    _abort_ref_label = "missing"
                if _abort_ref == 0:
                    live_pos.notes = (live_pos.notes or "") + "|abort_ref_missing"
                    log.warning(
                        "abort_tripwire skipped — no fresh price ref (not DexScreener)  token=%s",
                        live_pos.token_symbol,
                    )
                elif fill_price > _abort_ref * 1.30:
                    _abort_slip = (fill_price / _abort_ref - 1) * 100
                    log.warning(
                        "LIVE BUY ABORTED %s — fill %.1f%% above %s ($%.10f)  "
                        "fill=%.10f  buy_tx=%s",
                        live_pos.token_symbol, _abort_slip, _abort_ref_label, _abort_ref,
                        fill_price, buy_tx_sig,
                    )
                    sell_tx_sig = ""
                    try:
                        ex2 = MemeExecutor()
                        abort_sell = ex2.sell(live_pos.token_address, _live_size, fill_price, live_pos.chain, urgent=True)
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
                        f"|abort_slip:{_abort_slip:.1f}%vs{_abort_ref_label}"
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
                # cohort tag: oracle result is authoritative.
                # pp_silent alone is NOT graduation proof — T22 tokens are always
                # PP-silent but can be on bonding curve (oracle_bonding_curve=True).
                _cohort_tag  = (
                    "|cohort:bonding_curve"
                    if result.get("oracle_bonding_curve")
                    else ("|cohort:graduated" if result.get("pp_silent") else "|cohort:bonding_curve")
                )
                _canary_tag  = f"|canary_cap:{_canary_max}" if _canary_capped else ""
                live_pos.notes = f"{_dry_tag}live|tx:{result.get('tx_sig', '')}|fill:{fill_price:.10f}{_est_tag}{_slip_tag}{_cohort_tag}{_canary_tag}"
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
                except Exception as _alert_err:
                    log.warning("alert_live_buy failed: %s", _alert_err)
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
        # Fast pre-check without the per-position lock — eliminates lock allocation
        # overhead for already-closed positions on the hot monitor path.
        pos = self._positions.get(pos_id)
        if not pos or pos.status == "closed":
            return None

        # Per-position lock: prevents double-sell when the 0.5s poll loop and the
        # event-driven exit queue both fire simultaneously for the same position.
        # Second caller blocks here, then re-checks status and finds "closed" → no-op.
        with self._close_locks_meta:
            if pos_id not in self._close_locks:
                self._close_locks[pos_id] = threading.Lock()
            _pos_lock = self._close_locks[pos_id]

        with _pos_lock:
            # Re-read inside lock — another thread may have closed it while we waited.
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

            # Set status="closed" atomically inside the lock so any concurrent call
            # (0.5s poll loop and event-driven exit queue) sees it immediately.
            # The actual on-chain sell below happens outside the lock; on failure
            # the sell path restores status to "open" or "sell_stuck" safely.
            pos.exit_price = price or pos.current_price
            pos.exit_time  = time.time()
            pos.exit_reason = reason
            pos.status = "closed"
            self._jup_fallback_since.pop(pos_id, None)  # clean up dex_pair_loss tracker

        # Live execution gate — only sell on-chain if this position was a live buy
        _was_live_buy = bool(pos.notes and "live|tx:" in pos.notes)
        MAX_SELL_RETRIES = 5

        # reconciled_gone: balance already 0 on-chain — sell would fail and re-arm loop.
        # manual_sell: user sold outside the bot — nothing to sell on-chain.
        _skip_chain_sell = reason in ("reconciled_gone", "manual_sell")

        # Sell kill switch: /sells_off Telegram command disables on-chain sells.
        # Positions keep tracking, exits just don't fire the executor.
        if not _skip_chain_sell:
            try:
                from memecoin.kill_switch import live_sells_enabled as _lse
                if not _lse():
                    log.warning(
                        "SELL KILL SWITCH active — skipping on-chain sell for %s  reason=%s",
                        pos.token_symbol, reason,
                    )
                    _skip_chain_sell = True
            except Exception:
                pass

        if LIVE_TRADING and _was_live_buy and not _skip_chain_sell:
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
                # hard_stop_pp / trailing_stop_pp = same stop logic via PP event-driven thread.
                # Include them so the presigned path fires for both poll and PP-callback triggers.
                _STOP_REASONS   = frozenset({
                    "hard_stop", "trailing_stop",
                    "hard_stop_pp", "trailing_stop_pp",
                })
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
                            # Alert for presigned exits (feed_blind, hard_stop, etc.)
                            # The ladder path has its own alert at the sell confirm block.
                            try:
                                from app.alerts import alert_live_sell
                                # sol_received not measured for presigned — use 0 as placeholder.
                                # Append unconf flag to sig so alert shows uncertainty if needed.
                                _psig_tag = _psig if _pconf else f"{_psig}(unconf)"
                                alert_live_sell(pos, 0.0, _psig_tag)
                            except Exception as _alert_err:
                                log.warning("alert_live_sell (presigned) failed: %s", _alert_err)
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
                    # ── ExitRouter: classify token state + run PumpSwap local path ──────
                    # Additive layer — does NOT replace the executor path below.
                    # PUMPSWAP_LOCAL_SIM_ONLY=True (default): simulate only, then fall through
                    # to executor.  PUMPSWAP_LOCAL_SELL_ENABLED=True: simulate then send if ok.
                    _pumpswap_local_succeeded = False
                    # Tracks whether ExitRouter identified this token as graduated/uncertain so
                    # the executor escalation below doesn't miss it (pos.dex_id may still be
                    # "pumpfun" for Cat-2 tokens that graduated during the hold period).
                    _er_classified_graduated  = False
                    # Initialized here so is_rescue_eligible_error() can safely read them
                    # even if ExitRouter raises or EXIT_ROUTER_ENABLED=False.
                    _exit_state = None
                    _ps_result  = None
                    try:
                        from memecoin.config import EXIT_ROUTER_ENABLED as _er_enabled
                    except ImportError:
                        _er_enabled = False
                    # True when bonding curve oracle confirmed complete=False at buy time.
                    # T22 tokens are always PP-silent but not graduated — they must use
                    # PumpPortal (escalate=False), not PumpSwap/Jupiter rescue.
                    _oracle_bc = "|cohort:bonding_curve" in (pos.notes or "")

                    if _er_enabled:
                        try:
                            from memecoin import exit_router as _er
                            from memecoin import pumpportal_monitor as _ppm_mod
                            _exit_state = _er.classify(pos, _ppm_mod.monitor)
                            _exit_state_tag = f"|exit_state:{_exit_state.value}"
                            if _exit_state_tag not in (pos.notes or ""):
                                pos.notes = (pos.notes or "") + _exit_state_tag
                            # Flag graduated/uncertain so executor uses escalate=True on first attempt.
                            # Without this, Cat-2 tokens that graduated during the hold would hit the
                            # BC path → 6005 → graduated_unsellable → waste a full 60s retry cycle.
                            # Exception: oracle-confirmed bonding curve + MIGRATION_UNCERTAIN =
                            # T22 token still on BC (PP is always silent for T22). Do NOT escalate;
                            # PumpPortal handles these via the normal bonding-curve sell path.
                            if _exit_state in (
                                _er.TokenExitState.GRADUATED_PUMPSWAP,
                                _er.TokenExitState.MIGRATION_UNCERTAIN,
                            ):
                                if _oracle_bc and _exit_state == _er.TokenExitState.MIGRATION_UNCERTAIN:
                                    # Oracle says bonding curve — PP silence is normal for T22.
                                    # Skip escalation and PumpSwap/Jupiter; executor.sell handles it.
                                    log.info(
                                        "ExitRouter: MIGRATION_UNCERTAIN suppressed — "
                                        "oracle-confirmed bonding curve (T22)  token=%s",
                                        pos.token_address[:8],
                                    )
                                else:
                                    _er_classified_graduated = True
                            # Run pumpswap_local for GRADUATED_PUMPSWAP and MIGRATION_UNCERTAIN.
                            # MIGRATION_UNCERTAIN = Cat-2 token that graduated during hold but PP
                            # migration event was missed. Pool RPC lookup at sell time is the
                            # source of truth — if the pool exists, local sell can succeed.
                            # pumpswap_no_pool error class means token is still on BC → executor
                            # handles it normally.
                            # Skip for oracle-confirmed bonding curve — no pool will ever be found.
                            if _exit_state in (
                                _er.TokenExitState.GRADUATED_PUMPSWAP,
                                _er.TokenExitState.MIGRATION_UNCERTAIN,
                            ) and not (_oracle_bc and _exit_state == _er.TokenExitState.MIGRATION_UNCERTAIN):
                                from memecoin.config import CHAINS as _CHAINS_er
                                _chain_cfg_er = _CHAINS_er.get(pos.chain, {})
                                _rpc_url_er   = _chain_cfg_er.get(
                                    "rpc", "https://api.mainnet-beta.solana.com"
                                )
                                _ps_result = _er.run_pumpswap_local_path(pos, reason, _rpc_url_er)
                                _exit_route_tag = f"|exit_route:{_ps_result.get('route', '?')}"
                                if _exit_route_tag not in (pos.notes or ""):
                                    pos.notes = (pos.notes or "") + _exit_route_tag
                                if _ps_result.get("sim_error"):
                                    pos.notes = (pos.notes or "") + f"|sim_err:{_ps_result['sim_error']}"
                                # Only skip executor when local sell was truly sent + confirmed.
                                # SIM_ONLY mode returns success=False → executor runs as normal.
                                if _ps_result.get("success"):
                                    pos.notes = (
                                        (pos.notes or "")
                                        + f"|sell_tx:{_ps_result.get('tx_sig', '')}|pumpswap_local_ok"
                                    )
                                    log.info(
                                        "ExitRouter: PumpSwap local sell succeeded  mint=%s  sig=%s",
                                        pos.token_address[:8], _ps_result.get("tx_sig", "")[:16],
                                    )
                                    _pumpswap_local_succeeded = True
                        except Exception as _er_exc:
                            log.warning("ExitRouter error (non-fatal, falling through): %s", _er_exc)

                    # ── Jupiter rescue: fire before executor if local PumpSwap failed ──
                    _rescue_attempted       = False
                    _rescue_succeeded       = False
                    _rescue_class           = "fallback_allowed"
                    _ps_ec = _ps_result.get("error_class", "") if _ps_result is not None else ""
                    if not _pumpswap_local_succeeded and is_rescue_eligible_error(
                        error_class=_ps_ec,
                        exit_state=_exit_state.value if _exit_state is not None else "",
                        reason=reason,
                        oracle_bonding_curve=_oracle_bc,
                    ):
                        try:
                            from memecoin.jupiter_rescue import (
                                force_jupiter_rescue_sell,
                                classify_rescue_result as _classify_rescue,
                            )
                            try:
                                from app.alerts import _send as _alert_send
                                _alert_send(
                                    f"\u26a1 RESCUE {pos.token_symbol} — "
                                    f"attempting Jupiter rescue (reason={reason})"
                                )
                            except Exception:
                                pass
                            _resc             = force_jupiter_rescue_sell(pos, reason)
                            _rescue_attempted = True
                            _rescue_class     = _classify_rescue(_resc)

                            if _rescue_class == "sold":
                                _rescue_succeeded = True
                                _fill = _resc.get("fill_price") or pos.exit_price
                                if _fill:
                                    pos.exit_price = _fill
                                pos.notes = (pos.notes or "") + (
                                    f"|sell_tx:{_resc.get('tx_sig','')}|sell_fill:{_fill:.10f}"
                                    f"|route:JUPITER_RESCUE"
                                )
                                log.info(
                                    "Jupiter rescue succeeded  %s  sig=%s  sol=%.6f",
                                    pos.token_symbol, (_resc.get("tx_sig") or "")[:16],
                                    _resc.get("sol_received", 0),
                                )
                                try:
                                    from app.alerts import alert_live_sell
                                    alert_live_sell(pos, _resc.get("sol_received", 0), _resc.get("tx_sig", ""))
                                except Exception:
                                    pass

                            elif _rescue_class == "already_sold":
                                # Stale confirmed sig — position was already closed on-chain.
                                # Finalize without re-entering close_position (no double-sell).
                                log.info(
                                    "Jupiter rescue: already_sold  %s — finalizing",
                                    pos.token_symbol,
                                )
                                self._finalize_rescue_sell(pos.id, _resc)
                                return pos

                            elif _rescue_class == "pending":
                                # Tx sent but not yet confirmed — keep pending tag in notes,
                                # arm sell_stuck so retry loop re-checks confirmation.
                                # Never call executor.sell while a tx may still be inflight.
                                log.info(
                                    "Jupiter rescue: tx pending  %s  sig=%s — arming sell_stuck",
                                    pos.token_symbol, (_resc.get("tx_sig") or "")[:16],
                                )
                                try:
                                    from memecoin.config import (
                                        SELL_STUCK_RETRY_SEC as _srs,
                                        JUPITER_RESCUE_PENDING_TTL_SEC as _ttl,
                                    )
                                except ImportError:
                                    _srs = 60; _ttl = 30
                                pos.status = "sell_stuck"
                                self._positions[pos_id] = pos
                                self._sell_stuck_until[pos_id] = time.time() + max(_ttl * 2, _srs)
                                _save_positions(self._positions)
                                return pos

                            else:
                                # no_route / retry_no_send / fatal_no_send:
                                # No tx was sent. Arm controlled retry for rescue-eligible states.
                                # DO NOT call executor.sell — it has no path for graduated/uncertain.
                                log.info(
                                    "Jupiter rescue: %s for %s — arming migration retry, "
                                    "blocking executor.sell",
                                    _rescue_class, pos.token_symbol,
                                )
                                try:
                                    from memecoin.config import SELL_STUCK_RETRY_SEC as _srs
                                except ImportError:
                                    _srs = 60
                                self._arm_migration_retry(pos.id, _srs)
                                return pos

                        except Exception as _resc_exc:
                            log.warning("Jupiter rescue exception (non-fatal): %s", _resc_exc)

                    # Block executor.sell if rescue was attempted and result is not "fallback_allowed".
                    # This covers any exception path where _rescue_class stayed "fallback_allowed"
                    # despite _rescue_attempted=True — in that case we rely on the exception log
                    # and let executor try (the exception means rescue never sent a tx).
                    _rescue_blocks_executor = (
                        _rescue_attempted and _rescue_class not in ("fallback_allowed",)
                    )
                    if not _pumpswap_local_succeeded and not _rescue_succeeded and not _rescue_blocks_executor:
                        ex     = MemeExecutor()
                        # escalate=True when:
                        #   (a) this is a retry — previous full ladder failed
                        #   (b) token is graduated — pool="auto" returns Custom:6005;
                        #       escalate path uses pump-amm → Jupiter fallback.
                        #       Sources: cohort:graduated note (set at entry for Cat-3),
                        #       "graduated_exit" reason, OR ExitRouter classification
                        #       (catches Cat-2 tokens that graduated during hold).
                        _is_retry     = getattr(pos, "sell_attempts", 0) > 0
                        _is_graduated = (
                            "|cohort:graduated" in (pos.notes or "")
                            or reason == "graduated_exit"
                            or _er_classified_graduated
                        )
                    result = {}  # default: no executor result (set below if executor runs)
                    if not _pumpswap_local_succeeded and not _rescue_succeeded and not _rescue_blocks_executor:
                        if _is_graduated and not _is_retry:
                            log.info("SELL graduated token — pump-amm PRIMARY, Jupiter FALLBACK  token=%s",
                                     pos.token_address[:8])
                        _URGENT_REASONS = frozenset({
                            "hard_stop", "hard_stop_pp", "trailing_stop", "trailing_stop_pp",
                            "feed_blind", "graduated_exit", "dev_dump", "rug_lp", "velocity",
                            "abort_tripwire", "pre_graduation_exit",
                        })
                        result = ex.sell(
                            pos.token_address, pos.size_usd, pos.entry_price, pos.chain,
                            # Never escalate oracle-confirmed bonding curve tokens (T22).
                            # Escalation assumes PumpSwap graduation — BC tokens must always
                            # stay on PumpPortal path, even on retry (retry only ups slippage).
                            escalate=(False if _oracle_bc else (_is_retry or _is_graduated)),
                            urgent=(reason in _URGENT_REASONS),
                            # Pass tokens_held so local build can use exact count without RPC.
                            # Only valid for full exits (fraction=1.0 default); partial TPs
                            # pass known_token_count separately in _run_tp_sell_bg.
                            known_token_count=int(pos.tokens_held or 0),
                        )
                    if not _pumpswap_local_succeeded and result.get("success"):
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
                    elif not _pumpswap_local_succeeded and result.get("reason") == "zero_balance":
                        log.warning("Live sell %s — zero balance, tokens already sold. Closing.",
                                    pos.token_symbol)
                        pos.notes = (pos.notes or "") + "|sell_already_gone"
                        # Close immediately — do not fall through to retry handler.
                        # Retrying a zero-balance sell produces graduated_unsellable which
                        # triggers a 3-cycle loss write-off at $0 with misleading alerts.
                        pos.status      = "closed"
                        pos.exit_price  = pos.exit_price or 0.0
                        pos.exit_time   = time.time()
                        pos.exit_reason = "zero_balance"
                        self._positions[pos_id] = pos
                        _append_journal(pos)
                        del self._positions[pos_id]
                        _save_positions(self._positions)
                        with self._presigned_lock:
                            self._presigned_exits.pop(pos.token_address, None)
                            self._presigned_ts.pop(pos.token_address, None)
                            self._graduated_mints.discard(pos.token_address)
                        try:
                            from app.alerts import alert_position_close
                            alert_position_close(pos)
                        except Exception:
                            pass
                        return pos
                    elif not _pumpswap_local_succeeded and result.get("reason") == "graduated_unsellable":
                        # pump-amm + Jupiter both failed — token is mid-migration or pool is empty.
                        # 3 retries covers genuine migration lag (~2-3 min to settle).
                        # Honeypots will fail all 3 immediately — write off fast, stop burning fees.
                        MAX_GRADUATED_RETRIES = 3
                        _grad_attempts = self._graduated_retry_count.get(pos_id, 0) + 1
                        self._graduated_retry_count[pos_id] = _grad_attempts
                        if _grad_attempts >= MAX_GRADUATED_RETRIES:
                            # ── Fix B: check sigs + on-chain balance before writing $0 ──
                            import re as _re_gl
                            from memecoin.tx_meta import read_sol_delta as _rsd_gl
                            try:
                                from memecoin.config import WALLET_PUBKEY as _gl_wallet
                            except ImportError:
                                _gl_wallet = ""
                            _gl_recovered = False
                            if _gl_wallet:
                                _gl_sigs = _re_gl.findall(
                                    r'(?:sell_tx|sell_unconf|jupiter_rescue_pending|sell_pending):([A-Za-z0-9]+)',
                                    pos.notes or "",
                                )
                                for _gl_sig in reversed(_gl_sigs):
                                    _gl_res = _rsd_gl(_gl_sig, _gl_wallet)
                                    if _gl_res.get("ok") and (_gl_res.get("sol_delta") or 0) > 0:
                                        _gl_sol_delta = _gl_res["sol_delta"]
                                        # Convert chain delta to USD/token so exit_price matches entry_price units.
                                        try:
                                            from memecoin.executor import _sol_price_usd as _gl_sol_price_fn
                                            _gl_sol_price = _gl_sol_price_fn()
                                        except Exception:
                                            _gl_sol_price = 0.0
                                        # Tokens in this exit = remaining_fraction of original raw count.
                                        _gl_tokens_raw = (pos.tokens_held or 0) * (pos.remaining_fraction or 1.0)
                                        if _gl_tokens_raw > 0:
                                            _gl_tokens_human = _gl_tokens_raw / 1e6  # pump.fun decimals=6
                                        elif pos.entry_price > 0 and pos.size_usd > 0:
                                            # Fallback: estimate token count from entry cost
                                            _gl_tokens_human = (pos.size_usd * pos.remaining_fraction) / pos.entry_price
                                        else:
                                            _gl_tokens_human = 0.0
                                        if _gl_tokens_human > 0 and _gl_sol_price > 0:
                                            pos.exit_price = (_gl_sol_delta * _gl_sol_price) / _gl_tokens_human
                                        else:
                                            # Cannot compute USD/token — store 0; reconciler will fix
                                            pos.exit_price = 0.0
                                        pos.sol_received = _gl_sol_delta
                                        pos.exit_reason = "graduated_recovered"
                                        pos.notes = (
                                            (pos.notes or "")
                                            + f"|graduated_recovered:{_gl_sig[:8]}"
                                            + f"|sol_received:{_gl_sol_delta:.8f}"
                                        )
                                        log.warning(
                                            "graduated_recovered: sig confirmed sol_delta=%.6f  "
                                            "exit_price=%.8f USD/tok  sig=%s  pos=%s",
                                            _gl_sol_delta, pos.exit_price, _gl_sig[:16], pos_id,
                                        )
                                        _gl_recovered = True
                                        break

                                if not _gl_recovered:
                                    # Check on-chain balance — if tokens still held, defer
                                    try:
                                        from memecoin.executor import _token_balance as _gl_tb
                                        _gl_bal = _gl_tb(_gl_wallet, pos.token_address)
                                    except Exception:
                                        _gl_bal = -1
                                    if _gl_bal > 0:
                                        log.warning(
                                            "graduated_loss DEFERRED — tokens still on-chain (%d)  "
                                            "pos=%s  mint=%s",
                                            _gl_bal, pos_id, pos.token_address[:8],
                                        )
                                        self._arm_migration_retry(pos.id, 60)
                                        return pos

                            if _gl_recovered:
                                # Fall through to normal close with real fill
                                self._graduated_retry_count.pop(pos_id, None)
                                self._sell_stuck_until.pop(pos_id, None)
                                pos.status    = "closed"
                                pos.exit_time = time.time()
                                self._positions[pos_id] = pos
                                _append_journal(pos)
                                del self._positions[pos_id]
                                _save_positions(self._positions)
                                with self._presigned_lock:
                                    self._presigned_exits.pop(pos.token_address, None)
                                    self._presigned_ts.pop(pos.token_address, None)
                                    self._graduated_mints.discard(pos.token_address)
                                return pos

                            # Migration never settled or pool is permanently empty.
                            # Write off as total loss — no more Helius/RPC burn.
                            self._graduated_retry_count.pop(pos_id, None)
                            self._sell_stuck_until.pop(pos_id, None)
                            pos.status      = "closed"
                            pos.exit_price  = 0.0
                            pos.exit_time   = time.time()
                            pos.exit_reason = "graduated_loss"
                            pos.notes = (pos.notes or "") + f"|graduated_loss_after_{_grad_attempts}_retries"
                            self._positions[pos_id] = pos
                            _append_journal(pos)          # write CSV before removing from dict
                            del self._positions[pos_id]   # remove from live tracking
                            _save_positions(self._positions)
                            with self._presigned_lock:    # clean up presigned exit
                                self._presigned_exits.pop(pos.token_address, None)
                                self._presigned_ts.pop(pos.token_address, None)
                                self._graduated_mints.discard(pos.token_address)
                            log.error(
                                "GRADUATED LOSS %s — %d retries exhausted, writing off as $0.  mint=%s",
                                pos.token_symbol, _grad_attempts, pos.token_address,
                            )
                            try:
                                from app.alerts import _send
                                _send(
                                    f"\U0001f480 SELL FAILED {pos.token_symbol} — "
                                    f"pump-amm + Jupiter failed {_grad_attempts}x. "
                                    f"Jupiter rescue also failed. "
                                    f"Manual rescue needed in Phantom:\n"
                                    f"mint: {pos.token_address}"
                                )
                            except Exception:
                                pass
                            return pos
                        # Not yet at limit — schedule another retry in 60s
                        pos.status = "sell_stuck"
                        pos.exit_price  = 0.0
                        pos.exit_time   = 0.0
                        pos.exit_reason = ""
                        pos.sell_attempts = 0
                        if "|graduated_unsellable" not in (pos.notes or ""):
                            pos.notes = (pos.notes or "") + "|graduated_unsellable"
                        self._positions[pos_id] = pos
                        self._sell_stuck_until[pos_id] = time.time() + 60
                        _save_positions(self._positions)
                        log.error(
                            "SELL STUCK (graduated_unsellable) %s — pump-amm + Jupiter failed, "
                            "retry %d/%d in 60s.  mint=%s",
                            pos.token_symbol, _grad_attempts, MAX_GRADUATED_RETRIES, pos.token_address,
                        )
                        return pos
                    elif not _pumpswap_local_succeeded:
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
                            self._sell_stuck_until[pos_id] = time.time() + SELL_STUCK_RETRY_SEC
                            _save_positions(self._positions)
                            log.error(
                                "SELL STUCK %s — ladder exhausted, position stays open, "
                                "retry in %ds.  mint=%s",
                                pos.token_symbol, SELL_STUCK_RETRY_SEC, pos.token_address,
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
                self._sell_stuck_until[pos_id] = time.time() + SELL_STUCK_RETRY_SEC
                _save_positions(self._positions)
                log.error("SELL STUCK %s (exception path) — stays open, retry in %ds: %s",
                          pos.token_symbol, SELL_STUCK_RETRY_SEC, e)
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
        self._graduated_retry_count.pop(pos_id, None)  # clean up retry counter
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

    def _finalize_rescue_sell(self, pos_id: str, rescue_result: dict) -> None:
        """
        Close a position whose sell was already executed by force_jupiter_rescue_sell()
        from outside the normal close_position() path (e.g. scanner.py migration branch).

        Never calls executor.sell() or close_position(). Sets status=closed, journals,
        and sends the sell alert. Safe to call even if the position was concurrently
        removed (no-ops in that case).
        """
        with self._close_locks_meta:
            if pos_id not in self._close_locks:
                self._close_locks[pos_id] = threading.Lock()
            _lock = self._close_locks[pos_id]
        with _lock:
            pos = self._positions.get(pos_id)
            if not pos or pos.status == "closed":
                log.debug("_finalize_rescue_sell: pos %s already closed or missing — no-op", pos_id)
                return

            sig        = rescue_result.get("tx_sig", "")
            sol_recv   = float(rescue_result.get("sol_received") or 0.0)
            fill_price = rescue_result.get("fill_price") or pos.exit_price or 0.0

            pos.status      = "closed"
            pos.exit_reason = "jupiter_rescue"
            pos.exit_time   = time.time()
            pos.exit_price  = fill_price
            pos.notes = (pos.notes or "") + (
                f"|sell_tx:{sig}|route:JUPITER_RESCUE"
                f"|sol_received:{sol_recv:.6f}"
                + (f"|sell_fill:{fill_price:.10f}" if fill_price else "")
            )

            self._positions[pos_id] = pos
            _append_journal(pos)
            promote_to_winners(pos)
            del self._positions[pos_id]
            _save_positions(self._positions)
            self._sell_stuck_until.pop(pos_id, None)
            self._graduated_retry_count.pop(pos_id, None)

            try:
                from memecoin.pumpportal_monitor import monitor as _ppmon
                _ppmon.clear_creator(pos.token_address)
            except Exception:
                pass
            with self._presigned_lock:
                self._presigned_exits.pop(pos.token_address, None)
                self._presigned_ts.pop(pos.token_address, None)
                self._graduated_mints.discard(pos.token_address)

            log.info(
                "_finalize_rescue_sell: closed %s  sig=%s  sol=%.6f  fill=%.10f",
                pos.token_symbol, sig[:16] if sig else "", sol_recv, fill_price,
            )
            try:
                from app.alerts import alert_live_sell
                alert_live_sell(pos, sol_recv, sig)
            except Exception:
                pass
            try:
                alerts.alert_position_close(pos)
            except Exception:
                pass

    def _arm_migration_retry(self, pos_id: str, retry_sec: float) -> None:
        """
        Arm a MIGRATION_UNCERTAIN + no-pool position for sell_stuck retry.
        Sets sell_stuck status, records |migration_wait| + first-detection timestamp,
        and sets the retry timer. Called from scanner.py no-pool branch only.
        Never calls executor.sell().
        """
        pos = self._positions.get(pos_id)
        if not pos:
            return
        if "|migration_wait" not in (pos.notes or ""):
            pos.notes = (pos.notes or "") + f"|migration_wait|migration_uncertain_ts:{int(time.time())}"
        elif "|migration_uncertain_ts:" not in (pos.notes or ""):
            pos.notes = (pos.notes or "") + f"|migration_uncertain_ts:{int(time.time())}"
        pos.status = "sell_stuck"
        self._positions[pos_id] = pos
        self._sell_stuck_until[pos_id] = time.time() + retry_sec
        _save_positions(self._positions)

    def _save(self) -> None:
        """Persist current positions to disk (thin wrapper for scanner.py callers)."""
        _save_positions(self._positions)

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

            # FIX 3: per-position tick log — entry-to-exit price path for offline replay.
            # Appends (epoch, price_usd) every monitoring cycle (~2s) to
            # logs/price_paths/<mint>.csv. Used to optimize TRAIL_ACTIVATES_PCT /
            # TRAILING_STOP_PCT from real paths instead of guessing.
            if pos.current_price > 0:
                _append_price_tick(pos, pos.current_price)

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

    def manual_close_live(self, symbol: str, exit_price: float = 0.0) -> dict:
        """
        Close an open live position by symbol without attempting an on-chain sell.

        Use this when the user has already sold the position manually (e.g. on Phantom)
        and wants the bot to acknowledge the close, journal it, and stop looping.

        Called by the Telegram /manual_sold command.

        Returns {"ok": bool, "pos_id": str, "symbol": str, "exit_reason": str, "msg": str}
        """
        # Find the open live position by symbol (case-insensitive, match first found)
        sym_upper = symbol.strip().upper()
        target = None
        for pos in self._positions.values():
            if pos.status in ("open", "sell_stuck") and pos.notes and "live|tx:" in pos.notes:
                if pos.token_symbol.upper() == sym_upper:
                    target = pos
                    break

        if target is None:
            # Try partial match as fallback
            for pos in self._positions.values():
                if pos.status in ("open", "sell_stuck") and pos.notes and "live|tx:" in pos.notes:
                    if sym_upper in pos.token_symbol.upper():
                        target = pos
                        break

        if target is None:
            return {"ok": False, "msg": f"No open live position found for symbol '{symbol}'"}

        price = exit_price or target.current_price or target.entry_price
        target.notes = (target.notes or "") + "|manual_sold_via_tg"
        # Clear sell_stuck so close_position doesn't throttle
        self._sell_stuck_until.pop(target.id, None)
        self.close_position(target.id, "manual_sell", price)
        log.warning(
            "MANUAL SELL via Telegram: %s  pos=%s  price=%.10f",
            target.token_symbol, target.id, price,
        )
        return {
            "ok":    True,
            "pos_id": target.id,
            "symbol": target.token_symbol,
            "exit_reason": "manual_sell",
            "msg": f"Closed {target.token_symbol} as manual_sell at ${price:.8g}",
        }

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


# ---------------------------------------------------------------------------
# Module-level helper for executor.py (used by 6005 → local PumpSwap path)
# ---------------------------------------------------------------------------

def _get_open_position_by_token(token_address: str):
    """
    Return the first open Position (live or paper) matching token_address, or None.
    Used by executor.py's 6005-detected path to get a pos object for exit_router.
    """
    try:
        from memecoin.scanner import portfolio as _portfolio
        for pos in _portfolio.open_positions():
            if pos.token_address == token_address:
                return pos
    except Exception:
        pass
    # Fallback: load directly from positions file
    try:
        positions = _load_positions()
        for pos in positions.values():
            if pos.token_address == token_address and not pos.closed:
                return pos
    except Exception:
        pass
    return None
