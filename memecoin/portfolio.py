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


def effective_hard_stop_level(signal_price: float, entry_price: float, hard_stop_pct: float) -> float:
    """Compute effective hard-stop level = max(signal-anchored stop, fill-loss floor).

    Prevents TROONCH-style losses when slippage is high, while preserving
    signal structure when fill is close to signal.
    Paper behaviour unchanged: paper entry_price == signal_price.
    """
    try:
        from memecoin.config import MAX_LOSS_FROM_FILL_PCT as _mlffp
    except ImportError:
        _mlffp = 0.50  # default: 50% max loss from fill (stub-safe fallback)
    _sa = signal_price * (1 + hard_stop_pct) if signal_price > 0 else 0.0
    _fl = entry_price * (1 - _mlffp)
    return max(_sa, _fl)


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
    # P1': three-price benchmark fields (added 2026-07-09)
    "signal_dex_price",      # DexScreener price at signal time (stale indexer snapshot)
    "baseline_curve_price",  # preflight curve/PP baseline price at decision time
    "fill_price_field",      # actual on-chain fill price (blank for paper-only signals)
    "entry_source",          # "curve" | "pp_tick" | "dex_stale" | ""
]

# Stamp applied to every trade written from this session onward
CONFIG_TAG = "v7_entry_filters_2026-06-06"

# Accounting epoch — bump when position accounting logic changes so we can
# split reports cleanly.  Past rows are backfilled by tools/v7_journal_corrected.py.
# e1_baseline             : pre-2026-06-11 01:20 UTC (no PP exits, simple pnl)
# e2_pp_exits             : commit 81de8da — PP real-time exits wired to paper
# e3_pp_entries_anchored_stops: commit 9a2a332 — signal-anchored stops + this accounting fix
ACCOUNTING_EPOCH = "e4_rt_feed_quote_gate"


# ---------------------------------------------------------------------------
# C2 — Live entry program gate
# ---------------------------------------------------------------------------

def evaluate_live_entry_program_gate(
    classification,   # MintClassification from mint_classifier.py
    curve_observation=None,  # dict with "complete" key, or None
    config=None,      # unused for now, reserved
) -> dict:
    """
    Gate a live buy based on mint classification.

    Returns:
        {"allowed": True}  — proceed with buy
        {"allowed": False, "reason": str, "token_program": str}  — block buy

    Rules:
        - If classification is None or classification.error is set → block (unknown program)
        - If classification.token_program == "UNKNOWN" → block
        - If classification.is_tradeable is False (unsupported extensions) → block
        - SPL or T22_CLEAN → allow (T22 BC sells work; Jupiter rescue handles post-graduation)
    """
    if classification is None:
        return {
            "allowed": False,
            "reason": "unknown program: classification unavailable",
            "token_program": "UNKNOWN",
        }

    if classification.error is not None:
        return {
            "allowed": False,
            "reason": f"unknown program: classification error — {classification.error}",
            "token_program": getattr(classification, "token_program", "UNKNOWN"),
        }

    tp = classification.token_program

    if tp == "UNKNOWN":
        return {
            "allowed": False,
            "reason": "unknown token program",
            "token_program": "UNKNOWN",
        }

    if not classification.is_tradeable:
        unsup = getattr(classification, "unsupported_extensions", [])
        return {
            "allowed": False,
            "reason": f"unsupported extensions: {unsup}",
            "token_program": tp,
        }

    # SPL or T22_CLEAN — allow.
    # T22 bonding-curve sells work (bonding_curve_t22.py).
    # Post-graduation PumpSwap local fails for T22, but MU retry escalates to Jupiter rescue.
    return {"allowed": True}


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
    is_live: bool = False          # 4E: set True exactly once at live buy confirm; all routing keys off this
    mu_sell_total: int = 0         # 4D: cumulative sell windows attempted across all sell_stuck re-arms
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
    # P1': price benchmark fields
    signal_dex_price: float = 0.0    # DexScreener price at signal time (stale)
    baseline_curve_price: float = 0.0  # preflight curve/PP baseline at decision time
    fill_price_recorded: float = 0.0  # actual on-chain fill price (0 for paper-only)
    entry_source: str = ""           # "curve"|"pp_tick"|"dex_stale"|""
    baseline_price: float = 0.0   # preflight curve/PP baseline (same ref as stops/sizing)  # Phase 6.1
    # Z2/Z7: Structured execution state fields (epoch-protective, serialized with position)
    policy_cohort: str = ""           # strategy_pure_rider | legacy_graduation_guard | paper_reference
    lifecycle_state: str = ""         # bonding_curve | graduated | unknown | ""
    exit_intent_reason: str = ""      # original exit reason set when intent created (never overwritten by routing)
    exit_intent_ts: float = 0.0       # when exit intent was created
    exit_intent_policy: str = ""      # policy_cohort that created the exit intent
    venue_state_json: str = ""        # JSON-encoded per-venue state: {"primary": "pump_amm"|"bonding_curve"}
    pending_signature: str = ""       # last TX sig sent — preserved through restarts for duplicate-sell guard
    pending_signature_route: str = "" # route used for pending_signature
    pending_signature_ts: float = 0.0 # when pending_signature was sent

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

    @pnl_pct.setter
    def pnl_pct(self, _):
        raise AttributeError("pnl_pct is computed — set exit_price or realized_pnl_usd instead")

    @pnl_usd.setter
    def pnl_usd(self, _):
        raise AttributeError("pnl_usd is computed — set exit_price or realized_pnl_usd instead")


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
            # 4E backfill: existing positions serialized before is_live field was added
            if "is_live" not in d:
                d["is_live"] = bool(d.get("notes") and "live|tx:" in d["notes"])
            # Z2/Z7 backfill: structured execution state fields
            if "policy_cohort" not in d:
                d["policy_cohort"] = (
                    "strategy_pure_rider" if d.get("is_live") else "paper_reference"
                )
            if "lifecycle_state" not in d:
                _notes_bf = d.get("notes", "") or ""
                if "|cohort:graduated" in _notes_bf:
                    d["lifecycle_state"] = "graduated"
                elif "|cohort:bonding_curve" in _notes_bf:
                    d["lifecycle_state"] = "bonding_curve"
                else:
                    d["lifecycle_state"] = ""
            # Strip unknown keys so Position(**d) doesn't raise on old snapshots
            _known = set(Position.__dataclass_fields__)  # type: ignore[attr-defined]
            d = {k: v for k, v in d.items() if k in _known}
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
        # P1': three-price benchmark fields
        "signal_dex_price": pos.signal_dex_price or "",
        "baseline_curve_price": pos.baseline_curve_price or "",
        "fill_price_field": pos.fill_price_recorded or "",
        "entry_source": pos.entry_source or "dex_stale",
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


# JOURNAL CHOKE POINT — ALL close paths must call this function.
# Verify with: grep -n "_append_journal\|JOURNAL CHOKE POINT" portfolio.py
# Every path: close_position (normal, abort_tripwire, zero_balance, graduated_loss,
#             graduated_recovered), _finalize_rescue_sell (both fill and pending).
# abort_tripwire path at line ~1442 also calls _append_journal directly.
def _append_journal(pos: Position):
    # ── Telemetry: journal write ──
    try:
        from memecoin import telemetry as _tel
        _jt = _tel.get_trace_id_for_pos(pos.id)
        if _jt:
            _tel.event(_jt, "journal_write_started", pos_id=pos.id)
    except Exception:
        pass

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

        # ── Telemetry: journal written ──
        try:
            _jt2 = _tel.get_trace_id_for_pos(pos.id)
            if _jt2:
                _tel.event(_jt2, "journal_written",
                    pos_id=pos.id,
                    exit_price=pos.exit_price,
                    pnl_usd=pos.pnl_usd,
                    exit_reason=pos.exit_reason,
                    fill_estimated="|entry_estimated" in (pos.notes or ""),
                )
        except Exception:
            pass

        # If this was a live trade, also write to the live journal
        if pos.is_live:
            _ensure_journal_header(LIVE_JOURNAL_FILE)
            write_header = not LIVE_JOURNAL_FILE.exists() or LIVE_JOURNAL_FILE.stat().st_size == 0
            with open(LIVE_JOURNAL_FILE, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

    # DRY_RUN funnel counter — read + alert outside lock (non-critical, can be eventually consistent)
    if pos.is_live and "DRY_RUN" in (pos.notes or ""):
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
        # Per-position close lock: prevents double-sell when the 0.5s poll loop and
        # the event-driven exit queue both fire for the same position simultaneously.
        # Key: pos_id → Lock.  Acquired at close_position entry, released after
        # pos.status = "closed" is set.  Second caller blocks then sees "closed" → no-op.
        self._close_locks: dict[str, threading.Lock] = {}
        self._close_locks_meta = threading.Lock()  # guards _close_locks dict itself
        # sell_stuck throttle: pos_id → earliest time to retry sell
        # In-memory only — resets on restart (which itself gives a fresh attempt).
        self._sell_stuck_until: dict[str, float] = {}
        # 4C: TP inflight guard — never >1 concurrent TP thread per position per level
        # {pos_id: {level_key: earliest_retry_time}}  (0 = in-flight, future = cooldown)
        self._tp_inflight: dict[str, dict[str, float]] = {}
        # graduated_unsellable retry counter: pos_id → attempts
        # After MAX_GRADUATED_RETRIES the position is written off as a total loss.
        self._graduated_retry_count: dict[str, int] = {}
        # B4: Per-position per-venue state for the graduation fast window.
        # {pos_id: {venue_name: {"cooldown_until": float, "attempts": int, "last_result": str}}}
        self._venue_state: dict[str, dict[str, dict]] = {}
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
            # P1': capture DexScreener price at signal time for benchmark comparison
            signal_dex_price=getattr(signal, "_price_dex", 0.0) or signal.price_usd or 0.0,
            entry_source="pp_tick" if _use_pp else "dex_stale",
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

        pos.policy_cohort = "paper_reference"  # Z7: paper positions never execute on-chain
        self._positions[pos.id] = pos
        _save_positions(self._positions)
        log.info("Opened paper position %s  %s/%s @ $%.8f  dex=%s",
                 pos.id, pos.chain, pos.token_symbol, pos.entry_price, pos.dex_id)
        if signal.signal_type == "social_alert":
            try:
                from memecoin.health_monitor import bump_social_alert_paper as _bsap
                _bsap()
            except Exception:
                pass

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
            _curve_snap        = None  # populated if oracle path taken; passed to executor
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
                            _curve_snap["_preflight_ts"] = time.time()  # age stamp for executor passthrough
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
                            _curve_snap2["_preflight_ts"] = time.time()  # L1b: stamp for executor passthrough
                            _curve_complete2 = _curve_snap2.get("complete")
                            _curve_reason2   = _curve_snap2.get("reason", "")
                            if _curve_complete2 is False and (_curve_snap2.get("price_usd") or 0) > 0:
                                _pp_at_gate = _curve_snap2["price_usd"]
                                _baseline_source2 = "curve"
                                _curve_snap = _curve_snap2  # L1b: pass through to executor oracle gate
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
                _eff_stop = effective_hard_stop_level(
                    _sig_price, _pp_price, paper_pos.hard_stop_pct
                )
                _stop_dist = abs(_pp_price - _eff_stop) / _pp_price if _pp_price > 0 else abs(paper_pos.hard_stop_pct)
                if _stop_dist > 0:
                    _size_mult = _base_stop_pct / _stop_dist
                    _raw_size_mult = _size_mult            # before any clamp
                    _size_mult = max(0.5, min(1.0, _size_mult))
                    _orig_size = _live_size
                    _live_size = round(_live_size * _size_mult, 2)
                    live_pos.size_usd = _live_size  # fix: sync position size after norm so pnl_usd is correct
                    log.info(
                        "SIZE NORM %s: pp=%.8f stop=%.8f dist=%.1f%% "
                        "mult=%.2f  size $%.2f→$%.2f",
                        live_pos.token_symbol, _pp_price, _eff_stop,
                        _stop_dist * 100, _size_mult, _orig_size, _live_size,
                    )

                    # ── Shadow size-floor reporting — no behavior change ──
                    _hyp_size_mult_025 = max(0.25, min(1.0, _raw_size_mult))
                    _live_size_used_usd = _orig_size * _size_mult
                    _hyp_size_025_usd = _orig_size * _hyp_size_mult_025
                    _drift_pct = (_pp_price / _sig_price - 1.0) * 100 if _sig_price > 0 else 0.0
                    _stop_dist_from_fill = (_eff_stop / _pp_price - 1.0) * 100 if _pp_price > 0 else 0.0
                    log.info(
                        "SIZE_SHADOW %s: raw_mult=%.2f live_mult=%.2f hyp_mult_025=%.2f "
                        "live_size=$%.2f hyp_size_025=$%.2f drift=%.1f%% eff_stop=%.8g stop_dist=%.1f%%",
                        live_pos.token_symbol, _raw_size_mult, _size_mult, _hyp_size_mult_025,
                        _live_size_used_usd, _hyp_size_025_usd, _drift_pct, _eff_stop, _stop_dist_from_fill,
                    )
                    try:
                        from memecoin import telemetry as _tel
                        _shadow_trace = _tel.get_trace_id_for_pos(live_pos.id)
                        if _shadow_trace:
                            _tel.event(_shadow_trace, "size_shadow",
                                raw_mult=round(_raw_size_mult, 4),
                                live_mult=round(_size_mult, 4),
                                hyp_mult_025=round(_hyp_size_mult_025, 4),
                                live_size_usd=round(_live_size_used_usd, 2),
                                hyp_size_025_usd=round(_hyp_size_025_usd, 2),
                                drift_pct=round(_drift_pct, 2),
                                eff_stop=_eff_stop,
                                stop_dist_from_fill=round(_stop_dist_from_fill, 2),
                            )
                    except Exception:
                        pass

            # P1': store preflight curve baseline on paper position for honest benchmark
            # entry_source = "baseline_curve" when fetched from bonding curve RPC snapshot
            # entry_source = "baseline_pp"    when fetched from PumpPortal live price
            # The paper entry_price stays at signal time (PP or DexScreener fallback);
            # baseline_curve_price is the *preflight* price at decision time for replay audit.
            _pf_baseline_source = (_baseline_source if '_baseline_source' in dir() else
                                   (_baseline_source2 if '_baseline_source2' in dir() else ""))
            if _pp_price > 0:
                paper_pos.baseline_curve_price = _pp_price
                # Phase 3.1: paper entry anchored to preflight curve baseline (not stale DexScreener)
                paper_pos.entry_price = _pp_price
                paper_pos.current_price = _pp_price
                paper_pos.peak_price = _pp_price
                _bs_label = _pf_baseline_source or "pp"
                # Phase 3.2: renamed tags: "curve" / "pp_tick" / "dex_stale"
                paper_pos.entry_source = (
                    "curve" if "curve" in _bs_label
                    else "pp_tick"
                )
                # Phase 6.1: persistent baseline_price for abort comparison
                paper_pos.baseline_price = _pp_price
                live_pos.baseline_price = _pp_price

            # ── Telemetry: preflight done, buy build starting ──
            _entry_trace_id = getattr(signal, '_telemetry_trace_id', '') or ''
            _buy_build_start_ts = time.time()
            try:
                from memecoin import telemetry as _tel
                if _entry_trace_id:
                    # Update trace with actual pos_id now that we know it
                    with _tel._traces_lock:
                        _tmeta = _tel._traces.get(_entry_trace_id)
                        if _tmeta:
                            _tmeta["pos_id"] = live_pos.id
                    _tel.event(_entry_trace_id, "preflight_started",
                        preflight_ts=getattr(self, '_t_preflight_start', _buy_build_start_ts),
                    )
                    _tel.event(_entry_trace_id, "preflight_baseline_selected",
                        baseline_source=_baseline_source if '_baseline_source' in dir() else "unknown",
                        pp_price=_pp_price,
                        sig_price=_sig_price,
                    )
                    _tel.event(_entry_trace_id, "buy_build_started",
                        buy_build_start_ts=_buy_build_start_ts,
                        live_size_usd=_live_size,
                    )
            except Exception:
                pass

            # ── C2: Live entry program gate ──────────────────────────────────
            try:
                from memecoin.health_monitor import bump_live_attempt as _bla
                _bla()
            except Exception:
                pass
            try:
                from memecoin import mint_classifier as _mc
                _mint_cls = _mc.classify_mint(signal.token_address)
            except Exception as _mc_exc:
                log.warning("PROGRAM GATE: mint_classifier failed for %s: %s — blocking buy",
                            live_pos.token_symbol, _mc_exc)
                _mint_cls = None
            _curve_obs = _curve_snap if '_curve_snap' in dir() else None
            _pg_result = evaluate_live_entry_program_gate(_mint_cls, curve_observation=_curve_obs)
            if not _pg_result.get("allowed"):
                _pg_reason = _pg_result.get("reason", "unknown")
                _pg_tp     = _pg_result.get("token_program", "UNKNOWN")
                log.warning(
                    "PROGRAM GATE BLOCKED %s  token_program=%s  reason=%s",
                    live_pos.token_symbol, _pg_tp, _pg_reason,
                )
                try:
                    from memecoin.gate_logger import log_gate_block as _lgb
                    _lgb("program_gate", live_pos.chain, live_pos.token_address,
                         live_pos.token_symbol, pp_price=_pp_price if '_pp_price' in dir() else 0.0,
                         signal_price=_sig_price or 0,
                         size_usd=_live_size)
                except Exception:
                    pass
                try:
                    from memecoin.health_monitor import bump_gate_block as _bgb
                    _bgb(f"program_gate:{_pg_tp}:{_pg_reason[:60]}")
                except Exception:
                    pass
                return
            # ── end C2 ───────────────────────────────────────────────────────

            ex = MemeExecutor()
            result = ex.buy(signal.token_address, _live_size, signal.chain,
                            signal_price=_exec_signal_price,
                            max_slippage_pct=0.30,
                            dex_id=live_pos.dex_id,
                            preflight_oracle_result=_curve_snap)

            _buy_done_ts = time.time()
            # P8: set post-buy quiet window so Helius standby WS doesn't reconnect
            # during buy TX propagation (contends with confirmation)
            try:
                from memecoin.helius_account_monitor import helius_monitor as _ham
                _ham.set_post_buy_quiet()
            except Exception:
                pass
            try:
                if _entry_trace_id:
                    _tel.event(_entry_trace_id, "buy_build_done",
                        buy_done_ts=_buy_done_ts,
                        build_ms=round((_buy_done_ts - _buy_build_start_ts) * 1000, 1),
                        success=result.get("success", False),
                    )
            except Exception:
                pass

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
                # P5': Abort reference uses preflight CURVE BASELINE, not Jupiter quote.
                # Priority:
                # 1. _pp_price — preflight curve/PP baseline at decision time (FRESHEST)
                # 2. _pp_at_gate — PP price captured at gate time (dex-source type-2 path)
                # 3. Jupiter quote — live AMM price at build time
                # 4. Missing — skip abort, tag note, log warning
                # Never use DexScreener-derived price (stale 10-30s).
                # Note: Dog still aborts under this change (+61.5% vs baseline ~$0.0000214).
                # Comment: "Dog still aborts under this change (+48% vs baseline)."
                # Phase 6.2: prefer live_pos.baseline_price (persistent) over local _pp_price var
                _preflight_baseline = live_pos.baseline_price if live_pos.baseline_price > 0 else (_pp_price if '_pp_price' in dir() else 0)
                if _preflight_baseline > 0:
                    _abort_ref = _preflight_baseline
                    _abort_ref_label = "preflight_baseline"
                elif _pp_at_gate > 0:
                    _abort_ref = _pp_at_gate
                    _abort_ref_label = "pp_gate"
                elif _jup_ref > 0:
                    _abort_ref = _jup_ref
                    _abort_ref_label = "jup_quote"
                else:
                    _abort_ref = 0
                    _abort_ref_label = "missing"
                # Shadow log for 10 trades (always log, helps verify reference is correct)
                log.info(
                    "ABORT_REF_SHADOW fill=%.10f baseline=%.10f pp_gate=%.10f jup_quote=%.10f "
                    "ref_used=%s token=%s",
                    fill_price, _preflight_baseline, _pp_at_gate if '_pp_at_gate' in dir() else 0,
                    _jup_ref, _abort_ref_label, live_pos.token_symbol,
                )
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
                    # Phase 3.4: abort closes live_pos only; paper_pos continues independently — no duplicate paper row
                    # Write abort row to live journal so the burn is visible and auditable
                    # Phase 4.4: compute REAL pnl from abort_sell result
                    _abort_sol_recv = float((abort_sell or {}).get("sol_received") or 0.0)
                    _abort_sol_spent_usd = result.get("sol_spent_usd") or 0.0
                    _sol_price_est = 70.0  # fallback estimate
                    try:
                        from memecoin.config import SOL_USD_PRICE as _sol_p
                        _sol_price_est = _sol_p
                    except (ImportError, AttributeError):
                        pass
                    if _abort_sol_recv > 0 and _abort_sol_spent_usd > 0:
                        _abort_sol_spent = _abort_sol_spent_usd / _sol_price_est
                        _abort_pnl_usd = (_abort_sol_recv - _abort_sol_spent) * _sol_price_est
                        _abort_exit_price = (abort_sell or {}).get("fill_price") or fill_price
                    else:
                        _abort_pnl_usd = 0.0
                        _abort_exit_price = fill_price
                    live_pos.entry_price = fill_price
                    live_pos.current_price = fill_price
                    live_pos.peak_price = fill_price
                    live_pos.fill_price_recorded = fill_price  # P1'
                    live_pos.exit_price = _abort_exit_price
                    live_pos.realized_pnl_usd = _abort_pnl_usd
                    live_pos.sol_received = _abort_sol_recv
                    live_pos.exit_time = time.time()
                    live_pos.exit_reason = "abort_tripwire"
                    live_pos.status = "closed"
                    live_pos.notes = (
                        f"live|tx:{buy_tx_sig}|fill:{fill_price:.10f}"
                        f"|abort_slip:{_abort_slip:.1f}%vs{_abort_ref_label}"
                        + (f"|sell_tx:{sell_tx_sig}" if sell_tx_sig else "")
                        + (f"|sol_received:{_abort_sol_recv:.8f}" if _abort_sol_recv > 0 else "")
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
                # P1': record fill on live position for journal benchmark fields
                live_pos.fill_price_recorded = fill_price
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
                live_pos.is_live = True   # 4E: set exactly once here; all journal routing keys off this
                # Z2/Z7: set structured state fields at live buy confirm
                live_pos.policy_cohort  = "strategy_pure_rider"
                live_pos.lifecycle_state = (
                    "graduated" if result.get("pp_silent") and not result.get("oracle_bonding_curve")
                    else "bonding_curve"
                )
                paper_pos.notes = (paper_pos.notes or "") + f"|has_live_twin:{live_pos.id}"  # fix: suppress duplicate paper close alert
                # ── Paper twin: record fill but do NOT rebase entry ──────────────
                # P1': paper entry stays at preflight curve baseline (set before buy).
                # This gives an honest benchmark — replay shows ~+94% not +408%.
                # fill_price_recorded is stored for audit comparison only.
                # Live position's entry_price IS anchored to fill (stop logic needs it).
                paper_pos.fill_price_recorded = fill_price
                # Do NOT set paper_pos.entry_price = fill_price (P1' — wrong design).
                # Keep paper entry at baseline_curve_price set during preflight.
                # Only update tracking prices so stops fire correctly:
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
                    "screen=%.1fs  quote=%.2fs  submit=%.2fs  confirm=%.2fs  total=%.1fs | "
                    "build_ms=%.1f  sign_ms=%.1f  send_ms=%.1f  land_ms=%.1f  429_ms=%.1f"
                    "  http_build_ms=%.1f  confirm_detect_ms=%.1f  quote_ms=%.1f",
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
                    _timing.get("build_ms", 0),
                    _timing.get("sign_ms", 0),
                    _timing.get("send_ms", 0),
                    _timing.get("land_ms", 0),
                    _timing.get("rpc_429_wait_ms", 0),
                    _timing.get("http_build_ms", 0),
                    _timing.get("confirm_detect_ms", 0),
                    _timing.get("quote_ms", 0),
                )
                # ── Telemetry: buy confirmed + fill recorded ──
                # P4': buy_confirmed event includes sol_spent, tokens_received, fill_price, slip_pct
                try:
                    if _entry_trace_id:
                        _tel.event(_entry_trace_id, "buy_confirmed",
                            buy_confirmed_ts=_t_now,
                            sol_spent=result.get("sol_spent", 0),
                            tokens_received=result.get("tokens_received_raw", 0),
                            fill_price=fill_price,
                            slip_pct=round(_total_slip, 2) if _total_slip is not None else None,
                            tx_sig=result.get("tx_sig", ""),
                            entry_slippage_pct=result.get("entry_slippage_pct"),
                            jupiter_quote_price=_jup_price,
                            total_slip_pct=round(_total_slip, 2) if _total_slip is not None else None,
                        )
                        _tel.event(_entry_trace_id, "buy_fill_recorded",
                            fill_recorded_ts=time.time(),
                            pos_id=live_pos.id,
                            live_size_usd=_live_size,
                            signal_price=_sig_price,
                            pp_gate_price=_pp_at_gate if '_pp_at_gate' in dir() else 0,
                            fill_price=fill_price,
                            alert_to_fill_ms=round((_t_now - _t_receive) * 1000, 1) if _t_receive else None,
                        )
                except Exception:
                    pass

                try:
                    # X5: store fill confirm timestamp for first_price_ms measurement
                    live_pos._fill_confirm_ts = time.time()
                except Exception:
                    pass

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
                       price: float = 0.0, _t_detect: float = 0.0) -> Optional[Position]:
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
            # Z6: record exit_intent once (never overwritten by routing outcomes)
            if not pos.exit_intent_reason:
                pos.exit_intent_reason = reason
                pos.exit_intent_ts     = pos.exit_time
                pos.exit_intent_policy = getattr(pos, "policy_cohort", "")
            self._jup_fallback_since.pop(pos_id, None)  # clean up dex_pair_loss tracker

        # Live execution gate — only sell on-chain if this position was a live buy
        _t_close_enter = time.time()   # X3: for detect_ms / dispatch_ms telemetry
        _was_live_buy = pos.is_live
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

        # ── Telemetry: exit triggered ──
        _exit_trace_id = ""
        try:
            from memecoin import telemetry as _tel
            _exit_trace_id = _tel.get_trace_id_for_pos(pos_id)
            if not _exit_trace_id and _was_live_buy:
                # Position opened before telemetry — start a new trace
                _exit_trace_id = _tel.start_trace(
                    pos_id=pos_id,
                    mint=pos.token_address,
                    symbol=pos.token_symbol,
                    live_or_paper="live",
                )
            if _exit_trace_id:
                _tel.event(_exit_trace_id, "exit_triggered",
                    reason=reason,
                    trigger_price=price or pos.current_price,
                    trigger_source="close_position",
                    fraction=pos.remaining_fraction,
                    skip_chain_sell=_skip_chain_sell,
                    detect_ms=round((_t_close_enter - _t_detect) * 1000, 1) if _t_detect > 0 else None,
                    dispatch_ms=round((_t_close_enter - (_t_detect or _t_close_enter)) * 1000, 1) if _t_detect > 0 else None,
                )
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
                    "feed_blind", "pre_graduation_exit",   # X1: urgent exits eligible for presigned
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
                        from memecoin.executor import (
                            _send_transaction, _confirm_tx,
                            _mint_token_program_cache, _TOKEN22_PROGRAM_ID,
                            get_pumpfun_curve_complete,
                        )
                        # X1: skip T22 (L4 path not yet proven)
                        _tok_prog     = _mint_token_program_cache.get(pos.token_address, "")
                        _ps_is_t22    = (_tok_prog == _TOKEN22_PROGRAM_ID)
                        # X1: oracle gate — complete==False required
                        _ps_oracle_ok = False
                        if _ps_is_t22:
                            log.info("Presigned skip T22 %s — ladder", pos.token_symbol)
                        else:
                            try:
                                _ps_cv = get_pumpfun_curve_complete(pos.token_address)
                                _ps_oracle_ok = (_ps_cv.get("complete") is False)
                                if not _ps_oracle_ok:
                                    log.info(
                                        "Presigned skip graduated/missing %s reason=%s — ladder",
                                        pos.token_symbol, _ps_cv.get("reason", "?"),
                                    )
                            except Exception as _pog_e:
                                log.debug("Presigned oracle gate err %s: %s", pos.token_symbol, _pog_e)
                                _ps_oracle_ok = True  # err → attempt presigned
                        if not _ps_oracle_ok or _ps_is_t22:
                            with self._presigned_lock:
                                self._presigned_exits[pos.token_address] = _ps_bytes
                        else:
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
                                    "PRESIGNED FALLBACK token=%s presign_fallback reason=%s — ladder",
                                    pos.token_symbol, _pe,
                                )

                if not _presigned_used:
                    from memecoin.exit_orchestrator import (
                        ExitOrchestrator as _ExitOrch,
                        is_rescue_eligible as _is_rescue_elig,
                    )
                    orch = _ExitOrch(pos_id)
                    # ── ExitRouter: classify token state + run PumpSwap local path ──────
                    # Additive layer — does NOT replace the executor path below.
                    # PUMPSWAP_LOCAL_SIM_ONLY=True (default): simulate only, then fall through
                    # to executor.  PUMPSWAP_LOCAL_SELL_ENABLED=True: simulate then send if ok.
                    _pumpswap_local_succeeded = False
                    # Tracks whether ExitRouter identified this token as graduated/uncertain so
                    # the executor escalation below doesn't miss it (pos.dex_id may still be
                    # "pumpfun" for Cat-2 tokens that graduated during the hold period).
                    _er_classified_graduated  = False
                    # Initialized here so is_rescue_eligible() can safely read them
                    # even if ExitRouter raises or EXIT_ROUTER_ENABLED=False.
                    _exit_state = None
                    _ps_result  = None
                    try:
                        from memecoin.config import EXIT_ROUTER_ENABLED as _er_enabled
                    except ImportError:
                        _er_enabled = False
                    # Z4: Fresh venue classification using lifecycle_state field (authoritative).
                    # Falls back to notes-based cohort tag for positions that predate Z2.
                    _lifecycle = getattr(pos, "lifecycle_state", "")
                    # True when bonding curve oracle confirmed complete=False at buy time.
                    # T22 tokens are always PP-silent but not graduated — they must use
                    # PumpPortal (escalate=False), not PumpSwap/Jupiter rescue.
                    _oracle_bc = (
                        _lifecycle == "bonding_curve"             # Z4: authoritative field
                        or "|cohort:bonding_curve" in (pos.notes or "")  # legacy fallback
                    )

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
                    # B3: For oracle-confirmed graduated positions, executor runs FIRST.
                    # Jupiter rescue only fires after executor pump-amm fails.
                    # Detection: graduation_first_seen_ts stamp (set by B2 curve feed oracle).
                    _oracle_confirmed_graduated = (
                        "|graduation_first_seen_ts:" in (pos.notes or "")
                        and "|cohort:graduated" in (pos.notes or "")
                    )
                    if (not _pumpswap_local_succeeded
                            and not _oracle_confirmed_graduated
                            and _is_rescue_elig(
                        error_class=_ps_ec,
                        exit_state=_exit_state.value if _exit_state is not None else "",
                        reason=reason,
                        oracle_bonding_curve=_oracle_bc,
                    )):
                        try:
                            try:
                                from app.alerts import _send as _alert_send
                                _alert_send(
                                    f"\u26a1 RESCUE {pos.token_symbol} — "
                                    f"attempting Jupiter rescue (reason={reason})"
                                )
                            except Exception:
                                pass
                            _resc, _rescue_class = orch.dispatch_rescue(pos, reason)
                            _rescue_attempted = True

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

                            elif _rescue_class == "fatal_no_send":
                                # Structural failure (keypair/sign) — executor cannot help either.
                                log.warning(
                                    "Jupiter rescue: fatal_no_send for %s — arming migration retry, "
                                    "not falling through (structural failure)",
                                    pos.token_symbol,
                                )
                                try:
                                    from memecoin.config import SELL_STUCK_RETRY_SEC as _srs
                                except ImportError:
                                    _srs = 60
                                self._arm_migration_retry(pos.id, _srs)
                                return pos
                            else:
                                # no_route / retry_no_send: Jupiter can't route yet (pool not indexed).
                                # R3: Jupiter no_route CANNOT globally block pump-amm or BC routes.
                                # No tx was sent — fall through to executor.sell with escalate=True.
                                # executor will try pump-amm directly (no Jupiter indexing required).
                                log.info(
                                    "Jupiter rescue: %s for %s — no tx sent, "
                                    "falling through to executor (R3 venue isolation, pump-amm attempt)",
                                    _rescue_class, pos.token_symbol,
                                )
                                # Do NOT return — executor.sell runs below

                        except Exception as _resc_exc:
                            log.warning("Jupiter rescue exception (non-fatal): %s", _resc_exc)

                    # R3 (venue isolation): Only block executor when a real tx is pending (duplicate risk).
                    # no_route / retry_no_send are no-tx failures — executor may still try pump-amm.
                    # fatal_no_send and pending both return early above, so _rescue_class here is
                    # either "fallback_allowed" (rescue not attempted / exception path) or
                    # "no_route" / "retry_no_send" (no tx sent, fall-through allowed).
                    _rescue_blocks_executor = False  # R3: Jupiter result never globally blocks another venue
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
                        # G-batch Part 15: stamp first graduation detection for fast-window retry cadence.
                        if reason == "graduated_exit" and "|graduation_first_seen_ts:" not in (pos.notes or ""):
                            pos.notes = (pos.notes or "") + f"|graduation_first_seen_ts:{int(time.time())}"
                            self._positions[pos_id] = pos
                        _is_graduated = (
                            _lifecycle == "graduated"                # Z4: authoritative field
                            or "|cohort:graduated" in (pos.notes or "")  # legacy fallback
                            or reason == "graduated_exit"
                            or _er_classified_graduated
                        )
                    # B5: T22 graduated pump-amm gate.
                    # Check token program from classifier (not suffix heuristic).
                    _is_t22_graduated = False
                    _t22_pump_amm_allowed = False
                    if _is_graduated and pos.is_live:
                        try:
                            from memecoin.mint_classifier import get_token_program as _gtp
                            _tok_prog_b5 = _gtp(pos.token_address)
                            _is_t22_graduated = (_tok_prog_b5 == "T22")
                        except Exception:
                            _is_t22_graduated = False
                        if _is_t22_graduated:
                            try:
                                from memecoin.config import (
                                    T22_GRAD_PUMP_AMM_PROBE_ENABLED as _t22_probe,
                                    T22_GRAD_PUMP_AMM_ENABLED as _t22_enabled,
                                )
                            except ImportError:
                                _t22_probe = False; _t22_enabled = False
                            _t22_pump_amm_allowed = _t22_enabled or _t22_probe
                            log.info(
                                "B5 T22 grad gate  token=%s  probe=%s  enabled=%s  allowed=%s",
                                pos.token_symbol, _t22_probe, _t22_enabled, _t22_pump_amm_allowed,
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
                            escalate=(False if _oracle_bc
                                      else (False if (_is_t22_graduated and not _t22_pump_amm_allowed)
                                            else (_is_retry or _is_graduated))),
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
                        # P3 / Phase 4.3: sol_received/realized_pnl_usd populated from sell result before _save_positions()
                        _sol_recv = result.get("sol_received") or 0.0
                        if _sol_recv:
                            pos.sol_received = _sol_recv
                        # Accumulate realized_pnl_usd for the remaining fraction being closed
                        if pos.entry_price > 0 and fill:
                            _exit_pnl = (fill / pos.entry_price - 1.0) * pos.size_usd * pos.remaining_fraction
                            pos.realized_pnl_usd += _exit_pnl
                        _step = result.get("ladder_step", 1)
                        _all  = result.get("all_sigs", [])
                        _sigs_tag = f"|all_sigs:{','.join(_all)}" if len(_all) > 1 else ""
                        pos.notes = (
                            (pos.notes or "")
                            + f"|sell_tx:{result.get('tx_sig','')}|sell_fill:{fill:.10f}"
                            + (f"|sol_received:{_sol_recv:.8f}" if _sol_recv else "")
                            + (f"|sell_step:{_step}" if _step > 1 else "")
                            + _sigs_tag
                        )
                        log.info("Live sell confirmed %s  tx=%s  fill=%.10f",
                                 pos.token_symbol, result.get("tx_sig","")[:16], fill)
                        # ── Telemetry: sell confirmed ──
                        try:
                            if _exit_trace_id:
                                _tel.event(_exit_trace_id, "sell_confirmed",
                                    sell_confirmed_ts=time.time(),
                                    tx_sig=result.get("tx_sig", ""),
                                    fill_price=fill,
                                    sol_received=result.get("sol_received", 0),
                                    ladder_step=result.get("ladder_step", 1),
                                    route_used=result.get("route", "executor"),
                                    build_ms=result.get("timing", {}).get("build_ms"),
                                    send_ms=result.get("timing", {}).get("send_ms"),
                                    land_ms=result.get("timing", {}).get("land_ms"),
                                    meta_ms=result.get("timing", {}).get("meta_ms"),
                                )
                        except Exception:
                            pass
                        # B5 probe: append result row to logs/t22_grad_probe.jsonl
                        if _is_t22_graduated and _t22_pump_amm_allowed:
                            try:
                                import json as _pj
                                from pathlib import Path as _PP
                                _probe_path = _PP("logs/t22_grad_probe.jsonl")
                                _probe_row = {
                                    "ts": time.time(),
                                    "mint": pos.token_address,
                                    "symbol": pos.token_symbol,
                                    "route": result.get("route", "executor"),
                                    "success": result.get("success", False),
                                    "tx_sig": result.get("tx_sig", ""),
                                    "error_class": result.get("error_class", ""),
                                    "meta_err": result.get("meta_err"),
                                    "sol_received": result.get("sol_received", 0),
                                    "probe_mode": True,
                                }
                                with open(_probe_path, "a") as _pf:
                                    _pf.write(json.dumps(_probe_row) + "\n")
                            except Exception:
                                pass
                        try:
                            from app.alerts import alert_live_sell
                            alert_live_sell(pos, result.get("sol_received", 0), result.get("tx_sig", ""))
                        except Exception:
                            pass
                    elif not _pumpswap_local_succeeded and result.get("reason") == "zero_balance":
                        # Z5: If pending_signature is set, the zero balance may be from
                        # a previous TX we haven't confirmed yet — defer rather than close.
                        _z5_pending = getattr(pos, "pending_signature", "")
                        if _z5_pending:
                            log.warning(
                                "Z5 zero_balance deferred — pending_signature=%s may account "
                                "for balance  pos=%s",
                                _z5_pending[:16], pos_id,
                            )
                            pos.notes = (pos.notes or "") + f"|z5_zero_balance_deferred:{_z5_pending[:8]}"
                            pos.status      = "open"
                            pos.exit_price  = 0.0
                            pos.exit_time   = 0.0
                            pos.exit_reason = ""
                            self._positions[pos_id] = pos
                            self._sell_stuck_until[pos_id] = time.time() + 30
                            _save_positions(self._positions)
                            return pos
                        log.warning("Live sell %s — zero balance, tokens already sold. Closing.",
                                    pos.token_symbol)
                        pos.notes = (pos.notes or "") + "|sell_already_gone"
                        # Close immediately — do not fall through to retry handler.
                        # Retrying a zero-balance sell produces graduated_unsellable which
                        # triggers a 3-cycle loss write-off at $0 with misleading alerts.
                        pos.status      = "closed"
                        pos.exit_price  = pos.exit_price or 0.0
                        pos.exit_time   = time.time()
                        # Z6: strategy_pure_rider keeps original exit_reason; routing outcome in notes
                        if getattr(pos, "policy_cohort", "") == "strategy_pure_rider":
                            pos.notes = (pos.notes or "") + "|routing:zero_balance"
                        else:
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
                    # B3: Post-executor Jupiter fallback for oracle-confirmed graduated.
                    # Only fires if executor did NOT succeed AND oracle_confirmed_graduated.
                    if (not _pumpswap_local_succeeded
                            and _oracle_confirmed_graduated
                            and not result.get("success")
                            and result.get("reason") not in ("zero_balance",)
                            and _is_rescue_elig(
                                error_class=result.get("error_class", ""),
                                exit_state=_exit_state.value if _exit_state is not None else "",
                                reason=reason,
                                oracle_bonding_curve=_oracle_bc,
                            )):
                        log.info(
                            "B3 post-executor Jupiter fallback for %s "
                            "(executor pump-amm failed, trying Jupiter now)",
                            pos.token_symbol,
                        )
                        try:
                            _b3_resc, _b3_cls = orch.dispatch_rescue(pos, reason)
                            if _b3_cls == "sold":
                                _rescue_succeeded = True
                                _b3_fill = _b3_resc.get("fill_price") or pos.exit_price
                                pos.exit_price = _b3_fill
                                pos.notes = (pos.notes or "") + (
                                    f"|sell_tx:{_b3_resc.get('tx_sig','')}|sell_fill:{_b3_fill:.10f}"
                                    f"|route:JUPITER_RESCUE_B3"
                                )
                                result = {"success": True, "fill_price": _b3_fill,
                                          "sol_received": _b3_resc.get("sol_received", 0),
                                          "tx_sig": _b3_resc.get("tx_sig", ""),
                                          "route": "jupiter_rescue_b3"}
                                log.info("B3 Jupiter fallback succeeded %s sig=%s",
                                         pos.token_symbol, (_b3_resc.get("tx_sig",""))[:16])
                            elif _b3_cls == "pending":
                                try:
                                    from memecoin.config import SELL_STUCK_RETRY_SEC as _srs
                                except ImportError:
                                    _srs = 60
                                pos.status = "sell_stuck"
                                self._positions[pos_id] = pos
                                self._sell_stuck_until[pos_id] = time.time() + _srs
                                _save_positions(self._positions)
                                return pos
                            elif _b3_cls in ("no_route", "retry_no_send"):
                                self._arm_migration_retry(pos.id, 60)
                                return pos
                        except Exception as _b3_exc:
                            log.warning("B3 Jupiter fallback exception: %s", _b3_exc)
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
                                        # Z6: strategy_pure_rider preserves original exit_reason;
                                        # routing outcome goes to notes instead.
                                        if getattr(pos, "policy_cohort", "") == "strategy_pure_rider":
                                            pos.notes = (
                                                (pos.notes or "")
                                                + f"|routing:graduated_recovered:{_gl_sig[:8]}"
                                                + f"|sol_received:{_gl_sol_delta:.8f}"
                                            )
                                        else:
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

                            # Z5: Never write off if pending_signature not yet swept.
                            _z5_pending_gl = getattr(pos, "pending_signature", "")
                            if _z5_pending_gl and _z5_pending_gl not in (_gl_sigs if _gl_wallet else []):
                                log.warning(
                                    "Z5 graduated_loss deferred — pending_signature=%s not yet swept  pos=%s",
                                    _z5_pending_gl[:16], pos_id,
                                )
                                self._arm_migration_retry(pos.id, 60)
                                return pos

                            # Migration never settled or pool is permanently empty.
                            # Write off as total loss — no more Helius/RPC burn.
                            self._graduated_retry_count.pop(pos_id, None)
                            self._sell_stuck_until.pop(pos_id, None)
                            pos.status      = "closed"
                            pos.exit_price  = 0.0
                            pos.exit_time   = time.time()
                            # Z6: strategy_pure_rider preserves original exit_reason
                            if getattr(pos, "policy_cohort", "") == "strategy_pure_rider":
                                pos.notes = (pos.notes or "") + f"|routing:graduated_loss_after_{_grad_attempts}_retries"
                            else:
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
                        # Z5/Z2: track pending_signature for restart-safe write-off guard
                        if result.get("tx_sig"):
                            pos.pending_signature       = result["tx_sig"]
                            pos.pending_signature_route = result.get("route", "executor")
                            pos.pending_signature_ts    = time.time()
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
                            pos.mu_sell_total = getattr(pos, "mu_sell_total", 0) + 1
                            pos.sell_attempts = 0   # reset counter for next 60s window
                            if pos.mu_sell_total >= 8:
                                # 4D: terminal gate — require manual intervention
                                pos.status = "manual_required"
                                self._positions[pos_id] = pos
                                self._sell_stuck_until.pop(pos_id, None)  # never re-arm
                                _save_positions(self._positions)
                                log.error(
                                    "SELL TERMINAL %s — attempt 8 exhausted, manual sell required. "
                                    "mint=%s  /manual_sold %s",
                                    pos.token_symbol, pos.token_address, pos.token_symbol,
                                )
                                try:
                                    from app.alerts import _send
                                    _send(
                                        f"🚨 SELL TERMINAL {pos.token_symbol} — 8 sell windows failed.\n"
                                        f"Tokens still on-chain. Sell manually in Phantom then:\n"
                                        f"/manual_sold {pos.token_symbol}\n"
                                        f"mint={pos.token_address}"
                                    )
                                except Exception:
                                    pass
                            else:
                                pos.status = "sell_stuck"
                                pos.exit_price  = 0.0
                                pos.exit_time   = 0.0
                                pos.exit_reason = ""
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
        Close a position whose sell was already executed by the Jupiter rescue path
        (dispatch_rescue / exit_orchestrator) from outside the normal close_position()
        path (e.g. scanner.py migration branch).

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
            fill_price = rescue_result.get("fill_price") or 0.0
            if fill_price == 0.0 and sol_recv > 0.0:
                try:
                    from memecoin.tx_meta import compute_fill_price
                    from memecoin.executor import _sol_price_usd
                    _sol_usd = _sol_price_usd()
                    _tokens_raw = int(pos.tokens_held * pos.remaining_fraction)
                    if _tokens_raw > 0 and _sol_usd > 0:
                        fill_price = compute_fill_price(sol_recv, _tokens_raw, _sol_usd)
                except Exception as _fp_err:
                    log.debug("compute_fill_price failed in rescue finalize: %s", _fp_err)

            pos.status      = "closed"
            pos.exit_reason = "jupiter_rescue"
            pos.exit_time   = time.time()
            pos.notes = (pos.notes or "") + (
                f"|sell_tx:{sig}|route:JUPITER_RESCUE"
                f"|sol_received:{sol_recv:.6f}"
                + (f"|sell_fill:{fill_price:.10f}" if fill_price else "")
            )

            if fill_price > 0 and pos.entry_price > 0:
                pos.exit_price = fill_price
                # pnl_pct and pnl_usd are @property — computed from exit_price automatically.
                # DO NOT assign them directly (raises AttributeError → blocks journal write).
            else:
                # fill still unknown — don't alert $0/-100%
                pos.exit_price = 0.0
                pos.notes = (pos.notes or "") + "|fill_estimated|sol_parse_failed"
                # send estimation alert instead of bogus -100%
                try:
                    from app import alerts as _al
                    _al._send(
                        f"[LIVE SELL] {pos.token_symbol} (SOL)\n"
                        f"Reason:   jupiter_rescue\n"
                        f"Exit confirmed — fill reconciling (est. pending)\n"
                        f"SOL rcvd: {sol_recv:.4f}\n"
                        f"Tx:       {sig[:20]}..."
                    )
                except Exception:
                    pass
                # skip the normal alert_live_sell call below
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
                try:
                    from app import alerts
                    alerts.alert_position_close(pos)
                except Exception:
                    pass
                log.info("_finalize_rescue_sell: closed %s  sig=%s  sol=%.6f  fill=PENDING",
                         pos.token_symbol, sig[:16] if sig else "", sol_recv)
                return

            self._positions[pos_id] = pos
            try:
                _append_journal(pos)
            except Exception as _jex:
                log.error("_finalize_rescue_sell: journal write failed for %s: %s", pos.token_symbol, _jex)
                try:
                    from app.alerts import _send
                    _send(f"🚨 JOURNAL WRITE FAILED {pos.token_symbol} (jupiter_rescue): {_jex}")
                except Exception:
                    pass
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

    def _get_venue_state(self, pos_id: str, venue: str) -> dict:
        """Return mutable venue state dict for pos_id/venue. Creates if absent."""
        if pos_id not in self._venue_state:
            self._venue_state[pos_id] = {}
        if venue not in self._venue_state[pos_id]:
            self._venue_state[pos_id][venue] = {
                "cooldown_until": 0.0,
                "attempts": 0,
                "last_result": "",
            }
        return self._venue_state[pos_id][venue]

    def _record_venue_attempt(self, pos_id: str, venue: str, result: str,
                               cooldown_sec: float = 0.0) -> None:
        """Record a venue attempt and optionally set a cooldown."""
        vs = self._get_venue_state(pos_id, venue)
        vs["attempts"] += 1
        vs["last_result"] = result
        if cooldown_sec > 0:
            vs["cooldown_until"] = time.time() + cooldown_sec

    def _venue_in_cooldown(self, pos_id: str, venue: str) -> bool:
        """True if venue is still in cooldown (do not retry yet)."""
        vs = self._get_venue_state(pos_id, venue)
        return time.time() < vs["cooldown_until"]

    def _pump_amm_attempts(self, pos_id: str) -> int:
        """Return pump-amm attempt count for pos in fast window."""
        return self._get_venue_state(pos_id, "pump_amm")["attempts"]

    def _arm_migration_retry(self, pos_id: str, retry_sec: float) -> None:
        """
        Arm a MIGRATION_UNCERTAIN + no-pool position for sell_stuck retry.
        Sets sell_stuck status, records |migration_wait| + first-detection timestamp,
        and sets the retry timer.

        Fast window (Part 15): if graduation_first_seen_ts is in notes and within
        GRAD_FAST_WINDOW_SEC, uses GRAD_FAST_RETRY_SEC instead of retry_sec.
        This reduces graduation-window loss by polling pump-amm every 5s instead of 60s.
        After 60s, the existing MU ladder takes over at normal cadence.
        """
        import re as _re_mu_arm
        pos = self._positions.get(pos_id)
        if not pos:
            return

        # Fast window: check graduation_first_seen_ts
        try:
            from memecoin.config import GRAD_FAST_WINDOW_SEC, GRAD_FAST_RETRY_SEC
        except ImportError:
            GRAD_FAST_WINDOW_SEC = 60
            GRAD_FAST_RETRY_SEC = 5

        _grad_ts_m = _re_mu_arm.search(r'\|graduation_first_seen_ts:(\d+)', pos.notes or "")
        if _grad_ts_m:
            _grad_age = time.time() - int(_grad_ts_m.group(1))
            if _grad_age < GRAD_FAST_WINDOW_SEC:
                _actual_retry = GRAD_FAST_RETRY_SEC
                # B4: enforce fast-window pump-amm attempt cap.
                # After 3 pump-amm attempts, use 60s cadence (Jupiter will be tried at MU attempt 4+).
                _pa_attempts = self._pump_amm_attempts(pos_id)
                if _pa_attempts >= 3:
                    _actual_retry = retry_sec  # switch to normal MU cadence
                    log.info(
                        "GRAD FAST WINDOW %s: pump-amm attempts=%d >= 3, "
                        "switching to MU cadence (%ds)",
                        pos.token_symbol, _pa_attempts, _actual_retry,
                    )
                else:
                    log.info(
                        "GRAD FAST WINDOW %s: age=%.0fs < %ds — retry in %ds",
                        pos.token_symbol, _grad_age, GRAD_FAST_WINDOW_SEC, _actual_retry,
                    )
            else:
                _actual_retry = retry_sec
                log.info(
                    "GRAD FAST WINDOW %s: age=%.0fs >= %ds — normal retry %ds (MU ladder)",
                    pos.token_symbol, _grad_age, GRAD_FAST_WINDOW_SEC, _actual_retry,
                )
        else:
            _actual_retry = retry_sec

        if "|migration_wait" not in (pos.notes or ""):
            pos.notes = (pos.notes or "") + f"|migration_wait|migration_uncertain_ts:{int(time.time())}"
        elif "|migration_uncertain_ts:" not in (pos.notes or ""):
            pos.notes = (pos.notes or "") + f"|migration_uncertain_ts:{int(time.time())}"
        pos.status = "sell_stuck"
        self._positions[pos_id] = pos
        self._sell_stuck_until[pos_id] = time.time() + _actual_retry
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

            # X5: first_price_ms — time from fill confirm to first monitored price
            if (getattr(pos, '_fill_confirm_ts', 0) > 0
                    and not getattr(pos, '_first_price_logged', False)
                    and pos.current_price > 0):
                _fpm = (time.time() - pos._fill_confirm_ts) * 1000
                pos._first_price_logged = True
                log.info("FIRST_PRICE_MS token=%s ms=%.0f", pos.token_symbol, _fpm)
                try:
                    from memecoin import telemetry as _tel
                    _fpm_tid = _tel.get_trace_id_for_pos(pos.id)
                    if _fpm_tid:
                        _tel.event(_fpm_tid, "first_price_tick",
                            first_price_ms=round(_fpm, 1),
                            first_price=pos.current_price,
                        )
                except Exception:
                    pass

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
                                "is_live":              pos.is_live,
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

            # X3: capture trigger detection time before exit condition evaluation
            _t_trig = time.time()

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

            # 1. Hard stop — effective level = max(signal-anchored stop, fill-loss floor).
            #    Prevents TROONCH-style -70% losses when slippage is high, while preserving
            #    signal structure when fill is close to signal.
            #    MAX_LOSS_FROM_FILL_PCT=50% caps max fill-anchored loss.
            #    Paper behavior unchanged: paper entry_price == signal_price.
            #
            #    Example A (high slippage):
            #      signal=1.00  fill=2.17  hard_stop=-35%  MAX_LOSS=50%
            #      signal_stop = 1.00*0.65 = 0.65
            #      fill_floor  = 2.17*0.50 = 1.085
            #      effective   = max(0.65, 1.085) = 1.085  → -50% from fill
            #
            #    Example B (low slippage):
            #      signal=1.00  fill=1.10
            #      signal_stop = 0.65   fill_floor = 0.55
            #      effective   = max(0.65, 0.55) = 0.65  → signal anchor wins
            if not reason:
                _stop_lvl = effective_hard_stop_level(
                    pos.signal_price, pos.entry_price, pos.hard_stop_pct
                )
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
                # ── Telemetry: exit condition true (edge-trigger: once per reason) ──
                try:
                    from memecoin import telemetry as _tel
                    _mon_tid = _tel.get_trace_id_for_pos(pos.id)
                    if _mon_tid:
                        _evt_name = "tp_condition_true" if reason.startswith("whale_exit") else "exit_condition_true"
                        _tel.emit_once(
                            _mon_tid,
                            f"{_evt_name}:{reason}",   # edge-trigger key
                            _evt_name,
                            reason=reason,
                            trigger_price=pos.current_price,
                            gain_pct=round(gain * 100, 2),
                            peak_gain_pct=round(_peak_gain * 100, 2),
                        )
                except Exception:
                    pass
                closed = self.close_position(pos.id, reason, _t_detect=_t_trig)
                if closed:
                    exits.append({
                        "pos_id": pos.id,
                        "token_symbol": pos.token_symbol,
                        "reason": reason,
                        "pnl_pct": round(pos.pnl_pct * 100, 2),
                    })
            else:
                # check take-profit ladder
                _was_live_buy = pos.is_live
                import re as _re
                for tp_pct, tp_fraction in TP_LEVELS:
                    level_key = f"tp_{int(tp_pct*100)}"
                    if gain >= tp_pct and level_key not in pos.tp_levels_hit:
                        # Cooldown check: skip if a recent TP sell failed for this level
                        _now = time.time()
                        _cd_match = _re.search(
                            rf'\|tp_retry_cooldown:{_re.escape(level_key)}:(\d+)',
                            pos.notes or "",
                        )
                        if _cd_match and _now < float(_cd_match.group(1)):
                            continue  # still cooling down, skip this level
                        # cooldown expired — remove old tag, allow re-arm
                        if _cd_match:
                            pos.notes = _re.sub(
                                rf'\|tp_retry_cooldown:{_re.escape(level_key)}:\d+',
                                "", pos.notes or "",
                            )

                        # ── Telemetry: tp_condition_true (edge-trigger: once per level) ──
                        try:
                            from memecoin import telemetry as _tel
                            _tp_tid = _tel.get_trace_id_for_pos(pos.id)
                            if _tp_tid:
                                _tel.emit_once(
                                    _tp_tid,
                                    f"tp_condition_true:{level_key}",   # edge-trigger key
                                    "tp_condition_true",
                                    level_key=level_key,
                                    tp_pct=round(tp_pct * 100, 1),
                                    gain_pct=round(gain * 100, 2),
                                    trigger_price=pos.current_price,
                                )
                        except Exception:
                            pass

                        sell_frac = tp_fraction * pos.remaining_fraction
                        partial_usd = sell_frac * pos.size_usd

                        if LIVE_TRADING and _was_live_buy:
                            # 4C: TP inflight guard — never >1 concurrent thread per position/level
                            _tp_levels = self._tp_inflight.setdefault(pos.id, {})
                            _tp_ready_at = _tp_levels.get(level_key, 0)
                            if _tp_ready_at > time.time():
                                log.info("TP INFLIGHT guard: skipping duplicate dispatch "
                                         "pos=%s level=%s ready_in=%.1fs",
                                         pos.id, level_key, _tp_ready_at - time.time())
                            else:
                                # Mark in-flight (ready_at=inf blocks all re-dispatch until thread clears it)
                                _tp_levels[level_key] = float("inf")
                                # Dispatch live TP sell — do NOT mutate state yet.
                                # State mutates only on confirmed fill inside _run_tp_sell_bg.
                                _tp_thread = threading.Thread(
                                    target=self._run_tp_sell_bg,
                                    args=(pos.id, sell_frac, tp_pct, level_key),
                                    daemon=True,
                                )
                                _tp_thread.start()
                        else:
                            # Paper path: mutate immediately (no on-chain risk)
                            pos.tp_levels_hit.append(level_key)
                            pos.remaining_fraction -= sell_frac
                            pos.realized_pnl_usd += sell_frac * pos.size_usd * tp_pct

                        log.info(
                            "TP hit %s  %s +%.0f%%  selling %.0f%% ($%.2f)  realized=$%.2f",
                            pos.id, pos.token_symbol, gain * 100,
                            tp_fraction * 100, partial_usd, pos.realized_pnl_usd,
                        )
                        if not (LIVE_TRADING and _was_live_buy):
                            # Paper path: alert immediately
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

        State (tp_levels_hit, remaining_fraction, realized_pnl_usd) is mutated
        ONLY on confirmed fill. Failed sells add a cooldown tag and leave state
        untouched so the TP level can re-arm after cooldown.

        4C: On entry the dispatch guard set _tp_inflight[pos_id][level_key]=inf.
        On exit (success OR failure) we clear/set the cooldown so the next
        dispatch cycle can re-evaluate.
        """
        # ── Telemetry: TP sell triggered ──
        _tp_trace_id = ""
        try:
            from memecoin import telemetry as _tel
            _tp_trace_id = _tel.get_trace_id_for_pos(pos_id) or ""
            if _tp_trace_id:
                _tel.event(_tp_trace_id, "exit_queued",
                    reason=f"tp_{level_key}",
                    tp_pct=tp_pct,
                    sell_frac=sell_frac,
                )
        except Exception:
            pass

        try:
            from memecoin.executor import MemeExecutor as _MEx
        except Exception as _e:
            log.warning("LIVE TP BG: executor import failed: %s", _e)
            return

        def _clear_tp_inflight(cooldown_s: float = 0.0):
            """4C: clear in-flight marker; set cooldown if sell failed."""
            lvls = self._tp_inflight.get(pos_id, {})
            lvls[level_key] = time.time() + cooldown_s if cooldown_s > 0 else 0.0
            self._tp_inflight[pos_id] = lvls

        pos = self._positions.get(pos_id)
        if pos is None or pos.status != "open":
            log.info("LIVE TP BG: position %s already closed, skipping TP sell", pos_id)
            _clear_tp_inflight()
            return

        # Use the exact token count received at buy time (from tx postTokenBalances delta).
        _known_count = int(pos.tokens_held * sell_frac) if pos.tokens_held > 0 else 0
        if _known_count > 0:
            log.info("LIVE TP BG: using known_token_count=%d for %.0f%% sell  %s",
                     _known_count, sell_frac * 100, pos.token_symbol)
        else:
            log.info("LIVE TP BG: tokens_held=0, falling back to RPC balance query  %s",
                     pos.token_symbol)

        # --- Cohort routing ---
        _is_bc = bool(pos.notes and "cohort:bonding_curve" in (pos.notes or ""))
        _tp_is_grad = "|cohort:graduated" in (pos.notes or "")

        try:
            _tp_ex = _MEx()
            if _is_bc:
                # Bonding-curve tokens: force BC path, never PumpSwap/pump-amm
                _tp_r = _tp_ex.sell(
                    pos.token_address, pos.size_usd, pos.entry_price,
                    pos.chain, fraction=sell_frac,
                    escalate=False,
                    known_token_count=_known_count,
                    skip_pumpswap=True,
                )
            else:
                _tp_r = _tp_ex.sell(
                    pos.token_address, pos.size_usd, pos.entry_price,
                    pos.chain, fraction=sell_frac,
                    escalate=_tp_is_grad,
                    known_token_count=_known_count,
                )
        except Exception as _tp_err:
            log.warning("LIVE TP BG exception %s: %s — state unchanged", pos.token_symbol, _tp_err)
            # Add cooldown tag — no state mutation
            pos = self._positions.get(pos_id)
            if pos and pos.status == "open":
                _cooldown_key = f"|tp_retry_cooldown:{level_key}:{int(time.time() + 30)}"
                pos.notes = (pos.notes or "") + _cooldown_key
                _save_positions(self._positions)
            _clear_tp_inflight(cooldown_s=30.0)  # 4C: 30s cooldown before next dispatch
            return

        # Re-fetch position: it may have been closed during the tx confirmation
        pos = self._positions.get(pos_id)
        _clear_tp_inflight()  # 4C: sell returned (success or failure result below)

        if _tp_r.get("success"):
            _tp_fill_raw = _tp_r.get("fill_price")
            _tp_fill     = _tp_fill_raw if _tp_fill_raw else (pos.current_price if pos else 0.0)
            _tp_sig     = _tp_r.get("tx_sig", "")
            _real_pnl   = (
                (_tp_fill / pos.entry_price - 1) * sell_frac * pos.size_usd
                if pos and pos.entry_price > 0 else 0.0
            )

            if pos and pos.status == "open":
                # --- mutate state only here ---
                if level_key not in pos.tp_levels_hit:
                    pos.tp_levels_hit.append(level_key)
                    pos.remaining_fraction -= sell_frac
                    pos.realized_pnl_usd += _real_pnl
                pos.notes = (
                    (pos.notes or "")
                    + f"|{level_key}_tx:{_tp_sig}"
                    + f"|{level_key}_fill:{_tp_fill:.10f}"
                )
                _save_positions(self._positions)
                log.warning(
                    "LIVE TP SELL BG %s  tp=+%.0f%%  frac=%.0f%%  "
                    "fill=%.10f  real=$%.2f  tx=%s",
                    pos.token_symbol, tp_pct * 100, sell_frac * 100,
                    _tp_fill, _real_pnl, _tp_sig[:16],
                )
                # alert TP success
                try:
                    partial_usd = sell_frac * pos.size_usd
                    alerts.alert_tp_hit(pos, tp_pct, partial_usd)
                except Exception:
                    pass
            else:
                log.warning(
                    "LIVE TP SELL BG %s confirmed but position already closed  "
                    "fill=%.10f  realized=$%.2f  tx=%s",
                    pos_id, _tp_fill, _real_pnl, _tp_sig[:16],
                )

            try:
                _sym = pos.token_symbol if pos else pos_id
                from app.alerts import _send
                _send(
                    f"LIVE TP {_sym} +{tp_pct*100:.0f}%\n"
                    f"Sold: {sell_frac*100:.0f}% of position\n"
                    f"Locked: ${_real_pnl:.2f}\n"
                    + (f"Remaining: {pos.remaining_fraction*100:.0f}%\n" if pos and pos.status == "open" else "")
                    + f"tx: {_tp_sig[:20]}"
                )
            except Exception:
                pass
        else:
            # --- failure/revert: do NOT mutate remaining_fraction, realized_pnl, or tp_levels_hit ---
            _cooldown_key = f"|tp_retry_cooldown:{level_key}:{int(time.time() + 30)}"
            if pos and pos.status == "open":
                pos.notes = (pos.notes or "") + _cooldown_key
                _save_positions(self._positions)
            log.warning(
                "TP partial sell FAILED %s level=%s — cooldown 30s. reason=%s",
                pos.token_symbol if pos else pos_id,
                level_key,
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
            if pos.status in ("open", "sell_stuck") and pos.is_live:
                if pos.token_symbol.upper() == sym_upper:
                    target = pos
                    break

        if target is None:
            # Try partial match as fallback
            for pos in self._positions.values():
                if pos.status in ("open", "sell_stuck") and pos.is_live:
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

        # Sort merged result by exit_time ascending.
        # exit_time may be a unix timestamp float OR a legacy datetime string.
        def _exit_sort_key(r):
            v = r.get("exit_time") or ""
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
            try:
                from datetime import datetime, timezone
                return datetime.strptime(v[:19], "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc).timestamp()
            except Exception:
                return 0.0
        rows.sort(key=_exit_sort_key)
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
