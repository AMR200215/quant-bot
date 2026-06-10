"""
Real-time price monitor for open pump.fun positions via PumpPortal WebSocket.

Two modes:
  1. Position monitoring — subscribed for tokens we hold.  Price from virtual
     reserves (~1s latency) overrides DexScreener's 5-30s poll.
     portfolio.py reads prices via get_prices() and uses them when fresh (<15s).

  2. Screening accumulator — subscribed for type-1 TG tokens (no DexScreener
     data yet).  Accumulates per-mint: unique_buyers, buy/sell counts, SOL
     in/out, creator_sold flag.  scanner._run_screening_checks() reads state
     at T+30/60/120s and fires a paper-only signal if conditions are met.

Price formula (both modes):
  price_usd = (vSolInBondingCurve / (vTokensInBondingCurve / 1e6)) * sol_price_usd

Latency: ~1s from on-chain confirm to price update vs 10s DexScreener poll.

Threading model
---------------
  _run_thread       — reconnect loop; calls _connect() which is pure recv
  _sol_price_thread — refreshes SOL/USD via Jupiter every 60s; writes _sol_price
  _heartbeat_thread — sends WS ping every 20s; force-closes if silent >30s
  _pp_exit_thread   — lives in scanner.py, drains price-callback exit queue

The recv loop (_connect inner while) must contain ONLY:
    raw = ws.recv()
    → update _last_frame_ts
    → _handle_message()
No HTTP, no sleeps, no RPC ever.
"""

import base64
import json
import logging
import queue
import struct
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import requests

log = logging.getLogger(__name__)

WS_URL = "wss://pumpportal.fun/api/data"

# Price is considered stale if no update received in this many seconds.
PRICE_STALE_SEC = 15.0

# SOL price refresh interval — runs in its own thread, never in WS recv loop
SOL_PRICE_REFRESH_SEC = 60.0

# Max concurrent screening slots before LRU eviction
MAX_SCREENING_SLOTS = 30

# Reconnect delays: 0.5 → 1 → 2 → 5 (capped)
_RECONNECT_DELAY_BASE = 0.5
_RECONNECT_DELAY_MAX  = 5     # ≤ 5s as required

# Heartbeat: ping every N seconds; force-reconnect if no frame in ping+grace
HEARTBEAT_INTERVAL_SEC = 20
HEARTBEAT_GRACE_SEC    = 10   # close socket if silent for interval + grace = 30s


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScreeningState:
    """
    Per-mint accumulator for PumpPortal screening mode.
    Populated by trade events received while awaiting DexScreener indexing.
    """
    first_seen_ts:    float
    creator_pubkey:   Optional[str] = None
    first_seen_price: float         = 0.0
    latest_price:     float         = 0.0
    unique_buyers:    set           = field(default_factory=set)
    buy_count:        int           = 0
    sell_count:       int           = 0
    sol_in:           float         = 0.0
    sol_out:          float         = 0.0
    creator_sold:     bool          = False
    lru_ts:           float         = field(default_factory=time.time)

    @property
    def net_sol_inflow(self) -> float:
        return self.sol_in - self.sol_out

    @property
    def unique_buyer_count(self) -> int:
        return len(self.unique_buyers)


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class PumpPortalMonitor:
    """
    Maintains a live price cache for subscribed pump.fun tokens.

    Usage:
      monitor = PumpPortalMonitor()
      monitor.start()

      # Position monitoring
      monitor.subscribe({"MINT1", "MINT2"})
      prices = monitor.get_prices()   # {mint: price_usd}
      monitor.unsubscribe({"MINT1"})

      # Screening accumulator
      monitor.subscribe_screening("MINT3", creator_pubkey="PUBKEY")
      state = monitor.get_screening_state("MINT3")  # ScreeningState
      monitor.evict_screening({"MINT3"})
    """

    def __init__(self):
        # ── Position monitoring ───────────────────────────────────────────
        self._subscribed: set[str]   = set()
        self._sub_lock               = threading.Lock()
        # mint → (price_usd, timestamp)
        self._price_cache: dict[str, tuple[float, float]] = {}
        self._cache_lock             = threading.Lock()

        # ── Screening accumulator (LRU-ordered) ──────────────────────────
        self._screening: OrderedDict[str, ScreeningState] = OrderedDict()
        self._screening_lock         = threading.Lock()

        # ── Price update callbacks (event-driven stop detection) ──────────
        # Signature: fn(mint: str, price_usd: float) — must not block.
        self._price_callbacks: list  = []
        self._cb_lock                = threading.Lock()

        # ── SOL price (written by _sol_price_thread, read by _compute_price) ──
        self._sol_price: float       = 170.0
        self._sol_price_lock         = threading.Lock()

        # ── WS connection ─────────────────────────────────────────────────
        self._ws                     = None
        self._ws_lock                = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        # ── Heartbeat / frame-age tracking ───────────────────────────────
        # Updated on every successful ws.recv() — including pong frames.
        self._last_frame_ts: float   = 0.0
        self._frame_lock             = threading.Lock()

        # ── Connection telemetry ─────────────────────────────────────────
        self._conn_start_ts: float   = 0.0   # time of last successful connect
        self._drop_start_ts: float   = 0.0   # time of last drop (for gap_sec calc)
        # Daily counters — reset at UTC day boundary
        self._telemetry_day: str     = ""
        self._daily_drops:   int     = 0
        self._daily_gap_sec: float   = 0.0
        self._daily_blind_exits: int = 0
        self._telem_lock             = threading.Lock()

    # ------------------------------------------------------------------
    # Public API — lifecycle
    # ------------------------------------------------------------------

    def start(self, daemon: bool = True):
        """Start monitor thread + SOL price thread + heartbeat thread."""
        self._thread = threading.Thread(
            target=self._run, daemon=daemon, name="pumpportal-monitor"
        )
        self._thread.start()

        sol_t = threading.Thread(
            target=self._sol_price_thread, daemon=daemon, name="pp-sol-price"
        )
        sol_t.start()

        hb_t = threading.Thread(
            target=self._heartbeat_thread, daemon=daemon, name="pp-heartbeat"
        )
        hb_t.start()

        log.info("PumpPortal monitor started (monitor + sol-price + heartbeat threads)")

    # ------------------------------------------------------------------
    # Public API — position monitoring
    # ------------------------------------------------------------------

    def subscribe(self, mints: set):
        """Subscribe to trade events for held positions."""
        with self._sub_lock:
            new = mints - self._subscribed
            if not new:
                return
            self._subscribed |= new
        self._send_subscribe(new)
        log.info("PumpPortal subscribed to %d token(s): %s",
                 len(new), ", ".join(m[:8] for m in new))

    def unsubscribe(self, mints: set):
        """Unsubscribe held positions and clear their cached prices."""
        with self._sub_lock:
            gone = mints & self._subscribed
            if not gone:
                return
            self._subscribed -= gone
        # Only send WS unsubscribe if not also being screened
        with self._screening_lock:
            still_screening = gone & set(self._screening.keys())
        to_unsub = gone - still_screening
        if to_unsub:
            self._send_unsubscribe(to_unsub)
        with self._cache_lock:
            for m in gone:
                self._price_cache.pop(m, None)
        log.info("PumpPortal unsubscribed from %d token(s)", len(gone))

    def get_prices(self, max_age: float = PRICE_STALE_SEC) -> dict[str, float]:
        """
        Return fresh prices as {mint: price_usd} for position-monitored tokens.
        Only includes entries updated within max_age seconds.
        """
        now = time.time()
        with self._cache_lock:
            return {
                mint: price
                for mint, (price, ts) in self._price_cache.items()
                if now - ts <= max_age
            }

    def get_last_seen(self, mint: str) -> float:
        """
        Seconds since PumpPortal last sent a price tick for this mint.
        Returns float('inf') if the mint has never been seen.
        """
        with self._cache_lock:
            entry = self._price_cache.get(mint)
        if entry is None:
            return float("inf")
        _, ts = entry
        return time.time() - ts

    def add_price_callback(self, fn) -> None:
        """
        Register a callback invoked on every position-monitoring price update.
        Signature: fn(mint: str, price_usd: float) → None
        Called from the WS recv thread — must not block. Use a queue for heavy work.
        """
        with self._cb_lock:
            self._price_callbacks.append(fn)

    def increment_blind_exit_count(self) -> None:
        """Called by scanner when a feed_blind exit fires — tracked in daily telemetry."""
        with self._telem_lock:
            self._daily_blind_exits += 1

    def is_active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def is_connected(self) -> bool:
        """True when we have an open WS connection."""
        with self._ws_lock:
            return self._ws is not None

    def pp_last_frame_age(self) -> float:
        """Seconds since any WS frame was received. inf if never connected."""
        with self._frame_lock:
            ts = self._last_frame_ts
        if ts == 0.0:
            return float("inf")
        return time.time() - ts

    # ------------------------------------------------------------------
    # Public API — screening accumulator
    # ------------------------------------------------------------------

    def subscribe_screening(self, mint: str, creator_pubkey: Optional[str] = None):
        """
        Start accumulating trade data for a mint in screening mode.
        LRU-evicts oldest slot when MAX_SCREENING_SLOTS is reached.
        Safe to call if mint is already subscribed for position monitoring.
        """
        evicted_mint = None

        with self._screening_lock:
            if mint in self._screening:
                if creator_pubkey:
                    self._screening[mint].creator_pubkey = creator_pubkey
                self._screening.move_to_end(mint)
                return

            if len(self._screening) >= MAX_SCREENING_SLOTS:
                evicted_mint, _ = self._screening.popitem(last=False)  # oldest

            self._screening[mint] = ScreeningState(
                first_seen_ts=time.time(),
                creator_pubkey=creator_pubkey,
            )

        if evicted_mint:
            log.debug("PumpPortal screening LRU evict: %s", evicted_mint[:8])
            with self._sub_lock:
                held = evicted_mint in self._subscribed
            if not held:
                self._send_unsubscribe({evicted_mint})

        with self._sub_lock:
            already = mint in self._subscribed
        if not already:
            self._send_subscribe({mint})

        log.info("PumpPortal screening started: %s (creator=%s)",
                 mint[:8], (creator_pubkey or "unknown")[:8] if creator_pubkey else "unknown")

    def get_screening_state(self, mint: str) -> Optional[ScreeningState]:
        with self._screening_lock:
            return self._screening.get(mint)

    def evict_screening(self, mints: set):
        to_unsub = set()
        with self._screening_lock:
            for m in mints:
                self._screening.pop(m, None)
        with self._sub_lock:
            for m in mints:
                if m not in self._subscribed:
                    to_unsub.add(m)
        if to_unsub:
            self._send_unsubscribe(to_unsub)

    # ------------------------------------------------------------------
    # Background threads
    # ------------------------------------------------------------------

    def _sol_price_thread(self):
        """
        Refreshes SOL/USD price every 60s via Jupiter quote.
        Runs independently — never touches the WS recv loop.
        """
        while True:
            time.sleep(SOL_PRICE_REFRESH_SEC)
            self._refresh_sol_price()

    def _heartbeat_thread(self):
        """
        Sends a WS ping every 20s to keep the connection alive.
        If no frame of any kind has been received for >30s (interval + grace),
        force-closes the socket to trigger a reconnect.
        """
        while True:
            time.sleep(HEARTBEAT_INTERVAL_SEC)

            # Send ping via ws_lock so it doesn't race with recv reconnect
            with self._ws_lock:
                ws = self._ws
            if ws is not None:
                try:
                    ws.ping()
                    log.debug("PumpPortal heartbeat ping sent")
                except Exception as e:
                    log.debug("PumpPortal heartbeat ping failed: %s", e)

            # Check frame age — force reconnect if silent beyond grace window
            age = self.pp_last_frame_age()
            deadline = HEARTBEAT_INTERVAL_SEC + HEARTBEAT_GRACE_SEC
            if age > deadline:
                log.warning(
                    "PumpPortal heartbeat timeout: no frame for %.1fs "
                    "(expected ≤ %ds) — forcing reconnect",
                    age, deadline,
                )
                with self._ws_lock:
                    ws = self._ws
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # WebSocket reconnect loop
    # ------------------------------------------------------------------

    def _run(self):
        fail_count = 0
        while True:
            try:
                self._connect()
                fail_count = 0
            except Exception as e:
                fail_count += 1
                delay = min(
                    _RECONNECT_DELAY_BASE * (2 ** min(fail_count - 1, 3)),
                    _RECONNECT_DELAY_MAX,
                )
                log.warning("PumpPortal WS error (attempt %d): %s — retry in %.1fs",
                            fail_count, e, delay)
                # Accumulate gap time in daily telemetry
                with self._telem_lock:
                    self._daily_gap_sec += delay
                time.sleep(delay)

    def _connect(self):
        try:
            import websocket as _ws_mod
        except ImportError:
            log.warning("websocket-client not installed — PumpPortal monitor disabled. "
                        "Run: pip install websocket-client")
            time.sleep(3600)
            return

        ws = _ws_mod.create_connection(WS_URL, timeout=30)

        conn_start = time.time()
        self._conn_start_ts = conn_start

        with self._ws_lock:
            self._ws = ws

        # On reconnect: announce gap closed, update telemetry
        with self._telem_lock:
            if self._drop_start_ts > 0:
                gap = conn_start - self._drop_start_ts
                self._daily_gap_sec += gap
                self._drop_start_ts  = 0.0
        log.info("PumpPortal WebSocket connected (conn_start=%s)",
                 time.strftime("%H:%M:%S", time.gmtime(conn_start)))

        # Re-subscribe to all mints on reconnect
        with self._sub_lock:
            current = set(self._subscribed)
        with self._screening_lock:
            current |= set(self._screening.keys())
        if current:
            self._send_subscribe(current)

        # ── Pure recv loop — NO HTTP, NO sleeps, NO RPC ──────────────────
        try:
            while True:
                try:
                    raw = ws.recv()
                except Exception:
                    raise   # triggers reconnect via _run

                # Track liveness for heartbeat and blind-exit
                with self._frame_lock:
                    self._last_frame_ts = time.time()

                if not raw:
                    raise ConnectionError("empty recv — PumpPortal closed connection")

                try:
                    msg = json.loads(raw)
                except Exception:
                    continue   # pong or non-JSON control frame — frame_ts already updated

                self._handle_message(msg)
        finally:
            # Record drop telemetry before cleanup
            self._record_drop(conn_start)
            with self._ws_lock:
                self._ws = None

    def _record_drop(self, conn_start: float):
        """Log drop telemetry and accumulate daily counters."""
        now = time.time()
        today = time.strftime("%Y-%m-%d", time.gmtime(now))

        with self._frame_lock:
            last_frame_ts = self._last_frame_ts
        last_frame_age = (now - last_frame_ts) if last_frame_ts > 0 else float("inf")
        lifetime       = now - conn_start if conn_start > 0 else 0

        with self._telem_lock:
            # Day rollover — emit yesterday's summary
            if self._telemetry_day and today != self._telemetry_day:
                log.info(
                    "PumpPortal daily summary %s: drops=%d  gap_sec=%.0f  blind_exits=%d",
                    self._telemetry_day,
                    self._daily_drops,
                    self._daily_gap_sec,
                    self._daily_blind_exits,
                )
                self._daily_drops      = 0
                self._daily_gap_sec    = 0.0
                self._daily_blind_exits = 0
            self._telemetry_day  = today
            self._daily_drops   += 1
            self._drop_start_ts  = now   # gap starts now; closed in _connect on next success

        log.warning(
            "PumpPortal WS drop: ts=%s  lifetime=%.0fs  last_frame=%.1fs ago",
            time.strftime("%H:%M:%S", time.gmtime(now)),
            lifetime,
            last_frame_age,
        )

    # ------------------------------------------------------------------
    # WS send helpers
    # ------------------------------------------------------------------

    def _send_subscribe(self, mints: set):
        msg = json.dumps({"method": "subscribeTokenTrade", "keys": list(mints)})
        self._ws_send(msg)

    def _send_unsubscribe(self, mints: set):
        msg = json.dumps({"method": "unsubscribeTokenTrade", "keys": list(mints)})
        self._ws_send(msg)

    def _ws_send(self, msg: str):
        with self._ws_lock:
            ws = self._ws
        if ws is None:
            return
        try:
            ws.send(msg)
        except Exception as e:
            log.debug("PumpPortal ws_send failed: %s", e)

    # ------------------------------------------------------------------
    # Message handling + price derivation
    # ------------------------------------------------------------------

    def _compute_price(self, msg: dict) -> float:
        """Derive USD price from virtual reserves, falling back to trade amounts."""
        with self._sol_price_lock:
            sol_price = self._sol_price
        v_sol    = msg.get("vSolInBondingCurve")
        v_tokens = msg.get("vTokensInBondingCurve")
        if v_sol and v_tokens and float(v_tokens) > 0:
            return (float(v_sol) / (float(v_tokens) / 1e6)) * sol_price
        sol_amt   = msg.get("solAmount")
        token_amt = msg.get("tokenAmount")
        if sol_amt and token_amt and float(token_amt) > 0:
            return (float(sol_amt) / (float(token_amt) / 1e6)) * sol_price
        return 0.0

    def _handle_message(self, msg: dict):
        """
        Parse a PumpPortal trade event.
        Updates position-monitoring price cache and any active ScreeningState.
        Called only from the WS recv thread.
        """
        mint = msg.get("mint")
        if not mint:
            return

        price_usd = self._compute_price(msg)

        # ── Position-monitoring price cache ───────────────────────────────
        if price_usd > 0:
            with self._cache_lock:
                self._price_cache[mint] = (price_usd, time.time())
            log.debug("PumpPortal price update %s: $%.10f  (vSol=%s vTok=%s)",
                      mint[:8], price_usd,
                      msg.get("vSolInBondingCurve"), msg.get("vTokensInBondingCurve"))
            # Fire registered callbacks — must not block
            with self._cb_lock:
                cbs = list(self._price_callbacks)
            for cb in cbs:
                try:
                    cb(mint, price_usd)
                except Exception as _e:
                    log.debug("price callback error: %s", _e)

        # ── Screening accumulator ─────────────────────────────────────────
        with self._screening_lock:
            sc_state = self._screening.get(mint)
        if sc_state is not None:
            self._update_screening(sc_state, msg, price_usd)

    def _update_screening(self, state: ScreeningState, msg: dict, price_usd: float):
        tx_type = msg.get("txType", "")
        trader  = msg.get("traderPublicKey", "")
        sol_amt = float(msg.get("solAmount") or 0)

        if tx_type == "buy":
            state.buy_count += 1
            state.sol_in    += sol_amt
            if trader:
                state.unique_buyers.add(trader)
        elif tx_type == "sell":
            state.sell_count += 1
            state.sol_out    += sol_amt
            if trader and trader == state.creator_pubkey:
                state.creator_sold = True

        if price_usd > 0:
            if state.first_seen_price <= 0:
                state.first_seen_price = price_usd
            state.latest_price = price_usd

        state.lru_ts = time.time()

    def _refresh_sol_price(self):
        """Fetch current SOL/USD price via Jupiter quote. Called from _sol_price_thread."""
        try:
            resp = requests.get(
                "https://lite-api.jup.ag/swap/v1/quote",
                params={
                    "inputMint":  "So11111111111111111111111111111111111111112",
                    "outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                    "amount":     1_000_000_000,
                },
                timeout=5,
            )
            resp.raise_for_status()
            usdc_out = float(resp.json()["outAmount"])
            new_price = round(usdc_out / 1e6, 4)
            with self._sol_price_lock:
                self._sol_price = new_price
            log.debug("SOL price refreshed: $%.2f", new_price)
        except Exception as e:
            log.debug("SOL price refresh failed: %s", e)


# Module-level singleton — imported by scanner.py and helius_account_monitor.py
monitor = PumpPortalMonitor()
