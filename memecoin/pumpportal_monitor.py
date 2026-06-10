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
This is the difference between exiting at -35% and booking a -75% loss.
"""

import json
import logging
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import requests

log = logging.getLogger(__name__)

WS_URL = "wss://pumpportal.fun/api/data"

# Price is considered stale if no update received in this many seconds.
# DexScreener fallback kicks in for tokens exceeding this threshold.
PRICE_STALE_SEC = 15.0

# SOL price refresh interval (seconds). Uses Jupiter quote.
SOL_PRICE_REFRESH_SEC = 60.0

# Max concurrent screening slots before LRU eviction
MAX_SCREENING_SLOTS = 30

# Reconnect delay on WS failure
_RECONNECT_DELAY_BASE = 10
_RECONNECT_DELAY_MAX  = 120


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
    sol_in:           float         = 0.0    # SOL from buy txns
    sol_out:          float         = 0.0    # SOL from sell txns
    creator_sold:     bool          = False
    lru_ts:           float         = field(default_factory=time.time)

    @property
    def net_sol_inflow(self) -> float:
        return self.sol_in - self.sol_out

    @property
    def unique_buyer_count(self) -> int:
        return len(self.unique_buyers)


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
        # Each callback is called with (mint: str, price_usd: float) from the
        # WS recv thread. Must be non-blocking — long work belongs in a queue.
        self._price_callbacks: list  = []
        self._cb_lock                = threading.Lock()

        # ── Shared infrastructure ─────────────────────────────────────────
        self._sol_price: float       = 170.0
        self._sol_price_ts: float    = 0.0
        self._ws                     = None
        self._ws_lock                = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._pending: queue.Queue   = queue.Queue()

    # ------------------------------------------------------------------
    # Public API — position monitoring
    # ------------------------------------------------------------------

    def start(self, daemon: bool = True):
        self._thread = threading.Thread(
            target=self._run, daemon=daemon, name="pumpportal-monitor"
        )
        self._thread.start()
        log.info("PumpPortal monitor thread started")

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

    def is_active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

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

        # Subscribe WS only if not already subscribed as a held position
        with self._sub_lock:
            already = mint in self._subscribed
        if not already:
            self._send_subscribe({mint})

        log.info("PumpPortal screening started: %s (creator=%s)",
                 mint[:8], (creator_pubkey or "unknown")[:8] if creator_pubkey else "unknown")

    def get_screening_state(self, mint: str) -> Optional[ScreeningState]:
        """Return current ScreeningState for a mint, or None if not being screened."""
        with self._screening_lock:
            return self._screening.get(mint)

    def evict_screening(self, mints: set):
        """
        Remove mints from screening accumulator.
        Sends WS unsubscribe only for mints not held as open positions.
        """
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
    # WebSocket management
    # ------------------------------------------------------------------

    def _run(self):
        fail_count = 0
        while True:
            try:
                self._connect()
                fail_count = 0
            except Exception as e:
                fail_count += 1
                delay = min(_RECONNECT_DELAY_BASE * (2 ** min(fail_count - 1, 3)),
                            _RECONNECT_DELAY_MAX)
                log.warning("PumpPortal WS error (attempt %d): %s — retry in %ds",
                            fail_count, e, delay)
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
        log.info("PumpPortal WebSocket connected")

        with self._ws_lock:
            self._ws = ws

        # Re-subscribe to all mints (position monitoring + screening) on reconnect
        with self._sub_lock:
            current = set(self._subscribed)
        with self._screening_lock:
            current |= set(self._screening.keys())
        if current:
            self._send_subscribe(current)

        last_sol_refresh = 0.0

        while True:
            if time.time() - last_sol_refresh > SOL_PRICE_REFRESH_SEC:
                self._refresh_sol_price()
                last_sol_refresh = time.time()

            try:
                raw = ws.recv()
            except Exception:
                raise  # triggers reconnect

            if not raw:
                raise ConnectionError("empty recv")

            try:
                msg = json.loads(raw)
            except Exception:
                continue

            self._handle_message(msg)

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
        v_sol    = msg.get("vSolInBondingCurve")
        v_tokens = msg.get("vTokensInBondingCurve")
        if v_sol and v_tokens and float(v_tokens) > 0:
            return (float(v_sol) / (float(v_tokens) / 1e6)) * self._sol_price
        sol_amt   = msg.get("solAmount")
        token_amt = msg.get("tokenAmount")
        if sol_amt and token_amt and float(token_amt) > 0:
            return (float(sol_amt) / (float(token_amt) / 1e6)) * self._sol_price
        return 0.0

    def _handle_message(self, msg: dict):
        """
        Parse a PumpPortal trade event.
        Updates both the position-monitoring price cache and any active
        ScreeningState for this mint.
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
            # Fire registered callbacks (non-blocking — callers must use queues)
            with self._cb_lock:
                cbs = list(self._price_callbacks)
            for cb in cbs:
                try:
                    cb(mint, price_usd)
                except Exception as _e:
                    log.debug("price callback error: %s", _e)

        # ── Screening accumulator ─────────────────────────────────────────
        # Fetch state reference outside the lock; mutations are only from this
        # thread (WS recv loop) so no concurrent writer race.
        with self._screening_lock:
            sc_state = self._screening.get(mint)

        if sc_state is not None:
            self._update_screening(sc_state, msg, price_usd)

    def _update_screening(self, state: ScreeningState, msg: dict, price_usd: float):
        """Update a ScreeningState from a trade event."""
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

        # Track price evolution
        if price_usd > 0:
            if state.first_seen_price <= 0:
                state.first_seen_price = price_usd
            state.latest_price = price_usd

        state.lru_ts = time.time()

    def _refresh_sol_price(self):
        """Fetch current SOL/USD price via Jupiter quote."""
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
            self._sol_price = round(usdc_out / 1e6, 4)
            log.debug("PumpPortal SOL price refreshed: $%.2f", self._sol_price)
        except Exception as e:
            log.debug("PumpPortal SOL price refresh failed: %s", e)


# Module-level singleton — imported by scanner.py
monitor = PumpPortalMonitor()
