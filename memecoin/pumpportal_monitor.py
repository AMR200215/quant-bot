"""
Real-time price monitor for open pump.fun positions via PumpPortal WebSocket.

Subscribes to trade events for tokens we hold. Each trade event contains
virtual reserve data → we compute price directly from reserves, bypassing
DexScreener's 5-30s indexing lag.

Price formula:
  price_usd = (vSolInBondingCurve / (vTokensInBondingCurve / 1e6)) * sol_price_usd

Latency: ~1s from on-chain confirm to price update vs 10s DexScreener poll.
This is the difference between exiting at -35% and booking a -75% loss.

portfolio.py reads prices via get_prices() and uses them as overrides when
fresh (< 15s). DexScreener remains the fallback heartbeat for graduated tokens
that no longer appear in PumpPortal's stream.
"""

import json
import logging
import queue
import threading
import time
from typing import Optional

import requests

log = logging.getLogger(__name__)

WS_URL = "wss://pumpportal.fun/api/data"

# Price is considered stale if no update received in this many seconds.
# DexScreener fallback kicks in for tokens exceeding this threshold.
PRICE_STALE_SEC = 15.0

# SOL price refresh interval (seconds). Uses Jupiter quote.
SOL_PRICE_REFRESH_SEC = 60.0

# Reconnect delay on WS failure
_RECONNECT_DELAY_BASE = 10
_RECONNECT_DELAY_MAX  = 120


class PumpPortalMonitor:
    """
    Maintains a live price cache for subscribed pump.fun tokens.

    Usage:
      monitor = PumpPortalMonitor()
      monitor.start()
      monitor.subscribe({"MINT1", "MINT2"})
      prices = monitor.get_prices()   # {mint: (price_usd, timestamp)}
      monitor.unsubscribe({"MINT1"})
    """

    def __init__(self):
        self._subscribed: set[str]   = set()
        self._sub_lock               = threading.Lock()
        # mint → (price_usd, timestamp)
        self._price_cache: dict[str, tuple[float, float]] = {}
        self._cache_lock             = threading.Lock()
        self._sol_price: float       = 170.0
        self._sol_price_ts: float    = 0.0
        self._ws                     = None
        self._ws_lock                = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        # Commands queued while WS is connecting/reconnecting
        self._pending: queue.Queue   = queue.Queue()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, daemon: bool = True):
        self._thread = threading.Thread(
            target=self._run, daemon=daemon, name="pumpportal-monitor"
        )
        self._thread.start()
        log.info("PumpPortal monitor thread started")

    def subscribe(self, mints: set):
        """Subscribe to trade events for these mints."""
        with self._sub_lock:
            new = mints - self._subscribed
            if not new:
                return
            self._subscribed |= new
        self._send_subscribe(new)
        log.info("PumpPortal subscribed to %d token(s): %s",
                 len(new), ", ".join(m[:8] for m in new))

    def unsubscribe(self, mints: set):
        """Unsubscribe from these mints and clear their cached prices."""
        with self._sub_lock:
            gone = mints & self._subscribed
            if not gone:
                return
            self._subscribed -= gone
        self._send_unsubscribe(gone)
        with self._cache_lock:
            for m in gone:
                self._price_cache.pop(m, None)
        log.info("PumpPortal unsubscribed from %d token(s)", len(gone))

    def get_prices(self, max_age: float = PRICE_STALE_SEC) -> dict[str, float]:
        """
        Return fresh prices as {mint: price_usd} for positions we monitor.
        Only includes entries updated within max_age seconds.
        """
        now = time.time()
        with self._cache_lock:
            return {
                mint: price
                for mint, (price, ts) in self._price_cache.items()
                if now - ts <= max_age
            }

    def is_active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

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
            time.sleep(3600)  # don't spam reconnect attempts
            return

        ws = _ws_mod.create_connection(WS_URL, timeout=30)
        log.info("PumpPortal WebSocket connected")

        with self._ws_lock:
            self._ws = ws

        # Re-subscribe to all currently tracked mints (handles reconnect)
        with self._sub_lock:
            current = set(self._subscribed)
        if current:
            self._send_subscribe(current)

        last_sol_refresh = 0.0

        while True:
            # Refresh SOL price periodically
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

    def _handle_message(self, msg: dict):
        """
        Parse a PumpPortal trade event and update price cache.

        Price formula: price_usd = (vSolInBondingCurve / (vTokensInBondingCurve / 1e6))
                                   * sol_price_usd
        vSolInBondingCurve is in SOL; vTokensInBondingCurve is in raw units (6 decimals).
        """
        mint = msg.get("mint")
        if not mint:
            return

        v_sol    = msg.get("vSolInBondingCurve")
        v_tokens = msg.get("vTokensInBondingCurve")

        if v_sol and v_tokens and float(v_tokens) > 0:
            price_sol = float(v_sol) / (float(v_tokens) / 1e6)
            price_usd = price_sol * self._sol_price
            with self._cache_lock:
                self._price_cache[mint] = (price_usd, time.time())
            log.debug("PumpPortal price update %s: $%.10f  (vSol=%.4f vTok=%.0f)",
                      mint[:8], price_usd, v_sol, v_tokens)
        else:
            # Fallback: derive from solAmount / tokenAmount if reserves missing
            sol_amt   = msg.get("solAmount")
            token_amt = msg.get("tokenAmount")
            if sol_amt and token_amt and float(token_amt) > 0:
                price_sol = float(sol_amt) / (float(token_amt) / 1e6)
                price_usd = price_sol * self._sol_price
                with self._cache_lock:
                    self._price_cache[mint] = (price_usd, time.time())

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
