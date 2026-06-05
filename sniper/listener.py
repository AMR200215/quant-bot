"""
Pump.fun WebSocket listener.

Connects to wss://pumpportal.fun/api/data and streams:
  - New token creation events  (strategy: "launch")
  - Token migration events     (strategy: "migration" — Pump.fun → Raydium)

Runs in a background thread and pushes parsed events into a queue
for the scanner to consume. Auto-reconnects on disconnect.
"""

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

PUMPFUN_WS = "wss://pumpportal.fun/api/data"
_RECONNECT_DELAY = 5   # seconds between reconnect attempts


@dataclass
class PumpEvent:
    event_type: str          # "new_token" | "migration" | "trade"
    mint: str                # token mint address
    name: str
    symbol: str
    trader: str              # deployer wallet
    market_cap_sol: float    # current mcap in SOL
    initial_buy_sol: float   # SOL spent on initial buy
    bonding_curve: str       # bonding curve address
    has_twitter: bool
    has_telegram: bool
    has_website: bool
    timestamp: float         # unix seconds


def _parse_event(raw: dict) -> Optional[PumpEvent]:
    """Parse a raw Pump.fun WebSocket message into a PumpEvent."""
    tx_type = raw.get("txType", "")

    if tx_type not in ("create", "migration"):
        return None

    mint = raw.get("mint", "")
    if not mint:
        return None

    socials = raw.get("socials") or {}
    return PumpEvent(
        event_type="new_token" if tx_type == "create" else "migration",
        mint=mint,
        name=raw.get("name", ""),
        symbol=raw.get("symbol", ""),
        trader=raw.get("traderPublicKey", ""),
        market_cap_sol=float(raw.get("marketCapSol") or 0),
        initial_buy_sol=float(raw.get("solAmount") or 0),
        bonding_curve=raw.get("bondingCurveKey", ""),
        has_twitter=bool(raw.get("twitter") or socials.get("twitter")),
        has_telegram=bool(raw.get("telegram") or socials.get("telegram")),
        has_website=bool(raw.get("website") or socials.get("website")),
        timestamp=time.time(),
    )


class PumpListener:
    """
    Background WebSocket listener for Pump.fun events.

    Usage:
        listener = PumpListener(strategies=["migration"])
        listener.start()
        while True:
            event = listener.get(timeout=1)
            if event: process(event)
    """

    def __init__(self, strategies: list[str] = None):
        """
        strategies: list of "launch" and/or "migration"
        """
        self._strategies = set(strategies or ["migration"])
        self._queue: queue.Queue = queue.Queue(maxsize=500)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self, daemon: bool = True):
        self._thread = threading.Thread(
            target=self._run, daemon=daemon, name="pumpfun-listener"
        )
        self._thread.start()
        log.info("PumpListener started — strategies: %s", self._strategies)

    def stop(self):
        self._stop.set()

    def get(self, timeout: float = 1.0) -> Optional[PumpEvent]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self):
        while not self._stop.is_set():
            try:
                self._connect_and_listen()
            except Exception as e:
                log.warning("PumpListener error: %s — reconnecting in %ds", e, _RECONNECT_DELAY)
            if not self._stop.is_set():
                time.sleep(_RECONNECT_DELAY)

    def _connect_and_listen(self):
        import websocket  # websocket-client

        subscriptions = []
        if "launch" in self._strategies:
            subscriptions.append({"method": "subscribeNewToken"})
        if "migration" in self._strategies:
            # Pump.fun sends migration events via subscribeNewToken with txType="migration"
            subscriptions.append({"method": "subscribeNewToken"})

        # deduplicate subscriptions
        seen_methods = set()
        unique_subs = []
        for s in subscriptions:
            if s["method"] not in seen_methods:
                seen_methods.add(s["method"])
                unique_subs.append(s)

        ws = websocket.WebSocketApp(
            PUMPFUN_WS,
            on_open=lambda ws: self._on_open(ws, unique_subs),
            on_message=self._on_message,
            on_error=lambda ws, err: log.debug("WS error: %s", err),
            on_close=lambda ws, code, msg: log.debug("WS closed: %s %s", code, msg),
        )
        log.info("Connecting to Pump.fun WebSocket...")
        ws.run_forever(ping_interval=30, ping_timeout=10)

    def _on_open(self, ws, subscriptions: list):
        log.info("Pump.fun WS connected — subscribing to %d method(s)", len(subscriptions))
        for sub in subscriptions:
            ws.send(json.dumps(sub))

    def _on_message(self, ws, message: str):
        try:
            raw = json.loads(message)
        except json.JSONDecodeError:
            return

        event = _parse_event(raw)
        if not event:
            return

        # Filter by active strategy
        if event.event_type == "new_token" and "launch" not in self._strategies:
            return
        if event.event_type == "migration" and "migration" not in self._strategies:
            return

        try:
            self._queue.put_nowait(event)
        except queue.Full:
            log.debug("Pump listener queue full — dropping event %s", event.mint[:8])
