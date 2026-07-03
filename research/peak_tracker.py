"""
PeakTracker — subscribes to PumpPortal token-trade stream for newly alerted
tokens and records the highest price seen in the first TICK_PEAK_WINDOW_S
seconds (default 3 min).

Fixes the intra-interval peak undercount: T1m is the earliest poll but tokens
can peak at T30s.  This gives true tick-level peak for the bonding-curve window.

Writes to Supabase (columns must exist):
  price_peak_3m      FLOAT  — max USD price seen in window
  pct_change_peak_3m FLOAT  — % above entry price at alert time
  t_peak_3m_s        INT    — seconds after alert when peak occurred

Standalone — no memecoin/ imports.
"""

import asyncio
import json
import logging
import threading
import time
from datetime import datetime
from typing import Optional

import requests

from research.config import SUPABASE_URL, SUPABASE_KEY, TICK_PEAK_WINDOW_S, PP_WS_URL
from research.spool.writer import spool_dropped_field

log = logging.getLogger(__name__)

_SOL_MINT = "So11111111111111111111111111111111111111112"
# New optional columns — written only if they exist in Supabase
_PEAK_COLS = ("price_peak_3m", "pct_change_peak_3m", "t_peak_3m_s")


class PeakTracker:
    """
    Runs an asyncio loop in its own daemon thread.
    schedule_token() is thread-safe and can be called from the tracker thread.
    """

    def __init__(self):
        self._tracked: dict     = {}   # addr → state dict
        self._lock              = threading.Lock()
        self._pending: list     = []   # addrs awaiting first WS subscription
        self._pending_lock      = threading.Lock()
        self._sb                = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._sol_price: float  = 175.0

    def start(self):
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="research-peak"
        )
        self._thread.start()
        log.info("PeakTracker thread started")

    def schedule_token(
        self,
        token_address: str,
        alert_time: datetime,
        entry_price: Optional[float],
    ):
        """
        Called from tracker thread after a successful INSERT.
        Adds the token to the 3-min tick-peak tracking window.
        """
        with self._lock:
            if token_address in self._tracked:
                return
            ep = entry_price or 0.0
            self._tracked[token_address] = {
                "entry_price": ep,
                "max_price":   ep,
                "max_ts":      alert_time.timestamp(),
                "alert_ts":    alert_time.timestamp(),
                "expiry":      time.time() + TICK_PEAK_WINDOW_S,
                "done":        False,
            }
        with self._pending_lock:
            self._pending.append(token_address)
        # Signal the asyncio loop to subscribe this token immediately
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(lambda: None)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _init_supabase(self):
        try:
            from supabase import create_client
            self._sb = create_client(SUPABASE_URL, SUPABASE_KEY)
            log.info("PeakTracker: Supabase client initialised")
        except Exception as e:
            log.error("PeakTracker: Supabase init failed: %s", e)

    def _refresh_sol_price(self):
        try:
            r = requests.get(
                f"https://api.jup.ag/price/v2?ids={_SOL_MINT}", timeout=5
            )
            if r.status_code == 200:
                entry = (r.json().get("data") or {}).get(_SOL_MINT)
                if entry:
                    self._sol_price = float(entry.get("price") or self._sol_price)
        except Exception:
            pass

    def _price_from_msg(self, msg: dict) -> Optional[float]:
        """Derive USD price from bonding-curve reserves."""
        vsol = float(msg.get("vSolInBondingCurve") or 0)
        vtok = float(msg.get("vTokensInBondingCurve") or 0)
        if vsol > 0 and vtok > 0:
            return (vsol / vtok) * self._sol_price
        return None

    async def _ws_loop(self):
        """
        Persistent WebSocket to PumpPortal.
        Subscribes to subscribeTokenTrade for each tracked token.
        Reconnects on any error.
        """
        while True:
            try:
                import websockets as _ws_lib
                async with _ws_lib.connect(
                    PP_WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    log.info("PeakTracker: PP WebSocket connected")

                    # Re-subscribe all live tokens after reconnect
                    with self._lock:
                        for addr, st in self._tracked.items():
                            if not st["done"]:
                                await ws.send(json.dumps({
                                    "action": "subscribeTokenTrade",
                                    "tokenAddress": addr,
                                }))

                    async def _recv():
                        async for raw in ws:
                            try:
                                msg = json.loads(raw)
                                if msg.get("errors"):
                                    continue
                                mint = msg.get("mint")
                                if not mint:
                                    continue
                                price = self._price_from_msg(msg)
                                if price is None:
                                    continue
                                now = time.time()
                                with self._lock:
                                    st = self._tracked.get(mint)
                                    if st and not st["done"] and now < st["expiry"]:
                                        if price > st["max_price"]:
                                            st["max_price"] = price
                                            st["max_ts"]    = now
                            except Exception:
                                pass

                    async def _drain_pending():
                        """Subscribe new tokens as they arrive from schedule_token()."""
                        while True:
                            await asyncio.sleep(0.3)
                            with self._pending_lock:
                                new = list(self._pending)
                                self._pending.clear()
                            for addr in new:
                                try:
                                    await ws.send(json.dumps({
                                        "action": "subscribeTokenTrade",
                                        "tokenAddress": addr,
                                    }))
                                except Exception:
                                    # WS probably closed — re-queue for next connect
                                    with self._pending_lock:
                                        self._pending.insert(0, addr)
                                    return

                    await asyncio.gather(_recv(), _drain_pending())

            except Exception as e:
                log.warning("PeakTracker WS: %s — reconnect in 3s", e)
                await asyncio.sleep(3)

    async def _finalise_loop(self):
        """Every 10s: write peaks for expired tokens, purge old state."""
        _last_sol_refresh = 0.0
        while True:
            await asyncio.sleep(10)
            now = time.time()

            # Refresh SOL/USD every 60s (non-blocking)
            if now - _last_sol_refresh > 60:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._refresh_sol_price)
                _last_sol_refresh = now

            # Collect expired tracking windows
            expired = []
            with self._lock:
                for addr, st in list(self._tracked.items()):
                    if not st["done"] and now >= st["expiry"]:
                        st["done"] = True
                        expired.append((addr, dict(st)))
                # Purge done entries older than 1 h
                old = [a for a, s in self._tracked.items()
                       if s["done"] and s["expiry"] < now - 3600]
                for a in old:
                    del self._tracked[a]

            # Write peaks in executor so we don't block the event loop
            loop = asyncio.get_event_loop()
            for addr, st in expired:
                await loop.run_in_executor(None, self._write_peak, addr, st)

    def _write_peak(self, addr: str, st: dict):
        if not self._sb:
            return
        entry    = st["entry_price"]
        peak     = st["max_price"]
        alert_ts = st["alert_ts"]
        max_ts   = st["max_ts"]

        pct_peak = ((peak / entry - 1) * 100) if (entry > 0 and peak > entry) else None
        t_peak_s = int(max_ts - alert_ts)      if (peak > entry) else None

        update = {
            "price_peak_3m":       round(peak, 12) if peak > 0 else None,
            "pct_change_peak_3m":  round(pct_peak, 2) if pct_peak is not None else None,
            "t_peak_3m_s":         t_peak_s,
        }
        import re as _re
        from datetime import datetime, timezone
        _alert_time_iso = datetime.fromtimestamp(alert_ts, tz=timezone.utc).isoformat()
        _update = dict(update)
        for _attempt in range(4):
            try:
                self._sb.table("research_tokens") \
                    .update(_update) \
                    .eq("token_address", addr) \
                    .execute()
                log.info("PeakTracker %s | tick_peak=%.2f%% at T+%ds",
                         addr[:12], pct_peak or 0, t_peak_s or 0)
                return
            except Exception as e:
                e_str = str(e).lower()
                if "pgrst204" in e_str or "schema cache" in e_str:
                    m       = _re.search(r"'(\w+)'\s+column", str(e))
                    missing = m.group(1) if m else None
                    if missing and missing in _update:
                        spool_dropped_field(
                            token_address=addr, symbol="",
                            table="research_tokens", column=missing,
                            value=_update[missing], source_file="peak_tracker.py",
                            insert_context="peak_update",
                            alert_time=_alert_time_iso,
                        )
                        _update = {k: v for k, v in _update.items() if k != missing}
                    else:
                        log.warning("PeakTracker schema error (unrecognised col) for %s: %s",
                                    addr[:8], e)
                        return
                else:
                    log.warning("PeakTracker write error for %s: %s", addr[:8], e)
                    return

    def _run(self):
        self._init_supabase()
        self._refresh_sol_price()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        while True:
            try:
                loop.run_until_complete(
                    asyncio.gather(self._ws_loop(), self._finalise_loop())
                )
            except Exception as e:
                log.error("PeakTracker crashed: %s — restart in 5s", e)
                time.sleep(5)
