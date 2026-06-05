"""
Sniper scanner — orchestrates listener → filter → portfolio.

Three background threads:
  listener_thread  : Pump.fun WebSocket, pushes PumpEvents to queue
  snipe_thread     : consumes events, runs fast filter, opens paper positions
  monitor_thread   : updates prices on open positions every 30s, evaluates exits
"""

import logging
import threading
import time
from typing import Optional

from sniper.config import ACTIVE_STRATEGY, SNIPE_COOLDOWN_SEC
from sniper.listener import PumpListener, PumpEvent
from sniper.filter import fast_filter
from sniper.executor import get_current_price_sol
from sniper.portfolio import SniperPortfolio

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

portfolio = SniperPortfolio()
_listener: Optional[PumpListener] = None

# dedup: don't snipe same mint twice within cooldown window
_sniped: dict[str, float] = {}   # mint → timestamp

_lock = threading.Lock()
_started = False


def _is_duplicate(mint: str) -> bool:
    now = time.time()
    with _lock:
        last = _sniped.get(mint, 0)
        if now - last < SNIPE_COOLDOWN_SEC:
            return True
        _sniped[mint] = now
    return False


# ---------------------------------------------------------------------------
# Snipe thread — consumes WebSocket events
# ---------------------------------------------------------------------------

def _snipe_thread():
    log.info("Snipe thread started — strategy: %s", ACTIVE_STRATEGY)
    while True:
        try:
            event: Optional[PumpEvent] = _listener.get(timeout=2.0)
            if not event:
                continue

            # Filter by active strategy
            if ACTIVE_STRATEGY == "migration" and event.event_type != "migration":
                continue
            if ACTIVE_STRATEGY == "launch" and event.event_type != "new_token":
                continue

            if _is_duplicate(event.mint):
                continue

            # Fast filter (< 1s)
            result = fast_filter(event)
            if not result.passed:
                log.debug("SNIPE skip %s/%s — %s", event.symbol, event.mint[:8], result.reason)
                continue

            # Get entry price
            price_sol = get_current_price_sol(event.mint)
            if price_sol <= 0:
                log.debug("SNIPE skip %s — could not get price", event.mint[:8])
                continue

            # Open paper position
            try:
                portfolio.open_position(event, price_sol)
            except Exception as e:
                log.warning("SNIPE open_position failed for %s: %s", event.symbol, e)

        except Exception as e:
            log.warning("Snipe thread error: %s", e)
            time.sleep(1)


# ---------------------------------------------------------------------------
# Monitor thread — price updates + exit evaluation
# ---------------------------------------------------------------------------

def _monitor_thread():
    log.info("Sniper monitor started")
    while True:
        try:
            open_pos = portfolio.open_positions()
            if open_pos:
                prices = {}
                for pos in open_pos:
                    price = get_current_price_sol(pos.mint)
                    if price > 0:
                        prices[pos.mint] = price
                    time.sleep(0.3)   # gentle rate limiting

                exits = portfolio.update_prices(prices)
                for e in exits:
                    log.info("SNIPE EXIT  %s  %s  pnl=%.1f%%  $%.2f",
                             e["symbol"], e["reason"], e["pnl_pct"], e["pnl_usd"])
        except Exception as e:
            log.warning("Sniper monitor error: %s", e)

        time.sleep(30)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start(daemon: bool = True):
    global _listener, _started
    if _started:
        return
    _started = True

    strategies = [ACTIVE_STRATEGY]
    _listener = PumpListener(strategies=strategies)
    _listener.start(daemon=daemon)

    for target in [_snipe_thread, _monitor_thread]:
        t = threading.Thread(target=target, daemon=daemon)
        t.start()

    log.info("Sniper started — strategy=%s", ACTIVE_STRATEGY)


def get_open_positions() -> list[dict]:
    from dataclasses import asdict
    return [asdict(p) for p in portfolio.open_positions()]


def get_journal(limit: int = 200) -> list[dict]:
    return portfolio.load_journal()[-limit:]


def get_summary() -> dict:
    return portfolio.summary()


def manual_close(pos_id: str) -> Optional[dict]:
    from dataclasses import asdict
    pos = portfolio.manual_close(pos_id)
    return asdict(pos) if pos else None
