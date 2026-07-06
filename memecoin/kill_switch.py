"""kill_switch.py — runtime kill switches for live buys and live sells.

Auto-disabled on unknown sell failure (buys only by default).
Re-enable via Telegram commands /buys_on, /sells_on, or by restarting the bot.
"""
import logging
import threading

log = logging.getLogger(__name__)

_lock = threading.Lock()
_live_buys_enabled  = True   # runtime state; initialized from config at import
_live_sells_enabled = True   # independent sell kill switch

try:
    from memecoin.config import LIVE_BUYS_ENABLED as _cfg_enabled
    _live_buys_enabled = _cfg_enabled
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Buy switch
# ---------------------------------------------------------------------------

def live_buys_enabled() -> bool:
    with _lock:
        return _live_buys_enabled


def disable_live_buys(reason: str = "") -> None:
    global _live_buys_enabled
    with _lock:
        if _live_buys_enabled:
            _live_buys_enabled = False
            log.error("KILL SWITCH: live buys DISABLED  reason=%s", reason)
            try:
                from app.alerts import _send
                _send(f"🚨 KILL SWITCH: live buys disabled\nReason: {reason}\nManual re-enable required.\nSend /buys_on to re-enable.")
            except Exception:
                pass


def enable_live_buys(reason: str = "") -> None:
    global _live_buys_enabled
    with _lock:
        _live_buys_enabled = True
        log.warning("KILL SWITCH: live buys RE-ENABLED  reason=%s", reason)
        try:
            from app.alerts import _send
            _send(f"✅ Live buys RE-ENABLED  reason={reason or 'manual'}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Sell switch  (independent — stops close_position from executing on-chain sells)
# ---------------------------------------------------------------------------

def live_sells_enabled() -> bool:
    with _lock:
        return _live_sells_enabled


def disable_live_sells(reason: str = "") -> None:
    global _live_sells_enabled
    with _lock:
        if _live_sells_enabled:
            _live_sells_enabled = False
            log.error("KILL SWITCH: live sells DISABLED  reason=%s", reason)
            try:
                from app.alerts import _send
                _send(
                    f"🔴 Live sells DISABLED\nReason: {reason or 'manual'}\n"
                    f"Open positions will NOT be sold until re-enabled.\n"
                    f"Send /sells_on to re-enable."
                )
            except Exception:
                pass


def enable_live_sells(reason: str = "") -> None:
    global _live_sells_enabled
    with _lock:
        _live_sells_enabled = True
        log.warning("KILL SWITCH: live sells RE-ENABLED  reason=%s", reason)
        try:
            from app.alerts import _send
            _send(f"✅ Live sells RE-ENABLED  reason={reason or 'manual'}")
        except Exception:
            pass
