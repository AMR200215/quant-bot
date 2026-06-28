"""kill_switch.py — runtime kill switch for live buys.

Auto-disabled when an unknown sell failure occurs.
Re-enable manually by calling enable_live_buys() or restarting the bot
with LIVE_BUYS_ENABLED=True in config (the restart resets the runtime flag).
"""
import logging
import threading

log = logging.getLogger(__name__)

_lock = threading.Lock()
_live_buys_enabled = True  # runtime state; initialized from config at import

try:
    from memecoin.config import LIVE_BUYS_ENABLED as _cfg_enabled
    _live_buys_enabled = _cfg_enabled
except ImportError:
    pass


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
                _send(f"🚨 KILL SWITCH: live buys disabled\nReason: {reason}\nManual re-enable required.")
            except Exception:
                pass


def enable_live_buys(reason: str = "") -> None:
    global _live_buys_enabled
    with _lock:
        _live_buys_enabled = True
        log.warning("KILL SWITCH: live buys RE-ENABLED  reason=%s", reason)
