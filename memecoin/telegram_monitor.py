"""
Telegram channel monitor for social-driven token signals.

Monitors configured channels for token contract addresses.
When a Solana address is found in a message:
  1. Cross-references with DexScreener
  2. Runs safety screening
  3. Fires a social_alert signal if token is new (<60 min) and passes basic checks

Signals are tagged config_tag="social" and use $1 paper size for data collection.
Goal: measure whether social/CT-driven entries outperform pure volume signals.

Runs as a background asyncio thread inside the scanner process.

State machine:
  AUTH_REQUIRED      — session missing/expired; operator must run tg_auth.py
  NETWORK_ERROR      — transient connection failure; retrying with backoff
  RATE_LIMITED       — Telegram rate limit hit; backing off
  CONNECTED          — live, receiving messages
  CONNECTED_BUT_STALE — connected but no message for >2h
  THREAD_DEAD        — tg-monitor thread is not alive
"""

import asyncio
import enum
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

log = logging.getLogger(__name__)

# Mint address anchored to this channel's fixed alert template: the real
# token mint always sits on its own line immediately before the "USD:" (or
# "Dex:") line. A prior version matched every base58/hex substring anywhere
# in the message — including holder-table and other embedded strings — and
# fired one signal per match. Verified against 822 unique historical alert
# messages: the anchored pattern below matched correctly in all 822; taking
# the first raw match instead would have picked the wrong address in 33% of
# them, and filtering by a "pump"-suffix heuristic would have wrongly
# rejected 13% of real (non-suffixed) mints. See RECEIPTS.md 2026-08-08.
_SOL_MINT_ANCHOR_RE = re.compile(r'\n([1-9A-HJ-NP-Za-km-z]{32,44})\n\s*(?:USD:|Dex:)')
_BSC_MINT_ANCHOR_RE = re.compile(r'\n(0x[0-9a-fA-F]{40})\n\s*(?:USD:|Dex:)')

# Channels to monitor (without @)
CHANNELS = [
    "pumpdotfunalert",
]

# Cooldown: don't re-process same address within this many seconds.
# Only applied to addresses that passed DexScreener (have data).
# Addresses that failed no_dex_data are NOT added here — they retry freely.
_SEEN_ADDRESSES: dict[str, float] = {}
_SEEN_COOLDOWN = 300  # 5 minutes

# Retry queue: addresses that hit no_dex_data get retried after this delay.
# DexScreener typically indexes pump.fun tokens within 30-90 seconds of launch.
_NO_DEX_RETRY_DELAY = 45   # seconds between retries
_NO_DEX_MAX_RETRIES = 8    # 8 × 45s = 360s window; covers DexScreener worst-case lag (~120s)

# Stale threshold: CONNECTED_BUT_STALE after this many seconds with no message
_STALE_THRESHOLD_S = 2 * 3600   # 2 hours

# Auth-failure keywords in exception strings — do not retry fast on these
_AUTH_ERROR_KEYWORDS = ("eof when reading", "auth_key", "phone", "401",
                        "session", "unauthorized")


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class TGState(enum.Enum):
    AUTH_REQUIRED       = "AUTH_REQUIRED"
    NETWORK_ERROR       = "NETWORK_ERROR"
    RATE_LIMITED        = "RATE_LIMITED"
    CONNECTED           = "CONNECTED"
    CONNECTED_BUT_STALE = "CONNECTED_BUT_STALE"
    THREAD_DEAD         = "THREAD_DEAD"


# Module-level state (written by monitor thread, read by health endpoint)
_state_lock = threading.Lock()
_tg_state: TGState = TGState.NETWORK_ERROR
_last_connected: float = 0.0
_last_message_received: float = 0.0
_monitor_start: float = time.time()


def _set_state(new_state: TGState):
    global _tg_state
    with _state_lock:
        _tg_state = new_state


def _update_last_connected():
    global _last_connected
    with _state_lock:
        _last_connected = time.time()


def _update_last_message():
    global _last_message_received
    with _state_lock:
        _last_message_received = time.time()


def get_tg_state() -> dict:
    """Return current TG monitor state. Used by health endpoint."""
    with _state_lock:
        st   = _tg_state
        lc   = _last_connected
        lm   = _last_message_received
        up   = time.time() - _monitor_start
    return {
        "state":         st.value,
        "last_connected": lc or None,
        "last_message":  lm or None,
        "uptime_s":      round(up, 1),
    }


def _is_auth_error(exc: Exception) -> bool:
    """Return True if the exception string matches known auth/session failure patterns."""
    s = str(exc).lower()
    return any(kw in s for kw in _AUTH_ERROR_KEYWORDS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_fresh(address: str) -> bool:
    """Return True if we haven't successfully processed this address recently."""
    last = _SEEN_ADDRESSES.get(address, 0)
    if time.time() - last < _SEEN_COOLDOWN:
        return False
    _SEEN_ADDRESSES[address] = time.time()
    return True


def _mark_seen(address: str):
    """Mark address as successfully processed (start cooldown)."""
    _SEEN_ADDRESSES[address] = time.time()


def _extract_addresses(text: str) -> list[tuple[str, str]]:
    """Extract (chain, address) pairs from message text.

    One address per message, anchored to the alert template (see
    _SOL_MINT_ANCHOR_RE comment). If neither anchor matches — the channel's
    template changed — this fails closed (skips the message, logs a
    warning) rather than falling back to the old unanchored scan, which
    would just reintroduce the many-spurious-signals bug it replaces.
    """
    results = []
    sol_m = _SOL_MINT_ANCHOR_RE.search(text)
    if sol_m:
        addr = sol_m.group(1)
        if len(addr) >= 32 and _is_fresh(addr):
            results.append(("solana", addr))
    bsc_m = _BSC_MINT_ANCHOR_RE.search(text)
    if bsc_m:
        addr = bsc_m.group(1)
        if _is_fresh(addr):
            results.append(("bsc", addr))
    if not results and not sol_m and not bsc_m:
        log.warning("tg_monitor: no anchored mint found (alert format changed?) — skipping message: %r",
                    text[:150])
    return results


def _send_alert(msg: str):
    try:
        from app.alerts import send_alert as _sa
        _sa(msg)
    except Exception as e:
        log.debug("tg_monitor: alert send failed: %s", e)


def _log_executor_failure(future, address: str) -> None:
    """V8-TWIN-FIX VF4: done-callback for run_in_executor's Future so an
    exception that somehow escapes _screen_and_signal's own try/except
    (e.g. a BaseException subclass) can't disappear silently. Fired by
    the event loop when the background thread finishes -- never blocks
    the loop, never awaits the future."""
    try:
        exc = future.exception()
    except Exception:
        return   # cancelled or otherwise not retrievable — nothing to log
    if exc is not None:
        log.error("TG executor worker failed for %s: %r", address[:8], exc)


def _check_auth_sync(api_id: int, api_hash: str, session_file: str) -> bool:
    """
    Synchronously check if the Telethon session is authorised.
    Creates a temporary event loop to run the async check.
    Returns True if authorised, False otherwise.
    """
    try:
        from telethon import TelegramClient
    except ImportError:
        return False

    async def _inner():
        client = TelegramClient(session_file, api_id, api_hash)
        await client.connect()
        try:
            return await client.is_user_authorized()
        finally:
            await client.disconnect()

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_inner())
    except Exception as e:
        log.debug("auth check error: %s", e)
        return False
    finally:
        try:
            loop.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# TelegramMonitor
# ---------------------------------------------------------------------------

class TelegramMonitor:
    """
    Async Telegram monitor. Runs in a dedicated thread with its own event loop.
    Calls signal_callback(chain, address, message_text) for each new token found.

    Screening runs in a ThreadPoolExecutor so the Telegram event loop is never
    blocked by HTTP calls — rapid signals are all received immediately.

    no_dex_data addresses are retried after _NO_DEX_RETRY_DELAY seconds
    (DexScreener indexes most pump.fun tokens within 30-90s of launch).
    """

    def __init__(self, api_id: int, api_hash: str, signal_callback):
        self.api_id = api_id
        self.api_hash = api_hash
        self.signal_callback = signal_callback
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Thread pool for running synchronous screening without blocking event loop
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tg-screen")
        # Alarm cooldown: don't spam AUTH_REQUIRED alerts
        self._last_auth_alert: float = 0.0
        self._auth_alert_interval = 300   # 5 minutes between auth alerts

    def start(self, daemon: bool = True):
        """Start the monitor in a background thread."""
        self._thread = threading.Thread(
            target=self._run_loop, daemon=daemon, name="tg-monitor"
        )
        self._thread.start()
        log.warning("Telegram monitor thread started — channels: %s", CHANNELS)
        # Start watchdog thread
        _wt = threading.Thread(
            target=self._watchdog, daemon=True, name="tg-watchdog"
        )
        _wt.start()

    def _watchdog(self):
        """Watchdog: detect THREAD_DEAD and alert once per 30-min interval."""
        _last_dead_alert: float = 0.0
        while True:
            time.sleep(60)
            if self._thread is None:
                continue
            if not self._thread.is_alive():
                _set_state(TGState.THREAD_DEAD)
                now = time.time()
                if now - _last_dead_alert > 1800:
                    _last_dead_alert = now
                    log.error("HEALTH: tg-monitor thread is dead")
                    _send_alert("TELEGRAM THREAD_DEAD — tg-monitor thread not alive. Restart quantbot.")
            else:
                # Check for stale connection
                with _state_lock:
                    current_state = _tg_state
                    lm = _last_message_received
                if current_state == TGState.CONNECTED and lm > 0:
                    if time.time() - lm > _STALE_THRESHOLD_S:
                        _set_state(TGState.CONNECTED_BUT_STALE)
                        log.warning("TG feed connected but no message for >2h — CONNECTED_BUT_STALE")

    def _run_loop(self):
        global _monitor_start
        _monitor_start = time.time()
        backoff = 5

        while True:
            session_file = os.path.join(
                os.path.dirname(__file__), "data", "tg_session"
            )

            # Auth check before attempting connection
            authorized = _check_auth_sync(self.api_id, self.api_hash, session_file)
            if not authorized:
                _set_state(TGState.AUTH_REQUIRED)
                now = time.time()
                if now - self._last_auth_alert > self._auth_alert_interval:
                    self._last_auth_alert = now
                    log.error(
                        "TELEGRAM_AUTH_REQUIRED — session not authorised. "
                        "Run: python -m research.tg_auth"
                    )
                    _send_alert(
                        "TELEGRAM_AUTH_REQUIRED — Telethon session expired or missing. "
                        "SSH to server and run: python -m research.tg_auth"
                    )
                time.sleep(300)
                continue

            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._monitor())
                log.warning("Telegram monitor exited cleanly — restarting in %ds", backoff)
                _set_state(TGState.NETWORK_ERROR)
            except Exception as e:
                if _is_auth_error(e):
                    _set_state(TGState.AUTH_REQUIRED)
                    now = time.time()
                    if now - self._last_auth_alert > self._auth_alert_interval:
                        self._last_auth_alert = now
                        log.error(
                            "TELEGRAM_AUTH_REQUIRED (exception: %s) — run tg_auth.py", e
                        )
                        _send_alert(
                            "TELEGRAM_AUTH_REQUIRED — session error detected. "
                            "SSH to server and run: python -m research.tg_auth"
                        )
                    time.sleep(300)
                    continue
                else:
                    _set_state(TGState.NETWORK_ERROR)
                    log.error("Telegram monitor crashed: %s — restarting in %ds", e, backoff)
            finally:
                try:
                    self._loop.close()
                except Exception:
                    pass

            time.sleep(backoff)
            backoff = min(backoff * 2, 120)   # cap at 2 min

    def _screen_and_signal(self, chain: str, address: str, text: str,
                            attempt: int = 1):
        """
        Run in thread pool — does the blocking HTTP screening then fires callback.
        If DexScreener returns no_dex_data (_NoDexData raised), clears the seen
        entry and schedules a retry so the address isn't suppressed for 5 minutes.
        """
        try:
            self.signal_callback(chain, address, text)
        except Exception as e:
            # Check if it's a no_dex_data signal from the scanner
            if type(e).__name__ == "_NoDexData":
                # Clear from seen so retry can proceed without cooldown blocking it
                _SEEN_ADDRESSES.pop(address, None)
                self._schedule_retry(chain, address, text, attempt)
            else:
                log.warning("TG screen error %s: %s", address[:8], e)

    def _schedule_retry(self, chain: str, address: str, text: str, attempt: int):
        """Schedule a no_dex_data retry after _NO_DEX_RETRY_DELAY seconds."""
        if attempt > _NO_DEX_MAX_RETRIES:
            log.info("TG no_dex_data %s — max retries (%d) reached, giving up",
                     address[:8], _NO_DEX_MAX_RETRIES)
            return

        def _retry():
            time.sleep(_NO_DEX_RETRY_DELAY)
            # Only retry if still not in seen-cooldown (not processed by another path)
            if address not in _SEEN_ADDRESSES:
                log.info("TG no_dex_data retry %d/%d for %s",
                         attempt, _NO_DEX_MAX_RETRIES, address[:8])
                _SEEN_ADDRESSES[address] = time.time()  # reserve slot before retry
                try:
                    self.signal_callback(chain, address, text)
                except Exception as e:
                    log.warning("TG retry error %s: %s", address[:8], e)
                    # If still no_dex_data, clear reservation and try again
                    if address in _SEEN_ADDRESSES:
                        del _SEEN_ADDRESSES[address]
                    self._schedule_retry(chain, address, text, attempt + 1)

        t = threading.Thread(target=_retry, daemon=True,
                             name=f"tg-retry-{address[:8]}")
        t.start()

    async def _monitor(self):
        try:
            from telethon import TelegramClient, events
        except ImportError:
            log.warning("telethon not installed — Telegram monitor disabled. Run: pip install telethon")
            return

        session_file = os.path.join(
            os.path.dirname(__file__), "data", "tg_session"
        )

        async with TelegramClient(session_file, self.api_id, self.api_hash) as client:
            _set_state(TGState.CONNECTED)
            _update_last_connected()
            log.warning("Telegram client connected")

            @client.on(events.NewMessage(chats=CHANNELS))
            async def handler(event):
                _update_last_message()
                # If we were stale, flip back to CONNECTED on new message
                with _state_lock:
                    if _tg_state == TGState.CONNECTED_BUT_STALE:
                        pass
                _set_state(TGState.CONNECTED)

                text = event.raw_text or ""
                extra_urls = []
                if event.message and event.message.entities:
                    for ent in event.message.entities:
                        url = getattr(ent, "url", None)
                        if url:
                            extra_urls.append(url)
                combined = text + " " + " ".join(extra_urls)
                addresses = _extract_addresses(combined)
                for chain, address in addresses:
                    log.warning(
                        "TG signal: %s address=%s from channel=%s",
                        chain, address[:12], event.chat.username or "?"
                    )
                    # Run screening in thread pool — event loop returns immediately
                    # so the next Telegram message isn't delayed by HTTP calls
                    loop = asyncio.get_event_loop()
                    _fut = loop.run_in_executor(
                        self._executor,
                        self._screen_and_signal, chain, address, combined, 1
                    )
                    # V8-TWIN-FIX VF4: retrieve the Future's exception via a
                    # done-callback instead of leaving it unretrieved.
                    # _screen_and_signal already catches Exception internally
                    # (confirmed not the cause of the 15-candidate mystery --
                    # see docs/RECEIPTS.md's V8-TWIN-FIX section), so this
                    # covers the narrower residual case: something that
                    # escapes that catch (e.g. a BaseException subclass) or
                    # a wrapper-level failure in run_in_executor itself. Does
                    # not await the future -- purely a callback fired by the
                    # event loop once the background thread finishes, so the
                    # loop itself is never blocked.
                    _fut.add_done_callback(
                        lambda f, _addr=address: _log_executor_failure(f, _addr)
                    )

            await client.run_until_disconnected()
