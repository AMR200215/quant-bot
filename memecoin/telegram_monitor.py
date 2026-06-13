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
"""

import asyncio
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

log = logging.getLogger(__name__)

# Solana address pattern: base58, 32-44 chars
_SOL_ADDRESS_RE = re.compile(r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b')

# BSC address pattern: 0x + 40 hex chars
_BSC_ADDRESS_RE = re.compile(r'\b0x[0-9a-fA-F]{40}\b')

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
    """Extract (chain, address) pairs from message text."""
    results = []
    for addr in _SOL_ADDRESS_RE.findall(text):
        if len(addr) >= 32 and _is_fresh(addr):
            results.append(("solana", addr))
    for addr in _BSC_ADDRESS_RE.findall(text):
        if _is_fresh(addr):
            results.append(("bsc", addr))
    return results


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

    def start(self, daemon: bool = True):
        """Start the monitor in a background thread."""
        self._thread = threading.Thread(
            target=self._run_loop, daemon=daemon, name="tg-monitor"
        )
        self._thread.start()
        log.warning("Telegram monitor thread started — channels: %s", CHANNELS)

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._monitor())
        except Exception as e:
            log.error("Telegram monitor crashed: %s", e)

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
            log.warning("Telegram client connected")

            @client.on(events.NewMessage(chats=CHANNELS))
            async def handler(event):
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
                    loop.run_in_executor(
                        self._executor,
                        self._screen_and_signal, chain, address, combined, 1
                    )

            await client.run_until_disconnected()
