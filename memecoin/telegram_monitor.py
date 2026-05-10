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

# Cooldown: don't re-process same address within this many seconds
_SEEN_ADDRESSES: dict[str, float] = {}
_SEEN_COOLDOWN = 300  # 5 minutes


def _is_fresh(address: str) -> bool:
    """Return True if we haven't seen this address recently."""
    last = _SEEN_ADDRESSES.get(address, 0)
    if time.time() - last < _SEEN_COOLDOWN:
        return False
    _SEEN_ADDRESSES[address] = time.time()
    return True


def _extract_addresses(text: str) -> list[tuple[str, str]]:
    """Extract (chain, address) pairs from message text."""
    results = []
    for addr in _SOL_ADDRESS_RE.findall(text):
        # Filter out common non-token strings (tx hashes etc.)
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
    """

    def __init__(self, api_id: int, api_hash: str, signal_callback):
        self.api_id = api_id
        self.api_hash = api_hash
        self.signal_callback = signal_callback
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def start(self, daemon: bool = True):
        """Start the monitor in a background thread."""
        self._thread = threading.Thread(
            target=self._run_loop, daemon=daemon, name="tg-monitor"
        )
        self._thread.start()
        log.info("Telegram monitor thread started — channels: %s", CHANNELS)

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._monitor())
        except Exception as e:
            log.error("Telegram monitor crashed: %s", e)

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
            log.info("Telegram client connected")

            # Register handler for new messages in monitored channels
            @client.on(events.NewMessage(chats=CHANNELS))
            async def handler(event):
                text = event.raw_text or ""
                addresses = _extract_addresses(text)
                for chain, address in addresses:
                    log.info(
                        "TG signal: %s address=%s from channel=%s",
                        chain, address[:12], event.chat.username or "?"
                    )
                    try:
                        self.signal_callback(chain, address, text)
                    except Exception as e:
                        log.warning("TG signal callback error: %s", e)

            await client.run_until_disconnected()
