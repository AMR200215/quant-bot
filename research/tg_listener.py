"""
Telegram listener for pumpdotfunalert channel.

Independent Telethon session — separate session file from the trading bot's
memecoin/data/tg_session.  Two processes on the same session file would kick
each other out; two on different files (same account) is fine — Telegram
treats them like two logged-in devices.

On first run, call:  python -m research.setup
This authenticates the research session once and saves the session file.
"""

import asyncio
import logging
import queue
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from research.config import TG_API_ID, TG_API_HASH, TG_SESSION_FILE, TG_CHANNEL

log = logging.getLogger(__name__)

# Solana address: base58, 32–44 chars
_SOL_RE = re.compile(r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b')


@dataclass
class TGAlert:
    token_address: str
    chain: str
    alert_time: datetime
    raw_text: str


class TGListener:
    """
    Monitors pumpdotfunalert and puts TGAlert objects onto a queue.
    Dedup: same address ignored for 5 minutes after first parse.
    Runs in its own thread with its own asyncio event loop.
    """

    def __init__(self, out_queue: queue.Queue):
        self._q      = out_queue
        self._seen:  dict[str, float] = {}   # address → last_seen epoch
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="research-tg"
        )
        self._thread.start()
        log.info("TG listener thread started — channel: %s", TG_CHANNEL)

    def _run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._monitor())
        except Exception as e:
            log.error("TG listener crashed: %s", e, exc_info=True)

    def _is_fresh(self, address: str, now: float) -> bool:
        last = self._seen.get(address, 0)
        if now - last < 300:   # 5-minute dedup
            return False
        self._seen[address] = now
        return True

    def _extract_addresses(self, text: str) -> list[str]:
        now = __import__("time").time()
        found = []
        for addr in _SOL_RE.findall(text):
            if len(addr) >= 32 and self._is_fresh(addr, now):
                found.append(addr)
        return found

    async def _monitor(self):
        try:
            from telethon import TelegramClient, events
        except ImportError:
            log.error("telethon not installed — run: pip install telethon")
            return

        async with TelegramClient(TG_SESSION_FILE, TG_API_ID, TG_API_HASH) as client:
            log.info("TG research session connected")

            @client.on(events.NewMessage(chats=[TG_CHANNEL]))
            async def handler(event):
                text = event.raw_text or ""
                # Also check entity URLs (some bots embed address in hyperlink)
                extra = []
                if event.message and event.message.entities:
                    for ent in event.message.entities:
                        url = getattr(ent, "url", None)
                        if url:
                            extra.append(url)
                combined = text + " " + " ".join(extra)

                for addr in self._extract_addresses(combined):
                    alert = TGAlert(
                        token_address=addr,
                        chain="solana",
                        alert_time=datetime.now(timezone.utc),
                        raw_text=combined[:500],
                    )
                    self._q.put(alert)
                    log.info("TG alert queued: %s", addr[:12])

            await client.run_until_disconnected()
