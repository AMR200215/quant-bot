"""
Telegram listener for pumpdotfunalert channel.

Two listener implementations:

1. TGListener (original)
   Independent Telethon session — separate session file from the trading bot's
   memecoin/data/tg_session.  Requires interactive setup once via:
       python -m research.setup

2. FileQueueListener (preferred when scanner is co-located)
   Tails research/data/signal_queue.jsonl written by scanner._on_telegram_signal().
   No Telegram session or OTP required.  Zero risk of session conflicts.

research/main.py chooses FileQueueListener automatically when the TG session
file is absent or empty.
"""

import asyncio
import json
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from research.config import TG_API_ID, TG_API_HASH, TG_SESSION_FILE, TG_CHANNEL

log = logging.getLogger(__name__)

# Solana address: base58, 32–44 chars
_SOL_RE = re.compile(r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b')

_SIGNAL_QUEUE_PATH = Path(__file__).parent / "data" / "signal_queue.jsonl"


@dataclass
class TGAlert:
    token_address: str
    chain: str
    alert_time: datetime
    raw_text: str


# ---------------------------------------------------------------------------
# FileQueueListener — reads alerts written by scanner (no TG session needed)
# ---------------------------------------------------------------------------

class FileQueueListener:
    """
    Tails research/data/signal_queue.jsonl written by scanner._on_telegram_signal().

    On start: seeks to end of file (skips old alerts already in Supabase).
    On restart: seeks to end again (Supabase dedup prevents duplicates anyway).
    Polls every 0.5s — latency is negligible vs DexScreener's 30-90s indexing lag.
    """

    def __init__(self, out_queue: queue.Queue, queue_path: Path = _SIGNAL_QUEUE_PATH):
        self._q          = out_queue
        self._path       = queue_path
        self._seen:      dict[str, float] = {}   # address → last_seen epoch (5-min dedup)
        self._thread:    Optional[threading.Thread] = None
        self._last_pos:  int = 0

    def start(self):
        # Seek to end on first start — don't replay historical alerts
        if self._path.exists():
            self._last_pos = self._path.stat().st_size
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="research-file-queue"
        )
        self._thread.start()
        log.info(
            "FileQueueListener started — tailing %s (offset=%d)",
            self._path, self._last_pos,
        )

    @property
    def _thread_ref(self):
        return self._thread

    def _is_fresh(self, address: str, now: float) -> bool:
        last = self._seen.get(address, 0)
        if now - last < 300:
            return False
        self._seen[address] = now
        return True

    def _run(self):
        while True:
            try:
                if not self._path.exists():
                    time.sleep(1)
                    continue
                with open(self._path) as f:
                    f.seek(self._last_pos)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            e    = json.loads(line)
                            addr = e.get("token_address", "")
                            if not addr or len(addr) < 32:
                                continue
                            chain = e.get("chain", "solana")
                            now   = time.time()
                            if not self._is_fresh(addr, now):
                                continue
                            raw_ts = e.get("alert_time", "")
                            try:
                                alert_time = datetime.fromisoformat(
                                    raw_ts.replace("Z", "+00:00")
                                )
                            except Exception:
                                alert_time = datetime.now(timezone.utc)
                            alert = TGAlert(
                                token_address=addr,
                                chain=chain,
                                alert_time=alert_time,
                                raw_text=e.get("raw_text", ""),
                            )
                            try:
                                self._q.put_nowait(alert)
                            except queue.Full:
                                pass
                            log.info("FileQueue alert queued: %s", addr[:12])
                        except Exception:
                            pass
                    self._last_pos = f.tell()
            except Exception as _e:
                log.debug("FileQueueListener read error: %s", _e)
            time.sleep(0.5)


# ---------------------------------------------------------------------------
# TGListener — original Telethon-based listener (requires session file)
# ---------------------------------------------------------------------------

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
        """Restart loop — reconnects on disconnect or crash."""
        while True:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._monitor())
                log.warning("TG listener disconnected cleanly — reconnecting in 5s")
            except Exception as e:
                log.error("TG listener crashed: %s — reconnecting in 5s", e, exc_info=True)
            finally:
                try:
                    loop.close()
                except Exception:
                    pass
            time.sleep(5)

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
