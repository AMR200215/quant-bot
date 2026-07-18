"""
Research pipeline entry point.

Starts three threads:
  1. Listener     — puts TGAlert objects onto the queue (see below)
  2. Tracker      — snapshots tokens, writes to Supabase
  3. OutcomePoller — polls prices at category-specific intervals
  4. PeakTracker  — tick-level peak via PP WebSocket

Listener selection (automatic):
  FileQueueListener  — tails research/data/signal_queue.jsonl written by scanner.py.
                       Used when TG session file is absent or empty.
                       No Telegram OTP required.  Preferred when scanner is co-located.
  TGListener         — independent Telethon session (requires prior python -m research.setup).
                       Used only when a valid session file exists.

Run:
    python -m research.main
"""

import logging
import os
import queue
import signal
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

_MIN_SESSION_BYTES = 8192   # valid Telethon SQLite session is at least this size


def _session_valid(session_path: str) -> bool:
    p = Path(session_path + ".session")
    return p.exists() and p.stat().st_size >= _MIN_SESSION_BYTES


def main():
    from research.config import SUPABASE_URL, SUPABASE_KEY, TG_API_ID, TG_SESSION_FILE
    from research.tg_listener   import TGListener, FileQueueListener
    from research.tracker        import Tracker
    from research.outcome_poller import OutcomePoller
    from research.peak_tracker   import PeakTracker

    # Config checks
    if not SUPABASE_URL or not SUPABASE_KEY:
        log.error("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        sys.exit(1)
    if not TG_API_ID:
        log.error("TELEGRAM_API_ID must be set in .env")
        sys.exit(1)

    alert_queue  = queue.Queue(maxsize=500)
    poller       = OutcomePoller()
    peak_tracker = PeakTracker()
    tracker      = Tracker(
        in_queue=alert_queue,
        poll_schedule_cb=poller.schedule_token,
        peak_schedule_cb=peak_tracker.schedule_token,
    )

    # Choose listener: FileQueueListener when no valid TG session exists
    if _session_valid(TG_SESSION_FILE):
        listener = TGListener(out_queue=alert_queue)
        log.info("Listener: TGListener (Telethon session found)")
    else:
        queue_path = Path(__file__).parent / "data" / "signal_queue.jsonl"
        listener = FileQueueListener(out_queue=alert_queue, queue_path=queue_path)
        log.warning(
            "Listener: FileQueueListener (TG session absent/empty at %s.session) "
            "— tailing %s",
            TG_SESSION_FILE, queue_path,
        )

    poller.start()
    peak_tracker.start()
    tracker.start()
    listener.start()

    log.info("Research pipeline running — Ctrl+C to stop")

    # Graceful shutdown on SIGTERM (systemd stop)
    def _shutdown(sig, frame):
        log.info("Shutdown signal received")
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while True:
            time.sleep(30)
            qsize = alert_queue.qsize()
            if qsize > 50:
                log.warning("Alert queue backing up: %d items", qsize)
            # Watchdog: if listener thread dies, exit so systemd restarts
            t = getattr(listener, "_thread", None) or getattr(listener, "_thread_ref", None)
            if t and not t.is_alive():
                log.error("Listener thread died unexpectedly — exiting for systemd restart")
                sys.exit(1)
    except KeyboardInterrupt:
        log.info("Stopped by user")


if __name__ == "__main__":
    main()
