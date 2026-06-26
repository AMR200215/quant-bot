"""
Research pipeline entry point.

Starts three threads:
  1. TGListener   — watches pumpdotfunalert, puts tokens on queue
  2. Tracker      — snapshots tokens, writes to Supabase
  3. OutcomePoller — polls prices at category-specific intervals

Run:
    python -m research.main

First-time setup (TG session auth):
    python -m research.setup
"""

import logging
import queue
import signal
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    from research.config import SUPABASE_URL, SUPABASE_KEY, TG_API_ID, TG_SESSION_FILE
    from research.tg_listener   import TGListener
    from research.tracker        import Tracker
    from research.outcome_poller import OutcomePoller
    import os

    # Config checks
    if not SUPABASE_URL or not SUPABASE_KEY:
        log.error("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        sys.exit(1)
    if not TG_API_ID:
        log.error("TELEGRAM_API_ID must be set in .env")
        sys.exit(1)
    if not os.path.exists(TG_SESSION_FILE + ".session"):
        log.error(
            "TG research session not found at %s.session\n"
            "Run once:  python -m research.setup",
            TG_SESSION_FILE
        )
        sys.exit(1)

    alert_queue = queue.Queue(maxsize=500)
    poller      = OutcomePoller()
    tracker     = Tracker(
        in_queue=alert_queue,
        poll_schedule_cb=poller.schedule_token,
    )
    listener    = TGListener(out_queue=alert_queue)

    poller.start()
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
            # Watchdog: TG listener has its own restart loop, but if the thread
            # itself dies (unrecoverable), exit so systemd restarts the service.
            if not listener._thread.is_alive():
                log.error("TG listener thread died unexpectedly — exiting for systemd restart")
                sys.exit(1)
    except KeyboardInterrupt:
        log.info("Stopped by user")


if __name__ == "__main__":
    main()
