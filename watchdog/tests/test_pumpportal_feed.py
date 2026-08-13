"""watchdog/tests/test_pumpportal_feed.py — W6A fault injection.

Includes a regression guard for a real thing confirmed against production
logs before this check was written: PumpPortal reconnects every ~45-60s
by design (pre-warmed rotation WS). A naive "too many reconnects"
heuristic would false-positive against this permanently -- this check
must never evaluate reconnect frequency at all, only error/deadman
evidence."""

import unittest

from watchdog.checks import STATUS_CRITICAL, STATUS_OK, STATUS_UNKNOWN, STATUS_WARN
from watchdog.checks.pumpportal_feed import check_pumpportal_feed


class TestPumpPortalFeedFaultInjection(unittest.TestCase):

    def test_deadman_fire_is_critical(self):
        now = 1_000_000.0
        lines = [
            "INFO memecoin.pumpportal_monitor: PumpPortal WebSocket connected (conn_start=x)",
            "WARNING memecoin.pumpportal_monitor: PumpPortal tick deadman: abc123 open position "
            "silent >5min while subscribed — likely on stale fallback price",
        ]
        results = check_pumpportal_feed(now_ts=now, journal_lines=lines)
        self.assertEqual(results[0].status, STATUS_CRITICAL)
        self.assertIn("PRIMARY_FEED_DEGRADED", results[0].reason)

    def test_suppressed_weekly_deadman_note_is_not_critical(self):
        now = 1_000_000.0
        lines = [
            "INFO memecoin.pumpportal_monitor: PumpPortal tick deadman: 3 position(s) silent "
            ">5min (alert suppressed, LIVE_TRADING=false)",
        ]
        results = check_pumpportal_feed(now_ts=now, journal_lines=lines)
        self.assertNotEqual(results[0].status, STATUS_CRITICAL)

    def test_frequent_reconnects_with_no_errors_is_ok_not_flagged(self):
        """Regression: this is real, confirmed, by-design behavior (rotation
        WS) -- must never be treated as anomalous no matter how frequent."""
        now = 1_000_000.0
        lines = [f"INFO memecoin.pumpportal_monitor: PumpPortal WebSocket connected (conn_start={i})"
                 for i in range(30)]  # 30 reconnects in the lookback window
        results = check_pumpportal_feed(now_ts=now, journal_lines=lines)
        self.assertEqual(results[0].status, STATUS_OK)

    def test_error_more_recent_than_last_connect_is_warn(self):
        now = 1_000_000.0
        lines = [
            "INFO memecoin.pumpportal_monitor: PumpPortal WebSocket connected (conn_start=x)",
            "WARNING memecoin.pumpportal_monitor: PumpPortal WS error (attempt 3): timeout — retry in 4.0s",
        ]
        results = check_pumpportal_feed(now_ts=now, journal_lines=lines)
        self.assertEqual(results[0].status, STATUS_WARN)

    def test_connect_after_error_is_ok(self):
        now = 1_000_000.0
        lines = [
            "WARNING memecoin.pumpportal_monitor: PumpPortal WS error (attempt 3): timeout — retry in 4.0s",
            "INFO memecoin.pumpportal_monitor: PumpPortal WebSocket connected (conn_start=x)",
        ]
        results = check_pumpportal_feed(now_ts=now, journal_lines=lines)
        self.assertEqual(results[0].status, STATUS_OK)

    def test_no_journal_evidence_is_unknown(self):
        results = check_pumpportal_feed(now_ts=1_000_000.0, journal_lines=None)
        self.assertEqual(results[0].status, STATUS_UNKNOWN)

    def test_no_matching_lines_at_all_is_unknown(self):
        results = check_pumpportal_feed(now_ts=1_000_000.0, journal_lines=["unrelated log line"])
        self.assertEqual(results[0].status, STATUS_UNKNOWN)


if __name__ == "__main__":
    unittest.main()
