"""watchdog/tests/test_telegram_feed.py — W19 fault-injection #9-10, plus
evidence-unavailable handling."""

import unittest

from watchdog.checks import STATUS_CRITICAL, STATUS_OK, STATUS_UNKNOWN, STATUS_WARN
from watchdog.checks.telegram_feed import check_telegram_feed


class TestTelegramFeedFaultInjection(unittest.TestCase):

    def test_9_auth_required_is_critical(self):
        now = 1_000_000.0
        results = check_telegram_feed(
            now_ts=now,
            journal_lines=["2026-08-12 12:00:00 ERROR TELEGRAM_AUTH_REQUIRED — session not authorised"],
            funnel_events=[{"ts": now - 60, "stage": "telegram_received", "event_id": "e1"}],
        )
        self.assertEqual(results[0].status, STATUS_CRITICAL)
        self.assertIn("AUTH_REQUIRED", results[0].reason)

    def test_thread_dead_is_critical(self):
        now = 1_000_000.0
        results = check_telegram_feed(
            now_ts=now,
            journal_lines=["2026-08-12 ERROR HEALTH: tg-monitor thread is dead"],
            funnel_events=[{"ts": now - 60, "stage": "telegram_received", "event_id": "e1"}],
        )
        self.assertEqual(results[0].status, STATUS_CRITICAL)
        self.assertIn("thread", results[0].reason)

    def test_10_connected_and_legitimately_quiet_is_not_mislabeled_disconnected(self):
        """No error signal, no recent message -- must be WARN (ambiguous),
        never CRITICAL/DISCONNECTED. Silence alone is not proof of death."""
        now = 1_000_000.0
        results = check_telegram_feed(
            now_ts=now,
            journal_lines=["totally normal unrelated log line"],
            funnel_events=[{"ts": now - 5 * 3600, "stage": "telegram_received", "event_id": "e1"}],
            stale_threshold_s=2 * 3600,
        )
        self.assertEqual(results[0].status, STATUS_WARN)
        self.assertNotEqual(results[0].status, STATUS_CRITICAL)
        self.assertIn("ambiguous", results[0].reason)

    def test_recent_message_is_ok(self):
        now = 1_000_000.0
        results = check_telegram_feed(
            now_ts=now, journal_lines=["normal line"],
            funnel_events=[{"ts": now - 30, "stage": "telegram_received", "event_id": "e1"}],
        )
        self.assertEqual(results[0].status, STATUS_OK)

    def test_app_self_reported_stale_corroborates_but_does_not_escalate_to_critical(self):
        now = 1_000_000.0
        results = check_telegram_feed(
            now_ts=now,
            journal_lines=["TG feed connected but no message for >2h — CONNECTED_BUT_STALE"],
            funnel_events=[{"ts": now - 5 * 3600, "stage": "telegram_received", "event_id": "e1"}],
        )
        self.assertEqual(results[0].status, STATUS_WARN)
        self.assertTrue(results[0].evidence.get("app_reported_stale"))

    def test_no_journal_evidence_and_would_be_ok_downgrades_to_unknown(self):
        """Incomplete evidence (journalctl unavailable) must never let a
        funnel-only signal claim a confident OK -- the auth/thread-dead
        check couldn't run this time."""
        now = 1_000_000.0
        results = check_telegram_feed(
            now_ts=now, journal_lines=None, journal_fetch_failed=True,
            funnel_events=[{"ts": now - 30, "stage": "telegram_received", "event_id": "e1"}],
        )
        self.assertEqual(results[0].status, STATUS_UNKNOWN)

    def test_no_funnel_evidence_at_all_is_unknown(self):
        now = 1_000_000.0
        results = check_telegram_feed(now_ts=now, journal_lines=["normal"], funnel_events=[])
        self.assertEqual(results[0].status, STATUS_UNKNOWN)


if __name__ == "__main__":
    unittest.main()
