"""watchdog/tests/test_notifier.py — W19 fault-injection #20-22, plus the
UNKNOWN-must-not-mask-a-FIRING-incident regression found live during
Phase 1 development (see notifier.py's STATUS_UNKNOWN branch)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from watchdog import notifier as wd_notifier
from watchdog import state as wd_state
from watchdog.checks import CheckResult, STATUS_CRITICAL, STATUS_OK, STATUS_UNKNOWN, STATUS_WARN

HOUR = 3600


class TestNotifierFaultInjection(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.conn = wd_state.connect(Path(self._tmp.name) / "state.db")

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()

    def _critical(self, reason="dead"):
        return CheckResult(check_id="cron_execution.k5_nightly", status=STATUS_CRITICAL,
                            reason=reason, subject="k5_nightly", severity="CRITICAL")

    def test_20_twelve_consecutive_failures_send_one_initial_plus_reminders_not_twelve(self):
        with patch.object(wd_notifier, "send_telegram", return_value=True) as mock_send:
            t0 = 1_000_000.0
            for i in range(12):
                wd_notifier.process_results(self.conn, [self._critical()], now_ts=t0 + i, send=True)
            # First call fires immediately; none of the next 11 (all within
            # the same second, nowhere near the 2h reminder interval)
            # should trigger a second send.
            self.assertEqual(mock_send.call_count, 1,
                              f"expected exactly 1 Telegram call for 12 rapid identical "
                              f"failures, got {mock_send.call_count}")

            # Now simulate time passing past the first reminder interval.
            wd_notifier.process_results(
                self.conn, [self._critical()],
                now_ts=t0 + wd_notifier.CRITICAL_REMINDER_FIRST_SEC + 1, send=True,
            )
            self.assertEqual(mock_send.call_count, 2, "should get exactly one reminder after 2h")

    def test_21_recovery_sends_exactly_one_message(self):
        with patch.object(wd_notifier, "send_telegram", return_value=True) as mock_send:
            wd_notifier.process_results(self.conn, [self._critical()], now_ts=0, send=True)
            self.assertEqual(mock_send.call_count, 1)

            ok = CheckResult(check_id="cron_execution.k5_nightly", status=STATUS_OK,
                              reason="recovered", subject="k5_nightly")
            wd_notifier.process_results(self.conn, [ok], now_ts=1, send=True)
            self.assertEqual(mock_send.call_count, 2)

            # Repeating OK must not send a second recovery message -- there's
            # no active incident left to recover from.
            wd_notifier.process_results(self.conn, [ok], now_ts=2, send=True)
            self.assertEqual(mock_send.call_count, 2)

            inc = wd_state.get_incident(self.conn, "cron_execution.k5_nightly:k5_nightly")
            self.assertEqual(inc["state"], "RECOVERED")

    def test_22_notifier_send_failure_does_not_erase_the_incident(self):
        with patch.object(wd_notifier, "send_telegram", return_value=False):
            wd_notifier.process_results(self.conn, [self._critical()], now_ts=0, send=True)
        inc = wd_state.get_incident(self.conn, "cron_execution.k5_nightly:k5_nightly")
        self.assertIsNotNone(inc, "incident must persist even if the Telegram send itself failed")
        self.assertEqual(inc["state"], "FIRING")

    def test_unknown_does_not_downgrade_an_already_firing_incident(self):
        """Regression: UNKNOWN evidence (e.g. journalctl briefly
        unavailable) must never silently un-fire a real incident."""
        with patch.object(wd_notifier, "send_telegram", return_value=True) as mock_send:
            wd_notifier.process_results(self.conn, [self._critical()], now_ts=0, send=True)
            inc_before = wd_state.get_incident(self.conn, "cron_execution.k5_nightly:k5_nightly")
            self.assertEqual(inc_before["state"], "FIRING")

            unknown = CheckResult(check_id="cron_execution.k5_nightly", status=STATUS_UNKNOWN,
                                   reason="evidence unavailable", subject="k5_nightly")
            wd_notifier.process_results(self.conn, [unknown], now_ts=1, send=True)

            inc_after = wd_state.get_incident(self.conn, "cron_execution.k5_nightly:k5_nightly")
            self.assertEqual(inc_after["state"], "FIRING", "UNKNOWN must not downgrade FIRING")
            self.assertEqual(inc_after["consecutive_failures"], inc_before["consecutive_failures"],
                              "UNKNOWN must not extend the failure streak either")
            # And it must not have sent a second (spurious) alert.
            self.assertEqual(mock_send.call_count, 1)

    def test_unknown_with_no_existing_incident_does_not_accumulate_as_failures(self):
        unk = CheckResult(check_id="cron_static.k5_nightly", status=STATUS_UNKNOWN,
                           reason="evidence unavailable", subject="k5_nightly")
        with patch.object(wd_notifier, "send_telegram", return_value=True) as mock_send:
            for i in range(5):
                wd_notifier.process_results(self.conn, [unk], now_ts=float(i), send=True)
        inc = wd_state.get_incident(self.conn, "cron_static.k5_nightly:k5_nightly")
        self.assertEqual(inc["consecutive_failures"], 0)
        self.assertEqual(mock_send.call_count, 0, "UNKNOWN alone must never page")

    def test_warn_requires_min_consecutive_before_firing(self):
        warn = CheckResult(check_id="cron_execution.epoch_daily", status=STATUS_WARN,
                            reason="no receipt", subject="epoch_daily", severity="WARN")
        with patch.object(wd_notifier, "send_telegram", return_value=True) as mock_send:
            s1 = wd_notifier.process_results(self.conn, [warn], now_ts=0, send=True)
            self.assertEqual(mock_send.call_count, 0, "single WARN must not page immediately")
            s2 = wd_notifier.process_results(self.conn, [warn], now_ts=1, send=True)
            self.assertEqual(mock_send.call_count, 1, "2nd consecutive WARN should fire")


if __name__ == "__main__":
    unittest.main()
