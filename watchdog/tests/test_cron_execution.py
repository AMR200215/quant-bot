"""watchdog/tests/test_cron_execution.py — W19 fault-injection #2-6.

Timezone/DST note (item #6): the VPS runs in UTC and all schedule math
here operates on Unix epoch floats, which are timezone-unambiguous by
construction -- there is no wall-clock DST transition to get wrong in UTC.
This is a documented scope limitation, not an untested claim: if this
watchdog is ever deployed against a non-UTC scheduler, croniter's
timezone-aware datetime mode would need to be wired in instead of raw
epoch floats. test_6 below pins down that the underlying prev/next
arithmetic is at least correct across a real day boundary.
"""

import unittest

from watchdog.checks import STATUS_CRITICAL, STATUS_OK, STATUS_WARN
from watchdog.checks.cron_execution import evaluate_job_liveness

DAY = 86400
HOUR = 3600
MIN = 60


class TestCronExecutionFaultInjection(unittest.TestCase):

    def test_2_manual_receipt_does_not_satisfy_scheduler_liveness(self):
        """The exact K5 incident: a manual test run touched the artifact,
        making it *look* fresh, while the real scheduled job never ran.
        evaluate_job_liveness only ever receives scheduler-trigger receipts
        (the caller filters by trigger_type='scheduler') -- so a manual-only
        history must present here as latest_receipt=None."""
        schedule = "15 0 * * *"
        now_ts = 1786500000.0  # some time well past a day's 00:15 fire
        status, reason, _ = evaluate_job_liveness(
            schedule, grace_minutes=45, now_ts=now_ts, boot_time_ts=None,
            latest_receipt=None,  # caller already filtered out the manual receipt
        )
        self.assertEqual(status, STATUS_CRITICAL)
        self.assertIn("no scheduler execution receipt", reason)

    def test_3_stale_receipt_past_grace_alerts(self):
        schedule = "0 * * * *"  # hourly
        now_ts = 10 * HOUR
        stale_receipt = {"started_at": 5 * HOUR, "exit_code": 0}  # 5 cycles ago
        status, reason, evidence = evaluate_job_liveness(
            schedule, grace_minutes=10, now_ts=now_ts, boot_time_ts=None,
            latest_receipt=stale_receipt,
        )
        self.assertEqual(status, STATUS_CRITICAL)
        self.assertIn("predates", reason)
        self.assertLess(stale_receipt["started_at"], evidence["expected_prev_fire"])

    def test_4_not_yet_due_no_false_alarm(self):
        schedule = "0 * * * *"  # hourly, next due at top of hour
        now_ts = 30 * MIN  # 30 min past the hour -- previous fire was at t=0
        status, reason, _ = evaluate_job_liveness(
            schedule, grace_minutes=45, now_ts=now_ts, boot_time_ts=None,
            latest_receipt=None,
        )
        self.assertEqual(status, STATUS_OK, reason)
        self.assertIn("not yet overdue", reason)

    def test_5_boot_grace_no_immediate_false_alarm(self):
        """Machine rebooted after the most recently expected fire -- the
        job structurally could not have run for that cycle. Must not
        alarm on it; must wait for the next fire due *after* boot."""
        schedule = "0 * * * *"  # hourly
        boot_time_ts = 9 * HOUR + 50 * MIN  # rebooted just before the 10:00 fire
        now_ts = 9 * HOUR + 55 * MIN        # only 5 min after boot, next fire (10:00) not due yet
        status, reason, evidence = evaluate_job_liveness(
            schedule, grace_minutes=10, now_ts=now_ts, boot_time_ts=boot_time_ts,
            latest_receipt=None,
        )
        self.assertEqual(status, STATUS_OK, reason)
        self.assertIn("no scheduled fire has been due since boot", reason)

    def test_5b_boot_grace_expires_once_post_boot_fire_is_overdue(self):
        """After boot, once a fresh cycle's own deadline passes with still
        no receipt, that IS a real problem -- boot grace only excuses the
        one fire that couldn't have happened, not an indefinite pass."""
        schedule = "0 * * * *"
        boot_time_ts = 9 * HOUR + 50 * MIN
        now_ts = 10 * HOUR + 20 * MIN  # 20 min after the first post-boot fire (10:00), grace=10min
        status, reason, _ = evaluate_job_liveness(
            schedule, grace_minutes=10, now_ts=now_ts, boot_time_ts=boot_time_ts,
            latest_receipt=None,
        )
        self.assertEqual(status, STATUS_CRITICAL, reason)

    def test_6_prev_next_arithmetic_correct_across_day_boundary(self):
        schedule = "15 0 * * *"  # daily at 00:15
        # now = 00:10 the next day -- previous fire should be yesterday's 00:15,
        # not today's (which hasn't happened yet).
        now_ts = DAY + 10 * MIN
        status, reason, evidence = evaluate_job_liveness(
            schedule, grace_minutes=5, now_ts=now_ts, boot_time_ts=None,
            latest_receipt={"started_at": 15 * MIN, "exit_code": 0},  # yesterday's run
        )
        self.assertAlmostEqual(evidence["expected_prev_fire"], 15 * MIN, delta=1)
        self.assertEqual(status, STATUS_OK, reason)

    def test_nonzero_exit_code_is_warn_not_ok(self):
        schedule = "0 * * * *"
        now_ts = 2 * HOUR
        receipt = {"started_at": 1 * HOUR + 1, "exit_code": 1}
        status, reason, _ = evaluate_job_liveness(
            schedule, grace_minutes=5, now_ts=now_ts, boot_time_ts=None,
            latest_receipt=receipt,
        )
        self.assertEqual(status, STATUS_WARN)
        self.assertIn("exited 1", reason)


if __name__ == "__main__":
    unittest.main()
