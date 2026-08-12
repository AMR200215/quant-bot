"""watchdog/tests/test_runner.py — W19 fault-injection #25, #29, plus an
end-to-end smoke test of run_checks() against a real fixture cron.d dir."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from watchdog import runner as wd_runner
from watchdog import state as wd_state
from watchdog.checks import STATUS_CRITICAL, STATUS_UNKNOWN


class TestRunnerFaultInjection(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        (self.tmp_path / "cron.d").mkdir()
        self.conn = wd_state.connect(self.tmp_path / "state.db")

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()

    def test_25_one_crashing_check_does_not_prevent_others_from_running(self):
        registry = {
            "managed_cron_dir": str(self.tmp_path / "cron.d"),
            "jobs": [{"id": "j1", "cron_file": "quantbot-j1", "schedule": "0 * * * *",
                      "grace_minutes": 30, "severity": "CRITICAL", "profile": "fast"}],
        }
        with patch("watchdog.checks.cron_static.check_cron_static", side_effect=RuntimeError("boom")):
            results = wd_runner.run_checks(registry, "fast", self.conn)
        ids = [r.check_id for r in results]
        # The crashing check must surface as UNKNOWN, not silently vanish,
        # and cron_execution (a separate, independent check) must still run.
        self.assertTrue(any(r.status == STATUS_UNKNOWN and "cron_static" in r.check_id for r in results))
        self.assertTrue(any("cron_execution.j1" in cid for cid in ids),
                         "a crash in one check must not prevent an independent check from running")

    def test_end_to_end_missing_cron_file_produces_critical(self):
        registry = {
            "managed_cron_dir": str(self.tmp_path / "cron.d"),
            "jobs": [{"id": "j1", "cron_file": "quantbot-j1", "schedule": "0 * * * *",
                      "grace_minutes": 30, "severity": "CRITICAL", "profile": "fast"}],
        }
        results = wd_runner.run_checks(registry, "fast", self.conn)
        crit = [r for r in results if r.status == STATUS_CRITICAL]
        self.assertTrue(crit, "missing cron file must produce at least one CRITICAL result")

    def test_29_liveness_evidence_lives_in_sqlite_not_log_files(self):
        """Documents/enforces #29 by construction: job receipts and
        incident state are queried from the SQLite state DB, which is
        independent of any log file rotation policy -- a rotated/truncated
        log cannot erase watchdog liveness evidence."""
        wd_state.record_job_receipt(self.conn, "j1", "scheduler", started_at=1.0,
                                     finished_at=2.0, exit_code=0)
        # Simulate "log rotation" by deleting every file under a fake logs
        # dir -- state.db lives elsewhere and must be untouched.
        fake_logs = self.tmp_path / "logs"
        fake_logs.mkdir()
        (fake_logs / "some_cron.log").write_text("rotated away")
        (fake_logs / "some_cron.log").unlink()
        receipt = wd_state.get_latest_job_receipt(self.conn, "j1", trigger_type="scheduler")
        self.assertIsNotNone(receipt, "job receipt must survive independently of log file lifecycle")

    def test_self_test_mode_exercises_full_lifecycle_and_reports_pass(self):
        ok = wd_runner.run_self_test(self.conn)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
