"""watchdog/tests/test_exec_wrapper.py — proves the exec_wrapper records a
real scheduler-trigger receipt and that a receipt-write failure never
masks the wrapped command's own exit code (W5B's evidence source, and the
"fail loud for monitoring, never block the thing being monitored" rule)."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from watchdog import exec_wrapper, state as wd_state


class TestExecWrapper(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "state.db"

    def tearDown(self):
        self._tmp.cleanup()

    def test_records_scheduler_receipt_with_real_exit_code(self):
        rc = exec_wrapper.main([
            "--job-id", "j1", "--trigger", "scheduler", "--db-path", str(self.db_path),
            "--", sys.executable, "-c", "import sys; sys.exit(0)",
        ])
        self.assertEqual(rc, 0)
        conn = wd_state.connect(self.db_path)
        receipt = wd_state.get_latest_job_receipt(conn, "j1", trigger_type="scheduler")
        conn.close()
        self.assertIsNotNone(receipt)
        self.assertEqual(receipt["exit_code"], 0)
        self.assertEqual(receipt["trigger_type"], "scheduler")

    def test_nonzero_exit_code_propagates_and_is_recorded(self):
        rc = exec_wrapper.main([
            "--job-id", "j1", "--trigger", "scheduler", "--db-path", str(self.db_path),
            "--", sys.executable, "-c", "import sys; sys.exit(7)",
        ])
        self.assertEqual(rc, 7, "wrapper must propagate the wrapped command's real exit code")
        conn = wd_state.connect(self.db_path)
        receipt = wd_state.get_latest_job_receipt(conn, "j1", trigger_type="scheduler")
        conn.close()
        self.assertEqual(receipt["exit_code"], 7)

    def test_manual_trigger_is_recorded_distinctly(self):
        exec_wrapper.main([
            "--job-id", "j1", "--trigger", "manual", "--db-path", str(self.db_path),
            "--", sys.executable, "-c", "pass",
        ])
        conn = wd_state.connect(self.db_path)
        scheduler_receipt = wd_state.get_latest_job_receipt(conn, "j1", trigger_type="scheduler")
        manual_receipt = wd_state.get_latest_job_receipt(conn, "j1", trigger_type="manual")
        conn.close()
        self.assertIsNone(scheduler_receipt)
        self.assertIsNotNone(manual_receipt)

    def test_receipt_write_failure_never_masks_real_exit_code(self):
        with patch.object(wd_state, "record_job_receipt", side_effect=RuntimeError("disk full")):
            rc = exec_wrapper.main([
                "--job-id", "j1", "--trigger", "scheduler", "--db-path", str(self.db_path),
                "--", sys.executable, "-c", "import sys; sys.exit(3)",
            ])
        self.assertEqual(rc, 3, "a receipt-recording failure must never mask the job's real exit code")

    def test_default_trigger_is_scheduler(self):
        """A bare invocation (as cron itself would call it) must default to
        trigger=scheduler -- manual must be explicitly opted into, never
        the silent default, or every manual test run would masquerade as
        proof of a real scheduled execution."""
        exec_wrapper.main([
            "--job-id", "j1", "--db-path", str(self.db_path),
            "--", sys.executable, "-c", "pass",
        ])
        conn = wd_state.connect(self.db_path)
        receipt = wd_state.get_latest_job_receipt(conn, "j1", trigger_type="scheduler")
        conn.close()
        self.assertIsNotNone(receipt)


if __name__ == "__main__":
    unittest.main()
