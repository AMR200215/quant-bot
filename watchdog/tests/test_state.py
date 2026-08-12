"""watchdog/tests/test_state.py — W19 fault-injection #23, 24, 30."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from watchdog import state as wd_state


class TestStateFaultInjection(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "state.db"

    def tearDown(self):
        self._tmp.cleanup()

    def test_23_incident_state_survives_process_restart(self):
        """A fresh connect() (simulating a new process) must see the same
        incident row a prior process wrote -- no in-memory-only state."""
        conn1 = wd_state.connect(self.db_path)
        wd_state.upsert_incident(conn1, "fp1", "check.x", "subj", "CRITICAL", "FIRING",
                                  touch_notified=True, ts=100.0)
        conn1.close()

        conn2 = wd_state.connect(self.db_path)  # simulates a brand new process
        inc = wd_state.get_incident(conn2, "fp1")
        conn2.close()
        self.assertIsNotNone(inc)
        self.assertEqual(inc["state"], "FIRING")
        self.assertEqual(inc["last_notified"], 100.0)

    def test_24_singleton_lock_blocks_concurrent_instance(self):
        lock_path = Path(self._tmp.name) / "runner.lock"
        lock1 = wd_state.SingletonLock(lock_path)
        lock2 = wd_state.SingletonLock(lock_path)
        self.assertTrue(lock1.acquire())
        self.assertFalse(lock2.acquire(), "a second instance must not acquire the same lock")
        lock1.release()
        self.assertTrue(lock2.acquire(), "lock must be acquirable again after release")
        lock2.release()

    def test_24b_lock_release_is_idempotent_safe(self):
        lock_path = Path(self._tmp.name) / "runner.lock"
        lock = wd_state.SingletonLock(lock_path)
        self.assertTrue(lock.acquire())
        lock.release()
        lock.release()  # must not raise

    def test_30_missing_db_file_is_created_not_silently_all_green(self):
        """connect() against a path that doesn't exist yet must create a
        working, schema-initialized DB -- not silently no-op."""
        self.assertFalse(self.db_path.exists())
        conn = wd_state.connect(self.db_path)
        self.assertTrue(self.db_path.exists())
        # Schema must actually be usable, not just the file created.
        wd_state.record_run_start(conn, "r1", "fast", checks_due=1)
        conn.close()

    def test_30b_corrupt_db_file_fails_loud_not_silently_green(self):
        """A corrupt/non-sqlite file at the db path must raise, not be
        silently treated as an empty healthy database."""
        self.db_path.write_bytes(b"this is not a sqlite database file, just garbage bytes")
        with self.assertRaises(sqlite3.DatabaseError):
            conn = wd_state.connect(self.db_path)
            # init_db's executescript is what actually touches the corrupt
            # file structure and will raise.
            conn.execute("SELECT 1").fetchone()

    def test_job_receipt_trigger_type_is_validated(self):
        conn = wd_state.connect(self.db_path)
        with self.assertRaises(AssertionError):
            wd_state.record_job_receipt(conn, "job1", "not_a_valid_trigger",
                                         started_at=1.0, finished_at=2.0, exit_code=0)
        conn.close()

    def test_manual_receipt_never_returned_when_filtering_for_scheduler(self):
        conn = wd_state.connect(self.db_path)
        wd_state.record_job_receipt(conn, "job1", "manual", started_at=1.0,
                                     finished_at=2.0, exit_code=0)
        latest_scheduler = wd_state.get_latest_job_receipt(conn, "job1", trigger_type="scheduler")
        self.assertIsNone(latest_scheduler, "a manual receipt must never satisfy a scheduler-only query")
        latest_any = wd_state.get_latest_job_receipt(conn, "job1")
        self.assertIsNotNone(latest_any)
        conn.close()


if __name__ == "__main__":
    unittest.main()
