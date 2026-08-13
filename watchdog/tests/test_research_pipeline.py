"""watchdog/tests/test_research_pipeline.py — W6D fault injection."""

import datetime
import json
import tempfile
import unittest
from pathlib import Path

from watchdog.checks import STATUS_CRITICAL, STATUS_OK, STATUS_UNKNOWN, STATUS_WARN
from watchdog.checks.research_pipeline import check_queue_lag, check_spool_growth


class TestQueueLagFaultInjection(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)
        self.qpath = self.tmp_path / "signal_queue.jsonl"
        self.opath = self.tmp_path / ".queue_offset"

    def tearDown(self):
        self._tmp.cleanup()

    def test_caught_up_is_ok(self):
        self.qpath.write_text("x" * 1000)
        self.opath.write_text("1000")
        results = check_queue_lag(queue_path=self.qpath, offset_path=self.opath)
        self.assertEqual(results[0].status, STATUS_OK)

    def test_upstream_flowing_downstream_stalled_is_warn(self):
        """The exact W6D shape: queue file has grown well past the
        offset, meaning new signals arrived but the consumer hasn't
        processed them."""
        self.qpath.write_text("x" * 200_000)
        self.opath.write_text("1000")
        results = check_queue_lag(queue_path=self.qpath, offset_path=self.opath)
        self.assertEqual(results[0].status, STATUS_WARN)

    def test_small_gap_within_threshold_is_ok_not_a_false_alarm(self):
        self.qpath.write_text("x" * 5000)
        self.opath.write_text("4980")
        results = check_queue_lag(queue_path=self.qpath, offset_path=self.opath)
        self.assertEqual(results[0].status, STATUS_OK)

    def test_missing_queue_file_is_unknown(self):
        self.opath.write_text("0")
        results = check_queue_lag(queue_path=self.qpath, offset_path=self.opath)
        self.assertEqual(results[0].status, STATUS_UNKNOWN)

    def test_missing_offset_file_is_unknown(self):
        self.qpath.write_text("x" * 100)
        results = check_queue_lag(queue_path=self.qpath, offset_path=self.opath)
        self.assertEqual(results[0].status, STATUS_UNKNOWN)

    def test_offset_ahead_of_file_size_is_unknown_not_a_guess(self):
        """File rotation/truncation, not a stall -- must not be misreported
        either way."""
        self.qpath.write_text("x" * 100)
        self.opath.write_text("99999")
        results = check_queue_lag(queue_path=self.qpath, offset_path=self.opath)
        self.assertEqual(results[0].status, STATUS_UNKNOWN)


class TestSpoolGrowthFaultInjection(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "failed_inserts.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def _iso(self, now, offset_s):
        return datetime.datetime.fromtimestamp(now - offset_s, tz=datetime.timezone.utc).isoformat()

    def test_no_file_is_ok(self):
        results = check_spool_growth(failed_inserts_path=self.path)
        self.assertEqual(results[0].status, STATUS_OK)

    def test_below_min_sample_floor_no_false_alarm(self):
        """Per spec: never alert on n=1-2. The real production file has a
        historical bug's worth of old lines (dormant for 5+ days) -- a
        naive 'any lines exist' check would alarm on stale history."""
        now = 1_000_000.0
        lines = [json.dumps({"ts": self._iso(now, 60), "error": "boom"}) for _ in range(2)]
        self.path.write_text("\n".join(lines))
        results = check_spool_growth(now_ts=now, failed_inserts_path=self.path)
        self.assertEqual(results[0].status, STATUS_OK)

    def test_old_historical_lines_do_not_count_toward_recent_growth(self):
        """The real dormant-since-2026-08-08 bug: total line count on disk
        must not itself trigger an alert if none of it is recent."""
        now = 1_000_000.0
        old_lines = [json.dumps({"ts": self._iso(now, 30 * 24 * 3600), "error": "old bug"})
                     for _ in range(100)]
        self.path.write_text("\n".join(old_lines))
        results = check_spool_growth(now_ts=now, failed_inserts_path=self.path, lookback_seconds=2 * 3600)
        self.assertEqual(results[0].status, STATUS_OK)
        self.assertEqual(results[0].evidence["recent_count"], 0)

    def test_sustained_recent_growth_is_warn(self):
        now = 1_000_000.0
        lines = [json.dumps({"ts": self._iso(now, 60), "error": f"boom {i}"}) for i in range(10)]
        self.path.write_text("\n".join(lines))
        results = check_spool_growth(now_ts=now, failed_inserts_path=self.path)
        self.assertEqual(results[0].status, STATUS_WARN)

    def test_malformed_lines_are_skipped_not_fatal(self):
        now = 1_000_000.0
        lines = ["not json"] + [json.dumps({"ts": self._iso(now, 60), "error": "boom"}) for _ in range(6)]
        self.path.write_text("\n".join(lines))
        results = check_spool_growth(now_ts=now, failed_inserts_path=self.path)
        self.assertEqual(results[0].status, STATUS_WARN)
        self.assertEqual(results[0].evidence["recent_count"], 6)


if __name__ == "__main__":
    unittest.main()
