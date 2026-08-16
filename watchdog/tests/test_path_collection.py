"""watchdog/tests/test_path_collection.py — V8-FD Phase 1.5 (P15-7)
fault injection."""

import json
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from watchdog.checks import STATUS_OK, STATUS_UNKNOWN, STATUS_WARN
from watchdog.checks.path_collection import check_path_collection

HOUR = 3600


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class TestPathCollectionFaultInjection(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "path_collection_daily.json"

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, now, **overrides):
        base = {
            "day": "2026-08-15",
            "generated_at": _iso(now - 60),
            "tokens_scheduled": 128,
            "path_files": 27,
            "ticks": 41064,
            "pp_messages": 61390,
            "pp_daily_msg_budget": 50000,
            "budget_exceeded": True,
            "budget_dropped_tokens": 340,
            "path_yield_pct": 21.1,
            "sub_peak": 6,
        }
        base.update(overrides)
        self.path.write_text(json.dumps(base))

    def test_missing_status_file_is_unknown_not_ok(self):
        results = check_path_collection(status_path=self.path)
        self.assertEqual(results[0].status, STATUS_UNKNOWN)

    def test_healthy_day_is_ok(self):
        now = 1_000_000.0
        self._write(now, budget_exceeded=False, path_yield_pct=85.0, budget_dropped_tokens=0)
        results = check_path_collection(now_ts=now, status_path=self.path)
        self.assertEqual(results[0].status, STATUS_OK)

    def test_budget_exceeded_flags_warn_not_critical(self):
        """Real production case (2026-08-15): budget exceeded, low yield --
        must be WARN (known, cost-bounded constraint), never CRITICAL."""
        now = 1_000_000.0
        self._write(now)
        results = check_path_collection(now_ts=now, status_path=self.path)
        self.assertEqual(results[0].status, STATUS_WARN)
        self.assertIn("budget", results[0].reason.lower())

    def test_low_yield_without_budget_exceeded_still_flagged(self):
        now = 1_000_000.0
        self._write(now, budget_exceeded=False, path_yield_pct=10.0)
        results = check_path_collection(now_ts=now, status_path=self.path)
        self.assertEqual(results[0].status, STATUS_WARN)
        self.assertIn("yield", results[0].reason.lower())

    def test_stale_status_file_is_unknown(self):
        now = 1_000_000.0
        self._write(now - 40 * HOUR, generated_at=_iso(now - 40 * HOUR))
        results = check_path_collection(now_ts=now, status_path=self.path, stale_threshold_s=30 * HOUR)
        self.assertEqual(results[0].status, STATUS_UNKNOWN)

    def test_corrupt_status_file_is_unknown(self):
        self.path.write_text("not json {{{")
        results = check_path_collection(status_path=self.path)
        self.assertEqual(results[0].status, STATUS_UNKNOWN)

    def test_severity_capped_by_ceiling(self):
        now = 1_000_000.0
        self._write(now)
        results = check_path_collection(now_ts=now, status_path=self.path, severity_ceiling="WARN")
        self.assertEqual(results[0].status, "WARN")

    def test_evidence_includes_budget_dropped_count(self):
        now = 1_000_000.0
        self._write(now, budget_dropped_tokens=340)
        results = check_path_collection(now_ts=now, status_path=self.path)
        self.assertEqual(results[0].evidence.get("budget_dropped_tokens"), 340)


if __name__ == "__main__":
    unittest.main()
