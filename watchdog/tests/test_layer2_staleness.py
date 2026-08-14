"""watchdog/tests/test_layer2_staleness.py — W17 fault injection."""

import json
import tempfile
import unittest
from pathlib import Path

from watchdog.checks import STATUS_CRITICAL, STATUS_OK, STATUS_UNKNOWN
from watchdog.checks.layer2_staleness import check_layer2_staleness

HOUR = 3600


class TestLayer2StalenessFaultInjection(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "layer2_heartbeat.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_heartbeat_is_unknown_not_ok(self):
        results = check_layer2_staleness(heartbeat_path=self.path)
        self.assertEqual(results[0].status, STATUS_UNKNOWN)

    def test_recent_heartbeat_is_ok(self):
        now = 1_000_000.0
        self.path.write_text(json.dumps({
            "last_audit_id": "audit-x", "last_audit_completed_at": now - HOUR,
            "findings_count": 0, "critical_count": 0,
        }))
        results = check_layer2_staleness(now_ts=now, heartbeat_path=self.path)
        self.assertEqual(results[0].status, STATUS_OK)

    def test_stale_heartbeat_past_threshold_is_flagged(self):
        now = 1_000_000.0
        self.path.write_text(json.dumps({
            "last_audit_id": "audit-x", "last_audit_completed_at": now - 40 * HOUR,
            "findings_count": 0, "critical_count": 0,
        }))
        results = check_layer2_staleness(now_ts=now, heartbeat_path=self.path,
                                          stale_threshold_s=30 * HOUR)
        self.assertNotEqual(results[0].status, STATUS_OK)
        self.assertIn("stopped running", results[0].reason)

    def test_corrupt_heartbeat_is_unknown(self):
        self.path.write_text("not json {{{")
        results = check_layer2_staleness(heartbeat_path=self.path)
        self.assertEqual(results[0].status, STATUS_UNKNOWN)

    def test_severity_capped_by_ceiling(self):
        now = 1_000_000.0
        self.path.write_text(json.dumps({
            "last_audit_id": "audit-x", "last_audit_completed_at": now - 40 * HOUR,
        }))
        results = check_layer2_staleness(now_ts=now, heartbeat_path=self.path,
                                          stale_threshold_s=30 * HOUR, severity_ceiling="WARN")
        self.assertEqual(results[0].status, "WARN")


if __name__ == "__main__":
    unittest.main()
