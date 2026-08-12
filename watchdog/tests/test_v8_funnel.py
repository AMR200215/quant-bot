"""watchdog/tests/test_v8_funnel.py — W19 fault-injection #13-15, plus
the general no-false-green-on-missing-evidence cases."""

import json
import tempfile
import unittest
from pathlib import Path

from watchdog.checks import STATUS_CRITICAL, STATUS_OK, STATUS_UNKNOWN
from watchdog.checks.v8_funnel import check_v8_funnel


class TestV8FunnelFaultInjection(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "v8_funnel.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, events):
        self.path.write_text("\n".join(json.dumps(e) for e in events))

    def test_15_missing_terminal_disposition_beyond_grace_is_detected(self):
        now = 1_000_000.0
        self._write([
            {"ts": now - 500, "stage": "v8_gate_entered", "event_id": "ev1", "mint": "M1"},
            # no v8_gate_rejected or v8_opened for ev1 -- silent disappearance
        ])
        results = check_v8_funnel(now_ts=now, funnel_path=self.path, grace_seconds=120)
        self.assertEqual(results[0].status, STATUS_CRITICAL)
        self.assertIn("v8_gate_entered", results[0].reason)

    def test_13_ten_eligible_ten_explicitly_rejected_is_not_a_silent_disappearance(self):
        """Real production behavior: a low open-rate with explicit reject
        reasons for every candidate is healthy, not an anomaly. This check
        must never flag a 100%-explicitly-rejected batch."""
        now = 1_000_000.0
        events = []
        for i in range(10):
            events.append({"ts": now - 500, "stage": "v8_gate_entered", "event_id": f"ev{i}", "mint": f"M{i}"})
            events.append({"ts": now - 499, "stage": "v8_gate_rejected", "event_id": f"ev{i}", "mint": f"M{i}"})
        self._write(events)
        results = check_v8_funnel(now_ts=now, funnel_path=self.path, grace_seconds=120)
        self.assertEqual(results[0].status, STATUS_OK, results[0].reason)

    def test_14_single_event_within_grace_produces_no_false_alarm(self):
        now = 1_000_000.0
        self._write([
            {"ts": now - 10, "stage": "v8_gate_entered", "event_id": "ev1", "mint": "M1"},
            # only 10s old, well within the 120s grace -- too soon to expect resolution
        ])
        results = check_v8_funnel(now_ts=now, funnel_path=self.path, grace_seconds=120)
        self.assertEqual(results[0].status, STATUS_OK, results[0].reason)

    def test_two_stage_transitions_both_checked_independently(self):
        now = 1_000_000.0
        self._write([
            # ev1: stuck between add_signal_entered and v8_gate_entered/dedup_rejected
            {"ts": now - 500, "stage": "add_signal_entered", "event_id": "ev1", "mint": "M1"},
        ])
        results = check_v8_funnel(now_ts=now, funnel_path=self.path, grace_seconds=120)
        self.assertEqual(results[0].status, STATUS_CRITICAL)
        self.assertIn("add_signal_entered", results[0].evidence["missing_by_stage"])

    def test_dedup_rejected_counts_as_valid_terminal_for_add_signal_entered(self):
        now = 1_000_000.0
        self._write([
            {"ts": now - 500, "stage": "add_signal_entered", "event_id": "ev1", "mint": "M1"},
            {"ts": now - 499, "stage": "dedup_rejected", "event_id": "ev1", "mint": "M1"},
        ])
        results = check_v8_funnel(now_ts=now, funnel_path=self.path, grace_seconds=120)
        self.assertEqual(results[0].status, STATUS_OK, results[0].reason)

    def test_missing_file_is_unknown_not_ok(self):
        results = check_v8_funnel(now_ts=1_000_000.0, funnel_path=self.path)
        self.assertEqual(results[0].status, STATUS_UNKNOWN)

    def test_empty_file_is_unknown_not_ok(self):
        self.path.write_text("")
        results = check_v8_funnel(now_ts=1_000_000.0, funnel_path=self.path)
        self.assertEqual(results[0].status, STATUS_UNKNOWN)

    def test_malformed_lines_are_skipped_not_fatal(self):
        now = 1_000_000.0
        self.path.write_text(
            "not json at all\n"
            + json.dumps({"ts": now - 10, "stage": "v8_gate_entered", "event_id": "ev1", "mint": "M1"})
        )
        results = check_v8_funnel(now_ts=now, funnel_path=self.path, grace_seconds=120)
        self.assertEqual(results[0].status, STATUS_OK)


if __name__ == "__main__":
    unittest.main()
