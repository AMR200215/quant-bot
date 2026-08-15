"""research/tests/test_v8_head_to_head.py — V8-REWIRE VR14/VR18.

Run: python -m pytest research/tests/test_v8_head_to_head.py -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from research.scripts.v8_head_to_head import build_matrix


def _write(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events))


class TestV8HeadToHeadMatrix(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "v8_funnel.jsonl"

    def tearDown(self):
        self._tmp.cleanup()

    def test_all_four_cells_classified_correctly(self):
        _write(self.path, [
            {"ts": 1, "stage": "telegram_received", "event_id": "pp", "mint": "M1"},
            {"ts": 2, "stage": "screening_passed", "event_id": "pp", "mint": "M1"},
            {"ts": 3, "stage": "v8_opened", "event_id": "pp", "mint": "M1"},

            {"ts": 4, "stage": "telegram_received", "event_id": "pf", "mint": "M2"},
            {"ts": 5, "stage": "screening_passed", "event_id": "pf", "mint": "M2"},
            {"ts": 6, "stage": "v8_gate_rejected", "event_id": "pf", "mint": "M2"},

            {"ts": 7, "stage": "telegram_received", "event_id": "fp", "mint": "M3"},
            {"ts": 8, "stage": "screening_rejected", "event_id": "fp", "mint": "M3"},
            {"ts": 9, "stage": "v8_opened", "event_id": "fp", "mint": "M3"},

            {"ts": 10, "stage": "telegram_received", "event_id": "ff", "mint": "M4"},
            {"ts": 11, "stage": "screening_rejected", "event_id": "ff", "mint": "M4"},
            {"ts": 12, "stage": "v8_pass_unpriced", "event_id": "ff", "mint": "M4"},
        ])
        result = build_matrix(self.path, min_ts=None)
        m = result["matrix"]
        self.assertEqual([e["event_id"] for e in m["pass_pass"]], ["pp"])
        self.assertEqual([e["event_id"] for e in m["pass_fail"]], ["pf"])
        self.assertEqual([e["event_id"] for e in m["fail_pass"]], ["fp"])
        self.assertEqual([e["event_id"] for e in m["fail_fail"]], ["ff"])
        self.assertEqual(result["unresolved"], [])

    def test_v8_transport_duplicate_and_pass_unpriced_both_count_as_v8_fail(self):
        _write(self.path, [
            {"ts": 1, "stage": "telegram_received", "event_id": "e1", "mint": "M1"},
            {"ts": 2, "stage": "screening_passed", "event_id": "e1", "mint": "M1"},
            {"ts": 3, "stage": "v8_transport_duplicate", "event_id": "e1", "mint": "M1"},
        ])
        result = build_matrix(self.path, min_ts=None)
        self.assertEqual(len(result["matrix"]["pass_fail"]), 1)

    def test_unresolved_v8_side_excluded_from_matrix(self):
        _write(self.path, [
            {"ts": 1, "stage": "telegram_received", "event_id": "e1", "mint": "M1"},
            {"ts": 2, "stage": "screening_passed", "event_id": "e1", "mint": "M1"},
            # no V8 terminal stage yet -- still in flight
        ])
        result = build_matrix(self.path, min_ts=None)
        total_in_matrix = sum(len(v) for v in result["matrix"].values())
        self.assertEqual(total_in_matrix, 0)
        self.assertEqual(len(result["unresolved"]), 1)
        self.assertEqual(result["unresolved"][0]["v7"], "pass")
        self.assertIsNone(result["unresolved"][0]["v8"])

    def test_unresolved_v7_side_excluded_from_matrix(self):
        _write(self.path, [
            {"ts": 1, "stage": "telegram_received", "event_id": "e1", "mint": "M1"},
            {"ts": 2, "stage": "v8_opened", "event_id": "e1", "mint": "M1"},
            # no screening_passed/rejected recorded for this event_id
        ])
        result = build_matrix(self.path, min_ts=None)
        total_in_matrix = sum(len(v) for v in result["matrix"].values())
        self.assertEqual(total_in_matrix, 0)
        self.assertEqual(len(result["unresolved"]), 1)
        self.assertIsNone(result["unresolved"][0]["v7"])

    def test_min_ts_filters_pre_deploy_events(self):
        _write(self.path, [
            {"ts": 1, "stage": "telegram_received", "event_id": "old", "mint": "M1"},
            {"ts": 2, "stage": "screening_rejected", "event_id": "old", "mint": "M1"},
            {"ts": 3, "stage": "v8_opened", "event_id": "old", "mint": "M1"},

            {"ts": 100, "stage": "telegram_received", "event_id": "new", "mint": "M2"},
            {"ts": 101, "stage": "screening_rejected", "event_id": "new", "mint": "M2"},
            {"ts": 102, "stage": "v8_opened", "event_id": "new", "mint": "M2"},
        ])
        result = build_matrix(self.path, min_ts=50)
        self.assertEqual([e["event_id"] for e in result["matrix"]["fail_pass"]], ["new"])

    def test_malformed_lines_skipped(self):
        self.path.write_text(
            "not json\n" + json.dumps({"ts": 1, "stage": "telegram_received", "event_id": "e1", "mint": "M1"})
        )
        result = build_matrix(self.path, min_ts=None)   # must not raise
        self.assertEqual(len(result["unresolved"]), 1)


if __name__ == "__main__":
    unittest.main()
