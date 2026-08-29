"""research/tests/test_v8_execution_proxy_trend.py — YD-BATCH item YD3.

Run: python -m pytest research/tests/test_v8_execution_proxy_trend.py -v
"""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from research.v8_execution_proxy_trend import DayPoint, classify_trend, compute_trend


class TestClassifyTrend(unittest.TestCase):

    def test_insufficient_history_below_three_points(self):
        points = [DayPoint("2026-08-28", 5, 1, 20.0), DayPoint("2026-08-29", 6, 2, 33.3)]
        self.assertEqual(classify_trend(points), "INSUFFICIENT_HISTORY")

    def test_rising_when_delta_at_least_5pp(self):
        points = [
            DayPoint("2026-08-23", 10, 1, 10.0), DayPoint("2026-08-26", 20, 3, 15.0),
            DayPoint("2026-08-29", 30, 6, 20.0),
        ]
        self.assertEqual(classify_trend(points), "RISING")

    def test_plateaued_when_delta_small(self):
        points = [
            DayPoint("2026-08-23", 10, 4, 40.0), DayPoint("2026-08-26", 20, 8, 40.0),
            DayPoint("2026-08-29", 30, 12, 40.0),
        ]
        self.assertEqual(classify_trend(points), "PLATEAUED")

    def test_falling_when_delta_at_most_minus_5pp(self):
        points = [
            DayPoint("2026-08-23", 10, 5, 50.0), DayPoint("2026-08-26", 40, 8, 20.0),
            DayPoint("2026-08-29", 100, 10, 10.0),
        ]
        self.assertEqual(classify_trend(points), "FALLING")

    def test_days_with_zero_admitted_are_excluded_from_first_last_comparison(self):
        points = [
            DayPoint("2026-08-23", 0, 0, 0.0),   # no data yet this day -- must not count as "first"
            DayPoint("2026-08-26", 10, 4, 40.0),
            DayPoint("2026-08-27", 20, 8, 40.0),
            DayPoint("2026-08-29", 30, 12, 40.0),
        ]
        self.assertEqual(classify_trend(points), "PLATEAUED")


class TestComputeTrend(unittest.TestCase):

    def _write_admission_log(self, root, rows):
        out = root / "logs" / "research_admission"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "admission_log.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def _write_proxy_log(self, root, rows):
        out = root / "logs" / "research_execution_proxy"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "execution_proxy_log.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def test_no_era_gives_empty(self):
        class _FakeTable:
            def select(self, *a, **k): return self
            def eq(self, *a, **k): return self
            def range(self, *a, **k): return self
            def execute(self): return type("R", (), {"data": []})()
        class _FakeSb:
            def table(self, *a, **k): return _FakeTable()
        with tempfile.TemporaryDirectory() as d:
            points = compute_trend(_FakeSb(), Path(d))
        self.assertEqual(points, [])

    def test_cumulative_counts_grow_monotonically(self):
        now = datetime.now(timezone.utc)
        mints = [f"M{i}" for i in range(5)]

        class _FakeTable:
            def __init__(self):
                self._rows = [
                    {"token_address": m, "alert_time": (now - timedelta(days=1)).isoformat(),
                     "venue_state_at_signal": "CURVE_ACTIVE"}
                    for m in mints
                ]
            def select(self, *a, **k): return self
            def eq(self, *a, **k): return self
            def range(self, offset, end, **k):
                self._slice = self._rows[offset:end + 1] if offset == 0 else []
                return self
            def execute(self):
                return type("R", (), {"data": self._slice})()

        class _FakeSb:
            def table(self, *a, **k): return _FakeTable()

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            admit_rows = [
                {"ts": (now - timedelta(hours=20)).timestamp(), "token_address": m, "path_admitted": True}
                for m in mints
            ]
            self._write_admission_log(root, admit_rows)
            proxy_rows = [
                {"token_address": mints[0], "status": "OK",
                 "observed_at": (now - timedelta(hours=10)).isoformat()},
                {"token_address": mints[1], "status": "OK",
                 "observed_at": (now - timedelta(hours=1)).isoformat()},
            ]
            self._write_proxy_log(root, proxy_rows)

            # Patch era start to be well before all this synthetic data
            import research.v8_execution_proxy_trend as mod
            orig = mod.trustworthy_collection_era_start
            mod.trustworthy_collection_era_start = lambda root_arg: now - timedelta(days=3)
            try:
                points = compute_trend(_FakeSb(), root, window_days=3)
            finally:
                mod.trustworthy_collection_era_start = orig

        self.assertEqual(len(points), 3)
        # By the final (today) point, both admissions and both proxy observations count.
        self.assertEqual(points[-1].cumulative_admitted_n, 5)
        self.assertEqual(points[-1].cumulative_observed_n, 2)
        self.assertAlmostEqual(points[-1].coverage_pct, 40.0, places=1)
        # Monotonic non-decreasing cumulative counts across days.
        admitted_series = [p.cumulative_admitted_n for p in points]
        observed_series = [p.cumulative_observed_n for p in points]
        self.assertEqual(admitted_series, sorted(admitted_series))
        self.assertEqual(observed_series, sorted(observed_series))


if __name__ == "__main__":
    unittest.main()
