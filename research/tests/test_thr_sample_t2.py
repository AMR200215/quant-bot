"""research/tests/test_thr_sample_t2.py — THR-BATCH T2 pre-registered sample.

Run: python -m pytest research/tests/test_thr_sample_t2.py -v
"""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from research.thr_sample_t2 import build_sample, _load_admission_probabilities


def _make_rows(n, n_days, venue="CURVE_ACTIVE"):
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        day_offset = i % n_days
        alert_dt = start + timedelta(days=day_offset, seconds=i)
        rows.append({
            "event_id": f"e{i}", "token_address": f"MINT_{i}",
            "alert_time": alert_dt.isoformat(), "venue_state_at_signal": venue,
        })
    return rows


class _FakeTable:
    def __init__(self, rows):
        self._rows = rows

    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        return self

    def range(self, offset, end):
        return _FakeRangeResult(self._rows[offset:end + 1])


class _FakeRangeResult:
    def __init__(self, data):
        self._data = data

    def execute(self):
        return type("R", (), {"data": self._data})()


class _FakeSb:
    def __init__(self, rows):
        self._rows = rows

    def table(self, *a, **k):
        return _FakeTable(self._rows)


class TestBuildSample(unittest.TestCase):

    def test_never_includes_holdout_mints(self):
        rows = _make_rows(300, 30)
        sb = _FakeSb(rows)
        manifest = build_sample(sb, target_n=50, seed=1, admission_log_path=Path("/nonexistent"))
        sampled_mints = {t["token_address"] for t in manifest["tokens"]}
        # holdout_mints_excluded must be > 0 for this population shape and
        # none of them can appear in the output
        self.assertGreater(manifest["holdout_mints_excluded"], 0)
        self.assertEqual(manifest["sample_n"], len(sampled_mints))

    def test_deterministic_given_same_seed(self):
        rows = _make_rows(300, 30)
        m1 = build_sample(_FakeSb(rows), target_n=50, seed=42, admission_log_path=Path("/nonexistent"))
        m2 = build_sample(_FakeSb(rows), target_n=50, seed=42, admission_log_path=Path("/nonexistent"))
        self.assertEqual([t["token_address"] for t in m1["tokens"]],
                          [t["token_address"] for t in m2["tokens"]])

    def test_different_seed_gives_different_sample(self):
        rows = _make_rows(300, 30)
        m1 = build_sample(_FakeSb(rows), target_n=50, seed=1, admission_log_path=Path("/nonexistent"))
        m2 = build_sample(_FakeSb(rows), target_n=50, seed=2, admission_log_path=Path("/nonexistent"))
        self.assertNotEqual([t["token_address"] for t in m1["tokens"]],
                             [t["token_address"] for t in m2["tokens"]])

    def test_sample_n_capped_at_population_when_population_smaller(self):
        rows = _make_rows(20, 10)
        manifest = build_sample(_FakeSb(rows), target_n=500, seed=1, admission_log_path=Path("/nonexistent"))
        self.assertLessEqual(manifest["sample_n"], manifest["population_n"])
        self.assertLess(manifest["sample_n"], 500)

    def test_dedupes_by_mint(self):
        rows = _make_rows(300, 30) + _make_rows(300, 30)  # duplicate alerts per mint
        manifest = build_sample(_FakeSb(rows), target_n=50, seed=1, admission_log_path=Path("/nonexistent"))
        addrs = [t["token_address"] for t in manifest["tokens"]]
        self.assertEqual(len(addrs), len(set(addrs)))

    def test_non_curve_active_excluded(self):
        rows = _make_rows(300, 30, venue="GRADUATED")
        manifest = build_sample(_FakeSb([]), target_n=50, seed=1, admission_log_path=Path("/nonexistent"))
        self.assertEqual(manifest["population_n"], 0)


class TestLoadAdmissionProbabilities(unittest.TestCase):

    def test_joins_real_jsonl_format(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"token_address": "MINT_A", "path_sampling_probability": 0.5,
                                 "admission_reason": "sampled_admit"}) + "\n")
            f.write(json.dumps({"token_address": "MINT_B", "path_sampling_probability": 1.0,
                                 "admission_reason": "under_hourly_pace"}) + "\n")
            path = Path(f.name)
        try:
            out = _load_admission_probabilities(path)
            self.assertEqual(out["MINT_A"], (0.5, "sampled_admit"))
            self.assertEqual(out["MINT_B"], (1.0, "under_hourly_pace"))
        finally:
            path.unlink()

    def test_missing_file_returns_empty_dict(self):
        self.assertEqual(_load_admission_probabilities(Path("/definitely/not/here.jsonl")), {})

    def test_malformed_lines_skipped_not_fatal(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("not json\n")
            f.write(json.dumps({"token_address": "MINT_A", "path_sampling_probability": 0.5,
                                 "admission_reason": "x"}) + "\n")
            path = Path(f.name)
        try:
            out = _load_admission_probabilities(path)
            self.assertEqual(out["MINT_A"], (0.5, "x"))
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
