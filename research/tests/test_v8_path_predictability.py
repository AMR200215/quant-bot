"""research/tests/test_v8_path_predictability.py — YD-BATCH item YD1.

Run: python -m pytest research/tests/test_v8_path_predictability.py -v
"""

import csv
import tempfile
import unittest
from pathlib import Path

from research.v8_path_predictability import (
    _bucket_label, _realert_count_safe, _resolve_usability, find_best_condition,
    fit_logistic_and_auc, MintRecord, MIN_CELL_N, MATERIAL_LIFT_MIN_N,
    _PROGRESS_BUCKETS, _PROGRESS_LABELS,
)
from research.v8_feature_enforcement import check_features_allowed
from research.path_schema import PATH_HEADER

_IN_ERA_ISO = "2026-08-25T12:00:00+00:00"
_IN_ERA_TS_MS = 1787832000000  # ~2026-08-25T12:00:00Z in ms, exact value unimportant


def _write_csv(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PATH_HEADER)
        w.writeheader()
        for r in rows:
            full = {k: "" for k in PATH_HEADER}
            full.update(r)
            w.writerow(full)


def _real_row(ts_ms):
    return {
        "schema_version": "3", "ts_ms": str(ts_ms), "price_usd": "0.00005",
        "price_sol": "0.0000003", "vsol": "50.0", "vtok": "1000000000",
        "venue_state": "CURVE_ACTIVE", "source": "live_pp", "backfilled": "false",
        "data_status": "ok",
    }


def _event(mint, alert_iso, path_file=None):
    return {
        "event_id": f"e_{mint}", "token_address": mint, "alert_time": alert_iso,
        "progress_at_signal": 0.3, "venue_state_at_signal": "CURVE_ACTIVE",
        "path_file": path_file, "progress_capture_lag_ms": 500,
    }


class TestBucketLabel(unittest.TestCase):

    def test_boundary_goes_to_upper_bucket(self):
        self.assertEqual(_bucket_label(0.25, _PROGRESS_BUCKETS, _PROGRESS_LABELS), "25-50%")
        self.assertEqual(_bucket_label(0.24, _PROGRESS_BUCKETS, _PROGRESS_LABELS), "0-25%")

    def test_value_at_or_above_last_bucket_lo_falls_in_last(self):
        self.assertEqual(_bucket_label(0.95, _PROGRESS_BUCKETS, _PROGRESS_LABELS), "90%+")

    def test_value_beyond_all_ranges_falls_in_last_bucket(self):
        self.assertEqual(_bucket_label(5.0, _PROGRESS_BUCKETS, _PROGRESS_LABELS), "90%+")


class TestRealertCountSafe(unittest.TestCase):

    def test_no_realert_times_gives_zero(self):
        self.assertEqual(_realert_count_safe(None, "2026-08-25T12:00:00+00:00"), 0)
        self.assertEqual(_realert_count_safe([], "2026-08-25T12:00:00+00:00"), 0)

    def test_only_counts_times_strictly_before_alert_time(self):
        times = ["2026-08-25T11:00:00+00:00", "2026-08-25T11:30:00+00:00", "2026-08-25T13:00:00+00:00"]
        self.assertEqual(_realert_count_safe(times, "2026-08-25T12:00:00+00:00"), 2)

    def test_no_alert_time_gives_zero_not_crash(self):
        self.assertEqual(_realert_count_safe(["2026-08-25T11:00:00+00:00"], ""), 0)


class TestResolveUsability(unittest.TestCase):

    def test_ambiguous_mint_returns_none(self):
        event = _event("A", _IN_ERA_ISO)
        result = _resolve_usability(event, {"A"}, Path("/tmp"))
        self.assertIsNone(result)

    def test_no_path_file_returns_false(self):
        event = _event("A", _IN_ERA_ISO, path_file=None)
        result = _resolve_usability(event, set(), Path("/tmp"))
        self.assertFalse(result)

    def test_missing_on_disk_returns_false(self):
        event = _event("A", _IN_ERA_ISO, path_file="logs/research_paths/does-not-exist/A.csv")
        result = _resolve_usability(event, set(), Path("/tmp"))
        self.assertFalse(result)

    def test_valid_real_path_returns_true(self):
        mint = "GoodMint"
        rel = f"logs/research_paths/2026-08-25/{mint}.csv"
        event = _event(mint, _IN_ERA_ISO, path_file=rel)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_csv(root / rel, [_real_row(_IN_ERA_TS_MS), _real_row(_IN_ERA_TS_MS + 1000)])
            result = _resolve_usability(event, set(), root)
        self.assertTrue(result)

    def test_gz_rotated_path_still_resolves(self):
        import gzip
        mint = "GzMint"
        rel = f"logs/research_paths/2026-08-25/{mint}.csv"
        event = _event(mint, _IN_ERA_ISO, path_file=rel)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            plain = root / rel
            _write_csv(plain, [_real_row(_IN_ERA_TS_MS), _real_row(_IN_ERA_TS_MS + 1000)])
            with open(plain, "rb") as f_in, gzip.open(str(plain) + ".gz", "wb") as f_out:
                f_out.write(f_in.read())
            plain.unlink()
            result = _resolve_usability(event, set(), root)
        self.assertTrue(result)

    def test_backfilled_only_data_returns_false(self):
        mint = "BackfillOnly"
        rel = f"logs/research_paths/2026-08-25/{mint}.csv"
        event = _event(mint, _IN_ERA_ISO, path_file=rel)
        row = dict(_real_row(_IN_ERA_TS_MS))
        row["backfilled"] = "true"
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_csv(root / rel, [row])
            result = _resolve_usability(event, set(), root)
        self.assertFalse(result)


class TestFeatureEnforcementAssumptions(unittest.TestCase):
    """Guards against the module's hardcoded assumption (these 4 features
    are allowed_for_entry=true) silently drifting from the real registry."""

    def test_progress_vsol_velocity_realert_are_allowed_for_entry(self):
        violations = check_features_allowed(
            ["progress_at_signal", "vsol_at_signal", "channel_velocity_5m", "realert_times"], "entry",
        )
        self.assertEqual(violations, [])

    def test_top10_and_creator_holds_are_not_allowed_for_entry(self):
        violations = check_features_allowed(["top10_holder_pct", "creator_holds_pct"], "entry")
        self.assertEqual(len(violations), 2)


class TestFindBestCondition(unittest.TestCase):

    def test_finds_the_bucket_with_real_lift(self):
        records = []
        # baseline: mostly unusable, except velocity>=6 bucket which is mostly usable
        for i in range(MATERIAL_LIFT_MIN_N + 5):
            records.append(MintRecord(f"lo{i}", "2026-08-25T00:00:00+00:00", usable=False,
                                       features={"progress_at_signal": 0.3, "vsol_at_signal": 20,
                                                 "channel_velocity_5m": 0, "realert_count_safe": 0,
                                                 "hour_of_day": 12}))
        for i in range(MATERIAL_LIFT_MIN_N + 5):
            records.append(MintRecord(f"hi{i}", "2026-08-25T00:00:00+00:00", usable=True,
                                       features={"progress_at_signal": 0.3, "vsol_at_signal": 20,
                                                 "channel_velocity_5m": 8, "realert_count_safe": 0,
                                                 "hour_of_day": 12}))
        best = find_best_condition(records)
        self.assertIsNotNone(best)
        self.assertEqual(best["feature"], "channel_velocity_5m")
        self.assertEqual(best["bucket_label"], "6+")
        self.assertGreater(best["lift_pp"], 0)

    def test_empty_records_returns_none(self):
        self.assertIsNone(find_best_condition([]))

    def test_never_proposes_a_bucket_below_material_min_n(self):
        records = [
            MintRecord(f"m{i}", "2026-08-25T00:00:00+00:00", usable=(i % 2 == 0),
                       features={"progress_at_signal": 0.3, "vsol_at_signal": 20,
                                 "channel_velocity_5m": 0, "realert_count_safe": 0, "hour_of_day": 12})
            for i in range(5)   # well under MATERIAL_LIFT_MIN_N
        ]
        self.assertIsNone(find_best_condition(records))


class TestFitLogisticAndAuc(unittest.TestCase):

    def test_too_few_rows_reports_reason_not_crash(self):
        records = [
            MintRecord("a", "2026-08-25T00:00:00+00:00", usable=True,
                       features={"progress_at_signal": 0.3, "vsol_at_signal": 20,
                                 "channel_velocity_5m": 1, "realert_count_safe": 0, "hour_of_day": 12}),
        ]
        result = fit_logistic_and_auc(records)
        self.assertIsNotNone(result)
        self.assertIsNone(result["auc"])

    def test_separable_data_gives_high_auc(self):
        import random
        random.seed(0)
        records = []
        for i in range(60):
            usable = i % 2 == 0
            vsol = 80 if usable else 15   # strongly separating feature
            records.append(MintRecord(
                f"m{i}", "2026-08-25T00:00:00+00:00", usable=usable,
                features={"progress_at_signal": 0.3 + random.random() * 0.01,
                          "vsol_at_signal": vsol + random.random(),
                          "channel_velocity_5m": 2, "realert_count_safe": 0, "hour_of_day": 12},
            ))
        result = fit_logistic_and_auc(records)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result["auc"])
        self.assertGreater(result["auc"], 0.8)


if __name__ == "__main__":
    unittest.main()
