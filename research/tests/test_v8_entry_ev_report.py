"""research/tests/test_v8_entry_ev_report.py — first entry-EV (SELECTION)
evaluation, train+validation only, holdout structurally never read.

Run: python -m pytest research/tests/test_v8_entry_ev_report.py -v
"""

import unittest
from datetime import datetime, timedelta, timezone

from research.v8_entry_ev_report import (
    compute_entry_ev, _train_validation_outcome_rows, WINNER_THRESHOLD_PCT,
    build_report,
)
from research.v8_candidate_registry import CANDIDATES

_P0 = next(c for c in CANDIDATES if c["candidate_id"] == "V8-P0")


def _make_events(n, n_days, pct_values=None, progress=0.3, venue="CURVE_ACTIVE"):
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    events = []
    for i in range(n):
        day_offset = i % n_days
        alert_dt = start + timedelta(days=day_offset, seconds=i)
        pct = None if pct_values is None else pct_values[i % len(pct_values)]
        events.append({
            "event_id": f"e{i}", "token_address": f"MINT_{i}", "alert_time": alert_dt.isoformat(),
            "progress_at_signal": progress, "venue_state_at_signal": venue,
            "pct_change_peak": pct,
        })
    return events


class TestTrainValidationOutcomeRows(unittest.TestCase):

    def test_never_reads_holdout_structurally(self):
        """Grep-level guard: result.holdout must never appear as an
        operand anywhere in this function's source."""
        import inspect
        import research.v8_entry_ev_report as mod
        src = inspect.getsource(mod._train_validation_outcome_rows)
        self.assertNotIn("result.holdout", src)

    def test_too_few_events_returns_empty(self):
        events = [{"token_address": "A", "alert_time": "2026-08-01T00:00:00+00:00",
                   "venue_state_at_signal": "CURVE_ACTIVE", "pct_change_peak": 10.0}]
        self.assertEqual(_train_validation_outcome_rows(events), [])

    def test_only_venue_qualified_rows_considered(self):
        events = _make_events(300, 30, pct_values=[10.0], venue="UNKNOWN")
        self.assertEqual(_train_validation_outcome_rows(events), [])

    def test_null_pct_change_peak_excluded(self):
        events = _make_events(300, 30, pct_values=[None])
        rows = _train_validation_outcome_rows(events)
        self.assertEqual(rows, [])

    def test_real_split_returns_a_strict_subset_of_all_events(self):
        events = _make_events(300, 30, pct_values=[10.0, 60.0])
        rows = _train_validation_outcome_rows(events)
        self.assertGreater(len(rows), 0)
        self.assertLess(len(rows), 300)   # holdout bucket rows must be excluded


class TestComputeEntryEv(unittest.TestCase):

    def test_empty_gives_insufficient_not_error(self):
        r = compute_entry_ev("TEST", [], below_floor=True)
        self.assertEqual(r.n, 0)
        self.assertIsNone(r.win_rate_pct)

    def test_win_rate_uses_the_reused_threshold(self):
        rows = [{"pct_change_peak": 100.0}] * 5 + [{"pct_change_peak": 10.0}] * 5
        r = compute_entry_ev("TEST", rows, below_floor=False)
        self.assertEqual(r.n, 10)
        self.assertEqual(r.win_rate_pct, 50.0)
        self.assertEqual(WINNER_THRESHOLD_PCT, 50.0)

    def test_mean_and_median_computed_correctly(self):
        rows = [{"pct_change_peak": v} for v in [10.0, 20.0, 30.0, 40.0]]
        r = compute_entry_ev("TEST", rows, below_floor=False)
        self.assertEqual(r.mean_pct_change_peak, 25.0)
        self.assertEqual(r.median_pct_change_peak, 25.0)

    def test_percentiles_none_below_min_n(self):
        rows = [{"pct_change_peak": 10.0}, {"pct_change_peak": 20.0}]
        r = compute_entry_ev("TEST", rows, below_floor=False)
        self.assertIsNone(r.p25)

    def test_below_floor_flag_passed_through(self):
        r = compute_entry_ev("TEST", [{"pct_change_peak": 10.0}], below_floor=True)
        self.assertTrue(r.below_floor)


class TestBuildReport(unittest.TestCase):

    def test_covers_every_candidate(self):
        class _FakeTable:
            def select(self, *a, **k): return self
            def eq(self, *a, **k): return self
            def range(self, *a, **k): return self
            def execute(self): return type("R", (), {"data": []})()
        class _FakeSb:
            def table(self, *a, **k): return _FakeTable()

        results = build_report(_FakeSb())
        self.assertEqual({r.candidate_id for r in results}, {c["candidate_id"] for c in CANDIDATES})

    def test_below_floor_marked_when_not_in_selection_ready_set(self):
        class _FakeTable:
            def select(self, *a, **k): return self
            def eq(self, *a, **k): return self
            def range(self, *a, **k): return self
            def execute(self): return type("R", (), {"data": []})()
        class _FakeSb:
            def table(self, *a, **k): return _FakeTable()

        results = build_report(_FakeSb(), selection_ready_candidates={"V8-P0"})
        by_id = {r.candidate_id: r for r in results}
        self.assertFalse(by_id["V8-P0"].below_floor)
        self.assertTrue(by_id["V8-P3"].below_floor)


if __name__ == "__main__":
    unittest.main()
