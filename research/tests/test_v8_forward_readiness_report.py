"""research/tests/test_v8_forward_readiness_report.py — V8 DATA
RECOVERY batch item 9, CORRECTED by the V8 READINESS MONITOR
CORRECTION batch: automatic forward-readiness report.

Run: python -m pytest research/tests/test_v8_forward_readiness_report.py -v
"""

import csv
import tempfile
import unittest
from pathlib import Path

from research.v8_forward_readiness_report import (
    _progress_bucket, _estimate_velocity, compute_accumulation_velocity,
    FORWARD_READINESS_REPORT_VERSION, ForwardReadinessReport,
    _MIN_DAYS_FOR_VELOCITY_ESTIMATE, _query_candidate_evidence, _candidate_selection_ready,
    _compute_split_counts, _compute_diagnostics_feasibility, _collection_yield_execution_proxy_ready,
)
from research.v8_candidate_registry import CANDIDATES
from research.path_schema import PATH_HEADER

_BASELINE = next(c for c in CANDIDATES if c["candidate_id"] == "BASELINE-0")
_P0 = next(c for c in CANDIDATES if c["candidate_id"] == "V8-P0")
_P3 = next(c for c in CANDIDATES if c["candidate_id"] == "V8-P3")


def _write_path_file(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PATH_HEADER)
        w.writeheader()
        for r in rows:
            full = {k: "" for k in PATH_HEADER}
            full.update(r)
            w.writerow(full)


def _good_row(ts_ms):
    return {
        "schema_version": "2", "ts_ms": str(ts_ms), "price_usd": "0.00005", "price_sol": "0.0000003",
        "vsol": "50.0", "vtok": "1000000000", "venue_state": "CURVE_ACTIVE", "source": "live_pp",
        "backfilled": "false", "data_status": "ok",
    }


def _write_admission_log(root: Path, events: list, probability: float = 1.0, admitted: bool = True):
    """Synthetic logs/research_admission/admission_log.jsonl -- one
    admitted row per event's mint, matching the real per-mint schema
    research/peak_tracker.py writes (token_address, ts, path_admitted,
    path_sampling_probability, admission_reason)."""
    import json as _json
    out_dir = root / "logs" / "research_admission"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "admission_log.jsonl", "w") as f:
        for e in events:
            from datetime import datetime
            ts = datetime.fromisoformat(e["alert_time"].replace("Z", "+00:00")).timestamp()
            f.write(_json.dumps({
                "ts": ts, "token_address": e["token_address"], "utc_hour": 0,
                "path_eligible": True, "path_admitted": admitted,
                "path_sampling_probability": probability,
                "admission_reason": "under_hourly_pace" if probability >= 1.0 else "sampled_admit",
                "budget_used": 0, "budget_remaining": 100000,
            }) + "\n")


def _make_synthetic_events(n, n_days, progress=0.3, venue="CURVE_ACTIVE", start_day_str="2026-08-01"):
    """n distinct mints spread across n_days distinct calendar days --
    each mint alerts exactly once (no boundary-spanning ambiguity)."""
    from datetime import datetime, timedelta, timezone
    y, m, d = (int(x) for x in start_day_str.split("-"))
    start = datetime(y, m, d, tzinfo=timezone.utc)
    events = []
    for i in range(n):
        day_offset = i % n_days
        alert_dt = start + timedelta(days=day_offset, seconds=i)
        events.append({
            "event_id": f"e{i}", "token_address": f"MINT_{i}", "alert_time": alert_dt.isoformat(),
            "progress_at_signal": progress, "venue_state_at_signal": venue,
            "path_file": None, "progress_capture_lag_ms": 500, "pct_change_peak": 50.0,
            "_alert_ts_ms": int(alert_dt.timestamp() * 1000),
        })
    return events


class TestProgressBucket(unittest.TestCase):
    def test_buckets(self):
        self.assertEqual(_progress_bucket(0.3), "<50%")
        self.assertEqual(_progress_bucket(0.6), "50-70%")
        self.assertEqual(_progress_bucket(0.8), "70-85%")
        self.assertEqual(_progress_bucket(0.95), "85%+")
        self.assertEqual(_progress_bucket(None), "unknown")


class TestEstimateVelocity(unittest.TestCase):

    def test_insufficient_history_below_min_days(self):
        events = {"2026-08-19": 5}
        rate, label = _estimate_velocity(events, window_days=7)
        self.assertEqual(label, "INSUFFICIENT_HISTORY")

    def test_stable_when_low_variance(self):
        events = {"2026-08-15": 10, "2026-08-16": 11, "2026-08-17": 9, "2026-08-18": 10}
        rate, label = _estimate_velocity(events, window_days=7)
        self.assertEqual(label, "STABLE")
        self.assertAlmostEqual(rate, 10.0, delta=1.0)

    def test_too_variable_when_high_variance(self):
        events = {"2026-08-15": 1, "2026-08-16": 50, "2026-08-17": 2, "2026-08-18": 40}
        rate, label = _estimate_velocity(events, window_days=7)
        self.assertEqual(label, "TOO_VARIABLE_TO_PROJECT")

    def test_zero_events_is_insufficient_not_stable_zero(self):
        events = {"2026-08-15": 0, "2026-08-16": 0}
        rate, label = _estimate_velocity(events, window_days=7)
        self.assertEqual(label, "INSUFFICIENT_HISTORY")

    def test_never_raises_on_empty_dict(self):
        rate, label = _estimate_velocity({}, window_days=7)
        self.assertIsNone(rate)
        self.assertEqual(label, "INSUFFICIENT_HISTORY")


class TestComputeAccumulationVelocity(unittest.TestCase):

    def test_never_predicts_a_date(self):
        v = compute_accumulation_velocity([], [], [])
        self.assertFalse(hasattr(v, "estimated_completion_date"))
        self.assertFalse(hasattr(v, "eta"))

    def test_empty_inputs_give_insufficient_history(self):
        v = compute_accumulation_velocity([], [], [])
        self.assertEqual(v.stability_label, "INSUFFICIENT_HISTORY")

    def test_note_explains_no_prediction_policy(self):
        v = compute_accumulation_velocity([], [], [])
        self.assertIn("no completion date", v.note.lower())


class TestReportVersioning(unittest.TestCase):
    def test_version_is_2(self):
        self.assertEqual(FORWARD_READINESS_REPORT_VERSION, 2)

    def test_report_structurally_never_claims_holdout_evaluated(self):
        import inspect
        import research.v8_forward_readiness_report as mod
        src = inspect.getsource(mod)
        self.assertNotIn("holdout_evaluated=True", src)
        self.assertNotIn("holdout_evaluated = True", src)

    def test_diagnostics_feasibility_never_reads_holdout_bucket(self):
        """Structural check: _compute_diagnostics_feasibility only ever
        combines result.train + result.validation -- result.holdout must
        never appear as an operand anywhere in that function's source."""
        import inspect
        import research.v8_forward_readiness_report as mod
        src = inspect.getsource(mod._compute_diagnostics_feasibility)
        self.assertNotIn("result.holdout", src)


class TestSplitCounts(unittest.TestCase):

    def test_real_counts_not_hardcoded_zero(self):
        """The exact bug being fixed: train_n/validation_n/holdout_n used
        to be hardcoded 0 for every candidate."""
        events = _make_synthetic_events(n=300, n_days=30)
        counts = _compute_split_counts(events)
        self.assertGreater(counts.train_n, 0)
        self.assertGreater(counts.validation_n, 0)
        self.assertGreater(counts.holdout_n, 0)
        self.assertEqual(counts.train_n + counts.validation_n + counts.holdout_n + counts.boundary_purged_n, 300)

    def test_too_few_events_gives_zero_not_error(self):
        counts = _compute_split_counts([{"token_address": "A", "alert_time": "2026-08-01T00:00:00+00:00"}])
        self.assertEqual(counts.train_n, 0)
        self.assertEqual(counts.validation_n, 0)
        self.assertEqual(counts.holdout_n, 0)


class TestDiagnosticsFeasibility(unittest.TestCase):

    def test_feasible_with_enough_days_and_outcomes(self):
        events = _make_synthetic_events(n=100, n_days=20)
        result = _compute_diagnostics_feasibility(events)
        self.assertTrue(result.computable)
        self.assertGreater(result.train_validation_outcome_n, 0)

    def test_infeasible_with_too_few_events(self):
        result = _compute_diagnostics_feasibility([{"token_address": "A", "alert_time": "2026-08-01T00:00:00+00:00"}])
        self.assertFalse(result.computable)

    def test_infeasible_when_no_outcome_data(self):
        events = _make_synthetic_events(n=50, n_days=10)
        for e in events:
            e["pct_change_peak"] = None
        result = _compute_diagnostics_feasibility(events)
        self.assertFalse(result.computable)
        self.assertEqual(result.train_validation_outcome_n, 0)


class TestQueryCandidateEvidenceAndSelectionReady(unittest.TestCase):
    """Self-audit scenarios (item 7)."""

    def test_remains_false_with_many_venue_rows_but_zero_usable_paths(self):
        events = _make_synthetic_events(n=300, n_days=30)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # No path files written at all -- zero usable paths.
            evidence = _query_candidate_evidence(_BASELINE, events, [], root)
        self.assertEqual(evidence.path_coverage.usable_n, 0)
        self.assertFalse(_candidate_selection_ready(evidence, requires_venue=True))

    def test_remains_false_with_paths_but_zero_execution_proxies(self):
        # start_day_str="2026-08-20" -- inside the trustworthy collection
        # era (PRICE_CORRECTION_DEPLOY_UTC=2026-08-19) so this test
        # exercises the intended scenario (paths+admission present,
        # proxies missing) rather than accidentally passing because
        # events fell outside the era.
        events = _make_synthetic_events(n=300, n_days=30, start_day_str="2026-08-20")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for e in events:
                rel = f"logs/research_paths/2026-08-20/{e['token_address']}.csv"
                e["path_file"] = rel
                _write_path_file(root / rel, [_good_row(e["_alert_ts_ms"]), _good_row(e["_alert_ts_ms"] + 1000)])
            _write_admission_log(root, events)
            # A synthetic execution_proxy_log.jsonl entry (status != OK,
            # so it never counts as observed) is enough to anchor
            # pp_funded_era_start without an empty-log fallback scan.
            proxy_dir = root / "logs" / "research_execution_proxy"
            proxy_dir.mkdir(parents=True)
            (proxy_dir / "execution_proxy_log.jsonl").write_text(
                '{"token_address": "ANCHOR", "status": "UNAVAILABLE", "observed_at": "2026-08-20T00:00:00+00:00"}\n'
            )
            evidence = _query_candidate_evidence(_BASELINE, events, [], root)
        self.assertGreater(evidence.path_coverage.usable_n, 0)
        self.assertGreater(evidence.collection_yield.admitted_with_valid_usable_path_n, 0)
        self.assertEqual(evidence.collection_yield.execution_proxy_observed_n, 0)
        self.assertFalse(_candidate_selection_ready(evidence, requires_venue=True))

    def test_one_execution_proxy_observation_cannot_make_it_ready(self):
        events = _make_synthetic_events(n=300, n_days=30, start_day_str="2026-08-20")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for e in events:
                rel = f"logs/research_paths/2026-08-20/{e['token_address']}.csv"
                e["path_file"] = rel
                _write_path_file(root / rel, [_good_row(e["_alert_ts_ms"]), _good_row(e["_alert_ts_ms"] + 1000)])
            _write_admission_log(root, events)
            proxy_rows = [{"token_address": events[0]["token_address"], "status": "OK",
                           "observed_at": "2026-08-20T00:00:00+00:00"}]
            evidence = _query_candidate_evidence(_BASELINE, events, proxy_rows, root)
        self.assertEqual(evidence.collection_yield.execution_proxy_observed_n, 1)
        self.assertFalse(_collection_yield_execution_proxy_ready(evidence.collection_yield).ready)
        self.assertFalse(_candidate_selection_ready(evidence, requires_venue=True))

    def test_transitions_false_to_true_with_sufficiently_large_synthetic_dataset(self):
        events = _make_synthetic_events(n=300, n_days=30, progress=0.3, start_day_str="2026-08-20")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proxy_rows = []
            for e in events:
                rel = f"logs/research_paths/2026-08-20/{e['token_address']}.csv"
                e["path_file"] = rel
                _write_path_file(root / rel, [_good_row(e["_alert_ts_ms"]), _good_row(e["_alert_ts_ms"] + 1000)])
                proxy_rows.append({"token_address": e["token_address"], "status": "OK",
                                    "observed_at": e["alert_time"]})
            _write_admission_log(root, events)
            evidence = _query_candidate_evidence(_BASELINE, events, proxy_rows, root)

        self.assertGreaterEqual(evidence.forward_venue_qualified_n, 100)
        self.assertGreaterEqual(evidence.venue_qualified_unique_days, 14)
        cy = evidence.collection_yield
        self.assertFalse(cy.era_undetermined)
        self.assertEqual(cy.path_admitted_n, 300)
        self.assertTrue(cy.admitted_path_yield_pct >= 50.0)
        self.assertTrue(_collection_yield_execution_proxy_ready(cy).ready)
        self.assertTrue(evidence.diagnostics_feasibility.computable)
        self.assertTrue(_candidate_selection_ready(evidence, requires_venue=True))

    def test_candidate_specific_path_coverage_differs_between_candidates(self):
        """P0 has no progress filter (matches everything); a narrower
        candidate with fewer eligible events must get its OWN coverage
        number, not the global corpus's."""
        events = _make_synthetic_events(n=50, n_days=10, progress=0.9)  # only qualifies for P0, not P3 (<0.85)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for e in events:
                rel = f"logs/research_paths/2026-08-01/{e['token_address']}.csv"
                e["path_file"] = rel
                _write_path_file(root / rel, [_good_row(e["_alert_ts_ms"]), _good_row(e["_alert_ts_ms"] + 1000)])
            p0_evidence = _query_candidate_evidence(_P0, events, [], root)
            p3_evidence = _query_candidate_evidence(_P3, events, [], root)

        self.assertEqual(p0_evidence.path_coverage.eligible_n, 50)
        self.assertEqual(p3_evidence.path_coverage.eligible_n, 0)  # progress=0.9 excludes all from P3 (<0.85)

    def test_overlapping_events_do_not_double_count_proxy_coverage_across_candidates(self):
        """Same underlying events qualify for both P0 (no filter) and
        BASELINE-0 (progress<0.70); per-candidate observed_n must each
        reflect only that candidate's own eligible population, never a
        summed/global count."""
        events = _make_synthetic_events(n=20, n_days=5, progress=0.3)  # qualifies for both P0 and BASELINE-0
        proxy_rows = [{"token_address": e["token_address"], "status": "OK", "observed_at": e["alert_time"]}
                      for e in events]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            baseline_evidence = _query_candidate_evidence(_BASELINE, events, proxy_rows, root)
            p0_evidence = _query_candidate_evidence(_P0, events, proxy_rows, root)

        # Both see the same 20 observations against their own (here, identical) eligible sets --
        # neither is inflated by the other's count.
        self.assertEqual(baseline_evidence.execution_proxy_coverage.observed_n, 20)
        self.assertEqual(p0_evidence.execution_proxy_coverage.observed_n, 20)


if __name__ == "__main__":
    unittest.main()
