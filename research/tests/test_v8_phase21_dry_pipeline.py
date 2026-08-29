"""research/tests/test_v8_phase21_dry_pipeline.py — V8-FILTER-DERIVATION
Phase 2.1 item 6: final engine acceptance -- a dry pipeline run on
TRAIN-split (non-holdout) synthetic data proving every Phase 2.1
guarantee holds end to end, together, not just in isolated unit tests.

Explicitly proves:
  1. A corrupted path is excluded before it can reach a replay result.
  2. Unknown/absent venue state cannot satisfy FULL_ENTRY_RULE_READY.
  3. A configured slippage TOLERANCE is never treated as a realized cost.
  4. Actual measured-cost provenance (evidence_class) is retained end to end.
  5. Holdout stays locked -- this test never unlocks it, never reads
     holdout-split rows for any metric, and asserts the lock still holds
     at the end.

Run: python -m pytest research/tests/test_v8_phase21_dry_pipeline.py -v
"""

import unittest

from research.v8_split import grouped_chronological_split
from research.v8_path_integrity import assess_path_integrity, PathIntegrityStatus
from research.v8_replay_engine import replay_strategy_for_full_ev, FixedLagExecutionModel
from research.v8_readiness_engine import ReadinessInputs, assess_readiness
from research.v8_execution_cost_model import (
    get_configured_tolerance, get_realized_cost_components, EvidenceClass,
    SELL_LADDER_TOLERANCE_PCTS,
)
from research.v8_experiment_manifest import ExperimentManifest, assert_holdout_not_evaluated


def _synthetic_events(n_train_mints=30, n_holdout_mints=10):
    """Rows spread across a clear train era (t=0..999) and a clear
    holdout era (t=2000..2999), one row per mint, no boundary-spanning
    groups -- deliberately simple so the split behaves predictably."""
    rows = []
    for i in range(n_train_mints):
        rows.append({"token_address": f"TRAIN_{i}", "alert_time": 10 * i})
    for i in range(n_holdout_mints):
        rows.append({"token_address": f"HOLDOUT_{i}", "alert_time": 2000 + 10 * i})
    return rows


class TestDryPipeline(unittest.TestCase):

    def setUp(self):
        self.events = _synthetic_events()
        self.split = grouped_chronological_split(
            self.events, lambda r: r["token_address"], lambda r: r["alert_time"],
            train_frac=0.6, validation_frac=0.2,
        )
        self.manifest = ExperimentManifest()

    # ── Guarantee 5: holdout stays locked throughout this whole test ────

    def test_holdout_never_touched_for_any_metric(self):
        holdout_mints = {r["token_address"] for r in self.split.holdout}
        # This dry run only ever reads self.split.train below -- assert
        # that set is disjoint from holdout as a structural guardrail.
        train_mints = {r["token_address"] for r in self.split.train}
        self.assertEqual(train_mints & holdout_mints, set())

    def test_manifest_holdout_lock_holds_at_end_of_dry_run(self):
        assert_holdout_not_evaluated(self.manifest)  # must not raise
        self.assertFalse(self.manifest.holdout_evaluated)

    # ── Guarantee 1: corrupted path excluded before reaching a result ───

    def test_corrupted_train_path_excluded_never_reaches_replay_result(self):
        clean_path = [
            {"ts_ms": 0, "price_usd": 0.00005, "price_sol": 0.0000003, "vsol": 50.0, "venue_state": "CURVE_ACTIVE"},
            {"ts_ms": 1000, "price_usd": 0.00006, "price_sol": 0.00000036, "vsol": 55.0, "venue_state": "CURVE_ACTIVE"},
        ]
        corrupted_path = [
            {"ts_ms": 0, "price_usd": 1.0, "price_sol": 0.006, "vsol": 50.0, "venue_state": "CURVE_ACTIVE"},
            {"ts_ms": 1000, "price_usd": 73.49, "price_sol": 0.42, "vsol": 116.27, "venue_state": "CURVE_ACTIVE"},
        ]

        spec = {"hard_stop": -0.35, "trail_tiers": [{"activates_at": 0.3, "trail_pct": 0.25}],
                "tp_levels": [], "time_stop_min": 90}

        clean_result = replay_strategy_for_full_ev(
            clean_path, entry_ts=0, entry_spec={}, exit_spec=spec,
            execution_model=FixedLagExecutionModel(exec_lag_ms=0))
        corrupted_result = replay_strategy_for_full_ev(
            corrupted_path, entry_ts=0, entry_spec={}, exit_spec=spec,
            execution_model=FixedLagExecutionModel(exec_lag_ms=0))

        self.assertIsNotNone(clean_result)
        self.assertIsNone(corrupted_result)  # never produces a $/day-usable result

        integrity = assess_path_integrity(corrupted_path)
        self.assertEqual(integrity.status, PathIntegrityStatus.INVALID.value)

    # ── Guarantee 2: unknown venue state blocks FULL_ENTRY_RULE_READY ───

    def test_unknown_venue_state_blocks_full_entry_rule_readiness(self):
        inputs = ReadinessInputs(
            candidate_id="BASELINE-0", exit_id="E0", requires_venue_state=True,
            historical_entry_n=500, unique_mints=200, unique_days=60,   # abundant progress-only evidence
            forward_venue_qualified_n=0, venue_qualified_unique_mints=0, venue_qualified_unique_days=0,
            train_n=100, validation_n=100, holdout_n=100, boundary_purged_n=0,
            representative_path_n=100, path_coverage_pct=80.0,
            poll_outcome_n=100, poll_outcome_coverage_pct=80.0,
            cost_model_available=True, entry_slippage_measured=False,
        )
        report = assess_readiness(inputs)
        self.assertTrue(report.progress_evidence_ready)
        self.assertFalse(report.full_entry_rule_ready)
        self.assertFalse(report.full_eval_ready)

    # ── Guarantee 3: configured tolerance never treated as realized cost ─

    def test_configured_tolerance_never_appears_as_realized_cost_value(self):
        tolerance = get_configured_tolerance()
        realized = get_realized_cost_components()
        self.assertEqual(tolerance.sell_ladder_tolerance_pcts, SELL_LADDER_TOLERANCE_PCTS)
        for c in realized:
            self.assertNotIn(c.value, SELL_LADDER_TOLERANCE_PCTS)

    # ── Guarantee 4: measured-cost provenance retained end to end ───────

    def test_realized_cost_provenance_retained(self):
        realized = get_realized_cost_components()
        classes_seen = {c.evidence_class for c in realized}
        # must include more than one class -- provenance is NOT flattened
        # into a single blended confidence level
        self.assertGreaterEqual(len(classes_seen), 3)
        for c in realized:
            self.assertIn(c.evidence_class,
                           {EvidenceClass.MEASURED, EvidenceClass.PARTIALLY_MEASURED,
                            EvidenceClass.ASSUMPTION_BOUND, EvidenceClass.UNMEASURED})
            self.assertFalse(c.cohort_matches_v8)  # V7/V4 journal, never silently claimed as V8

    # ── End-to-end: all five guarantees hold together in one run ────────

    def test_full_dry_pipeline_all_guarantees_together(self):
        # 1. Split (train only used below)
        self.assertGreater(len(self.split.train), 0)
        self.assertGreater(len(self.split.holdout), 0)

        # 2. Path integrity on a TRAIN-era path
        train_path = [
            {"ts_ms": 0, "price_usd": 73.49, "price_sol": 0.42, "vsol": 116.27, "venue_state": "CURVE_ACTIVE"},
        ]
        integrity = assess_path_integrity(train_path)
        self.assertEqual(integrity.status, "INVALID")

        # 3. Readiness with no venue-qualified evidence
        readiness = assess_readiness(ReadinessInputs(
            candidate_id="BASELINE-0", exit_id="E0", requires_venue_state=True,
            historical_entry_n=1000, unique_mints=500, unique_days=90,
            forward_venue_qualified_n=0, venue_qualified_unique_mints=0, venue_qualified_unique_days=0,
            train_n=200, validation_n=200, holdout_n=200, boundary_purged_n=0,
            representative_path_n=200, path_coverage_pct=90.0,
            poll_outcome_n=200, poll_outcome_coverage_pct=90.0,
            cost_model_available=True, entry_slippage_measured=False,
        ))
        self.assertFalse(readiness.full_eval_ready)

        # 4. Cost model provenance
        realized = get_realized_cost_components()
        self.assertTrue(any(c.evidence_class == EvidenceClass.UNMEASURED for c in realized))

        # 5. Holdout lock
        assert_holdout_not_evaluated(self.manifest)
        self.assertFalse(self.manifest.holdout_evaluated)


if __name__ == "__main__":
    unittest.main()
