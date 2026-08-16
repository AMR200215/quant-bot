"""research/tests/test_v8_phase2_foundations.py — V8-FILTER-DERIVATION
Phase 2 (parallel, per P16 explicit authorization): feature enforcement,
candidate registry, grouped chronological split, experiment manifest.

Run: python -m pytest research/tests/test_v8_phase2_foundations.py -v
"""

import unittest

from research.v8_feature_enforcement import (
    FeatureNotAllowedError, FeatureNotRegisteredError,
    assert_features_allowed, check_features_allowed,
)
from research.v8_candidate_registry import CANDIDATES, MAX_CONDITIONS_PER_CANDIDATE, registry_hash
from research.v8_split import grouped_chronological_split
from research.v8_experiment_manifest import ExperimentManifest, build_manifest_from_current_state


class TestFeatureEnforcement(unittest.TestCase):

    def test_allowed_entry_features_pass(self):
        violations = check_features_allowed(
            ["progress_at_signal", "venue_state_at_signal"], "entry")
        self.assertEqual(violations, [])

    def test_post_trade_feature_rejected_for_entry(self):
        with self.assertRaises(FeatureNotAllowedError):
            assert_features_allowed(["pct_change_peak"], "entry")

    def test_smart_money_rejected_for_entry(self):
        with self.assertRaises(FeatureNotAllowedError):
            assert_features_allowed(["smart_money_hit"], "entry")

    def test_unregistered_feature_fails_closed(self):
        with self.assertRaises(FeatureNotRegisteredError):
            assert_features_allowed(["some_field_nobody_registered"], "entry")

    def test_dex_screener_field_rejected_for_entry(self):
        """T0+snapshot fields are not usable at V8's real fork point."""
        with self.assertRaises(FeatureNotAllowedError):
            assert_features_allowed(["price_change_5m"], "entry")

    def test_invalid_stage_raises(self):
        with self.assertRaises(ValueError):
            check_features_allowed(["progress_at_signal"], "not_a_real_stage")


class TestCandidateRegistry(unittest.TestCase):

    def test_baseline_zero_always_present(self):
        ids = {c["candidate_id"] for c in CANDIDATES}
        self.assertIn("BASELINE-0", ids)

    def test_no_candidate_exceeds_complexity_cap(self):
        for c in CANDIDATES:
            self.assertLessEqual(len(c["conditions"]), MAX_CONDITIONS_PER_CANDIDATE,
                                  f"{c['candidate_id']} exceeds the {MAX_CONDITIONS_PER_CANDIDATE}-condition cap")

    def test_every_candidate_only_uses_allowed_entry_features(self):
        """The registry itself must obey the enforcement module -- not
        just candidates built later at evaluation time."""
        for c in CANDIDATES:
            violations = check_features_allowed(c["required_features"], "entry")
            self.assertEqual(violations, [], f"{c['candidate_id']}: {violations}")

    def test_every_candidate_has_a_rationale(self):
        for c in CANDIDATES:
            self.assertTrue(c.get("rationale"), f"{c['candidate_id']} has no rationale")

    def test_registry_hash_deterministic(self):
        self.assertEqual(registry_hash(), registry_hash())

    def test_registry_hash_changes_if_candidates_change(self):
        import research.v8_candidate_registry as reg
        original_hash = reg.registry_hash()
        reg.CANDIDATES.append({"candidate_id": "TEST-TEMP", "conditions": []})
        try:
            self.assertNotEqual(reg.registry_hash(), original_hash)
        finally:
            reg.CANDIDATES.pop()

    def test_baseline_matches_v8_paper_gate_exactly(self):
        baseline = next(c for c in CANDIDATES if c["candidate_id"] == "BASELINE-0")
        progress_cond = next(c for c in baseline["conditions"] if c["feature"] == "progress_at_signal")
        self.assertEqual(progress_cond["value"], 0.70)


class TestGroupedChronologicalSplit(unittest.TestCase):

    def _rows(self, specs):
        # specs: list of (token_address, alert_time)
        return [{"token_address": t, "alert_time": a} for t, a in specs]

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            grouped_chronological_split([], lambda r: r["token_address"], lambda r: r["alert_time"])

    def test_invalid_fractions_raise(self):
        rows = self._rows([("A", 0)])
        with self.assertRaises(ValueError):
            grouped_chronological_split(rows, lambda r: r["token_address"], lambda r: r["alert_time"],
                                         train_frac=0.7, validation_frac=0.4)

    def test_basic_chronological_split_by_time_not_row_count(self):
        # 100 rows uniformly spread over time for token A (first 60% of
        # time), only 2 rows for token B in the last 20% -- a row-count
        # split would put almost everything in train; a TIME split must
        # not be dominated by row density.
        rows = self._rows([(f"A{i}", i) for i in range(100)]) + self._rows([("B1", 95), ("B2", 96)])
        result = grouped_chronological_split(rows, lambda r: r["token_address"], lambda r: r["alert_time"])
        train_addrs = {r["token_address"] for r in result.train}
        holdout_addrs = {r["token_address"] for r in result.holdout}
        self.assertIn("B1", holdout_addrs)
        self.assertNotIn("B1", train_addrs)

    def test_same_mint_never_appears_in_two_splits(self):
        # Token X alerted once early (train era) and again late (holdout
        # era) -- must land ENTIRELY in one split (its first-seen time).
        rows = self._rows([("X", 0), ("X", 99), ("Y", 50)])
        result = grouped_chronological_split(rows, lambda r: r["token_address"], lambda r: r["alert_time"],
                                              train_frac=0.5, validation_frac=0.25)
        splits_containing_x = sum(
            1 for split in (result.train, result.validation, result.holdout)
            if any(r["token_address"] == "X" for r in split)
        )
        self.assertEqual(splits_containing_x, 1)

    def test_ambiguous_groups_counted_not_silently_dropped(self):
        rows = self._rows([("X", 0), ("X", 99)])   # spans the whole range -> definitely ambiguous
        result = grouped_chronological_split(rows, lambda r: r["token_address"], lambda r: r["alert_time"],
                                              train_frac=0.5, validation_frac=0.25)
        self.assertGreaterEqual(result.ambiguous_groups, 1)
        # both rows still present somewhere -- not dropped
        total = len(result.train) + len(result.validation) + len(result.holdout)
        self.assertEqual(total, 2)

    def test_all_rows_preserved_across_splits(self):
        rows = self._rows([(f"T{i}", i) for i in range(50)])
        result = grouped_chronological_split(rows, lambda r: r["token_address"], lambda r: r["alert_time"])
        total = len(result.train) + len(result.validation) + len(result.holdout)
        self.assertEqual(total, len(rows))

    def test_group_count_matches_distinct_tokens(self):
        rows = self._rows([("A", 0), ("A", 1), ("B", 2), ("C", 3)])
        result = grouped_chronological_split(rows, lambda r: r["token_address"], lambda r: r["alert_time"],
                                              train_frac=0.5, validation_frac=0.25)
        self.assertEqual(result.group_count, 3)


class TestExperimentManifest(unittest.TestCase):

    def test_default_holdout_evaluated_is_false(self):
        m = ExperimentManifest()
        self.assertFalse(m.holdout_evaluated)

    def test_default_smart_money_version_is_none(self):
        """P15-8/FD6: must never default to reusing v1."""
        m = ExperimentManifest()
        self.assertIsNone(m.smart_money_version)

    def test_run_id_unique_per_instance(self):
        m1, m2 = ExperimentManifest(), ExperimentManifest()
        self.assertNotEqual(m1.run_id, m2.run_id)

    def test_build_from_current_state_populates_real_registry_hash(self):
        from research.v8_candidate_registry import registry_hash
        m = build_manifest_from_current_state()
        self.assertEqual(m.candidate_registry_hash, registry_hash())
        self.assertGreater(m.clean_cohort_version, 0)

    def test_to_dict_is_json_serialisable(self):
        import json
        m = build_manifest_from_current_state()
        json.dumps(m.to_dict())   # must not raise


if __name__ == "__main__":
    unittest.main()
