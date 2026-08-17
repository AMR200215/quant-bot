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
from research.v8_candidate_registry import (
    CANDIDATES, MAX_CONDITIONS_PER_CANDIDATE, registry_hash,
    assert_registry_frozen, CANDIDATE_REGISTRY_FROZEN_SHA256, CANDIDATE_REGISTRY_FROZEN_COUNT,
)
from research.v8_split import grouped_chronological_split
from research.v8_experiment_manifest import ExperimentManifest, build_manifest_from_current_state
from research.v8_clean_cohort import CANDIDATE0_FULL_GATE_HISTORICAL_N, CANDIDATE0_PROGRESS_HALF_IS_NOT_A_VALIDATION


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

    def test_no_duplicate_candidate_identity(self):
        """P2-2 regression: no two candidate_ids may encode the exact same
        rule (same conditions, order-independent). BASELINE-0 and P2's
        progress<0.70+CURVE_ACTIVE were previously counted twice."""
        def _canonical(conds):
            return frozenset((c["feature"], c["op"], c["value"]) for c in conds)

        seen = {}
        for c in CANDIDATES:
            key = _canonical(c["conditions"])
            self.assertNotIn(key, seen,
                              f"{c['candidate_id']} duplicates {seen.get(key)} -- same rule, two identities")
            seen[key] = c["candidate_id"]

    def test_no_separate_v8_p2_entry(self):
        ids = {c["candidate_id"] for c in CANDIDATES}
        self.assertNotIn("V8-P2", ids, "V8-P2 is identical to BASELINE-0 and must not be a separate entry")

    def test_no_unverified_velocity_threshold(self):
        """P2-3: channel_velocity_5m<=5 failed its reproducibility audit
        (real median=0, not ~5) and was removed, not retuned. No candidate
        may use channel_velocity_5m until a fresh, dated audit re-adds it."""
        for c in CANDIDATES:
            features = {cond["feature"] for cond in c["conditions"]}
            self.assertNotIn("channel_velocity_5m", features,
                              f"{c['candidate_id']} uses channel_velocity_5m without a passing audit")
        ids = {c["candidate_id"] for c in CANDIDATES}
        self.assertNotIn("V8-P2-LOWVEL", ids)
        self.assertNotIn("V8-P3-LOWVEL", ids)

    def test_p2_4_frozen_v1_registry_matches_live_state(self):
        """P2-4: the frozen v1 hash/count must match the live CANDIDATES
        exactly right now. If this fails, CANDIDATES drifted after the
        freeze and needs an explicit experiment v2, not a quiet edit."""
        self.assertEqual(len(CANDIDATES), CANDIDATE_REGISTRY_FROZEN_COUNT)
        self.assertEqual(registry_hash(), CANDIDATE_REGISTRY_FROZEN_SHA256)
        assert_registry_frozen()  # must not raise

    def test_p2_4_frozen_v1_registry_has_exactly_the_expected_core(self):
        ids = {c["candidate_id"] for c in CANDIDATES}
        self.assertEqual(ids, {"BASELINE-0", "V8-P0", "V8-P1", "V8-P3"})

    def test_assert_registry_frozen_raises_on_drift(self):
        import research.v8_candidate_registry as reg
        reg.CANDIDATES.append({"candidate_id": "TEST-DRIFT", "conditions": []})
        try:
            with self.assertRaises(RuntimeError):
                reg.assert_registry_frozen()
        finally:
            reg.CANDIDATES.pop()


class TestP2_10_ProgressHalfCohortNotAValidation(unittest.TestCase):

    def test_full_gate_historical_n_stays_unknown(self):
        """venue_state_at_signal is not persisted historically and must
        never be approximated from dex_id -- so the FULL gate's historical
        n must stay None, not silently backfilled."""
        self.assertIsNone(CANDIDATE0_FULL_GATE_HISTORICAL_N)

    def test_progress_half_flagged_as_not_a_validation(self):
        self.assertTrue(CANDIDATE0_PROGRESS_HALF_IS_NOT_A_VALIDATION)


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
        # era) -- P2-1: must NOT land in train (that would move the later,
        # holdout-era observation backward -- temporal leakage). The whole
        # group must be purged from every split instead.
        rows = self._rows([("X", 0), ("X", 99), ("Y", 50)])
        result = grouped_chronological_split(rows, lambda r: r["token_address"], lambda r: r["alert_time"],
                                              train_frac=0.5, validation_frac=0.25)
        splits_containing_x = sum(
            1 for split in (result.train, result.validation, result.holdout)
            if any(r["token_address"] == "X" for r in split)
        )
        self.assertEqual(splits_containing_x, 0)
        self.assertNotIn("X", {r["token_address"] for r in result.train},
                          "X's later (holdout-era) row must never move a train-era row's group into train")

    def test_boundary_spanning_group_purged_not_silently_dropped(self):
        rows = self._rows([("X", 0), ("X", 99)])   # spans the whole range -> boundary-spanning
        result = grouped_chronological_split(rows, lambda r: r["token_address"], lambda r: r["alert_time"],
                                              train_frac=0.5, validation_frac=0.25)
        self.assertGreaterEqual(result.boundary_spanning_groups, 1)
        # both rows excluded (purged), counted, not silently vanished
        total = len(result.train) + len(result.validation) + len(result.holdout)
        self.assertEqual(total, 0)
        self.assertEqual(result.purged_rows, 2)

    def test_regression_early_train_row_and_later_holdout_row_both_purged(self):
        """P2-1 exact regression case: same mint, one row naturally in the
        train era, one row naturally in the holdout era. Both rows must be
        purged; NEITHER may end up in train (the original bug moved both
        into train via the group's first-seen timestamp)."""
        rows = self._rows(
            [(f"FILLER{i}", i) for i in range(1, 90)]  # spread filler across the range so cutoffs are meaningful
            + [("MINT_X", 2), ("MINT_X", 95)]           # 2 -> train era, 95 -> holdout era
        )
        result = grouped_chronological_split(rows, lambda r: r["token_address"], lambda r: r["alert_time"],
                                              train_frac=0.6, validation_frac=0.2)
        all_addrs_by_split = {
            "train": {r["token_address"] for r in result.train},
            "validation": {r["token_address"] for r in result.validation},
            "holdout": {r["token_address"] for r in result.holdout},
        }
        self.assertNotIn("MINT_X", all_addrs_by_split["train"])
        self.assertNotIn("MINT_X", all_addrs_by_split["validation"])
        self.assertNotIn("MINT_X", all_addrs_by_split["holdout"])
        self.assertGreaterEqual(result.boundary_spanning_groups, 1)

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
