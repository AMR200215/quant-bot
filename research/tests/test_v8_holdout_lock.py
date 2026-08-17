"""research/tests/test_v8_holdout_lock.py — V8-FILTER-DERIVATION
Phase 2 (P2-12): the enforceable holdout lock.

Run: python -m pytest research/tests/test_v8_holdout_lock.py -v
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from research.v8_experiment_manifest import (
    ExperimentManifest, build_manifest_from_current_state,
    PHASE3_HOLDOUT_UNLOCK_SENTINEL,
    assert_holdout_not_evaluated, unlock_holdout_for_phase3,
    check_v1_not_invalidated,
    HoldoutStillLockedViolation, ExperimentV1InvalidatedError,
)


class TestManifestProvenanceFields(unittest.TestCase):

    def test_exit_registry_hash_populated(self):
        m = build_manifest_from_current_state()
        self.assertTrue(m.exit_registry_hash)

    def test_execution_cost_model_version_populated(self):
        m = build_manifest_from_current_state()
        self.assertEqual(m.execution_cost_model_version, 1)

    def test_feature_registry_hash_populated_and_deterministic(self):
        m1 = build_manifest_from_current_state()
        m2 = build_manifest_from_current_state()
        self.assertTrue(m1.feature_registry_hash)
        self.assertEqual(m1.feature_registry_hash, m2.feature_registry_hash)


class TestAssertHoldoutNotEvaluated(unittest.TestCase):

    def test_passes_on_fresh_manifest(self):
        m = ExperimentManifest()
        assert_holdout_not_evaluated(m)  # must not raise

    def test_raises_once_holdout_evaluated(self):
        m = ExperimentManifest()
        unlock_holdout_for_phase3(m, PHASE3_HOLDOUT_UNLOCK_SENTINEL)
        with self.assertRaises(HoldoutStillLockedViolation):
            assert_holdout_not_evaluated(m)


class TestUnlockHoldoutForPhase3(unittest.TestCase):

    def test_wrong_confirmation_string_refused(self):
        m = ExperimentManifest()
        with self.assertRaises(ValueError):
            unlock_holdout_for_phase3(m, "yes please")
        self.assertFalse(m.holdout_evaluated)

    def test_missing_confirmation_refused(self):
        m = ExperimentManifest()
        with self.assertRaises(TypeError):
            unlock_holdout_for_phase3(m)  # confirmation is required, no default

    def test_correct_sentinel_unlocks(self):
        m = ExperimentManifest()
        unlock_holdout_for_phase3(m, PHASE3_HOLDOUT_UNLOCK_SENTINEL)
        self.assertTrue(m.holdout_evaluated)

    def test_double_unlock_same_manifest_refused(self):
        m = ExperimentManifest()
        unlock_holdout_for_phase3(m, PHASE3_HOLDOUT_UNLOCK_SENTINEL)
        with self.assertRaises(RuntimeError):
            unlock_holdout_for_phase3(m, PHASE3_HOLDOUT_UNLOCK_SENTINEL)


class TestCheckV1NotInvalidated(unittest.TestCase):

    def setUp(self):
        self.tmp_root = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def _write_manifest(self, run_id, holdout_evaluated, candidate_hash, exit_hash=None):
        out_dir = self.tmp_root / "logs" / "research_reports" / "v8_filter_selection" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(json.dumps({
            "run_id": run_id,
            "holdout_evaluated": holdout_evaluated,
            "candidate_registry_hash": candidate_hash,
            "exit_registry_hash": exit_hash,
        }))

    def test_no_manifests_at_all_is_fine(self):
        check_v1_not_invalidated(repo_root=self.tmp_root)  # must not raise

    def test_non_holdout_manifest_ignored_even_with_wrong_hash(self):
        self._write_manifest("run-a", holdout_evaluated=False, candidate_hash="stale-hash-doesnt-matter")
        check_v1_not_invalidated(repo_root=self.tmp_root)  # must not raise -- holdout never evaluated

    def test_holdout_manifest_with_matching_live_hash_passes(self):
        from research.v8_candidate_registry import registry_hash as live_hash
        from research.v8_exit_registry import registry_hash as live_exit_hash
        self._write_manifest("run-b", holdout_evaluated=True,
                              candidate_hash=live_hash(), exit_hash=live_exit_hash())
        check_v1_not_invalidated(repo_root=self.tmp_root)  # must not raise

    def test_holdout_manifest_with_stale_candidate_hash_invalidates(self):
        self._write_manifest("run-c", holdout_evaluated=True, candidate_hash="THIS-IS-NOW-STALE")
        with self.assertRaises(ExperimentV1InvalidatedError):
            check_v1_not_invalidated(repo_root=self.tmp_root)

    def test_holdout_manifest_with_stale_exit_hash_invalidates(self):
        from research.v8_candidate_registry import registry_hash as live_hash
        self._write_manifest("run-d", holdout_evaluated=True,
                              candidate_hash=live_hash(), exit_hash="THIS-IS-NOW-STALE")
        with self.assertRaises(ExperimentV1InvalidatedError):
            check_v1_not_invalidated(repo_root=self.tmp_root)


if __name__ == "__main__":
    unittest.main()
