"""research/tests/test_v8_final_state.py — V8 DATA RECOVERY batch item
11: the one final state machine.

Run: python -m pytest research/tests/test_v8_final_state.py -v
"""

import unittest

from research.v8_final_state import (
    check_engine_ready, check_forward_pipeline_healthy,
    RECENT_INVALID_RATE_UNHEALTHY_THRESHOLD_PCT, MIN_RECENT_LIVE_ROWS_TO_JUDGE,
)


class TestEngineReady(unittest.TestCase):

    def test_engine_ready_true_against_real_frozen_registries(self):
        """The frozen v1 registries this whole batch built on top of
        must still pass their own freeze checks."""
        result = check_engine_ready()
        self.assertTrue(result.entry_registry_ok)
        self.assertTrue(result.exit_registry_ok)
        self.assertTrue(result.holdout_lock_ok)
        self.assertTrue(result.statistical_module_ok)
        self.assertTrue(result.ready)
        self.assertEqual(result.reasons, [])


class TestForwardPipelineHealthy(unittest.TestCase):

    def test_unknown_status_never_silently_true(self):
        """With no path files present (a fresh/empty repo_root), status
        must be UNKNOWN, not HEALTHY -- never assume health from
        silence."""
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            result = check_forward_pipeline_healthy(repo_root=Path(d))
        self.assertEqual(result.status, "UNKNOWN")
        self.assertEqual(result.recent_live_rows_checked, 0)
        self.assertIsNone(result.recent_invalid_rate_pct)

    def test_threshold_is_far_below_the_prefix_corruption_rate(self):
        """Sanity-anchor: the unhealthy threshold must be well below
        Phase 2.1's measured 51.5% pre-fix invalid rate, so a real
        recurrence would trip it immediately."""
        self.assertLess(RECENT_INVALID_RATE_UNHEALTHY_THRESHOLD_PCT, 51.5)

    def test_min_rows_to_judge_is_a_real_floor_not_zero(self):
        self.assertGreater(MIN_RECENT_LIVE_ROWS_TO_JUDGE, 0)


if __name__ == "__main__":
    unittest.main()
