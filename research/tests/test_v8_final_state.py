"""research/tests/test_v8_final_state.py — V8 DATA RECOVERY batch item
11, CORRECTED by the V8 READINESS MONITOR CORRECTION batch: the one
final state machine, component-based pipeline health.

Run: python -m pytest research/tests/test_v8_final_state.py -v
"""

import unittest
from pathlib import Path

from research.v8_final_state import (
    check_engine_ready, check_forward_pipeline_healthy, check_live_pp_paths_flow,
    check_path_integrity_quality, check_execution_proxy_flow, _combine_component_statuses,
    ComponentHealth, RECENT_INVALID_RATE_UNHEALTHY_THRESHOLD_PCT, MIN_RECENT_ROWS_TO_JUDGE,
)


class TestEngineReady(unittest.TestCase):

    def test_engine_ready_true_against_real_frozen_registries(self):
        result = check_engine_ready()
        self.assertTrue(result.entry_registry_ok)
        self.assertTrue(result.exit_registry_ok)
        self.assertTrue(result.holdout_lock_ok)
        self.assertTrue(result.statistical_module_ok)
        self.assertTrue(result.ready)
        self.assertEqual(result.reasons, [])


class TestCombineComponentStatuses(unittest.TestCase):
    """Item 4's exact aggregation rule."""

    def _c(self, status):
        return ComponentHealth(name="x", status=status, n_checked=10, detail="", note="")

    def test_all_healthy_is_healthy(self):
        self.assertEqual(_combine_component_statuses([self._c("HEALTHY")] * 5), "HEALTHY")

    def test_any_unhealthy_dominates(self):
        components = [self._c("HEALTHY")] * 4 + [self._c("UNHEALTHY")]
        self.assertEqual(_combine_component_statuses(components), "UNHEALTHY")

    def test_unknown_without_unhealthy_gives_unknown(self):
        components = [self._c("HEALTHY")] * 4 + [self._c("UNKNOWN")]
        self.assertEqual(_combine_component_statuses(components), "UNKNOWN")

    def test_unhealthy_beats_unknown(self):
        components = [self._c("UNKNOWN"), self._c("UNHEALTHY"), self._c("HEALTHY")]
        self.assertEqual(_combine_component_statuses(components), "UNHEALTHY")

    def test_a_single_unknown_component_prevents_overall_healthy(self):
        """The exact requirement: a silent/dead execution-proxy collector
        must prevent HEALTHY even if every price/path component looks
        fine."""
        components = [self._c("HEALTHY"), self._c("HEALTHY"), self._c("HEALTHY"),
                      self._c("HEALTHY"), self._c("UNKNOWN")]
        self.assertNotEqual(_combine_component_statuses(components), "HEALTHY")


class TestLivePPPathsFlow(unittest.TestCase):

    def test_unknown_with_no_data(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            result = check_live_pp_paths_flow(Path(d))
        self.assertEqual(result.status, "UNKNOWN")
        self.assertEqual(result.n_checked, 0)


class TestPathIntegrityQuality(unittest.TestCase):

    def test_unknown_with_no_data(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            result = check_path_integrity_quality(Path(d))
        self.assertEqual(result.status, "UNKNOWN")

    def test_threshold_far_below_prefix_corruption_rate(self):
        self.assertLess(RECENT_INVALID_RATE_UNHEALTHY_THRESHOLD_PCT, 51.5)

    def test_unhealthy_on_demonstrated_post_fix_corruption(self):
        """Self-audit item 7: pipeline health must become UNHEALTHY if
        the same price-corruption pattern Phase 2.1 found ever recurs --
        proven with a synthetic corpus mostly full of the real corrupted
        pattern (vsol/vtok implying ~$73B mcap), not just asserted."""
        import csv
        import tempfile
        from datetime import datetime, timezone
        from research.path_schema import PATH_HEADER

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            day_dir = root / "logs" / "research_paths" / today
            day_dir.mkdir(parents=True)
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

            def _row(ts_ms, corrupted):
                base = {
                    "schema_version": "2", "ts_ms": str(ts_ms),
                    "price_usd": "73.49" if corrupted else "0.00005",
                    "price_sol": "0.42" if corrupted else "0.0000003",
                    "vsol": "116.27" if corrupted else "50.0",
                    "vtok": "279900000", "venue_state": "CURVE_ACTIVE",
                    "source": "live_pp", "backfilled": "false", "data_status": "ok",
                }
                return base

            with open(day_dir / "MINT_CORRUPT.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=PATH_HEADER)
                w.writeheader()
                for i in range(10):
                    full = {k: "" for k in PATH_HEADER}
                    full.update(_row(now_ms + i, corrupted=(i < 9)))  # 9/10 corrupted
                    w.writerow(full)

            result = check_path_integrity_quality(root)

        self.assertEqual(result.status, "UNHEALTHY")
        self.assertGreaterEqual(result.n_checked, MIN_RECENT_ROWS_TO_JUDGE)


class TestExecutionProxyFlow(unittest.TestCase):

    def test_unknown_when_collector_silent(self):
        """A silently-dead collector (empty/missing log) must report
        UNKNOWN, never HEALTHY from silence."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            result = check_execution_proxy_flow(Path(d))
        self.assertEqual(result.status, "UNKNOWN")
        self.assertIn("silently dead", result.note.lower())


class TestCheckForwardPipelineHealthy(unittest.TestCase):

    def test_no_supabase_client_gives_unknown_for_those_components_not_a_crash(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            result = check_forward_pipeline_healthy(sb=None, repo_root=Path(d))
        self.assertEqual(result.status, "UNKNOWN")
        names = {c.name for c in result.components}
        self.assertIn("progress_at_signal_flow", names)
        self.assertIn("venue_state_at_signal_flow", names)
        self.assertIn("live_pp_paths_flow", names)
        self.assertIn("path_integrity_quality", names)
        self.assertIn("execution_proxy_flow", names)

    def test_five_independent_components_checked(self):
        """Item 4's core requirement: all five named streams must each
        get their own component, not one merged check."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            result = check_forward_pipeline_healthy(sb=None, repo_root=Path(d))
        self.assertEqual(len(result.components), 5)


class TestMinRowsFloor(unittest.TestCase):
    def test_min_rows_to_judge_is_a_real_floor_not_zero(self):
        self.assertGreater(MIN_RECENT_ROWS_TO_JUDGE, 0)


if __name__ == "__main__":
    unittest.main()
