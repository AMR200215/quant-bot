"""
test_tracker_progress.py — PROGRESS-FIX PF13 regression tests #12, #15, and
a direct test of the event_id-persistence fix made alongside PF10.

  12. pp_snapshot_ok stays False for a default-only (all-zero) PP snapshot —
      direct unit test of the PF7 fix in research/tracker.py's
      _enrich_with_pp. Before the fix this was set unconditionally True
      whenever a snapshot dict existed at all, regardless of whether any
      field in it was a real observation.
  15. research/analysis/backfill_progress_from_paths.py (PF10) never calls
      any live RPC / present-day curve-state function — every recovered
      value must come from a token's own historical path file, not
      whatever the chain looks like today.
  (extra) event_id is now persisted for LIVE (non-backfilled) rows too, not
      only backfilled ones — this was a real gap found while building PF10:
      research_tokens rows were losing event_id after insert for live
      alerts, breaking any later Research<->capture cross-reference.

Run with:
    python -m unittest research/tests/test_tracker_progress.py
"""

import sys
import types
import unittest
from pathlib import Path

import importlib

if "research.config" not in sys.modules:
    _real_config = importlib.import_module("research.config")
    _config_stub = types.ModuleType("research.config")
    for _k, _v in vars(_real_config).items():
        setattr(_config_stub, _k, _v)
    sys.modules["research.config"] = _config_stub

import research.tracker as tracker  # noqa: E402


class TestPpSnapshotOkTruthful(unittest.TestCase):
    """PF13 #12 / PF7."""

    def test_default_only_snapshot_is_not_ok(self):
        """A snapshot dict that exists but carries only zero/default values
        (the exact PF1 race: ScreeningState created but no real tick ever
        arrived) must report pp_snapshot_ok=False."""
        snap = {}
        pp = {"pp_price": 0.0, "pp_vsol": 0.0, "pp_buys": 0, "pp_sells": 0,
              "pp_sol_in": 0.0, "sol_price": 0.0}
        out = tracker._enrich_with_pp(snap, pp)
        self.assertFalse(out["pp_snapshot_ok"])

    def test_empty_pp_dict_is_not_ok(self):
        out = tracker._enrich_with_pp({}, {})
        self.assertFalse(out["pp_snapshot_ok"])

    def test_real_price_observation_is_ok(self):
        snap = {}
        pp = {"pp_price": 0.000012, "pp_vsol": 0.0, "pp_buys": 0, "pp_sells": 0,
              "pp_sol_in": 0.0, "sol_price": 0.0}
        out = tracker._enrich_with_pp(snap, pp)
        self.assertTrue(out["pp_snapshot_ok"])

    def test_real_vsol_only_is_ok(self):
        """vsol alone (no price yet) still counts as a real observation."""
        snap = {}
        pp = {"pp_price": 0.0, "pp_vsol": 12.5, "pp_buys": 0, "pp_sells": 0,
              "pp_sol_in": 0.0, "sol_price": 0.0}
        out = tracker._enrich_with_pp(snap, pp)
        self.assertTrue(out["pp_snapshot_ok"])


class TestEventIdPersistedForLiveRows(unittest.TestCase):
    """Regression test for the gap found while building PF10: event_id was
    only ever added to the Supabase row for backfilled alerts."""

    def test_insert_source_stores_event_id_unconditionally(self):
        src = Path(tracker.__file__).read_text()
        # The unconditional assignment must exist outside the
        # `if alert.backfilled:` branch.
        self.assertIn('_pp_extras["event_id"] = alert.event_id', src,
            "event_id must be stored for every row, not only backfilled ones "
            "— otherwise live rows lose their event_id after insert, breaking "
            "any later cross-reference back to the capture that produced "
            "their progress_at_signal.")


class TestBackfillNeverUsesPresentDayState(unittest.TestCase):
    """PF13 #15 — PF10's historical recovery script must be pure path-file
    replay, never a live RPC / present-day curve lookup."""

    def test_backfill_script_imports_no_live_rpc_functions(self):
        import research.analysis.backfill_progress_from_paths as backfill
        src = Path(backfill.__file__).read_text()
        forbidden = [
            "get_curve_state_batch", "get_curve_prices_batch",
            "requests.post", "requests.get",
            "capture_progress_async", "wait_for_capture",
        ]
        hits = [f for f in forbidden if f in src]
        self.assertEqual(hits, [],
            f"backfill_progress_from_paths.py references live-state calls "
            f"it must never use for historical recovery: {hits}")

    def test_backfill_script_only_reads_path_files_for_ticks(self):
        import research.analysis.backfill_progress_from_paths as backfill
        src = Path(backfill.__file__).read_text()
        self.assertIn("load_path_file", src,
            "recovery must come from the token's own recorded path file")


if __name__ == "__main__":
    unittest.main()
