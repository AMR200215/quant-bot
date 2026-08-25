"""research/tests/test_v8_replay_engine_refactor.py — V8-FILTER-DERIVATION
Phase 2 (P2-6/FD14): reusable replay-engine interface + exit registry.

Run: python -m pytest research/tests/test_v8_replay_engine_refactor.py -v
"""

import unittest

from research.v8_replay_engine import (
    replay_strategy, replay_strategy_for_full_ev, FixedLagExecutionModel, ReplayResult,
)
from research.analysis.replay_exits import _replay_one, _V7_SPEC
from research.v8_exit_registry import (
    EXIT_CANDIDATES, MIDTRADE_CANDIDATES, MIDTRADE_STATUS,
    registry_hash, assert_registry_frozen,
    EXIT_REGISTRY_FROZEN_SHA256, EXIT_REGISTRY_FROZEN_EXIT_COUNT, EXIT_REGISTRY_FROZEN_MIDTRADE_COUNT,
)

_SPEC = {
    "hard_stop": -0.35,
    "trail_tiers": [
        {"activates_at": 0.30, "trail_pct": 0.25},
        {"activates_at": 1.00, "trail_pct": 0.25},
        {"activates_at": 3.00, "trail_pct": 0.15},
    ],
    "tp_levels": [],
    "time_stop_min": 90,
    "time_stop_min_gain": 0.30,
    "profit_lock_min_gain": 0.40,
    "profit_lock_max_gain": 1.00,
    "profit_lock_stall_sec": 60,
}


class TestReplayEngineInterface(unittest.TestCase):

    def test_too_short_path_returns_none(self):
        self.assertIsNone(replay_strategy(
            [{"ts_ms": 0, "price_usd": 1.0}], entry_ts=0, entry_spec={},
            exit_spec=_SPEC, execution_model=FixedLagExecutionModel()))

    def test_zero_entry_price_returns_none(self):
        rows = [{"ts_ms": 0, "price_usd": 0.0}, {"ts_ms": 1000, "price_usd": 1.0}]
        self.assertIsNone(replay_strategy(
            rows, entry_ts=0, entry_spec={}, exit_spec=_SPEC,
            execution_model=FixedLagExecutionModel()))

    def test_hard_stop_triggers(self):
        rows = [{"ts_ms": 0, "price_usd": 1.0}, {"ts_ms": 1000, "price_usd": 0.60}]
        r = replay_strategy(rows, entry_ts=0, entry_spec={}, exit_spec=_SPEC,
                             execution_model=FixedLagExecutionModel(exec_lag_ms=0))
        self.assertEqual(r.exit_reason, "hard_stop")
        self.assertIsInstance(r, ReplayResult)

    def test_path_end_exit_when_nothing_triggers(self):
        rows = [{"ts_ms": 0, "price_usd": 1.0}, {"ts_ms": 1000, "price_usd": 1.05}]
        r = replay_strategy(rows, entry_ts=0, entry_spec={}, exit_spec=_SPEC,
                             execution_model=FixedLagExecutionModel(exec_lag_ms=0))
        self.assertEqual(r.exit_reason, "path_end")

    def test_entry_ts_is_never_assumed_to_be_rows_zero(self):
        """P2-6: the interface must key off the given entry_ts, not rows[0] --
        even if rows[0] is a much earlier/different-priced tick."""
        rows = [
            {"ts_ms": 0, "price_usd": 9.0},      # NOT the entry
            {"ts_ms": 1000, "price_usd": 1.0},    # real entry
            {"ts_ms": 2000, "price_usd": 1.05},
        ]
        r = replay_strategy(rows, entry_ts=1000, entry_spec={}, exit_spec=_SPEC,
                             execution_model=FixedLagExecutionModel(exec_lag_ms=0))
        self.assertEqual(r.entry_price, 1.0)
        self.assertEqual(r.entry_ts_ms, 1000)

    def test_tp_ladder_partial_exits_weighted_average(self):
        spec = dict(_SPEC)
        spec["tp_levels"] = [(0.50, 0.5), (1.00, 1.0)]  # 50% out at +50%, rest at +100%
        rows = [
            {"ts_ms": 0, "price_usd": 1.0},
            {"ts_ms": 1000, "price_usd": 1.50},
            {"ts_ms": 2000, "price_usd": 2.00},
        ]
        r = replay_strategy(rows, entry_ts=0, entry_spec={}, exit_spec=spec,
                             execution_model=FixedLagExecutionModel(exec_lag_ms=0))
        # Inherited from the pre-refactor engine (preserved, not "fixed" --
        # P2-6 is a refactor, not a behavior change): when the TP ladder's
        # last leg fully closes the position, a zero-fraction "_final"
        # marker is appended on top of the real fill, so partial_exits
        # counts 3 legs for 2 real fills. It doesn't affect PnL (its
        # fraction is 0, contributes nothing to the weighted average).
        self.assertEqual(r.partial_exits, 3)
        # weighted avg of 1.5*0.5 + 2.0*0.5 (+ 2.0*0) = 1.75 -> pnl=+75%
        self.assertAlmostEqual(r.pnl_pct, 75.0, places=1)


class TestReplayStrategyForFullEV(unittest.TestCase):
    """Phase 2.1 item 2b: the mandatory path-integrity choke point."""

    def test_corrupted_path_never_produces_a_result_through_full_ev_entrypoint(self):
        corrupted_rows = [
            {"ts_ms": 0, "price_usd": 1.0, "price_sol": 0.006, "vsol": 50.0, "venue_state": "CURVE_ACTIVE"},
            {"ts_ms": 1000, "price_usd": 73.49292385903, "price_sol": 0.419959564909,
             "vsol": 116.26907755036314, "venue_state": "CURVE_ACTIVE"},
        ]
        result = replay_strategy_for_full_ev(
            corrupted_rows, entry_ts=0, entry_spec={}, exit_spec=_SPEC,
            execution_model=FixedLagExecutionModel(exec_lag_ms=0))
        self.assertIsNone(result)

    def test_clean_path_still_produces_a_result_through_full_ev_entrypoint(self):
        clean_rows = [
            {"ts_ms": 0, "price_usd": 0.00005, "price_sol": 0.0000003, "vsol": 50.0, "venue_state": "CURVE_ACTIVE"},
            {"ts_ms": 1000, "price_usd": 0.00006, "price_sol": 0.00000036, "vsol": 55.0, "venue_state": "CURVE_ACTIVE"},
        ]
        result = replay_strategy_for_full_ev(
            clean_rows, entry_ts=0, entry_spec={}, exit_spec=_SPEC,
            execution_model=FixedLagExecutionModel(exec_lag_ms=0))
        self.assertIsNotNone(result)


class TestReplayExitsWrapperUnchanged(unittest.TestCase):
    """P2-6: replay_exits.py's _replay_one must produce output identical in
    shape and value to what it produced before the refactor -- verified by
    driving both the dict wrapper and the new engine on the same input and
    comparing every field."""

    def test_wrapper_matches_direct_engine_call(self):
        rows = [
            {"ts_ms": 0, "price_usd": 1.0},
            {"ts_ms": 1000, "price_usd": 1.10},
            {"ts_ms": 2000, "price_usd": 0.60},
        ]
        wrapped = _replay_one(rows, _V7_SPEC, exec_lag_ms=0)
        direct = replay_strategy(rows, entry_ts=rows[0]["ts_ms"], entry_spec={},
                                  exit_spec=_V7_SPEC, execution_model=FixedLagExecutionModel(exec_lag_ms=0))
        self.assertEqual(wrapped["exit_reason"], direct.exit_reason)
        self.assertEqual(wrapped["pnl_pct"], direct.pnl_pct)
        self.assertEqual(wrapped["hold_time_s"], direct.hold_time_s)
        self.assertEqual(wrapped["partial_exits"], direct.partial_exits)
        self.assertEqual(wrapped["exit_price"], direct.exit_price)

    def test_wrapper_returns_dict_not_dataclass(self):
        rows = [{"ts_ms": 0, "price_usd": 1.0}, {"ts_ms": 1000, "price_usd": 1.05}]
        result = _replay_one(rows, _V7_SPEC, exec_lag_ms=0)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()),
                          {"exit_price", "exit_reason", "pnl_pct", "hold_time_s", "partial_exits"})


class TestExitRegistry(unittest.TestCase):

    def test_four_exit_candidates(self):
        # Experiment v2 (2026-08-26): E3 added, see research/v8_exit_registry.py's
        # EXPERIMENT V2 docstring for the derivation.
        self.assertEqual(len(EXIT_CANDIDATES), 4)
        ids = {c["exit_id"] for c in EXIT_CANDIDATES}
        self.assertEqual(ids, {"E0", "E1", "E2", "E3"})

    def test_no_midtrade_rule_is_explicit_valid_status(self):
        self.assertEqual(MIDTRADE_CANDIDATES, [])
        self.assertEqual(MIDTRADE_STATUS, "NO_MIDTRADE_RULE_SUPPORTED")

    def test_registry_hash_deterministic(self):
        self.assertEqual(registry_hash(), registry_hash())

    def test_frozen_hash_matches_live_state(self):
        self.assertEqual(len(EXIT_CANDIDATES), EXIT_REGISTRY_FROZEN_EXIT_COUNT)
        self.assertEqual(len(MIDTRADE_CANDIDATES), EXIT_REGISTRY_FROZEN_MIDTRADE_COUNT)
        self.assertEqual(registry_hash(), EXIT_REGISTRY_FROZEN_SHA256)
        assert_registry_frozen()  # must not raise

    def test_assert_registry_frozen_raises_on_drift(self):
        import research.v8_exit_registry as reg
        reg.EXIT_CANDIDATES.append({"exit_id": "TEST-DRIFT", "spec": {}})
        try:
            with self.assertRaises(RuntimeError):
                reg.assert_registry_frozen()
        finally:
            reg.EXIT_CANDIDATES.pop()

    def test_e0_matches_current_v7_spec_exactly(self):
        e0 = next(c for c in EXIT_CANDIDATES if c["exit_id"] == "E0")
        self.assertEqual(e0["spec"]["hard_stop"], _V7_SPEC["hard_stop"])
        self.assertEqual(e0["spec"]["trail_tiers"], _V7_SPEC["trail_tiers"])

    def test_e3_isolates_only_time_stop_min_vs_e0(self):
        """E3's hard_stop/trail_tiers/tp_levels/profit_lock params must be
        identical to E0 -- the depth-based derivation was outlier-dominated
        and deliberately not used (see EXPERIMENT V2 docstring), so only
        time_stop_min should differ."""
        e0 = next(c for c in EXIT_CANDIDATES if c["exit_id"] == "E0")["spec"]
        e3 = next(c for c in EXIT_CANDIDATES if c["exit_id"] == "E3")["spec"]
        self.assertEqual(e3["hard_stop"], e0["hard_stop"])
        self.assertEqual(e3["trail_tiers"], e0["trail_tiers"])
        self.assertEqual(e3["tp_levels"], e0["tp_levels"])
        self.assertEqual(e3["time_stop_min_gain"], e0["time_stop_min_gain"])
        self.assertEqual(e3["profit_lock_min_gain"], e0["profit_lock_min_gain"])
        self.assertEqual(e3["profit_lock_max_gain"], e0["profit_lock_max_gain"])
        self.assertEqual(e3["profit_lock_stall_sec"], e0["profit_lock_stall_sec"])
        self.assertNotEqual(e3["time_stop_min"], e0["time_stop_min"])
        self.assertEqual(e3["time_stop_min"], 7)

    def test_e3_time_stop_is_shortest_of_all_candidates(self):
        stops = {c["exit_id"]: c["spec"]["time_stop_min"] for c in EXIT_CANDIDATES}
        self.assertEqual(min(stops, key=stops.get), "E3")


if __name__ == "__main__":
    unittest.main()
