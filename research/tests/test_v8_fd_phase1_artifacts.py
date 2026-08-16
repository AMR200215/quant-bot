"""research/tests/test_v8_fd_phase1_artifacts.py — V8-FILTER-DERIVATION
Phase 1 (FD35): structural tests for the artifacts Phase 1 produces.

These test the STRUCTURE and internal consistency of the registry/cohort
definition, not live Supabase data (which isn't deterministic test
input) — e.g. "every feature entry has the required keys" and "no
POST-TRADE feature is ever marked allowed_for_entry", so a future edit
can't silently reintroduce a lookahead-bias bug without a test failing.

Run: python -m pytest research/tests/test_v8_fd_phase1_artifacts.py -v
"""

import unittest
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).parent.parent.parent
_REGISTRY_PATH = _REPO_ROOT / "research" / "v8_feature_registry.yaml"

_REQUIRED_KEYS = {
    "name", "semantic", "source", "availability_class", "availability_delay_ms",
    "live_deployable", "coverage_note", "leakage_risk",
    "allowed_for_entry", "allowed_for_midtrade", "allowed_for_exit",
}

_VALID_CLASSES = {"T0", "T0+capture", "T0+snapshot", "T+Ns", "POST-TRADE"}


def _load_registry() -> dict:
    return yaml.safe_load(_REGISTRY_PATH.read_text())


class TestFeatureRegistryStructure(unittest.TestCase):

    def test_registry_file_exists_and_parses(self):
        self.assertTrue(_REGISTRY_PATH.exists())
        data = _load_registry()
        self.assertIn("features", data)
        self.assertIsInstance(data["features"], list)
        self.assertGreater(len(data["features"]), 0)

    def test_every_feature_has_required_keys(self):
        data = _load_registry()
        for f in data["features"]:
            missing = _REQUIRED_KEYS - set(f.keys())
            self.assertEqual(missing, set(), f"{f.get('name')} missing keys: {missing}")

    def test_every_feature_has_a_valid_availability_class(self):
        data = _load_registry()
        for f in data["features"]:
            self.assertIn(f["availability_class"], _VALID_CLASSES,
                           f"{f['name']} has unknown availability_class {f['availability_class']!r}")

    def test_no_post_trade_feature_is_allowed_for_entry(self):
        """The core FD3 lookahead-prevention invariant, enforced structurally:
        if a future edit ever sets allowed_for_entry: true on a POST-TRADE
        feature, this test fails immediately."""
        data = _load_registry()
        for f in data["features"]:
            if f["availability_class"] == "POST-TRADE":
                self.assertFalse(f["allowed_for_entry"],
                                  f"{f['name']} is POST-TRADE but allowed_for_entry=True")
                self.assertFalse(f["allowed_for_midtrade"],
                                  f"{f['name']} is POST-TRADE but allowed_for_midtrade=True")

    def test_smart_money_fields_locked_ineligible(self):
        """FD6 verdict, enforced structurally so it can't silently regress."""
        data = _load_registry()
        by_name = {f["name"]: f for f in data["features"]}
        for name in ("smart_money_hit", "smart_money_count"):
            self.assertIn(name, by_name)
            f = by_name[name]
            self.assertFalse(f["allowed_for_entry"])
            self.assertIn("SMART_MONEY_NOT_ELIGIBLE_FOR_HISTORICAL_SELECTION",
                           f["leakage_risk"])

    def test_dex_id_flags_the_known_graduation_bug(self):
        data = _load_registry()
        by_name = {f["name"]: f for f in data["features"]}
        self.assertIn("dex_id", by_name)
        self.assertIn("UNRELIABLE", by_name["dex_id"]["leakage_risk"])

    def test_raw_realert_count_not_allowed_but_realert_times_conditionally_is(self):
        data = _load_registry()
        by_name = {f["name"]: f for f in data["features"]}
        self.assertFalse(by_name["realert_count"]["allowed_for_entry"])
        self.assertTrue(by_name["realert_times"]["allowed_for_entry"])

    def test_pct_change_peak_never_allowed_anywhere(self):
        data = _load_registry()
        by_name = {f["name"]: f for f in data["features"]}
        for name in ("pct_change_peak", "pct_change_peak_3m"):
            f = by_name[name]
            self.assertFalse(f["allowed_for_entry"])
            self.assertFalse(f["allowed_for_midtrade"])
            self.assertFalse(f["allowed_for_exit"])


class TestCleanCohortManifest(unittest.TestCase):

    def test_module_imports_and_has_required_attrs(self):
        from research import v8_clean_cohort as cc
        self.assertIsInstance(cc.V8_CLEAN_COHORT_VERSION, int)
        self.assertGreaterEqual(cc.V8_CLEAN_COHORT_VERSION, 1)
        self.assertIsInstance(cc.GATES, list)
        self.assertGreater(len(cc.GATES), 0)
        self.assertIsInstance(cc.KNOWN_GAPS, list)

    def test_gates_are_monotonically_non_increasing(self):
        """Each successive gate must narrow or hold the cohort, never grow it --
        a real bug if it ever does (would mean a gate isn't actually a filter)."""
        from research import v8_clean_cohort as cc
        counts = [g["live_count"] for g in sorted(cc.GATES, key=lambda g: g["order"])]
        for i in range(1, len(counts)):
            self.assertLessEqual(counts[i], counts[i - 1],
                                  f"gate {i+1} count {counts[i]} exceeds prior gate count {counts[i-1]}")

    def test_gates_have_required_fields(self):
        from research import v8_clean_cohort as cc
        for g in cc.GATES:
            for key in ("order", "field", "condition", "live_count", "pct_of_total"):
                self.assertIn(key, g)


class TestPhase15Corrections(unittest.TestCase):
    """P15-1/P15-2/P15-3: the follow-up corrections must actually be
    present and internally consistent, not just prose in a docstring."""

    def test_candidate0_progress_half_is_a_positive_int_not_hardcoded_to_one_bucket(self):
        from research import v8_clean_cohort as cc
        self.assertIsInstance(cc.CANDIDATE0_PROGRESS_HALF_N, int)
        self.assertGreater(cc.CANDIDATE0_PROGRESS_HALF_N, 1,
            "must be the sum of <50% and 50-70%, not just the 50-70% bucket alone")

    def test_candidate0_full_gate_historical_n_is_explicitly_unknown(self):
        """venue_state_at_signal isn't persisted historically -- this must
        stay None (never fabricated), per P15-4's explicit instruction."""
        from research import v8_clean_cohort as cc
        self.assertIsNone(cc.CANDIDATE0_FULL_GATE_HISTORICAL_N)

    def test_clean_cohort_date_range_is_distinct_from_overall_table_range(self):
        from research import v8_clean_cohort as cc
        rng = cc.CLEAN_COHORT_DATE_RANGE
        self.assertIn("min_alert_time", rng)
        self.assertIn("max_alert_time", rng)
        self.assertIn("unique_calendar_days", rng)
        # the overall table spans ~2 months (Jun21-Aug15); the clean
        # cohort must NOT claim that same span -- it's a much narrower window
        self.assertLess(rng["unique_calendar_days"], 30)

    def test_progress_policy_candidates_include_p0_through_p3_and_no_dense_grid(self):
        from research import v8_clean_cohort as cc
        ids = {c["id"] for c in cc.PROGRESS_POLICY_CANDIDATES}
        self.assertEqual(ids, {"P0", "P1", "P2", "P3"})
        # explicitly NOT a fine-grained threshold sweep (FD8's anti-fishing rule)
        self.assertLessEqual(len(cc.PROGRESS_POLICY_CANDIDATES), 6)

    def test_p2_matches_current_candidate0_threshold(self):
        from research import v8_clean_cohort as cc
        p2 = next(c for c in cc.PROGRESS_POLICY_CANDIDATES if c["id"] == "P2")
        self.assertIn("0.70", p2["rule"])

    def test_p15_9_precondition_constants_exist(self):
        from research import v8_clean_cohort as cc
        self.assertIsInstance(cc.ENGINE_DESIGN_READY_MEANS, str)
        self.assertIsInstance(cc.SELECTION_DATA_READY_MEANS, str)


if __name__ == "__main__":
    unittest.main()
