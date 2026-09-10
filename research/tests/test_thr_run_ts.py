"""research/tests/test_thr_run_ts.py — TS-BATCH grid derivation.

Run: python -m pytest research/tests/test_thr_run_ts.py -v
"""

import unittest
from datetime import datetime
from unittest.mock import patch

from research.thr_run_ts import (
    build_trail_tiers, build_grid, build_exit_spec, prepare_population,
    score_cell, qualifies, E0_HARD_STOP, E0_TIME_STOP_MIN, E0_TIER2_ARM, E0_TIER3_ARM,
    ARM_VALUES, WIDTH_VALUES, TP_VARIANTS,
)
from research.v8_candidate_registry import CANDIDATES

_V8_P0 = next(c for c in CANDIDATES if c["candidate_id"] == "V8-P0")
_ALERT_TIME = "2026-09-01T00:00:00+00:00"


def _alert_ts_ms():
    return int(datetime.fromisoformat(_ALERT_TIME).timestamp() * 1000)


def _synthetic_rows(price_multipliers, tick_seconds=1):
    from research.path_schema import PATH_SCHEMA_VERSION
    alert_ts_ms = _alert_ts_ms()
    base_price_sol = 1e-6
    sol_price_usd = 200.0
    rows = []
    for i, mult in enumerate(price_multipliers):
        price_sol = base_price_sol * mult
        rows.append({
            "schema_version": str(PATH_SCHEMA_VERSION), "research_event_id": "", "event_id": f"e{i}",
            "ts_ms": alert_ts_ms + i * tick_seconds * 1000, "price_usd": price_sol * sol_price_usd,
            "price_sol": price_sol, "side": "buy",
            "token_amount": 100.0, "sol_amount": 1.0, "vsol": 30.0, "vtok": 1_000_000_000,
            "source": "reconstructed_curve_exact", "venue_state": "CURVE_ACTIVE",
            "backfilled": "true", "data_status": "ok", "trader_pk": "",
        })
    return rows


class TestBuildTrailTiers(unittest.TestCase):

    def test_tier1_uses_grid_arm_and_width(self):
        tiers = build_trail_tiers(arm=0.40, width=0.30)
        self.assertEqual(tiers[0], {"activates_at": 0.40, "trail_pct": 0.30})

    def test_tier2_and_tier3_arm_points_fixed_at_e0_values(self):
        tiers = build_trail_tiers(arm=0.75, width=0.50)
        self.assertEqual(tiers[1]["activates_at"], E0_TIER2_ARM)
        self.assertEqual(tiers[2]["activates_at"], E0_TIER3_ARM)

    def test_tier3_width_preserves_e0_ratio(self):
        tiers = build_trail_tiers(arm=0.30, width=0.25)
        # E0 itself: tier1/2 width=0.25, tier3 width=0.15 -> ratio 0.6
        self.assertAlmostEqual(tiers[2]["trail_pct"], 0.25 * 0.6, places=4)

    def test_tier2_width_matches_tier1(self):
        tiers = build_trail_tiers(arm=0.50, width=0.40)
        self.assertEqual(tiers[0]["trail_pct"], tiers[1]["trail_pct"])


class TestBuildGrid(unittest.TestCase):

    def test_grid_size_is_32(self):
        grid = build_grid()
        self.assertEqual(len(grid), len(ARM_VALUES) * len(WIDTH_VALUES) * len(TP_VARIANTS))
        self.assertEqual(len(grid), 32)

    def test_labels_unique(self):
        grid = build_grid()
        labels = [c["label"] for c in grid]
        self.assertEqual(len(labels), len(set(labels)))

    def test_hard_stop_and_time_stop_fixed_at_e0_values_every_cell(self):
        grid = build_grid()
        for cell in grid:
            self.assertEqual(cell["spec"]["hard_stop"], E0_HARD_STOP)
            self.assertEqual(cell["spec"]["time_stop_min"], E0_TIME_STOP_MIN)

    def test_tp_variant_reflected_in_spec(self):
        grid = build_grid()
        notp_cells = [c for c in grid if c["tp_variant"] == "notp"]
        tp_cells = [c for c in grid if c["tp_variant"] == "tp"]
        self.assertTrue(all(c["spec"]["tp_levels"] == [] for c in notp_cells))
        self.assertTrue(all(c["spec"]["tp_levels"] == [(0.50, 0.25), (1.00, 0.25)] for c in tp_cells))


class TestPreparePopulation(unittest.TestCase):

    def test_includes_both_forward_and_reconstructed_sources(self):
        rows_a = _synthetic_rows([1.0, 1.0, 1.0, 2.0, 1.5])
        rows_b = _synthetic_rows([1.0, 1.0, 1.0, 1.1, 1.1])
        event_a = {"event_id": "ea", "token_address": "MINTA", "alert_time": _ALERT_TIME}
        event_b = {"event_id": "eb", "token_address": "MINTB", "alert_time": _ALERT_TIME}
        by_mint = {"MINTA": event_a, "MINTB": event_b}

        def fake_load(mint, by_mint_arg, root_arg):
            if mint == "MINTA":
                return rows_a, event_a, "forward"
            return rows_b, event_b, "reconstructed"

        with patch("research.thr_run_ts._load_path_rows", side_effect=fake_load):
            population = prepare_population(_V8_P0, {"MINTA", "MINTB"}, by_mint, set(), None)

        self.assertEqual(len(population), 2)
        sources = {p["mint"]: p["source"] for p in population}
        self.assertEqual(sources["MINTA"], "forward")
        self.assertEqual(sources["MINTB"], "reconstructed")

    def test_winner_classification_correct(self):
        rows_winner = _synthetic_rows([1.0, 1.0, 1.0, 2.0, 1.5])   # peak +100%
        rows_loser = _synthetic_rows([1.0, 1.0, 1.0, 1.1, 1.05])   # peak +10%
        event_w = {"event_id": "ew", "token_address": "MINTW", "alert_time": _ALERT_TIME}
        event_l = {"event_id": "el", "token_address": "MINTL", "alert_time": _ALERT_TIME}
        by_mint = {"MINTW": event_w, "MINTL": event_l}

        def fake_load(mint, by_mint_arg, root_arg):
            if mint == "MINTW":
                return rows_winner, event_w, "forward"
            return rows_loser, event_l, "forward"

        with patch("research.thr_run_ts._load_path_rows", side_effect=fake_load):
            population = prepare_population(_V8_P0, {"MINTW", "MINTL"}, by_mint, set(), None)

        by_mint_out = {p["mint"]: p for p in population}
        self.assertTrue(by_mint_out["MINTW"]["is_winner"])
        self.assertFalse(by_mint_out["MINTL"]["is_winner"])


class TestScoreCell(unittest.TestCase):

    def test_censoring_detected_and_pessimistically_imputed(self):
        # Short path: reaches a nice gain then simply stops recording --
        # no exit rule ever fires except path_end.
        rows = _synthetic_rows([1.0, 1.0, 1.0, 1.6, 1.6])
        population = [{
            "mint": "MINT1", "rows": rows, "entry_ts_ms": rows[0]["ts_ms"],
            "source": "forward", "peak_gain_pct": 60.0, "is_winner": True,
        }]
        spec = build_exit_spec(arm=0.75, width=0.50, tp_levels=[])  # arms very late -- won't fire here
        result = score_cell(population, spec)
        combined = result["combined"]
        self.assertEqual(combined["n"], 1)
        self.assertEqual(combined["censored_fraction"], 1.0)
        # pessimistic EV must equal the spec's own hard_stop floor + cost, not the rosy last price
        expected_floor = round(spec["hard_stop"] * 100 + (-1.99), 2)
        self.assertAlmostEqual(combined["net_mean_ev_pct_pessimistic"], expected_floor, places=1)
        # optimistic EV should NOT equal the floor (it's the better, uncensored-looking number)
        self.assertNotAlmostEqual(combined["net_mean_ev_pct"], expected_floor, places=1)

    def test_forward_only_is_strict_subset_of_combined(self):
        rows = _synthetic_rows([1.0, 1.0, 1.0, 1.3, 1.2, 1.2, 1.2, 1.2])
        population = [
            {"mint": "F1", "rows": rows, "entry_ts_ms": rows[0]["ts_ms"],
             "source": "forward", "peak_gain_pct": 30.0, "is_winner": False},
            {"mint": "R1", "rows": rows, "entry_ts_ms": rows[0]["ts_ms"],
             "source": "reconstructed", "peak_gain_pct": 30.0, "is_winner": False},
        ]
        spec = build_exit_spec(arm=0.30, width=0.25, tp_levels=[])
        result = score_cell(population, spec)
        self.assertEqual(result["forward_only"]["n"], 1)
        self.assertEqual(result["combined"]["n"], 2)

    def test_winner_vs_nonwinner_bucketed_separately(self):
        winner_rows = _synthetic_rows([1.0, 1.0, 1.0, 2.0] + [1.5] * 10)
        loser_rows = _synthetic_rows([1.0, 1.0, 1.0, 0.7] + [0.6] * 10)
        population = [
            {"mint": "W1", "rows": winner_rows, "entry_ts_ms": winner_rows[0]["ts_ms"],
             "source": "forward", "peak_gain_pct": 100.0, "is_winner": True},
            {"mint": "L1", "rows": loser_rows, "entry_ts_ms": loser_rows[0]["ts_ms"],
             "source": "forward", "peak_gain_pct": 0.0, "is_winner": False},
        ]
        spec = build_exit_spec(arm=0.30, width=0.25, tp_levels=[])
        result = score_cell(population, spec)
        combined = result["combined"]
        self.assertEqual(combined["winner_n"], 1)
        self.assertEqual(combined["nonwinner_n"], 1)
        self.assertIsNotNone(combined["winner_mean_capture"])
        self.assertIsNotNone(combined["nonwinner_mean_net_pnl_pct"])


class TestQualifies(unittest.TestCase):

    def _stats(self, ev_pess, hard_stop_rate, n=50):
        s = {"n": n, "net_mean_ev_pct_pessimistic": ev_pess, "hard_stop_hit_rate_pct": hard_stop_rate}
        return {"forward_only": s, "combined": s}

    def test_qualifies_when_beats_e0_and_hard_stop_within_tolerance(self):
        cell = self._stats(ev_pess=5.0, hard_stop_rate=40.0)
        e0 = self._stats(ev_pess=2.0, hard_stop_rate=38.0)
        result = qualifies(cell, e0)
        self.assertTrue(result["qualifies"])

    def test_disqualified_when_pessimistic_ev_does_not_beat_e0(self):
        cell = self._stats(ev_pess=1.0, hard_stop_rate=38.0)
        e0 = self._stats(ev_pess=2.0, hard_stop_rate=38.0)
        result = qualifies(cell, e0)
        self.assertFalse(result["qualifies"])
        self.assertTrue(any("does not beat E0" in r for r in result["reasons"]))

    def test_disqualified_when_hard_stop_degraded_beyond_tolerance(self):
        cell = self._stats(ev_pess=5.0, hard_stop_rate=55.0)
        e0 = self._stats(ev_pess=2.0, hard_stop_rate=38.0)  # delta = 17pp > 10pp tolerance
        result = qualifies(cell, e0)
        self.assertFalse(result["qualifies"])
        self.assertTrue(any("degraded" in r for r in result["reasons"]))

    def test_small_hard_stop_degradation_within_tolerance_passes(self):
        cell = self._stats(ev_pess=5.0, hard_stop_rate=45.0)
        e0 = self._stats(ev_pess=2.0, hard_stop_rate=38.0)  # delta = 7pp < 10pp tolerance
        result = qualifies(cell, e0)
        self.assertTrue(result["qualifies"])


if __name__ == "__main__":
    unittest.main()
