"""research/tests/test_thr_winner_exit_analysis.py

Run: python -m pytest research/tests/test_thr_winner_exit_analysis.py -v
"""

import unittest
from datetime import datetime
from unittest.mock import patch

from research.thr_winner_exit_analysis import find_real_winners, analyze_exit_for_winners
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


class TestFindRealWinners(unittest.TestCase):

    def test_token_reaching_winner_threshold_is_found(self):
        # Flat lead-in at 1.0 (covers the T0+capture nominal delay
        # regardless of exactly which early tick becomes the entry tick),
        # then a clean run to 2.0 (+100% relative to any of the flat
        # entry-candidate ticks) -- comfortably clears the 50% threshold.
        rows = _synthetic_rows([1.0, 1.0, 1.0, 2.0, 1.5] + [1.5] * 5)
        event = {"event_id": "e0", "token_address": "MINT1", "alert_time": _ALERT_TIME}
        by_mint = {"MINT1": event}

        with patch("research.thr_winner_exit_analysis._load_path_rows",
                   return_value=(rows, event, "forward")):
            winners = find_real_winners(_V8_P0, {"MINT1"}, by_mint, set(), None)

        self.assertEqual(len(winners), 1)
        self.assertAlmostEqual(winners[0]["peak_gain_pct"], 100.0, delta=0.5)

    def test_token_never_reaching_threshold_excluded(self):
        rows = _synthetic_rows([1.0, 1.1, 1.2, 1.15])  # peak +20%, below 50% threshold
        event = {"event_id": "e0", "token_address": "MINT1", "alert_time": _ALERT_TIME}
        by_mint = {"MINT1": event}

        with patch("research.thr_winner_exit_analysis._load_path_rows",
                   return_value=(rows, event, "forward")):
            winners = find_real_winners(_V8_P0, {"MINT1"}, by_mint, set(), None)

        self.assertEqual(winners, [])

    def test_reconstructed_source_excluded(self):
        rows = _synthetic_rows([1.0, 2.0, 2.0])
        event = {"event_id": "e0", "token_address": "MINT1", "alert_time": _ALERT_TIME}
        by_mint = {"MINT1": event}

        with patch("research.thr_winner_exit_analysis._load_path_rows",
                   return_value=(rows, event, "reconstructed")):
            winners = find_real_winners(_V8_P0, {"MINT1"}, by_mint, set(), None)

        self.assertEqual(winners, [])

    def test_ambiguous_mint_excluded(self):
        rows = _synthetic_rows([1.0, 2.0, 2.0])
        event = {"event_id": "e0", "token_address": "MINT1", "alert_time": _ALERT_TIME}
        by_mint = {"MINT1": event}

        with patch("research.thr_winner_exit_analysis._load_path_rows",
                   return_value=(rows, event, "forward")):
            winners = find_real_winners(_V8_P0, {"MINT1"}, by_mint, {"MINT1"}, None)

        self.assertEqual(winners, [])


class TestAnalyzeExitForWinners(unittest.TestCase):

    def test_time_stop_cutting_off_a_slow_winner_is_captured(self):
        # Winner takes a long time (well past a short time_stop) to reach
        # +50%, and never exceeds +30% before the deadline -- time_stop
        # should fire, well before the eventual peak.
        rows = _synthetic_rows(
            [1.0] * 10          # flat for 10s (below the 30% time-stop-gain floor)
            + [1.6] * 5,        # then a late jump to +60% (past the time-stop check point)
            tick_seconds=60,    # 1 minute per tick -> 10 minutes flat
        )
        event = {"event_id": "e0", "token_address": "MINT1", "alert_time": _ALERT_TIME}
        winner = {
            "mint": "MINT1", "rows": rows, "entry_ts_ms": rows[0]["ts_ms"],
            "entry_price": rows[0]["price_usd"], "peak_gain_pct": 60.0, "peak_ts_ms": rows[-1]["ts_ms"],
        }
        exit_spec_dict = {"exit_id": "E3", "spec": {
            "hard_stop": -0.35, "trail_tiers": [], "tp_levels": [],
            "time_stop_min": 7, "time_stop_min_gain": 0.30,
        }}
        result = analyze_exit_for_winners([winner], exit_spec_dict)
        self.assertEqual(result["n_winners_evaluated"], 1)
        self.assertIn("time_stop", result["by_exit_reason"])
        cf = result["by_exit_reason"]["time_stop"]["mean_captured_fraction"]
        self.assertLess(cf, 1.0)  # captured less than the eventual peak

    def test_replay_refused_path_skipped_not_crash(self):
        winner = {"mint": "MINT1", "rows": [], "entry_ts_ms": 0, "entry_price": 1.0,
                  "peak_gain_pct": 60.0, "peak_ts_ms": 0}
        exit_spec_dict = {"exit_id": "E0", "spec": {"hard_stop": -0.35, "trail_tiers": [], "tp_levels": [],
                                                       "time_stop_min": 90}}
        result = analyze_exit_for_winners([winner], exit_spec_dict)
        self.assertEqual(result["n_winners_evaluated"], 0)


if __name__ == "__main__":
    unittest.main()
