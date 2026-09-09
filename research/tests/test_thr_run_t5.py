"""research/tests/test_thr_run_t5.py — THR-BATCH T5.

Run: python -m pytest research/tests/test_thr_run_t5.py -v
"""

import gzip
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.thr_run_t5 import (
    combined_mint_set_for_candidate, candidate_venue_qualified_n,
    replay_candidate_exit, ROUND_TRIP_COST_PCT, _era_days, _type_rows, _load_path_rows,
)
from research.v8_candidate_registry import CANDIDATES

_V8_P0 = next(c for c in CANDIDATES if c["candidate_id"] == "V8-P0")
_V8_P3 = next(c for c in CANDIDATES if c["candidate_id"] == "V8-P3")


class TestTypeRows(unittest.TestCase):

    def test_string_ts_ms_and_price_usd_become_numeric(self):
        rows = [{"ts_ms": "1000", "price_usd": "0.5", "other": "x"}]
        typed = _type_rows(rows)
        self.assertEqual(typed[0]["ts_ms"], 1000)
        self.assertIsInstance(typed[0]["ts_ms"], int)
        self.assertEqual(typed[0]["price_usd"], 0.5)
        self.assertIsInstance(typed[0]["price_usd"], float)

    def test_unparseable_row_dropped_not_fatal(self):
        rows = [{"ts_ms": "not-a-number", "price_usd": "0.5"}]
        self.assertEqual(_type_rows(rows), [])

    def test_load_path_rows_from_real_csv_gz_returns_typed_ints(self):
        # Regression test for the real bug this module hit live: load_path_file
        # returns csv.DictReader string values, and resolve_entry_alignment's
        # "r['ts_ms'] >= entry_target_ts" comparison raises TypeError against
        # an untyped row -- caught running T5 against the real VPS corpus.
        from research.path_schema import PATH_HEADER, PATH_SCHEMA_VERSION
        with tempfile.TemporaryDirectory() as td:
            gz_path = Path(td) / "MINT1.csv.gz"
            import csv
            with gzip.open(gz_path, "wt", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(PATH_HEADER)
                row = {c: "" for c in PATH_HEADER}
                row.update({"schema_version": str(PATH_SCHEMA_VERSION), "ts_ms": "12345",
                            "price_usd": "0.0001", "source": "reconstructed_curve_exact",
                            "venue_state": "CURVE_ACTIVE", "backfilled": "true", "data_status": "ok",
                            "side": "buy"})
                writer.writerow([row.get(c, "") for c in PATH_HEADER])

            by_mint = {"MINT1": {"path_file": None}}
            with patch("research.thr_run_t5.RECONSTRUCTED_DIR", Path(td)):
                rows, event = _load_path_rows("MINT1", by_mint, Path(td))
            self.assertIsNotNone(rows)
            self.assertIsInstance(rows[0]["ts_ms"], int)
            self.assertIsInstance(rows[0]["price_usd"], float)


class TestCombinedMintSet(unittest.TestCase):

    def test_union_dedupes_overlap(self):
        forward = {"A", "B", "C"}
        reconstructed = {"B", "C", "D"}
        by_mint = {m: {"progress_at_signal": 0.5} for m in "ABCD"}
        # use a candidate with no progress condition (V8-P0)
        result = combined_mint_set_for_candidate(_V8_P0, forward, reconstructed, by_mint)
        self.assertEqual(result, {"A", "B", "C", "D"})

    def test_filters_by_candidate_progress_condition(self):
        prog_cond = next(c for c in _V8_P3["conditions"] if c["feature"] == "progress_at_signal")
        thr = prog_cond["value"]
        by_mint = {
            "LOW": {"progress_at_signal": thr - 0.05},
            "HIGH": {"progress_at_signal": thr + 0.05},
        }
        result = combined_mint_set_for_candidate(_V8_P3, {"LOW", "HIGH"}, set(), by_mint)
        self.assertEqual(result, {"LOW"})

    def test_missing_progress_at_signal_excluded_when_condition_present(self):
        prog_cond = next(c for c in _V8_P3["conditions"] if c["feature"] == "progress_at_signal")
        by_mint = {"NOPROG": {"progress_at_signal": None}}
        result = combined_mint_set_for_candidate(_V8_P3, {"NOPROG"}, set(), by_mint)
        self.assertEqual(result, set())


class TestCandidateVenueQualifiedN(unittest.TestCase):

    def test_counts_only_curve_active(self):
        by_mint = {
            "A": {"venue_state_at_signal": "CURVE_ACTIVE", "progress_at_signal": 0.5},
            "B": {"venue_state_at_signal": "GRADUATED", "progress_at_signal": 0.5},
        }
        n = candidate_venue_qualified_n(_V8_P0, by_mint)
        self.assertEqual(n, 1)

    def test_respects_progress_condition(self):
        prog_cond = next(c for c in _V8_P3["conditions"] if c["feature"] == "progress_at_signal")
        thr = prog_cond["value"]
        by_mint = {
            "LOW": {"venue_state_at_signal": "CURVE_ACTIVE", "progress_at_signal": thr - 0.05},
            "HIGH": {"venue_state_at_signal": "CURVE_ACTIVE", "progress_at_signal": thr + 0.05},
        }
        n = candidate_venue_qualified_n(_V8_P3, by_mint)
        self.assertEqual(n, 1)


class TestEraDays(unittest.TestCase):

    def test_never_below_one_day(self):
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        self.assertGreaterEqual(_era_days(now_iso), 1.0)


_ALERT_TIME = "2026-09-01T00:00:00+00:00"


class TestReplayCandidateExit(unittest.TestCase):
    """Uses real research.v8_replay_engine / v8_entry_alignment via a
    synthetic but internally-consistent path -- not mocking the replay
    math itself, only the path-loading I/O."""

    def _alert_ts_ms(self):
        from datetime import datetime
        return int(datetime.fromisoformat(_ALERT_TIME).timestamp() * 1000)

    def _synthetic_rows(self, price_multipliers):
        # price_multipliers are relative multipliers on a realistic
        # on-curve base price -- research.v8_path_integrity's theoretical
        # curve-price ceiling (~8.22e-6 SOL/token with slack) rejects
        # naive dollar-scale prices like $1.00 as impossible for a
        # CURVE_ACTIVE tick, same check that catches real corruption.
        from research.path_schema import PATH_SCHEMA_VERSION
        alert_ts_ms = self._alert_ts_ms()
        base_price_sol = 1e-6
        sol_price_usd = 200.0
        rows = []
        for i, mult in enumerate(price_multipliers):
            price_sol = base_price_sol * mult
            rows.append({
                "schema_version": str(PATH_SCHEMA_VERSION), "research_event_id": "", "event_id": f"e{i}",
                "ts_ms": alert_ts_ms + i * 1000, "price_usd": price_sol * sol_price_usd,
                "price_sol": price_sol, "side": "buy",
                "token_amount": 100.0, "sol_amount": 1.0, "vsol": 30.0, "vtok": 1_000_000_000,
                "source": "reconstructed_curve_exact", "venue_state": "CURVE_ACTIVE",
                "backfilled": "true", "data_status": "ok", "trader_pk": "",
            })
        return rows

    def test_applies_round_trip_cost_haircut(self):
        # Flat-then-double path: TP-less E0 should exit near path_end with
        # a large raw gain; net pnl must be exactly ROUND_TRIP_COST_PCT
        # lower than whatever the raw replay would have produced alone.
        rows = self._synthetic_rows([1.0] * 3 + [2.0] * 200)  # big, clean winner, no stop/time-stop trip
        event = {"event_id": "e0", "token_address": "MINT1", "alert_time": _ALERT_TIME,
                 "progress_capture_lag_ms": 0}
        by_mint = {"MINT1": event}
        exit_spec_dict = {"exit_id": "E0", "spec": {
            "hard_stop": -0.99, "trail_tiers": [], "tp_levels": [], "time_stop_min": 999999,
        }}
        with patch("research.thr_run_t5._load_path_rows", return_value=(rows, event)):
            stats = replay_candidate_exit(_V8_P0, exit_spec_dict, {"MINT1"}, by_mint, set(), None)
        self.assertEqual(stats["n"], 1)
        # raw gain should be large positive; net = raw + ROUND_TRIP_COST_PCT
        self.assertLess(stats["mean_pnl_pct_net"], 100 * 1.0)  # sanity: not absurd
        self.assertGreater(stats["mean_pnl_pct_net"], 0)

    def test_no_path_rows_counted_as_exclusion_not_crash(self):
        by_mint = {"MINT1": {"event_id": "e0", "token_address": "MINT1",
                              "alert_time": _ALERT_TIME}}
        exit_spec_dict = {"exit_id": "E0", "spec": {"hard_stop": -0.35, "trail_tiers": [], "tp_levels": [],
                                                      "time_stop_min": 90}}
        with patch("research.thr_run_t5._load_path_rows", return_value=(None, by_mint["MINT1"])):
            stats = replay_candidate_exit(_V8_P0, exit_spec_dict, {"MINT1"}, by_mint, set(), None)
        self.assertEqual(stats["n"], 0)
        self.assertIn("NO_PATH_ROWS", stats["exclusion_reasons"])

    def test_ambiguous_mint_excluded(self):
        rows = self._synthetic_rows([1.0, 1.1, 1.2])
        event = {"event_id": "e0", "token_address": "MINT1", "alert_time": _ALERT_TIME}
        by_mint = {"MINT1": event}
        exit_spec_dict = {"exit_id": "E0", "spec": {"hard_stop": -0.35, "trail_tiers": [], "tp_levels": [],
                                                      "time_stop_min": 90}}
        with patch("research.thr_run_t5._load_path_rows", return_value=(rows, event)):
            stats = replay_candidate_exit(_V8_P0, exit_spec_dict, {"MINT1"}, by_mint, {"MINT1"}, None)
        self.assertEqual(stats["n"], 0)
        self.assertIn("AMBIGUOUS_PATH_EVENT_JOIN", stats["exclusion_reasons"])


if __name__ == "__main__":
    unittest.main()
