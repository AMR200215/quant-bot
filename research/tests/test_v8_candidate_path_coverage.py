"""research/tests/test_v8_candidate_path_coverage.py — V8 READINESS
MONITOR CORRECTION batch item 2: candidate-specific path coverage.

Run: python -m pytest research/tests/test_v8_candidate_path_coverage.py -v
"""

import csv
import tempfile
import unittest
from pathlib import Path

from research.v8_candidate_path_coverage import compute_candidate_path_coverage
from research.v8_candidate_registry import CANDIDATES
from research.path_schema import PATH_HEADER

_BASELINE = next(c for c in CANDIDATES if c["candidate_id"] == "BASELINE-0")


def _write_path_file(path: Path, rows: list):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PATH_HEADER)
        w.writeheader()
        for r in rows:
            full = {k: "" for k in PATH_HEADER}
            full.update(r)
            w.writerow(full)


_GOOD_ROW = {
    "schema_version": "2", "ts_ms": "1000", "price_usd": "0.00005", "price_sol": "0.0000003",
    "vsol": "50.0", "vtok": "1000000000", "venue_state": "CURVE_ACTIVE", "source": "live_pp",
    "backfilled": "false", "data_status": "ok",
}


class TestComputeCandidatePathCoverage(unittest.TestCase):

    def test_basic_coverage_with_one_usable_path(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path_dir = root / "logs" / "research_paths" / "2026-08-19"
            path_dir.mkdir(parents=True)
            rel_path = "logs/research_paths/2026-08-19/MINT_A.csv"
            _write_path_file(root / rel_path, [dict(_GOOD_ROW, ts_ms="0"), dict(_GOOD_ROW, ts_ms="1000")])

            events = [{"event_id": "e1", "token_address": "MINT_A", "alert_time": "1970-01-01T00:00:00+00:00",
                       "path_file": rel_path}]
            result = compute_candidate_path_coverage(events, events, _BASELINE, root)

        self.assertEqual(result.eligible_n, 1)
        self.assertEqual(result.usable_n, 1)
        self.assertEqual(result.coverage_pct, 100.0)

    def test_ambiguous_mint_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path_dir = root / "logs" / "research_paths" / "2026-08-19"
            path_dir.mkdir(parents=True)
            rel_path = "logs/research_paths/2026-08-19/MINT_A.csv"
            _write_path_file(root / rel_path, [dict(_GOOD_ROW, ts_ms="0")])

            # Two events sharing the same mint -> ambiguous
            all_events = [
                {"event_id": "e1", "token_address": "MINT_A", "alert_time": "1970-01-01T00:00:00+00:00", "path_file": rel_path},
                {"event_id": "e2", "token_address": "MINT_A", "alert_time": "1970-01-01T00:10:00+00:00", "path_file": rel_path},
            ]
            result = compute_candidate_path_coverage(all_events, all_events, _BASELINE, root)

        self.assertEqual(result.eligible_n, 2)
        self.assertEqual(result.usable_n, 0)
        self.assertEqual(result.ambiguous_excluded_n, 2)

    def test_no_path_file_counted_separately(self):
        events = [{"event_id": "e1", "token_address": "MINT_A", "alert_time": "1970-01-01T00:00:00+00:00", "path_file": None}]
        result = compute_candidate_path_coverage(events, events, _BASELINE, Path("/tmp"))
        self.assertEqual(result.no_path_file_n, 1)
        self.assertEqual(result.usable_n, 0)

    def test_missing_file_on_disk_counted_separately(self):
        events = [{"event_id": "e1", "token_address": "MINT_A", "alert_time": "1970-01-01T00:00:00+00:00",
                   "path_file": "logs/research_paths/does-not-exist/MINT_A.csv"}]
        result = compute_candidate_path_coverage(events, events, _BASELINE, Path("/tmp"))
        self.assertEqual(result.path_file_missing_or_empty_n, 1)

    def test_corrupted_path_excluded_via_integrity_check(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path_dir = root / "logs" / "research_paths" / "2026-08-19"
            path_dir.mkdir(parents=True)
            rel_path = "logs/research_paths/2026-08-19/MINT_BAD.csv"
            bad_row = dict(_GOOD_ROW, ts_ms="1000", price_usd="73.49", price_sol="0.42", vsol="116.27")
            # A second, later tick so entry-alignment can resolve to a real
            # executable entry point (decision_available_ts = alert_ts + the
            # candidate's nominal T0+capture delay) -- otherwise the single
            # corrupted tick at ts=0 never survives alignment to reach the
            # integrity check at all, which would test the wrong exclusion path.
            _write_path_file(root / rel_path, [dict(_GOOD_ROW, ts_ms="0"), bad_row])

            events = [{"event_id": "e1", "token_address": "MINT_BAD", "alert_time": "1970-01-01T00:00:00+00:00",
                       "path_file": rel_path}]
            result = compute_candidate_path_coverage(events, events, _BASELINE, root)

        self.assertEqual(result.usable_n, 0)
        self.assertEqual(result.integrity_invalid_or_unknown_n, 1)

    def test_eligible_n_matches_input_length_regardless_of_outcome(self):
        events = [{"event_id": f"e{i}", "token_address": f"M{i}", "alert_time": "1970-01-01T00:00:00+00:00",
                   "path_file": None} for i in range(5)]
        result = compute_candidate_path_coverage(events, events, _BASELINE, Path("/tmp"))
        self.assertEqual(result.eligible_n, 5)

    def test_empty_candidate_events_gives_zero_coverage_not_error(self):
        result = compute_candidate_path_coverage([], [], _BASELINE, Path("/tmp"))
        self.assertEqual(result.eligible_n, 0)
        self.assertEqual(result.coverage_pct, 0.0)

    def test_gz_rotated_path_file_is_found_not_reported_missing(self):
        """V8 forward-readiness health check incident finding: peak_tracker.py
        gzips the previous UTC day's path files in place without updating
        research_tokens.path_file. A path recorded as "*.csv" that only
        exists on disk as "*.csv.gz" must still be found and read -- this
        was previously misclassified as path_file_missing_or_empty_n,
        which silently zeroed out coverage for every path older than the
        current UTC day."""
        import gzip
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path_dir = root / "logs" / "research_paths" / "2026-08-19"
            path_dir.mkdir(parents=True)
            rel_path = "logs/research_paths/2026-08-19/MINT_A.csv"

            # Write the plain CSV, then gzip it and remove the original --
            # matches PeakTracker._gzip_directory's real behavior exactly.
            plain = root / rel_path
            _write_path_file(plain, [dict(_GOOD_ROW, ts_ms="0"), dict(_GOOD_ROW, ts_ms="1000")])
            with open(plain, "rb") as f_in, gzip.open(str(plain) + ".gz", "wb") as f_out:
                f_out.write(f_in.read())
            plain.unlink()

            events = [{"event_id": "e1", "token_address": "MINT_A", "alert_time": "1970-01-01T00:00:00+00:00",
                       "path_file": rel_path}]
            result = compute_candidate_path_coverage(events, events, _BASELINE, root)

        self.assertEqual(result.usable_n, 1)
        self.assertEqual(result.path_file_missing_or_empty_n, 0)
        self.assertEqual(result.coverage_pct, 100.0)


if __name__ == "__main__":
    unittest.main()
