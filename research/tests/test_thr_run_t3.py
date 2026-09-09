"""research/tests/test_thr_run_t3.py — THR-BATCH T3 reconstruction run.

Run: python -m pytest research/tests/test_thr_run_t3.py -v
"""

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.thr_run_t3 import process_token, _write_reconstructed_csv, run
from research.path_schema import load_path_file

# Same real, live-verified event blob used in test_thr_reconstruct_paths.py.
_REAL_EVENT_1_MINT = "HnAVbEMfF1iLdBsSF2YGf9LHWvurwX63cTGHyKaWpump"
_REAL_EVENT_1_B64 = (
    "vdt/007mYe75TNxlkX/y4DNd7rh9Do9vZnNu+iiqsfk/uK0O4tQ5v7diQAMAAAAAuTh1EsIBAAAA"
    "OQMBrV7X0XyRxiF7MdJ5+4mQfDDjkGir/FRIYipc+OzgbZ9qAAAAAPVS2gIHAAAAXeBZ0zzMAwD1"
    "prYGAAAAAF1IR4erzQIAY4NzAA6iLLJk00r/ZKBLXvq/u3TdzQSJl7GYFUfX0RBfAAAAAAAAAF3o"
    "BwAAAAAAu+ySfNwWzXDRM33O5FlZksrSG9USMWObXh6Cy2LiS2QeAAAAAAAAAEZ/AgAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAHNlbGwAAAAAAAAAAAAAAAAAAAAA"
    "AIgTAAAAAAAALvQDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAC3YkAD"
    "AAAAAPVS2gIHAAAA9aa2BgAAAAA="
)

_TOKEN_ENTRY = {"token_address": _REAL_EVENT_1_MINT, "alert_time": "2026-09-04T00:00:00+00:00",
                 "event_id": "e1"}
# ts=1788833248 is the real event's own timestamp; use it as alert_time so the
# reconstructed row lands close to the vsol_at_signal-style anchor in tests
# that need a comparable point.
_DETAIL_NO_REF = {"price_t1m": None, "price_t3m": None, "price_t5m": None,
                    "price_t10m": None, "price_t20m": None, "vsol_at_signal": None}


def _tx_with_event(block_time=1788833248):
    return {"blockTime": block_time, "meta": {"err": None, "logMessages": [f"Program data: {_REAL_EVENT_1_B64}"]}}


class TestProcessToken(unittest.TestCase):

    def test_no_sigs_gives_no_sigs_status(self):
        with patch("research.backfill_paths._fetch_sigs", return_value=[]):
            with tempfile.TemporaryDirectory() as td:
                result = process_token(_TOKEN_ENTRY, _DETAIL_NO_REF, 200.0, Path(td))
        self.assertEqual(result["status"], "NO_SIGS")
        self.assertFalse(result["valid_usable_path"])

    def test_no_matching_event_gives_no_exact_rows(self):
        with patch("research.backfill_paths._fetch_sigs", return_value=["sig1"]), \
             patch("research.backfill_paths._parse_txs_std",
                   return_value=[{"blockTime": 1, "meta": {"err": None, "logMessages": []}}]):
            with tempfile.TemporaryDirectory() as td:
                result = process_token(_TOKEN_ENTRY, _DETAIL_NO_REF, 200.0, Path(td))
        self.assertEqual(result["status"], "NO_EXACT_ROWS")
        self.assertFalse(result["valid_usable_path"])

    def test_valid_reconstruction_with_insufficient_xval_data_is_usable(self):
        with patch("research.backfill_paths._fetch_sigs", return_value=["sig1"]), \
             patch("research.backfill_paths._parse_txs_std", return_value=[_tx_with_event()]):
            with tempfile.TemporaryDirectory() as td:
                out_dir = Path(td)
                result = process_token(_TOKEN_ENTRY, _DETAIL_NO_REF, 200.0, out_dir)
                self.assertEqual(result["status"], "RECONSTRUCTED")
                self.assertEqual(result["integrity_status"], "VALID")
                self.assertEqual(result["xval_status"], "INSUFFICIENT_DATA")  # no ref points in _DETAIL_NO_REF
                self.assertTrue(result["valid_usable_path"])
                self.assertIsNotNone(result["path_file"])
                self.assertTrue(Path(result["path_file"]).exists())

    def test_xval_fail_marks_path_not_usable_and_does_not_write_file(self):
        # A reference point that disagrees wildly with the real reconstructed
        # price forces xval FAIL.
        bad_detail = dict(_DETAIL_NO_REF)
        bad_detail["price_t1m"] = 999.0  # nowhere near the real ~$0.00004 price
        entry = dict(_TOKEN_ENTRY, alert_time="2026-09-04T00:00:00+00:00")
        with patch("research.backfill_paths._fetch_sigs", return_value=["sig1"]), \
             patch("research.backfill_paths._parse_txs_std", return_value=[_tx_with_event(block_time=1788833248)]):
            with tempfile.TemporaryDirectory() as td:
                out_dir = Path(td)
                # alert 60s before the event's own ts so T1m lands exactly at ts
                import datetime
                alert_dt = datetime.datetime.fromtimestamp(1788833248 - 60, tz=datetime.timezone.utc)
                entry["alert_time"] = alert_dt.isoformat()
                result = process_token(entry, bad_detail, 200.0, out_dir)
        self.assertEqual(result["xval_status"], "FAIL")
        self.assertFalse(result["valid_usable_path"])
        self.assertIsNone(result["path_file"])
        self.assertEqual(list(out_dir.glob("*.csv.gz")), [])


class TestWriteReconstructedCsv(unittest.TestCase):

    def test_written_file_round_trips_through_load_path_file(self):
        rows = [{
            "schema_version": "3", "research_event_id": "", "event_id": "abc",
            "ts_ms": 1000, "price_usd": 0.0001, "price_sol": 0.0000005,
            "side": "buy", "token_amount": 100.0, "sol_amount": 0.01,
            "vsol": 30.0, "vtok": 1000000000, "source": "reconstructed_curve_exact",
            "venue_state": "CURVE_ACTIVE", "backfilled": "true", "data_status": "ok",
            "trader_pk": "",
        }]
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            path_str = _write_reconstructed_csv("TESTMINT", rows, out_dir)
            gz_path = Path(path_str)
            self.assertTrue(gz_path.exists())
            with gzip.open(gz_path, "rt") as f:
                content = f.read()
            self.assertIn("reconstructed_curve_exact", content)
            loaded, warnings = load_path_file(gz_path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["source"], "reconstructed_curve_exact")


class TestRunEndToEnd(unittest.TestCase):

    def test_manifest_written_with_correct_counts(self):
        sample = {"seed": 1, "generated_at": "x", "tokens": [_TOKEN_ENTRY]}
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            sample_file = td_path / "sample.json"
            sample_file.write_text(json.dumps(sample))
            results_file = td_path / "results.json"
            out_dir = td_path / "out"

            fake_sb_data = [dict(_TOKEN_ENTRY, **_DETAIL_NO_REF)]

            class _FakeTable:
                def select(self, *a, **k): return self
                def in_(self, *a, **k): return self
                def execute(self): return type("R", (), {"data": fake_sb_data})()

            class _FakeSb:
                def table(self, *a, **k): return _FakeTable()

            with patch("research.config.SUPABASE_URL", "x"), \
                 patch("research.config.SUPABASE_KEY", "y"), \
                 patch("supabase.create_client", return_value=_FakeSb()), \
                 patch("research.thr_run_t3._get_sol_price", return_value=200.0), \
                 patch("research.backfill_paths._fetch_sigs", return_value=["sig1"]), \
                 patch("research.backfill_paths._parse_txs_std", return_value=[_tx_with_event()]):
                manifest = run(sample_file=sample_file, results_file=results_file, out_dir=out_dir)

            self.assertEqual(manifest["n_attempted"], 1)
            self.assertEqual(manifest["n_reconstructed"], 1)
            self.assertEqual(manifest["n_valid_usable_path"], 1)
            self.assertTrue(results_file.exists())
            saved = json.loads(results_file.read_text())
            self.assertEqual(saved["n_valid_usable_path"], 1)


if __name__ == "__main__":
    unittest.main()
