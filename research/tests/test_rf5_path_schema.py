"""
RF5 path schema tests.

All tests use unittest + mock only.
No network calls, no actual file writes (uses tmp_path via tempfile).
"""

import csv
import gzip
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure the project root is on the path
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from research.path_schema import (
    PATH_SCHEMA_VERSION,
    PATH_REQUIRED_FIELDS,
    PATH_HEADER,
    validate_row,
    normalise_legacy_row,
    load_path_file,
)


def _make_valid_live_row(**overrides) -> dict:
    """Return a complete, valid live row dict."""
    row = {
        "schema_version":    str(PATH_SCHEMA_VERSION),
        "research_event_id": "abc123",
        "event_id":          "evt456",
        "ts_ms":             "1700000000000",
        "price_usd":         "0.000001234",
        "price_sol":         "0.000000007",
        "side":              "buy",
        "token_amount":      "1000000",
        "sol_amount":        "0.5",
        "vsol":              "50.0",
        "source":            "live_pp",
        "venue_state":       "CURVE_ACTIVE",
        "backfilled":        "false",
        "data_status":       "ok",
    }
    row.update(overrides)
    return row


def _make_valid_backfill_row(**overrides) -> dict:
    """Return a complete, valid backfill row dict."""
    row = {
        "schema_version":    str(PATH_SCHEMA_VERSION),
        "research_event_id": "uuid-1234",
        "event_id":          "deadbeef" * 4,
        "ts_ms":             "1700000001000",
        "price_usd":         "0.000000987",
        "price_sol":         "0.0",
        "side":              "unknown",
        "token_amount":      "0",
        "sol_amount":        "0.3",
        "vsol":              "0",
        "source":            "backfill_helius",
        "venue_state":       "UNKNOWN",
        "backfilled":        "true",
        "data_status":       "ok",
    }
    row.update(overrides)
    return row


def _write_csv_to_tmp(rows: list[dict], *, gzipped: bool = False) -> Path:
    """Write rows (list of dicts) to a temp file, return Path."""
    suffix = ".csv.gz" if gzipped else ".csv"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    tmp_path = Path(tmp)
    if gzipped:
        opener = gzip.open(tmp_path, "wt", newline="", encoding="utf-8")
    else:
        opener = open(tmp_path, "w", newline="", encoding="utf-8")
    with opener as f:
        writer = csv.DictWriter(f, fieldnames=PATH_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return tmp_path


class TestValidateRow(unittest.TestCase):

    def test_valid_live_row_passes_validation(self):
        row = _make_valid_live_row()
        valid, err = validate_row(row)
        self.assertTrue(valid, f"Expected valid, got error: {err}")
        self.assertEqual(err, "")

    def test_valid_backfill_row_with_unknown_side(self):
        row = _make_valid_backfill_row()
        valid, err = validate_row(row)
        self.assertTrue(valid, f"Expected valid for backfill+unknown side: {err}")

    def test_missing_required_field_produces_partial_status(self):
        # ts_ms is required — remove it
        row = _make_valid_live_row()
        del row["ts_ms"]
        valid, err = validate_row(row)
        self.assertFalse(valid)
        self.assertIn("ts_ms", err)

    def test_missing_price_usd_fails(self):
        row = _make_valid_live_row()
        del row["price_usd"]
        valid, err = validate_row(row)
        self.assertFalse(valid)
        self.assertIn("price_usd", err)

    def test_missing_backfilled_fails(self):
        row = _make_valid_live_row()
        del row["backfilled"]
        valid, err = validate_row(row)
        self.assertFalse(valid)
        self.assertIn("backfilled", err)

    def test_live_row_with_unknown_side_fails_not_hard_error(self):
        """Live rows with 'unknown' side are invalid (not a hard exception)."""
        row = _make_valid_live_row(side="unknown")
        valid, err = validate_row(row)
        self.assertFalse(valid)
        self.assertIn("live row", err)

    def test_invalid_ts_ms_fails(self):
        row = _make_valid_live_row(ts_ms="not_a_number")
        valid, err = validate_row(row)
        self.assertFalse(valid)
        self.assertIn("ts_ms", err)

    def test_invalid_price_usd_fails(self):
        row = _make_valid_live_row(price_usd="abc")
        valid, err = validate_row(row)
        self.assertFalse(valid)
        self.assertIn("price_usd", err)

    def test_invalid_backfilled_value_fails(self):
        row = _make_valid_live_row(backfilled="yes")
        valid, err = validate_row(row)
        self.assertFalse(valid)
        self.assertIn("backfilled", err)

    def test_invalid_side_fails(self):
        row = _make_valid_live_row(side="long")
        valid, err = validate_row(row)
        self.assertFalse(valid)
        self.assertIn("side", err)


class TestNormaliseLegacyRow(unittest.TestCase):

    def test_legacy_row_normalised_to_schema_v1(self):
        """Old PC1 row (5 columns, no schema_version) is normalised to v0 with defaults."""
        legacy = {
            "ts_ms":      "1700000000000",
            "price_usd":  "0.000001",
            "side":       "buy",
            "sol_amount": "0.5",
            "vsol":       "50.0",
        }
        result = normalise_legacy_row(legacy)
        self.assertEqual(result["schema_version"], "0")
        self.assertEqual(result["backfilled"], "false")   # vsol>0 → live
        self.assertEqual(result["source"], "live_pp")
        self.assertIn("data_status", result)
        # All PATH_HEADER fields must be present
        for col in PATH_HEADER:
            self.assertIn(col, result, f"Missing column: {col}")

    def test_legacy_backfill_row_normalised(self):
        """Old PC2 row (6 columns with source=backfill) is normalised correctly."""
        legacy = {
            "ts_ms":      "1700000002000",
            "price_usd":  "0.0000005",
            "side":       "buy",
            "sol_amount": "0.1",
            "vsol":       "0",
            "source":     "backfill",
        }
        result = normalise_legacy_row(legacy)
        self.assertEqual(result["schema_version"], "0")
        self.assertEqual(result["backfilled"], "true")    # source contains "backfill"
        for col in PATH_HEADER:
            self.assertIn(col, result, f"Missing column: {col}")

    def test_legacy_row_with_missing_side_gets_unknown(self):
        legacy = {
            "ts_ms":     "1700000003000",
            "price_usd": "0.000002",
            "sol_amount": "0.2",
            "vsol":      "30.0",
        }
        result = normalise_legacy_row(legacy)
        self.assertEqual(result["side"], "unknown")
        # Live row with unknown side → data_status=partial
        self.assertEqual(result["data_status"], "partial")

    def test_schema_version_compatibility_v0_then_v1(self):
        """v0 row normalised; v1 row passes validate_row correctly."""
        # v0 legacy row
        legacy = {
            "ts_ms": "1700000005000",
            "price_usd": "0.000003",
            "side": "sell",
            "sol_amount": "0.4",
            "vsol": "60.0",
        }
        normalised = normalise_legacy_row(legacy)
        self.assertEqual(normalised["schema_version"], "0")

        # v1 row passes validate_row directly
        v1_row = _make_valid_live_row()
        valid, err = validate_row(v1_row)
        self.assertTrue(valid, err)

    def test_normalise_never_raises(self):
        """normalise_legacy_row must not raise on pathological input."""
        for bad_input in [{}, {"side": None, "ts_ms": "", "price_usd": "x"}]:
            try:
                result = normalise_legacy_row(bad_input)
                self.assertIsInstance(result, dict)
            except Exception as e:
                self.fail(f"normalise_legacy_row raised on {bad_input!r}: {e}")


class TestUnknownFutureField(unittest.TestCase):

    def test_unknown_future_field_silently_ignored(self):
        """Fields in the file that aren't in PATH_HEADER are stripped by load_path_file."""
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(suffix=".csv")
            os.close(fd)
            tmp_path = Path(tmp)
            extended_header = PATH_HEADER + ["future_field_xyz", "another_new_col"]
            with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=extended_header)
                writer.writeheader()
                row = _make_valid_live_row()
                row["future_field_xyz"] = "some_value"
                row["another_new_col"] = "42"
                writer.writerow(row)

            rows, warnings = load_path_file(tmp_path)
            self.assertEqual(len(rows), 1)
            # Unknown fields must NOT appear in the returned row
            self.assertNotIn("future_field_xyz", rows[0])
            self.assertNotIn("another_new_col", rows[0])
        finally:
            if tmp and Path(tmp).exists():
                os.unlink(tmp)


class TestLoadPathFile(unittest.TestCase):

    def test_load_path_file_handles_gzip(self):
        """load_path_file correctly reads .csv.gz files."""
        rows_in = [_make_valid_live_row(), _make_valid_backfill_row()]
        tmp = _write_csv_to_tmp(rows_in, gzipped=True)
        try:
            rows, warnings = load_path_file(tmp)
            self.assertEqual(len(rows), 2)
        finally:
            tmp.unlink(missing_ok=True)

    def test_load_path_file_returns_warnings_on_bad_rows_not_raise(self):
        """Rows with missing required fields produce warnings, not exceptions."""
        fd, tmp = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        tmp_path = Path(tmp)
        try:
            with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(PATH_HEADER)
                # Good row
                good = _make_valid_live_row()
                writer.writerow([good.get(c, "") for c in PATH_HEADER])
                # Bad row — ts_ms missing
                bad = _make_valid_live_row(ts_ms="")
                writer.writerow([bad.get(c, "") for c in PATH_HEADER])

            rows, warnings = load_path_file(tmp_path)
            # Both rows returned (bad one with data_status=partial)
            self.assertEqual(len(rows), 2)
            partial_rows = [r for r in rows if r.get("data_status") == "partial"]
            self.assertGreater(len(partial_rows), 0)
            self.assertGreater(len(warnings), 0)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_load_path_file_returns_empty_on_missing_file(self):
        """load_path_file must not raise on a non-existent file."""
        rows, warnings = load_path_file(Path("/nonexistent/path/fake.csv"))
        self.assertEqual(rows, [])
        self.assertGreater(len(warnings), 0)

    def test_load_path_file_sorted_by_ts_ms(self):
        """Rows are returned sorted by ts_ms ascending."""
        row1 = _make_valid_live_row(ts_ms="1700000005000")
        row2 = _make_valid_live_row(ts_ms="1700000001000", side="sell")
        tmp = _write_csv_to_tmp([row1, row2])
        try:
            rows, _ = load_path_file(tmp)
            self.assertEqual(len(rows), 2)
            self.assertLessEqual(int(rows[0]["ts_ms"]), int(rows[1]["ts_ms"]))
        finally:
            tmp.unlink(missing_ok=True)

    def test_load_path_file_normalises_legacy_rows(self):
        """Legacy rows (no schema_version) are normalised, not rejected."""
        fd, tmp = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        tmp_path = Path(tmp)
        try:
            # Write with old 5-column header (no schema_version)
            legacy_header = ["ts_ms", "price_usd", "side", "sol_amount", "vsol"]
            with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(legacy_header)
                writer.writerow(["1700000010000", "0.000002", "buy", "0.5", "45.0"])

            rows, warnings = load_path_file(tmp_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["schema_version"], "0")
            self.assertEqual(rows[0]["backfilled"], "false")
        finally:
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# N7(a): trader_pk field (post-measurement batch, 2026-07-30)
# ---------------------------------------------------------------------------

class TestTraderPk(unittest.TestCase):

    def test_trader_pk_in_header(self):
        self.assertIn("trader_pk", PATH_HEADER)

    def test_trader_pk_not_required(self):
        """Old rows (pre-N7a) must not be rejected for lacking trader_pk."""
        self.assertNotIn("trader_pk", PATH_REQUIRED_FIELDS)

    def test_row_missing_trader_pk_defaults_to_empty_string(self):
        row = _make_valid_live_row()
        row.pop("trader_pk", None)
        normalised = normalise_legacy_row(row)
        self.assertEqual(normalised["trader_pk"], "")
        valid, err = validate_row(normalised)
        self.assertTrue(valid, err)

    def test_row_with_trader_pk_preserved(self):
        row = _make_valid_live_row(trader_pk="7xKXtg2CW3xkV4wtjaVwNXQ1Vsswz1SB4NQTQfxL7Q4x")
        normalised = normalise_legacy_row(row)
        self.assertEqual(normalised["trader_pk"], "7xKXtg2CW3xkV4wtjaVwNXQ1Vsswz1SB4NQTQfxL7Q4x")

    def test_load_path_file_tolerates_pre_n7a_rows(self):
        """A schema_version=1 file (no trader_pk column at all) must still load."""
        fd, tmp = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        tmp_path = Path(tmp)
        try:
            v1_header = [c for c in PATH_HEADER if c != "trader_pk"]
            with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(v1_header)
                writer.writerow(["1", "", "", "1700000010000", "0.000002", "0.0",
                                  "buy", "0", "0.5", "45.0", "live_pp",
                                  "CURVE_ACTIVE", "false", "ok"])
            rows, warnings = load_path_file(tmp_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].get("trader_pk"), "")
            self.assertEqual(rows[0]["data_status"], "ok")
        finally:
            tmp_path.unlink(missing_ok=True)


class TestPeakTrackerSchemaConsistency(unittest.TestCase):
    """Regression test for the _SCHEMA_VER/_CSV_HEADER bug (N7a fix, 2026-07-30):
    peak_tracker.py had a stale local 5-column _CSV_HEADER and referenced an
    undefined _SCHEMA_VER name in its tick writerow() call — every tick write
    raised NameError, silently swallowed by a bare except, leaving every path
    file on disk header-only. This asserts the module now sources both from
    the canonical schema instead of a local/undefined copy.
    """

    def test_csv_header_matches_canonical_path_header(self):
        import research.peak_tracker as pt
        self.assertEqual(pt._CSV_HEADER, PATH_HEADER)

    def test_schema_ver_matches_canonical_version(self):
        import research.peak_tracker as pt
        self.assertEqual(pt._SCHEMA_VER, PATH_SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
