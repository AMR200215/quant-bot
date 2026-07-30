"""
N6: v8_vs_v7_daily.py — tests for the RECEIPTS.md upsert logic and day-stats
aggregation. No real file writes to docs/RECEIPTS.md (uses in-memory strings
/ tmp files).

Run: python -m pytest research/tests/test_v8_vs_v7_daily.py -v
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from research.scripts.v8_vs_v7_daily import _day_stats, _upsert_row, _SECTION_HEADER


class TestUpsertRow(unittest.TestCase):

    def test_creates_section_when_absent(self):
        text = "# Doc\n\nsome content\n"
        out = _upsert_row(text, "2026-07-30", "| 2026-07-30 | 1 | +1.0% | 1 | +1.0% | g |")
        self.assertIn(_SECTION_HEADER, out)
        self.assertIn("2026-07-30", out)

    def test_rerun_same_day_replaces_not_duplicates(self):
        text = "# Doc\n"
        out = _upsert_row(text, "2026-07-30", "| 2026-07-30 | 1 | +1.0% | 1 | +1.0% | g |")
        out2 = _upsert_row(out, "2026-07-30", "| 2026-07-30 | 9 | +9.0% | 9 | +9.0% | g |")
        self.assertEqual(out2.count("2026-07-30"), 1)
        self.assertIn("| 9 |", out2)

    def test_second_day_appends_without_disturbing_first(self):
        text = "# Doc\n"
        out = _upsert_row(text, "2026-07-30", "| 2026-07-30 | 1 | +1.0% | 1 | +1.0% | g |")
        out2 = _upsert_row(out, "2026-07-31", "| 2026-07-31 | 2 | +2.0% | 2 | +2.0% | g |")
        self.assertIn("2026-07-30", out2)
        self.assertIn("2026-07-31", out2)

    def test_does_not_disturb_other_sections(self):
        text = "# Doc\n\n### Other section\nkeep me\n"
        out = _upsert_row(text, "2026-07-30", "| 2026-07-30 | 1 | +1.0% | 1 | +1.0% | g |")
        self.assertIn("### Other section", out)
        self.assertIn("keep me", out)


class TestDayStats(unittest.TestCase):

    def _write_journal(self, rows):
        fd, path = tempfile.mkstemp(suffix=".csv")
        import os
        os.close(fd)
        p = Path(path)
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["exit_time", "pnl_pct"])
            writer.writeheader()
            writer.writerows(rows)
        return p

    def test_missing_file_returns_zero(self):
        n, pct = _day_stats(Path("/tmp/does-not-exist-v8-test.csv"), "2026-07-30")
        self.assertEqual((n, pct), (0, 0.0))

    def test_filters_by_day_prefix(self):
        p = self._write_journal([
            {"exit_time": "2026-07-30 10:00:00", "pnl_pct": "10.0"},
            {"exit_time": "2026-07-30 11:00:00", "pnl_pct": "-2.0"},
            {"exit_time": "2026-07-29 09:00:00", "pnl_pct": "50.0"},
            {"exit_time": "", "pnl_pct": ""},   # still open — excluded
        ])
        try:
            n, mean_pct = _day_stats(p, "2026-07-30")
            self.assertEqual(n, 2)
            self.assertAlmostEqual(mean_pct, 4.0)   # (10 + -2) / 2
        finally:
            p.unlink(missing_ok=True)

    def test_bad_pnl_value_does_not_raise(self):
        p = self._write_journal([{"exit_time": "2026-07-30 10:00:00", "pnl_pct": "garbage"}])
        try:
            n, mean_pct = _day_stats(p, "2026-07-30")
            self.assertEqual(n, 1)
            self.assertEqual(mean_pct, 0.0)
        finally:
            p.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
