"""
C5 — Execution receipts: 7 unit tests.
All tests use temp files, never the real logs/ dir. R6: calls production API directly.
"""
import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from memecoin.execution_receipts import write_receipt, read_receipts


class TestExecutionReceipts(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        self._tmp.close()
        self._path = self._tmp.name

    def tearDown(self):
        try:
            os.unlink(self._path)
        except OSError:
            pass

    def test_write_receipt_produces_valid_json_line(self):
        """write_receipt appends a valid JSON line to the file."""
        write_receipt(
            {"sig": "abc123", "action": "buy", "mint": "mintA", "sol_amount": 0.01},
            _path_override=self._path,
        )
        with open(self._path) as f:
            lines = [l.strip() for l in f if l.strip()]
        self.assertEqual(len(lines), 1)
        rec = json.loads(lines[0])
        self.assertEqual(rec["sig"], "abc123")
        self.assertEqual(rec["action"], "buy")

    def test_read_receipts_returns_all_when_no_filter(self):
        """read_receipts with no filters returns all written receipts."""
        for i in range(3):
            write_receipt({"sig": f"sig{i}", "action": "sell", "mint": f"mint{i}"},
                          _path_override=self._path)
        results = read_receipts(_path_override=self._path)
        self.assertEqual(len(results), 3)

    def test_read_receipts_filters_by_mint(self):
        """read_receipts(mint=...) returns only receipts for that mint."""
        write_receipt({"sig": "s1", "action": "buy", "mint": "mintA"}, _path_override=self._path)
        write_receipt({"sig": "s2", "action": "buy", "mint": "mintB"}, _path_override=self._path)
        results = read_receipts(mint="mintA", _path_override=self._path)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["mint"], "mintA")

    def test_read_receipts_filters_by_pos_id(self):
        """read_receipts(pos_id=...) returns only receipts for that pos_id."""
        write_receipt({"sig": "s1", "action": "buy", "pos_id": "pos_X"}, _path_override=self._path)
        write_receipt({"sig": "s2", "action": "buy", "pos_id": "pos_Y"}, _path_override=self._path)
        results = read_receipts(pos_id="pos_X", _path_override=self._path)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["pos_id"], "pos_X")

    def test_read_receipts_filters_by_since_ts(self):
        """read_receipts(since_ts=...) excludes receipts before the cutoff."""
        # Write a receipt explicitly with an old timestamp
        old = {
            "sig": "old_sig",
            "action": "buy",
            "sent_at_utc": "2020-01-01T00:00:00.000000Z",
        }
        with open(self._path, "a") as f:
            f.write(json.dumps(old) + "\n")
        # Write a fresh receipt (auto-ts = now)
        write_receipt({"sig": "new_sig", "action": "buy"}, _path_override=self._path)
        cutoff = 1_700_000_000.0  # Nov 2023 — well after 2020, well before now
        results = read_receipts(since_ts=cutoff, _path_override=self._path)
        sigs = [r["sig"] for r in results]
        self.assertNotIn("old_sig", sigs)
        self.assertIn("new_sig", sigs)

    def test_secret_fields_are_stripped(self):
        """Fields with secret names are not written to disk."""
        write_receipt(
            {
                "sig": "s1",
                "action": "buy",
                "private_key": "SUPERSECRET",
                "api_key": "ALSOSECRET",
                "sol_amount": 0.05,
            },
            _path_override=self._path,
        )
        with open(self._path) as f:
            content = f.read()
        self.assertNotIn("SUPERSECRET", content)
        self.assertNotIn("ALSOSECRET", content)
        # non-secret fields still present
        self.assertIn("sol_amount", content)

    def test_write_receipt_never_raises_on_bad_input(self):
        """write_receipt silently ignores None or empty input."""
        # None input
        write_receipt(None, _path_override=self._path)  # type: ignore
        # Empty dict
        write_receipt({}, _path_override=self._path)
        # Missing sig/action
        write_receipt({"foo": None, "bar": float("nan")}, _path_override=self._path)
        # File should still be readable
        results = read_receipts(_path_override=self._path)
        # At least the two valid empty-ish dicts should appear
        self.assertGreaterEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
