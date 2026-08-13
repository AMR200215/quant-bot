"""watchdog/tests/test_batch_claims.py — W9 fault injection.

Uses real fixture batch manifests + a real RECEIPTS.md-shaped fixture
file (via tools.batch_verify's own path constants, patched) rather than
mocking verify_batch() itself -- the whole point of this check is
correctly interpreting batch_verify's real output, so exercising the
real function end to end is the stronger test.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from watchdog.checks import STATUS_CRITICAL, STATUS_OK, STATUS_WARN
from watchdog.checks.batch_claims import check_batch_verify


class TestBatchClaimsFaultInjection(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.batches_dir = self.root / "batches"
        self.batches_dir.mkdir()
        self.receipts = self.root / "docs" / "RECEIPTS.md"
        self.receipts.parent.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def _patch_bv_paths(self, bv):
        return patch.multiple(
            bv, _BATCHES_DIR=self.batches_dir, _RECEIPTS=self.receipts,
        )

    def test_all_green_batch_is_ok(self):
        import tools.batch_verify as bv
        self.receipts.write_text("## MYBATCH\ncommit abc123 done\n")
        (self.batches_dir / "mybatch.yaml").write_text(
            "batch_id: mybatch\ncommit: abc123\nitems:\n"
            "  - id: A1\n    receipts_section: MYBATCH\n    receipt_complete: true\n"
        )
        with self._patch_bv_paths(bv):
            results = check_batch_verify()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, STATUS_OK)

    def test_partial_batch_is_warn_not_silently_ok(self):
        """The exact real gap: batch_verify's own CLI exits 0 for this,
        identical to full-GREEN. Must be WARN here, not OK."""
        import tools.batch_verify as bv
        self.receipts.write_text("## MYBATCH\nsome other commit here\n")
        (self.batches_dir / "mybatch.yaml").write_text(
            "batch_id: mybatch\ncommit: abc123\nitems:\n"
            "  - id: A1\n    receipts_section: MYBATCH\n    receipt_complete: true\n"
        )
        with self._patch_bv_paths(bv):
            results = check_batch_verify()
        self.assertEqual(results[0].status, STATUS_WARN)
        self.assertIn("A1", results[0].evidence["partial_item_ids"])

    def test_fail_batch_is_capped_at_default_warn_ceiling(self):
        """Default severity_ceiling=WARN -- a hard FAIL is a documentation/
        completeness gap, not by default a page-worthy production
        incident. Must still be WARN, never silently OK."""
        import tools.batch_verify as bv
        (self.batches_dir / "mybatch.yaml").write_text(
            "batch_id: mybatch\ncommit: abc123\nitems:\n"
            "  - id: A1\n    receipts_section: DOES_NOT_EXIST\n    receipt_complete: true\n"
        )
        with self._patch_bv_paths(bv):
            results = check_batch_verify()
        self.assertEqual(results[0].status, STATUS_WARN)
        self.assertIn("A1", results[0].evidence["fail_item_ids"])

    def test_fail_batch_reaches_critical_when_ceiling_allows(self):
        import tools.batch_verify as bv
        (self.batches_dir / "mybatch.yaml").write_text(
            "batch_id: mybatch\ncommit: abc123\nitems:\n"
            "  - id: A1\n    receipts_section: DOES_NOT_EXIST\n    receipt_complete: true\n"
        )
        with self._patch_bv_paths(bv):
            results = check_batch_verify(severity_ceiling="CRITICAL")
        self.assertEqual(results[0].status, STATUS_CRITICAL)

    def test_no_manifests_is_ok(self):
        import tools.batch_verify as bv
        with self._patch_bv_paths(bv):
            results = check_batch_verify()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, STATUS_OK)

    def test_receipt_complete_false_is_partial_not_ok(self):
        import tools.batch_verify as bv
        self.receipts.write_text("## MYBATCH\ncommit abc123\n")
        (self.batches_dir / "mybatch.yaml").write_text(
            "batch_id: mybatch\ncommit: abc123\nitems:\n"
            "  - id: A1\n    receipts_section: MYBATCH\n    receipt_complete: false\n"
        )
        with self._patch_bv_paths(bv):
            results = check_batch_verify()
        self.assertEqual(results[0].status, STATUS_WARN)


if __name__ == "__main__":
    unittest.main()
