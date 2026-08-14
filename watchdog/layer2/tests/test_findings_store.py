"""watchdog/layer2/tests/test_findings_store.py"""

import json
import tempfile
import unittest
from pathlib import Path

from watchdog.layer2.audit_prompt import Finding
from watchdog.layer2.evidence_bundle import build_evidence_bundle
from watchdog.layer2.findings_store import read_heartbeat, write_audit_artifacts, write_heartbeat


class TestFindingsStore(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo_root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _bundle(self):
        return build_evidence_bundle({"collected_at": 1.0, "host": {"a": 1}}, audit_id="audit-test1")

    def _finding(self, severity="CRITICAL"):
        return Finding(finding_id="F1", severity=severity, component="cron", claim="c",
                        observed_ground_truth="o", expected_ground_truth="e",
                        evidence_ids=["EV001"], operational_impact="i",
                        suggested_next_step="n", confidence="high", audit_id="audit-test1")

    def test_writes_all_expected_files(self):
        bundle = self._bundle()
        audit_dir = write_audit_artifacts(self.repo_root, bundle, "ground truth text",
                                           [self._finding()], "[{}]")
        self.assertTrue((audit_dir / "evidence.json").exists())
        self.assertTrue((audit_dir / "ground_truth.txt").exists())
        self.assertTrue((audit_dir / "findings.json").exists())
        self.assertTrue((audit_dir / "findings.md").exists())
        self.assertEqual((audit_dir / "ground_truth.txt").read_text(), "ground truth text")

    def test_findings_json_roundtrips(self):
        bundle = self._bundle()
        f = self._finding()
        audit_dir = write_audit_artifacts(self.repo_root, bundle, "gt", [f], "[]")
        data = json.loads((audit_dir / "findings.json").read_text())
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["evidence_ids"], ["EV001"])

    def test_empty_findings_produces_readable_md(self):
        bundle = self._bundle()
        audit_dir = write_audit_artifacts(self.repo_root, bundle, "gt", [], "[]")
        md = (audit_dir / "findings.md").read_text()
        self.assertIn("No findings", md)

    def test_findings_md_orders_by_severity(self):
        bundle = self._bundle()
        findings = [self._finding("INFO"), self._finding("CRITICAL"), self._finding("WARN")]
        audit_dir = write_audit_artifacts(self.repo_root, bundle, "gt", findings, "[]")
        md = (audit_dir / "findings.md").read_text()
        crit_pos = md.index("[CRITICAL]")
        warn_pos = md.index("[WARN]")
        info_pos = md.index("[INFO]")
        self.assertLess(crit_pos, warn_pos)
        self.assertLess(warn_pos, info_pos)

    def test_repeated_audit_ids_do_not_collide_with_different_content(self):
        """Different audit_id -> different directory; same audit_id would
        overwrite (by design -- a re-run of the SAME audit_id is
        idempotent, not a historical rewrite of a DIFFERENT one)."""
        bundle1 = build_evidence_bundle({"collected_at": 1.0, "host": {"a": 1}}, audit_id="audit-A")
        bundle2 = build_evidence_bundle({"collected_at": 2.0, "host": {"a": 2}}, audit_id="audit-B")
        d1 = write_audit_artifacts(self.repo_root, bundle1, "gt1", [], "[]")
        d2 = write_audit_artifacts(self.repo_root, bundle2, "gt2", [], "[]")
        self.assertNotEqual(d1, d2)
        self.assertTrue(d1.exists())
        self.assertTrue(d2.exists())

    def test_heartbeat_roundtrip(self):
        bundle = self._bundle()
        write_heartbeat(self.repo_root, bundle, [self._finding()], now_ts=12345.0)
        data = read_heartbeat(self.repo_root)
        self.assertEqual(data["last_audit_id"], "audit-test1")
        self.assertEqual(data["last_audit_completed_at"], 12345.0)
        self.assertEqual(data["critical_count"], 1)

    def test_missing_heartbeat_returns_none(self):
        self.assertIsNone(read_heartbeat(self.repo_root))

    def test_corrupt_heartbeat_returns_none_not_raises(self):
        path = self.repo_root / "logs" / "watchdog" / "layer2_heartbeat.json"
        path.parent.mkdir(parents=True)
        path.write_text("not json {{{")
        self.assertIsNone(read_heartbeat(self.repo_root))


if __name__ == "__main__":
    unittest.main()
