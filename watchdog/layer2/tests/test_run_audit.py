"""watchdog/layer2/tests/test_run_audit.py — full pipeline wiring test
with fakes, no real SSH connection or Anthropic API key needed."""

import json
import tempfile
import unittest
from pathlib import Path

from watchdog.layer2.findings_store import read_heartbeat
from watchdog.layer2.run_audit import run_audit


class TestRunAuditPipeline(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo_root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _fake_evidence(self):
        return {"collected_at": 1000.0, "host": {"a": 1},
                "receipts_tail": {"mtime": 1.0, "tail": "STATUS: COMPLETE"}}

    def test_full_pipeline_no_findings(self):
        call_count = [0]
        def fake_model(prompt):
            call_count[0] += 1
            if call_count[0] == 1:
                return "ground truth: EV001 shows host is up."
            return "[]"

        sent = []
        result = run_audit(self._fake_evidence(), fake_model, self.repo_root,
                            send_telegram_fn=lambda t: sent.append(t))

        self.assertEqual(result["findings_count"], 0)
        self.assertEqual(call_count[0], 2)  # ground truth pass + comparison pass
        self.assertEqual(sent, [])  # no actionable findings -> no telegram send

        heartbeat = read_heartbeat(self.repo_root)
        self.assertIsNotNone(heartbeat)
        self.assertEqual(heartbeat["findings_count"], 0)

    def test_full_pipeline_with_critical_finding_sends_telegram(self):
        def fake_model(prompt):
            if "Evidence Bundle" in prompt:
                return "ground truth: EV001 shows host up, EV002 claims COMPLETE but no supporting evidence"
            return json.dumps([{
                "finding_id": "F1", "severity": "CRITICAL", "component": "receipts",
                "claim": "STATUS: COMPLETE", "observed_ground_truth": "no supporting evidence",
                "expected_ground_truth": "should be backed by evidence",
                "evidence_ids": ["EV001"], "operational_impact": "claim may be false",
                "suggested_next_step": "check manually", "confidence": "medium",
            }])

        sent = []
        result = run_audit(self._fake_evidence(), fake_model, self.repo_root,
                            send_telegram_fn=lambda t: sent.append(t))

        self.assertEqual(result["findings_count"], 1)
        self.assertEqual(result["critical_count"], 1)
        self.assertEqual(len(sent), 1)
        self.assertIn("CRITICAL", sent[0])
        self.assertIn("EV001", sent[0])

        heartbeat = read_heartbeat(self.repo_root)
        self.assertEqual(heartbeat["critical_count"], 1)

    def test_info_only_finding_does_not_trigger_telegram(self):
        def fake_model(prompt):
            if "Evidence Bundle" in prompt:
                return "ground truth"
            return json.dumps([{
                "finding_id": "F1", "severity": "INFO", "component": "x",
                "evidence_ids": ["EV001"], "claim": "", "observed_ground_truth": "",
                "expected_ground_truth": "", "operational_impact": "", "suggested_next_step": "",
                "confidence": "low",
            }])
        sent = []
        run_audit(self._fake_evidence(), fake_model, self.repo_root,
                   send_telegram_fn=lambda t: sent.append(t))
        self.assertEqual(sent, [])

    def test_no_send_telegram_fn_does_not_crash(self):
        def fake_model(prompt):
            return "ground truth" if "Evidence Bundle" in prompt else "[]"
        result = run_audit(self._fake_evidence(), fake_model, self.repo_root, send_telegram_fn=None)
        self.assertEqual(result["findings_count"], 0)

    def test_audit_artifacts_written_to_disk(self):
        def fake_model(prompt):
            return "ground truth" if "Evidence Bundle" in prompt else "[]"
        result = run_audit(self._fake_evidence(), fake_model, self.repo_root)
        audit_dir = self.repo_root / "logs" / "watchdog" / "audits" / result["audit_id"]
        self.assertTrue((audit_dir / "evidence.json").exists())
        self.assertTrue((audit_dir / "findings.md").exists())


class TestAnthropicAdapterEmptyResponse(unittest.TestCase):
    """Regression for a real bug found on the first live GitHub Actions
    run: the ground-truth pass returned 0 chars of text, and the
    comparison pass still produced "6 valid findings" grounded in
    nothing. make_anthropic_call_model must refuse to silently return
    empty text -- an empty ground truth means findings would be compared
    against nothing, not that nothing is wrong."""

    def _fake_response(self, content_blocks, stop_reason="end_turn"):
        from unittest.mock import MagicMock
        resp = MagicMock()
        resp.content = content_blocks
        resp.stop_reason = stop_reason
        resp.usage = MagicMock()
        return resp

    def test_empty_text_response_raises_not_silently_returns_empty(self):
        from unittest.mock import MagicMock, patch
        from watchdog.layer2.run_audit import make_anthropic_call_model

        # Simulate a response with only a non-text block (e.g. the model
        # stopped before producing any real text) -- the exact shape of
        # the real failure is unknown (no raw response was captured on
        # the live run), so this covers the general "no usable text"
        # case regardless of which specific block type caused it.
        empty_block = MagicMock(spec=[])  # no .text attribute at all
        fake_response = self._fake_response([empty_block], stop_reason="max_tokens")

        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = fake_response
            call_model = make_anthropic_call_model("fake-api-key")
            with self.assertRaises(RuntimeError) as ctx:
                call_model("some prompt")
            self.assertIn("empty", str(ctx.exception).lower())
            self.assertIn("max_tokens", str(ctx.exception))

    def test_whitespace_only_response_also_raises(self):
        from unittest.mock import MagicMock, patch
        from watchdog.layer2.run_audit import make_anthropic_call_model

        text_block = MagicMock()
        text_block.text = "   \n  "
        fake_response = self._fake_response([text_block])

        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = fake_response
            call_model = make_anthropic_call_model("fake-api-key")
            with self.assertRaises(RuntimeError):
                call_model("some prompt")

    def test_real_text_response_passes_through_normally(self):
        from unittest.mock import MagicMock, patch
        from watchdog.layer2.run_audit import make_anthropic_call_model

        text_block = MagicMock()
        text_block.text = "real ground truth content, EV001 shows X"
        fake_response = self._fake_response([text_block])

        with patch("anthropic.Anthropic") as MockClient:
            MockClient.return_value.messages.create.return_value = fake_response
            call_model = make_anthropic_call_model("fake-api-key")
            result = call_model("some prompt")
        self.assertEqual(result, "real ground truth content, EV001 shows X")


if __name__ == "__main__":
    unittest.main()
