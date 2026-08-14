"""watchdog/layer2/tests/test_audit_prompt.py — W15/W16 fault injection.

Uses a fake call_model function -- no real API calls, no API key needed.
The point of these tests is proving the STRUCTURAL ordering guarantee
(Call 1 literally cannot see claims text, because it was never in the
prompt) and the findings-parsing robustness, not testing real model
output quality.
"""

import unittest

from watchdog.layer2.audit_prompt import (
    Finding, parse_findings, run_comparison_pass, run_ground_truth_pass,
)
from watchdog.layer2.evidence_bundle import build_evidence_bundle


class TestOrderingGuarantee(unittest.TestCase):

    def test_ground_truth_pass_prompt_never_contains_claims_text(self):
        """The core W15 guarantee, proven structurally: capture exactly
        what prompt text was sent to the model in Call 1, assert the
        claims content is nowhere in it."""
        raw = {"collected_at": 1.0, "host": {"a": 1},
               "receipts_tail": {"mtime": 1.0, "tail": "STATUS: FULLY_LIVE_VERIFIED_TRUST_ME"}}
        bundle = build_evidence_bundle(raw)

        captured_prompts = []
        def fake_model(prompt):
            captured_prompts.append(prompt)
            return "ground truth: EV001 shows host state (a=1)."

        run_ground_truth_pass(fake_model, bundle)
        self.assertEqual(len(captured_prompts), 1)
        self.assertNotIn("FULLY_LIVE_VERIFIED_TRUST_ME", captured_prompts[0])

    def test_comparison_pass_receives_its_own_prior_ground_truth(self):
        raw = {"collected_at": 1.0, "host": {"a": 1}}
        bundle = build_evidence_bundle(raw)
        prior_ground_truth = "EV001 shows host is up (a=1). No systemd evidence present."

        captured_prompts = []
        def fake_model(prompt):
            captured_prompts.append(prompt)
            return "[]"

        run_comparison_pass(fake_model, prior_ground_truth, bundle)
        self.assertIn(prior_ground_truth, captured_prompts[0])

    def test_comparison_pass_receives_claims_labeled_as_assertions_not_evidence(self):
        raw = {"collected_at": 1.0, "receipts_tail": {"mtime": 1.0, "tail": "COMPLETE"}}
        bundle = build_evidence_bundle(raw)
        captured = []
        def fake_model(prompt):
            captured.append(prompt)
            return "[]"
        run_comparison_pass(fake_model, "ground truth here", bundle)
        self.assertIn("ASSERTIONS, not evidence", captured[0])
        self.assertIn("COMPLETE", captured[0])


class TestFindingsParsing(unittest.TestCase):

    def test_valid_findings_parsed(self):
        raw = '''[{"finding_id": "F1", "severity": "critical", "component": "cron",
                    "claim": "job runs daily", "observed_ground_truth": "no receipt",
                    "expected_ground_truth": "receipt should exist",
                    "evidence_ids": ["EV003"], "operational_impact": "job may be dead",
                    "suggested_next_step": "check journalctl", "confidence": "high"}]'''
        findings = parse_findings(raw, audit_id="a1")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].severity, "CRITICAL")
        self.assertEqual(findings[0].evidence_ids, ["EV003"])
        self.assertEqual(findings[0].audit_id, "a1")

    def test_finding_without_evidence_ids_is_discarded(self):
        """A finding whose factual premise can't be tied to evidence is invalid
        per W14's explicit requirement -- must not survive parsing."""
        raw = '''[{"finding_id": "F1", "severity": "CRITICAL", "component": "x",
                    "evidence_ids": []}]'''
        findings = parse_findings(raw)
        self.assertEqual(len(findings), 0)

    def test_invalid_severity_is_discarded(self):
        raw = '''[{"finding_id": "F1", "severity": "SUPER_BAD", "component": "x",
                    "evidence_ids": ["EV001"]}]'''
        findings = parse_findings(raw)
        self.assertEqual(len(findings), 0)

    def test_empty_array_is_valid_zero_findings(self):
        findings = parse_findings("[]")
        self.assertEqual(findings, [])

    def test_malformed_json_returns_empty_not_raises(self):
        findings = parse_findings("this is not json at all { [ broken")
        self.assertEqual(findings, [])

    def test_markdown_fenced_json_is_unwrapped(self):
        raw = '```json\n[{"finding_id": "F1", "severity": "WARN", "component": "x", "evidence_ids": ["EV001"]}]\n```'
        findings = parse_findings(raw)
        self.assertEqual(len(findings), 1)

    def test_non_list_json_returns_empty(self):
        findings = parse_findings('{"not": "a list"}')
        self.assertEqual(findings, [])

    def test_non_dict_items_in_list_are_skipped(self):
        raw = '["just a string", {"finding_id": "F1", "severity": "WARN", "component": "x", "evidence_ids": ["EV001"]}]'
        findings = parse_findings(raw)
        self.assertEqual(len(findings), 1)


if __name__ == "__main__":
    unittest.main()
