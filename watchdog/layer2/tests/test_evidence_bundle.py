"""watchdog/layer2/tests/test_evidence_bundle.py"""

import unittest

from watchdog.layer2.evidence_bundle import build_evidence_bundle, _redact


class TestEvidenceBundle(unittest.TestCase):

    def test_ev_ids_assigned_deterministically(self):
        raw = {"collected_at": 100.0, "host": {"a": 1}, "git": {"b": 2}, "cron": {"c": 3}}
        bundle = build_evidence_bundle(raw, audit_id="test1")
        ids = [(i.ev_id, i.section) for i in bundle.items]
        # section order is fixed regardless of dict insertion order
        self.assertEqual(ids, [("EV001", "host"), ("EV002", "git"), ("EV003", "cron")])

    def test_same_raw_dump_produces_same_hash(self):
        raw = {"collected_at": 100.0, "host": {"a": 1}}
        b1 = build_evidence_bundle(raw, audit_id="a")
        b2 = build_evidence_bundle(raw, audit_id="b")
        self.assertEqual(b1.raw_bundle_sha256, b2.raw_bundle_sha256)

    def test_different_content_produces_different_hash(self):
        b1 = build_evidence_bundle({"collected_at": 1.0, "host": {"a": 1}})
        b2 = build_evidence_bundle({"collected_at": 1.0, "host": {"a": 2}})
        self.assertNotEqual(b1.raw_bundle_sha256, b2.raw_bundle_sha256)

    def test_get_retrieves_by_ev_id(self):
        raw = {"collected_at": 1.0, "host": {"a": 1}, "git": {"b": 2}}
        bundle = build_evidence_bundle(raw)
        item = bundle.get("EV002")
        self.assertEqual(item.section, "git")
        self.assertIsNone(bundle.get("EV999"))

    def test_unknown_sections_still_included_after_known_ones(self):
        raw = {"collected_at": 1.0, "mystery_section": {"x": 1}, "host": {"a": 1}}
        bundle = build_evidence_bundle(raw)
        sections = [i.section for i in bundle.items]
        self.assertEqual(sections, ["host", "mystery_section"])

    def test_redaction_masks_known_secret_patterns(self):
        text = 'TELEGRAM_BOT_TOKEN=123456:ABCDEF-ghijklmnop other stuff'
        redacted = _redact(text)
        self.assertNotIn("123456:ABCDEF", redacted)
        self.assertIn("[REDACTED]", redacted)

    def test_prompt_text_includes_all_ev_ids(self):
        raw = {"collected_at": 1.0, "host": {"a": 1}, "git": {"b": 2}}
        bundle = build_evidence_bundle(raw)
        text = bundle.to_prompt_text()
        self.assertIn("EV001", text)
        self.assertIn("EV002", text)
        self.assertIn(bundle.raw_bundle_sha256, text)

    def test_secret_in_evidence_content_is_redacted_in_prompt_text(self):
        raw = {"collected_at": 1.0, "cron": {"content": "SOLANA_PRIVATE_KEY=abcXYZ123secretvalue"}}
        bundle = build_evidence_bundle(raw)
        text = bundle.to_prompt_text()
        self.assertNotIn("abcXYZ123secretvalue", text)

    def test_receipts_prose_hidden_from_evidence_view_by_default(self):
        """The core W15 guarantee: RECEIPTS.md's claim text must not be
        visible during the ground-truth pass."""
        raw = {"collected_at": 1.0,
               "receipts_tail": {"mtime": 123.0, "length": 500, "tail": "STATUS: LIVE_VERIFIED, all good"}}
        bundle = build_evidence_bundle(raw)
        evidence_view = bundle.to_prompt_text(hide_claims=True)
        self.assertNotIn("LIVE_VERIFIED", evidence_view)
        self.assertIn("123.0", evidence_view)  # mtime (fact) stays visible

    def test_receipts_prose_visible_when_claims_not_hidden(self):
        raw = {"collected_at": 1.0,
               "receipts_tail": {"mtime": 123.0, "length": 500, "tail": "STATUS: LIVE_VERIFIED, all good"}}
        bundle = build_evidence_bundle(raw)
        full_view = bundle.to_prompt_text(hide_claims=False)
        self.assertIn("LIVE_VERIFIED", full_view)

    def test_claims_text_contains_only_claims_designated_content(self):
        raw = {"collected_at": 1.0, "host": {"a": 1},
               "receipts_tail": {"mtime": 123.0, "tail": "STATUS: LIVE_VERIFIED"}}
        bundle = build_evidence_bundle(raw)
        claims = bundle.to_claims_text()
        self.assertIn("LIVE_VERIFIED", claims)
        self.assertNotIn('"a": 1', claims)  # host section is evidence, not a claim

    def test_claims_text_empty_bundle_says_so_explicitly(self):
        raw = {"collected_at": 1.0, "host": {"a": 1}}
        bundle = build_evidence_bundle(raw)
        claims = bundle.to_claims_text()
        self.assertIn("no claims content", claims)


if __name__ == "__main__":
    unittest.main()
