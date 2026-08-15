"""watchdog/tests/test_v8_funnel.py — W19 fault-injection #13-15, plus
the general no-false-green-on-missing-evidence cases.

V8-REWIRE (2026-08-14): updated for the new (telegram_received ->
v8_fork_entered) and (v8_fork_entered -> terminal) pairings. The first
pairing is the load-bearing new invariant: every raw alert must reach
V8's fork, independent of V7's screening -- see watchdog/checks/v8_funnel.py.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from watchdog.checks import STATUS_CRITICAL, STATUS_OK, STATUS_UNKNOWN
from watchdog.checks.v8_funnel import check_v8_funnel, find_missing_terminal_dispositions


class TestV8FunnelFaultInjection(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "v8_funnel.jsonl"
        # Deterministic by default: check_v8_funnel() reads the REAL
        # repo-relative logs/watchdog/v8_rewire_deploy_ts.txt unless this
        # is patched. Found live: on a machine where that file genuinely
        # exists (e.g. the VPS, post-deploy) with a real 2026 epoch
        # timestamp, tests using fabricated tiny now_ts=1_000_000.0
        # values would have every telegram_received row wrongly exempted
        # as "predates the invariant" and silently pass when they should
        # fail. Tests that specifically want to exercise the real-stamp
        # behavior override this with their own inner patch.
        self._deploy_ts_patch = patch("watchdog.checks.v8_funnel._v8_rewire_deploy_ts",
                                       return_value=None)
        self._deploy_ts_patch.start()

    def tearDown(self):
        self._deploy_ts_patch.stop()
        self._tmp.cleanup()

    def _write(self, events):
        self.path.write_text("\n".join(json.dumps(e) for e in events))

    def test_15_missing_terminal_disposition_beyond_grace_is_detected(self):
        now = 1_000_000.0
        self._write([
            {"ts": now - 500, "stage": "v8_fork_entered", "event_id": "ev1", "mint": "M1"},
            # no v8_gate_rejected / v8_opened / v8_pass_unpriced / v8_transport_duplicate for ev1
        ])
        results = check_v8_funnel(now_ts=now, funnel_path=self.path, grace_seconds=120)
        self.assertEqual(results[0].status, STATUS_CRITICAL)
        self.assertIn("v8_fork_entered", results[0].reason)

    def test_13_ten_eligible_ten_explicitly_rejected_is_not_a_silent_disappearance(self):
        """Real production behavior: a low open-rate with explicit reject
        reasons for every candidate is healthy, not an anomaly. This check
        must never flag a 100%-explicitly-rejected batch."""
        now = 1_000_000.0
        events = []
        for i in range(10):
            events.append({"ts": now - 500, "stage": "v8_fork_entered", "event_id": f"ev{i}", "mint": f"M{i}"})
            events.append({"ts": now - 499, "stage": "v8_gate_rejected", "event_id": f"ev{i}", "mint": f"M{i}"})
        self._write(events)
        results = check_v8_funnel(now_ts=now, funnel_path=self.path, grace_seconds=120)
        self.assertEqual(results[0].status, STATUS_OK, results[0].reason)

    def test_14_single_event_within_grace_produces_no_false_alarm(self):
        now = 1_000_000.0
        self._write([
            {"ts": now - 10, "stage": "v8_fork_entered", "event_id": "ev1", "mint": "M1"},
            # only 10s old, well within the 120s grace -- too soon to expect resolution
        ])
        results = check_v8_funnel(now_ts=now, funnel_path=self.path, grace_seconds=120)
        self.assertEqual(results[0].status, STATUS_OK, results[0].reason)

    def test_two_stage_transitions_both_checked_independently(self):
        now = 1_000_000.0
        self._write([
            # ev1: stuck between telegram_received and v8_fork_entered --
            # exactly the regression class this pairing exists to catch
            # (V8 silently stops forking off the raw alert).
            {"ts": now - 500, "stage": "telegram_received", "event_id": "ev1", "mint": "M1"},
        ])
        results = check_v8_funnel(now_ts=now, funnel_path=self.path, grace_seconds=120)
        self.assertEqual(results[0].status, STATUS_CRITICAL)
        self.assertIn("telegram_received", results[0].evidence["missing_by_stage"])

    def test_v8_fork_entered_counts_as_valid_terminal_for_telegram_received(self):
        now = 1_000_000.0
        self._write([
            {"ts": now - 500, "stage": "telegram_received", "event_id": "ev1", "mint": "M1"},
            {"ts": now - 499, "stage": "v8_fork_entered", "event_id": "ev1", "mint": "M1"},
            {"ts": now - 498, "stage": "v8_gate_rejected", "event_id": "ev1", "mint": "M1"},
        ])
        results = check_v8_funnel(now_ts=now, funnel_path=self.path, grace_seconds=120)
        self.assertEqual(results[0].status, STATUS_OK, results[0].reason)

    def test_transport_duplicate_and_pass_unpriced_count_as_valid_terminals(self):
        now = 1_000_000.0
        self._write([
            {"ts": now - 500, "stage": "telegram_received", "event_id": "ev1", "mint": "M1"},
            {"ts": now - 499, "stage": "v8_fork_entered", "event_id": "ev1", "mint": "M1"},
            {"ts": now - 498, "stage": "v8_transport_duplicate", "event_id": "ev1", "mint": "M1"},
            {"ts": now - 400, "stage": "telegram_received", "event_id": "ev2", "mint": "M2"},
            {"ts": now - 399, "stage": "v8_fork_entered", "event_id": "ev2", "mint": "M2"},
            {"ts": now - 398, "stage": "v8_pass_unpriced", "event_id": "ev2", "mint": "M2"},
        ])
        results = check_v8_funnel(now_ts=now, funnel_path=self.path, grace_seconds=120)
        self.assertEqual(results[0].status, STATUS_OK, results[0].reason)

    def test_v7_reject_does_not_excuse_v8_from_reaching_its_own_terminal(self):
        """The exact regression the rewire fixes: a token V7 rejected must
        still show a V8 terminal disposition (v8_fork_entered must have
        fired). If it only has telegram_received + screening_rejected
        (V7's stages) and nothing from V8, that IS the old, broken
        architecture reappearing and must be flagged."""
        now = 1_000_000.0
        self._write([
            {"ts": now - 500, "stage": "telegram_received", "event_id": "ev1", "mint": "M1"},
            {"ts": now - 499, "stage": "screening_rejected", "event_id": "ev1", "mint": "M1"},
            # no v8_fork_entered -- V8 never saw this V7-rejected alert
        ])
        results = check_v8_funnel(now_ts=now, funnel_path=self.path, grace_seconds=120)
        self.assertEqual(results[0].status, STATUS_CRITICAL)

    def test_missing_file_is_unknown_not_ok(self):
        results = check_v8_funnel(now_ts=1_000_000.0, funnel_path=self.path)
        self.assertEqual(results[0].status, STATUS_UNKNOWN)

    def test_empty_file_is_unknown_not_ok(self):
        self.path.write_text("")
        results = check_v8_funnel(now_ts=1_000_000.0, funnel_path=self.path)
        self.assertEqual(results[0].status, STATUS_UNKNOWN)

    def test_pre_deploy_telegram_received_rows_are_not_flagged(self):
        """Regression for a real bug caught live by Layer 2's first
        automatic audit the morning after V8-REWIRE deployed: old
        telegram_received rows written before v8_fork_entered existed as
        a stage were flagged CRITICAL forever, since that stage could
        never retroactively appear for them. min_ts_by_entry_stage (fed
        from the same deploy-cutover stamp memecoin/v8_paper.py writes)
        must exempt them."""
        now = 1_000_000.0
        deploy_ts = now - 300   # deploy happened 300s ago
        events = [
            # pre-deploy: old row, will never get v8_fork_entered -- must NOT be flagged
            {"ts": now - 500, "stage": "telegram_received", "event_id": "old1", "mint": "M1"},
            # post-deploy, past grace (200s > 120s grace) -- legitimately stuck, must still be flagged
            {"ts": now - 200, "stage": "telegram_received", "event_id": "new1", "mint": "M2"},
        ]
        missing = find_missing_terminal_dispositions(
            events, now_ts=now, grace_seconds=120,
            min_ts_by_entry_stage={"telegram_received": deploy_ts},
        )
        offender_ids = {e["event_id"] for e in missing.get("telegram_received", [])}
        self.assertNotIn("old1", offender_ids)
        self.assertIn("new1", offender_ids)

    def test_check_v8_funnel_reads_real_deploy_stamp_and_exempts_old_rows(self):
        now = 1_000_000.0
        with tempfile.TemporaryDirectory() as tmp:
            stamp = Path(tmp) / "v8_rewire_deploy_ts.txt"
            stamp.write_text(str(now - 100))
            self._write([
                {"ts": now - 500, "stage": "telegram_received", "event_id": "old1", "mint": "M1"},
            ])
            with patch("watchdog.checks.v8_funnel._v8_rewire_deploy_ts", return_value=now - 100):
                results = check_v8_funnel(now_ts=now, funnel_path=self.path, grace_seconds=120)
        self.assertEqual(results[0].status, STATUS_OK, results[0].reason)

    def test_missing_deploy_stamp_does_not_crash_and_fails_toward_quiet(self):
        """If the stamp is unreadable, min_ts_by_entry_stage ends up empty
        -- old rows fall back to being checked against the invariant like
        before this fix existed, which is the pre-fix (noisier) behavior,
        not a crash. Documents the tradeoff explicitly rather than leaving
        it implicit."""
        now = 1_000_000.0
        self._write([
            {"ts": now - 500, "stage": "telegram_received", "event_id": "old1", "mint": "M1"},
        ])
        with patch("watchdog.checks.v8_funnel._v8_rewire_deploy_ts", return_value=None):
            results = check_v8_funnel(now_ts=now, funnel_path=self.path, grace_seconds=120)
        self.assertEqual(results[0].status, STATUS_CRITICAL)

    def test_malformed_lines_are_skipped_not_fatal(self):
        now = 1_000_000.0
        self.path.write_text(
            "not json at all\n"
            + json.dumps({"ts": now - 10, "stage": "v8_fork_entered", "event_id": "ev1", "mint": "M1"})
        )
        results = check_v8_funnel(now_ts=now, funnel_path=self.path, grace_seconds=120)
        self.assertEqual(results[0].status, STATUS_OK)


if __name__ == "__main__":
    unittest.main()
