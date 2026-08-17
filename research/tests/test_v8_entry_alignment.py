"""research/tests/test_v8_entry_alignment.py — V8-FILTER-DERIVATION
Phase 2 (P2-7): real entry-time alignment, never assumes rows[0]==entry.

Run: python -m pytest research/tests/test_v8_entry_alignment.py -v
"""

import unittest

from research.v8_entry_alignment import (
    resolve_entry_alignment, find_ambiguous_mints,
    EntryAlignment, EntryAlignmentExclusion,
)
from research.v8_candidate_registry import CANDIDATES

_BASELINE = next(c for c in CANDIDATES if c["candidate_id"] == "BASELINE-0")  # progress+venue_state

# All 4 frozen v1 candidates require venue_state_at_signal (T0+capture) --
# there's no real registered candidate that's pure T0. Use a synthetic
# candidate to exercise the T0 (zero-delay) code path directly.
_PURE_T0_CANDIDATE = {
    "candidate_id": "TEST-PURE-T0",
    "conditions": [{"feature": "alert_time", "op": "exists", "value": True}],
    "required_features": ["alert_time"],
}


def _event(event_id="e1", token_address="MINT_A", alert_time="2026-08-10T00:00:00+00:00", lag_ms=None):
    e = {"event_id": event_id, "token_address": token_address, "alert_time": alert_time}
    if lag_ms is not None:
        e["progress_capture_lag_ms"] = lag_ms
    return e


def _rows(ts_price_pairs):
    return [{"ts_ms": ts, "price_usd": price} for ts, price in ts_price_pairs]


class TestAmbiguousJoin(unittest.TestCase):

    def test_single_event_per_mint_not_ambiguous(self):
        events = [_event(token_address="A"), _event(token_address="B")]
        self.assertEqual(find_ambiguous_mints(events), set())

    def test_realerted_mint_is_ambiguous(self):
        events = [_event(event_id="e1", token_address="A"), _event(event_id="e2", token_address="A")]
        self.assertEqual(find_ambiguous_mints(events), {"A"})

    def test_ambiguous_mint_excluded_not_guessed(self):
        events = [_event(event_id="e1", token_address="A"), _event(event_id="e2", token_address="A")]
        ambiguous = find_ambiguous_mints(events)
        rows = _rows([(0, 1.0), (1000, 1.1)])
        result = resolve_entry_alignment(events[0], rows, _BASELINE, ambiguous)
        self.assertIsInstance(result, EntryAlignmentExclusion)
        self.assertEqual(result.reason, "AMBIGUOUS_PATH_EVENT_JOIN")


class TestDecisionDelay(unittest.TestCase):

    def test_t0_candidate_no_delay(self):
        alert_ms = 1_000_000
        event = _event(alert_time="1970-01-01T00:16:40+00:00")  # 1_000_000 ms epoch
        rows = _rows([(alert_ms, 1.0), (alert_ms + 100, 1.05)])
        result = resolve_entry_alignment(event, rows, _PURE_T0_CANDIDATE, set())
        self.assertIsInstance(result, EntryAlignment)
        self.assertEqual(result.entry_source, "T0")
        self.assertEqual(result.decision_available_ts, alert_ms)
        self.assertEqual(result.entry_ts_ms, alert_ms)  # first tick is exactly at target

    def test_t0_capture_uses_real_lag_when_present(self):
        alert_ms = 1_000_000
        event = _event(alert_time="1970-01-01T00:16:40+00:00", lag_ms=750)
        rows = _rows([(alert_ms, 1.0), (alert_ms + 750, 1.05), (alert_ms + 900, 1.10)])
        result = resolve_entry_alignment(event, rows, _BASELINE, set())
        self.assertEqual(result.entry_source, "T0+capture:real_lag")
        self.assertEqual(result.decision_available_ts, alert_ms + 750)
        self.assertEqual(result.entry_ts_ms, alert_ms + 750)  # exact match tick
        self.assertEqual(result.entry_lag_ms, 0)

    def test_t0_capture_falls_back_to_nominal_when_no_real_lag(self):
        alert_ms = 1_000_000
        event = _event(alert_time="1970-01-01T00:16:40+00:00")  # no lag_ms
        rows = _rows([(alert_ms, 1.0), (alert_ms + 2000, 1.05)])
        result = resolve_entry_alignment(event, rows, _BASELINE, set())
        self.assertEqual(result.entry_source, "T0+capture:nominal_fallback")
        self.assertGreater(result.decision_available_ts, alert_ms)

    def test_entry_price_is_not_rows_zero_when_delay_pushes_entry_later(self):
        """The core P2-7 guarantee: entry must NOT be rows[0] if a real
        delay means the bot couldn't have acted at rows[0]'s price."""
        alert_ms = 1_000_000
        event = _event(alert_time="1970-01-01T00:16:40+00:00", lag_ms=750)
        rows = _rows([
            (alert_ms, 9.99),          # rows[0] -- NOT executable (before feature availability)
            (alert_ms + 750, 1.00),    # real entry tick
            (alert_ms + 1500, 1.10),
        ])
        result = resolve_entry_alignment(event, rows, _BASELINE, set())
        self.assertEqual(result.entry_price, 1.00)
        self.assertNotEqual(result.entry_price, 9.99)


class TestRealRegistryDelayClass(unittest.TestCase):

    def test_all_frozen_v1_candidates_require_venue_state_t0_capture(self):
        """Documents a real fact about the current frozen v1 registry: every
        candidate (including P0) requires venue_state_at_signal, which is
        T0+capture -- so none of them take the zero-delay T0 path today."""
        for c in CANDIDATES:
            self.assertIn("venue_state_at_signal", c["required_features"])


class TestExclusions(unittest.TestCase):

    def test_no_path_rows(self):
        result = resolve_entry_alignment(_event(), [], _BASELINE, set())
        self.assertEqual(result.reason, "NO_PATH_ROWS")

    def test_bad_alert_time(self):
        event = _event(alert_time="not-a-timestamp")
        result = resolve_entry_alignment(event, _rows([(0, 1.0)]), _BASELINE, set())
        self.assertEqual(result.reason, "BAD_ALERT_TIME")

    def test_no_executable_tick_after_target_never_nearest_match(self):
        """If every tick is BEFORE the decision-available target, this must
        be an explicit exclusion -- never silently fall back to the
        nearest (too-early) tick."""
        alert_ms = 1_000_000
        event = _event(alert_time="1970-01-01T00:16:40+00:00", lag_ms=5000)
        rows = _rows([(alert_ms, 1.0), (alert_ms + 100, 1.05)])  # path ends long before target
        result = resolve_entry_alignment(event, rows, _BASELINE, set())
        self.assertIsInstance(result, EntryAlignmentExclusion)
        self.assertEqual(result.reason, "NO_EXECUTABLE_TICK_AFTER_TARGET")


if __name__ == "__main__":
    unittest.main()
