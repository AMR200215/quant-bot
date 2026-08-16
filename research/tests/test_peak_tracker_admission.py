"""research/tests/test_peak_tracker_admission.py — V8-FD P16-3/P16-4/
P16-6: budget-paced admission controller.

Root cause (V8-FD Phase 1.5, P15-5): "spend until PP_DAILY_MSG_BUDGET is
hit, then permanently drop every later token for the rest of the UTC
day" created a time-of-day selection bias in the naturalistic path
sample. These tests cover the replacement pacing function and the
structural guarantee that admission never depends on any outcome/
strategy-correlated signal.

Run: python -m pytest research/tests/test_peak_tracker_admission.py -v
"""

import ast
import unittest
from pathlib import Path

from research.peak_tracker import HOURS_PER_DAY, PeakTracker, _admission_probability


class TestAdmissionProbabilityPure(unittest.TestCase):
    """Pure function -- no asyncio, no live state."""

    def test_under_hourly_pace_admits_freely(self):
        self.assertEqual(_admission_probability(0, 4166.67, 0, 100_000), 1.0)
        self.assertEqual(_admission_probability(4000, 4166.67, 4000, 100_000), 1.0)

    def test_exactly_at_hourly_pace_starts_decaying(self):
        p = _admission_probability(4166.67, 4166.67, 4166, 100_000)
        self.assertAlmostEqual(p, 1.0, places=2)

    def test_double_hourly_pace_is_roughly_half_probability(self):
        hourly = 4166.67
        p = _admission_probability(2 * hourly, hourly, 8000, 100_000)
        self.assertAlmostEqual(p, 0.5, places=2)

    def test_quadruple_hourly_pace_is_roughly_quarter_probability(self):
        hourly = 4166.67
        p = _admission_probability(4 * hourly, hourly, 16000, 100_000)
        self.assertAlmostEqual(p, 0.25, places=2)

    def test_never_hits_a_hard_zero_from_hourly_pressure_alone(self):
        """The whole point of P16-3: no finite hourly overage should ever
        produce a literal 0% chance the way the old hard-cutoff did."""
        hourly = 4166.67
        p = _admission_probability(1_000_000, hourly, 50_000, 100_000)
        self.assertGreater(p, 0.0)

    def test_daily_ceiling_is_a_true_hard_stop(self):
        """The one place 0.0 is correct -- the actual approved cost bound
        (P16-2) must never be exceeded by an unbounded amount."""
        p = _admission_probability(0, 4166.67, 100_000, 100_000)
        self.assertEqual(p, 0.0)
        p2 = _admission_probability(0, 4166.67, 150_000, 100_000)
        self.assertEqual(p2, 0.0)

    def test_probability_always_in_valid_range(self):
        for used_hour in (0, 100, 5000, 50_000, 1_000_000):
            for used_day in (0, 50_000, 99_999, 100_000, 200_000):
                p = _admission_probability(used_hour, 4166.67, used_day, 100_000)
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)

    def test_zero_budget_never_admits(self):
        self.assertEqual(_admission_probability(0, 0, 0, 0), 0.0)
        self.assertEqual(_admission_probability(0, 4166.67, 0, 0), 0.0)


class TestNoStrategySignalInAdmission(unittest.TestCase):
    """V8-FD P16-6: admission must be decidable from pacing/budget state
    alone -- never progress_at_signal, V7/V8 pass state, or any outcome
    field. Enforced structurally (AST parameter inspection), not just by
    convention, so a future edit can't silently reintroduce it."""

    _FORBIDDEN_PARAM_SUBSTRINGS = (
        "progress", "smart_money", "outcome", "peak", "winner", "loser",
        "pct_change", "v7", "v8", "dex_id", "venue_state",
    )

    def test_admission_probability_signature_has_no_strategy_or_outcome_params(self):
        import research.peak_tracker as pt
        import inspect
        sig = inspect.signature(pt._admission_probability)
        param_names = [p.lower() for p in sig.parameters.keys()]
        for forbidden in self._FORBIDDEN_PARAM_SUBSTRINGS:
            hits = [p for p in param_names if forbidden in p]
            self.assertEqual(hits, [],
                f"_admission_probability's signature contains a strategy/outcome-"
                f"correlated parameter matching {forbidden!r}: {hits}")

    def test_admission_probability_only_takes_four_pacing_params(self):
        import research.peak_tracker as pt
        import inspect
        sig = inspect.signature(pt._admission_probability)
        self.assertEqual(
            set(sig.parameters.keys()),
            {"messages_used_this_hour", "hourly_budget", "messages_used_today", "daily_budget"},
        )

    def test_drain_pending_does_not_reference_progress_or_outcome_fields(self):
        """Static check on the actual call site -- the admission decision
        code path itself must never branch on a candidate's own data."""
        src = Path(__import__("research.peak_tracker", fromlist=["x"]).__file__).read_text()
        start = src.index("async def _drain_pending")
        end = src.index("async def _finalise_loop") if "async def _finalise_loop" in src[start:] else len(src)
        # bound the search to _drain_pending's own body reasonably
        end = start + src[start:].index("await asyncio.gather(_recv(), _drain_pending())")
        body = src[start:end]
        for forbidden in ("progress_at_signal", "smart_money", "pct_change_peak", "v8_paper", "v7"):
            self.assertNotIn(forbidden, body,
                f"_drain_pending()'s admission logic references {forbidden!r} -- "
                f"naturalistic path collection must stay independent of strategy labels")


class TestHourBucketAndRollover(unittest.TestCase):

    def setUp(self):
        self.pt = PeakTracker.__new__(PeakTracker)
        self.pt._hourly_stats = {}
        self.pt._current_hour = -1
        self.pt._messages_this_hour = 0

    def test_hour_bucket_lazily_initialised_with_all_required_fields(self):
        b = self.pt._hour_bucket(5)
        for key in ("path_eligible", "path_admitted", "subscriptions_started",
                    "ticks_ge1", "ticks_ge2", "usable_paths", "budget_messages"):
            self.assertIn(key, b)
            self.assertEqual(b[key], 0)

    def test_hour_bucket_persists_across_calls(self):
        b1 = self.pt._hour_bucket(3)
        b1["path_eligible"] += 1
        b2 = self.pt._hour_bucket(3)
        self.assertEqual(b2["path_eligible"], 1)
        self.assertIs(b1, b2)

    def test_maybe_roll_hour_resets_hourly_message_counter_on_change(self):
        # Fabricate two timestamps guaranteed to be in different UTC hours.
        base = 1_755_000_000.0  # arbitrary fixed epoch
        hour1 = self.pt._maybe_roll_hour(base)   # first touch: _current_hour starts at -1, always "changes"
        self.assertEqual(self.pt._messages_this_hour, 0)
        self.pt._messages_this_hour = 999
        same_hour = self.pt._maybe_roll_hour(base + 1)   # +1s, same hour -- must NOT reset
        self.assertEqual(same_hour, hour1)
        self.assertEqual(self.pt._messages_this_hour, 999)
        hour2 = self.pt._maybe_roll_hour(base + 3700)   # +1h1m40s -> next hour
        self.assertNotEqual(hour1, hour2)
        self.assertEqual(self.pt._messages_this_hour, 0)

    def test_hours_per_day_constant_is_24(self):
        self.assertEqual(HOURS_PER_DAY, 24)


if __name__ == "__main__":
    unittest.main()
