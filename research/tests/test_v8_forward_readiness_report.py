"""research/tests/test_v8_forward_readiness_report.py — V8 DATA
RECOVERY batch item 9: automatic forward-readiness report.

Run: python -m pytest research/tests/test_v8_forward_readiness_report.py -v
"""

import unittest

from research.v8_forward_readiness_report import (
    _progress_bucket, _estimate_velocity, compute_accumulation_velocity,
    FORWARD_READINESS_REPORT_VERSION, ForwardReadinessReport,
    _MIN_DAYS_FOR_VELOCITY_ESTIMATE,
)


class TestProgressBucket(unittest.TestCase):
    def test_buckets(self):
        self.assertEqual(_progress_bucket(0.3), "<50%")
        self.assertEqual(_progress_bucket(0.6), "50-70%")
        self.assertEqual(_progress_bucket(0.8), "70-85%")
        self.assertEqual(_progress_bucket(0.95), "85%+")
        self.assertEqual(_progress_bucket(None), "unknown")


class TestEstimateVelocity(unittest.TestCase):

    def test_insufficient_history_below_min_days(self):
        events = {"2026-08-19": 5}
        rate, label = _estimate_velocity(events, window_days=7)
        self.assertEqual(label, "INSUFFICIENT_HISTORY")

    def test_stable_when_low_variance(self):
        events = {"2026-08-15": 10, "2026-08-16": 11, "2026-08-17": 9, "2026-08-18": 10}
        rate, label = _estimate_velocity(events, window_days=7)
        self.assertEqual(label, "STABLE")
        self.assertAlmostEqual(rate, 10.0, delta=1.0)

    def test_too_variable_when_high_variance(self):
        events = {"2026-08-15": 1, "2026-08-16": 50, "2026-08-17": 2, "2026-08-18": 40}
        rate, label = _estimate_velocity(events, window_days=7)
        self.assertEqual(label, "TOO_VARIABLE_TO_PROJECT")

    def test_zero_events_is_insufficient_not_stable_zero(self):
        events = {"2026-08-15": 0, "2026-08-16": 0}
        rate, label = _estimate_velocity(events, window_days=7)
        self.assertEqual(label, "INSUFFICIENT_HISTORY")

    def test_never_raises_on_empty_dict(self):
        rate, label = _estimate_velocity({}, window_days=7)
        self.assertIsNone(rate)
        self.assertEqual(label, "INSUFFICIENT_HISTORY")


class TestComputeAccumulationVelocity(unittest.TestCase):

    def test_never_predicts_a_date(self):
        """The dataclass must not carry any 'estimated_completion_date'
        style field -- descriptive rate only."""
        v = compute_accumulation_velocity([], [], [])
        self.assertFalse(hasattr(v, "estimated_completion_date"))
        self.assertFalse(hasattr(v, "eta"))

    def test_empty_inputs_give_insufficient_history(self):
        v = compute_accumulation_velocity([], [], [])
        self.assertEqual(v.stability_label, "INSUFFICIENT_HISTORY")

    def test_note_explains_no_prediction_policy(self):
        v = compute_accumulation_velocity([], [], [])
        self.assertIn("no completion date", v.note.lower())


class TestReportVersioning(unittest.TestCase):
    def test_version_is_1(self):
        self.assertEqual(FORWARD_READINESS_REPORT_VERSION, 1)

    def test_report_structurally_never_claims_holdout_evaluated(self):
        """holdout_evaluated must default False and there is no setter
        anywhere in this module that could flip it."""
        import inspect
        import research.v8_forward_readiness_report as mod
        src = inspect.getsource(mod)
        self.assertNotIn("holdout_evaluated=True", src)
        self.assertNotIn("holdout_evaluated = True", src)


if __name__ == "__main__":
    unittest.main()
