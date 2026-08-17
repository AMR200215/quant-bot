"""research/tests/test_v8_ipw.py — V8-FILTER-DERIVATION Phase 2 (P2-8):
inverse probability weighting design (not yet applied).

Run: python -m pytest research/tests/test_v8_ipw.py -v
"""

import unittest

from research.v8_ipw import compute_ipw_weight, diagnose_admission_log, IPW_APPLIED_IN_PIPELINE


class TestComputeIPWWeight(unittest.TestCase):

    def test_full_probability_gives_weight_one(self):
        self.assertEqual(compute_ipw_weight(1.0), 1.0)

    def test_half_probability_gives_weight_two(self):
        self.assertEqual(compute_ipw_weight(0.5), 2.0)

    def test_zero_probability_raises(self):
        with self.assertRaises(ValueError):
            compute_ipw_weight(0.0)

    def test_negative_probability_raises(self):
        with self.assertRaises(ValueError):
            compute_ipw_weight(-0.1)

    def test_probability_above_one_raises(self):
        with self.assertRaises(ValueError):
            compute_ipw_weight(1.5)


class TestDiagnoseAdmissionLog(unittest.TestCase):

    def test_empty_rows_raises(self):
        with self.assertRaises(ValueError):
            diagnose_admission_log([])

    def test_unweighted_and_weighted_counts_kept_separate(self):
        rows = [
            {"path_sampling_probability": 1.0},
            {"path_sampling_probability": 0.5},
            {"path_sampling_probability": 0.25},
        ]
        d = diagnose_admission_log(rows)
        self.assertEqual(d.unweighted_n_effective, 3)
        # weighted effective n = 1 + 2 + 4 = 7, strictly > unweighted
        self.assertAlmostEqual(d.weighted_n_effective, 7.0)
        self.assertGreater(d.weighted_n_effective, d.unweighted_n_effective)

    def test_zero_probability_row_counted_not_silently_dropped(self):
        rows = [{"path_sampling_probability": 1.0}, {"path_sampling_probability": 0.0}]
        d = diagnose_admission_log(rows)
        self.assertEqual(d.n_admitted, 2)
        self.assertEqual(d.n_zero_probability, 1)

    def test_real_admission_log_shape_all_positive_probability(self):
        """Mirrors the live-verified P2-8 finding: 43 admitted rows,
        zero with probability 0.0. If this regresses (a future admitted
        row has probability<=0), n_zero_probability must surface it,
        not silently vanish it."""
        real_probs = [1.0, 1.0, 0.89548, 0.467587, 1.0, 1.0, 1.0, 0.838026,
                      0.607209, 1.0, 1.0, 1.0, 1.0, 1.0, 0.366719, 1.0]
        rows = [{"path_sampling_probability": p} for p in real_probs]
        d = diagnose_admission_log(rows)
        self.assertEqual(d.n_zero_probability, 0)
        self.assertEqual(d.n_admitted, len(real_probs))


class TestNotYetApplied(unittest.TestCase):

    def test_ipw_applied_flag_is_false(self):
        self.assertFalse(IPW_APPLIED_IN_PIPELINE)


if __name__ == "__main__":
    unittest.main()
