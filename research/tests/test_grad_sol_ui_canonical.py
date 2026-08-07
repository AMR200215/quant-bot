"""
test_grad_sol_ui_canonical.py — PROGRESS-FIX PF9 regression test.

Guards against GRAD_SOL_UI drift: prior to 2026-08-07, this value (115.0)
was independently hardcoded/redefined in research/tracker.py,
research/analysis/report.py, research/analysis/path_stats.py, and
memecoin/v8_paper.py, with no mechanism to catch them silently diverging
if one copy were ever edited without the others.

research.config.GRAD_SOL_UI and memecoin.config.GRAD_SOL_UI remain two
separate module-level constants by design (research and memecoin are
decoupled services) — this test asserts they still agree, and that every
other module that needs the graduation threshold imports one of these two
canonical sources rather than defining its own.

Run with:
    python -m unittest research/tests/test_grad_sol_ui_canonical.py
"""

import unittest


class TestGradSolUiCanonical(unittest.TestCase):

    def test_research_and_memecoin_config_agree(self):
        from research.config import GRAD_SOL_UI as research_grad
        from memecoin.config import GRAD_SOL_UI as memecoin_grad
        self.assertEqual(
            research_grad, memecoin_grad,
            "research.config.GRAD_SOL_UI and memecoin.config.GRAD_SOL_UI "
            "have drifted apart — these must stay in sync manually since "
            "they're separate modules by design; PF9 only guarantees no "
            "THIRD independent copy exists.",
        )

    def test_path_stats_uses_canonical_value(self):
        from research.config import GRAD_SOL_UI
        from research.analysis import path_stats
        self.assertEqual(path_stats._GRAD_SOL, GRAD_SOL_UI)

    def test_v8_paper_uses_canonical_value(self):
        from memecoin.config import GRAD_SOL_UI
        from memecoin import v8_paper
        self.assertEqual(v8_paper._GRAD_SOL, GRAD_SOL_UI)

    def test_tracker_progress_calc_uses_canonical_value(self):
        """research/tracker.py no longer has a bare 115.0 literal — it must
        import GRAD_SOL_UI and use it in the progress_at_signal calc."""
        import inspect
        from research import tracker
        src = inspect.getsource(tracker)
        self.assertNotIn(
            "/ 115.0", src,
            "research/tracker.py has a bare 115.0 literal again — use the "
            "imported GRAD_SOL_UI constant instead.",
        )
        self.assertIn("GRAD_SOL_UI", src)

    def test_report_progress_calc_uses_canonical_value(self):
        import inspect
        from research.analysis import report
        src = inspect.getsource(report)
        self.assertNotIn(
            "/ 115.0", src,
            "research/analysis/report.py has a bare 115.0 literal again — "
            "use the imported GRAD_SOL_UI constant instead.",
        )
        self.assertIn("GRAD_SOL_UI", src)

    def test_canonical_value_is_115(self):
        """Not a calibration change (PF9 explicitly forbids that) — just
        pins the current agreed value so a silent edit is caught."""
        from research.config import GRAD_SOL_UI
        self.assertEqual(GRAD_SOL_UI, 115.0)


if __name__ == "__main__":
    unittest.main()
