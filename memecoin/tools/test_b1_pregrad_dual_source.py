"""B1: PP-silent pre-graduation exit fires from curve reserves."""
import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from memecoin.scanner import select_vsol_source  # noqa: E402


class TestB1DualSourcePregrad(unittest.TestCase):
    def test_pp_silent_curve_crosses_threshold_fires_exit(self):
        """When PP vSOL=0 but curve reserves cross 97.75 SOL, exit fires exactly once."""
        GRAD_SOL = 115.0
        PREGRAD_PCT = 0.85

        now = 1000.1  # fresh (0.1s old)
        curve_vsol = {"mintAAA": (98.0, 1000.0)}

        vsol, source = select_vsol_source("mintAAA", 0.0, curve_vsol, now)

        self.assertGreater(vsol, 0, "vSOL must be nonzero from curve")
        self.assertEqual(source, "curve", "Source must be curve when PP silent")
        _progress = vsol / GRAD_SOL
        self.assertGreaterEqual(_progress, PREGRAD_PCT,
                                f"Progress {_progress:.2%} must >= {PREGRAD_PCT:.2%}")

    def test_stale_curve_entry_not_used(self):
        """Stale curve entry (>5s) is ignored — no fabricated trigger."""
        now = 1010.0
        curve_vsol = {"mintBBB": (98.0, 1000.0)}  # 10s old → stale

        vsol, source = select_vsol_source("mintBBB", 0.0, curve_vsol, now)

        self.assertEqual(vsol, 0.0, "Stale curve entry must not be used")
        self.assertEqual(source, "pp")

    def test_pp_fresh_preferred_over_curve(self):
        """When PP vSOL is fresh, curve is not consulted."""
        curve_vsol = {"mintCCC": (98.0, 999.0)}  # would trigger if used
        now = 1000.0

        vsol, source = select_vsol_source("mintCCC", 50.0, curve_vsol, now)

        self.assertEqual(source, "pp")
        self.assertEqual(vsol, 50.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
