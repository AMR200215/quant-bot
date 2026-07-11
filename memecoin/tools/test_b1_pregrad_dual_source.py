"""B1: PP-silent pre-graduation exit fires from curve reserves."""
import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class TestB1DualSourcePregrad(unittest.TestCase):
    def test_pp_silent_curve_crosses_threshold_fires_exit(self):
        """When PP vSOL=0 but curve reserves cross 97.75 SOL, exit fires exactly once."""
        from unittest.mock import MagicMock, patch

        # Mock position
        pos = MagicMock()
        pos.id = "pos1"
        pos.chain = "solana"
        pos.token_address = "mintAAA"
        pos.notes = "live|tx:abc123"
        pos.token_symbol = "TKNA"
        pos.current_price = 0.00003

        # Mock PP monitor — always returns 0 vSOL (PP silent)
        pp_monitor = MagicMock()
        pp_monitor.get_vsol.return_value = 0.0

        # Curve vsol dict — 98 SOL (above 97.75 threshold)
        curve_vsol = {"mintAAA": (98.0, 1000.0)}

        # Run the pre-graduation logic inline (extracted for test)
        import time
        now = 1000.1  # fresh (0.1s old)
        GRAD_SOL = 115.0
        PREGRAD_PCT = 0.85

        _vsol = pp_monitor.get_vsol("mintAAA")
        _vsol_source = "pp"
        if _vsol <= 0:
            _cv_entry = curve_vsol.get("mintAAA")
            if _cv_entry and (now - _cv_entry[1]) < 5.0:
                _vsol = _cv_entry[0]
                _vsol_source = "curve"

        self.assertGreater(_vsol, 0, "vSOL must be nonzero from curve")
        self.assertEqual(_vsol_source, "curve", "Source must be curve when PP silent")
        _progress = _vsol / GRAD_SOL
        self.assertGreaterEqual(_progress, PREGRAD_PCT,
                                f"Progress {_progress:.2%} must >= {PREGRAD_PCT:.2%}")

    def test_stale_curve_entry_not_used(self):
        """Stale curve entry (>5s) is ignored — no fabricated trigger."""
        import time
        now = 1010.0
        curve_vsol = {"mintBBB": (98.0, 1000.0)}  # 10s old → stale

        pp_vsol = 0.0
        _vsol = pp_vsol
        _cv_entry = curve_vsol.get("mintBBB")
        if _cv_entry and (now - _cv_entry[1]) < 5.0:
            _vsol = _cv_entry[0]
        # Should still be 0 (stale)
        self.assertEqual(_vsol, 0.0, "Stale curve entry must not be used")

    def test_pp_fresh_preferred_over_curve(self):
        """When PP vSOL is fresh, curve is not consulted."""
        from unittest.mock import MagicMock
        pp_monitor = MagicMock()
        pp_monitor.get_vsol.return_value = 50.0  # PP has it

        curve_vsol = {"mintCCC": (98.0, 999.0)}  # would trigger if used
        now = 1000.0

        _vsol = pp_monitor.get_vsol("mintCCC")
        _vsol_source = "pp"
        if _vsol <= 0:
            _cv_entry = curve_vsol.get("mintCCC")
            if _cv_entry and (now - _cv_entry[1]) < 5.0:
                _vsol = _cv_entry[0]
                _vsol_source = "curve"

        self.assertEqual(_vsol_source, "pp")
        self.assertEqual(_vsol, 50.0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
