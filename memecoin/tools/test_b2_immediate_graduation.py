"""B2: Oracle-confirmed graduation dispatches immediately (no 30s delay)."""
import sys, os, time, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from memecoin.scanner import stamp_and_dispatch_graduation  # noqa: E402
from unittest.mock import MagicMock


class TestB2ImmediateGraduation(unittest.TestCase):
    def test_complete_true_dispatches_without_delay(self):
        """complete=True at T0 → close_position called within simulation, no 30s wait."""
        close_calls = []
        graduated_exit_fired = set()

        pos = MagicMock()
        pos.id = "pos_b2"
        pos.token_symbol = "TKNA"
        pos.notes = "live|tx:abc|cohort:bonding_curve"
        pos.exit_reason = None
        pos.current_price = 0.00003

        portfolio = MagicMock()
        portfolio._positions = {}
        portfolio.close_position.side_effect = lambda pid, reason, price: close_calls.append(
            (reason, pid, time.time())
        )

        now = time.time()
        dispatched = stamp_and_dispatch_graduation(pos, portfolio, graduated_exit_fired, now)

        self.assertTrue(dispatched, "stamp_and_dispatch_graduation must return True")
        self.assertEqual(len(close_calls), 1, "close_position must be called once")
        elapsed = close_calls[0][2] - now
        self.assertLess(elapsed, 5.0, f"Dispatch must be immediate, got {elapsed:.1f}s")
        self.assertIn(pos.id, graduated_exit_fired)

    def test_no_double_dispatch(self):
        """graduation_first_seen_ts already set → dispatch only fires once."""
        close_calls = []
        graduated_exit_fired = set()

        pos = MagicMock()
        pos.id = "pos_b2b"
        pos.token_symbol = "TKNB"
        pos.notes = "live|tx:abc|cohort:graduated|graduation_first_seen_ts:1000"
        pos.exit_reason = None
        pos.current_price = 0.00003

        portfolio = MagicMock()
        portfolio._positions = {}
        portfolio.close_position.side_effect = lambda pid, reason, price: close_calls.append(1)

        now = time.time()
        # First dispatch
        stamp_and_dispatch_graduation(pos, portfolio, graduated_exit_fired, now)
        # Second loop iteration (simulate re-entry)
        stamp_and_dispatch_graduation(pos, portfolio, graduated_exit_fired, now)

        self.assertEqual(len(close_calls), 1, "close_position must only be called once")


if __name__ == "__main__":
    unittest.main(verbosity=2)
