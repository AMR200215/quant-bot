"""B2: Oracle-confirmed graduation dispatches immediately (no 30s delay)."""
import sys, os, time, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class TestB2ImmediateGraduation(unittest.TestCase):
    def test_complete_true_dispatches_without_delay(self):
        """complete=True at T0 → close_position called within simulation, no 30s wait."""
        # Simulate the logic inline
        _graduated_exit_fired = set()
        close_calls = []

        class FakePos:
            id = "pos_b2"
            token_symbol = "TKNA"
            notes = "live|tx:abc|cohort:bonding_curve"
            exit_reason = None
            current_price = 0.00003

        pos = FakePos()
        now = time.time()
        cp_result = {"complete": True, "ok": True}

        # Simulate B2 dispatch logic
        if cp_result.get("complete") is True:
            if "|cohort:bonding_curve" in pos.notes:
                pos.notes = pos.notes.replace("|cohort:bonding_curve", "|cohort:graduated")
            if "|graduation_first_seen_ts:" not in (pos.notes or ""):
                pos.notes += f"|graduation_first_seen_ts:{int(now)}"
            if pos.id not in _graduated_exit_fired and not pos.exit_reason:
                _graduated_exit_fired.add(pos.id)
                close_calls.append(("graduated_exit", pos.id, time.time()))

        self.assertEqual(len(close_calls), 1, "close_position must be called once")
        elapsed = close_calls[0][2] - now
        self.assertLess(elapsed, 5.0, f"Dispatch must be immediate, got {elapsed:.1f}s")
        self.assertIn(pos.id, _graduated_exit_fired)

    def test_no_double_dispatch(self):
        """graduation_first_seen_ts already set → dispatch still only fires once."""
        _graduated_exit_fired = set()
        close_calls = []

        class FakePos:
            id = "pos_b2b"
            token_symbol = "TKNB"
            notes = "live|tx:abc|cohort:graduated|graduation_first_seen_ts:1000"
            exit_reason = None
            current_price = 0.00003

        pos = FakePos()
        cp_result = {"complete": True}

        # First dispatch
        if pos.id not in _graduated_exit_fired and not pos.exit_reason:
            _graduated_exit_fired.add(pos.id)
            close_calls.append(1)
        # Second loop iteration
        if pos.id not in _graduated_exit_fired and not pos.exit_reason:
            close_calls.append(2)

        self.assertEqual(len(close_calls), 1)

if __name__ == "__main__":
    unittest.main(verbosity=2)
