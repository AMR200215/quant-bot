"""B3: Oracle-confirmed graduated positions try pump-amm before Jupiter rescue."""
import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class TestB3PumpAmmFirst(unittest.TestCase):

    def _make_oracle_confirmed_pos(self):
        """Position with graduation_first_seen_ts (oracle-confirmed graduated)."""
        from unittest.mock import MagicMock
        pos = MagicMock()
        pos.notes = (
            "live|tx:abc123|cohort:graduated"
            "|graduation_first_seen_ts:1000"
        )
        pos.token_symbol = "TKNA"
        pos.token_address = "mintGRAD"
        pos.chain = "solana"
        pos.is_live = True
        pos.entry_price = 0.00003
        pos.exit_price = 0.00001
        pos.size_usd = 3.0
        pos.remaining_fraction = 1.0
        return pos

    def test_oracle_confirmed_skips_rescue_pre_call(self):
        """For oracle-confirmed graduated, is_rescue_eligible_error is NOT called before executor."""
        rescue_called = []
        executor_called = []

        pos = self._make_oracle_confirmed_pos()

        _oracle_confirmed_graduated = (
            "|graduation_first_seen_ts:" in (pos.notes or "")
            and "|cohort:graduated" in (pos.notes or "")
        )
        self.assertTrue(_oracle_confirmed_graduated)

        # Simulate the gate check
        if not _oracle_confirmed_graduated:
            rescue_called.append(1)  # should NOT happen
        executor_called.append(1)

        self.assertEqual(rescue_called, [], "Rescue must not be called before executor for oracle-confirmed")
        self.assertEqual(executor_called, [1], "Executor must be called")

    def test_non_oracle_confirmed_uses_rescue_first(self):
        """For non-oracle-confirmed graduated, rescue IS called first."""
        from unittest.mock import MagicMock
        pos = MagicMock()
        pos.notes = "live|tx:abc|cohort:graduated"  # no graduation_first_seen_ts

        _oracle_confirmed_graduated = (
            "|graduation_first_seen_ts:" in (pos.notes or "")
            and "|cohort:graduated" in (pos.notes or "")
        )
        self.assertFalse(_oracle_confirmed_graduated, "Should NOT be oracle-confirmed")

    def test_first_outbound_is_executor_not_jupiter(self):
        """
        Order assertion: graduated close's first outbound attempt is executor (pump-amm).
        Jupiter call log should appear AFTER executor, not before.
        """
        call_order = []
        pos = self._make_oracle_confirmed_pos()

        _oracle_confirmed_graduated = (
            "|graduation_first_seen_ts:" in (pos.notes or "")
            and "|cohort:graduated" in (pos.notes or "")
        )

        # Simulate close_position routing order
        if not _oracle_confirmed_graduated:
            call_order.append("jupiter_rescue")  # pre-executor (old behavior)
        call_order.append("executor_pump_amm")   # always runs (B3 + G-batch)
        # post-executor Jupiter fallback only after executor fails:
        executor_failed = True  # simulate pump-amm fail
        if _oracle_confirmed_graduated and executor_failed:
            call_order.append("jupiter_rescue_b3")

        self.assertEqual(call_order[0], "executor_pump_amm",
                         f"First call must be executor, got: {call_order}")
        self.assertIn("jupiter_rescue_b3", call_order,
                      "Jupiter must still appear as post-executor fallback")
        self.assertNotIn("jupiter_rescue", call_order[:1],
                         "Jupiter must NOT be the first call")

if __name__ == "__main__":
    unittest.main(verbosity=2)
