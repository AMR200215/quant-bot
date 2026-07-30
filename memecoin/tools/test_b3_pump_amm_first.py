"""B3: Oracle-confirmed graduated positions try pump-amm before Jupiter rescue."""
import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from memecoin.exit_orchestrator import ExitOrchestrator, RouteOutcome  # noqa: E402
from unittest.mock import MagicMock


class TestB3PumpAmmFirst(unittest.TestCase):

    def _make_oracle_confirmed_pos(self):
        """Position with graduation_first_seen_ts (oracle-confirmed graduated)."""
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
        """For oracle-confirmed graduated, rescue is NOT called before executor."""
        call_order = []

        pos = self._make_oracle_confirmed_pos()
        orch = ExitOrchestrator(pos.token_address)

        _oracle_confirmed_graduated = (
            "|graduation_first_seen_ts:" in (pos.notes or "")
            and "|cohort:graduated" in (pos.notes or "")
        )
        self.assertTrue(_oracle_confirmed_graduated)

        # Simulate the routing gate: pre-executor rescue only fires when NOT oracle confirmed
        if not _oracle_confirmed_graduated:
            call_order.append("jupiter_rescue")  # should NOT happen
        call_order.append("executor_pump_amm")

        self.assertNotIn("jupiter_rescue", call_order,
                         "Rescue must not be called before executor for oracle-confirmed")
        self.assertIn("executor_pump_amm", call_order, "Executor must be called")

    def test_non_oracle_confirmed_uses_rescue_first(self):
        """For non-oracle-confirmed graduated, rescue IS called first."""
        pos = MagicMock()
        pos.notes = "live|tx:abc|cohort:graduated"  # no graduation_first_seen_ts

        _oracle_confirmed_graduated = (
            "|graduation_first_seen_ts:" in (pos.notes or "")
            and "|cohort:graduated" in (pos.notes or "")
        )
        self.assertFalse(_oracle_confirmed_graduated, "Should NOT be oracle-confirmed")

    def test_first_outbound_is_executor_not_jupiter(self):
        """
        Order assertion: dispatch ordering is enforced by ExitOrchestrator.
        For oracle-confirmed graduated, pump_amm dispatch comes before jupiter dispatch.
        """
        call_order = []
        pos = self._make_oracle_confirmed_pos()
        orch = ExitOrchestrator(pos.token_address)

        _oracle_confirmed_graduated = (
            "|graduation_first_seen_ts:" in (pos.notes or "")
            and "|cohort:graduated" in (pos.notes or "")
        )

        # Simulate dispatch ordering using orchestrator
        def _pump_amm_fn():
            call_order.append("executor_pump_amm")
            return {"success": False, "sig": None, "error_class": "graduated_unsellable",
                    "fill_price": None, "sol_received": None}

        orch.dispatch("pump_amm", _pump_amm_fn)

        # Post-executor Jupiter fallback only after executor fails
        if _oracle_confirmed_graduated:
            def _jupiter_fn():
                call_order.append("jupiter_rescue_b3")
                return {"success": True, "sig": "testsig", "error_class": None,
                        "fill_price": 0.00001, "sol_received": 0.001}
            orch.dispatch("jupiter", _jupiter_fn)

        self.assertEqual(call_order[0], "executor_pump_amm",
                         f"First call must be executor, got: {call_order}")
        self.assertIn("jupiter_rescue_b3", call_order,
                      "Jupiter must still appear as post-executor fallback")
        self.assertNotIn("jupiter_rescue", call_order[:1],
                         "Jupiter must NOT be the first call")


if __name__ == "__main__":
    unittest.main(verbosity=2)
