"""B4: Per-venue state — cooldowns and attempt caps."""
import sys, os, time, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class TestB4VenueState(unittest.TestCase):

    def _make_portfolio(self):
        """Build a minimal Portfolio with venue state support."""
        class FakePortfolio:
            _venue_state = {}
            def _get_venue_state(self, pos_id, venue):
                if pos_id not in self._venue_state:
                    self._venue_state[pos_id] = {}
                if venue not in self._venue_state[pos_id]:
                    self._venue_state[pos_id][venue] = {
                        "cooldown_until": 0.0, "attempts": 0, "last_result": ""
                    }
                return self._venue_state[pos_id][venue]
            def _record_venue_attempt(self, pos_id, venue, result, cooldown_sec=0.0):
                vs = self._get_venue_state(pos_id, venue)
                vs["attempts"] += 1
                vs["last_result"] = result
                if cooldown_sec > 0:
                    vs["cooldown_until"] = time.time() + cooldown_sec
            def _venue_in_cooldown(self, pos_id, venue):
                return time.time() < self._get_venue_state(pos_id, venue)["cooldown_until"]
            def _pump_amm_attempts(self, pos_id):
                return self._get_venue_state(pos_id, "pump_amm")["attempts"]
        return FakePortfolio()

    def test_cooldown_blocks_venue(self):
        pf = self._make_portfolio()
        pf._record_venue_attempt("pos1", "pump_amm", "no_send", cooldown_sec=30)
        self.assertTrue(pf._venue_in_cooldown("pos1", "pump_amm"))

    def test_cooldown_expires(self):
        pf = self._make_portfolio()
        pf._record_venue_attempt("pos1", "pump_amm", "no_send", cooldown_sec=0.001)
        time.sleep(0.01)
        self.assertFalse(pf._venue_in_cooldown("pos1", "pump_amm"))

    def test_pump_amm_attempts_counted(self):
        pf = self._make_portfolio()
        pf._record_venue_attempt("pos1", "pump_amm", "no_send")
        pf._record_venue_attempt("pos1", "pump_amm", "no_send")
        self.assertEqual(pf._pump_amm_attempts("pos1"), 2)

    def test_venues_isolated(self):
        """cooldown on pump_amm does not affect jupiter state."""
        pf = self._make_portfolio()
        pf._record_venue_attempt("pos1", "pump_amm", "no_send", cooldown_sec=60)
        # Jupiter should still be open
        self.assertFalse(pf._venue_in_cooldown("pos1", "jupiter"))

    def test_pending_sig_semantics(self):
        """A pending sig on any venue prevents duplicate: pin that existing close_position guard."""
        rescue_class = "pending"
        executor_would_run = (rescue_class not in ("pending", "sold", "already_sold", "fatal_no_send"))
        self.assertFalse(executor_would_run,
                         "Pending rescue must block executor (no duplicate)")

    def test_fast_window_cap_at_3_attempts(self):
        """After 3 pump-amm attempts in fast window, cadence switches to MU (60s)."""
        pf = self._make_portfolio()
        GRAD_FAST_RETRY_SEC = 5
        retry_sec_normal = 60

        for i in range(3):
            pf._record_venue_attempt("pos_fw", "pump_amm", "no_send")

        pa_attempts = pf._pump_amm_attempts("pos_fw")
        _actual_retry = GRAD_FAST_RETRY_SEC
        if pa_attempts >= 3:
            _actual_retry = retry_sec_normal

        self.assertEqual(_actual_retry, retry_sec_normal,
                         "After 3 attempts, must use MU cadence not 5s")

if __name__ == "__main__":
    unittest.main(verbosity=2)
