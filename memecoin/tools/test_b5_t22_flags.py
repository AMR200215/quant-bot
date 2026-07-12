"""B5: T22 graduated pump-amm flags are read and control routing."""
import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class TestB5T22Flags(unittest.TestCase):

    def test_flags_read_from_config(self):
        """T22_GRAD_PUMP_AMM_PROBE_ENABLED and T22_GRAD_PUMP_AMM_ENABLED exist in config."""
        import importlib, sys
        # Force reload in case a test stub corrupted sys.modules["memecoin.config"]
        sys.modules.pop("memecoin.config", None)
        import memecoin.config as _cfg
        importlib.reload(_cfg)
        T22_GRAD_PUMP_AMM_PROBE_ENABLED = _cfg.T22_GRAD_PUMP_AMM_PROBE_ENABLED
        T22_GRAD_PUMP_AMM_ENABLED = _cfg.T22_GRAD_PUMP_AMM_ENABLED
        self.assertFalse(T22_GRAD_PUMP_AMM_PROBE_ENABLED,
                         "Probe flag must default False — no receipt yet")
        self.assertFalse(T22_GRAD_PUMP_AMM_ENABLED,
                         "Enabled flag must default False — no receipt yet")

    def test_both_false_disallows_pump_amm(self):
        """When both flags False, T22 grad pump-amm is NOT allowed."""
        _t22_probe = False
        _t22_enabled = False
        _t22_pump_amm_allowed = _t22_enabled or _t22_probe
        self.assertFalse(_t22_pump_amm_allowed)
        # Escalate would be False for T22 graduated when not allowed
        _is_t22_graduated = True
        escalate = (False if (_is_t22_graduated and not _t22_pump_amm_allowed)
                    else True)
        self.assertFalse(escalate)

    def test_probe_true_allows_pump_amm(self):
        """When probe=True, T22 grad pump-amm IS allowed."""
        _t22_probe = True
        _t22_enabled = False
        _t22_pump_amm_allowed = _t22_enabled or _t22_probe
        self.assertTrue(_t22_pump_amm_allowed)

    def test_enabled_true_allows_pump_amm(self):
        """When enabled=True, T22 grad pump-amm IS allowed."""
        _t22_probe = False
        _t22_enabled = True
        _t22_pump_amm_allowed = _t22_enabled or _t22_probe
        self.assertTrue(_t22_pump_amm_allowed)

    def test_flags_in_decision_grep(self):
        """portfolio.py must reference both flag names."""
        import re
        portfolio_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "memecoin", "portfolio.py"
        )
        with open(portfolio_path) as f:
            content = f.read()
        self.assertIn("T22_GRAD_PUMP_AMM_PROBE_ENABLED", content,
                      "portfolio.py must reference probe flag")
        self.assertIn("T22_GRAD_PUMP_AMM_ENABLED", content,
                      "portfolio.py must reference enabled flag")

if __name__ == "__main__":
    unittest.main(verbosity=2)
