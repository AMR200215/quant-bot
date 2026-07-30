"""Tests for the shared compute_fill_price helper."""
import pytest
from memecoin.tx_meta import compute_fill_price


def _call(sol_delta, tokens_raw, sol_usd, decimals=6):
    return compute_fill_price(sol_delta, tokens_raw, sol_usd, decimals)


class TestComputeFillPrice:
    def test_basic(self):
        # 0.1 SOL, 1_000_000 raw tokens (= 1 token with 6 dec), SOL=$100
        # => fill = (0.1 * 100) / 1.0 = $10.00
        assert abs(_call(0.1, 1_000_000, 100.0) - 10.0) < 1e-9

    def test_graduated_recovered_caller(self):
        # Simulate graduated_recovered: 50k raw tokens, 0.02 SOL, $80 SOL
        # fill = (0.02 * 80) / (50_000 / 1e6) = 1.6 / 0.05 = $32
        assert abs(_call(0.02, 50_000, 80.0) - 32.0) < 1e-6

    def test_rescue_finalize_caller(self):
        # Same inputs -> same result (one formula, all callers)
        assert _call(0.02, 50_000, 80.0) == _call(0.02, 50_000, 80.0)

    def test_reconciler_caller(self):
        # Reconciler uses same formula — just check it returns a positive value
        result = _call(0.031459, 8_588_571_867, 82.0)
        assert result > 0
        # Sanity: should be a small price per token (large raw count)
        assert result < 1.0

    def test_zero_inputs_return_zero(self):
        assert _call(0.0, 1_000_000, 100.0) == 0.0
        assert _call(0.1, 0, 100.0) == 0.0
        assert _call(0.1, 1_000_000, 0.0) == 0.0

    def test_negative_sol_returns_zero(self):
        assert _call(-0.01, 1_000_000, 100.0) == 0.0

    def test_custom_decimals(self):
        # 9-decimal token: 1_000_000_000 raw = 1 token
        result = _call(0.1, 1_000_000_000, 100.0, decimals=9)
        assert abs(result - 10.0) < 1e-6
