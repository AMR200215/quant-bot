"""
test_t22_exit.py — Dry-run tests for T22 exit scenarios.

Tests:
  1. T22 token detection (check_token_program on a known T22 mint)
  2. T22 extension classification
  3. BC T22 sell TX build (BONDING_CURVE_T22_LOCAL)
  4. BC T22 simulate
  5. Jupiter price impact guard (mock >35% impact quote)
  6. Kill switch trigger and re-enable
  7. Canary mode size capping

Usage:
    python scripts/test_t22_exit.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PASS = "PASS"
FAIL = "FAIL"

_results = []

def _test(name: str, fn):
    try:
        result = fn()
        status = PASS if result else FAIL
        _results.append((name, status, ""))
        print(f"  [{status}] {name}")
        return result
    except Exception as e:
        _results.append((name, FAIL, str(e)))
        print(f"  [FAIL] {name} — {e}")
        return None


def test_t22_detection():
    """Test 1: check_token_program on a known T22 mint (USDC = SPL, not T22)."""
    from memecoin.screener import check_token_program
    # USDC is SPL — should return is_token2022=False
    usdc = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
    result = check_token_program(usdc)
    assert "is_token2022" in result, "missing is_token2022 key"
    assert result["token_program"] != "", "token_program should not be empty for USDC"
    assert result["is_token2022"] == False, f"USDC should not be T22, got {result}"
    return True


def test_t22_extension_classification():
    """Test 2: Extension type classification logic."""
    from memecoin.screener import _T22_SAFE_EXTENSIONS, _T22_TRANSFER_HOOK_ID, _T22_EXTENSION_NAMES
    # Transfer hook should be type 14
    assert _T22_TRANSFER_HOOK_ID == 14, "TransferHook should be type 14"
    # Safe list should not include transfer hook
    assert 14 not in _T22_SAFE_EXTENSIONS, "TransferHook (14) must not be in safe list"
    # All known types should have names
    for i in range(24):
        assert i in _T22_EXTENSION_NAMES, f"Extension type {i} missing from name map"
    return True


def test_bc_t22_build():
    """Test 3: BC T22 sell TX build (will fail without live RPC/keypair but should import cleanly)."""
    from memecoin.bonding_curve_t22 import build_bc_t22_sell_tx, TOKEN_PROGRAM_T22
    # Verify module loads and function is callable
    assert callable(build_bc_t22_sell_tx), "build_bc_t22_sell_tx should be callable"
    assert TOKEN_PROGRAM_T22 == "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
    return True


def test_bc_t22_simulate():
    """Test 4: BC T22 simulate function is importable and returns correct shape."""
    from memecoin.bonding_curve_t22 import simulate_bc_t22_sell
    assert callable(simulate_bc_t22_sell), "simulate_bc_t22_sell should be callable"
    # Call with dummy args — should fail gracefully (no keypair)
    result = simulate_bc_t22_sell(
        token_mint="11111111111111111111111111111111",
        wallet_pubkey="11111111111111111111111111111111",
        token_amount=0,
        rpc_url="https://api.mainnet-beta.solana.com",
    )
    assert "sim_ok" in result, "missing sim_ok in result"
    assert "error_class" in result, "missing error_class in result"
    assert result["sim_ok"] == False, "sim should fail for dummy inputs"
    return True


def test_jupiter_price_impact_guard():
    """Test 5: Jupiter price impact guard blocks high-impact quotes."""
    # Mock a quote dict with high price impact
    mock_quote = {"priceImpactPct": "0.40", "outAmount": "1000000"}  # 40%

    from memecoin.config import MAX_JUPITER_EXIT_PRICE_IMPACT_PCT, ALLOW_JUPITER_PANIC_EXIT
    assert MAX_JUPITER_EXIT_PRICE_IMPACT_PCT == 35, f"Expected 35, got {MAX_JUPITER_EXIT_PRICE_IMPACT_PCT}"

    impact_pct = float(mock_quote.get("priceImpactPct", 0)) * 100
    assert impact_pct == 40.0, f"Expected 40.0, got {impact_pct}"
    assert impact_pct > MAX_JUPITER_EXIT_PRICE_IMPACT_PCT, "40% should exceed 35% limit"
    assert ALLOW_JUPITER_PANIC_EXIT == False, "Panic exit should be disabled by default"
    # Logic: should block
    would_block = impact_pct > MAX_JUPITER_EXIT_PRICE_IMPACT_PCT and not ALLOW_JUPITER_PANIC_EXIT
    assert would_block, "Impact guard should block this trade"
    return True


def test_kill_switch_trigger():
    """Test 6: Kill switch disable and re-enable."""
    import memecoin.kill_switch as ks

    # Re-enable first to get clean state
    ks.enable_live_buys("test_setup")
    assert ks.live_buys_enabled() == True, "Kill switch should start enabled"

    # Disable
    ks.disable_live_buys("test_unknown_sell_failure")
    assert ks.live_buys_enabled() == False, "Kill switch should be disabled after trigger"

    # Re-enable
    ks.enable_live_buys("test_manual_reenable")
    assert ks.live_buys_enabled() == True, "Kill switch should be re-enabled"

    return True


def test_canary_mode_size_cap():
    """Test 7: Canary mode caps trade size."""
    from memecoin.config import LIVE_CANARY_MODE, EXIT_SYSTEM_VALIDATED, MAX_CANARY_TRADE_USD

    assert LIVE_CANARY_MODE == True, "Canary mode should be enabled by default"
    assert EXIT_SYSTEM_VALIDATED == False, "System should not be validated yet"
    assert MAX_CANARY_TRADE_USD == 3, f"Expected MAX_CANARY_TRADE_USD=3, got {MAX_CANARY_TRADE_USD}"

    # Simulate the cap logic
    test_sizes = [5.0, 10.0, 3.0, 1.0, 0.5]
    for size in test_sizes:
        if LIVE_CANARY_MODE and not EXIT_SYSTEM_VALIDATED:
            capped = min(size, MAX_CANARY_TRADE_USD)
        else:
            capped = size
        expected = min(size, 3.0)
        assert capped == expected, f"Expected {expected} for size={size}, got {capped}"

    return True


def main():
    print("\n=== T22 Exit System Test Suite ===\n")

    _test("T22 token detection (USDC=SPL)", test_t22_detection)
    _test("T22 extension classification", test_t22_extension_classification)
    _test("BC T22 sell TX build (import)", test_bc_t22_build)
    _test("BC T22 simulate (graceful fail)", test_bc_t22_simulate)
    _test("Jupiter price impact guard (40% > 35%)", test_jupiter_price_impact_guard)
    _test("Kill switch trigger and re-enable", test_kill_switch_trigger)
    _test("Canary mode size capping", test_canary_mode_size_cap)

    passed = sum(1 for _, s, _ in _results if s == PASS)
    total  = len(_results)
    print(f"\n=== Results: {passed}/{total} passed ===")

    if passed < total:
        for name, status, err in _results:
            if status == FAIL:
                print(f"  FAIL: {name}" + (f" — {err}" if err else ""))
        sys.exit(1)
    else:
        print("All tests passed.")


if __name__ == "__main__":
    main()
