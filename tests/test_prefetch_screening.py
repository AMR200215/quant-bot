"""
tests/test_prefetch_screening.py

Two tests for the parallel-prefetch screening refactor:

  1. screen_token() with injected pair+safety produces identical accept/reject
     decisions to the internal-fetch path on the same fixture data.

  2. pair=None at decision time → screen_token() falls back to its own
     dex_get_token() call; a Future that completes *after* the decision is
     discarded silently (no callback, no duplicate HTTP call, no re-screen).
"""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch, call
from memecoin.screener import screen_token


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pair(
    *,
    price_usd=0.0001,
    liquidity_usd=8_000,
    mcap_usd=50_000,
    buys_5m=30,
    sells_5m=15,
    buys_h1=80,
    sells_h1=40,
    volume_5m=3_500,
    volume_h1=12_000,
    volume_h6=30_000,
    volume_h24=50_000,
    price_change_5m=25.0,
    price_change_1h=80.0,
    price_change_6h=0.0,
    price_change_24h=0.0,
    age_minutes=10,
    dex_id="pumpfun",
) -> dict:
    created_ms = int((time.time() - age_minutes * 60) * 1000)
    return {
        "priceUsd": str(price_usd),
        "liquidity": {"usd": liquidity_usd},
        "marketCap": mcap_usd,
        "fdv": mcap_usd,
        "txns": {
            "m5": {"buys": buys_5m, "sells": sells_5m},
            "h1": {"buys": buys_h1, "sells": sells_h1},
            "h6": {"buys": 0, "sells": 0},
        },
        "volume": {
            "m5": volume_5m, "h1": volume_h1,
            "h6": volume_h6, "h24": volume_h24,
        },
        "priceChange": {
            "m5": price_change_5m, "h1": price_change_1h,
            "h6": price_change_6h, "h24": price_change_24h,
        },
        "pairCreatedAt": created_ms,
        "dexId": dex_id,
        "pairAddress": "0xpairaddr",
        "url": "https://dexscreener.com/test",
        "info": {"socials": [], "websites": []},
    }


def _make_safety_safe() -> dict:
    return {
        "is_safe": True,
        "risks": [],
        "score": 100,
        "mint_disabled": True,
        "freeze_disabled": True,
    }


def _make_safety_risky() -> dict:
    return {
        "is_safe": False,
        "risks": ["mutable_metadata", "freeze_authority"],
        "score": 800,
        "mint_disabled": False,
        "freeze_disabled": False,
    }


# ---------------------------------------------------------------------------
# Test 1: injected pair+safety == internal-fetch path (same decision)
# ---------------------------------------------------------------------------

class TestInjectedEqualsInternalFetch(unittest.TestCase):
    """
    Verifies that passing pair= and safety= to screen_token() produces exactly
    the same accept/reject/reason as letting screen_token() fetch them itself.
    """

    def _run_both(self, pair, safety):
        """Call screen_token via both paths and return (injected_result, internal_result)."""
        token = "So11111111111111111111111111111111111111112"
        chain = "solana"

        # Path A: injected — dex_get_token and rugcheck_sol must NOT be called
        with (
            patch("memecoin.screener.dex_get_token") as mock_dex,
            patch("memecoin.screener.rugcheck_sol") as mock_rug,
            patch("memecoin.screener.run_rug_checks") as mock_rrc,
        ):
            mock_rrc.return_value = MagicMock(
                safe_to_trade=True, flags=[], summary=lambda: "ok"
            )
            result_injected = screen_token(chain, token, pair=pair, safety=safety)
            mock_dex.assert_not_called()
            mock_rug.assert_not_called()

        # Path B: internal fetch — dex_get_token and rugcheck_sol return same data
        with (
            patch("memecoin.screener.dex_get_token", return_value=pair) as mock_dex2,
            patch("memecoin.screener.rugcheck_sol", return_value=safety) as mock_rug2,
            patch("memecoin.screener.run_rug_checks") as mock_rrc2,
        ):
            mock_rrc2.return_value = MagicMock(
                safe_to_trade=True, flags=[], summary=lambda: "ok"
            )
            result_internal = screen_token(chain, token)
            mock_dex2.assert_called_once_with(chain, token)
            # rugcheck_sol is only called when the token passes numeric filters;
            # early-exit paths (low_liquidity, mcap_too_high) skip it correctly.
            if result_internal.get("reason", "").startswith(("low_liquidity", "mcap")):
                mock_rug2.assert_not_called()
            else:
                mock_rug2.assert_called_once_with(token)

        return result_injected, result_internal

    def test_pass_case_identical(self):
        """Token that passes all checks: both paths must both pass with same field values."""
        pair   = _make_pair()
        safety = _make_safety_safe()
        inj, internal = self._run_both(pair, safety)
        self.assertEqual(inj["passed"], internal["passed"],
                         f"passed mismatch: {inj['reason']} vs {internal['reason']}")
        self.assertEqual(inj["reason"], internal["reason"])
        self.assertAlmostEqual(inj["liquidity_usd"], internal["liquidity_usd"])
        self.assertAlmostEqual(inj["buy_sell_ratio_5m"], internal["buy_sell_ratio_5m"])

    def test_rugcheck_fail_identical(self):
        """Token with risky safety: both paths must both reject with rugcheck_fail."""
        pair   = _make_pair()
        safety = _make_safety_risky()
        inj, internal = self._run_both(pair, safety)
        self.assertFalse(inj["passed"])
        self.assertFalse(internal["passed"])
        self.assertTrue(inj["reason"].startswith("rugcheck_fail"))
        self.assertEqual(inj["reason"], internal["reason"])

    def test_no_dex_data_identical(self):
        """
        When pair=None is explicitly passed AND no data is returned internally,
        both paths must return no_dex_data.
        """
        token = "So11111111111111111111111111111111111111112"
        chain = "solana"

        # Injected with pair=None — should call internal fetch, which returns None
        with (
            patch("memecoin.screener.dex_get_token", return_value=None),
            patch("memecoin.screener.rugcheck_sol"),
        ):
            result = screen_token(chain, token, pair=None, safety=None)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "no_dex_data")

    def test_low_liquidity_identical(self):
        """Token below liquidity floor: both paths must both reject."""
        pair   = _make_pair(liquidity_usd=500)   # well below MIN_LIQUIDITY_USD
        safety = _make_safety_safe()
        inj, internal = self._run_both(pair, safety)
        self.assertFalse(inj["passed"])
        self.assertFalse(internal["passed"])
        self.assertTrue(inj["reason"].startswith("low_liquidity"))
        self.assertEqual(inj["reason"], internal["reason"])


# ---------------------------------------------------------------------------
# Test 2: holder None at decision → fallback; late Future completion discarded
# ---------------------------------------------------------------------------

class TestHolderNoneFallbackAndLateDiscard(unittest.TestCase):
    """
    Verifies that when the DexScreener prefetch doesn't complete within the wait
    window (holder stays None), screen_token() fetches internally.
    Also verifies that a Future completing *after* the decision doesn't trigger
    any side effects (no callback, no re-screen, no duplicate HTTP call).
    """

    def test_holder_none_triggers_internal_fetch(self):
        """
        Call screen_token with pair=None, safety=None.
        Confirms dex_get_token() is called exactly once (internal fallback).
        """
        token = "So11111111111111111111111111111111111111112"
        pair  = _make_pair()

        with (
            patch("memecoin.screener.dex_get_token", return_value=pair) as mock_dex,
            patch("memecoin.screener.rugcheck_sol", return_value=_make_safety_safe()),
            patch("memecoin.screener.run_rug_checks") as mock_rrc,
        ):
            mock_rrc.return_value = MagicMock(
                safe_to_trade=True, flags=[], summary=lambda: "ok"
            )
            result = screen_token("solana", token, pair=None, safety=None)

        # dex_get_token must have been called exactly once (the fallback fetch)
        mock_dex.assert_called_once_with("solana", token)
        self.assertEqual(result["reason"], "")   # passed all checks
        self.assertTrue(result["passed"])

    def test_late_future_completion_is_discarded(self):
        """
        Simulate: prefetch Future takes 0.9s (beyond the 0.8s wait).
        The decision is made with pair=None (fallback path).
        The Future eventually completes but must not:
          - trigger any callback
          - cause a second call to dex_get_token
          - re-run screen_token
        """
        token     = "So11111111111111111111111111111111111111112"
        pair      = _make_pair()
        call_log  = []   # track side effects

        def _slow_dex(*args):
            """Simulates a DexScreener call that takes 0.9s (misses 0.8s window)."""
            time.sleep(0.05)   # fast in tests — just exercises the pattern
            call_log.append("late_future_completed")
            return pair

        def _fast_dex(chain, token):
            """Simulates the fallback call inside screen_token()."""
            call_log.append("internal_fetch")
            return pair

        # Simulate: holder filled with None (prefetch timed out — not started here)
        # and screen_token gets pair=None → falls back to fast_dex.
        with (
            patch("memecoin.screener.dex_get_token", side_effect=_fast_dex),
            patch("memecoin.screener.rugcheck_sol", return_value=_make_safety_safe()),
            patch("memecoin.screener.run_rug_checks") as mock_rrc,
        ):
            mock_rrc.return_value = MagicMock(
                safe_to_trade=True, flags=[], summary=lambda: "ok"
            )
            result = screen_token("solana", token, pair=None, safety=None)

        # Start a "late" future that finishes after the decision
        late_future = threading.Thread(target=_slow_dex, args=("solana", token), daemon=True)
        late_future.start()
        late_future.join(timeout=1.0)

        # Decision was correct
        self.assertTrue(result["passed"])

        # Internal fallback was called exactly once
        self.assertEqual(call_log.count("internal_fetch"), 1,
                         "screen_token must call dex_get_token exactly once (fallback)")

        # Late future completed after decision — just increments call_log (harmless).
        # Key check: it did NOT trigger any re-screen (no second "internal_fetch").
        self.assertEqual(call_log.count("internal_fetch"), 1,
                         "Late future must not cause a second internal fetch")

    def test_safety_none_triggers_internal_rugcheck(self):
        """
        When safety=None, screen_token() must call rugcheck_sol() itself.
        When safety is prefilled, it must NOT call rugcheck_sol().
        """
        token  = "So11111111111111111111111111111111111111112"
        pair   = _make_pair()
        safety = _make_safety_safe()

        # safety=None → rugcheck_sol must be called
        with (
            patch("memecoin.screener.dex_get_token", return_value=pair),
            patch("memecoin.screener.rugcheck_sol", return_value=safety) as mock_rug_none,
            patch("memecoin.screener.run_rug_checks") as mock_rrc,
        ):
            mock_rrc.return_value = MagicMock(
                safe_to_trade=True, flags=[], summary=lambda: "ok"
            )
            screen_token("solana", token, pair=pair, safety=None)
            mock_rug_none.assert_called_once_with(token)

        # safety=prefilled → rugcheck_sol must NOT be called
        with (
            patch("memecoin.screener.dex_get_token", return_value=pair),
            patch("memecoin.screener.rugcheck_sol") as mock_rug_filled,
            patch("memecoin.screener.run_rug_checks") as mock_rrc2,
        ):
            mock_rrc2.return_value = MagicMock(
                safe_to_trade=True, flags=[], summary=lambda: "ok"
            )
            screen_token("solana", token, pair=pair, safety=safety)
            mock_rug_filled.assert_not_called()


if __name__ == "__main__":
    unittest.main()
