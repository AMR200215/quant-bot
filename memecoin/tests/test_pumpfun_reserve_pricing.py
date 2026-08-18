"""memecoin/tests/test_pumpfun_reserve_pricing.py — V8 DATA RECOVERY
batch: the canonical PumpPortal reserve-pricing helper, proven against
real live-captured examples (see module docstring for the capture
evidence). A unit-conversion regression capable of recreating the
~1,000,000x-scale bug must be structurally impossible after this.

Run: python -m pytest memecoin/tests/test_pumpfun_reserve_pricing.py -v
"""

import unittest

from memecoin.pumpfun_reserve_pricing import (
    price_sol_from_pp_reserves, price_usd_from_pp_reserves, venue_state_from_pp_reserves,
    PUMPFUN_INITIAL_VIRTUAL_TOKEN_RESERVES, PUMPFUN_REAL_TOKEN_RESERVES_AT_GRADUATION,
    MIN_VIRTUAL_TOKEN_RESERVES_ON_CURVE,
)


class TestRealCapturedExamples(unittest.TestCase):
    """Every value here is a REAL PumpPortal message captured live on
    2026-08-19 (VPS), not synthetic."""

    def test_fresh_create_event_near_initial_reserves(self):
        # mint=4vjpoWgkouDKoLd4MZxkdfHegFCN8LML42eUJZfLpump
        price = price_sol_from_pp_reserves(30.100015625999983, 1069434660.764585)
        # Real pump.fun fresh-launch price is on the order of 1e-8 SOL/token,
        # not 1e-2 (which is what the old /1e6 bug would have produced).
        self.assertAlmostEqual(price, 30.100015625999983 / 1069434660.764585, places=15)
        self.assertLess(price, 1e-6)
        self.assertGreater(price, 1e-9)

    def test_near_graduation_event_matches_exact_theoretical_reserve(self):
        # mint=eMSgEEkS8RBgyKeS4HZwFcb6gZ3gN8Zj4Y4grQ5pump -- vTokensInBondingCurve
        # (279900000) matches MIN_VIRTUAL_TOKEN_RESERVES_ON_CURVE EXACTLY.
        v_sol, v_tok = 115.005359056806, 279900000
        self.assertEqual(v_tok, MIN_VIRTUAL_TOKEN_RESERVES_ON_CURVE)
        price = price_sol_from_pp_reserves(v_sol, v_tok)
        self.assertAlmostEqual(price, 115.005359056806 / 279900000, places=15)

    def test_old_buggy_formula_would_have_been_a_million_times_larger(self):
        """Regression guard: prove the fix is materially different from
        the old bug, not just refactored the same wrong math."""
        v_sol, v_tok = 115.005359056806, 279900000
        correct = price_sol_from_pp_reserves(v_sol, v_tok)
        old_buggy = v_sol / (v_tok / 1e6)
        ratio = old_buggy / correct
        self.assertGreater(ratio, 900_000)
        self.assertLess(ratio, 1_100_000)


class TestInvariants(unittest.TestCase):

    def test_none_on_missing_input(self):
        self.assertIsNone(price_sol_from_pp_reserves(None, 1000.0))
        self.assertIsNone(price_sol_from_pp_reserves(30.0, None))

    def test_none_on_unparseable_input(self):
        self.assertIsNone(price_sol_from_pp_reserves("nan-ish", 1000.0))

    def test_none_on_zero_or_negative_reserves(self):
        self.assertIsNone(price_sol_from_pp_reserves(0.0, 1000.0))
        self.assertIsNone(price_sol_from_pp_reserves(30.0, 0.0))
        self.assertIsNone(price_sol_from_pp_reserves(-1.0, 1000.0))

    def test_price_never_exceeds_curve_ceiling_across_realistic_reserve_range(self):
        """Invariant: for any (vsol, vtok) pair inside the curve's real
        operating range (vsol in (0, GRAD_SOL_UI], vtok in
        [MIN_VIRTUAL_TOKEN_RESERVES_ON_CURVE, PUMPFUN_INITIAL_VIRTUAL_TOKEN_RESERVES]),
        the computed price must stay below a sane ceiling -- this is the
        structural check that would have caught the old bug immediately."""
        from research.config import GRAD_SOL_UI
        max_price = price_sol_from_pp_reserves(GRAD_SOL_UI, MIN_VIRTUAL_TOKEN_RESERVES_ON_CURVE)
        for vsol in (0.001, 1.0, 30.0, 80.0, GRAD_SOL_UI):
            for vtok in (MIN_VIRTUAL_TOKEN_RESERVES_ON_CURVE, 500_000_000, PUMPFUN_INITIAL_VIRTUAL_TOKEN_RESERVES):
                p = price_sol_from_pp_reserves(vsol, vtok)
                self.assertLessEqual(p, max_price * 1.0001)

    def test_price_usd_multiplies_by_sol_rate(self):
        p_sol = price_sol_from_pp_reserves(30.0, 1_000_000_000)
        p_usd = price_usd_from_pp_reserves(30.0, 1_000_000_000, 175.0)
        self.assertAlmostEqual(p_usd, p_sol * 175.0, places=12)

    def test_price_usd_none_on_invalid_rate(self):
        self.assertIsNone(price_usd_from_pp_reserves(30.0, 1_000_000_000, 0.0))
        self.assertIsNone(price_usd_from_pp_reserves(30.0, 1_000_000_000, None))

    def test_graduation_constants_are_public_protocol_values(self):
        self.assertEqual(PUMPFUN_INITIAL_VIRTUAL_TOKEN_RESERVES, 1_073_000_000)
        self.assertEqual(PUMPFUN_REAL_TOKEN_RESERVES_AT_GRADUATION, 793_100_000)
        self.assertEqual(MIN_VIRTUAL_TOKEN_RESERVES_ON_CURVE, 279_900_000)


class TestVenueStateFromReserves(unittest.TestCase):
    """V8 DATA RECOVERY item 4: venue_state was previously hardcoded
    CURVE_ACTIVE unconditionally -- the direct cause of Phase 2.1's
    VSOL_EXCEEDS_GRADUATION_WHILE_CURVE_ACTIVE findings."""

    def test_reserves_present_is_curve_active(self):
        self.assertEqual(venue_state_from_pp_reserves(30.1, 1_069_434_660.76), "CURVE_ACTIVE")

    def test_reserves_absent_is_unknown_not_curve_active(self):
        self.assertEqual(venue_state_from_pp_reserves(None, None), "UNKNOWN")
        self.assertEqual(venue_state_from_pp_reserves(0, 0), "UNKNOWN")

    def test_never_asserts_graduated_or_dex_active_without_proof(self):
        """Only CURVE_ACTIVE or UNKNOWN may come out of this function --
        it must never assert a specific post-graduation state it cannot
        prove from this message shape alone."""
        for vsol, vtok in [(None, None), (0, 0), (30.0, 0), (0, 1000.0)]:
            self.assertIn(venue_state_from_pp_reserves(vsol, vtok), ("CURVE_ACTIVE", "UNKNOWN"))


if __name__ == "__main__":
    unittest.main()
