"""research/tests/test_v8_execution_proxy.py — V8 DATA RECOVERY batch
item 7: $2/$5 paper execution-cost proxy collector.

Run: python -m pytest research/tests/test_v8_execution_proxy.py -v
"""

import unittest

from research.v8_execution_proxy import (
    simulate_curve_buy, simulate_round_trip, build_curve_observation, build_unavailable_observation,
    EXECUTION_PROXY_UNAVAILABLE, EXECUTION_PROXY_MODEL_VERSION, PUMPFUN_TRADING_FEE_RATE,
)

# Real reserve snapshot, live-captured 2026-08-19 (fresh launch, VPS)
_REAL_VSOL = 30.100015625999983
_REAL_VTOK = 1069434660.764585


class TestSimulateCurveBuy(unittest.TestCase):

    def test_invalid_inputs_return_none(self):
        self.assertIsNone(simulate_curve_buy(0, 1000, 1.0))
        self.assertIsNone(simulate_curve_buy(30, 0, 1.0))
        self.assertIsNone(simulate_curve_buy(30, 1000, 0))
        self.assertIsNone(simulate_curve_buy(30, 1000, -1))
        self.assertIsNone(simulate_curve_buy(30, 1000, 1.0, fee_rate=1.5))

    def test_price_impact_positive_for_a_real_buy(self):
        result = simulate_curve_buy(_REAL_VSOL, _REAL_VTOK, notional_sol=0.05)
        self.assertIsNotNone(result)
        self.assertGreater(result.price_impact_pct, 0)  # buying pushes price up
        self.assertGreater(result.executable_buy_price_sol, result.reference_spot_price_sol)

    def test_fee_matches_configured_rate(self):
        result = simulate_curve_buy(_REAL_VSOL, _REAL_VTOK, notional_sol=1.0, fee_rate=0.01)
        self.assertAlmostEqual(result.fee_sol, 0.01, places=10)

    def test_larger_notional_has_larger_price_impact(self):
        small = simulate_curve_buy(_REAL_VSOL, _REAL_VTOK, notional_sol=0.01)
        large = simulate_curve_buy(_REAL_VSOL, _REAL_VTOK, notional_sol=1.0)
        self.assertGreater(large.price_impact_pct, small.price_impact_pct)

    def test_reserves_shift_conserves_the_swap_invariant_direction(self):
        result = simulate_curve_buy(_REAL_VSOL, _REAL_VTOK, notional_sol=0.05)
        self.assertGreater(result.new_vsol, _REAL_VSOL)   # SOL added to curve
        self.assertLess(result.new_vtok, _REAL_VTOK)      # tokens removed from curve


class TestSimulateRoundTrip(unittest.TestCase):

    def test_round_trip_loses_value_to_fees_and_impact(self):
        """An immediate buy+sell (no time for real price movement) must
        lose money -- two fee legs plus price impact both ways."""
        result = simulate_round_trip(_REAL_VSOL, _REAL_VTOK, notional_sol=0.05)
        self.assertIsNotNone(result)
        self.assertLess(result.round_trip_pct, 0)
        self.assertLess(result.round_trip_value_sol, 0.05)

    def test_round_trip_none_on_invalid_input(self):
        self.assertIsNone(simulate_round_trip(0, 1000, 1.0))

    def test_round_trip_pct_is_size_invariant_dominated_by_fees(self):
        """Exact constant-product AMM property: an immediate buy-then-sell
        with no intervening trades cancels the price-impact term exactly
        (proof: buy dy=y*dx/(x+dx) tokens, sell them back into (x+dx,y-dy)
        yields exactly dx again) -- so round-trip loss is size-invariant,
        driven purely by the two fee legs (~2x fee_rate), NOT by trade
        size. (Per-leg price_impact_pct DOES grow with size -- see
        TestSimulateCurveBuy.test_larger_notional_has_larger_price_impact
        -- but it cancels out of the round trip specifically.)"""
        small = simulate_round_trip(_REAL_VSOL, _REAL_VTOK, notional_sol=0.01)
        large = simulate_round_trip(_REAL_VSOL, _REAL_VTOK, notional_sol=1.0)
        self.assertAlmostEqual(small.round_trip_pct, large.round_trip_pct, places=1)
        expected_fee_loss_pct = (1 - (1 - PUMPFUN_TRADING_FEE_RATE) ** 2) * -100
        self.assertAlmostEqual(small.round_trip_pct, expected_fee_loss_pct, places=1)


class TestBuildCurveObservation(unittest.TestCase):

    def test_real_2_and_5_dollar_observations(self):
        for notional in (2.0, 5.0):
            obs = build_curve_observation(
                event_id="e1", token_address="MINT", notional_usd=notional,
                sol_price_usd=175.0, vsol=_REAL_VSOL, vtok=_REAL_VTOK,
            )
            self.assertEqual(obs.status, "OK")
            self.assertEqual(obs.venue, "CURVE_ACTIVE")
            self.assertEqual(obs.data_source, "pp_reserve_snapshot")
            self.assertGreater(obs.executable_buy_price_usd, obs.reference_spot_price_usd)
            self.assertGreater(obs.price_impact_pct, 0)
            self.assertLess(obs.round_trip_pct, 0)
            self.assertEqual(obs.curve_reserves_used, {"vsol": _REAL_VSOL, "vtok": _REAL_VTOK})
            self.assertEqual(obs.model_version, EXECUTION_PROXY_MODEL_VERSION)

    def test_unavailable_on_bad_sol_price(self):
        obs = build_curve_observation(
            event_id="e1", token_address="MINT", notional_usd=2.0,
            sol_price_usd=0.0, vsol=_REAL_VSOL, vtok=_REAL_VTOK,
        )
        self.assertEqual(obs.status, EXECUTION_PROXY_UNAVAILABLE)
        self.assertIsNone(obs.executable_buy_price_usd)

    def test_unavailable_on_zero_reserves(self):
        obs = build_curve_observation(
            event_id="e1", token_address="MINT", notional_usd=2.0,
            sol_price_usd=175.0, vsol=0, vtok=0,
        )
        self.assertEqual(obs.status, EXECUTION_PROXY_UNAVAILABLE)

    def test_lag_computed_when_timestamps_given(self):
        obs = build_curve_observation(
            event_id="e1", token_address="MINT", notional_usd=2.0,
            sol_price_usd=175.0, vsol=_REAL_VSOL, vtok=_REAL_VTOK,
            alert_ts_ms=1000, observed_ts_ms=1500,
        )
        self.assertEqual(obs.lag_from_alert_ms, 500)

    def test_lag_none_when_timestamps_absent(self):
        obs = build_curve_observation(
            event_id="e1", token_address="MINT", notional_usd=2.0,
            sol_price_usd=175.0, vsol=_REAL_VSOL, vtok=_REAL_VTOK,
        )
        self.assertIsNone(obs.lag_from_alert_ms)


class TestBuildUnavailableObservation(unittest.TestCase):

    def test_never_fabricates_a_cost(self):
        obs = build_unavailable_observation("e1", "MINT", 2.0, venue="GRADUATED_OR_DEX")
        self.assertEqual(obs.status, EXECUTION_PROXY_UNAVAILABLE)
        self.assertIsNone(obs.executable_buy_price_usd)
        self.assertIsNone(obs.price_impact_pct)
        self.assertIsNone(obs.fee_usd)

    def test_venue_label_preserved(self):
        obs = build_unavailable_observation("e1", "MINT", 2.0, venue="UNKNOWN")
        self.assertEqual(obs.venue, "UNKNOWN")


class TestProtocolConstant(unittest.TestCase):
    def test_fee_rate_is_public_protocol_value(self):
        self.assertEqual(PUMPFUN_TRADING_FEE_RATE, 0.01)


if __name__ == "__main__":
    unittest.main()
