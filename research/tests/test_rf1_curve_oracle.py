"""
test_rf1_curve_oracle.py — Unit tests for RF1 bonding-curve oracle.

All network calls are mocked.  No pytest dependency — runs with:
    python -m unittest research/tests/test_rf1_curve_oracle.py
or:
    python -m unittest discover -s research/tests -p "test_rf1_*.py"

Test cases:
  test_curve_active_returns_price       — valid account data → CURVE_ACTIVE + price
  test_graduated_complete_flag          — complete=True → GRADUATED, no price from curve
  test_account_missing_not_graduation   — null account → CURVE_MISSING, not GRADUATED
  test_rpc_error_not_graduation         — requests.post raises → RPC_ERROR, not GRADUATED
  test_parse_error_fails_closed         — malformed base64 → PARSE_ERROR
  test_stale_sol_usd_rejects_price      — sol_usd_age > max → sol_usd_stale
  test_partial_batch_some_succeed       — 3 mints, 2 succeed, 1 null
  test_decimal_mismatch_explicit        — verify price calc uses pump.fun 6-decimal logic
"""

import base64
import json
import struct
import sys
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Config stub: inherit ALL attributes from the real config so that other
# modules imported in the same pytest session (e.g. peak_tracker) don't
# fail with AttributeError / ImportError when the stub is still cached in
# sys.modules.  Only the few attributes relevant to curve_oracle are
# overridden — everything else comes from the real module.
# ---------------------------------------------------------------------------
import importlib
import types

# Load the real config (safe — all vars have os.getenv defaults)
_real_config = importlib.import_module("research.config")

_config_stub = types.ModuleType("research.config")
# Copy every attribute from the real config into the stub
for _k, _v in vars(_real_config).items():
    setattr(_config_stub, _k, _v)

# Override only what curve_oracle tests need to control
_config_stub.SOL_USD_MAX_CACHE_AGE_S = 120
_config_stub.CURVE_BATCH_SIZE        = 100
_config_stub.GRAD_SOL_UI             = 115.0
_config_stub.HELIUS_API_KEY          = "test-helius-key"
_config_stub.HELIUS_RPC_URL          = "https://mainnet.helius-rpc.com/?api-key=test"
_config_stub.GRAD_VSOL_THRESHOLD     = 79.0

# Register the stub BEFORE importing curve_oracle
sys.modules["research.config"] = _config_stub

# Now import the module under test
from research import curve_oracle  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers for building valid bonding-curve account bytes
# ---------------------------------------------------------------------------

def _build_curve_account_bytes(
    virtual_token_reserves: int = 500_000_000_000_000,   # 500M tokens * 1e6 (6 decimals)
    virtual_sol_reserves:   int = 35_000_000_000,        # 35 SOL in lamports
    real_token_reserves:    int = 200_000_000_000_000,
    real_sol_reserves:      int = 10_000_000_000,
    token_total_supply:     int = 1_000_000_000_000_000,
    complete:               bool = False,
) -> bytes:
    """
    Build a fake bonding-curve account buffer matching the pump.fun layout.
    Layout:
      [0:8]   discriminator (8 arbitrary bytes)
      [8:16]  virtualTokenReserves  (u64 LE)
      [16:24] virtualSolReserves    (u64 LE)
      [24:32] realTokenReserves     (u64 LE)
      [32:40] realSolReserves       (u64 LE)
      [40:48] tokenTotalSupply      (u64 LE)
      [48]    complete              (bool / u8)
    """
    discriminator = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    body = struct.pack(
        "<QQQQQ?",
        virtual_token_reserves,
        virtual_sol_reserves,
        real_token_reserves,
        real_sol_reserves,
        token_total_supply,
        complete,
    )
    return discriminator + body


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def _rpc_ok_response(accounts: list) -> dict:
    """Build a valid getMultipleAccounts JSON-RPC response."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "context": {"slot": 123456},
            "value": accounts,
        },
    }


def _account_entry(raw_bytes: bytes, owner: str = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P") -> dict:
    return {
        "data": [_b64(raw_bytes), "base64"],
        "executable": False,
        "lamports": 2039280,
        "owner": owner,
        "rentEpoch": 0,
    }


# ── Fake mint addresses (not real, just recognisable in tests) ───────────────
MINT_A = "TokenAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
MINT_B = "TokenBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
MINT_C = "TokenCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class TestCurveOracle(unittest.TestCase):

    def _mock_post(self, response_dict: dict):
        """Return a MagicMock for requests.post that yields response_dict."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = response_dict
        mock_resp.raise_for_status.return_value = None
        mock = MagicMock(return_value=mock_resp)
        return mock

    # ── 1. Valid curve account → CURVE_ACTIVE + non-zero price ───────────────

    def test_curve_active_returns_price(self):
        """Valid account data with complete=False → CURVE_ACTIVE and a real price."""
        raw = _build_curve_account_bytes(
            virtual_token_reserves=500_000_000_000_000,  # 500M tokens (6 dec)
            virtual_sol_reserves=35_000_000_000,          # 35 SOL in lamports
            complete=False,
        )
        rpc_resp = _rpc_ok_response([_account_entry(raw)])

        with patch("research.curve_oracle.derive_curve_address", return_value="FakeCurveAddr1"):
            with patch("requests.post", self._mock_post(rpc_resp)):
                results = curve_oracle.get_curve_prices_batch(
                    mints=[MINT_A],
                    helius_key="test-key",
                    sol_price_usd=150.0,
                    sol_price_age_s=5.0,
                )

        self.assertIn(MINT_A, results)
        r = results[MINT_A]
        self.assertEqual(r["venue_state"], "CURVE_ACTIVE")
        self.assertIsNone(r["failure_reason"])
        self.assertIsNotNone(r["price_usd"])
        self.assertGreater(r["price_usd"], 0)
        self.assertIsNotNone(r["vsol_ui"])
        self.assertAlmostEqual(r["vsol_ui"], 35.0, places=3)

        # Price formula check (FIXED 2026-08-07, PROGRESS-FIX PF0 — was
        # tautologically re-deriving the buggy no-decimal-conversion
        # formula as "expected"; now the dimensionally correct one):
        # vsol_sol = 35_000_000_000 / 1e9 = 35 SOL
        # vtoken_ui = 500_000_000_000_000 / 1e6 = 500,000,000 tokens
        # price_usd = (vsol_sol / vtoken_ui) * sol_usd
        vsol_sol  = 35_000_000_000 / 1e9
        vtoken_ui = 500_000_000_000_000 / 1e6
        expected_price_usd = (vsol_sol / vtoken_ui) * 150.0
        self.assertAlmostEqual(expected_price_usd, 1.05e-5, places=10)  # = $0.0000105/token
        self.assertAlmostEqual(r["price_usd"], expected_price_usd, places=15)
        self.assertIsNotNone(r["rpc_latency_ms"])
        self.assertEqual(r["curve_address"], "FakeCurveAddr1")

    # ── 2. complete=True → GRADUATED, no price ───────────────────────────────

    def test_graduated_complete_flag(self):
        """complete=True in account data → GRADUATED venue_state, no price."""
        raw = _build_curve_account_bytes(
            virtual_token_reserves=1_000_000,
            virtual_sol_reserves=115_000_000_000,
            complete=True,   # GRADUATED
        )
        rpc_resp = _rpc_ok_response([_account_entry(raw)])

        with patch("research.curve_oracle.derive_curve_address", return_value="FakeCurveAddr2"):
            with patch("requests.post", self._mock_post(rpc_resp)):
                results = curve_oracle.get_curve_prices_batch(
                    mints=[MINT_A],
                    helius_key="test-key",
                    sol_price_usd=150.0,
                    sol_price_age_s=5.0,
                )

        r = results[MINT_A]
        self.assertEqual(r["venue_state"], "GRADUATED")
        self.assertIsNone(r["failure_reason"])
        self.assertIsNone(r["price_usd"])
        self.assertTrue(r["complete"])

    # ── 3. Null account → CURVE_MISSING, NOT graduation ──────────────────────

    def test_account_missing_not_graduation(self):
        """Null account in getMultipleAccounts → CURVE_MISSING, never GRADUATED."""
        rpc_resp = _rpc_ok_response([None])  # null account

        with patch("research.curve_oracle.derive_curve_address", return_value="FakeCurveAddr3"):
            with patch("requests.post", self._mock_post(rpc_resp)):
                results = curve_oracle.get_curve_prices_batch(
                    mints=[MINT_A],
                    helius_key="test-key",
                    sol_price_usd=150.0,
                    sol_price_age_s=5.0,
                )

        r = results[MINT_A]
        self.assertEqual(r["venue_state"], "CURVE_MISSING")
        self.assertEqual(r["failure_reason"], "curve_account_missing")
        # CRITICAL: must NOT be GRADUATED
        self.assertNotEqual(r["venue_state"], "GRADUATED")
        self.assertIsNone(r["price_usd"])

    # ── 4. RPC error → RPC_ERROR, NOT graduation ─────────────────────────────

    def test_rpc_error_not_graduation(self):
        """requests.post raises ConnectionError → RPC_ERROR, never GRADUATED."""
        with patch("research.curve_oracle.derive_curve_address", return_value="FakeCurveAddr4"):
            with patch("requests.post", side_effect=ConnectionError("network down")):
                results = curve_oracle.get_curve_prices_batch(
                    mints=[MINT_A],
                    helius_key="test-key",
                    sol_price_usd=150.0,
                    sol_price_age_s=5.0,
                )

        r = results[MINT_A]
        self.assertEqual(r["venue_state"], "RPC_ERROR")
        self.assertEqual(r["failure_reason"], "curve_rpc_error")
        # CRITICAL: must NOT be GRADUATED
        self.assertNotEqual(r["venue_state"], "GRADUATED")
        self.assertIsNone(r["price_usd"])

    # ── 5. Malformed base64 → PARSE_ERROR ────────────────────────────────────

    def test_parse_error_fails_closed(self):
        """Malformed base64 string → PARSE_ERROR, never GRADUATED."""
        bad_account = {
            "data": ["not!!valid!!base64@@@@", "base64"],
            "executable": False,
            "lamports": 1000,
            "owner": "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P",
            "rentEpoch": 0,
        }
        rpc_resp = _rpc_ok_response([bad_account])

        with patch("research.curve_oracle.derive_curve_address", return_value="FakeCurveAddr5"):
            with patch("requests.post", self._mock_post(rpc_resp)):
                results = curve_oracle.get_curve_prices_batch(
                    mints=[MINT_A],
                    helius_key="test-key",
                    sol_price_usd=150.0,
                    sol_price_age_s=5.0,
                )

        r = results[MINT_A]
        self.assertIn(r["venue_state"], ("PARSE_ERROR",))
        self.assertNotEqual(r["venue_state"], "GRADUATED")
        self.assertIsNone(r["price_usd"])

    # ── 6. Stale SOL/USD → sol_usd_stale failure ─────────────────────────────

    def test_stale_sol_usd_rejects_price(self):
        """SOL/USD price age > SOL_USD_MAX_CACHE_AGE_S → failure_reason=sol_usd_stale."""
        raw = _build_curve_account_bytes(
            virtual_token_reserves=500_000_000_000_000,
            virtual_sol_reserves=35_000_000_000,
            complete=False,
        )
        rpc_resp = _rpc_ok_response([_account_entry(raw)])

        stale_age = _config_stub.SOL_USD_MAX_CACHE_AGE_S + 1   # 121s — over the limit

        with patch("research.curve_oracle.derive_curve_address", return_value="FakeCurveAddr6"):
            with patch("requests.post", self._mock_post(rpc_resp)):
                results = curve_oracle.get_curve_prices_batch(
                    mints=[MINT_A],
                    helius_key="test-key",
                    sol_price_usd=150.0,
                    sol_price_age_s=stale_age,
                )

        r = results[MINT_A]
        self.assertEqual(r["failure_reason"], "sol_usd_stale")
        self.assertIsNone(r["price_usd"])
        self.assertNotEqual(r["venue_state"], "GRADUATED")

    # ── 7. Partial batch: 3 mints, 2 succeed, 1 null ─────────────────────────

    def test_partial_batch_some_succeed(self):
        """Batch of 3 mints: MINT_A succeeds, MINT_B graduated, MINT_C null account."""
        raw_a = _build_curve_account_bytes(
            virtual_token_reserves=500_000_000_000_000,
            virtual_sol_reserves=30_000_000_000,
            complete=False,
        )
        raw_b = _build_curve_account_bytes(
            virtual_token_reserves=100_000,
            virtual_sol_reserves=115_000_000_000,
            complete=True,   # graduated
        )
        # MINT_C → null

        rpc_resp = _rpc_ok_response([
            _account_entry(raw_a),
            _account_entry(raw_b),
            None,                    # MINT_C account missing
        ])

        def _fake_derive(mint: str) -> str:
            return f"FakeCurve_{mint[:8]}"

        with patch("research.curve_oracle.derive_curve_address", side_effect=_fake_derive):
            with patch("requests.post", self._mock_post(rpc_resp)):
                results = curve_oracle.get_curve_prices_batch(
                    mints=[MINT_A, MINT_B, MINT_C],
                    helius_key="test-key",
                    sol_price_usd=150.0,
                    sol_price_age_s=5.0,
                )

        # MINT_A: CURVE_ACTIVE with price
        self.assertEqual(results[MINT_A]["venue_state"], "CURVE_ACTIVE")
        self.assertIsNotNone(results[MINT_A]["price_usd"])
        self.assertGreater(results[MINT_A]["price_usd"], 0)

        # MINT_B: GRADUATED, no price
        self.assertEqual(results[MINT_B]["venue_state"], "GRADUATED")
        self.assertIsNone(results[MINT_B]["price_usd"])
        self.assertTrue(results[MINT_B]["complete"])

        # MINT_C: CURVE_MISSING, NOT graduation
        self.assertEqual(results[MINT_C]["venue_state"], "CURVE_MISSING")
        self.assertEqual(results[MINT_C]["failure_reason"], "curve_account_missing")
        self.assertNotEqual(results[MINT_C]["venue_state"], "GRADUATED")

    # ── 8. Price decimal correctness ──────────────────────────────────────────

    def test_decimal_mismatch_explicit_handling(self):
        """
        Price formula: price_sol = vSol_lamports / vToken_lamports
        price_usd = price_sol * sol_usd / 1e9

        With 6-decimal tokens (pump.fun), 1 token = 1_000_000 base units.
        Verify the formula is NOT incorrectly applying an extra 1e6 divisor.

        Setup: 1 SOL (1e9 lamports) backing 1M tokens (1e6 * 1e6 = 1e12 base units)
        Expected price per token = 1 SOL / 1_000_000 tokens = 1e-6 SOL = 1e-6 * sol_usd
        In lamport terms: 1e9 / 1e12 * 150 / 1e9 = 150 / 1e12 = 1.5e-10 USD per base unit
        Per human token (1e6 base units): 1.5e-10 * 1e6 = 1.5e-4 USD = $0.00015
        """
        v_sol_lamports   = 1_000_000_000        # 1 SOL
        v_token_lamports = 1_000_000_000_000    # 1M tokens * 1e6 decimals

        raw = _build_curve_account_bytes(
            virtual_token_reserves=v_token_lamports,
            virtual_sol_reserves=v_sol_lamports,
            complete=False,
        )
        rpc_resp = _rpc_ok_response([_account_entry(raw)])

        sol_usd = 150.0
        with patch("research.curve_oracle.derive_curve_address", return_value="FakeCurveAddrDec"):
            with patch("requests.post", self._mock_post(rpc_resp)):
                results = curve_oracle.get_curve_prices_batch(
                    mints=[MINT_A],
                    helius_key="test-key",
                    sol_price_usd=sol_usd,
                    sol_price_age_s=5.0,
                )

        r = results[MINT_A]
        self.assertEqual(r["venue_state"], "CURVE_ACTIVE")
        self.assertIsNotNone(r["price_usd"])

        # FIXED 2026-08-07 (PROGRESS-FIX PF0): this assertion previously
        # re-derived "expected" using the SAME buggy formula as the
        # implementation ((v_sol/v_token)*sol_usd/1e9, no token-decimal
        # conversion) — a tautological check that could never fail
        # regardless of correctness, even though this test's own docstring
        # already stated the correct answer ($0.00015). Now asserts against
        # the actual dimensionally-correct value.
        vsol_sol = v_sol_lamports / 1e9
        vtoken_ui = v_token_lamports / 1e6
        expected = (vsol_sol / vtoken_ui) * sol_usd
        self.assertAlmostEqual(expected, 0.00015, places=10)   # sanity: matches docstring
        self.assertAlmostEqual(r["price_usd"], expected, places=15)

        # FIXED 2026-08-07 (PROGRESS-FIX PF0): r["price_usd"] is now ALREADY
        # price-per-human-readable-token (the fix converts virtual_token_reserves
        # to UI units internally) — no further *1e6 adjustment needed. The old
        # version of this block assumed price_usd was still "per base unit" and
        # multiplied by 1e6, which would now be wrong by exactly that factor.
        self.assertAlmostEqual(r["price_usd"], 1.5e-4, places=10)

    def test_progress_fix_pf0_documented_fixture(self):
        """
        PROGRESS-FIX PF0 fixture — exact sample from
        docs/PUMPFUN_COMPATIBILITY_REPORT.md:
            virtual_token_reserves = 1,063,494,656,015,142
            virtual_sol_reserves   = 3,107,652,233
        At ~$150 SOL/USD (back-derived to land on the report's documented
        approximate scale), correct price ~= $4.38e-7/token. The pre-fix
        formula would have given ~$4.38e-13/token (1,000,000x too small).
        """
        v_token_lamports = 1_063_494_656_015_142
        v_sol_lamports   = 3_107_652_233
        sol_usd          = 150.0

        raw = _build_curve_account_bytes(
            virtual_token_reserves=v_token_lamports,
            virtual_sol_reserves=v_sol_lamports,
            complete=False,
        )
        rpc_resp = _rpc_ok_response([_account_entry(raw)])

        with patch("research.curve_oracle.derive_curve_address", return_value="FakeCurveAddrPF0"):
            with patch("requests.post", self._mock_post(rpc_resp)):
                results = curve_oracle.get_curve_prices_batch(
                    mints=[MINT_A],
                    helius_key="test-key",
                    sol_price_usd=sol_usd,
                    sol_price_age_s=5.0,
                )

        r = results[MINT_A]
        self.assertEqual(r["venue_state"], "CURVE_ACTIVE")
        self.assertIsNotNone(r["price_usd"])

        # Correct (fixed) value: on the ~$4.38e-7 scale, not ~$4.38e-13
        self.assertGreater(r["price_usd"], 1e-8)     # rules out the old 1e6x-too-small bug
        self.assertLess(r["price_usd"], 1e-5)
        self.assertAlmostEqual(r["price_usd"], 4.3835e-7, delta=1e-9)

        # Explicitly assert the OLD buggy value is NOT what we get
        buggy_value = (v_sol_lamports / v_token_lamports) * sol_usd / 1e9
        self.assertNotAlmostEqual(r["price_usd"], buggy_value, places=10)
        self.assertAlmostEqual(buggy_value, 4.3835e-13, delta=1e-15)   # confirms old bug's scale

    # ── 9. RPC JSON error field → RPC_ERROR, not graduation ──────────────────

    def test_rpc_json_error_field_not_graduation(self):
        """RPC response with 'error' field → RPC_ERROR for all mints in batch."""
        rpc_resp = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32005, "message": "Rate limit exceeded"},
        }

        with patch("research.curve_oracle.derive_curve_address", return_value="FakeCurveAddrErr"):
            with patch("requests.post", self._mock_post(rpc_resp)):
                results = curve_oracle.get_curve_prices_batch(
                    mints=[MINT_A, MINT_B],
                    helius_key="test-key",
                    sol_price_usd=150.0,
                    sol_price_age_s=5.0,
                )

        for mint in (MINT_A, MINT_B):
            self.assertIn(mint, results)
            r = results[mint]
            self.assertEqual(r["venue_state"], "RPC_ERROR")
            self.assertEqual(r["failure_reason"], "curve_rpc_error")
            self.assertNotEqual(r["venue_state"], "GRADUATED")

    # ── 10. Zero virtualTokenReserves → parse error, not division by zero ────

    def test_zero_virtual_token_reserves_no_div_zero(self):
        """virtualTokenReserves=0 → PARSE_ERROR (no ZeroDivisionError)."""
        raw = _build_curve_account_bytes(
            virtual_token_reserves=0,            # degenerate
            virtual_sol_reserves=35_000_000_000,
            complete=False,
        )
        rpc_resp = _rpc_ok_response([_account_entry(raw)])

        with patch("research.curve_oracle.derive_curve_address", return_value="FakeCurveAddrZero"):
            with patch("requests.post", self._mock_post(rpc_resp)):
                results = curve_oracle.get_curve_prices_batch(
                    mints=[MINT_A],
                    helius_key="test-key",
                    sol_price_usd=150.0,
                    sol_price_age_s=5.0,
                )

        r = results[MINT_A]
        self.assertIsNone(r["price_usd"])
        self.assertNotEqual(r["venue_state"], "GRADUATED")

    # ── 11. get_sol_usd_cached thread-safety smoke test ──────────────────────

    def test_get_sol_usd_cached_returns_tuple(self):
        """get_sol_usd_cached returns (float, float) even if Jupiter fails."""
        with patch("requests.get", side_effect=ConnectionError("no network")):
            price, age = curve_oracle.get_sol_usd_cached()
        # Must return a 2-tuple of floats (even if price=0 from empty cache)
        self.assertIsInstance(price, float)
        self.assertIsInstance(age, float)

    # ── 12. derive_curve_address returns a string or None ────────────────────

    def test_derive_curve_address_returns_string_or_none(self):
        """derive_curve_address for a known mint should return a base58 string or None."""
        # Use a valid-looking (but fake) mint address
        mint = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC mint (real key format)
        result = curve_oracle.derive_curve_address(mint)
        # Either a string (successful derivation) or None (solders not available / error)
        self.assertTrue(result is None or isinstance(result, str))
        if result is not None:
            # Should be a non-empty base58 string (44 chars for a Solana pubkey)
            self.assertGreater(len(result), 0)


class TestParseAccountDirectly(unittest.TestCase):
    """Direct unit tests for _parse_curve_account (internal function)."""

    def test_account_too_short_returns_layout_unknown(self):
        """Data buffer shorter than 49 bytes → curve_layout_unknown."""
        short_data = base64.b64encode(b"\x00" * 10).decode()
        result = curve_oracle._parse_curve_account(
            short_data, "MintXXXX", sol_price_usd=150.0, sol_price_age_s=5.0
        )
        self.assertEqual(result["failure_reason"], "curve_layout_unknown")
        self.assertIsNone(result["price_usd"])

    def test_stale_sol_price_checked_before_decode(self):
        """Stale SOL price should be caught before any parsing attempt."""
        # Even garbage data shouldn't reach the parser
        garbage = base64.b64encode(b"\xff" * 100).decode()
        result = curve_oracle._parse_curve_account(
            garbage, "MintXXXX", sol_price_usd=0.0, sol_price_age_s=200.0
        )
        self.assertEqual(result["failure_reason"], "sol_usd_stale")

    def test_complete_false_yields_curve_active(self):
        """Minimal valid account with complete=False → CURVE_ACTIVE."""
        raw = b"\x00" * 8 + struct.pack("<QQQQQ?",
            1_000_000_000_000,  # virtualTokenReserves
            30_000_000_000,     # virtualSolReserves
            500_000_000_000,
            15_000_000_000,
            1_000_000_000_000_000,
            False,
        )
        data_b64 = base64.b64encode(raw).decode()
        result = curve_oracle._parse_curve_account(
            data_b64, "MintXXXX", sol_price_usd=150.0, sol_price_age_s=10.0
        )
        self.assertEqual(result["venue_state"], "CURVE_ACTIVE")
        self.assertIsNone(result["failure_reason"])
        self.assertIsNotNone(result["price_usd"])
        self.assertGreater(result["price_usd"], 0)
        self.assertAlmostEqual(result["vsol_ui"], 30.0, places=3)


if __name__ == "__main__":
    unittest.main()
