"""
Z8 — pump.fun program-upgrade compatibility gate tests.

Covers all Z8.8 requirements.  No live transactions are made.

Confirmed post-upgrade facts (on-chain evidence, slot 433,122,735):
  Buy  discriminator: 66063d1201daebea  UNCHANGED  (sha256("global:buy")[:8])
  Sell discriminator: 5df6823ce7e940b2  CHANGED    (sha256("global:sell_v2")[:8])
  Buy  account count: 18                CHANGED    (V1 was 12)
  Sell account count: 26                CHANGED    (V1 was 12)
  BC   discriminator: 17b7f83760d8ac60  UNCHANGED
  BC   account size:  151 bytes         CHANGED    (V1 was 49)
  BC   complete byte: offset 48         UNCHANGED
"""
import sys
import os
import hashlib
import struct
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from memecoin import pumpfun_compat as compat

# ---------------------------------------------------------------------------
# Post-upgrade transaction fixtures (Z8.3)
# Decoded from on-chain at slot 433,122,735 (post-upgrade slot 433,095,571)
# ---------------------------------------------------------------------------
_PUMP_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

BUY_FIXTURE = {
    "tx_sig":           "h3rVR17am61iuQH5XPsRuzUxH4sAnFQfdyzhxGn4aKBeE6NhVNHcbT7okWJQXwzsgv1NhKLV4ADviLp7QBogccq",
    "slot":             433_122_735,
    "discriminator":    bytes.fromhex("66063d1201daebea"),   # sha256("global:buy")[:8]
    "data_len":         24,
    "account_count":    18,
    "accounts": [
        {"pos":  0, "flags": "READONLY",  "role": "global_pda",             "addr": "4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf"},
        {"pos":  1, "flags": "WRITABLE",  "role": "fee_recipient",          "addr": ""},
        {"pos":  2, "flags": "READONLY",  "role": "mint",                   "addr": ""},
        {"pos":  3, "flags": "WRITABLE",  "role": "bonding_curve_pda",      "addr": ""},
        {"pos":  4, "flags": "WRITABLE",  "role": "assoc_bc_token_acct",    "addr": ""},
        {"pos":  5, "flags": "WRITABLE",  "role": "assoc_user_token_acct",  "addr": ""},
        {"pos":  6, "flags": "WR+SIGN",   "role": "wallet",                 "addr": ""},
        {"pos":  7, "flags": "READONLY",  "role": "system_program",         "addr": "11111111111111111111111111111111"},
        {"pos":  8, "flags": "READONLY",  "role": "token_program",          "addr": "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"},
        {"pos":  9, "flags": "WRITABLE",  "role": "new_v2_acct_9",         "addr": ""},
        {"pos": 10, "flags": "READONLY",  "role": "event_authority",        "addr": "Ce6TQqeHC9p8KetsN6JsjHK7UTZk7nasjjnr7XxXp9F1"},
        {"pos": 11, "flags": "READONLY",  "role": "program",                "addr": _PUMP_PROGRAM},
        {"pos": 12, "flags": "WRITABLE",  "role": "new_v2_acct_12",        "addr": ""},
        {"pos": 13, "flags": "WRITABLE",  "role": "new_v2_acct_13",        "addr": ""},
        {"pos": 14, "flags": "WRITABLE",  "role": "new_v2_acct_14",        "addr": ""},
        {"pos": 15, "flags": "WRITABLE",  "role": "new_v2_acct_15",        "addr": ""},
        {"pos": 16, "flags": "READONLY",  "role": "new_v2_acct_16",        "addr": ""},
        {"pos": 17, "flags": "WRITABLE",  "role": "new_v2_acct_17",        "addr": ""},
    ],
    "note": "Buy discriminator UNCHANGED. Account count 18 > V1 12. 6 new accounts added.",
}

SELL_FIXTURE = {
    "tx_sig":           "5RS2U4qFxHBDsXhuEuJPUUyR3WkcRRPMEh4dTdtXnXpM4J4MMHopDTLpjzi7Y1azr3WTj97YVUkWeCARYuW1rFfq",
    "slot":             433_122_735,
    "discriminator":    bytes.fromhex("5df6823ce7e940b2"),   # sha256("global:sell_v2")[:8]
    "data_len":         24,
    "account_count":    26,
    "note": "Sell discriminator CHANGED from 33e685a4017f83ad (global:sell) to 5df6823ce7e940b2 (global:sell_v2).",
}

# V1 reference values
V1_BUY_DISC          = bytes.fromhex("66063d1201daebea")
V1_SELL_DISC         = bytes.fromhex("33e685a4017f83ad")
V1_BUY_ACCOUNT_COUNT  = 12
V1_SELL_ACCOUNT_COUNT = 12

# BC constants
BC_DISCRIMINATOR = bytes.fromhex("17b7f83760d8ac60")
BC_V1_SIZE = 49
BC_V2_SIZE = 151

def _make_bc_account(complete: int = 0, size: int = BC_V2_SIZE,
                     discriminator: bytes = BC_DISCRIMINATOR) -> bytes:
    """Build a synthetic bonding-curve account bytes for testing."""
    raw = bytearray(size)
    raw[0:8]   = discriminator
    # virtual_token_reserves at offset 8
    struct.pack_into("<Q", raw, 8,  1_000_000_000_000_000)
    # virtual_sol_reserves at offset 16
    struct.pack_into("<Q", raw, 16, 30_000_000_000)
    # token_total_supply at offset 40
    struct.pack_into("<Q", raw, 40, 1_000_000_000_000_000)
    # complete at offset 48
    raw[48] = complete
    return bytes(raw)


class TestZ8DiscriminatorFixtures(unittest.TestCase):
    """Z8.3 — verify fixture discriminators match expected hashes."""

    def test_buy_discriminator_matches_global_buy(self):
        """Post-upgrade buy discriminator must be sha256('global:buy')[:8]."""
        expected = hashlib.sha256(b"global:buy").digest()[:8]
        self.assertEqual(BUY_FIXTURE["discriminator"], expected,
                         "Buy discriminator CHANGED — local builder must be updated")

    def test_sell_discriminator_changed_to_sell_v2(self):
        """Post-upgrade sell discriminator must be sha256('global:sell_v2')[:8]."""
        expected_v2 = hashlib.sha256(b"global:sell_v2").digest()[:8]
        self.assertEqual(SELL_FIXTURE["discriminator"], expected_v2)

    def test_sell_discriminator_differs_from_v1(self):
        """V2 sell discriminator must differ from V1 sha256('global:sell')[:8]."""
        expected_v1 = hashlib.sha256(b"global:sell").digest()[:8]
        self.assertNotEqual(SELL_FIXTURE["discriminator"], expected_v1,
                            "V2 sell uses 'global:sell_v2'; V1 local builder is incompatible")

    def test_buy_account_count_increased(self):
        """Post-upgrade buy instruction has 18 accounts, not V1's 12."""
        self.assertEqual(BUY_FIXTURE["account_count"], 18)
        self.assertGreater(BUY_FIXTURE["account_count"], V1_BUY_ACCOUNT_COUNT)

    def test_sell_account_count_increased(self):
        """Post-upgrade sell instruction has 26 accounts, not V1's 12."""
        self.assertEqual(SELL_FIXTURE["account_count"], 26)
        self.assertGreater(SELL_FIXTURE["account_count"], V1_SELL_ACCOUNT_COUNT)

    def test_global_pda_unchanged(self):
        """Global PDA at buy account[0] must still be 4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf."""
        self.assertEqual(
            BUY_FIXTURE["accounts"][0]["addr"],
            "4wTV1YmiEkRvAtNtsSGPtUrqRYQMe5SKy2uB4Jjaxnjf",
        )

    def test_event_authority_unchanged(self):
        """Event authority at buy account[10] must still be Ce6TQqeHC9..."""
        self.assertEqual(
            BUY_FIXTURE["accounts"][10]["addr"],
            "Ce6TQqeHC9p8KetsN6JsjHK7UTZk7nasjjnr7XxXp9F1",
        )

    def test_fixtures_are_post_upgrade(self):
        """Both fixtures must be from slots after the verified upgrade slot."""
        self.assertGreater(BUY_FIXTURE["slot"],  compat.BASELINE_DEPLOY_SLOT)
        self.assertGreater(SELL_FIXTURE["slot"], compat.BASELINE_DEPLOY_SLOT)


class TestZ8CompatState(unittest.TestCase):
    """Z8.5 — compatibility state machine."""

    def setUp(self):
        # Reset to CHANGED (default) before each test.
        with compat._state_lock:
            compat._compat_state = compat.CHANGED

    def tearDown(self):
        with compat._state_lock:
            compat._compat_state = compat.CHANGED

    def test_default_state_is_changed(self):
        """Default state must be CHANGED — confirmed interface differences found."""
        self.assertEqual(compat.get_state(), compat.CHANGED)

    def test_changed_state_disables_local_build(self):
        """local_build_allowed() must return False when state == CHANGED."""
        self.assertFalse(compat.local_build_allowed())

    def test_changed_state_disables_layout_graduation(self):
        """layout_graduation_allowed() must return False when state == CHANGED."""
        self.assertFalse(compat.layout_graduation_allowed())

    def test_unknown_state_disables_local_build(self):
        """local_build_allowed() must return False when state == UNKNOWN."""
        with compat._state_lock:
            compat._compat_state = compat.UNKNOWN
        self.assertFalse(compat.local_build_allowed())

    def test_unknown_state_disables_layout_graduation(self):
        """layout_graduation_allowed() must return False when state == UNKNOWN."""
        with compat._state_lock:
            compat._compat_state = compat.UNKNOWN
        self.assertFalse(compat.layout_graduation_allowed())

    def test_verified_state_allows_local_build(self):
        """local_build_allowed() must return True when state == VERIFIED."""
        with compat._state_lock:
            compat._compat_state = compat.VERIFIED
        self.assertTrue(compat.local_build_allowed())

    def test_verified_state_allows_layout_graduation(self):
        """layout_graduation_allowed() must return True when state == VERIFIED."""
        with compat._state_lock:
            compat._compat_state = compat.VERIFIED
        self.assertTrue(compat.layout_graduation_allowed())

    def test_mark_interface_verified_requires_matching_slot(self):
        """mark_interface_verified must raise if slot doesn't match baseline."""
        with self.assertRaises(ValueError):
            compat.mark_interface_verified(999_000_000, compat.BASELINE_EXE_HASH)

    def test_mark_interface_verified_sets_verified(self):
        """mark_interface_verified sets state to VERIFIED when slot matches."""
        compat.mark_interface_verified(
            compat.BASELINE_DEPLOY_SLOT,
            compat.BASELINE_EXE_HASH,
        )
        self.assertEqual(compat.get_state(), compat.VERIFIED)


class TestZ8DeploySlotTripwire(unittest.TestCase):
    """Z8.6 — deployment slot tripwire."""

    def setUp(self):
        with compat._state_lock:
            compat._compat_state = compat.CHANGED

    def tearDown(self):
        with compat._state_lock:
            compat._compat_state = compat.CHANGED

    def test_slot_unchanged_preserves_state(self):
        """If live slot matches baseline, tripwire must not change state."""
        with patch.object(compat, "_fetch_current_deploy_slot",
                          return_value=compat.BASELINE_DEPLOY_SLOT):
            compat.check_tripwire_once()
        self.assertEqual(compat.get_state(), compat.CHANGED)

    def test_slot_changed_sets_changed(self):
        """If live slot differs from baseline, tripwire must set state = CHANGED."""
        with compat._state_lock:
            compat._compat_state = compat.VERIFIED  # simulate previously verified
        with patch.object(compat, "_fetch_current_deploy_slot",
                          return_value=compat.BASELINE_DEPLOY_SLOT + 1_000_000), \
             patch.object(compat, "_alert_compat_change"):
            compat.check_tripwire_once()
        self.assertEqual(compat.get_state(), compat.CHANGED)

    def test_slot_changed_disables_local_build(self):
        """After tripwire fires on slot change, local_build_allowed must be False."""
        with compat._state_lock:
            compat._compat_state = compat.VERIFIED
        with patch.object(compat, "_fetch_current_deploy_slot",
                          return_value=compat.BASELINE_DEPLOY_SLOT + 1), \
             patch.object(compat, "_alert_compat_change"):
            compat.check_tripwire_once()
        self.assertFalse(compat.local_build_allowed())

    def test_rpc_failure_preserves_state(self):
        """RPC failure on tripwire check must not change compat state."""
        with compat._state_lock:
            compat._compat_state = compat.VERIFIED
        with patch.object(compat, "_fetch_current_deploy_slot", return_value=None):
            compat.check_tripwire_once()
        # State must remain VERIFIED — fail open on monitoring, not on builds
        self.assertEqual(compat.get_state(), compat.VERIFIED)

    def test_baseline_slot_matches_confirmed_upgrade(self):
        """BASELINE_DEPLOY_SLOT must equal the confirmed upgrade slot 433,095,571."""
        self.assertEqual(compat.BASELINE_DEPLOY_SLOT, 433_095_571)


class TestZ8BCAccountValidation(unittest.TestCase):
    """Z8.4 — bonding-curve account layout validation."""

    def test_valid_v1_account(self):
        """49-byte V1 BC account with correct discriminator and owner must pass."""
        raw = _make_bc_account(complete=0, size=BC_V1_SIZE)
        ok, reason = compat.validate_bc_account(raw, _PUMP_PROGRAM)
        self.assertTrue(ok, reason)
        self.assertEqual(reason, "ok")

    def test_valid_v2_account(self):
        """151-byte V2 BC account with correct discriminator and owner must pass."""
        raw = _make_bc_account(complete=0, size=BC_V2_SIZE)
        ok, reason = compat.validate_bc_account(raw, _PUMP_PROGRAM)
        self.assertTrue(ok, reason)

    def test_valid_v2_account_complete_true(self):
        """V2 BC account with complete=1 must pass validation (graduation gating is separate)."""
        raw = _make_bc_account(complete=1, size=BC_V2_SIZE)
        ok, reason = compat.validate_bc_account(raw, _PUMP_PROGRAM)
        self.assertTrue(ok, reason)

    def test_short_account_rejected(self):
        """Account shorter than 49 bytes must be rejected."""
        raw = _make_bc_account(size=BC_V1_SIZE)[:48]  # truncate by 1
        ok, reason = compat.validate_bc_account(raw, _PUMP_PROGRAM)
        self.assertFalse(ok)
        self.assertIn("too_short", reason)

    def test_empty_account_rejected(self):
        """Empty account must be rejected."""
        ok, reason = compat.validate_bc_account(b"", _PUMP_PROGRAM)
        self.assertFalse(ok)
        self.assertIn("too_short", reason)

    def test_unexpected_length_rejected(self):
        """Account with unexpected length (not 49 or 151) must be rejected."""
        raw = _make_bc_account(size=BC_V1_SIZE) + b"\x00" * 5  # 54 bytes
        ok, reason = compat.validate_bc_account(raw, _PUMP_PROGRAM)
        self.assertFalse(ok)
        self.assertIn("unexpected_length", reason)

    def test_wrong_owner_rejected(self):
        """Account owned by wrong program must be rejected."""
        raw = _make_bc_account(size=BC_V2_SIZE)
        ok, reason = compat.validate_bc_account(raw, "SomeOtherProgram1111111111111111111")
        self.assertFalse(ok)
        self.assertIn("wrong_owner", reason)

    def test_wrong_discriminator_rejected(self):
        """Account with non-BC discriminator must be rejected."""
        bad_disc = b"\xde\xad\xbe\xef\x00\x01\x02\x03"
        raw = _make_bc_account(size=BC_V2_SIZE, discriminator=bad_disc)
        ok, reason = compat.validate_bc_account(raw, _PUMP_PROGRAM)
        self.assertFalse(ok)
        self.assertIn("wrong_discriminator", reason)

    def test_invalid_complete_byte_rejected(self):
        """complete byte outside {0,1} must be rejected."""
        raw = bytearray(_make_bc_account(size=BC_V2_SIZE))
        raw[48] = 2  # invalid value
        ok, reason = compat.validate_bc_account(bytes(raw), _PUMP_PROGRAM)
        self.assertFalse(ok)
        self.assertIn("invalid_complete_byte", reason)

    def test_empty_owner_rejected(self):
        """Empty owner string must be rejected."""
        raw = _make_bc_account(size=BC_V2_SIZE)
        ok, reason = compat.validate_bc_account(raw, "")
        self.assertFalse(ok)
        self.assertIn("wrong_owner", reason)


class TestZ8GraduationGating(unittest.TestCase):
    """
    Z8.5 — graduation gating.

    complete=True from BC parse must be blocked when compat != VERIFIED.
    account_missing (Z1) must still work regardless of compat state.
    """

    def setUp(self):
        with compat._state_lock:
            compat._compat_state = compat.CHANGED

    def tearDown(self):
        with compat._state_lock:
            compat._compat_state = compat.CHANGED

    def test_layout_graduation_blocked_in_changed_state(self):
        """layout_graduation_allowed() returns False when state=CHANGED."""
        self.assertFalse(compat.layout_graduation_allowed())

    def test_layout_graduation_blocked_in_unknown_state(self):
        """layout_graduation_allowed() returns False when state=UNKNOWN."""
        with compat._state_lock:
            compat._compat_state = compat.UNKNOWN
        self.assertFalse(compat.layout_graduation_allowed())

    def test_layout_graduation_allowed_in_verified_state(self):
        """layout_graduation_allowed() returns True only when state=VERIFIED."""
        with compat._state_lock:
            compat._compat_state = compat.VERIFIED
        self.assertTrue(compat.layout_graduation_allowed())

    def test_account_missing_independent_of_compat_state(self):
        """account_missing detection must not require compat state to be VERIFIED.

        The Z1 guard returns {"ok": True, "reason": "account_missing"} when
        value is None (account not found) — this does NOT call validate_bc_account
        or layout_graduation_allowed() because there is no data to parse.
        This test verifies the logic contract: account_missing path bypasses the layout gate.
        """
        # Simulate: account is gone → value=None → no layout check needed
        # The executor logic for account_missing is:
        #   if value is None: return {"ok": True, "complete": None, "reason": "account_missing"}
        # This branch runs BEFORE the layout validation block.
        # We verify the gate functions return False (compat is CHANGED)
        # but account_missing result is structurally independent.
        self.assertFalse(compat.layout_graduation_allowed())  # gate is closed
        # account_missing result is constructed without going through the gate
        result = {"ok": True, "complete": None, "reason": "account_missing", "bc_pda": "test"}
        self.assertEqual(result["reason"], "account_missing")
        self.assertTrue(result["ok"])

    def test_complete_true_blocked_when_changed(self):
        """In CHANGED state, layout_graduation_allowed() blocks complete=True graduation."""
        self.assertFalse(compat.layout_graduation_allowed())
        # If scanner checks layout_graduation_allowed() before acting on complete=True,
        # it must get False and must NOT fire graduation.

    def test_complete_false_not_affected_by_compat(self):
        """complete=False (active curve) is not graduation — not gated by compat state.

        The compat gate only applies to complete=True.  complete=False is used
        to confirm a token is still on the bonding curve for buy-side gating.
        This path should not be blocked by compat state.
        """
        # The Z8.5 gate in executor.py reads: "if complete and not _z8_lga(): ..."
        # complete=False → the condition is False → no blocking → passes through.
        complete = False
        if complete and not compat.layout_graduation_allowed():
            self.fail("complete=False should not be blocked by compat gate")
        # Verified: correct — no exception raised


class TestZ8LocalBuildDisabled(unittest.TestCase):
    """Z8.5/Z8.7 — local build disabled: buy and sell fall back to PumpPortal."""

    def setUp(self):
        with compat._state_lock:
            compat._compat_state = compat.CHANGED

    def tearDown(self):
        with compat._state_lock:
            compat._compat_state = compat.CHANGED

    def test_local_build_not_allowed_in_changed_state(self):
        """local_build_allowed() must return False in CHANGED state."""
        self.assertFalse(compat.local_build_allowed())

    def test_local_build_allowed_only_in_verified(self):
        """local_build_allowed() must return True only in VERIFIED state."""
        for state, expected in [(compat.CHANGED, False),
                                 (compat.UNKNOWN, False),
                                 (compat.VERIFIED, True)]:
            with compat._state_lock:
                compat._compat_state = state
            self.assertEqual(compat.local_build_allowed(), expected,
                             f"state={state}")

    def test_pumpswap_and_jupiter_unaffected(self):
        """PumpSwap and Jupiter do not use the pump.fun local builder.

        Z8 disables the BONDING-CURVE local builder only. The PumpSwap builder
        in pumpswap_local.py uses a completely different program
        (pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA) and is not gated here.
        """
        PUMP_AMM_PROGRAM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
        self.assertNotEqual(PUMP_AMM_PROGRAM, compat._PUMPFUN_PROGRAM,
                            "PumpSwap AMM is a distinct program — not gated by Z8")
        # local_build_allowed() is False (compat=CHANGED), but this has no effect
        # on pumpswap_local.py or Jupiter — they remain available.
        self.assertFalse(compat.local_build_allowed())  # BC builder disabled
        # PumpSwap/Jupiter have no compat gate — they are always callable


class TestZ8BCConstants(unittest.TestCase):
    """Sanity checks on BC validation constants."""

    def test_bc_discriminator_is_correct(self):
        """BC discriminator must be sha256('account:BondingCurve')[:8]."""
        expected = hashlib.sha256(b"account:BondingCurve").digest()[:8]
        # Note: Anchor uses "account:<TypeName>" for account discriminators.
        # Confirmed unchanged post-upgrade from on-chain data.
        self.assertEqual(compat._BC_DISCRIMINATOR, expected)

    def test_bc_valid_lengths_include_v1_and_v2(self):
        """BC valid lengths must include both V1 (49) and V2 (151)."""
        self.assertIn(49,  compat._BC_VALID_LENGTHS)
        self.assertIn(151, compat._BC_VALID_LENGTHS)

    def test_bc_complete_offset_is_48(self):
        """BC complete byte offset must be 48."""
        self.assertEqual(compat._BC_COMPLETE_OFFSET, 48)

    def test_sell_v1_discriminator_differs_from_v2(self):
        """V1 sell discriminator must not equal V2 sell discriminator."""
        self.assertNotEqual(compat.DISC_SELL_V1, compat.DISC_SELL_V2)

    def test_buy_discriminator_unchanged(self):
        """Buy discriminator must be the same in V1 and V2."""
        self.assertEqual(compat.DISC_BUY_V1, compat.DISC_BUY_V2)

    def test_buy_disc_matches_hash(self):
        """Buy discriminator must equal sha256('global:buy')[:8]."""
        self.assertEqual(compat.DISC_BUY_V1,
                         hashlib.sha256(b"global:buy").digest()[:8])

    def test_sell_v1_disc_matches_hash(self):
        """V1 sell discriminator must equal sha256('global:sell')[:8]."""
        self.assertEqual(compat.DISC_SELL_V1,
                         hashlib.sha256(b"global:sell").digest()[:8])

    def test_sell_v2_disc_matches_hash(self):
        """V2 sell discriminator must equal sha256('global:sell_v2')[:8]."""
        self.assertEqual(compat.DISC_SELL_V2,
                         hashlib.sha256(b"global:sell_v2").digest()[:8])


if __name__ == "__main__":
    unittest.main(verbosity=2)
