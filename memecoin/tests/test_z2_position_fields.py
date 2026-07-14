"""
Z2/Z7 — Position structured execution state fields.

Tests:
  1. New fields exist on Position with correct defaults
  2. Fields serialize/deserialize correctly via asdict() + Position(**d)
  3. Backfill in _load_positions: policy_cohort from is_live flag
  4. Backfill: lifecycle_state from notes cohort tags
  5. Restart safety: pending_signature survives round-trip through disk
  6. policy_cohort assignment: paper positions get "paper_reference",
     live positions get "strategy_pure_rider"
"""
import sys
import os
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from memecoin.portfolio import Position, _load_positions


def _make_pos(**kwargs) -> Position:
    """Build a minimal Position for testing."""
    defaults = dict(
        id="test001",
        signal_id="sig001",
        chain="solana",
        token_address="AAAA",
        token_symbol="TEST",
        signal_type="social_alert",
        strength="medium",
        whale_count=0,
        whale_tiers=[],
        whales_involved=[],
    )
    defaults.update(kwargs)
    return Position(**defaults)


class TestZ2PositionFields(unittest.TestCase):

    def test_new_fields_exist_with_defaults(self):
        """All Z2/Z7 fields exist on Position with empty/zero defaults."""
        pos = _make_pos()
        self.assertEqual(pos.policy_cohort, "")
        self.assertEqual(pos.lifecycle_state, "")
        self.assertEqual(pos.exit_intent_reason, "")
        self.assertEqual(pos.exit_intent_ts, 0.0)
        self.assertEqual(pos.exit_intent_policy, "")
        self.assertEqual(pos.venue_state_json, "")
        self.assertEqual(pos.pending_signature, "")
        self.assertEqual(pos.pending_signature_route, "")
        self.assertEqual(pos.pending_signature_ts, 0.0)

    def test_serialise_deserialise_roundtrip(self):
        """asdict + Position(**d) round-trips new fields correctly."""
        pos = _make_pos(
            policy_cohort="strategy_pure_rider",
            lifecycle_state="graduated",
            exit_intent_reason="hard_stop",
            exit_intent_ts=1234567890.0,
            exit_intent_policy="strategy_pure_rider",
            venue_state_json='{"primary": "pump_amm"}',
            pending_signature="TestSig1111111111111111111111",
            pending_signature_route="executor",
            pending_signature_ts=1234567891.0,
        )
        d = asdict(pos)
        pos2 = Position(**d)
        self.assertEqual(pos2.policy_cohort, "strategy_pure_rider")
        self.assertEqual(pos2.lifecycle_state, "graduated")
        self.assertEqual(pos2.exit_intent_reason, "hard_stop")
        self.assertEqual(pos2.exit_intent_ts, 1234567890.0)
        self.assertEqual(pos2.pending_signature, "TestSig1111111111111111111111")
        self.assertEqual(pos2.pending_signature_route, "executor")
        self.assertEqual(pos2.venue_state_json, '{"primary": "pump_amm"}')

    def test_backfill_policy_cohort_from_is_live(self):
        """Positions loaded from disk backfill policy_cohort from is_live flag."""
        pos_live  = asdict(_make_pos(id="L001", is_live=True))
        pos_paper = asdict(_make_pos(id="p001", is_live=False))
        # Remove Z2 fields to simulate old snapshot
        for f in ("policy_cohort", "lifecycle_state", "exit_intent_reason",
                  "exit_intent_ts", "exit_intent_policy", "venue_state_json",
                  "pending_signature", "pending_signature_route", "pending_signature_ts"):
            pos_live.pop(f, None)
            pos_paper.pop(f, None)

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump([pos_live, pos_paper], f)
            tmp_path = Path(f.name)

        try:
            from memecoin.config import POSITIONS_FILE
            with patch("memecoin.portfolio.POSITIONS_FILE", tmp_path):
                loaded = _load_positions()
        finally:
            tmp_path.unlink(missing_ok=True)

        self.assertEqual(loaded["L001"].policy_cohort, "strategy_pure_rider")
        self.assertEqual(loaded["p001"].policy_cohort, "paper_reference")

    def test_backfill_lifecycle_state_from_notes(self):
        """Lifecycle state backfills from cohort notes tags."""
        pos_bc   = asdict(_make_pos(id="bc1", notes="|cohort:bonding_curve"))
        pos_grad = asdict(_make_pos(id="gd1", notes="|cohort:graduated"))
        pos_none = asdict(_make_pos(id="nn1", notes=""))
        for p in (pos_bc, pos_grad, pos_none):
            p.pop("lifecycle_state", None)
            p.pop("policy_cohort", None)

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump([pos_bc, pos_grad, pos_none], f)
            tmp_path = Path(f.name)

        try:
            from memecoin.config import POSITIONS_FILE
            with patch("memecoin.portfolio.POSITIONS_FILE", tmp_path):
                loaded = _load_positions()
        finally:
            tmp_path.unlink(missing_ok=True)

        self.assertEqual(loaded["bc1"].lifecycle_state, "bonding_curve")
        self.assertEqual(loaded["gd1"].lifecycle_state, "graduated")
        self.assertEqual(loaded["nn1"].lifecycle_state, "")

    def test_pending_signature_survives_disk_roundtrip(self):
        """pending_signature is preserved after save+load (restart safety)."""
        SIG = "5mqkMBkc111111111111111111111111111111111111111111111"
        pos = asdict(_make_pos(
            id="L999",
            is_live=True,
            policy_cohort="strategy_pure_rider",
            pending_signature=SIG,
            pending_signature_route="executor",
            pending_signature_ts=9999999.0,
        ))

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump([pos], f)
            tmp_path = Path(f.name)

        try:
            from memecoin.config import POSITIONS_FILE
            with patch("memecoin.portfolio.POSITIONS_FILE", tmp_path):
                loaded = _load_positions()
        finally:
            tmp_path.unlink(missing_ok=True)

        self.assertEqual(loaded["L999"].pending_signature, SIG)
        self.assertEqual(loaded["L999"].pending_signature_route, "executor")
        self.assertEqual(loaded["L999"].pending_signature_ts, 9999999.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
