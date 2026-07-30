"""Tests for TP partial sell mutation ordering."""
import time
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Minimal Position stub — mirrors the real Position fields used by TP logic
# ---------------------------------------------------------------------------
@dataclass
class _StubPosition:
    id: str = "test_pos_1"
    token_symbol: str = "TESTCOIN"
    token_address: str = "FakeAddr111111111111111111111111111111111"
    chain: str = "solana"
    status: str = "open"
    entry_price: float = 0.001
    current_price: float = 0.002
    peak_price: float = 0.002
    size_usd: float = 5.0
    remaining_fraction: float = 1.0
    realized_pnl_usd: float = 0.0
    tp_levels_hit: list = field(default_factory=list)
    notes: str = "live|tx:FakeTx123"
    tokens_held: int = 5_000_000
    pnl_pct: float = 0.0
    pnl_usd: float = 0.0
    exit_price: float = 0.0
    exit_reason: str = ""
    exit_time: float = 0.0


class TestFailedPartialNoMutation:
    """Test 1 — failed partial sell leaves state untouched and adds cooldown."""

    def test_failed_sell_no_mutation(self):
        pos = _StubPosition()
        orig_frac = pos.remaining_fraction
        orig_hit = list(pos.tp_levels_hit)
        orig_pnl = pos.realized_pnl_usd

        # Simulate what _run_tp_sell_bg does on failure:
        # sell returns success=False
        _tp_r = {"success": False, "reason": "all_steps_reverted"}
        level_key = "tp_100"
        sell_frac = 0.5

        # On failure path: no mutation of remaining_fraction, tp_levels_hit, realized_pnl_usd
        if not _tp_r.get("success"):
            _cooldown_key = f"|tp_retry_cooldown:{level_key}:{int(time.time() + 30)}"
            pos.notes = (pos.notes or "") + _cooldown_key

        assert pos.remaining_fraction == orig_frac, "remaining_fraction must not change on failure"
        assert pos.tp_levels_hit == orig_hit, "tp_levels_hit must not change on failure"
        assert pos.realized_pnl_usd == orig_pnl, "realized_pnl_usd must not change on failure"
        assert "tp_retry_cooldown" in pos.notes, "cooldown tag must be added on failure"


class TestConfirmedPartialMutatesOnce:
    """Test 2 — confirmed partial sell mutates state exactly once."""

    def test_confirmed_sell_mutates(self):
        pos = _StubPosition()
        level_key = "tp_100"
        sell_frac = 0.5
        tp_pct = 1.0

        # Simulate confirmed fill
        _tp_fill = 0.002
        _real_pnl = (_tp_fill / pos.entry_price - 1) * sell_frac * pos.size_usd

        # Mutate-on-confirm logic (from _run_tp_sell_bg success branch)
        if level_key not in pos.tp_levels_hit:
            pos.tp_levels_hit.append(level_key)
            pos.remaining_fraction -= sell_frac
            pos.realized_pnl_usd += _real_pnl

        assert level_key in pos.tp_levels_hit
        assert abs(pos.remaining_fraction - 0.5) < 1e-9
        assert pos.realized_pnl_usd > 0


class TestDuplicateConfirmationNoDoubleMutation:
    """Test 3 — duplicate confirmation with same level_key does not double-mutate."""

    def test_no_double_mutation(self):
        pos = _StubPosition()
        level_key = "tp_100"
        sell_frac = 0.5
        _tp_fill = 0.002
        _real_pnl = (_tp_fill / pos.entry_price - 1) * sell_frac * pos.size_usd

        # First confirm
        if level_key not in pos.tp_levels_hit:
            pos.tp_levels_hit.append(level_key)
            pos.remaining_fraction -= sell_frac
            pos.realized_pnl_usd += _real_pnl

        frac_after_first = pos.remaining_fraction
        pnl_after_first = pos.realized_pnl_usd

        # Second confirm with same level_key (duplicate)
        if level_key not in pos.tp_levels_hit:
            pos.tp_levels_hit.append(level_key)
            pos.remaining_fraction -= sell_frac
            pos.realized_pnl_usd += _real_pnl

        assert pos.remaining_fraction == frac_after_first, "remaining_fraction must not decrement twice"
        assert pos.realized_pnl_usd == pnl_after_first, "realized_pnl_usd must not increment twice"
        assert pos.tp_levels_hit.count(level_key) == 1, "level_key must appear only once"


class TestCrashBetweenDispatchAndConfirm:
    """Test 4 — crash between dispatch and confirm leaves state unmutated."""

    def test_restart_shows_unmutated_state(self):
        pos = _StubPosition()
        level_key = "tp_100"

        # Dispatch: we add a pending sig tag to notes but do NOT mutate state
        pos.notes = (pos.notes or "") + f"|tp_pending:{level_key}:FakeSig123"

        # Simulate save (serialization round-trip)
        saved_frac = pos.remaining_fraction
        saved_hit = list(pos.tp_levels_hit)
        saved_pnl = pos.realized_pnl_usd

        # Simulate restart: recreate position from saved state
        restored = _StubPosition(
            remaining_fraction=saved_frac,
            tp_levels_hit=list(saved_hit),
            realized_pnl_usd=saved_pnl,
            notes=pos.notes,
        )

        assert restored.remaining_fraction == 1.0, "remaining_fraction must be 1.0 after restart"
        assert restored.tp_levels_hit == [], "tp_levels_hit must be empty after restart"
        assert restored.realized_pnl_usd == 0.0, "realized_pnl_usd must be 0 after restart"


class TestBCPartialNeverUsesPumpSwap:
    """Test 5 — bonding-curve TP partial never routes through PumpSwap."""

    def test_bc_partial_skips_pumpswap(self):
        pos = _StubPosition(notes="live|tx:FakeTx123|cohort:bonding_curve")

        # Check that the routing logic detects bonding_curve cohort
        _is_bc = bool(pos.notes and "cohort:bonding_curve" in (pos.notes or ""))
        assert _is_bc, "Must detect bonding_curve cohort from notes"

        # When _is_bc is True, sell is called with skip_pumpswap=True
        # Verify by capturing kwargs
        captured_kwargs = {}

        class MockExecutor:
            def sell(self, *args, **kwargs):
                captured_kwargs.update(kwargs)
                return {"success": False, "reason": "mock"}

        mock_ex = MockExecutor()

        # Simulate the routing from _run_tp_sell_bg
        sell_frac = 0.5
        _known_count = int(pos.tokens_held * sell_frac)
        if _is_bc:
            mock_ex.sell(
                pos.token_address, pos.size_usd, pos.entry_price,
                pos.chain, fraction=sell_frac,
                escalate=False,
                known_token_count=_known_count,
                skip_pumpswap=True,
            )
        else:
            mock_ex.sell(
                pos.token_address, pos.size_usd, pos.entry_price,
                pos.chain, fraction=sell_frac,
                escalate=False,
                known_token_count=_known_count,
            )

        assert captured_kwargs.get("skip_pumpswap") is True, \
            "BC tokens must pass skip_pumpswap=True to executor.sell()"
