"""
Unit test: signal-anchored stop logic.

Scenario (from spec):
  signal_price = 1.00
  fill_price   = 1.25   (25% slippage above signal)
  hard_stop    = -0.35

  stop_level (signal-anchored) = 1.00 × 0.65 = 0.65
  stop_level (fill-anchored)   = 1.25 × 0.65 = 0.8125

  At price 0.82:
    • fill-anchored  stop → 0.82 < 0.8125 → STOPS OUT (wrong)
    • signal-anchored stop → 0.82 > 0.65  → SURVIVES   (correct)

  At price 1.50:
    • (1.50 - 1.25) / 1.25 = +20% from fill → WINNER
"""

import sys
import os

# Allow running without the full bot installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ─── Pure-logic helper (mirrors scanner.py / portfolio.py stop logic) ────────

def signal_anchored_stop_level(entry_price: float, signal_price: float,
                                hard_stop_pct: float) -> float:
    """Return the price level at which a hard stop should fire."""
    if signal_price > 0 and entry_price > signal_price:
        return signal_price * (1 + hard_stop_pct)
    return entry_price * (1 + hard_stop_pct)


def fill_anchored_stop_level(entry_price: float, hard_stop_pct: float) -> float:
    return entry_price * (1 + hard_stop_pct)


def should_stop(current_price: float, stop_level: float) -> bool:
    return current_price <= stop_level


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_stop_level_is_signal_anchored():
    """stop_level must be anchored to signal price, not fill price."""
    stop = signal_anchored_stop_level(
        entry_price=1.25, signal_price=1.00, hard_stop_pct=-0.35
    )
    assert abs(stop - 0.65) < 1e-9, f"Expected 0.65, got {stop}"


def test_fill_anchored_stop_level_for_reference():
    """Confirm the old (wrong) stop would have been 0.8125."""
    stop = fill_anchored_stop_level(entry_price=1.25, hard_stop_pct=-0.35)
    assert abs(stop - 0.8125) < 1e-9, f"Expected 0.8125, got {stop}"


def test_position_survives_dip_to_082():
    """
    Spec scenario: signal=1.00, fill=1.25, hard_stop=-0.35, dip to 0.82.
    Signal-anchored stop level = 0.65.  price 0.82 > 0.65 → must survive.
    (Note: fill-anchored stop = 0.8125, and 0.82 > 0.8125 so it also
    survives at exactly 0.82.  The critical difference is between 0.65–0.81:
    see test_signal_anchor_rescues_position_in_danger_zone below.)
    """
    stop_sa = signal_anchored_stop_level(1.25, 1.00, -0.35)  # 0.65
    price   = 0.82
    assert not should_stop(price, stop_sa), (
        f"Signal-anchored: price {price} should NOT stop at level {stop_sa}"
    )


def test_signal_anchor_rescues_position_in_danger_zone():
    """
    At price 0.75 (between signal-anchored 0.65 and fill-anchored 0.8125):
    fill-anchored stops the position; signal-anchored does not.
    This is the core correctness proof for the feature.
    """
    stop_sa = signal_anchored_stop_level(1.25, 1.00, -0.35)  # 0.65
    stop_fa = fill_anchored_stop_level(1.25, -0.35)           # 0.8125
    price   = 0.75   # between the two stop levels

    assert not should_stop(price, stop_sa), (
        f"Signal-anchored: price {price} should NOT stop at level {stop_sa}"
    )
    assert should_stop(price, stop_fa), (
        f"Fill-anchored: price {price} SHOULD stop at level {stop_fa}"
    )


def test_position_closes_as_winner_at_150():
    """After surviving the dip, price 1.50 → +20% from fill → winner."""
    entry_price = 1.25
    exit_price  = 1.50
    pnl_pct = (exit_price - entry_price) / entry_price
    assert pnl_pct > 0, f"Expected positive PnL, got {pnl_pct:.2%}"
    assert abs(pnl_pct - 0.20) < 1e-9, f"Expected +20%, got {pnl_pct:.2%}"


def test_no_anchor_when_fill_equals_signal():
    """When fill == signal, both methods give the same stop level."""
    stop_sa = signal_anchored_stop_level(1.00, 1.00, -0.35)
    stop_fa = fill_anchored_stop_level(1.00, -0.35)
    assert abs(stop_sa - stop_fa) < 1e-9


def test_no_anchor_when_signal_price_zero():
    """If signal_price is 0 (unknown), fall back to fill-anchored."""
    stop_sa = signal_anchored_stop_level(1.25, 0.0, -0.35)
    stop_fa = fill_anchored_stop_level(1.25, -0.35)
    assert abs(stop_sa - stop_fa) < 1e-9


def test_size_normalisation():
    """
    Size multiplier formula: size_mult = base_stop_pct / stop_dist_from_fill
    floor at 0.5×, cap at 1.0×.

    signal=1.00, pp=1.25, hard_stop=-0.35
      stop_level      = 0.65
      stop_dist_fill  = (1.25 - 0.65) / 1.25 = 0.48
      size_mult       = 0.35 / 0.48 ≈ 0.7292
    """
    signal_price  = 1.00
    pp_price      = 1.25
    hard_stop_pct = -0.35

    stop_level     = signal_price * (1 + hard_stop_pct)         # 0.65
    stop_dist_fill = (pp_price - stop_level) / pp_price          # 0.48
    size_mult      = abs(hard_stop_pct) / stop_dist_fill         # ~0.7292
    size_mult      = max(0.5, min(1.0, size_mult))

    assert abs(stop_level - 0.65) < 1e-9
    assert abs(stop_dist_fill - 0.48) < 1e-9
    assert 0.72 < size_mult < 0.74, f"Expected ~0.73, got {size_mult:.4f}"


def test_size_normalisation_floor():
    """Extreme slippage (fill 3× signal): floor at 0.5×."""
    signal_price  = 1.00
    pp_price      = 3.00          # very high fill
    hard_stop_pct = -0.35

    stop_level     = signal_price * (1 + hard_stop_pct)  # 0.65
    stop_dist_fill = (pp_price - stop_level) / pp_price  # (3.00-0.65)/3.00 ≈ 0.783
    size_mult      = abs(hard_stop_pct) / stop_dist_fill # 0.35/0.783 ≈ 0.447 → floored
    size_mult      = max(0.5, min(1.0, size_mult))
    assert size_mult == 0.5, f"Expected floor 0.5, got {size_mult}"


if __name__ == "__main__":
    tests = [
        test_stop_level_is_signal_anchored,
        test_fill_anchored_stop_level_for_reference,
        test_position_survives_dip_to_082,
        test_position_closes_as_winner_at_150,
        test_no_anchor_when_fill_equals_signal,
        test_no_anchor_when_signal_price_zero,
        test_size_normalisation,
        test_size_normalisation_floor,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
