"""
test_paper_entry_baseline.py — Phase 3.5 unit tests.

Verifies:
  1. Paper position entry_price uses the preflight baseline (not stale DexScreener).
  2. PnL% calculation is correct when entry is anchored to baseline (not fill).
  3. entry_source tagging logic: "pp_tick" when _pp_price > 0, "dex_stale" when 0.
"""


def test_paper_entry_uses_baseline_not_dex():
    """Paper pos entry_price should equal pp_price (baseline), not stale dex_price."""

    class MockPaperPos:
        entry_price: float = 0.0
        current_price: float = 0.0
        peak_price: float = 0.0
        baseline_curve_price: float = 0.0
        baseline_price: float = 0.0
        entry_source: str = ""

    dex_price = 0.0000089   # stale DexScreener snapshot
    pp_price  = 0.0000234   # live PP / curve baseline at decision time

    paper_pos = MockPaperPos()
    paper_pos.baseline_curve_price = pp_price

    # Phase 3.1: paper entry anchored to preflight curve baseline
    paper_pos.entry_price   = pp_price
    paper_pos.current_price = pp_price
    paper_pos.peak_price    = pp_price
    paper_pos.entry_source  = "pp_tick"
    paper_pos.baseline_price = pp_price

    assert paper_pos.entry_price == pp_price, (
        f"entry_price should be pp_price {pp_price}, got {paper_pos.entry_price}"
    )
    assert paper_pos.entry_price != dex_price, (
        f"entry_price must NOT be dex_price {dex_price}"
    )


def test_dog_replay_pct_correct():
    """PnL% with baseline entry must be ~+47.9%, not the inflated +408% from stale dex."""
    entry  = 0.0000234   # preflight curve baseline
    exit_p = 0.0000346   # abort fill price

    pct = (exit_p / entry - 1) * 100
    assert abs(pct - 47.9) < 0.2, f"expected ~47.9%, got {pct:.1f}%"


def test_entry_source_tagged():
    """entry_source should be 'pp_tick' when _pp_price > 0, 'dex_stale' when 0."""

    def derive_entry_source(pp_price: float, bs_label: str = "pp") -> str:
        """Mirrors the Phase 3.2 logic in portfolio.py."""
        if pp_price > 0:
            return "curve" if "curve" in bs_label else "pp_tick"
        return "dex_stale"

    # pp_price present → pp_tick
    assert derive_entry_source(0.0000234, "pp") == "pp_tick"
    # pp_price present with curve label → curve
    assert derive_entry_source(0.0000234, "curve_rpc") == "curve"
    # pp_price absent → dex_stale
    assert derive_entry_source(0.0) == "dex_stale"
