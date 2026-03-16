"""Risk management helpers for sizing and drawdown control."""


def kelly_fraction(p: float, price: float) -> float:
    """Compute the Kelly fraction for a binary contract."""
    if price <= 0 or price >= 1:
        return 0.0

    odds = (1 / price) - 1
    f = (p * odds - (1 - p)) / odds
    return max(0.0, f)


def fractional_kelly_size(
    bankroll: float,
    p: float,
    price: float,
    fraction: float = 0.25,
    max_risk: float = 0.02,
) -> float:
    """Convert a Kelly fraction into a capped dollar position size."""
    raw_kelly = kelly_fraction(p, price)
    scaled_fraction = raw_kelly * fraction
    capped_fraction = min(scaled_fraction, max_risk)
    return bankroll * capped_fraction


def impact_estimate(order_size_dollars: float, liquidity_depth: float) -> float:
    """Estimate market impact as a share of visible liquidity depth."""
    if liquidity_depth <= 0:
        return 1.0
    return order_size_dollars / liquidity_depth


def drawdown(current_equity: float, peak_equity: float) -> float:
    """Compute drawdown from the equity peak."""
    if peak_equity <= 0:
        return 0.0
    return 1 - (current_equity / peak_equity)


if __name__ == "__main__":
    bankroll = 5000
    p = 0.67
    price = 0.55

    raw_kelly = kelly_fraction(p, price)
    size = fractional_kelly_size(bankroll, p, price)
    impact = impact_estimate(size, liquidity_depth=4000)
    dd = drawdown(current_equity=4500, peak_equity=5000)

    print("Kelly fraction:", round(raw_kelly, 4))
    print("Position size:", round(size, 2))
    print("Impact estimate:", round(impact, 4))
    print("Drawdown:", round(dd, 4))
