"""Classify markets using structural market characteristics rather than keywords."""

from app.data_client import Market


def classify_market(market: Market) -> tuple[str, str]:
    """Classify a market by price, volume, and liquidity quality."""
    if market.yes_price < 0.02 or market.yes_price > 0.98:
        return ("excluded", "extreme probability")
    if market.volume < 50000:
        return ("excluded", "low volume")
    if market.liquidity_depth < 1000:
        return ("excluded", "low liquidity")

    if (
        market.volume >= 200000
        and market.liquidity_depth >= 10000
        and 0.10 <= market.yes_price <= 0.90
    ):
        return ("core", "high-quality market")

    return ("novelty", "medium-quality or uncertain market")


if __name__ == "__main__":
    sample = Market(
        market_id="test",
        question="Will BTC hit 100k?",
        yes_price=0.6,
        no_price=0.4,
        volume=150000,
        liquidity_depth=5000,
    )
    print(classify_market(sample))
