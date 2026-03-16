"""Market data structures and mock market loaders for the quant bot."""

from dataclasses import dataclass
from typing import List


@dataclass
class Market:
    """Simple market model for binary prediction markets."""

    market_id: str
    question: str
    yes_price: float
    no_price: float
    volume: float
    liquidity_depth: float


def fetch_mock_markets() -> List[Market]:
    """Return mock markets until real API fetching is added."""
    return [
        Market(
            market_id="btc-weekly-90k",
            question="Will BTC close above 90k this week?",
            yes_price=0.58,
            no_price=0.43,
            volume=185000.0,
            liquidity_depth=25000.0,
        ),
        Market(
            market_id="state-race-y",
            question="Will candidate X win state Y?",
            yes_price=0.41,
            no_price=0.60,
            volume=92000.0,
            liquidity_depth=18000.0,
        ),
        Market(
            market_id="fed-cut-month",
            question="Will the Fed cut rates this month?",
            yes_price=0.34,
            no_price=0.67,
            volume=140000.0,
            liquidity_depth=22000.0,
        ),
    ]


if __name__ == "__main__":
    markets = fetch_mock_markets()
    for market in markets:
        print(market)
