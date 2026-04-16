"""Backtest the quant model against real historical Polymarket data.

Requires data/historical_dataset.csv produced by:
  python -m app.fetch_historical
  python -m app.build_dataset
"""

import csv
from pathlib import Path
from types import SimpleNamespace

from app.edge import estimate_edge
from app.model import estimate_probability
from app.state import settings

DATASET_FILE = Path("data/historical_dataset.csv")

THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05]


def load_dataset(path: Path = DATASET_FILE) -> list[dict]:
    """Load the historical dataset CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run: python -m app.fetch_historical\n"
            "Then: python -m app.build_dataset"
        )
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def row_to_market(row: dict) -> object:
    """Convert a dataset row into a market-like object for the model."""
    yes_price = float(row["yes_price"])
    return SimpleNamespace(
        market_id=row["market_id"],
        question=row["question"],
        yes_price=yes_price,
        no_price=round(1 - yes_price, 4),
        volume=float(row["volume"]),
        liquidity_depth=float(row["liquidity_depth"]),
        days_to_resolution=None,
    )


def run_backtest(rows: list[dict]) -> None:
    """Run threshold sweep over the historical dataset."""
    total = len(rows)
    print(f"Dataset size: {total} markets")
    print()

    for threshold in THRESHOLDS:
        bankroll = float(settings.bankroll)
        trades = 0
        wins = 0
        losses = 0
        skipped = 0

        for row in rows:
            market = row_to_market(row)
            actual_outcome = row["actual_outcome"]

            if market.volume < settings.min_volume:
                skipped += 1
                continue

            estimate = estimate_probability(market)
            posterior = estimate.posterior
            edge = estimate_edge(market, posterior)

            if edge.preferred_side == "buy_yes":
                signal_edge = edge.adjusted_edge_yes
                price = market.yes_price
                bot_wins = actual_outcome == "yes"
            else:
                signal_edge = edge.adjusted_edge_no
                price = market.no_price
                bot_wins = actual_outcome == "no"

            if signal_edge < threshold:
                skipped += 1
                continue

            if 0.45 < posterior < 0.55:
                skipped += 1
                continue

            size = 100 * edge.risk_multiplier

            if bot_wins:
                pnl = size * ((1 / price) - 1)
                wins += 1
            else:
                pnl = -size
                losses += 1

            bankroll += pnl
            trades += 1

        win_rate = wins / trades if trades else 0.0
        roi = (bankroll - float(settings.bankroll)) / float(settings.bankroll)

        print("=" * 60)
        print(f"THRESHOLD: {threshold:.2f}  |  Trades: {trades}  |  Skipped: {skipped}")
        print("=" * 60)
        print(f"Win rate:       {win_rate:.1%}  ({wins}W / {losses}L)")
        print(f"Final bankroll: ${bankroll:,.2f}")
        print(f"ROI:            {roi:+.1%}")
        print()


def main() -> None:
    print("=" * 60)
    print("HISTORICAL BACKTEST")
    print("=" * 60)
    print()

    rows = load_dataset()
    run_backtest(rows)


if __name__ == "__main__":
    main()
