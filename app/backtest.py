"""Simple backtesting helpers for the quant bot."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

from app.bayes import bayes_update
from app.risk import fractional_kelly_size
from app.signals import choose_side

LOG_FILE = Path("logs/backtest_results.csv")


@dataclass
class BacktestRow:
    """Minimal historical market row used for mock backtests."""

    market_id: str
    question: str
    yes_price: float
    no_price: float
    outcome: str


def load_mock_backtest_data() -> List[BacktestRow]:
    """Return simple mock rows for a first-pass backtest."""
    return [
        BacktestRow(
            market_id="btc-1",
            question="Will BTC close above 90k this week?",
            yes_price=0.58,
            no_price=0.43,
            outcome="yes",
        ),
        BacktestRow(
            market_id="fed-1",
            question="Will the Fed cut rates this month?",
            yes_price=0.34,
            no_price=0.67,
            outcome="no",
        ),
        BacktestRow(
            market_id="election-1",
            question="Will candidate X win state Y?",
            yes_price=0.41,
            no_price=0.60,
            outcome="no",
        ),
        BacktestRow(
            market_id="oil-1",
            question="Will oil close above 85 this Friday?",
            yes_price=0.49,
            no_price=0.52,
            outcome="yes",
        ),
        BacktestRow(
            market_id="jobs-1",
            question="Will payrolls beat expectations this month?",
            yes_price=0.62,
            no_price=0.39,
            outcome="no",
        ),
    ]


def log_trade(row: list) -> None:
    """Append an executed trade row to the backtest CSV log."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_FILE.exists()

    with LOG_FILE.open("a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(
                [
                    "market_id",
                    "question",
                    "side",
                    "posterior",
                    "size",
                    "pnl",
                    "bankroll_after",
                    "outcome",
                ]
            )
        writer.writerow(row)


def run_backtest(bankroll: float = 5000.0) -> None:
    """Run a minimal mock backtest over static rows."""
    rows = load_mock_backtest_data()
    current_bankroll = bankroll
    trades = 0
    wins = 0
    losses = 0

    for row in rows:
        prior = row.yes_price
        likelihood_true = 0.65
        likelihood_false = 0.35
        posterior = bayes_update(prior, likelihood_true, likelihood_false)
        signal = choose_side(
            model_p=posterior,
            yes_price=row.yes_price,
            no_price=row.no_price,
            fee_buffer=0.01,
        )

        if signal["side"] == "hold":
            continue

        if signal["side"] == "buy_yes":
            size = fractional_kelly_size(
                bankroll=current_bankroll,
                p=posterior,
                price=row.yes_price,
                fraction=0.25,
            )
            if row.outcome == "yes":
                pnl = size * ((1 / row.yes_price) - 1)
                wins += 1
            else:
                pnl = -size
                losses += 1
        else:
            size = fractional_kelly_size(
                bankroll=current_bankroll,
                p=1 - posterior,
                price=row.no_price,
                fraction=0.25,
            )
            if row.outcome == "no":
                pnl = size * ((1 / row.no_price) - 1)
                wins += 1
            else:
                pnl = -size
                losses += 1

        current_bankroll += pnl
        trades += 1

        log_trade(
            [
                row.market_id,
                row.question,
                signal["side"],
                round(posterior, 4),
                round(size, 2),
                round(pnl, 2),
                round(current_bankroll, 2),
                row.outcome,
            ]
        )

        print(
            f"Question: {row.question} | "
            f"Side: {signal['side']} | "
            f"Size: {size:.2f} | "
            f"PnL: {pnl:.2f} | "
            f"Bankroll: {current_bankroll:.2f}"
        )

    print(f"Starting bankroll: {bankroll:.2f}")
    print(f"Final bankroll: {current_bankroll:.2f}")
    print(f"Total trades: {trades}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")


if __name__ == "__main__":
    run_backtest()
