"""Profit-focused analytics on the historical backtest dataset.

Runs the model on data/historical_dataset.csv and produces:
- Overall accuracy and calibration
- Win rate and ROI by edge threshold (optimal cut finder)
- Category performance breakdown
- YES vs NO side bias
- Sharpe ratio and max drawdown simulation
- Recommendation: best threshold to use live
"""

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

from app.edge import estimate_edge
from app.model import estimate_probability
from app.state import settings

DATASET_FILE = Path("data/historical_dataset.csv")
CALIBRATION_FILE = Path("data/calibration.json")
THRESHOLDS = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
STARTING_BANKROLL = float(settings.bankroll)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: Path = DATASET_FILE) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run: python -m app.fetch_historical\n"
            "Then: python -m app.build_dataset"
        )
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def row_to_market(row: dict) -> object:
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


def score_row(row: dict) -> dict:
    """Run the full model pipeline on a dataset row and return a scored record."""
    market = row_to_market(row)
    estimate = estimate_probability(market)
    posterior = estimate.posterior
    edge = estimate_edge(market, posterior)

    if edge.preferred_side == "buy_yes":
        signal_edge = edge.adjusted_edge_yes
        trade_price = market.yes_price
        bot_wins = row["actual_outcome"] == "yes"
    else:
        signal_edge = edge.adjusted_edge_no
        trade_price = market.no_price
        bot_wins = row["actual_outcome"] == "no"

    return {
        "market_id": row["market_id"],
        "question": row["question"],
        "category": row.get("category", ""),
        "actual_outcome": row["actual_outcome"],
        "yes_price": market.yes_price,
        "posterior": posterior,
        "preferred_side": edge.preferred_side,
        "signal_edge": signal_edge,
        "confidence": edge.confidence,
        "risk_score": edge.risk_score,
        "risk_multiplier": edge.risk_multiplier,
        "trade_price": trade_price,
        "bot_wins": bot_wins,
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibration_analysis(scored: list[dict]) -> None:
    """Check if the model's posterior matches actual outcome rates."""
    buckets: dict[str, list[bool]] = {}
    for record in scored:
        p = record["posterior"]
        # Bucket by the predicted probability for the preferred side
        if record["preferred_side"] == "buy_yes":
            pred_p = p
            actual_win = record["actual_outcome"] == "yes"
        else:
            pred_p = 1 - p
            actual_win = record["actual_outcome"] == "no"

        label = f"{int(pred_p * 10) * 10}-{int(pred_p * 10) * 10 + 10}%"
        buckets.setdefault(label, []).append(actual_win)

    print("[Calibration — Predicted vs Actual Win Rate]")
    print(f"{'Predicted':>14}  {'Actual':>8}  {'Count':>6}  {'Gap':>8}")
    print("-" * 44)

    ordered = sorted(buckets.keys(), key=lambda x: int(x.split("-")[0]))
    total_gap = 0.0
    for label in ordered:
        wins = buckets[label]
        pred_mid = (int(label.split("-")[0]) + 5) / 100
        actual_rate = sum(wins) / len(wins) if wins else 0
        gap = actual_rate - pred_mid
        total_gap += abs(gap)
        flag = " <-- overconfident" if gap < -0.10 else (" <-- underconfident" if gap > 0.10 else "")
        print(
            f"{label:>14}  {actual_rate:>7.1%}  {len(wins):>6}  {gap:>+7.1%}{flag}"
        )

    avg_gap = total_gap / len(buckets) if buckets else 0
    print(f"\nMean absolute calibration error: {avg_gap:.1%}")
    print()


def save_calibration_json(scored: list[dict]) -> None:
    """Persist calibration bucket data to data/calibration.json.

    The web UI reads this file to render a reliability diagram without
    re-running the full analytics pipeline.
    """
    buckets: dict[str, list[bool]] = {}
    for record in scored:
        p = record["posterior"]
        if record["preferred_side"] == "buy_yes":
            pred_p = p
            actual_win = record["actual_outcome"] == "yes"
        else:
            pred_p = 1 - p
            actual_win = record["actual_outcome"] == "no"

        label = f"{int(pred_p * 10) * 10}-{int(pred_p * 10) * 10 + 10}%"
        buckets.setdefault(label, []).append(actual_win)

    rows = []
    for label in sorted(buckets.keys(), key=lambda x: int(x.split("-")[0])):
        wins = buckets[label]
        pred_mid = (int(label.split("-")[0]) + 5) / 100
        actual_rate = sum(wins) / len(wins) if wins else 0
        rows.append(
            {
                "bucket": label,
                "predicted": round(pred_mid, 3),
                "actual": round(actual_rate, 4),
                "count": len(wins),
                "gap": round(actual_rate - pred_mid, 4),
            }
        )

    mean_ace = sum(abs(r["gap"]) for r in rows) / len(rows) if rows else 0
    output = {
        "buckets": rows,
        "mean_absolute_calibration_error": round(mean_ace, 4),
        "n_markets": len(scored),
    }

    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CALIBRATION_FILE.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"Calibration data saved → {CALIBRATION_FILE}")


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def threshold_sweep(scored: list[dict]) -> float:
    """Find the optimal edge threshold by ROI and win rate."""
    print("[Threshold Sweep — Finding Optimal Cut]")
    print(
        f"{'Threshold':>10}  {'Trades':>7}  {'Win%':>7}  {'ROI':>8}  {'Sharpe':>8}"
    )
    print("-" * 50)

    best_threshold = THRESHOLDS[0]
    best_roi = float("-inf")

    for threshold in THRESHOLDS:
        bankroll = STARTING_BANKROLL
        peak = STARTING_BANKROLL
        trade_pnls: list[float] = []
        wins = 0

        for record in scored:
            if record["signal_edge"] < threshold:
                continue
            if not (0.05 <= record["yes_price"] <= 0.95):
                continue

            size = 100 * record["risk_multiplier"]
            price = record["trade_price"]

            if record["bot_wins"]:
                pnl = size * ((1 / price) - 1)
                wins += 1
            else:
                pnl = -size

            bankroll += pnl
            peak = max(peak, bankroll)
            trade_pnls.append(pnl)

        n = len(trade_pnls)
        if n == 0:
            print(f"{threshold:>10.3f}  {'—':>7}  {'—':>7}  {'—':>8}  {'—':>8}")
            continue

        win_rate = wins / n
        roi = (bankroll - STARTING_BANKROLL) / STARTING_BANKROLL

        avg_pnl = sum(trade_pnls) / n
        std_pnl = math.sqrt(sum((p - avg_pnl) ** 2 for p in trade_pnls) / n) if n > 1 else 0
        sharpe = (avg_pnl / std_pnl) if std_pnl > 0 else 0

        flag = " <-- best" if roi > best_roi and n >= 5 else ""
        if roi > best_roi and n >= 5:
            best_roi = roi
            best_threshold = threshold

        print(
            f"{threshold:>10.3f}  {n:>7}  {win_rate:>7.1%}  {roi:>+8.1%}"
            f"  {sharpe:>8.2f}{flag}"
        )

    print()
    return best_threshold


# ---------------------------------------------------------------------------
# Category breakdown
# ---------------------------------------------------------------------------

def category_breakdown(scored: list[dict]) -> None:
    """Win rate and trade count by market category."""
    cats: dict[str, dict] = {}

    for record in scored:
        cat = record["category"].strip() or "unknown"
        if cat not in cats:
            cats[cat] = {"trades": 0, "wins": 0}
        cats[cat]["trades"] += 1
        if record["bot_wins"]:
            cats[cat]["wins"] += 1

    rows = sorted(cats.items(), key=lambda x: x[1]["trades"], reverse=True)

    print("[Category Breakdown]")
    print(f"{'Category':>30}  {'Trades':>7}  {'Win%':>7}")
    print("-" * 48)
    for cat, data in rows[:15]:
        win_rate = data["wins"] / data["trades"] if data["trades"] else 0
        print(f"{cat[:30]:>30}  {data['trades']:>7}  {win_rate:>7.1%}")
    print()


# ---------------------------------------------------------------------------
# Side bias
# ---------------------------------------------------------------------------

def side_bias(scored: list[dict]) -> None:
    """Check if the model systematically favors YES or NO incorrectly."""
    yes_trades = [r for r in scored if r["preferred_side"] == "buy_yes"]
    no_trades = [r for r in scored if r["preferred_side"] == "buy_no"]

    def _stats(records: list[dict]) -> tuple[int, float]:
        if not records:
            return 0, 0.0
        wins = sum(1 for r in records if r["bot_wins"])
        return len(records), wins / len(records)

    yes_n, yes_wr = _stats(yes_trades)
    no_n, no_wr = _stats(no_trades)

    print("[Side Bias — YES vs NO]")
    print(f"  buy_yes: {yes_n:>4} trades  win rate: {yes_wr:.1%}")
    print(f"  buy_no:  {no_n:>4} trades  win rate: {no_wr:.1%}")

    if yes_wr > no_wr + 0.10:
        print("  Insight: Model performs better on YES side — trust YES signals more.")
    elif no_wr > yes_wr + 0.10:
        print("  Insight: Model performs better on NO side — trust NO signals more.")
    else:
        print("  Insight: No strong side bias detected.")
    print()


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------

def drawdown_analysis(scored: list[dict], threshold: float) -> None:
    """Simulate bankroll path and compute max drawdown at the given threshold."""
    bankroll = STARTING_BANKROLL
    peak = STARTING_BANKROLL
    max_dd = 0.0
    path: list[float] = [bankroll]

    for record in scored:
        if record["signal_edge"] < threshold:
            continue
        if not (0.05 <= record["yes_price"] <= 0.95):
            continue

        size = 100 * record["risk_multiplier"]
        price = record["trade_price"]

        pnl = size * ((1 / price) - 1) if record["bot_wins"] else -size
        bankroll += pnl
        peak = max(peak, bankroll)
        dd = (peak - bankroll) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
        path.append(bankroll)

    print(f"[Drawdown Analysis — threshold={threshold:.3f}]")
    print(f"  Starting bankroll: ${STARTING_BANKROLL:,.2f}")
    print(f"  Final bankroll:    ${bankroll:,.2f}")
    print(f"  Max drawdown:      {max_dd:.1%}")
    print(f"  Trades simulated:  {len(path) - 1}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("PROFIT ANALYTICS")
    print("=" * 60)
    print()

    rows = load_dataset()
    print(f"Dataset: {len(rows)} historical markets\n")

    scored = [score_row(row) for row in rows]

    calibration_analysis(scored)
    save_calibration_json(scored)
    best_threshold = threshold_sweep(scored)
    side_bias(scored)
    category_breakdown(scored)
    drawdown_analysis(scored, best_threshold)

    print("=" * 60)
    print(f"RECOMMENDED THRESHOLD: {best_threshold:.3f}")
    print("Use this in the live scan for highest historical ROI.")
    print("=" * 60)


if __name__ == "__main__":
    main()
