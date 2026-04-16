import csv
from datetime import datetime, UTC
from pathlib import Path

from app.data_client import enrich_with_momentum, get_market_snapshot
from app.edge import estimate_edge
from app.external_signals import get_external_consensus
from app.model import estimate_probability
from app.portfolio import get_status, is_halted
from app.state import settings

LOG_FILE = Path("logs/live_scan_results.csv")


def describe_adjusted_edge(value: float) -> str:
    if value < 0.02:
        return "weak"
    if value < 0.05:
        return "moderate"
    return "strong"


def describe_confidence(value: float) -> str:
    if value < 0.30:
        return "low"
    if value < 0.60:
        return "medium"
    return "high"


def describe_risk_score(value: float) -> str:
    if value < 0.30:
        return "low"
    if value < 0.60:
        return "medium"
    return "high"


def log_live_rows(rows: list[list]) -> None:
    """Append live scan rows to the CSV log."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_FILE.exists()

    with LOG_FILE.open("a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "market_id",
                    "question",
                    "yes_price",
                    "posterior",
                    "preferred_side",
                    "adjusted_edge",
                    "confidence",
                    "risk_score",
                    "risk_multiplier",
                    "volume",
                    "liquidity_depth",
                ]
            )
        writer.writerows(rows)


def main() -> None:
    # --- Circuit breaker ---
    if is_halted(settings.max_drawdown):
        status = get_status()
        print("=" * 60)
        print("CIRCUIT BREAKER ACTIVE — no new signals emitted.")
        print(f"  Drawdown: {status['drawdown']:.1%}  |  "
              f"Limit: {settings.max_drawdown:.1%}")
        print(f"  Current equity: ${status['current_equity']}  |  "
              f"Peak: ${status['peak_equity']}")
        print("  Run `from app.portfolio import reset_halt; reset_halt()` to clear.")
        print("=" * 60)
        return

    snapshot = get_market_snapshot(settings.data_source)

    # Enrich with 7-day momentum (adds ~80 ms/market via CLOB API)
    if snapshot.source == "live":
        enrich_with_momentum(snapshot.markets)

    rows: list[list] = []
    records: list[dict] = []

    for market in snapshot.markets:
        if market.yes_price < 0.02 or market.yes_price > 0.98:
            continue
        if market.volume < 50000:
            continue
        if market.liquidity_depth < 1000:
            continue
        if market.days_to_resolution is not None and market.days_to_resolution <= 0:
            continue

        estimate = estimate_probability(market)
        posterior = estimate.posterior

        # Blend with external consensus (Manifold + Metaculus + X) when enabled
        external = {}
        if settings.use_external_signals and snapshot.source == "live":
            external = get_external_consensus(
                market.question,
                twitter_bearer_token=settings.twitter_bearer_token,
            )
            if external.get("consensus_p") is not None:
                # 60% trained model, 40% external consensus
                posterior = round(
                    0.60 * posterior + 0.40 * external["consensus_p"], 4
                )
                posterior = max(0.02, min(0.98, posterior))

        edge = estimate_edge(market, posterior)

        if edge.preferred_side == "buy_yes":
            adjusted_edge = edge.adjusted_edge_yes
            final_edge = edge.final_edge_yes
            signal_value = edge.final_signal_yes
        else:
            adjusted_edge = edge.adjusted_edge_no
            final_edge = edge.final_edge_no
            signal_value = edge.final_signal_no

        # Skip signals that don't clear the fee-adjusted minimum edge
        if adjusted_edge < settings.min_ev:
            continue

        # Momentum filter: if momentum contradicts the signal, skip
        if market.momentum_7d is not None:
            if edge.preferred_side == "buy_yes" and market.momentum_7d < -0.05:
                continue  # price falling, don't fight the tape
            if edge.preferred_side == "buy_no" and market.momentum_7d > 0.05:
                continue  # price rising, don't fight the tape

        timestamp = datetime.now(UTC).isoformat()
        record = {
            "timestamp": timestamp,
            "market_id": market.market_id,
            "question": market.question,
            "yes_price": market.yes_price,
            "days_to_resolution": market.days_to_resolution,
            "posterior": posterior,
            "preferred_side": edge.preferred_side,
            "adjusted_edge": adjusted_edge,
            "final_edge": final_edge,
            "resolution_quality_score": edge.resolution_quality_score,
            "final_signal": signal_value,
            "confidence": edge.confidence,
            "risk_score": edge.risk_score,
            "risk_multiplier": edge.risk_multiplier,
            "maturity_score": edge.maturity_score,
            "volume": market.volume,
            "liquidity_depth": market.liquidity_depth,
            "momentum_7d": market.momentum_7d,
            "manifold_p": external.get("manifold_p"),
            "metaculus_p": external.get("metaculus_p"),
            "x_sentiment": external.get("x_sentiment"),
            "consensus_p": external.get("consensus_p"),
        }
        records.append(record)
        rows.append(
            [
                timestamp,
                market.market_id,
                market.question,
                market.yes_price,
                posterior,
                edge.preferred_side,
                adjusted_edge,
                edge.confidence,
                edge.risk_score,
                edge.risk_multiplier,
                market.volume,
                market.liquidity_depth,
            ]
        )

    records.sort(key=lambda item: item["final_signal"], reverse=True)
    log_live_rows(rows)

    print("=" * 60)
    print("LIVE SCAN RESULTS")
    print("=" * 60)
    print(f"Data source: {snapshot.source}")
    print(f"Markets scored: {len(records)}")
    print("=" * 60)
    print()

    for rank, record in enumerate(records[:5], start=1):
        print("-" * 60)
        print(f"Rank: {rank}")
        print(f"Market: {record['question']}")
        print(f"ID: {record['market_id']}")
        print()
        print(f"Posterior: {record['posterior']:.4f}")
        print(f"Side: {record['preferred_side']}")
        print(
            f"Adjusted Edge: {record['adjusted_edge']:.4f} "
            f"[{describe_adjusted_edge(record['adjusted_edge'])}; weak < 0.02 | moderate 0.02–0.05 | strong > 0.05]"
        )
        if record["days_to_resolution"] is None:
            print("Days to Resolution: unknown")
        else:
            print(f"Days to Resolution: {record['days_to_resolution']:.2f}")
        print(
            f"Maturity Score: {record['maturity_score']:.2f} "
            "[fast 1.00 | medium 0.75–0.50 | slow 0.25]"
        )
        print(f"Final Edge: {record['final_edge']:.4f}")
        print(
            f"Resolution Quality Score: "
            f"{record['resolution_quality_score']:.2f}"
        )
        print(f"Final Signal: {record['final_signal']:.4f}")
        print(
            f"Confidence: {record['confidence']:.4f} "
            f"[{describe_confidence(record['confidence'])}; low < 0.30 | medium 0.30–0.60 | high > 0.60]"
        )
        print(
            f"Risk Score: {record['risk_score']:.4f} "
            f"[{describe_risk_score(record['risk_score'])}; low < 0.30 | medium 0.30–0.60 | high > 0.60]"
        )
        print(f"Risk Multiplier: {record['risk_multiplier']:.2f}")
        print(f"Volume: {record['volume']:.2f}")
        print(f"Liquidity: {record['liquidity_depth']:.2f}")
        if record["momentum_7d"] is not None:
            direction = "up" if record["momentum_7d"] > 0 else "down"
            print(f"Momentum (7d): {record['momentum_7d']:+.4f}  [{direction}]")
        if record["consensus_p"] is not None:
            print(
                f"External Consensus: {record['consensus_p']:.4f}  "
                f"(Manifold={record['manifold_p']}  "
                f"Metaculus={record['metaculus_p']}  "
                f"X={record.get('x_sentiment')})"
            )
        print()


if __name__ == "__main__":
    main()
