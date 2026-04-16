"""Live scan for Polymarket markets across configurable time windows.

Usage:
  python -m app.scan_near_term              # default: 2–5 days
  python -m app.scan_near_term 2 60         # 2 days to 2 months
  python -m app.scan_near_term 1 7 0.005    # 1–7 days, threshold 0.005

Fetches live markets, filters by days-to-resolution window, runs the full
model pipeline, and ranks opportunities by signal strength.
Results are saved to the journal automatically.
"""

import sys

from app.data_client import enrich_with_momentum, fetch_markets_by_days
from app.edge import estimate_edge
from app.external_signals import get_external_consensus
from app.market_journal import append_journal_record
from app.model import estimate_probability
from app.portfolio import get_status, is_halted
from app.state import settings

DEFAULT_MIN_DAYS = 2
DEFAULT_MAX_DAYS = 60
DEFAULT_THRESHOLD = 0.005
FETCH_LIMIT = 100

# Paper trading logs everything above this edge — lower than MIN_EV so we
# collect data even when the model isn't confident enough to signal a real trade
PAPER_TRADE_MIN_EDGE = 0.01


def describe_edge(value: float) -> str:
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


def describe_risk(value: float) -> str:
    if value < 0.30:
        return "low"
    if value < 0.60:
        return "medium"
    return "high"


def main(
    min_days: float = DEFAULT_MIN_DAYS,
    max_days: float = DEFAULT_MAX_DAYS,
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    # --- Circuit breaker ---
    if is_halted(settings.max_drawdown):
        status = get_status()
        print("=" * 60)
        print("CIRCUIT BREAKER ACTIVE — no new signals emitted.")
        print(f"  Drawdown: {status['drawdown']:.1%}  |  Limit: {settings.max_drawdown:.1%}")
        print("=" * 60)
        return

    print("=" * 60)
    print(f"MARKET SCAN  ({min_days:.0f}–{max_days:.0f} days to close)")
    print("=" * 60)
    print(f"Fetching markets closing in {min_days:.0f}–{max_days:.0f} days (no junk)...")

    markets = fetch_markets_by_days(min_days=min_days, max_days=max_days, limit=FETCH_LIMIT)

    # Enrich with 7-day price momentum
    enrich_with_momentum(markets)

    candidates = []

    for market in markets:
        days = market.days_to_resolution
        if days is None:
            continue
        if market.yes_price < 0.05 or market.yes_price > 0.95:
            continue
        if market.volume < 1000:
            continue

        estimate = estimate_probability(market)
        posterior = estimate.posterior

        # Blend with external consensus when available
        external = get_external_consensus(
            market.question,
            twitter_bearer_token=settings.twitter_bearer_token,
        )
        if external.get("consensus_p") is not None:
            posterior = round(0.60 * posterior + 0.40 * external["consensus_p"], 4)
            posterior = max(0.02, min(0.98, posterior))

        edge = estimate_edge(market, posterior)

        if edge.preferred_side == "buy_yes":
            signal_edge = edge.final_signal_yes
            adjusted_edge = edge.adjusted_edge_yes
        else:
            signal_edge = edge.final_signal_no
            adjusted_edge = edge.adjusted_edge_no

        # Paper trade threshold — log everything above 1% edge
        # regardless of whether it passes the strict real-trade filter
        is_real_signal = (
            adjusted_edge >= max(threshold, settings.min_ev)
            and not (
                market.momentum_7d is not None
                and (
                    (edge.preferred_side == "buy_yes" and market.momentum_7d < -0.05)
                    or (edge.preferred_side == "buy_no" and market.momentum_7d > 0.05)
                )
            )
        )

        if adjusted_edge < PAPER_TRADE_MIN_EDGE:
            continue

        candidates.append(
            {
                "market": market,
                "posterior": posterior,
                "estimate": estimate,
                "edge": edge,
                "signal_edge": signal_edge,
                "adjusted_edge": adjusted_edge,
                "days": days,
                "external": external,
                "is_real_signal": is_real_signal,
            }
        )

    candidates.sort(key=lambda x: x["signal_edge"], reverse=True)

    total_scanned = sum(
        1 for m in markets
        if 0.05 <= m.yes_price <= 0.95 and m.volume >= 1000
    )
    real_signals  = [c for c in candidates if c["is_real_signal"]]
    paper_only    = [c for c in candidates if not c["is_real_signal"]]

    print(
        f"Markets in window: {total_scanned}  |  "
        f"Real signals: {len(real_signals)}  |  "
        f"Paper-only: {len(paper_only)}\n"
    )

    if not candidates:
        print("No opportunities found above paper trade threshold (1%).")
        return

    for rank, c in enumerate(candidates[:10], start=1):
        market = c["market"]
        edge   = c["edge"]
        label  = "REAL SIGNAL" if c["is_real_signal"] else "paper trade"

        append_journal_record(
            market_id=market.market_id,
            question=market.question,
            yes_price=market.yes_price,
            posterior=c["posterior"],
            preferred_side=edge.preferred_side,
            adjusted_edge=c["adjusted_edge"],
            final_signal=c["signal_edge"],
            confidence=edge.confidence,
            risk_score=edge.risk_score,
            risk_multiplier=edge.risk_multiplier,
            days_to_resolution=c["days"],
            maturity_score=edge.maturity_score,
            resolution_quality_score=edge.resolution_quality_score,
            notes=f"{label} | scan: {min_days:.0f}-{max_days:.0f}d window",
        )

        print("-" * 60)
        print(f"Rank #{rank}  [{label}]")
        print(f"Market:    {market.question}")
        print(f"ID:        {market.market_id}")
        print(f"Closes in: {c['days']:.1f} days")
        print()
        print(f"Side:      {edge.preferred_side}")
        print(f"Signal:    {c['signal_edge']:.4f}  [{describe_edge(c['adjusted_edge'])}]")
        print(f"Adj Edge:  {c['adjusted_edge']:.4f}")
        print(
            f"Posterior: {c['posterior']:.4f}  "
            f"(market: {market.yes_price:.4f}  |  "
            f"gap: {c['posterior'] - market.yes_price:+.4f})"
        )
        print(f"Rationale: {c['estimate'].rationale}")
        print(
            f"Confidence:{edge.confidence:.4f}  [{describe_confidence(edge.confidence)}]  |  "
            f"Risk: {edge.risk_score:.4f}  [{describe_risk(edge.risk_score)}]"
        )
        print(f"Volume:    ${market.volume:,.0f}  |  Liquidity: ${market.liquidity_depth:,.0f}")
        if market.momentum_7d is not None:
            direction = "up" if market.momentum_7d > 0 else "down"
            print(f"Momentum (7d): {market.momentum_7d:+.4f}  [{direction}]")
        ext = c["external"]
        if ext.get("consensus_p") is not None:
            print(
                f"External:  consensus={ext['consensus_p']:.4f}  "
                f"(Manifold={ext['manifold_p']}  "
                f"Metaculus={ext['metaculus_p']}  "
                f"X={ext.get('x_sentiment')})"
            )
        print()

    print(f"Top {min(len(candidates), 10)} logged to journal ({len(real_signals)} real, {len(paper_only)} paper).")
    print("=" * 60)


if __name__ == "__main__":
    args = sys.argv[1:]
    min_d = float(args[0]) if len(args) > 0 else DEFAULT_MIN_DAYS
    max_d = float(args[1]) if len(args) > 1 else DEFAULT_MAX_DAYS
    thr = float(args[2]) if len(args) > 2 else DEFAULT_THRESHOLD
    main(min_d, max_d, thr)
