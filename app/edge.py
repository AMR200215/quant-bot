"""Compute market edge and confidence-adjusted edge for the quant bot."""

import csv
from dataclasses import dataclass
from pathlib import Path

from app.data_client import Market

# ---------------------------------------------------------------------------
# Category accuracy — loaded once from historical_dataset.csv
# ---------------------------------------------------------------------------

_CATEGORY_ACCURACY: dict[str, float] | None = None


def _load_category_accuracy() -> dict[str, float]:
    """Compute per-category direction accuracy from historical_dataset.csv.

    Direction accuracy = fraction of markets where (yes_price > 0.5) correctly
    predicted the YES outcome.  Returns a multiplier relative to the overall
    average, clamped to [0.6, 1.4].  Categories with fewer than 20 samples
    get a neutral multiplier of 1.0.
    """
    global _CATEGORY_ACCURACY
    if _CATEGORY_ACCURACY is not None:
        return _CATEGORY_ACCURACY

    path = Path("data/historical_dataset.csv")
    if not path.exists():
        _CATEGORY_ACCURACY = {}
        return _CATEGORY_ACCURACY

    counts: dict[str, list[int]] = {}  # category -> [correct, total]
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            cat = (row.get("category") or "other").strip() or "other"
            outcome = (row.get("actual_outcome") or "").lower().strip()
            if outcome not in ("yes", "no"):
                continue
            try:
                yes_price = float(row.get("yes_price", 0.5))
            except (TypeError, ValueError):
                continue
            counts.setdefault(cat, [0, 0])
            counts[cat][1] += 1
            if (yes_price > 0.5) == (outcome == "yes"):
                counts[cat][0] += 1

    total_correct = sum(v[0] for v in counts.values())
    total_all = sum(v[1] for v in counts.values())
    overall_acc = total_correct / total_all if total_all > 0 else 0.65

    multipliers: dict[str, float] = {}
    for cat, (correct, total) in counts.items():
        if total < 20:
            multipliers[cat] = 1.0
            continue
        cat_acc = correct / total
        raw_mult = cat_acc / overall_acc
        multipliers[cat] = round(max(0.6, min(1.4, raw_mult)), 4)

    _CATEGORY_ACCURACY = multipliers
    return _CATEGORY_ACCURACY


@dataclass
class EdgeEstimate:
    market_id: str
    posterior: float
    edge_yes: float
    edge_no: float
    confidence: float
    risk_score: float
    risk_multiplier: float
    maturity_score: float
    adjusted_edge_yes: float
    adjusted_edge_no: float
    final_edge_yes: float
    final_edge_no: float
    resolution_quality_score: float
    final_signal_yes: float
    final_signal_no: float
    preferred_side: str
    rationale: str


def compute_confidence(market: Market, logit: float = 0.0) -> float:
    """Estimate confidence from market structure, model conviction, and category accuracy.

    Four components:
      - volume_score:      market trading activity (35%)
      - liquidity_score:   depth available to trade (35%)
      - distance_score:    how far the price is from 50/50 (15%)
      - model_conviction:  magnitude of the logistic regression logit (15%)

    The base score is then scaled by a per-category accuracy multiplier derived
    from historical resolution data (how reliably each category's markets resolve
    in the direction the market prices imply).
    """
    volume_score     = min(market.volume / 500000, 1.0)
    liquidity_score  = min(market.liquidity_depth / 50000, 1.0)
    distance_score   = min(abs(market.yes_price - 0.5) / 0.5, 1.0)
    # logit of ~3 maps to ~95% sigmoid — treat that as maximum conviction
    model_conviction = min(abs(logit) / 3.0, 1.0)

    base = (
        0.35 * volume_score
        + 0.35 * liquidity_score
        + 0.15 * distance_score
        + 0.15 * model_conviction
    )

    cat = (getattr(market, "category", None) or "other").strip() or "other"
    cat_multiplier = _load_category_accuracy().get(cat, 1.0)

    return round(min(max(base * cat_multiplier, 0.0), 1.0), 6)


def compute_risk_score(market: Market) -> float:
    """Estimate structural market risk from liquidity, volume, and uncertainty."""
    liquidity_risk = 1 - min(market.liquidity_depth / 50000, 1.0)
    volume_risk = 1 - min(market.volume / 500000, 1.0)
    uncertainty_risk = 1 - min(abs(market.yes_price - 0.5) / 0.5, 1.0)

    risk_score = (
        0.4 * liquidity_risk + 0.3 * volume_risk + 0.3 * uncertainty_risk
    )
    return min(max(risk_score, 0.0), 1.0)


def compute_maturity_score(market: Market) -> float:
    """Score markets by time to resolution, favoring near-term markets."""
    if market.days_to_resolution is None:
        return 0.50
    if market.days_to_resolution <= 7:
        return 1.00
    if market.days_to_resolution <= 30:
        return 0.75
    if market.days_to_resolution <= 90:
        return 0.50
    return 0.25


def compute_resolution_quality_score(market: Market) -> float:
    """Score how cleanly a market is structured for practical resolution."""
    score = 1.0

    if not market.question:
        return 0.5

    q = market.question.lower()

    if "before" in q:
        score *= 0.5
    if "ever" in q:
        score *= 0.5
    if "return" in q:
        score *= 0.4
    if "gta vi" in q:
        score *= 0.4

    if "this week" in q or "this month" in q:
        score *= 1.1

    return min(max(score, 0.1), 1.0)


def estimate_edge(market: Market, posterior: float, logit: float = 0.0) -> EdgeEstimate:
    """Estimate raw and confidence-adjusted edge for a market."""
    edge_yes = posterior - market.yes_price
    edge_no = (1 - posterior) - market.no_price
    confidence = compute_confidence(market, logit=logit)
    risk_score = compute_risk_score(market)
    if risk_score < 0.30:
        risk_multiplier = 1.0
    elif risk_score < 0.60:
        risk_multiplier = 0.6
    else:
        risk_multiplier = 0.3
    maturity_score = compute_maturity_score(market)
    adjusted_edge_yes = edge_yes * confidence
    adjusted_edge_no = edge_no * confidence
    final_edge_yes = adjusted_edge_yes * maturity_score
    final_edge_no = adjusted_edge_no * maturity_score
    resolution_quality_score = compute_resolution_quality_score(market)
    final_signal_yes = final_edge_yes * resolution_quality_score
    final_signal_no = final_edge_no * resolution_quality_score
    preferred_side = "buy_yes" if final_signal_yes >= final_signal_no else "buy_no"
    rationale = (
        "Edge adjusted by confidence, maturity, and resolution quality "
        "derived from market structure"
    )

    return EdgeEstimate(
        market_id=market.market_id,
        posterior=posterior,
        edge_yes=edge_yes,
        edge_no=edge_no,
        confidence=confidence,
        risk_score=risk_score,
        risk_multiplier=risk_multiplier,
        maturity_score=maturity_score,
        adjusted_edge_yes=adjusted_edge_yes,
        adjusted_edge_no=adjusted_edge_no,
        final_edge_yes=final_edge_yes,
        final_edge_no=final_edge_no,
        resolution_quality_score=resolution_quality_score,
        final_signal_yes=final_signal_yes,
        final_signal_no=final_signal_no,
        preferred_side=preferred_side,
        rationale=rationale,
    )


if __name__ == "__main__":
    sample_market = Market(
        market_id="btc-weekly-90k",
        question="Will BTC close above 90k this week?",
        yes_price=0.58,
        no_price=0.43,
        volume=185000.0,
        liquidity_depth=25000.0,
    )
    posterior = 0.6926
    estimate = estimate_edge(sample_market, posterior)
    main_adjusted_edge = (
        estimate.adjusted_edge_yes
        if estimate.preferred_side == "buy_yes"
        else estimate.adjusted_edge_no
    )

    print("=" * 60)
    print("EDGE ESTIMATE")
    print("=" * 60)
    print(f"Market ID: {estimate.market_id}")
    print()
    print("[Main Signal]")
    print(
        f"Adjusted Edge: {main_adjusted_edge:.4f}   "
        "[weak < 0.02 | moderate 0.02–0.05 | strong > 0.05]"
    )
    print(
        f"Preferred Side: {estimate.preferred_side}         "
        "[buy_yes or buy_no]"
    )
    print(
        f"Confidence: {estimate.confidence:.4f}             "
        "[low < 0.30 | medium 0.30–0.60 | high > 0.60]"
    )
    print()
    print("[Supporting Detail]")
    print(f"Posterior: {estimate.posterior:.4f}")
    print(f"Raw YES Edge: {estimate.edge_yes:.4f}")
    print(f"Raw NO Edge: {estimate.edge_no:.4f}")
    print(f"Adjusted YES Edge: {estimate.adjusted_edge_yes:.4f}")
    print(f"Adjusted NO Edge: {estimate.adjusted_edge_no:.4f}")
    print()
    print(f"Rationale: {estimate.rationale}")
    print("=" * 60)
