import re
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from app.data_client import fetch_live_markets, get_market_snapshot
from app.edge import estimate_edge
from app.external_signals import get_external_consensus
from app.market_journal import append_journal_record
from app.model import estimate_probability
from app.state import settings

# Signals above this are in the historically best-performing band (1-3% adj edge).
# Signals above 5% have shown inverse accuracy — model is fighting the market.
MIN_ACTIONABLE_EDGE = 0.01
MAX_RELIABLE_EDGE   = 0.05

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"

STOPWORDS = {
    "the",
    "a",
    "an",
    "will",
    "when",
    "what",
    "where",
    "how",
    "is",
    "are",
    "to",
    "for",
    "of",
    "in",
    "on",
    "by",
    "and",
    "or",
    "out",
    "as",
    "end",
    "this",
    "that",
    "be",
    "again",
}


def normalize_words(text: str) -> Set[str]:
    """Normalize text into a filtered word set."""
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return {
        word for word in normalized.split() if word and word not in STOPWORDS
    }


def extract_input_type(query: str) -> Tuple[str, str]:
    """Classify the query as an event slug, market id, or plain text."""
    normalized = query.strip().rstrip("/")

    if "polymarket.com/event/" in normalized:
        slug = normalized.split("/event/", 1)[1]
        slug = slug.split("?", 1)[0].split("#", 1)[0].strip("/")
        return "event_slug", slug

    if normalized.isdigit():
        return "market_id", normalized

    return "plain_text", query


def fetch_event_by_slug(slug: str) -> Optional[Dict[str, Any]]:
    """Fetch a Polymarket event by slug."""
    try:
        response = requests.get(
            f"{GAMMA_BASE_URL}/events/slug/{slug}",
            timeout=20,
        )
    except requests.RequestException:
        return None

    if response.status_code == 200:
        return response.json()
    return None


def extract_markets_from_event(event_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract market payloads from an event response."""
    markets = event_payload.get("markets")
    if isinstance(markets, list):
        return markets
    return []


def find_market_by_id(market_id: str, markets: list) -> Optional[object]:
    """Find a market by exact market id."""
    for market in markets:
        if market.market_id == market_id:
            return market
    return None


def search_local_candidates(query: str, limit: int = 200) -> List[Dict[str, Any]]:
    """Search a larger local live market universe with overlap scoring."""
    markets = fetch_live_markets(limit=limit)
    query_words = normalize_words(query)
    candidates: List[Dict[str, Any]] = []

    for market in markets:
        market_words = normalize_words(market.question)
        overlap = len(query_words & market_words)
        if overlap <= 0:
            continue

        coverage = overlap / max(len(query_words), 1)
        candidates.append(
            {
                "market": market,
                "overlap": overlap,
                "coverage": coverage,
                "volume": market.volume,
            }
        )

    candidates.sort(
        key=lambda item: (item["overlap"], item["coverage"], item["volume"]),
        reverse=True,
    )
    return candidates


def safe_float(value: Any, default: float) -> float:
    """Convert a value to float when possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def payload_to_market(payload: Dict[str, Any]) -> object:
    """Convert a direct event market payload into a market-like object."""
    yes_price = safe_float(payload.get("lastTradePrice"), float("nan"))
    if not 0 < yes_price < 1:
        yes_price = safe_float(payload.get("price"), 0.50)

    volume = safe_float(payload.get("volume"), 0.0)
    liquidity_depth = safe_float(
        payload.get("liquidity"), max(volume * 0.1, 1000.0)
    )

    return SimpleNamespace(
        market_id=str(payload.get("id", "")),
        question=payload.get("question") or payload.get("title") or "Unknown market",
        yes_price=yes_price,
        no_price=round(1 - yes_price, 4),
        volume=volume,
        liquidity_depth=liquidity_depth,
        days_to_resolution=None,
    )


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


def format_days_to_resolution(value: Optional[float]) -> str:
    """Format time to resolution for terminal output."""
    if value is None:
        return "unknown"
    return f"{value:.2f}"


def analyze_market(market: object) -> None:
    """Run the existing analysis output for a selected market."""
    estimate = estimate_probability(market)
    posterior = estimate.posterior

    # Blend with external consensus when available (Manifold / Metaculus)
    external = get_external_consensus(
        market.question,
        twitter_bearer_token=settings.twitter_bearer_token,
    )
    external_sources = external.get("sources", 0)
    if external.get("consensus_p") is not None:
        weight = 0.20 if external_sources == 1 else 0.35
        posterior = round(weight * external["consensus_p"] + (1 - weight) * posterior, 4)
        posterior = max(0.02, min(0.98, posterior))

    edge = estimate_edge(market, posterior, logit=estimate.logit)

    if edge.preferred_side == "buy_yes":
        signal_edge = edge.final_signal_yes
        adjusted_edge = edge.adjusted_edge_yes
    else:
        signal_edge = edge.final_signal_no
        adjusted_edge = edge.adjusted_edge_no

    append_journal_record(
        market_id=market.market_id,
        question=market.question,
        yes_price=market.yes_price,
        posterior=posterior,
        preferred_side=edge.preferred_side,
        adjusted_edge=adjusted_edge,
        final_signal=signal_edge,
        confidence=edge.confidence,
        risk_score=edge.risk_score,
        risk_multiplier=edge.risk_multiplier,
        days_to_resolution=market.days_to_resolution,
        maturity_score=edge.maturity_score,
        resolution_quality_score=edge.resolution_quality_score,
        notes="logged from analyze_market",
    )

    print("=" * 60)
    print("MARKET ANALYSIS")
    print("=" * 60)
    print(f"Market: {market.question}")
    print(f"ID: {market.market_id}")
    print()
    print("[Model]")
    print(f"Rationale: {estimate.rationale}")
    print(f"Posterior: {posterior:.4f}  (market: {market.yes_price:.4f}  gap: {posterior - market.yes_price:+.4f})")
    if external.get("consensus_p") is not None:
        print(
            f"External:  consensus={external['consensus_p']:.4f}  "
            f"(Manifold={external['manifold_p']}  Metaculus={external['metaculus_p']})"
        )
    print()
    print("[Signal]")
    print(f"Side: {edge.preferred_side}")
    print(
        f"Adjusted Edge: {signal_edge:.4f} "
        f"[{describe_adjusted_edge(signal_edge)}; weak < 0.02 | moderate 0.02–0.05 | strong > 0.05]"
    )
    if adjusted_edge < MIN_ACTIONABLE_EDGE:
        print(f"  !! Edge below 1% — likely noise.")
    elif adjusted_edge > MAX_RELIABLE_EDGE:
        print(
            f"  !! Edge above 5% — model is strongly disagreeing with the market; "
            "historically these have lower accuracy than the 1-3% band."
        )
    print(
        f"Confidence: {edge.confidence:.4f} "
        f"[{describe_confidence(edge.confidence)}; low < 0.30 | medium 0.30–0.60 | high > 0.60]"
    )
    print(
        f"Risk Score: {edge.risk_score:.4f} "
        f"[{describe_risk_score(edge.risk_score)}; low < 0.30 | medium 0.30–0.60 | high > 0.60]"
    )
    print(f"Risk Multiplier: {edge.risk_multiplier:.2f}")
    print()
    print("[Timing]")
    print(f"Days to Resolution: {format_days_to_resolution(market.days_to_resolution)}")
    print(f"Maturity Score: {edge.maturity_score:.2f}")
    print()
    print("[Quality]")
    print(f"Resolution Quality Score: {edge.resolution_quality_score:.2f}")
    print()
    print("[Final]")
    print(f"Final Signal: {signal_edge:.4f}")
    print(f"Volume: {market.volume:.2f}")
    print(f"Liquidity: {market.liquidity_depth:.2f}")
    print()
    print("Journal: saved")
    print("=" * 60)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 -m app.analyze_market <market id or question>")
        return

    query = " ".join(sys.argv[1:])
    input_type, extracted_value = extract_input_type(query)

    if input_type == "event_slug":
        event_payload = fetch_event_by_slug(extracted_value)
        if not event_payload:
            print("No event found for that Polymarket URL/slug.")
            return

        event_markets = extract_markets_from_event(event_payload)
        if not event_markets:
            print("Event found, but no markets were returned.")
            return

        if len(event_markets) == 1:
            analyze_market(payload_to_market(event_markets[0]))
            return

        print("This event contains multiple markets/options:")
        for candidate in event_markets:
            question = candidate.get("question") or candidate.get("title") or "Unknown market"
            print(f"- {candidate.get('id', '')} | {question}")
        print("Please rerun using an exact market ID or exact market question.")
        return

    if input_type == "market_id":
        markets = fetch_live_markets(limit=200)
        market = find_market_by_id(extracted_value, markets)
        if market is None:
            print("No market found for that market ID.")
            return
        analyze_market(market)
        return

    markets = fetch_live_markets(limit=200)
    if not markets:
        snapshot = get_market_snapshot("live")
        markets = snapshot.markets
    normalized_query = query.lower().strip()

    for market in markets:
        if market.question.lower() == normalized_query:
            analyze_market(market)
            return

    for market in markets:
        if normalized_query in market.question.lower():
            analyze_market(market)
            return

    candidates = search_local_candidates(query, limit=200)
    if not candidates:
        print("No strong match found. Closest matches:")
        return

    best_candidate = candidates[0]
    if (
        best_candidate["overlap"] >= 2
        and best_candidate["coverage"] >= 0.4
    ):
        print("Using closest match:")
        analyze_market(best_candidate["market"])
        return

    print("No strong match found. Closest matches:")
    for candidate in candidates[:5]:
        market = candidate["market"]
        print(
            f"- {market.market_id} | {market.question} | "
            f"overlap={candidate['overlap']} | coverage={candidate['coverage']:.2f}"
        )


if __name__ == "__main__":
    main()
