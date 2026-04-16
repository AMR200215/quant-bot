import re
import sys
from typing import Any, Dict, List, Set, Tuple

import requests

from app.data_client import fetch_live_markets

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


def normalize_candidate(
    payload: Dict[str, Any], source: str
) -> Dict[str, Any]:
    """Normalize API payloads into a consistent candidate shape."""
    return {
        "kind": "api",
        "market_id": str(payload.get("id")) if payload.get("id") is not None else None,
        "question": payload.get("question") or payload.get("title") or payload.get("name") or "Unknown market",
        "source": source,
    }


def is_usable_public_candidate(candidate: Dict[str, Any], query: str) -> bool:
    """Filter noisy public-search candidates."""
    question = (candidate.get("question") or "").strip()
    if not question:
        return False

    query_words = normalize_words(query)
    question_words = normalize_words(question)
    if query_words and not (query_words & question_words):
        return False

    year_match = re.search(r"\b(20\d{2})\b", question)
    if year_match and int(year_match.group(1)) <= 2021:
        return False

    return True


def search_via_public_api(query: str) -> List[Dict[str, Any]]:
    """Try public Polymarket discovery endpoints without crashing on failure."""
    endpoints = [
        (f"{GAMMA_BASE_URL}/search", {"query": query}),
        (f"{GAMMA_BASE_URL}/markets", {"search": query, "limit": 25}),
        (f"{GAMMA_BASE_URL}/events", {"search": query, "limit": 25}),
    ]
    candidates: List[Dict[str, Any]] = []

    for url, params in endpoints:
        try:
            response = requests.get(url, params=params, timeout=20)
            if response.status_code != 200:
                continue
            payload = response.json()
        except (requests.RequestException, ValueError):
            continue

        if not isinstance(payload, list):
            continue

        for item in payload:
            if not isinstance(item, dict):
                continue

            if "markets" in item and isinstance(item["markets"], list):
                for market in item["markets"]:
                    if isinstance(market, dict):
                        candidate = normalize_candidate(market, "public_search")
                        if is_usable_public_candidate(candidate, query):
                            candidates.append(candidate)
            else:
                candidate = normalize_candidate(item, "public_search")
                if is_usable_public_candidate(candidate, query):
                    candidates.append(candidate)

        if candidates:
            break

    return candidates


def search_via_local_markets(query: str, limit: int = 200) -> List[Dict[str, Any]]:
    """Search a larger local market universe with overlap and coverage scoring."""
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
                "kind": "market",
                "market_id": market.market_id,
                "question": market.question,
                "overlap": overlap,
                "coverage": coverage,
                "volume": market.volume,
                "liquidity_depth": market.liquidity_depth,
                "source": "local_market_search",
            }
        )

    candidates.sort(
        key=lambda item: (
            item["overlap"],
            item["coverage"],
            item["volume"],
        ),
        reverse=True,
    )
    return candidates


def merge_candidates(
    local_candidates: List[Dict[str, Any]],
    api_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge candidates, preferring live local results and deduplicating by id+question."""
    merged: List[Dict[str, Any]] = []
    seen = set()

    for candidate in local_candidates + api_candidates:
        key = (candidate.get("market_id"), candidate.get("question"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(candidate)

    return merged


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 -m app.search_markets <query>")
        return

    query = " ".join(sys.argv[1:])
    input_type, extracted_value = extract_input_type(query)

    if input_type == "event_slug":
        print("Direct event URL/slug detected. Use app.analyze_market for exact analysis.")
        print(f"Slug: {extracted_value}")
        return

    if input_type == "market_id":
        print("Direct market ID detected. Use app.analyze_market for exact analysis.")
        print(f"Market ID: {extracted_value}")
        return

    local_candidates = search_via_local_markets(query, limit=200)
    api_candidates = search_via_public_api(query)
    candidates = merge_candidates(local_candidates, api_candidates)

    live_candidates = [
        candidate for candidate in candidates if candidate.get("source") == "local_market_search"
    ]
    supplementary_candidates = [
        candidate for candidate in candidates if candidate.get("source") != "local_market_search"
    ]

    print("=" * 60)
    print("MARKET SEARCH RESULTS")
    print("=" * 60)
    print(f"Query: {query}")
    print("=" * 60)
    print()

    if not live_candidates and not supplementary_candidates:
        print("No matches found.")
        return

    if live_candidates:
        print("[Live Candidates]")
        for rank, candidate in enumerate(live_candidates[:10], start=1):
            print("-" * 60)
            print(f"Rank: {rank}")
            print(f"Market ID: {candidate.get('market_id')}")
            print(f"Question: {candidate.get('question')}")
            print(f"Source: {candidate.get('source')}")
            if "overlap" in candidate:
                print(
                    f"Overlap/Coverage: "
                    f"{candidate['overlap']} / {candidate['coverage']:.2f}"
                )
            if "volume" in candidate:
                print(f"Volume: {candidate['volume']:.2f}")
            if "liquidity_depth" in candidate:
                print(f"Liquidity: {candidate['liquidity_depth']:.2f}")
            print()

    if supplementary_candidates:
        print("[Supplementary Search Results]")
        for rank, candidate in enumerate(supplementary_candidates[:5], start=1):
            print("-" * 60)
            print(f"Rank: {rank}")
            print(f"Market ID: {candidate.get('market_id')}")
            print(f"Question: {candidate.get('question')}")
            print(f"Source: {candidate.get('source')}")
            if "overlap" in candidate:
                print(
                    f"Overlap/Coverage: "
                    f"{candidate['overlap']} / {candidate['coverage']:.2f}"
                )
            if "volume" in candidate:
                print(f"Volume: {candidate['volume']:.2f}")
            if "liquidity_depth" in candidate:
                print(f"Liquidity: {candidate['liquidity_depth']:.2f}")
            print()


if __name__ == "__main__":
    main()
