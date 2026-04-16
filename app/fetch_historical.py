"""Fetch resolved Polymarket markets with historical price snapshots.

Fetches recently closed markets from the Gamma API, then retrieves the
YES-token's earliest available CLOB price (the "opening" price) as a
pre-resolution snapshot. Saves results to data/historical_raw.json.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
OUTPUT_FILE = Path("data/historical_raw.json")

# Price range considered "informative" (market still genuinely uncertain)
MIN_SNAPSHOT_PRICE = 0.05
MAX_SNAPSHOT_PRICE = 0.95

# Minimum volume to include a market
MIN_VOLUME = 1000

# Minimum outcome price to declare a winner (handles near-1.0 floats)
OUTCOME_THRESHOLD = 0.99

# Maximum markets to fetch per run
DEFAULT_LIMIT = 6000

# Trivial short-term market patterns to skip
SKIP_PATTERNS = ["up or down", "o/u", "over/under", "most kills", "first blood"]


def _parse_json_field(raw: Any) -> list:
    """Parse a JSON string field or return an empty list."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def _to_unix(date_str: str) -> Optional[int]:
    """Convert an ISO date string to a Unix timestamp."""
    if not date_str:
        return None
    try:
        normalized = date_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except (ValueError, TypeError):
        return None


def _is_trivial(question: str) -> bool:
    """Return True for markets that are noise for model evaluation."""
    q = question.lower()
    return any(pat in q for pat in SKIP_PATTERNS)


def fetch_midlife_price(token_id: str, created_ts: int, end_ts: int) -> Optional[float]:
    """Fetch the YES-token price from the middle of the market's lifetime.

    Uses a 12-hour window centred at the market's midpoint (50% through its
    life).  This matches what the live bot sees when it scans an active market —
    a price that has already incorporated early information but is not yet
    close to resolution.

    Previously this function used the first 6 hours after open (opening price),
    which is a very different price distribution from what the model sees at
    inference time (current live price of an ongoing market).  Aligning the
    training snapshot with inference ensures the logistic regression learns the
    right mapping.
    """
    lifetime = end_ts - created_ts
    if lifetime <= 0:
        return None

    mid_ts = created_ts + lifetime // 2
    window_start = mid_ts - 21600  # 6 h before midpoint
    window_end   = mid_ts + 21600  # 6 h after midpoint

    try:
        response = requests.get(
            f"{CLOB_BASE_URL}/prices-history",
            params={
                "market": token_id,
                "startTs": window_start,
                "endTs": window_end,
                "fidelity": 60,
            },
            timeout=20,
        )
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    history = response.json().get("history", [])
    if not history:
        return None

    prices = [entry["p"] for entry in history if "p" in entry]
    if not prices:
        return None

    # Median of the window to smooth noise
    prices.sort()
    return prices[len(prices) // 2]


def fetch_closed_markets(limit: int = DEFAULT_LIMIT) -> List[Dict[str, Any]]:
    """Fetch recently closed markets from the Gamma API."""
    markets = []
    seen_ids: set = set()
    offset = 0

    while len(markets) < limit:
        batch_size = min(100, limit - len(markets))

        try:
            response = requests.get(
                f"{GAMMA_BASE_URL}/markets",
                params={
                    "closed": "true",
                    "limit": batch_size,
                    "offset": offset,
                    "order": "closedTime",
                    "ascending": "false",
                },
                timeout=20,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Failed to fetch closed markets: {exc}"
            ) from exc

        items = response.json()
        if not items:
            break

        new_items = 0
        for item in items:
            market_id = str(item.get("id", ""))
            if not market_id or market_id in seen_ids:
                continue

            question = item.get("question", "")
            if _is_trivial(question):
                continue

            volume = float(item.get("volumeNum") or 0)
            if volume < MIN_VOLUME:
                continue

            outcome_prices = _parse_json_field(item.get("outcomePrices"))
            outcomes = _parse_json_field(item.get("outcomes"))

            if len(outcome_prices) != 2 or len(outcomes) != 2:
                continue

            try:
                p0, p1 = float(outcome_prices[0]), float(outcome_prices[1])
            except (ValueError, TypeError):
                continue

            if p0 >= OUTCOME_THRESHOLD and p1 < 0.5:
                actual_outcome = outcomes[0].lower()
            elif p1 >= OUTCOME_THRESHOLD and p0 < 0.5:
                actual_outcome = outcomes[1].lower()
            else:
                continue  # not yet resolved or ambiguous

            clob_token_ids = _parse_json_field(item.get("clobTokenIds"))
            if not clob_token_ids or not clob_token_ids[0]:
                continue

            yes_token_id = clob_token_ids[0]
            created_ts = _to_unix(item.get("createdAt"))
            end_ts = _to_unix(item.get("endDate"))

            if not created_ts or not end_ts:
                continue

            seen_ids.add(market_id)
            markets.append(
                {
                    "market_id": market_id,
                    "question": question,
                    "yes_token_id": yes_token_id,
                    "end_date": item.get("endDate"),
                    "created_ts": created_ts,
                    "end_ts": end_ts,
                    "volume": volume,
                    "liquidity": float(item.get("liquidityNum") or 0),
                    "actual_outcome": actual_outcome,
                    "outcomes": outcomes,
                    "category": item.get("category", ""),
                }
            )
            new_items += 1

            if len(markets) >= limit:
                break

        if len(items) < batch_size or new_items == 0:
            break

        offset += batch_size

    return markets


def enrich_with_prices(
    markets: List[Dict[str, Any]], delay_ms: int = 50
) -> List[Dict[str, Any]]:
    """Add midlife price snapshots to each market record."""
    enriched = []
    total = len(markets)

    for i, market in enumerate(markets):
        price = fetch_midlife_price(
            market["yes_token_id"], market["created_ts"], market["end_ts"]
        )

        # Only keep the price if it's informationally useful
        if price is not None and not (MIN_SNAPSHOT_PRICE <= price <= MAX_SNAPSHOT_PRICE):
            price = None

        market["snapshot_yes_price"] = price

        if (i + 1) % 50 == 0 or (i + 1) == total:
            found = sum(1 for m in enriched if m.get("snapshot_yes_price") is not None)
            found += 1 if price is not None else 0
            print(
                f"  Progress: {i + 1}/{total} fetched, "
                f"{found} with price snapshots"
            )

        enriched.append(market)
        time.sleep(delay_ms / 1000)

    return enriched


def main() -> None:
    print(f"Fetching recently closed markets (min volume: ${MIN_VOLUME:,.0f})...")
    markets = fetch_closed_markets(limit=DEFAULT_LIMIT)
    print(f"Found {len(markets)} resolved markets.")

    if not markets:
        print("No markets to enrich. Exiting.")
        return

    print("\nFetching opening price snapshots (first 6h of trading)...")
    enriched = enrich_with_prices(markets)

    with_price = [m for m in enriched if m.get("snapshot_yes_price") is not None]
    print(f"\nMarkets with price snapshots: {len(with_price)} / {len(enriched)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w") as f:
        json.dump(enriched, f, indent=2, default=str)

    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
