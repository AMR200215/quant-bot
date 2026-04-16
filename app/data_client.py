"""Market data ingestion helpers for the quant bot.

Step 1 of DMDO: Data -> Model -> Decision -> Output.
This module starts with a clean mock-data adapter so the rest of the bot can
be built and tested before connecting a live prediction-market API.
"""

from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, List, Optional

import requests

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"

# Patterns that identify machine-generated junk markets
_TRIVIAL_PATTERNS = [
    "up or down",
    "close above $",
    "close below $",
    "be above $",
    "be below $",
    "dip to $",
    "reach $",
    "hit $",
    "drop to $",
    "o/u ",
    "over/under",
    "most kills",
    "first blood",
    "first to ",
    "highest temperature",
    "lowest temperature",
    "temperature in",
    "°f on ",
    "°c on ",
    "between ",
]


def _is_trivial_market(question: str) -> bool:
    """Return True for machine-generated daily/5-min noise markets."""
    q = question.lower()
    return any(p in q for p in _TRIVIAL_PATTERNS)


CLOB_BASE_URL = "https://clob.polymarket.com"


@dataclass
class Market:
    market_id: str
    question: str
    yes_price: float
    no_price: float
    volume: float
    liquidity_depth: float
    end_date: Optional[str] = None
    days_to_resolution: Optional[float] = None
    # CLOB token ID for the YES outcome — needed for momentum enrichment
    clob_token_id: Optional[str] = None
    # Price 7 days ago; positive = market moved up toward YES, negative = down
    momentum_7d: Optional[float] = None


@dataclass
class IngestionSnapshot:
    source: str
    markets: List[Market]


def _safe_float(value: Any, default: float) -> float:
    """Convert a value to float when possible, otherwise return a default."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compute_days_to_resolution(end_date_str: Optional[str]) -> Optional[float]:
    """Compute time to market resolution in days from a UTC end timestamp."""
    if not end_date_str:
        return None

    try:
        normalized = end_date_str.replace("Z", "+00:00")
        end_date = datetime.fromisoformat(normalized)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (end_date - now).total_seconds() / 86400
    except (TypeError, ValueError):
        return None


def fetch_mock_markets() -> List[Market]:
    """Return a small set of realistic mock prediction markets."""
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


def fetch_live_markets(limit: int = 20) -> List[Market]:
    """Fetch active open markets from the Polymarket Gamma API."""
    markets: List[Market] = []
    seen_ids = set()
    offset = 0

    while len(markets) < limit:
        batch_size = min(100, limit - len(markets))

        try:
            response = requests.get(
                f"{GAMMA_BASE_URL}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": batch_size,
                    "offset": offset,
                },
                timeout=20,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Failed to fetch live Polymarket markets: {exc}"
            ) from exc

        items = response.json()
        if not items:
            break

        new_items = 0
        for item in items:
            market_id = str(item.get("id", ""))
            if not market_id or market_id in seen_ids:
                continue

            question = str(item.get("question", "Unknown market")).strip()
            raw_last_trade_price = item.get("lastTradePrice")
            raw_price = item.get("price")

            if raw_last_trade_price is None and raw_price is None:
                continue

            yes_price = _safe_float(raw_last_trade_price, float("nan"))
            if not 0 < yes_price < 1:
                yes_price = _safe_float(raw_price, float("nan"))

            if not question or not 0 < yes_price < 1:
                continue

            if _is_trivial_market(question):
                continue

            volume = _safe_float(item.get("volume"), 0.0)
            liquidity_depth = _safe_float(
                item.get("liquidity"), max(volume * 0.1, 1000.0)
            )
            end_date = item.get("endDate")
            days_to_resolution = _compute_days_to_resolution(end_date)

            clob_ids = item.get("clobTokenIds") or []
            if isinstance(clob_ids, str):
                try:
                    import json as _json
                    clob_ids = _json.loads(clob_ids)
                except Exception:
                    clob_ids = []
            clob_token_id = str(clob_ids[0]) if clob_ids else None

            markets.append(
                Market(
                    market_id=market_id,
                    question=question,
                    yes_price=yes_price,
                    no_price=round(1 - yes_price, 4),
                    volume=volume,
                    liquidity_depth=liquidity_depth,
                    end_date=end_date,
                    days_to_resolution=days_to_resolution,
                    clob_token_id=clob_token_id,
                )
            )
            seen_ids.add(market_id)
            new_items += 1

            if len(markets) >= limit:
                break

        if len(items) < batch_size or new_items == 0:
            break

        offset += batch_size

    return markets


def fetch_markets_by_days(
    min_days: float = 2,
    max_days: float = 60,
    limit: int = 100,
) -> List[Market]:
    """Fetch live non-trivial markets filtered to a days-to-resolution window.

    Uses endDate-ascending order and binary-searches for the right starting
    offset so we don't have to page through thousands of daily junk markets.
    """
    markets: List[Market] = []
    seen_ids: set = set()

    # Binary search for the offset where days_to_resolution >= min_days
    lo, hi = 0, 30000
    start_offset = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            resp = requests.get(
                f"{GAMMA_BASE_URL}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": 5,
                    "offset": mid,
                    "order": "endDate",
                    "ascending": "true",
                },
                timeout=20,
            )
            items = resp.json() if resp.status_code == 200 else []
        except requests.RequestException:
            items = []

        if not items:
            hi = mid - 1
            continue

        days = _compute_days_to_resolution(items[0].get("endDate"))
        if days is None:
            hi = mid - 1
            continue

        if days < min_days:
            lo = mid + 1
            start_offset = mid
        else:
            hi = mid - 1

    # Now paginate forward from start_offset collecting real markets
    offset = max(0, start_offset - 200)

    while len(markets) < limit:
        batch_size = min(100, (limit - len(markets)) * 5)  # over-fetch to compensate for filtering

        try:
            resp = requests.get(
                f"{GAMMA_BASE_URL}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": batch_size,
                    "offset": offset,
                    "order": "endDate",
                    "ascending": "true",
                },
                timeout=20,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to fetch markets by days: {exc}") from exc

        items = resp.json()
        if not items:
            break

        all_past_window = True
        for item in items:
            market_id = str(item.get("id", ""))
            if not market_id or market_id in seen_ids:
                continue

            question = str(item.get("question", "")).strip()
            if not question or _is_trivial_market(question):
                continue

            raw_last_trade_price = item.get("lastTradePrice")
            raw_price = item.get("price")
            yes_price = _safe_float(raw_last_trade_price, float("nan"))
            if not 0 < yes_price < 1:
                yes_price = _safe_float(raw_price, float("nan"))
            if not 0 < yes_price < 1:
                continue

            end_date = item.get("endDate")
            days_to_resolution = _compute_days_to_resolution(end_date)

            if days_to_resolution is not None and days_to_resolution < min_days:
                continue

            if days_to_resolution is not None and days_to_resolution <= max_days:
                all_past_window = False

            if days_to_resolution is not None and days_to_resolution > max_days:
                continue

            volume = _safe_float(item.get("volume"), 0.0)
            liquidity_depth = _safe_float(
                item.get("liquidity"), max(volume * 0.1, 1000.0)
            )

            clob_ids = item.get("clobTokenIds") or []
            if isinstance(clob_ids, str):
                try:
                    import json as _json
                    clob_ids = _json.loads(clob_ids)
                except Exception:
                    clob_ids = []
            clob_token_id = str(clob_ids[0]) if clob_ids else None

            markets.append(
                Market(
                    market_id=market_id,
                    question=question,
                    yes_price=yes_price,
                    no_price=round(1 - yes_price, 4),
                    volume=volume,
                    liquidity_depth=liquidity_depth,
                    end_date=end_date,
                    days_to_resolution=days_to_resolution,
                    clob_token_id=clob_token_id,
                )
            )
            seen_ids.add(market_id)

            if len(markets) >= limit:
                break

        if all_past_window and offset > start_offset:
            break

        offset += batch_size

    return markets


def _fetch_price_7d_ago(clob_token_id: str) -> Optional[float]:
    """Return the YES-token price 7 days ago using the CLOB price history API."""
    import time as _time
    now_ts = int(_time.time())
    start_ts = now_ts - 7 * 86400 - 3600  # 7 days ago ±1 h window
    end_ts = now_ts - 7 * 86400 + 3600

    try:
        resp = requests.get(
            f"{CLOB_BASE_URL}/prices-history",
            params={
                "market": clob_token_id,
                "startTs": start_ts,
                "endTs": end_ts,
                "fidelity": 60,
            },
            timeout=10,
        )
    except requests.RequestException:
        return None

    if resp.status_code != 200:
        return None

    history = resp.json().get("history", [])
    prices = [entry["p"] for entry in history if "p" in entry]
    if not prices:
        return None

    prices.sort()
    return round(prices[len(prices) // 2], 4)


def enrich_with_momentum(markets: List[Market], delay_ms: int = 80) -> None:
    """Fetch 7-day price history and compute momentum in-place.

    momentum_7d = current yes_price − price_7d_ago
      > 0  market is drifting toward YES  (bullish momentum)
      < 0  market is drifting toward NO   (bearish momentum)

    Only markets with a clob_token_id are enriched; others are left as None.
    Operates in-place — no return value.
    """
    import time as _time
    for market in markets:
        if not market.clob_token_id:
            continue
        price_7d = _fetch_price_7d_ago(market.clob_token_id)
        if price_7d is not None:
            market.momentum_7d = round(market.yes_price - price_7d, 4)
        _time.sleep(delay_ms / 1000)


def get_market_snapshot(source: str = "mock") -> IngestionSnapshot:
    """Return a normalized ingestion snapshot.

    Later, this function can switch between mock data and live API data without
    changing the rest of the bot.
    """
    if source == "mock":
        return IngestionSnapshot(source="mock", markets=fetch_mock_markets())
    if source == "live":
        return IngestionSnapshot(source="live", markets=fetch_live_markets())

    raise ValueError(f"Unsupported data source: {source}")


if __name__ == "__main__":
    try:
        snapshot = get_market_snapshot("live")
    except Exception as exc:
        print(exc)
        snapshot = get_market_snapshot("mock")

    print(f"Source: {snapshot.source}")
    print(f"Markets fetched: {len(snapshot.markets)}")
    for market in snapshot.markets:
        print(market)
