"""Build a clean historical dataset CSV from raw historical market data.

Reads data/historical_raw.json (produced by fetch_historical.py) and
outputs data/historical_dataset.csv with normalized fields ready for
backtesting.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

RAW_FILE = Path("data/historical_raw.json")
OUTPUT_FILE = Path("data/historical_dataset.csv")

DATASET_HEADER = [
    "market_id",
    "question",
    "yes_price",
    "volume",
    "liquidity_depth",
    "actual_outcome",
    "end_date",
    "category",
    "snapshot_date",    # ISO timestamp of when this market was fetched
    "days_to_resolution", # days from snapshot to end_date (for model training)
]

# Price range we consider "informative" (not already nearly resolved)
MIN_PRICE = 0.05
MAX_PRICE = 0.95

_CATEGORY_RULES = [
    ("crypto",    ["btc", "bitcoin", "ethereum", "eth", "solana", "sol", "xrp",
                   "dogecoin", "doge", "crypto", "bnb", "hyperliquid"]),
    ("stocks",    ["nasdaq", "s&p", "sp500", "aapl", "apple", "meta", "google",
                   "googl", "amazon", "amzn", "tesla", "tsla", "nvda", "nvidia",
                   "microsoft", "msft", "stock", "close above", "close below"]),
    ("politics",  ["trump", "biden", "election", "congress", "senate", "president",
                   "vote", "democrat", "republican", "bill", "law", "cabinet",
                   "tariff", "elon", "musk", "doge"]),
    ("sports",    ["nba", "nfl", "mlb", "nhl", "ufc", "mma", "soccer", "football",
                   "basketball", "baseball", "hockey", "tennis", "golf", "formula",
                   "f1", "champion", "playoff", "super bowl", "world cup",
                   "premier league", "la liga", "bundesliga", "serie a"]),
    ("esports",   ["lol", "league of legends", "cs:", "csgo", "dota", "valorant",
                   "esport", "counter-strike", "overwatch", "iem", "esl",
                   "rift rivals", "emea masters", "kill"]),
    ("economics", ["fed", "interest rate", "inflation", "gdp", "recession",
                   "unemployment", "fomc", "central bank", "cpi", "jobs"]),
    ("weather",   ["temperature", "weather", "rainfall", "snow", "hurricane",
                   "celsius", "fahrenheit"]),
]


def infer_category(question: str) -> str:
    """Infer a market category from its question text."""
    q = question.lower()
    for category, keywords in _CATEGORY_RULES:
        if any(kw in q for kw in keywords):
            return category
    return "other"


def load_raw(path: Path = RAW_FILE) -> List[Dict[str, Any]]:
    """Load raw historical market data."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m app.fetch_historical` first."
        )
    with path.open() as f:
        return json.load(f)


def normalize_outcome(raw: str, outcomes: List[str]) -> Optional[str]:
    """Normalize an outcome string to 'yes' or 'no'.

    Uses the market's outcomes list to map the winner to yes/no.
    The first outcome is treated as YES, the second as NO.
    """
    if not raw or not outcomes:
        return None

    raw_lower = raw.lower().strip()

    if raw_lower in ("yes", "no"):
        return raw_lower

    # Map first outcome to 'yes', second to 'no'
    if len(outcomes) >= 2:
        if raw_lower == outcomes[0].lower().strip():
            return "yes"
        if raw_lower == outcomes[1].lower().strip():
            return "no"

    return None


def _compute_days_to_resolution(snapshot_date: str, end_date: str) -> str:
    """Compute days from snapshot_date to end_date. Returns '' on any parse error."""
    try:
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        snap = datetime.strptime(snapshot_date, fmt).replace(tzinfo=timezone.utc)
        end  = datetime.strptime(end_date[:19] + "Z", fmt).replace(tzinfo=timezone.utc)
        days = max(0.0, (end - snap).total_seconds() / 86400)
        return str(round(days, 1))
    except Exception:
        return ""


def build_records(raw_markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert raw market records into clean dataset rows."""
    records = []
    skipped_no_price = 0
    skipped_extreme = 0
    skipped_outcome = 0

    # Use build time as fallback snapshot_date for legacy records without one
    build_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for item in raw_markets:
        price = item.get("snapshot_yes_price")
        if price is None:
            skipped_no_price += 1
            continue

        try:
            price = float(price)
        except (TypeError, ValueError):
            skipped_no_price += 1
            continue

        if not (MIN_PRICE <= price <= MAX_PRICE):
            skipped_extreme += 1
            continue

        outcomes = item.get("outcomes", [])
        raw_outcome = item.get("actual_outcome", "")
        normalized = normalize_outcome(raw_outcome, outcomes)

        if normalized not in ("yes", "no"):
            skipped_outcome += 1
            continue

        question      = item.get("question", "")
        category      = item.get("category") or infer_category(question)
        end_date      = item.get("end_date", "")
        snapshot_date = item.get("snapshot_date", build_time)
        days_to_res   = _compute_days_to_resolution(snapshot_date, end_date)

        records.append(
            {
                "market_id":        item.get("market_id", ""),
                "question":         question,
                "yes_price":        round(price, 4),
                "volume":           round(float(item.get("volume") or 0), 2),
                "liquidity_depth":  round(float(item.get("liquidity") or 0), 2),
                "actual_outcome":   normalized,
                "end_date":         end_date,
                "category":         category,
                "snapshot_date":    snapshot_date,
                "days_to_resolution": days_to_res,
            }
        )

    print(f"  Skipped — no price snapshot:  {skipped_no_price}")
    print(f"  Skipped — extreme price:       {skipped_extreme}")
    print(f"  Skipped — unclear outcome:     {skipped_outcome}")
    print(f"  Records kept:                  {len(records)}")

    return records


def write_csv(records: List[Dict[str, Any]], path: Path = OUTPUT_FILE) -> None:
    """Write dataset records to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_HEADER)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    print(f"Loading raw data from {RAW_FILE}...")
    raw = load_raw()
    print(f"Loaded {len(raw)} raw market records.")
    print()

    print("Building dataset...")
    records = build_records(raw)

    if not records:
        print("\nNo usable records. Check fetch_historical output.")
        return

    write_csv(records)
    print(f"\nDataset saved to {OUTPUT_FILE}")

    yes_count = sum(1 for r in records if r["actual_outcome"] == "yes")
    no_count = len(records) - yes_count
    print(f"YES outcomes: {yes_count} ({yes_count / len(records):.1%})")
    print(f"NO outcomes:  {no_count} ({no_count / len(records):.1%})")


if __name__ == "__main__":
    main()
