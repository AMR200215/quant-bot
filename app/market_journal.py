import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Optional

JOURNAL_FILE = Path("logs/market_journal.csv")

JOURNAL_HEADER = [
    "timestamp",
    "market_id",
    "question",
    "yes_price",
    "posterior",
    "preferred_side",
    "adjusted_edge",
    "final_signal",
    "confidence",
    "risk_score",
    "risk_multiplier",
    "days_to_resolution",
    "maturity_score",
    "resolution_quality_score",
    "actual_outcome",
    "bot_correct",
    "notes",
]


def ensure_journal_exists() -> None:
    """Ensure the journal file and its parent directory exist."""
    JOURNAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not JOURNAL_FILE.exists():
        with JOURNAL_FILE.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(JOURNAL_HEADER)


def is_already_logged(market_id: str) -> bool:
    """Return True if this market already has an unresolved entry in the journal."""
    ensure_journal_exists()
    with JOURNAL_FILE.open("r", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if row["market_id"] == market_id and not row["actual_outcome"]:
                return True
    return False


def append_journal_record(
    market_id: str,
    question: str,
    yes_price: float,
    posterior: float,
    preferred_side: str,
    adjusted_edge: float,
    final_signal: float,
    confidence: float,
    risk_score: float,
    risk_multiplier: float,
    days_to_resolution: Optional[float],
    maturity_score: float,
    resolution_quality_score: float,
    notes: str = "",
) -> bool:
    """Append a new market analysis record to the journal.

    Skips markets that already have an unresolved entry (deduplication).
    Returns True if the record was written, False if skipped.
    """
    if is_already_logged(market_id):
        return False

    ensure_journal_exists()
    timestamp = datetime.now(UTC).isoformat()
    days_value = "" if days_to_resolution is None else days_to_resolution

    with JOURNAL_FILE.open("a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                timestamp,
                market_id,
                question,
                yes_price,
                posterior,
                preferred_side,
                adjusted_edge,
                final_signal,
                confidence,
                risk_score,
                risk_multiplier,
                days_value,
                maturity_score,
                resolution_quality_score,
                "",
                "",
                notes,
            ]
        )
    return True


def load_journal_records() -> List[Dict]:
    """Load all market journal records."""
    ensure_journal_exists()
    with JOURNAL_FILE.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)


def update_journal_outcome(
    market_id: str,
    actual_outcome: str,
    notes: str = "",
) -> bool:
    """Update the most recent matching journal record with an outcome."""
    ensure_journal_exists()
    rows = load_journal_records()

    target_index = None
    for index in range(len(rows) - 1, -1, -1):
        if rows[index]["market_id"] == market_id:
            target_index = index
            break

    if target_index is None:
        return False

    row = rows[target_index]
    row["actual_outcome"] = actual_outcome
    row["notes"] = notes if notes else row["notes"]

    preferred_side = row.get("preferred_side", "")
    outcome_normalized = actual_outcome.lower()
    if preferred_side == "buy_yes" and outcome_normalized == "yes":
        row["bot_correct"] = "true"
    elif preferred_side == "buy_no" and outcome_normalized == "no":
        row["bot_correct"] = "true"
    else:
        row["bot_correct"] = "false"

    with JOURNAL_FILE.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=JOURNAL_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    return True


def display_value(value: str, fallback: str) -> str:
    """Return a fallback label for empty journal values."""
    return value if value else fallback


if __name__ == "__main__":
    records = load_journal_records()
    if not records:
        print("Market journal is empty.")
    else:
        latest = records[-1]
        print("=" * 60)
        print("LATEST JOURNAL ENTRY")
        print("=" * 60)
        print(f"Timestamp: {latest['timestamp']}")
        print(f"Market ID: {latest['market_id']}")
        print(f"Question: {latest['question']}")
        print()
        print("[Signal]")
        print(f"Preferred Side: {latest['preferred_side']}")
        print(f"Posterior: {latest['posterior']}")
        print(f"Adjusted Edge: {latest['adjusted_edge']}")
        print(f"Final Signal: {latest['final_signal']}")
        print(f"Confidence: {latest['confidence']}")
        print(f"Risk Score: {latest['risk_score']}")
        print(f"Risk Multiplier: {latest['risk_multiplier']}")
        print()
        print("[Timing]")
        print(
            f"Days to Resolution: "
            f"{display_value(latest['days_to_resolution'], 'unknown')}"
        )
        print(f"Maturity Score: {latest['maturity_score']}")
        print()
        print("[Quality]")
        print(
            f"Resolution Quality Score: "
            f"{latest['resolution_quality_score']}"
        )
        print()
        print("[Outcome]")
        print(
            f"Actual Outcome: "
            f"{display_value(latest['actual_outcome'], 'not recorded')}"
        )
        print(
            f"Bot Correct: "
            f"{display_value(latest['bot_correct'], 'not recorded')}"
        )
        print()
        print(f"Notes: {latest['notes']}")
        print("=" * 60)
