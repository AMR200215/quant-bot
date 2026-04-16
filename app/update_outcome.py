import sys

from app.market_journal import load_journal_records, update_journal_outcome


def print_usage() -> None:
    print("Usage: python -m app.update_outcome <market_id> <yes|no> [optional notes]")


def main() -> None:
    if len(sys.argv) < 3:
        print_usage()
        return

    market_id = sys.argv[1]
    actual_outcome = sys.argv[2].lower()
    notes = " ".join(sys.argv[3:]) if len(sys.argv) > 3 else ""

    if actual_outcome not in {"yes", "no"}:
        print_usage()
        return

    records = load_journal_records()
    matching_records = [row for row in records if row.get("market_id") == market_id]
    if not matching_records:
        print("No journal entry found for that market ID.")
        return

    updated = update_journal_outcome(market_id, actual_outcome, notes)
    if not updated:
        print("No matching market could be updated.")
        return

    updated_records = load_journal_records()
    latest_match = None
    for row in reversed(updated_records):
        if row.get("market_id") == market_id:
            latest_match = row
            break

    print("=" * 60)
    print("OUTCOME UPDATED")
    print("=" * 60)
    print(f"Market ID: {market_id}")
    print(f"Actual Outcome: {actual_outcome}")
    print("Status: saved")
    print("=" * 60)

    if latest_match:
        print(f"Question: {latest_match.get('question', '')}")
        print(f"Preferred Side: {latest_match.get('preferred_side', '')}")
        print(f"Bot Correct: {latest_match.get('bot_correct', '')}")


if __name__ == "__main__":
    main()
