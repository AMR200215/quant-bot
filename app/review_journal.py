import csv
from pathlib import Path
from typing import Callable

from app.market_journal import JOURNAL_FILE


def load_records() -> list[dict]:
    """Load journal rows if the journal file exists."""
    if not Path(JOURNAL_FILE).exists():
        return []

    with Path(JOURNAL_FILE).open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)


def to_float(value: str, default: float = 0.0) -> float:
    """Safely convert a string value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def resolved_records(records: list[dict]) -> list[dict]:
    """Keep only journal rows with resolved outcomes and correctness."""
    return [
        row
        for row in records
        if row.get("actual_outcome", "").strip()
        and row.get("bot_correct", "").strip()
    ]


def bucket_final_signal(value: float) -> str:
    if value < 0.01:
        return "<0.01"
    if value < 0.03:
        return "0.01-0.03"
    if value < 0.05:
        return "0.03-0.05"
    return ">0.05"


def bucket_confidence(value: float) -> str:
    if value < 0.30:
        return "low"
    if value < 0.60:
        return "medium"
    return "high"


def bucket_risk(value: float) -> str:
    if value < 0.30:
        return "low"
    if value < 0.60:
        return "medium"
    return "high"


def summarize_bucket(
    records: list[dict], bucket_fn: Callable[[float], str], field_name: str
) -> dict[str, dict]:
    """Group resolved records into buckets and compute win rates."""
    summary: dict[str, dict] = {}

    for row in resolved_records(records):
        bucket = bucket_fn(to_float(row.get(field_name, "")))
        if bucket not in summary:
            summary[bucket] = {"count": 0, "correct_count": 0, "win_rate": 0.0}

        summary[bucket]["count"] += 1
        if row.get("bot_correct", "").lower() == "true":
            summary[bucket]["correct_count"] += 1

    for bucket, values in summary.items():
        count = values["count"]
        values["win_rate"] = values["correct_count"] / count if count else 0.0

    return summary


def summarize_side(records: list[dict]) -> dict[str, dict]:
    """Compute win-rate summary by preferred side."""
    summary: dict[str, dict] = {}

    for row in resolved_records(records):
        side = row.get("preferred_side", "")
        if side not in summary:
            summary[side] = {"count": 0, "correct_count": 0, "win_rate": 0.0}

        summary[side]["count"] += 1
        if row.get("bot_correct", "").lower() == "true":
            summary[side]["correct_count"] += 1

    for side, values in summary.items():
        count = values["count"]
        values["win_rate"] = values["correct_count"] / count if count else 0.0

    return summary


def print_bucket_section(
    title: str, order: list[str], summary: dict[str, dict]
) -> None:
    """Print a formatted bucket summary section."""
    print(title)
    for bucket in order:
        values = summary.get(bucket, {"count": 0, "win_rate": 0.0})
        print(
            f"{bucket:<10} | count={values['count']:<3} | "
            f"win_rate={values['win_rate']:.2%}"
        )
    print()


def main() -> None:
    records = load_records()
    resolved = resolved_records(records)
    unresolved_count = len(records) - len(resolved)

    print("=" * 60)
    print("JOURNAL REVIEW")
    print("=" * 60)
    print(f"Total records: {len(records)}")
    print(f"Resolved records: {len(resolved)}")
    print(f"Unresolved records: {unresolved_count}")
    print("=" * 60)
    print()

    if not resolved:
        print("No resolved records yet. Add actual outcomes first.")
        return

    total_correct = sum(
        1 for row in resolved if row.get("bot_correct", "").lower() == "true"
    )
    overall_win_rate = total_correct / len(resolved)

    print("[Overall]")
    print(f"Resolved Trades: {len(resolved)}")
    print(f"Correct Calls: {total_correct}")
    print(f"Win Rate: {overall_win_rate:.2%}")
    print()

    side_summary = summarize_side(records)
    print("[By Side]")
    for side in ["buy_yes", "buy_no"]:
        values = side_summary.get(side, {"count": 0, "correct_count": 0, "win_rate": 0.0})
        print(
            f"{side}: count={values['count']} | "
            f"correct={values['correct_count']} | "
            f"win_rate={values['win_rate']:.2%}"
        )
    print()

    final_signal_summary = summarize_bucket(records, bucket_final_signal, "final_signal")
    confidence_summary = summarize_bucket(records, bucket_confidence, "confidence")
    risk_summary = summarize_bucket(records, bucket_risk, "risk_score")

    print_bucket_section(
        "[By Final Signal]",
        ["<0.01", "0.01-0.03", "0.03-0.05", ">0.05"],
        final_signal_summary,
    )
    print_bucket_section(
        "[By Confidence]",
        ["low", "medium", "high"],
        confidence_summary,
    )
    print_bucket_section(
        "[By Risk]",
        ["low", "medium", "high"],
        risk_summary,
    )

    best_signal_bucket = max(
        final_signal_summary.items(),
        key=lambda item: item[1]["win_rate"],
    )[0]
    worst_signal_bucket = min(
        final_signal_summary.items(),
        key=lambda item: item[1]["win_rate"],
    )[0]

    yes_rate = side_summary.get("buy_yes", {"win_rate": 0.0})["win_rate"]
    no_rate = side_summary.get("buy_no", {"win_rate": 0.0})["win_rate"]
    better_side = "buy_yes" if yes_rate >= no_rate else "buy_no"

    low_confidence_rate = confidence_summary.get("low", {"win_rate": 0.0})["win_rate"]
    medium_confidence_rate = confidence_summary.get("medium", {"win_rate": 0.0})["win_rate"]
    high_confidence_rate = confidence_summary.get("high", {"win_rate": 0.0})["win_rate"]

    high_risk_rate = risk_summary.get("high", {"win_rate": 0.0})["win_rate"]
    low_risk_rate = risk_summary.get("low", {"win_rate": 0.0})["win_rate"]

    print("[Insights]")
    print(f"- Best signal bucket: {best_signal_bucket}")
    print(f"- Worst signal bucket: {worst_signal_bucket}")
    print(f"- Better performing side: {better_side}")

    if low_confidence_rate < max(medium_confidence_rate, high_confidence_rate):
        print("- Low-confidence markets appear to underperform stronger setups.")
    else:
        print("- Low-confidence markets do not obviously underperform yet.")

    if high_risk_rate < low_risk_rate:
        print("- High-risk markets appear to underperform lower-risk setups.")
    else:
        print("- High-risk markets do not obviously underperform yet.")


if __name__ == "__main__":
    main()
