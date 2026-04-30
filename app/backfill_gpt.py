"""Backfill GPT verdicts for unresolved journal entries that have none.

Reads market_journal.csv, finds rows where:
  - actual_outcome is empty (not yet resolved)
  - gpt_verdict is empty (logged before OPENAI_API_KEY was set)

Calls gpt-4o-search-preview for each and writes the verdict back.

Usage:
    python -m app.backfill_gpt            # dry run — show what would be called
    python -m app.backfill_gpt --run      # actually call GPT and update journal
"""

import csv
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from app.gpt_analyst import GPT_MIN_EDGE, analyze as gpt_analyze
from app.market_journal import JOURNAL_FILE, JOURNAL_HEADER

DRY_RUN   = "--run"     not in sys.argv
REFRESH   = "--refresh" in sys.argv  # re-run even markets that already have a verdict


def load_rows() -> list[dict]:
    with JOURNAL_FILE.open("r", newline="") as f:
        return list(csv.DictReader(f))


def save_rows(rows: list[dict]) -> None:
    with JOURNAL_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=JOURNAL_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = load_rows()

    if REFRESH:
        # Re-run all unresolved markets regardless of existing verdict
        targets = [
            (i, r) for i, r in enumerate(rows)
            if not r.get("actual_outcome", "").strip()
            and float(r.get("adjusted_edge") or 0) >= GPT_MIN_EDGE
        ]
        print(f"REFRESH mode — re-running all {len(targets)} unresolved markets above edge threshold")
    else:
        targets = [
            (i, r) for i, r in enumerate(rows)
            if not r.get("actual_outcome", "").strip()
            and r.get("gpt_verdict", "").strip() in ("", "skipped")
        ]
        print(f"Unresolved rows with no GPT verdict (empty or skipped): {len(targets)}")

    if DRY_RUN:
        print("DRY RUN — pass --run to actually call GPT\n")
        above_threshold = [
            (i, r) for i, r in targets
            if float(r.get("adjusted_edge") or 0) >= GPT_MIN_EDGE
        ]
        below_threshold = len(targets) - len(above_threshold)
        print(f"  Would call GPT on: {len(above_threshold)} markets (edge >= {GPT_MIN_EDGE:.0%})")
        print(f"  Would skip (edge too low): {below_threshold}\n")
        for _, r in above_threshold:
            edge = float(r.get("adjusted_edge") or 0)
            print(f"  [{r['preferred_side']}] {r['question'][:65]}  edge={edge:.3f}")
        return

    updated = 0
    skipped_edge = 0
    errors = 0

    for i, row in targets:
        edge = float(row.get("adjusted_edge") or 0)
        if edge < GPT_MIN_EDGE:
            skipped_edge += 1
            continue

        print(f"Calling GPT: {row['question'][:70]}")
        result = gpt_analyze(
            question=row["question"],
            yes_price=float(row["yes_price"]),
            posterior=float(row["posterior"]),
            preferred_side=row["preferred_side"],
            adjusted_edge=edge,
        )

        verdict = result["verdict"]
        reasoning = result["reasoning"]
        icon = {"confirm": "✓", "reject": "✗", "neutral": "~", "error": "!", "skipped": "-"}.get(verdict, "?")
        print(f"  [{icon} {verdict.upper()}] {reasoning}\n")

        if verdict in ("error", "skipped"):
            errors += 1
        else:
            rows[i]["gpt_verdict"] = verdict
            rows[i]["gpt_reasoning"] = reasoning
            updated += 1

    if updated > 0:
        save_rows(rows)
        print(f"\nUpdated {updated} rows. Skipped {skipped_edge} (edge < {GPT_MIN_EDGE:.0%}). Errors: {errors}.")
        print("Journal saved. Run 'python -m app.review_journal' to see GPT accuracy breakdown.")
    else:
        print(f"\nNo rows updated. Skipped {skipped_edge} (edge < {GPT_MIN_EDGE:.0%}). Errors: {errors}.")


if __name__ == "__main__":
    main()
