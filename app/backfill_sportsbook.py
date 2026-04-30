"""Backfill sportsbook_p and kalshi_p for unresolved journal entries that have none.

Usage:
    python -m app.backfill_sportsbook          # dry run — show what would be fetched
    python -m app.backfill_sportsbook --run    # actually fetch and update journal
"""

import csv
import sys

from dotenv import load_dotenv
load_dotenv()

import os
from app.external_signals import fetch_odds_probability, fetch_kalshi_probability
from app.market_journal import JOURNAL_FILE, JOURNAL_HEADER

DRY_RUN = "--run" not in sys.argv

ODDS_API_KEY  = os.getenv("ODDS_API_KEY", "")
KALSHI_KEY    = os.getenv("KALSHI_API_KEY", "")
KALSHI_KEY_ID = os.getenv("KALSHI_KEY_ID", "")


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

    targets = [
        (i, r) for i, r in enumerate(rows)
        if not r.get("actual_outcome", "").strip()          # unresolved only
        and not r.get("sportsbook_p", "").strip()           # missing sportsbook
    ]

    print(f"Unresolved rows without sportsbook_p: {len(targets)}")

    if DRY_RUN:
        print("DRY RUN — pass --run to actually fetch\n")
        for _, r in targets[:20]:
            print(f"  {r['question'][:70]}")
        if len(targets) > 20:
            print(f"  ... and {len(targets) - 20} more")
        return

    updated = 0
    no_match = 0

    for i, row in targets:
        q = row["question"]
        sp = fetch_odds_probability(q, ODDS_API_KEY)
        kp = fetch_kalshi_probability(q, KALSHI_KEY, KALSHI_KEY_ID)

        if sp is not None or kp is not None:
            rows[i]["sportsbook_p"] = str(round(sp, 4)) if sp is not None else ""
            rows[i]["kalshi_p"]     = str(round(kp, 4)) if kp is not None else ""
            updated += 1
            parts = []
            if sp is not None: parts.append(f"sportsbook={sp:.4f}")
            if kp is not None: parts.append(f"kalshi={kp:.4f}")
            print(f"  MATCH  {q[:60]}  →  {', '.join(parts)}")
        else:
            no_match += 1

    if updated > 0:
        save_rows(rows)

    print(f"\nUpdated: {updated}  |  No match: {no_match}")


if __name__ == "__main__":
    main()
