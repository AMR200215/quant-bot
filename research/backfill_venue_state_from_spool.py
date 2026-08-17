"""
Recovers venue_state_at_signal values that tracker.py captured correctly
but couldn't persist because the column didn't exist in production yet
(P2-0, 2026-08-17 — the column now exists, see research/supabase_schema.sql).

This is NOT a fabrication/inference backfill (research/v8_clean_cohort.py's
KNOWN_GAPS explicitly forbids inferring venue_state from dex_id). It
recovers REAL, already-captured values: tracker.py's insert retry loop
(_insert_research_token, PGRST204 handling) strips any column PostgREST
doesn't recognize and spools the dropped (column, value) pair to
research/spool/dropped_fields.jsonl instead of losing it — exactly so
this recovery is possible once the schema catches up.

Matches each spooled venue_state_at_signal entry to its research_tokens
row by (token_address, alert_time) — the same two fields tracker.py used
to build that row — and updates ONLY rows where venue_state_at_signal is
currently NULL (idempotent; safe to re-run; never overwrites a value
that's already there, e.g. from a fresh non-dropped insert).

Run:
    python -m research.backfill_venue_state_from_spool [--dry-run]
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("backfill_venue_state_from_spool")

_SPOOL_PATH = Path(__file__).parent / "spool" / "dropped_fields.jsonl"


def _load_venue_state_drops() -> list[dict]:
    if not _SPOOL_PATH.exists():
        return []
    out = []
    for line in _SPOOL_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("table") == "research_tokens" and r.get("column") == "venue_state_at_signal":
            out.append(r)
    return out


def main():
    parser = argparse.ArgumentParser(description="Recover spooled venue_state_at_signal values")
    parser.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    args = parser.parse_args()

    from supabase import create_client
    from research.config import SUPABASE_URL, SUPABASE_KEY
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    drops = _load_venue_state_drops()
    log.info("Found %d spooled venue_state_at_signal entries", len(drops))
    if not drops:
        print("Nothing to recover.")
        return

    updated = 0
    already_set = 0
    no_match = 0
    ambiguous = 0
    errors = 0

    for d in drops:
        token_address = d.get("token_address")
        alert_time = d.get("alert_time")
        value = d.get("value")
        if not token_address or not alert_time or not value:
            errors += 1
            continue

        resp = (sb.table("research_tokens")
                .select("id,venue_state_at_signal")
                .eq("token_address", token_address)
                .eq("alert_time", alert_time)
                .execute())
        rows = resp.data or []

        if not rows:
            no_match += 1
            log.warning("No matching row for %s @ %s", token_address[:8], alert_time)
            continue
        if len(rows) > 1:
            # Same mint + same exact alert_time landing twice would be a real
            # ambiguity (matches P2-7's AMBIGUOUS_PATH_EVENT_JOIN reasoning) --
            # skip rather than guess which row this value belongs to.
            ambiguous += 1
            log.warning("Ambiguous match (%d rows) for %s @ %s -- skipped", len(rows), token_address[:8], alert_time)
            continue

        row = rows[0]
        if row.get("venue_state_at_signal") is not None:
            already_set += 1
            continue

        if args.dry_run:
            updated += 1
            continue

        try:
            sb.table("research_tokens").update({"venue_state_at_signal": value}).eq("id", row["id"]).execute()
            updated += 1
        except Exception as e:
            errors += 1
            log.error("Update failed for row id=%s: %s", row["id"], e)

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Recovery summary:")
    print(f"  spooled entries examined: {len(drops)}")
    print(f"  {'would update' if args.dry_run else 'updated'}:         {updated}")
    print(f"  already set (skipped):    {already_set}")
    print(f"  no matching row:          {no_match}")
    print(f"  ambiguous match (skipped): {ambiguous}")
    print(f"  errors:                   {errors}")


if __name__ == "__main__":
    main()
