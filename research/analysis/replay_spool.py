"""
research/analysis/replay_spool.py — Replay spooled failed inserts and dropped fields.

Run after fixing schema to recover data that was rejected by Supabase.

    python -m research.analysis.replay_spool [--dry-run]

Idempotent: skips rows already in Supabase (by token_address + alert_date unique key).
Preserves original timestamps where available.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_SPOOL_DIR           = Path(__file__).parent.parent / "spool"
_FAILED_INSERTS_PATH = _SPOOL_DIR / "failed_inserts.jsonl"
_DROPPED_FIELDS_PATH = _SPOOL_DIR / "dropped_fields.jsonl"


def _load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _init_supabase():
    from research.config import SUPABASE_URL, SUPABASE_KEY
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _already_exists(sb, token_address: str, alert_time_iso: str) -> bool:
    """Check if a row with this token+date already exists (unique index guard)."""
    try:
        date_str = alert_time_iso[:10]
        resp = sb.table("research_tokens") \
            .select("id") \
            .eq("token_address", token_address) \
            .gte("alert_time", f"{date_str}T00:00:00Z") \
            .lt("alert_time",  f"{date_str}T23:59:59Z") \
            .limit(1) \
            .execute()
        return len(resp.data) > 0
    except Exception:
        return False


def replay_failed_inserts(sb, dry_run: bool) -> dict:
    rows = _load_jsonl(_FAILED_INSERTS_PATH)
    stats = {"total": len(rows), "replayed": 0, "skipped_existing": 0, "still_failed": 0}

    for entry in rows:
        row          = entry.get("row", {})
        token_address = entry.get("token_address") or row.get("token_address", "")
        alert_time    = row.get("alert_time", entry.get("ts", ""))

        if not token_address:
            stats["still_failed"] += 1
            continue

        if _already_exists(sb, token_address, alert_time):
            log.info("SKIP existing  %s  %s", token_address[:12], alert_time[:10])
            stats["skipped_existing"] += 1
            continue

        if dry_run:
            log.info("DRY-RUN would INSERT %s  %s", token_address[:12], alert_time[:10])
            stats["replayed"] += 1
            continue

        try:
            sb.table("research_tokens").insert(row, returning="minimal").execute()
            log.info("REPLAY INSERT  %s  %s", token_address[:12], alert_time[:10])
            stats["replayed"] += 1
        except Exception as e:
            log.warning("Still failed  %s: %s", token_address[:12], str(e)[:80])
            stats["still_failed"] += 1

    return stats


def replay_dropped_fields(sb, dry_run: bool) -> dict:
    """
    Replay dropped fields using the most precise event key available:
      1. row_id — targets exactly one Supabase row (preferred)
      2. (token_address, alert_date) — targets one row via unique index
      3. no key — skipped (reported as skipped_no_event_key)

    Never updates by token_address alone (would corrupt tokens with multiple
    alert rows on different dates).
    """
    rows  = _load_jsonl(_DROPPED_FIELDS_PATH)
    stats = {
        "total": len(rows),
        "replayed": 0,
        "skipped_no_row": 0,
        "skipped_no_event_key": 0,
        "still_failed": 0,
    }

    # Group by target key: ("id", row_id) or ("token_date", token_address, date_str)
    # Key → {"token_address": str, "alert_time": str, "key": tuple, "updates": dict}
    by_key: dict = {}

    for entry in rows:
        col  = entry.get("column")
        val  = entry.get("value")
        if not col:
            continue

        token_address = entry.get("token_address", "")
        alert_time    = entry.get("alert_time", "")
        row_id        = entry.get("row_id")

        if row_id is not None:
            k = ("id", str(row_id))
        elif token_address and alert_time:
            date_str = alert_time[:10]
            k = ("token_date", token_address, date_str)
        else:
            stats["skipped_no_event_key"] += 1
            log.info("SKIP no event key for col=%s token=%s", col, token_address[:12] if token_address else "?")
            continue

        if k not in by_key:
            by_key[k] = {"key": k, "token_address": token_address,
                         "alert_time": alert_time, "updates": {}}
        by_key[k]["updates"][col] = val

    for target in by_key.values():
        k       = target["key"]
        updates = target["updates"]
        addr    = target["token_address"]
        at      = target["alert_time"]

        if dry_run:
            if k[0] == "id":
                log.info("DRY-RUN would UPDATE id=%s  cols=%s", k[1], list(updates))
            else:
                log.info("DRY-RUN would UPDATE token=%s  date=%s  cols=%s",
                         addr[:12], k[2], list(updates))
            stats["replayed"] += len(updates)
            continue

        # Resolve exact Supabase row id
        try:
            if k[0] == "id":
                resp = sb.table("research_tokens").select("id").eq("id", k[1]).limit(1).execute()
            else:
                date_str = k[2]
                resp = sb.table("research_tokens").select("id") \
                    .eq("token_address", addr) \
                    .gte("alert_time", f"{date_str}T00:00:00Z") \
                    .lt("alert_time",  f"{date_str}T23:59:59Z") \
                    .limit(1).execute()

            if not resp.data:
                log.info("SKIP no row  key=%s", k)
                stats["skipped_no_row"] += len(updates)
                continue
            exact_id = resp.data[0]["id"]
        except Exception as e:
            log.warning("Lookup failed key=%s: %s", k, e)
            stats["still_failed"] += len(updates)
            continue

        try:
            sb.table("research_tokens").update(updates).eq("id", exact_id).execute()
            log.info("REPLAY UPDATE  id=%s  cols=%s", exact_id, list(updates))
            stats["replayed"] += len(updates)
        except Exception as e:
            log.warning("Still failed update id=%s: %s", exact_id, str(e)[:80])
            stats["still_failed"] += len(updates)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Replay spooled research data into Supabase")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without writing to Supabase")
    args = parser.parse_args()

    if args.dry_run:
        log.info("DRY-RUN mode — no writes to Supabase")

    sb = None if args.dry_run else _init_supabase()

    print(f"\n{'='*60}")
    print("REPLAY SPOOL")
    print(f"{'='*60}")

    fi_stats = replay_failed_inserts(sb, args.dry_run)
    df_stats = replay_dropped_fields(sb, args.dry_run)

    print(f"\n── Failed inserts {'─'*42}")
    print(f"  total in spool      : {fi_stats['total']}")
    print(f"  replayed_rows       : {fi_stats['replayed']}")
    print(f"  skipped_existing    : {fi_stats['skipped_existing']}")
    print(f"  still_failed        : {fi_stats['still_failed']}")

    print(f"\n── Dropped fields {'─'*42}")
    print(f"  total in spool      : {df_stats['total']}")
    print(f"  replayed_fields     : {df_stats['replayed']}")
    print(f"  skipped_no_row      : {df_stats['skipped_no_row']}")
    print(f"  skipped_no_event_key: {df_stats['skipped_no_event_key']}")
    print(f"  still_failed        : {df_stats['still_failed']}")

    still_failed = fi_stats["still_failed"] + df_stats["still_failed"]
    if still_failed:
        print(f"\nWARN — {still_failed} items still failed after replay.")
        print("       Check schema migration is applied, then re-run.")
        sys.exit(1)
    else:
        print(f"\nPASS — All spooled data replayed successfully.")


if __name__ == "__main__":
    main()
