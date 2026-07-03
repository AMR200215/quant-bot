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
    rows  = _load_jsonl(_DROPPED_FIELDS_PATH)
    stats = {"total": len(rows), "replayed": 0, "skipped_no_row": 0, "still_failed": 0}

    # Group by token_address to batch updates
    by_token: dict = {}
    for entry in rows:
        addr = entry.get("token_address", "")
        if not addr:
            continue
        if addr not in by_token:
            by_token[addr] = {}
        col = entry.get("column")
        val = entry.get("value")
        if col:
            by_token[addr][col] = val

    for token_address, updates in by_token.items():
        if dry_run:
            log.info("DRY-RUN would UPDATE %s  cols=%s", token_address[:12], list(updates))
            stats["replayed"] += len(updates)
            continue

        # Verify row exists before updating
        try:
            resp = sb.table("research_tokens") \
                .select("id") \
                .eq("token_address", token_address) \
                .limit(1) \
                .execute()
            if not resp.data:
                log.info("SKIP no row for %s", token_address[:12])
                stats["skipped_no_row"] += len(updates)
                continue
        except Exception as e:
            log.warning("Lookup failed for %s: %s", token_address[:12], e)
            stats["still_failed"] += len(updates)
            continue

        try:
            sb.table("research_tokens").update(updates).eq("token_address", token_address).execute()
            log.info("REPLAY UPDATE  %s  cols=%s", token_address[:12], list(updates))
            stats["replayed"] += len(updates)
        except Exception as e:
            log.warning("Still failed update %s: %s", token_address[:12], str(e)[:80])
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
