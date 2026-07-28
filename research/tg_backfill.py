"""
research/tg_backfill.py — backfill missed TG alerts from channel history.

Walks forward through pumpdotfunalert message history from a known last-good
message ID (or a cutoff date), extracts Solana addresses, and appends them to
signal_queue.jsonl with backfill provenance fields.

Idempotent: generates a deterministic event_id per (channel, message_id, address)
and skips rows that already exist in Supabase with that event_id.

Usage:
    python -m research.tg_backfill
    python -m research.tg_backfill --dry-run
    python -m research.tg_backfill --limit 5000
    python -m research.tg_backfill --min-id 12345
    python -m research.tg_backfill --since 2026-07-14T23:43:00

State persisted to: research/data/tg_backfill_state.json
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_SOL_ADDRESS_RE = re.compile(r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b')

_CHANNEL = "pumpdotfunalert"
_SIGNAL_QUEUE_PATH = Path(__file__).parent / "data" / "signal_queue.jsonl"
_STATE_PATH        = Path(__file__).parent / "data" / "tg_backfill_state.json"

# Session file shared with memecoin/telegram_monitor.py
_SESSION_FILE = str(
    Path(__file__).parent.parent / "memecoin" / "data" / "tg_session"
)


def _get_credentials() -> tuple[int, str]:
    api_id_raw = os.environ.get("TELEGRAM_API_ID", "")
    api_hash   = os.environ.get("TELEGRAM_API_HASH", "")
    if not api_id_raw or not api_hash:
        print(
            "ERROR: TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in environment.",
            file=sys.stderr,
        )
        sys.exit(1)
    return int(api_id_raw), api_hash


def _make_event_id(channel: str, message_id: int, address: str) -> str:
    """Deterministic 16-char hex event ID — stable across re-runs."""
    raw = f"tg:{channel}:{message_id}:{address}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _extract_addresses(text: str) -> list[str]:
    """Extract Solana addresses from message text."""
    found = []
    for addr in _SOL_ADDRESS_RE.findall(text):
        if len(addr) >= 32:
            found.append(addr)
    return list(dict.fromkeys(found))   # deduplicate while preserving order


def _load_state() -> dict:
    if _STATE_PATH.exists():
        try:
            return json.loads(_STATE_PATH.read_text())
        except Exception as e:
            log.warning("Failed to read backfill state: %s", e)
    return {}


def _save_state(state: dict):
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(state, indent=2))


def _check_event_exists_supabase(event_ids: list[str]) -> set[str]:
    """
    Check which event_ids already exist in Supabase.
    Returns set of already-existing event_ids.
    """
    if not event_ids:
        return set()
    try:
        from research.config import SUPABASE_URL, SUPABASE_KEY
        from supabase import create_client
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        # Supabase PostgREST: filter with 'in' operator
        resp = (
            sb.table("research_tokens")
            .select("event_id")
            .in_("event_id", event_ids)
            .execute()
        )
        return {row["event_id"] for row in (resp.data or [])}
    except Exception as e:
        log.warning("Supabase event_id check failed: %s — assuming none exist", e)
        return set()


def _append_to_queue(entries: list[dict], dry_run: bool):
    """Append backfill entries to signal_queue.jsonl."""
    if dry_run:
        for e in entries:
            print(f"  [DRY-RUN] would insert: {e['token_address'][:12]} "
                  f"msg_id={e['tg_message_id']} time={e['alert_time']}")
        return
    _SIGNAL_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_SIGNAL_QUEUE_PATH, "a") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


async def _run_backfill(
    min_id: int,
    limit: int,
    dry_run: bool,
    batch_id: str,
    since_dt: datetime | None,
):
    try:
        from telethon import TelegramClient
    except ImportError:
        print("ERROR: telethon not installed. Run: pip install telethon", file=sys.stderr)
        sys.exit(1)

    api_id, api_hash = _get_credentials()

    client = TelegramClient(_SESSION_FILE, api_id, api_hash)
    await client.connect()

    if not await client.is_user_authorized():
        print(
            "ERROR: Telegram session not authorised. Run: python -m research.tg_auth",
            file=sys.stderr,
        )
        await client.disconnect()
        sys.exit(1)

    log.info("Connected. Backfilling from %s (min_id=%d, limit=%d)", _CHANNEL, min_id, limit)

    messages_scanned = 0
    addresses_found  = 0
    new_inserts      = 0
    skipped_existing = 0
    earliest_time: datetime | None = None
    latest_time:   datetime | None = None
    highest_id     = min_id

    # Batch event_ids for Supabase check
    _BATCH_CHECK_SIZE = 100
    pending_entries: list[dict] = []
    pending_event_ids: list[str] = []

    async def _flush(force: bool = False):
        nonlocal new_inserts, skipped_existing
        if not pending_entries:
            return
        if not force and len(pending_entries) < _BATCH_CHECK_SIZE:
            return
        existing = _check_event_exists_supabase(pending_event_ids)
        to_write = [e for e in pending_entries if e["event_id"] not in existing]
        skip_count = len(pending_entries) - len(to_write)
        skipped_existing += skip_count
        new_inserts += len(to_write)
        _append_to_queue(to_write, dry_run)
        pending_entries.clear()
        pending_event_ids.clear()

    try:
        async for msg in client.iter_messages(
            _CHANNEL,
            min_id=min_id,
            limit=limit,
            reverse=True,    # walk forward (oldest first)
        ):
            if msg is None or not msg.text:
                continue

            # Apply date filter if --since was given
            if since_dt and msg.date:
                msg_dt = msg.date.replace(tzinfo=timezone.utc) if msg.date.tzinfo is None else msg.date
                if msg_dt < since_dt:
                    continue

            messages_scanned += 1
            highest_id = max(highest_id, msg.id)

            # Also scan entity URLs
            extra_urls = []
            if msg.entities:
                for ent in msg.entities:
                    url = getattr(ent, "url", None)
                    if url:
                        extra_urls.append(url)
            combined = (msg.text or "") + " " + " ".join(extra_urls)

            addrs = _extract_addresses(combined)
            if not addrs:
                continue

            addresses_found += len(addrs)
            msg_dt = msg.date
            if msg_dt:
                if msg_dt.tzinfo is None:
                    msg_dt = msg_dt.replace(tzinfo=timezone.utc)
                alert_time_str = msg_dt.isoformat().replace("+00:00", "Z")
                if earliest_time is None or msg_dt < earliest_time:
                    earliest_time = msg_dt
                if latest_time is None or msg_dt > latest_time:
                    latest_time = msg_dt
            else:
                alert_time_str = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            raw_text = (msg.text or "")[:500]

            for addr in addrs:
                event_id = _make_event_id(_CHANNEL, msg.id, addr)
                entry = {
                    "token_address":    addr,
                    "chain":            "solana",
                    "alert_time":       alert_time_str,
                    "raw_text":         raw_text,
                    "backfilled":       True,
                    "backfill_batch_id": batch_id,
                    "source":           "telegram_history",
                    "tg_message_id":    msg.id,
                    "event_id":         event_id,
                }
                pending_entries.append(entry)
                pending_event_ids.append(event_id)

            await _flush()

            if messages_scanned % 500 == 0:
                log.info(
                    "Progress: %d messages scanned, %d addresses, %d new, %d skipped",
                    messages_scanned, addresses_found, new_inserts, skipped_existing,
                )

    finally:
        await _flush(force=True)
        await client.disconnect()

    # Persist state
    state = _load_state()
    state["last_processed_message_id"] = highest_id
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    state["last_batch_id"] = batch_id
    if not dry_run:
        _save_state(state)

    print("\n--- Backfill summary ---")
    print(f"Messages scanned : {messages_scanned}")
    print(f"Addresses found  : {addresses_found}")
    print(f"New inserts      : {new_inserts}")
    print(f"Skipped (exists) : {skipped_existing}")
    print(f"Earliest alert   : {earliest_time.isoformat() if earliest_time else 'N/A'}")
    print(f"Latest alert     : {latest_time.isoformat() if latest_time else 'N/A'}")
    print(f"Highest msg ID   : {highest_id}")
    print(f"Batch ID         : {batch_id}")
    if dry_run:
        print("(DRY-RUN — no files written)")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill missed TG alerts from channel history into signal_queue.jsonl.",
    )
    parser.add_argument(
        "--min-id", type=int, default=None,
        help="Start from this Telegram message ID (exclusive). "
             "Defaults to last_processed_message_id from state file, or prompts operator.",
    )
    parser.add_argument(
        "--since", type=str, default=None,
        help="Only process messages on or after this ISO datetime, e.g. 2026-07-14T23:43:00",
    )
    parser.add_argument(
        "--limit", type=int, default=10000,
        help="Maximum number of messages to process (default: 10000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be inserted — no writes to disk or Supabase.",
    )
    parser.add_argument(
        "--batch-id", type=str, default=None,
        help="Override auto-generated backfill_batch_id.",
    )
    args = parser.parse_args()

    # Resolve min_id
    state = _load_state()
    min_id = args.min_id
    if min_id is None:
        min_id = state.get("last_processed_message_id")
    if min_id is None:
        try:
            raw = input(
                "No known last message ID. Enter approximate Telegram message ID to start from "
                "(or press Enter to start from the very beginning — may be slow): "
            ).strip()
            min_id = int(raw) if raw else 0
        except (ValueError, EOFError):
            min_id = 0

    # Resolve batch_id
    now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    batch_id = args.batch_id or f"backfill_{now_str}"

    # Resolve --since
    since_dt: datetime | None = None
    if args.since:
        try:
            since_dt = datetime.fromisoformat(args.since.replace("Z", "+00:00"))
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=timezone.utc)
        except ValueError as e:
            print(f"ERROR: invalid --since value: {e}", file=sys.stderr)
            sys.exit(1)

    log.info(
        "Starting backfill: channel=%s min_id=%d limit=%d dry_run=%s batch_id=%s",
        _CHANNEL, min_id, args.limit, args.dry_run, batch_id,
    )

    asyncio.run(_run_backfill(
        min_id=min_id,
        limit=args.limit,
        dry_run=args.dry_run,
        batch_id=batch_id,
        since_dt=since_dt,
    ))


if __name__ == "__main__":
    main()
