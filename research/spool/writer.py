"""
Spool writer — append-only local durability for data that Supabase rejected.

Usage:
    from research.spool.writer import spool_dropped_field, spool_failed_insert

Called from tracker.py, peak_tracker.py, outcome_poller.py whenever a field
is stripped or a row insert fails.  Never raises — spool failure itself is
logged but never crashes the caller.
"""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

_SPOOL_DIR            = Path(__file__).parent
_DROPPED_FIELDS_PATH  = _SPOOL_DIR / "dropped_fields.jsonl"
_FAILED_INSERTS_PATH  = _SPOOL_DIR / "failed_inserts.jsonl"

# Serial lock — spool files are written from multiple threads
_lock = threading.Lock()

# CODE_VERSION — set once at import; callers can override via set_code_version()
_CODE_VERSION = "research-2026-07-03"

# Daily summary counters  {column: count}  — reset at midnight
_daily_dropped: dict = {}
_daily_date:    str  = ""


def set_code_version(v: str):
    global _CODE_VERSION
    _CODE_VERSION = v


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(path: Path, obj: dict):
    """Append one JSON line to path.  Thread-safe."""
    try:
        line = json.dumps(obj, default=str) + "\n"
        with _lock:
            with open(path, "a") as fh:
                fh.write(line)
    except Exception as e:
        log.error("Spool write failed (%s): %s", path.name, e)


# ── Daily summary ─────────────────────────────────────────────────────────────

def _record_daily(column: str):
    global _daily_date, _daily_dropped
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if today != _daily_date:
        if _daily_dropped:
            log.warning(
                "Spool daily summary (%s): %d dropped-field events by column: %s",
                _daily_date, sum(_daily_dropped.values()),
                {k: v for k, v in sorted(_daily_dropped.items(), key=lambda x: -x[1])},
            )
        _daily_date    = today
        _daily_dropped = {}
    _daily_dropped[column] = _daily_dropped.get(column, 0) + 1


# ── Public API ────────────────────────────────────────────────────────────────

def spool_dropped_field(
    *,
    token_address: str,
    symbol: str,
    table: str,
    column: str,
    value,
    source_file: str,
    insert_context: str,   # 'base_row' | 'peak_update' | 'finalize' | 'outcome'
    alert_time: str = "",  # ISO string; used by replay_spool to target exact row
    row_id=None,           # Supabase row id; preferred over token+date for replay
):
    """
    Record one field that was stripped because the Supabase column didn't exist.
    Logs an immediate WARNING and appends to dropped_fields.jsonl.

    alert_time and row_id are used by replay_spool.py to target the exact row.
    Pass them where available — at minimum token_address + alert_time is required
    for idempotent replay; row_id is preferred when known.
    """
    log.warning(
        "SPOOL dropped_field: table=%s col=%s token=%s ctx=%s  (will replay after schema fix)",
        table, column, token_address[:12], insert_context,
    )
    _record_daily(column)
    _append(_DROPPED_FIELDS_PATH, {
        "ts":             _ts(),
        "token_address":  token_address,
        "symbol":         symbol,
        "table":          table,
        "column":         column,
        "value":          value,
        "source_file":    source_file,
        "code_version":   _CODE_VERSION,
        "insert_context": insert_context,
        "alert_time":     alert_time,
        "row_id":         row_id,
    })


def spool_failed_insert(
    *,
    token_address: str,
    symbol: str,
    table: str,
    row: dict,
    error: str,
    source_file: str,
    insert_context: str = "base_row",
    alert_time: str = "",
):
    """
    Record a full row that could not be inserted at all (after all retries).
    Logs an immediate ERROR and appends to failed_inserts.jsonl.
    """
    log.error(
        "SPOOL failed_insert: table=%s token=%s err=%s  (row spooled for replay)",
        table, token_address[:12], error[:120],
    )
    _append(_FAILED_INSERTS_PATH, {
        "ts":             _ts(),
        "token_address":  token_address,
        "symbol":         symbol,
        "table":          table,
        "row":            row,
        "error":          error,
        "source_file":    source_file,
        "code_version":   _CODE_VERSION,
        "insert_context": insert_context,
        "alert_time":     alert_time,
    })


def dropped_field_count(since_date: str = "") -> int:
    """Count rows in dropped_fields.jsonl, optionally filtered by date prefix."""
    return _count_spool(_DROPPED_FIELDS_PATH, since_date)


def failed_insert_count(since_date: str = "") -> int:
    """Count rows in failed_inserts.jsonl, optionally filtered by date prefix."""
    return _count_spool(_FAILED_INSERTS_PATH, since_date)


def _count_spool(path: Path, since_date: str = "") -> int:
    if not path.exists():
        return 0
    count = 0
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if since_date:
                    try:
                        ts = json.loads(line).get("ts", "")
                        if ts < since_date:
                            continue
                    except Exception:
                        pass
                count += 1
    except Exception:
        pass
    return count
