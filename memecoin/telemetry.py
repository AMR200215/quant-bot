"""
memecoin/telemetry.py — Unified trade telemetry black-box recorder.

Public API:
    start_trace(pos_id, mint, symbol, live_or_paper, **fields) -> trace_id
    event(trace_id, event_name, **fields)
    finish_trace(trace_id, **fields)
    summarize_trace(trace_id) -> dict | None
    get_trace_id_for_pos(pos_id) -> str | None

Storage:
    logs/trade_telemetry.jsonl   — append-only canonical event log
    logs/trade_telemetry_summary.csv  — optional per-trace summary (written at finish_trace)

Config (read from memecoin/config.py or defaults):
    TELEMETRY_ENABLED = True
    TELEMETRY_HEARTBEAT_SEC = 10
    TELEMETRY_FILE = logs/trade_telemetry.jsonl

Schema version: "1.0"
"""

import csv
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — read from config.py or use defaults
# ---------------------------------------------------------------------------
try:
    from memecoin.config import TELEMETRY_ENABLED  # type: ignore
except (ImportError, AttributeError):
    TELEMETRY_ENABLED = True

try:
    from memecoin.config import TELEMETRY_HEARTBEAT_SEC  # type: ignore
except (ImportError, AttributeError):
    TELEMETRY_HEARTBEAT_SEC = 10

try:
    from memecoin.config import TELEMETRY_FILE  # type: ignore
except (ImportError, AttributeError):
    TELEMETRY_FILE = "logs/trade_telemetry.jsonl"

SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
_traces: dict[str, dict] = {}
_trace_events: dict[str, list[dict]] = {}   # trace_id → list of event dicts
_write_lock = threading.Lock()
_traces_lock = threading.Lock()

_SUMMARY_FIELDS = [
    "trace_id", "pos_id", "pair_id", "mint", "symbol", "live_or_paper",
    "started_at", "finished_at", "duration_sec", "n_events",
    "first_event", "last_event",
]


def _telemetry_path() -> Path:
    """Resolve the JSONL file path (relative to repo root)."""
    p = Path(TELEMETRY_FILE)
    if not p.is_absolute():
        p = Path(__file__).parent.parent / p
    return p


def _summary_path() -> Path:
    p = _telemetry_path()
    return p.parent / "trade_telemetry_summary.csv"


def _write_event(record: dict) -> None:
    """Append a JSON line to the telemetry file. Thread-safe, non-blocking on errors."""
    if not TELEMETRY_ENABLED:
        return
    try:
        p = _telemetry_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str) + "\n"
        with _write_lock:
            with open(p, "a") as f:
                f.write(line)
    except Exception as exc:
        log.debug("telemetry write error: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_trace(
    pos_id: str,
    mint: str,
    symbol: str,
    live_or_paper: str,
    pair_id: str | None = None,
    **fields,
) -> str:
    """Start a new trace and emit a trace_started event. Returns trace_id."""
    trace_id = f"tr_{pos_id}_{int(time.time())}"
    meta = {
        "pos_id": pos_id,
        "mint": mint,
        "symbol": symbol,
        "live_or_paper": live_or_paper,
        "pair_id": pair_id or "",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "started_monotonic": time.monotonic(),
    }
    meta.update(fields)
    with _traces_lock:
        _traces[trace_id] = meta
        _trace_events[trace_id] = []
    event(trace_id, "trace_started", **fields)
    return trace_id


def event(trace_id: str, event_name: str, **fields) -> None:
    """Emit a single telemetry event. Never raises."""
    try:
        if not TELEMETRY_ENABLED:
            return
        with _traces_lock:
            tmeta = _traces.get(trace_id, {})
        record = {
            "schema_version": SCHEMA_VERSION,
            "timestamp_wall": datetime.now(timezone.utc).isoformat() + "Z",
            "timestamp_monotonic": time.monotonic(),
            "trace_id": trace_id,
            "pair_id": tmeta.get("pair_id", ""),
            "pos_id": tmeta.get("pos_id", ""),
            "mint": tmeta.get("mint", ""),
            "symbol": tmeta.get("symbol", ""),
            "live_or_paper": tmeta.get("live_or_paper", ""),
            "event_name": event_name,
        }
        record.update(fields)
        _write_event(record)
        with _traces_lock:
            evts = _trace_events.get(trace_id)
            if evts is not None:
                evts.append(record)
    except Exception as exc:
        log.debug("telemetry event error: %s", exc)


def finish_trace(trace_id: str, **fields) -> None:
    """Emit trace_finished, write summary CSV row, remove from in-memory store."""
    try:
        event(trace_id, "trace_finished", **fields)
        _write_summary(trace_id)
        with _traces_lock:
            _traces.pop(trace_id, None)
            _trace_events.pop(trace_id, None)
    except Exception as exc:
        log.debug("telemetry finish_trace error: %s", exc)


def summarize_trace(trace_id: str) -> dict | None:
    """Return in-memory trace metadata + events. None if not found."""
    with _traces_lock:
        meta = _traces.get(trace_id)
        evts = _trace_events.get(trace_id)
    if meta is None:
        return None
    return {
        "meta": dict(meta),
        "events": list(evts) if evts else [],
    }


def get_trace_id_for_pos(pos_id: str) -> str | None:
    """Find an active trace by pos_id. Returns trace_id or None."""
    with _traces_lock:
        for tid, meta in _traces.items():
            if meta.get("pos_id") == pos_id:
                return tid
    return None


def emit_once(trace_id: str, event_key: str, event_name: str, **fields) -> bool:
    """
    Edge-trigger guard: emit event_name only if event_key has not been emitted
    for this trace. Suppresses all subsequent calls with the same key.

    Example keys:
      "exit_condition_true:hard_stop"   — once per position per exit reason
      "tp_condition_true:tp_30"         — once per TP level per position

    The emitted set is stored in _traces[trace_id]["_emitted"] and is cleared
    when finish_trace() removes the trace.

    Returns True if emitted, False if suppressed (or error).
    """
    try:
        with _traces_lock:
            meta = _traces.get(trace_id)
            if meta is None:
                return False
            emitted: set = meta.setdefault("_emitted", set())
            if event_key in emitted:
                return False
            emitted.add(event_key)
        # emit outside the lock — event() is already thread-safe
        event(trace_id, event_name, **fields)
        return True
    except Exception as exc:
        log.debug("telemetry emit_once error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_summary(trace_id: str) -> None:
    """Write a single summary row to the CSV."""
    try:
        with _traces_lock:
            meta = _traces.get(trace_id, {})
            evts = _trace_events.get(trace_id, [])
        if not meta:
            return
        now = datetime.now(timezone.utc).isoformat()
        started = meta.get("started_at", "")
        dur = time.monotonic() - meta.get("started_monotonic", time.monotonic())
        row = {
            "trace_id": trace_id,
            "pos_id": meta.get("pos_id", ""),
            "pair_id": meta.get("pair_id", ""),
            "mint": meta.get("mint", ""),
            "symbol": meta.get("symbol", ""),
            "live_or_paper": meta.get("live_or_paper", ""),
            "started_at": started,
            "finished_at": now,
            "duration_sec": round(dur, 2),
            "n_events": len(evts),
            "first_event": evts[0].get("event_name", "") if evts else "",
            "last_event": evts[-1].get("event_name", "") if evts else "",
        }
        p = _summary_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        write_header = not p.exists() or p.stat().st_size == 0
        with _write_lock:
            with open(p, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
                if write_header:
                    w.writeheader()
                w.writerow(row)
    except Exception as exc:
        log.debug("telemetry summary write error: %s", exc)
