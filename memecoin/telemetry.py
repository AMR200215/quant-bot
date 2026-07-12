"""
memecoin/telemetry.py — Unified trade telemetry black-box recorder.

Public API:
    start_trace(pos_id, mint, symbol, live_or_paper, **fields) -> trace_id
    event(trace_id, event_name, **fields)
    finish_trace(trace_id, **fields)
    summarize_trace(trace_id) -> dict | None
    get_trace_id_for_pos(pos_id) -> str | None
    bind_position(trace_id, pos_id) -> None
    link_pair(trace_id, pair_id) -> None
    restore_trace_for_position(pos_id, mint, symbol, live_or_paper, pair_id) -> str

Canonical event names (use these constants, never bare strings):
    E_* constants below

Storage:
    logs/trade_telemetry.jsonl         — append-only canonical event log
    logs/trade_telemetry_summary.csv   — per-trace summary (written at finish_trace)

Config (read from memecoin/config.py or defaults):
    TELEMETRY_ENABLED = True
    TELEMETRY_HEARTBEAT_SEC = 10
    TELEMETRY_FILE = logs/trade_telemetry.jsonl
    TELEMETRY_MAX_LINES = 50000        — rotate when file exceeds this many lines
    TELEMETRY_RETENTION_DAYS = 30      — drop events older than this on rotation

Schema version: "1.1"
"""

import csv
import json
import logging
import os
import re
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
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

try:
    from memecoin.config import TELEMETRY_MAX_LINES  # type: ignore
except (ImportError, AttributeError):
    TELEMETRY_MAX_LINES = 50_000

try:
    from memecoin.config import TELEMETRY_RETENTION_DAYS  # type: ignore
except (ImportError, AttributeError):
    TELEMETRY_RETENTION_DAYS = 30

SCHEMA_VERSION = "1.1"

# ---------------------------------------------------------------------------
# Canonical event name constants  (30+ entries covering full lifecycle)
# Entry phase
E_ALERT_RECEIVED            = "alert_received"
E_PREFLIGHT_STARTED         = "preflight_started"
E_PREFLIGHT_BASELINE        = "preflight_baseline_selected"
E_PREFLIGHT_BLOCKED         = "preflight_blocked"
E_ENTRY_GATE_CHECKED        = "entry_gate_checked"
E_ENTRY_GATE_BLOCKED        = "entry_gate_blocked"
E_BUY_BUILD_STARTED         = "buy_build_started"
E_BUY_BUILD_DONE            = "buy_build_done"
E_BUY_SENT                  = "buy_sent"
E_BUY_CONFIRMED             = "buy_confirmed"
E_BUY_FILL_RECORDED         = "buy_fill_recorded"
E_BUY_FAILED                = "buy_failed"
E_ABORT_TRIPWIRE            = "abort_tripwire"
E_ABORT_SELL_SENT           = "abort_sell_sent"
E_ABORT_SELL_CONFIRMED      = "abort_sell_confirmed"
# Monitoring phase
E_PRICE_TICK                = "price_tick"
E_TP_CONDITION_TRUE         = "tp_condition_true"
E_TP_INFLIGHT               = "tp_inflight"
E_EXIT_CONDITION_TRUE       = "exit_condition_true"
E_EXIT_QUEUED               = "exit_queued"
E_EXIT_TRIGGERED            = "exit_triggered"
E_PREGRAD_THRESHOLD         = "pregrad_threshold_crossed"
E_GRAD_DETECTED             = "graduation_detected"
E_GRAD_FAST_WINDOW          = "grad_fast_window_start"
# Exit / sell phase
E_SELL_BUILD_STARTED        = "sell_build_started"
E_SELL_SENT                 = "sell_sent"
E_SELL_CONFIRMED            = "sell_confirmed"
E_SELL_FAILED               = "sell_failed"
E_SELL_STUCK                = "sell_stuck"
E_SELL_ROUTE_CHANGED        = "sell_route_changed"
# MU / recovery
E_MU_ATTEMPT                = "mu_retry_attempt"
E_MU_ESCALATE               = "mu_escalate_jupiter"
E_MU_FINAL_GATE             = "mu_final_gate"
E_MU_MANUAL_REQUIRED        = "mu_manual_required"
E_RESCUE_STARTED            = "rescue_started"
E_RESCUE_SUCCEEDED          = "rescue_succeeded"
E_RESCUE_FAILED             = "rescue_failed"
E_RECONCILED_GONE           = "reconciled_gone"
# Journal / receipt
E_JOURNAL_WRITTEN           = "journal_written"
E_JOURNAL_BACKFILLED        = "journal_backfilled"
E_RECEIPT_WRITTEN           = "receipt_written"
E_RECEIPT_PROMOTED          = "receipt_promoted"
# Lifecycle
E_TRACE_STARTED             = "trace_started"
E_TRACE_FINISHED            = "trace_finished"
E_SIZE_SHADOW               = "size_shadow"

# ---------------------------------------------------------------------------
# Secret field patterns to redact before writing
# ---------------------------------------------------------------------------
_SECRET_PATTERNS = re.compile(
    r'(?i)(private|(?:api|helius|rpc|wallet|auth|solana)_key|secret_key|seed|mnemonic|password|keypair)',
)


def _redact(record: dict) -> dict:
    """Return a copy of record with secret fields removed."""
    return {k: v for k, v in record.items() if not _SECRET_PATTERNS.search(k)}


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
_traces: dict[str, dict] = {}
_trace_events: dict[str, list[dict]] = {}   # trace_id → list of event dicts
_pos_to_trace: dict[str, str] = {}          # pos_id → trace_id index
_write_lock = threading.Lock()
_traces_lock = threading.Lock()
_line_counter = 0                            # approximate line count for rotation gate

_SUMMARY_FIELDS = [
    "trace_id", "pos_id", "pair_id", "mint", "symbol", "live_or_paper",
    "started_at", "finished_at", "duration_sec", "n_events",
    "first_event", "last_event",
]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _telemetry_path() -> Path:
    """Resolve the JSONL file path (relative to repo root)."""
    p = Path(TELEMETRY_FILE)
    if not p.is_absolute():
        p = Path(__file__).parent.parent / p
    return p


def _summary_path() -> Path:
    p = _telemetry_path()
    return p.parent / "trade_telemetry_summary.csv"


# ---------------------------------------------------------------------------
# UTC timestamp helper (fixes isoformat()+Z double-suffix bug)
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    """Return current UTC time as ISO8601 with Z suffix, no +00:00 duplication."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def _write_event(record: dict) -> None:
    """Append a redacted JSON line to the telemetry file. Thread-safe, never raises."""
    if not TELEMETRY_ENABLED:
        return
    global _line_counter
    try:
        p = _telemetry_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        safe = _redact(record)
        line = json.dumps(safe, default=str) + "\n"
        with _write_lock:
            with open(p, "a") as f:
                f.write(line)
            _line_counter += 1
            if _line_counter >= TELEMETRY_MAX_LINES:
                _maybe_rotate(p)
                _line_counter = 0
    except Exception as exc:
        log.debug("telemetry write error: %s", exc)


def _maybe_rotate(p: Path) -> None:
    """Drop lines older than TELEMETRY_RETENTION_DAYS. Called under _write_lock."""
    try:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=TELEMETRY_RETENTION_DAYS)
        ).timestamp()
        kept = []
        with open(p) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                    wall = rec.get("timestamp_wall", "")
                    ts_str = wall.rstrip("Z")
                    ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc).timestamp()
                    if ts >= cutoff:
                        kept.append(raw)
                except Exception:
                    kept.append(raw)  # keep lines we can't parse
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w") as f:
            f.write("\n".join(kept) + "\n")
        tmp.replace(p)
    except Exception as exc:
        log.debug("telemetry rotation error: %s", exc)


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
    """Start a new trace and emit trace_started. Returns collision-safe trace_id (UUID)."""
    trace_id = f"tr_{uuid.uuid4().hex}"
    meta = {
        "pos_id": pos_id,
        "mint": mint,
        "symbol": symbol,
        "live_or_paper": live_or_paper,
        "pair_id": pair_id or "",
        "started_at": _utc_now_iso(),
        "started_monotonic": time.monotonic(),
    }
    meta.update(fields)
    with _traces_lock:
        _traces[trace_id] = meta
        _trace_events[trace_id] = []
        if pos_id:
            _pos_to_trace[pos_id] = trace_id
    event(trace_id, E_TRACE_STARTED, **fields)
    return trace_id


def bind_position(trace_id: str, pos_id: str) -> None:
    """Associate pos_id with an existing trace. Used when pos_id is assigned after start_trace."""
    try:
        with _traces_lock:
            meta = _traces.get(trace_id)
            if meta is not None:
                meta["pos_id"] = pos_id
                _pos_to_trace[pos_id] = trace_id
    except Exception as exc:
        log.debug("telemetry bind_position error: %s", exc)


def link_pair(trace_id: str, pair_id: str) -> None:
    """Set pair_id on an existing trace (supports signal_id pairing for summarizer)."""
    try:
        with _traces_lock:
            meta = _traces.get(trace_id)
            if meta is not None:
                meta["pair_id"] = pair_id
    except Exception as exc:
        log.debug("telemetry link_pair error: %s", exc)


def restore_trace_for_position(
    pos_id: str,
    mint: str,
    symbol: str,
    live_or_paper: str,
    pair_id: str | None = None,
) -> str:
    """
    Recover or create a trace for a position that survived a process restart.

    If a trace is already active for pos_id, return its trace_id.
    Otherwise start a fresh trace tagged with event restore_trace.
    """
    with _traces_lock:
        existing = _pos_to_trace.get(pos_id)
    if existing and existing in _traces:
        return existing
    trace_id = start_trace(pos_id, mint, symbol, live_or_paper, pair_id=pair_id)
    event(trace_id, "trace_restored", pos_id=pos_id)
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
            "timestamp_wall": _utc_now_iso(),
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
        event(trace_id, E_TRACE_FINISHED, **fields)
        _write_summary(trace_id)
        with _traces_lock:
            meta = _traces.pop(trace_id, {})
            _trace_events.pop(trace_id, None)
            pos_id = meta.get("pos_id", "")
            if pos_id and _pos_to_trace.get(pos_id) == trace_id:
                _pos_to_trace.pop(pos_id, None)
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
        tid = _pos_to_trace.get(pos_id)
        if tid and tid in _traces:
            return tid
        # fallback linear scan (handles races where index is stale)
        for t, meta in _traces.items():
            if meta.get("pos_id") == pos_id:
                return t
    return None


def emit_once(trace_id: str, event_key: str, event_name: str, **fields) -> bool:
    """
    Edge-trigger guard: emit event_name only if event_key has not been emitted
    for this trace. Suppresses all subsequent calls with the same key.

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
        now = _utc_now_iso()
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
