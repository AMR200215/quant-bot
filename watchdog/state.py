"""watchdog/state.py — durable storage layer.

SQLite in WAL mode. All writes go through explicit transactions (BEGIN
IMMEDIATE ... COMMIT) so a crash mid-write leaves the previous committed
row intact rather than a half-written one — this is the "atomic rename or
SQLite transaction semantics" requirement from the design spec.

This module is pure storage: it has no opinion about incident lifecycle
policy (that's watchdog/notifier.py) or what a healthy check looks like
(that's watchdog/checks/*). It only records what happened.
"""

from __future__ import annotations

import fcntl
import json
import os
import socket
import sqlite3
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = REPO_ROOT / "logs" / "watchdog" / "state.db"
DEFAULT_LOCK_PATH = REPO_ROOT / "logs" / "watchdog" / "runner.lock"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS watchdog_runs (
    run_id TEXT PRIMARY KEY,
    host TEXT NOT NULL,
    boot_id TEXT,
    profile TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    duration_ms REAL,
    git_sha TEXT,
    watchdog_version TEXT,
    checks_due INTEGER,
    checks_completed INTEGER,
    final_runner_status TEXT
);

CREATE TABLE IF NOT EXISTS check_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    check_id TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT,
    evidence TEXT,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_check_results_check_id ON check_results(check_id, ts);

CREATE TABLE IF NOT EXISTS incidents (
    fingerprint TEXT PRIMARY KEY,
    check_id TEXT NOT NULL,
    subject TEXT,
    severity TEXT NOT NULL,
    state TEXT NOT NULL,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    last_notified REAL,
    last_evidence TEXT,
    recovered_at REAL
);

CREATE TABLE IF NOT EXISTS job_receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    trigger_type TEXT NOT NULL,
    scheduled_for REAL,
    started_at REAL NOT NULL,
    finished_at REAL,
    exit_code INTEGER,
    duration_ms REAL,
    git_sha TEXT,
    output_artifact TEXT,
    run_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_job_receipts_job_id ON job_receipts(job_id, started_at);

CREATE TABLE IF NOT EXISTS digest_state (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def now() -> float:
    return time.time()


def new_run_id() -> str:
    return uuid.uuid4().hex[:16]


def get_git_sha(cwd: Optional[Path] = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd or REPO_ROOT,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except Exception:
        return "unknown"


def get_host() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=10000;")
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)


# ---------------------------------------------------------------------------
# Singleton lock (W1: no overlapping watchdog executions)
# ---------------------------------------------------------------------------

class SingletonLock:
    """Non-blocking flock-based singleton. Returns None from acquire() if
    another instance already holds the lock — caller must treat that as
    UNKNOWN/SKIPPED, never as a silent success."""

    def __init__(self, lock_path: Optional[Path] = None):
        self.lock_path = Path(lock_path) if lock_path else DEFAULT_LOCK_PATH
        self._fd = None

    def acquire(self) -> bool:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError(f"could not acquire singleton lock: {self.lock_path}")
        return self

    def __exit__(self, *exc):
        self.release()


# ---------------------------------------------------------------------------
# watchdog_runs (W2: self-heartbeat)
# ---------------------------------------------------------------------------

def record_run_start(conn: sqlite3.Connection, run_id: str, profile: str,
                      checks_due: int) -> None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            """INSERT INTO watchdog_runs
               (run_id, host, boot_id, profile, started_at, git_sha,
                watchdog_version, checks_due, final_runner_status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'in_progress')""",
            (run_id, get_host(), get_boot_id(), profile, now(),
             get_git_sha(), _watchdog_version(), checks_due),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def record_run_finish(conn: sqlite3.Connection, run_id: str,
                       checks_completed: int, final_runner_status: str) -> None:
    started = conn.execute(
        "SELECT started_at FROM watchdog_runs WHERE run_id=?", (run_id,)
    ).fetchone()
    finished_at = now()
    duration_ms = (finished_at - started["started_at"]) * 1000 if started else None
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            """UPDATE watchdog_runs SET finished_at=?, duration_ms=?,
               checks_completed=?, final_runner_status=? WHERE run_id=?""",
            (finished_at, duration_ms, checks_completed, final_runner_status, run_id),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def last_successful_run(conn: sqlite3.Connection, profile: Optional[str] = None) -> Optional[dict]:
    q = "SELECT * FROM watchdog_runs WHERE final_runner_status='ok'"
    args: list = []
    if profile:
        q += " AND profile=?"
        args.append(profile)
    q += " ORDER BY started_at DESC LIMIT 1"
    row = conn.execute(q, args).fetchone()
    return dict(row) if row else None


def _watchdog_version() -> str:
    try:
        from watchdog import WATCHDOG_VERSION
        return WATCHDOG_VERSION
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# check_results
# ---------------------------------------------------------------------------

def record_check_result(conn: sqlite3.Connection, run_id: str, check_id: str,
                         status: str, reason: str = "", evidence: Optional[dict] = None) -> None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            """INSERT INTO check_results (run_id, check_id, status, reason, evidence, ts)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (run_id, check_id, status, reason, json.dumps(evidence or {}), now()),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


# ---------------------------------------------------------------------------
# incidents
# ---------------------------------------------------------------------------

def get_incident(conn: sqlite3.Connection, fingerprint: str) -> Optional[dict]:
    row = conn.execute("SELECT * FROM incidents WHERE fingerprint=?", (fingerprint,)).fetchone()
    return dict(row) if row else None


def list_active_incidents(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM incidents WHERE state IN ('SUSPECT','FIRING') ORDER BY first_seen"
    ).fetchall()
    return [dict(r) for r in rows]


def touch_incident_seen(conn: sqlite3.Connection, fingerprint: str,
                         ts: Optional[float] = None) -> None:
    """Updates last_seen only -- for UNKNOWN evidence against an incident
    that's already FIRING. UNKNOWN is not proof of recovery, so it must
    never silently downgrade a real firing incident's state/severity or
    reset its consecutive_failures streak; it also isn't a new failure,
    so it must not increment that streak either."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("UPDATE incidents SET last_seen=? WHERE fingerprint=?",
                      (ts if ts is not None else now(), fingerprint))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def upsert_incident(conn: sqlite3.Connection, fingerprint: str, check_id: str,
                     subject: str, severity: str, state: str,
                     evidence: Optional[dict] = None, reset_consecutive: bool = False,
                     touch_notified: bool = False, recovered: bool = False,
                     ts: Optional[float] = None) -> dict:
    """Insert or update an incident row. Caller (notifier.py) decides the
    state-transition policy; this just persists it durably.

    `ts` should be the caller's own notion of "now" (e.g. notifier.py's
    now_ts) rather than defaulting silently to a fresh wall-clock read --
    otherwise reminder-interval math done against a caller-supplied clock
    (tests, or any future replay/backfill use) silently compares against
    the wrong clock."""
    ts = ts if ts is not None else now()
    conn.execute("BEGIN IMMEDIATE")
    try:
        existing = conn.execute(
            "SELECT * FROM incidents WHERE fingerprint=?", (fingerprint,)
        ).fetchone()
        ev_json = json.dumps(evidence or {})
        if existing is None:
            consecutive = 0 if reset_consecutive else 1
            conn.execute(
                """INSERT INTO incidents
                   (fingerprint, check_id, subject, severity, state, first_seen,
                    last_seen, consecutive_failures, last_notified, last_evidence, recovered_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (fingerprint, check_id, subject, severity, state, ts, ts,
                 consecutive, ts if touch_notified else None, ev_json,
                 ts if recovered else None),
            )
        else:
            consecutive = 0 if reset_consecutive else existing["consecutive_failures"] + 1
            last_notified = ts if touch_notified else existing["last_notified"]
            recovered_at = ts if recovered else existing["recovered_at"]
            conn.execute(
                """UPDATE incidents SET check_id=?, subject=?, severity=?, state=?,
                   last_seen=?, consecutive_failures=?, last_notified=?,
                   last_evidence=?, recovered_at=? WHERE fingerprint=?""",
                (check_id, subject, severity, state, ts, consecutive, last_notified,
                 ev_json, recovered_at, fingerprint),
            )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return get_incident(conn, fingerprint)


# ---------------------------------------------------------------------------
# job_receipts (W5B: proof a scheduled job actually ran, not just that its
# output artifact's mtime looks fresh)
# ---------------------------------------------------------------------------

def record_job_receipt(conn: sqlite3.Connection, job_id: str, trigger_type: str,
                        started_at: float, finished_at: Optional[float],
                        exit_code: Optional[int], scheduled_for: Optional[float] = None,
                        output_artifact: str = "", run_id: str = "") -> None:
    assert trigger_type in ("scheduler", "manual"), trigger_type
    duration_ms = (finished_at - started_at) * 1000 if finished_at else None
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            """INSERT INTO job_receipts
               (job_id, trigger_type, scheduled_for, started_at, finished_at,
                exit_code, duration_ms, git_sha, output_artifact, run_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (job_id, trigger_type, scheduled_for, started_at, finished_at,
             exit_code, duration_ms, get_git_sha(), output_artifact, run_id),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def get_latest_job_receipt(conn: sqlite3.Connection, job_id: str,
                            trigger_type: Optional[str] = None) -> Optional[dict]:
    q = "SELECT * FROM job_receipts WHERE job_id=?"
    args: list[Any] = [job_id]
    if trigger_type:
        q += " AND trigger_type=?"
        args.append(trigger_type)
    q += " ORDER BY started_at DESC LIMIT 1"
    row = conn.execute(q, args).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# digest_state (tracks "did we already send today's digest")
# ---------------------------------------------------------------------------

def get_digest_marker(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM digest_state WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_digest_marker(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "INSERT INTO digest_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
