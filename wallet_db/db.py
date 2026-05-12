"""
Wallet intelligence database.

Uses PostgreSQL (Supabase) in production, SQLite for local dev/testing.
Set DATABASE_URL env var to switch:
  - Not set / "sqlite"  → SQLite at memecoin/data/wallet_intelligence.db
  - postgresql://...    → Postgres (Supabase)
"""

import os
import sqlite3
from pathlib import Path
from typing import Any

DATABASE_URL = os.getenv("DATABASE_URL", "")
_USE_POSTGRES = DATABASE_URL.startswith("postgresql") or DATABASE_URL.startswith("postgres")

SQLITE_PATH = Path(__file__).parent.parent / "memecoin" / "data" / "wallet_intelligence.db"

# ---------------------------------------------------------------------------
# Schema (Postgres-flavoured; SQLite adapter translates where needed)
# ---------------------------------------------------------------------------

_PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS wallets (
    address         TEXT NOT NULL,
    chain           TEXT NOT NULL,
    name            TEXT DEFAULT '',
    source          TEXT DEFAULT 'initial',
    first_seen_ts   BIGINT NOT NULL,
    last_trade_ts   BIGINT,
    current_tier    TEXT,
    current_score   DOUBLE PRECISION DEFAULT 0.0,
    cluster_id      INTEGER,
    status          TEXT NOT NULL DEFAULT 'active',
    PRIMARY KEY (address, chain)
);

CREATE TABLE IF NOT EXISTS wallet_trades (
    id              BIGSERIAL PRIMARY KEY,
    wallet_address  TEXT NOT NULL,
    chain           TEXT NOT NULL,
    token_address   TEXT NOT NULL,
    side            TEXT NOT NULL,
    token_amount    DOUBLE PRECISION,
    native_amount   DOUBLE PRECISION,
    usd_value       DOUBLE PRECISION,
    price_at_trade  DOUBLE PRECISION,
    block_time      BIGINT NOT NULL,
    tx_hash         TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS token_outcomes (
    token_address   TEXT NOT NULL,
    chain           TEXT NOT NULL,
    launch_time     BIGINT,
    launch_price    DOUBLE PRECISION,
    peak_price      DOUBLE PRECISION,
    peak_time       BIGINT,
    peak_multiple   DOUBLE PRECISION,
    current_price   DOUBLE PRECISION,
    status          TEXT DEFAULT 'active',
    last_updated    BIGINT NOT NULL,
    PRIMARY KEY (token_address, chain)
);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id      SERIAL PRIMARY KEY,
    primary_wallet  TEXT,
    member_count    INTEGER DEFAULT 0,
    score           DOUBLE PRECISION DEFAULT 0.0,
    last_updated    BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS discovery_queue (
    id              BIGSERIAL PRIMARY KEY,
    wallet_address  TEXT NOT NULL,
    chain           TEXT NOT NULL,
    source          TEXT NOT NULL,
    context         TEXT DEFAULT '',
    discovered_at   BIGINT NOT NULL,
    evaluated       INTEGER DEFAULT 0,
    evaluation_result TEXT
);

CREATE TABLE IF NOT EXISTS wallet_scores_history (
    id              BIGSERIAL PRIMARY KEY,
    wallet_address  TEXT NOT NULL,
    chain           TEXT NOT NULL,
    score_date      TEXT NOT NULL,
    tier            TEXT,
    hit_rate        DOUBLE PRECISION,
    median_return   DOUBLE PRECISION,
    trade_count     INTEGER,
    UNIQUE (wallet_address, chain, score_date)
);

CREATE INDEX IF NOT EXISTS idx_trades_wallet_time
    ON wallet_trades (wallet_address, chain, block_time DESC);

CREATE INDEX IF NOT EXISTS idx_trades_token_time
    ON wallet_trades (token_address, chain, block_time);

CREATE INDEX IF NOT EXISTS idx_wallets_status_tier
    ON wallets (status, current_tier);

CREATE INDEX IF NOT EXISTS idx_wallets_last_trade
    ON wallets (last_trade_ts DESC);

CREATE INDEX IF NOT EXISTS idx_outcomes_peak
    ON token_outcomes (chain, peak_multiple DESC);

CREATE INDEX IF NOT EXISTS idx_discovery_pending
    ON discovery_queue (evaluated, discovered_at);
"""

_SQLITE_SCHEMA = _PG_SCHEMA \
    .replace("BIGSERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT") \
    .replace("SERIAL PRIMARY KEY",    "INTEGER PRIMARY KEY AUTOINCREMENT") \
    .replace("BIGINT",                "INTEGER") \
    .replace("DOUBLE PRECISION",      "REAL")


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

class _PgConn:
    """Thin wrapper so Postgres behaves like our sqlite3 usage."""

    def __init__(self, dsn: str):
        import psycopg2
        import psycopg2.extras
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False

    def execute(self, sql: str, params=()) -> Any:
        sql = self._pg_sql(sql)
        cur = self._conn.cursor(cursor_factory=__import__("psycopg2").extras.RealDictCursor)
        cur.execute(sql, params)
        return cur

    def executemany(self, sql: str, seq):
        sql = self._pg_sql(sql)
        cur = self._conn.cursor()
        cur.executemany(sql, seq)

    def executescript(self, sql: str):
        """Run a multi-statement SQL block."""
        cur = self._conn.cursor()
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    cur.execute(stmt)
                except Exception:
                    pass   # IF NOT EXISTS makes this safe

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()

    @staticmethod
    def _pg_sql(sql: str) -> str:
        """Convert SQLite ? placeholders to Postgres %s."""
        return sql.replace("?", "%s") \
                  .replace("INSERT OR IGNORE INTO", "INSERT INTO") \
                  .replace("INSERT OR REPLACE INTO", "INSERT INTO") \
                  .replace("ON CONFLICT(", "ON CONFLICT (")


def get_conn():
    """Return a DB connection. Caller is responsible for closing."""
    if _USE_POSTGRES:
        return _PgConn(DATABASE_URL)
    else:
        SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(SQLITE_PATH), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn


def init_db():
    """Create schema. Safe to call multiple times."""
    conn = get_conn()
    schema = _PG_SCHEMA if _USE_POSTGRES else _SQLITE_SCHEMA
    if _USE_POSTGRES:
        conn.executescript(schema)
    else:
        conn.executescript(schema)
    conn.commit()
    conn.close()


def get_active_wallets(chain: str = "solana", tiers=("S", "A", "B")) -> list[dict]:
    """
    Return active tiered wallets.
    Integration point — will replace whale_wallets_sol.json in Phase 7.
    """
    conn = get_conn()
    placeholders = ",".join(["%s" if _USE_POSTGRES else "?"] * len(tiers))
    rows = conn.execute(
        f"""
        SELECT address, name, current_tier, current_score
        FROM wallets
        WHERE chain = {'%s' if _USE_POSTGRES else '?'}
          AND status = 'active'
          AND current_tier IN ({placeholders})
        ORDER BY current_score DESC
        """,
        (chain, *tiers),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    backend = "Postgres (Supabase)" if _USE_POSTGRES else f"SQLite ({SQLITE_PATH})"
    print(f"Database initialised — {backend}")
