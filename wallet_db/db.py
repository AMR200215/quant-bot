"""
Wallet intelligence database — SQLite with WAL mode.
Single connection factory used by all wallet_db modules.

DB file: data/wallet_intelligence.db (next to memecoin/data/)
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "memecoin" / "data" / "wallet_intelligence.db"

_SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS wallets (
    address         TEXT NOT NULL,
    chain           TEXT NOT NULL,
    name            TEXT DEFAULT '',
    source          TEXT DEFAULT 'initial',
    first_seen_ts   INTEGER NOT NULL,
    last_trade_ts   INTEGER,
    current_tier    TEXT,
    current_score   REAL DEFAULT 0.0,
    cluster_id      INTEGER,
    status          TEXT NOT NULL DEFAULT 'active',
    PRIMARY KEY (address, chain)
);

CREATE TABLE IF NOT EXISTS wallet_trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    wallet_address  TEXT NOT NULL,
    chain           TEXT NOT NULL,
    token_address   TEXT NOT NULL,
    side            TEXT NOT NULL,
    token_amount    REAL,
    native_amount   REAL,
    usd_value       REAL,
    price_at_trade  REAL,
    block_time      INTEGER NOT NULL,
    tx_hash         TEXT NOT NULL UNIQUE,
    FOREIGN KEY (wallet_address, chain) REFERENCES wallets(address, chain)
);

CREATE TABLE IF NOT EXISTS token_outcomes (
    token_address   TEXT NOT NULL,
    chain           TEXT NOT NULL,
    launch_time     INTEGER,
    launch_price    REAL,
    peak_price      REAL,
    peak_time       INTEGER,
    peak_multiple   REAL,
    current_price   REAL,
    status          TEXT DEFAULT 'active',
    last_updated    INTEGER NOT NULL,
    PRIMARY KEY (token_address, chain)
);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    primary_wallet  TEXT,
    member_count    INTEGER DEFAULT 0,
    score           REAL DEFAULT 0.0,
    last_updated    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS discovery_queue (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    wallet_address  TEXT NOT NULL,
    chain           TEXT NOT NULL,
    source          TEXT NOT NULL,
    discovered_at   INTEGER NOT NULL,
    evaluated       INTEGER DEFAULT 0,
    evaluation_result TEXT
);

CREATE TABLE IF NOT EXISTS wallet_scores_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    wallet_address  TEXT NOT NULL,
    chain           TEXT NOT NULL,
    score_date      TEXT NOT NULL,
    tier            TEXT,
    hit_rate        REAL,
    median_return   REAL,
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


def get_conn() -> sqlite3.Connection:
    """Return a WAL-mode connection. Caller is responsible for closing."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create schema if not already present. Safe to call multiple times."""
    conn = get_conn()
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()


def get_active_wallets(chain: str = "solana", tiers=("S", "A", "B")) -> list[dict]:
    """
    Return active wallets at the given tiers.
    This is the integration point — replaces reading whale_wallets_sol.json.
    Returns list of dicts with 'address', 'name', 'current_tier'.
    """
    conn = get_conn()
    placeholders = ",".join("?" * len(tiers))
    rows = conn.execute(
        f"""
        SELECT address, name, current_tier, current_score
        FROM wallets
        WHERE chain = ? AND status = 'active' AND current_tier IN ({placeholders})
        ORDER BY current_score DESC
        """,
        (chain, *tiers),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    print(f"Database initialised at {DB_PATH}")
