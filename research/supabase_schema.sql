-- Research pipeline schema — source of truth
-- Run once to create; for live databases run the ALTER TABLE migration block below.
--
-- ── MIGRATION (run on existing DB) ──────────────────────────────────────────
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS pp_vsol              FLOAT;
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS pp_snapshot_ok       BOOL;
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS top10_holder_pct     FLOAT;
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS creator_holds_pct    FLOAT;
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS price_t1m            FLOAT;
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS price_peak_3m        FLOAT;
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS pct_change_peak_3m   FLOAT;
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS t_peak_3m_s          INT;
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS pct_change_t1m       FLOAT;
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS pct_change_t3m       FLOAT;
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS pct_change_t5m       FLOAT;
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS pct_change_t10m      FLOAT;
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS pct_change_t15m      FLOAT;
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS pct_change_t20m      FLOAT;
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS pct_change_t30m      FLOAT;
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS data_partial          BOOL  DEFAULT FALSE;
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS v7_traded             BOOL  DEFAULT FALSE;
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS v7_traded_at          TIMESTAMPTZ;
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS code_version          TEXT;
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS channel_velocity_5m   INT;
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS tg_message_text       TEXT;
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS snapshot_attempts     INT   DEFAULT 0;
-- ── W2 migration (2026-07-18) ────────────────────────────────────────────────
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS smart_money_hit       BOOL;
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS smart_money_count     INT;
-- ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS progress_at_signal    FLOAT;
-- ── PC1 migration (2026-07-18) ───────────────────────────────────────────────
-- DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN path_file TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
-- ── RF0 backfill provenance migration (2026-07-28) ───────────────────────────
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN backfilled BOOL DEFAULT FALSE; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN source TEXT DEFAULT 'telegram_live'; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN event_id TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN backfill_batch_id TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
-- ── RF1 migration (2026-07-28) — per-interval provenance ─────────────────────
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN price_source_t1m TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN price_status_t1m TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN price_observed_at_t1m TIMESTAMPTZ; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN price_source_t3m TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN price_status_t3m TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN price_observed_at_t3m TIMESTAMPTZ; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN price_source_t5m TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN price_status_t5m TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN price_observed_at_t5m TIMESTAMPTZ; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN price_source_t10m TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN price_status_t10m TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN price_observed_at_t10m TIMESTAMPTZ; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN price_source_t15m TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN price_status_t15m TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN price_observed_at_t15m TIMESTAMPTZ; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN price_source_t20m TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN price_status_t20m TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN price_observed_at_t20m TIMESTAMPTZ; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN price_source_t30m TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN price_status_t30m TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN price_observed_at_t30m TIMESTAMPTZ; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN venue_state_final TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
-- ── RF3 migration (2026-07-28) — tiered watch window metadata ────────────────
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN path_extension_count  INT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN path_stop_reason      TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN path_watch_duration_s INT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN path_valid_tick_count INT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
-- ── RF4 migration (2026-07-28) — realert tracking ────────────────────────────
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN realert_count        INT DEFAULT 0; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN realert_times        JSONB DEFAULT '[]'; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN realert_message_ids  JSONB DEFAULT '[]'; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN last_realert_time    TIMESTAMPTZ; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
-- ── RF6 migration (2026-07-28) — versioned smart-money scoring ───────────────
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN smart_money_hit_v1 BOOL; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN smart_money_count_v1 INT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN smart_money_data_ok_v1 BOOL; EXCEPTION WHEN duplicate_column THEN NULL; END $$;

-- ── PROGRESS-FIX PF8 migration (2026-08-08) ───────────────────────────────────
-- Source-provenanced progress capture (see memecoin/progress_capture.py).
-- progress_at_signal/pp_vsol are RETAINED for compatibility (pp_vsol was
-- previously the only source; progress_at_signal's meaning is now
-- vsol_at_signal / GRAD_SOL_UI regardless of which source produced it).
-- pp_snapshot_ok's semantics are also fixed here (PF7): it must only be
-- TRUE when a genuine PumpPortal observation existed, never for a
-- freshly-created, never-updated ScreeningState.
--
-- NOTE: this repo has no DDL-execution path from application code (no
-- DATABASE_URL / direct Postgres connection configured, only the
-- PostgREST-based SUPABASE_URL/SUPABASE_KEY, which cannot run ALTER TABLE).
-- Apply this block manually via the Supabase SQL editor. Until applied,
-- research/tracker.py's existing PGRST204-retry-and-strip logic means the
-- new fields are silently dropped on INSERT rather than failing — degraded
-- but not broken.
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN vsol_at_signal          FLOAT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN progress_source         TEXT;  EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN progress_observed_at    TIMESTAMPTZ; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN progress_capture_lag_ms INT;   EXCEPTION WHEN duplicate_column THEN NULL; END $$;
-- PF8 correction (2026-08-08, found during PF12 live verification): the
-- code always produces a float (ProgressCapture.progress_capture_lag_ms =
-- round((now - alert_ts) * 1000, 1), e.g. 451.9) -- INT rejects any
-- fractional value ("invalid input syntax for type integer"), which was
-- silently stuck via research/spool's PGRST204 handling (that path only
-- catches unknown-column errors, not type-mismatch errors, so these rows
-- kept failing with max_retries_exceeded). Idempotent to re-run.
ALTER TABLE research_tokens ALTER COLUMN progress_capture_lag_ms TYPE FLOAT;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN progress_status         TEXT;  EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN progress_data_ok        BOOL;  EXCEPTION WHEN duplicate_column THEN NULL; END $$;
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN progress_schema_version INT;   EXCEPTION WHEN duplicate_column THEN NULL; END $$;
-- event_id already exists (RF0 migration, line ~36) — reused as-is for PF5's
-- event-keyed identity, not redefined here.
CREATE INDEX IF NOT EXISTS idx_rt_event_id ON research_tokens (event_id);
-- ── END MIGRATION ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS research_tokens (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    token_address           TEXT NOT NULL,
    symbol                  TEXT,
    chain                   TEXT NOT NULL DEFAULT 'solana',
    alert_time              TIMESTAMPTZ NOT NULL,
    category                TEXT,   -- 'social_alert_bc' | 'social_alert_grad' | 'unknown'

    -- ── Context fields (point-in-time, cannot reconstruct later) ─────────────
    tg_message_text         TEXT,           -- raw TG alert text (first 500 chars)
    channel_velocity_5m     INT,            -- tokens logged in last 5 min at alert time
    code_version            TEXT,           -- research pipeline version at ingest time

    -- ── T=0 market snapshot ──────────────────────────────────────────────────
    snapshot_ok             BOOL DEFAULT FALSE,
    snapshot_attempts       INT  DEFAULT 0,
    price_usd               FLOAT,
    mcap_usd                FLOAT,
    liquidity_usd           FLOAT,
    fdv                     FLOAT,
    age_minutes             FLOAT,
    volume_5m               FLOAT,
    volume_1h               FLOAT,
    buys_5m                 INT,
    sells_5m                INT,
    buy_sell_ratio_5m       FLOAT,
    buys_1h                 INT,
    sells_1h                INT,
    buy_sell_ratio_1h       FLOAT,
    price_change_5m         FLOAT,
    price_change_1h         FLOAT,
    price_change_6h         FLOAT,

    -- ── Safety / social ───────────────────────────────────────────────────────
    dex_id                  TEXT,
    has_twitter             BOOL,
    has_telegram            BOOL,
    has_website             BOOL,
    rugcheck_score          FLOAT,
    mint_disabled           BOOL,
    freeze_disabled         BOOL,
    top10_holder_pct        FLOAT,
    creator_holds_pct       FLOAT,

    -- ── PumpPortal realtime fields ────────────────────────────────────────────
    pp_vsol                 FLOAT,          -- vSolInBondingCurve at alert time
    pp_snapshot_ok          BOOL,           -- TRUE if PP data was merged
    progress_at_signal      FLOAT,          -- pp_vsol / 115 (0→1, bonding-curve completion)

    -- ── Smart-money features (W2, backfilled from Helius early-buyer data) ───
    smart_money_hit         BOOL,           -- TRUE if ≥1 smart-wallet in first 30 buyers
    smart_money_count       INT,            -- count of smart wallets in first 30 buyers

    -- ── Outcome poll prices (NULL = polled but no price; never set to 0.0) ────
    price_t1m               FLOAT,
    price_t3m               FLOAT,
    price_t5m               FLOAT,
    price_t10m              FLOAT,
    price_t15m              FLOAT,
    price_t20m              FLOAT,
    price_t30m              FLOAT,

    -- ── RF1 per-interval provenance ──────────────────────────────────────────
    -- source: "curve_account" | "dexscreener" | "jupiter" | NULL
    -- status: failure_reason (e.g. "curve_rpc_error") or NULL if success
    -- observed_at: UTC timestamp when the poll fired
    price_source_t1m        TEXT,
    price_status_t1m        TEXT,
    price_observed_at_t1m   TIMESTAMPTZ,
    price_source_t3m        TEXT,
    price_status_t3m        TEXT,
    price_observed_at_t3m   TIMESTAMPTZ,
    price_source_t5m        TEXT,
    price_status_t5m        TEXT,
    price_observed_at_t5m   TIMESTAMPTZ,
    price_source_t10m       TEXT,
    price_status_t10m       TEXT,
    price_observed_at_t10m  TIMESTAMPTZ,
    price_source_t15m       TEXT,
    price_status_t15m       TEXT,
    price_observed_at_t15m  TIMESTAMPTZ,
    price_source_t20m       TEXT,
    price_status_t20m       TEXT,
    price_observed_at_t20m  TIMESTAMPTZ,
    price_source_t30m       TEXT,
    price_status_t30m       TEXT,
    price_observed_at_t30m  TIMESTAMPTZ,
    venue_state_final       TEXT,           -- CURVE_ACTIVE | GRADUATED | UNKNOWN

    -- ── Tick-level peak (15-min window from PumpPortal WebSocket) ───────────
    price_peak_3m           FLOAT,
    pct_change_peak_3m      FLOAT,
    t_peak_3m_s             INT,

    -- ── Trade-path file (PC1) ─────────────────────────────────────────────
    path_file               TEXT,             -- relative path: logs/research_paths/YYYY-MM-DD/<mint>.csv

    -- ── Derived outcomes (computed when outcome_complete fires) ───────────────
    pct_change_t1m          FLOAT,
    pct_change_t3m          FLOAT,
    pct_change_t5m          FLOAT,
    pct_change_t10m         FLOAT,
    pct_change_t15m         FLOAT,
    pct_change_t20m         FLOAT,
    pct_change_t30m         FLOAT,
    pct_change_peak         FLOAT,          -- max % gain across all poll intervals
    peak_interval           TEXT,           -- e.g. 'T10m'
    time_to_peak_min        FLOAT,
    outcome_complete        BOOL DEFAULT FALSE,
    data_partial            BOOL DEFAULT FALSE,  -- TRUE if any expected poll was NULL at finalize

    -- ── Backfill provenance ───────────────────────────────────────────────────
    backfilled              BOOL DEFAULT FALSE,          -- TRUE if row came from backfill
    source                  TEXT DEFAULT 'telegram_live', -- 'telegram_live' | 'telegram_history'
    event_id                TEXT,                        -- sha256(tg:channel:msg_id:addr)[:16]
    backfill_batch_id       TEXT,                        -- e.g. 'backfill_20260728_001'

    -- ── Trading bot overlap ───────────────────────────────────────────────────
    v7_traded               BOOL DEFAULT FALSE,
    v7_traded_at            TIMESTAMPTZ,

    created_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rt_alert_time    ON research_tokens (alert_time);
CREATE INDEX IF NOT EXISTS idx_rt_category      ON research_tokens (category);
CREATE INDEX IF NOT EXISTS idx_rt_snapshot_ok   ON research_tokens (snapshot_ok);
CREATE INDEX IF NOT EXISTS idx_rt_outcome       ON research_tokens (outcome_complete);
CREATE INDEX IF NOT EXISTS idx_rt_token_address ON research_tokens (token_address);
CREATE UNIQUE INDEX IF NOT EXISTS idx_rt_token_day
    ON research_tokens (token_address, DATE(alert_time));


-- ── Outcome poll log ──────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS research_outcome_polls (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    token_address   TEXT NOT NULL,
    interval_label  TEXT NOT NULL,
    scheduled_at    TIMESTAMPTZ,
    polled_at       TIMESTAMPTZ,
    price_usd       FLOAT,
    mcap_usd        FLOAT,
    liquidity_usd   FLOAT,
    late            BOOL DEFAULT FALSE,
    error           TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rop_token ON research_outcome_polls (token_address);
CREATE INDEX IF NOT EXISTS idx_rop_sched ON research_outcome_polls (scheduled_at);
