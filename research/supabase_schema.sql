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

    -- ── Outcome poll prices (NULL = polled but no price; never set to 0.0) ────
    price_t1m               FLOAT,
    price_t3m               FLOAT,
    price_t5m               FLOAT,
    price_t10m              FLOAT,
    price_t15m              FLOAT,
    price_t20m              FLOAT,
    price_t30m              FLOAT,

    -- ── Tick-level peak (3-min window from PumpPortal WebSocket) ─────────────
    price_peak_3m           FLOAT,
    pct_change_peak_3m      FLOAT,
    t_peak_3m_s             INT,

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
