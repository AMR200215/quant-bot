-- Research pipeline schema
-- Run once in Supabase SQL editor: https://supabase.com/dashboard → SQL Editor

-- ── Main token table ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS research_tokens (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    token_address           TEXT NOT NULL,
    symbol                  TEXT,
    chain                   TEXT NOT NULL DEFAULT 'solana',
    alert_time              TIMESTAMPTZ NOT NULL,
    category                TEXT,   -- 'social_alert_bc' | 'social_alert_grad' | 'unknown'

    -- T=0 market snapshot (NULL if DexScreener not indexed yet)
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

    -- safety / social (from DexScreener + rugcheck)
    dex_id                  TEXT,
    has_twitter             BOOL,
    has_telegram            BOOL,
    has_website             BOOL,
    rugcheck_score          FLOAT,   -- 0–1000, lower = safer (NULL if fetch failed)
    mint_disabled           BOOL,
    freeze_disabled         BOOL,

    -- screener fields (raw values — decisions computed at analysis time)
    -- thresholds at time of capture stored in config_snapshot column
    -- context captured at alert time (can't reconstruct later)
    symbol                  TEXT,                 -- token symbol from DexScreener (NULL if snap_ok=False)
    tg_message_text         TEXT,                 -- raw TG message text (first 500 chars)
    channel_velocity_5m     INT,                  -- # of tokens logged in last 5 min at alert time

    snapshot_ok             BOOL DEFAULT FALSE,   -- TRUE once DexScreener returned data
    snapshot_attempts       INT DEFAULT 0,        -- how many DexScreener retries before data

    -- outcomes — NULL until outcome_poller fills them
    -- social_alert_bc windows: T+3m, T+5m, T+10m, T+20m
    price_T3m               FLOAT,
    price_T5m               FLOAT,
    price_T10m              FLOAT,
    price_T20m              FLOAT,
    -- social_alert_grad windows: T+15m, T+30m
    price_T15m              FLOAT,
    price_T30m              FLOAT,

    -- derived outcomes (computed when outcome_complete fires)
    pct_change_T5m          FLOAT,
    pct_change_T10m         FLOAT,
    pct_change_T20m         FLOAT,
    pct_change_T30m         FLOAT,
    pct_change_peak         FLOAT,   -- max % gain within observation window
    time_to_peak_min        FLOAT,   -- minutes from alert to peak poll
    peak_interval           TEXT,    -- which interval captured the peak (e.g. 'T10m')
    outcome_complete        BOOL DEFAULT FALSE,

    created_at              TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rt_alert_time      ON research_tokens (alert_time);
CREATE INDEX IF NOT EXISTS idx_rt_category        ON research_tokens (category);
CREATE INDEX IF NOT EXISTS idx_rt_snapshot_ok     ON research_tokens (snapshot_ok);
CREATE INDEX IF NOT EXISTS idx_rt_outcome         ON research_tokens (outcome_complete);
CREATE INDEX IF NOT EXISTS idx_rt_token_address   ON research_tokens (token_address);
-- Unique: one row per token per day (first mention wins)
CREATE UNIQUE INDEX IF NOT EXISTS idx_rt_token_day
    ON research_tokens (token_address, DATE(alert_time));


-- ── Outcome poll log ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS research_outcome_polls (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    token_address   TEXT NOT NULL,
    interval_label  TEXT NOT NULL,   -- 'T3m' | 'T5m' | 'T10m' | 'T20m' | 'T15m' | 'T30m'
    scheduled_at    TIMESTAMPTZ NOT NULL,
    polled_at       TIMESTAMPTZ,
    price_usd       FLOAT,
    mcap_usd        FLOAT,
    liquidity_usd   FLOAT,
    late            BOOL DEFAULT FALSE,   -- TRUE if polled after restart (past-due)
    error           TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rop_token   ON research_outcome_polls (token_address);
CREATE INDEX IF NOT EXISTS idx_rop_sched   ON research_outcome_polls (scheduled_at);
