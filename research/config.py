"""
Research pipeline configuration.
Completely standalone — no imports from memecoin/.

Thresholds here are a snapshot of what the trading bot uses at build time.
They're stored so analysis scripts can answer "what would the bot have decided"
without re-running the live screener.  When trading bot thresholds change,
update the SCREENER_* constants here too and note the date.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")   # service_role key (not anon — needs UPDATE)

# ---------------------------------------------------------------------------
# Telegram (same credentials as trading bot, different session file)
# ---------------------------------------------------------------------------
TG_API_ID    = int(os.getenv("TELEGRAM_API_ID", "0"))
TG_API_HASH  = os.getenv("TELEGRAM_API_HASH", "")
TG_CHANNEL   = "pumpdotfunalert"

# Session file — MUST be different from memecoin/data/tg_session
# Two Telethon clients on the same session file = one kicks the other out.
TG_SESSION_FILE = str(Path(__file__).parent / "data" / "tg_research_session")

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# DexScreener
# ---------------------------------------------------------------------------
DEXSCREENER_BASE   = "https://api.dexscreener.com"
DEX_RETRY_COUNT    = 3      # retries before inserting NULL snapshot
DEX_RETRY_DELAY_S  = 30     # seconds between retries (token indexes within 60s usually)
DEX_CALL_DELAY_S   = 1.0    # min seconds between DexScreener calls (rate limit courtesy)

# ---------------------------------------------------------------------------
# Rugcheck
# ---------------------------------------------------------------------------
RUGCHECK_BASE = "https://api.rugcheck.xyz/v1"

# ---------------------------------------------------------------------------
# Token categories and observation windows
# Derived from 708-trade analysis of actual hold-time-to-win data.
# ---------------------------------------------------------------------------
CATEGORY_INTERVALS = {
    # bonding curve: T1m added to catch early pumps that peak before T3m
    "social_alert_bc":   ["T1m", "T3m", "T5m", "T10m", "T20m"],
    # graduated (pumpswap): almost all resolve at 30 min time_stop
    "social_alert_grad": ["T15m", "T30m"],
    # unknown (DexScreener not indexed at alert time): cover both windows
    "unknown":           ["T5m", "T10m", "T20m", "T30m"],
}

# Interval label → minutes offset from alert_time
INTERVAL_MINUTES = {
    "T1m":  1,
    "T3m":  3,
    "T5m":  5,
    "T10m": 10,
    "T15m": 15,
    "T20m": 20,
    "T30m": 30,
}

# ---------------------------------------------------------------------------
# Screener threshold snapshot (trading bot values as of 2026-06-21)
# Analysis scripts use these to compute screener_passed at query time.
# Update date comment when these change in the trading bot.
# ---------------------------------------------------------------------------
SCREENER_MIN_LIQUIDITY_USD       = 8_000
SCREENER_MAX_MCAP_USD            = 8_000_000
SCREENER_MIN_BUY_SELL_RATIO_5M  = 0.55
SCREENER_MIN_VOL_5M              = 2_000
SCREENER_MAX_VOL_5M              = 15_000
SCREENER_MAX_VOL_H1              = 20_000
SCREENER_MAX_PRICE_CHANGE_5M     = 500     # >500% in 5m = blow-off top risk
SCREENER_MAX_RUGCHECK_SCORE      = 500     # rugcheck 0-1000: lower = safer; >500 = risky

# ---------------------------------------------------------------------------
# Dedup window: ignore same token seen within this many hours
# ---------------------------------------------------------------------------
DEDUP_WINDOW_HOURS = 24

# ---------------------------------------------------------------------------
# Outcome poller restart lookback: on restart, rebuild heap for tokens
# logged in the last N hours that have incomplete outcomes.
# ---------------------------------------------------------------------------
POLLER_LOOKBACK_HOURS = 2
