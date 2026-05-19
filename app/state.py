import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

data_source = os.getenv("DATA_SOURCE", "live")


@dataclass
class Settings:
    """
    Central configuration object for the bot.
    Values are loaded from environment variables (.env).
    """

    polygon_api_key: str = os.getenv("POLYGON_API_KEY", "")

    # External signal sources
    openai_api_key: str        = os.getenv("OPENAI_API_KEY", "")
    twitter_bearer_token: str  = os.getenv("TWITTER_BEARER_TOKEN", "")
    odds_api_key: str          = os.getenv("ODDS_API_KEY", "")
    kalshi_api_key: str        = os.getenv("KALSHI_API_KEY", "")
    kalshi_key_id: str         = os.getenv("KALSHI_KEY_ID", "")
    use_external_signals: bool = os.getenv("USE_EXTERNAL_SIGNALS", "true").lower() == "true"
    data_source: str = data_source

    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    bankroll: float = float(os.getenv("BANKROLL", 5000))

    fractional_kelly: float = float(os.getenv("FRACTIONAL_KELLY", 0.25))

    max_drawdown: float = float(os.getenv("MAX_DRAWDOWN", 0.20))

    # Minimum adjusted edge to emit a signal.  Must exceed the ~2.5% fee buffer
    # to generate any real profit; 5% gives a meaningful margin above costs.
    min_ev: float = float(os.getenv("MIN_EV", 0.05))

    min_volume: float = float(os.getenv("MIN_VOLUME", 5000))

    max_impact: float = float(os.getenv("MAX_IMPACT", 0.03))

    # ── Polymarket execution ──────────────────────────────────────────────────
    # Polygon wallet private key (hex, no 0x prefix)
    poly_private_key: str   = os.getenv("POLY_PRIVATE_KEY", "")
    # Polymarket CLOB API credentials (generated at polymarket.com/profile)
    poly_api_key: str        = os.getenv("POLY_API_KEY", "")
    poly_api_secret: str     = os.getenv("POLY_API_SECRET", "")
    poly_api_passphrase: str = os.getenv("POLY_API_PASSPHRASE", "")
    # Set to "true" to send real orders; anything else = dry-run only
    live_trading: bool       = os.getenv("LIVE_TRADING", "false").lower() == "true"
    # Per-trade size in USDC
    pm_position_size: float  = float(os.getenv("PM_POSITION_SIZE", 10.0))
    # Maximum simultaneous open Polymarket positions
    pm_max_positions: int    = int(os.getenv("PM_MAX_POSITIONS", 5))
    # Stop scanning for the day if cumulative loss exceeds this (USDC)
    pm_daily_loss_limit: float = float(os.getenv("PM_DAILY_LOSS_LIMIT", 40.0))
    # Minimum adjusted edge required to place an order (mirrors memory: 6%+)
    pm_min_edge: float       = float(os.getenv("PM_MIN_EDGE", 0.06))


# Singleton config instance
settings = Settings()
