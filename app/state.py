import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Settings:
    """
    Central configuration object for the bot.
    Values are loaded from environment variables (.env).
    """

    polygon_api_key: str = os.getenv("POLYGON_API_KEY", "")

    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    bankroll: float = float(os.getenv("BANKROLL", 5000))

    fractional_kelly: float = float(os.getenv("FRACTIONAL_KELLY", 0.25))

    max_drawdown: float = float(os.getenv("MAX_DRAWDOWN", 0.20))

    min_ev: float = float(os.getenv("MIN_EV", 0.03))

    min_volume: float = float(os.getenv("MIN_VOLUME", 5000))

    max_impact: float = float(os.getenv("MAX_IMPACT", 0.03))


# Singleton config instance
settings = Settings()
