"""
One-time TG session setup for the research pipeline.

Run:  python -m research.setup

This authenticates a Telethon session using the same account as the trading bot
but saves to a DIFFERENT session file (research/data/tg_research_session).
Two Telethon clients on different session files = two simultaneous logins
(like two devices) — Telegram allows this.

You only need to run this once per server.  After authentication the session
file persists and research/main.py will connect automatically.
"""

import asyncio
import sys


async def _setup():
    try:
        from telethon import TelegramClient
    except ImportError:
        print("ERROR: telethon not installed.  Run: pip install telethon")
        sys.exit(1)

    from research.config import TG_API_ID, TG_API_HASH, TG_SESSION_FILE

    if not TG_API_ID or not TG_API_HASH:
        print("ERROR: TELEGRAM_API_ID and TELEGRAM_API_HASH must be in .env")
        sys.exit(1)

    print(f"Authenticating Telegram research session...")
    print(f"Session file: {TG_SESSION_FILE}.session")
    print()

    async with TelegramClient(TG_SESSION_FILE, TG_API_ID, TG_API_HASH) as client:
        me = await client.get_me()
        print(f"Authenticated as: {me.first_name} (@{me.username})")
        print()

        # Verify we can see the channel
        from research.config import TG_CHANNEL
        try:
            entity = await client.get_entity(TG_CHANNEL)
            print(f"Channel access OK: {entity.title} (@{TG_CHANNEL})")
        except Exception as e:
            print(f"WARNING: Could not resolve channel {TG_CHANNEL}: {e}")
            print("The session was saved but channel access may need checking.")

    print()
    print("Setup complete.  Start the pipeline with:  python -m research.main")


if __name__ == "__main__":
    asyncio.run(_setup())
