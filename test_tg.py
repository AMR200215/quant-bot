"""Quick test: verify Telethon can reach @pumpdotfunalert and fetch recent messages."""
import asyncio
from telethon import TelegramClient

SESSION = '/root/quant-bot/memecoin/data/tg_session'
API_ID  = 20897927
API_HASH = '0b406cc8e99aad885fa771ac1d0a67b4'

async def test():
    async with TelegramClient(SESSION, API_ID, API_HASH) as client:
        try:
            entity = await client.get_entity('pumpdotfunalert')
            print('OK — channel found:', entity.title, 'id:', entity.id)
            async for msg in client.iter_messages(entity, limit=5):
                print('MSG:', msg.date, repr((msg.text or '')[:100]))
        except Exception as e:
            print('ERROR:', e)

asyncio.run(test())
