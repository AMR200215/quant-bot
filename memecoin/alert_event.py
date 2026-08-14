"""memecoin/alert_event.py — V8-REWIRE VR1/VR2: the one object V7 and V8
are both allowed to see.

TelegramAlertEvent carries only facts that exist before either strategy
forms an opinion: the raw alert itself, and objective, source-neutral
observations about the token (progress/venue state come from
memecoin.progress_capture, keyed by event_id, computed independently of
both V7's screen_token() and V8's gate). It never carries a V7 opinion
(screen result, strength, dex_id-as-filtered-by-V7, dedup state, position
state) and V8 code must never import or accept memecoin.scanner.Signal.

Constructed once per raw Telegram alert, in memecoin/scanner.py's
_on_telegram_signal(), before screen_token() runs. V7 continues to build
its own Signal from the same raw inputs independently -- this object is
not a replacement for Signal, it's the shared ancestor both branches fork
from.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TelegramAlertEvent:
    event_id: str
    chain: str
    token_address: str
    alert_ts: float
    message_text: str = ""
    token_symbol: str = ""
