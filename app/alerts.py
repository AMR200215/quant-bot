"""Alert helpers for the quant bot.

Sends Telegram messages to the user's personal chat via a bot.
Credentials are read from env vars:
  TELEGRAM_BOT_TOKEN — from @BotFather
  TELEGRAM_CHAT_ID   — your personal chat ID (run get_chat_id() to find it)
"""

import logging
import os

import requests

log = logging.getLogger(__name__)

_BOT_TOKEN = ""
_CHAT_ID   = ""


def init():
    """Load credentials from environment. Call once at startup."""
    global _BOT_TOKEN, _CHAT_ID
    _BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    _CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
    if _BOT_TOKEN and _CHAT_ID:
        log.info("Telegram alerts enabled (chat_id=%s)", _CHAT_ID)
    else:
        log.info("Telegram alerts disabled — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")


def _send(text: str) -> bool:
    if not _BOT_TOKEN or not _CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{_BOT_TOKEN}/sendMessage"
        r = requests.post(url, json={"chat_id": _CHAT_ID, "text": text}, timeout=10)
        return r.status_code == 200
    except Exception as e:
        log.debug("Alert send failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Formatted alert types
# ---------------------------------------------------------------------------

def alert_position_open(sig, pos) -> bool:
    """Fired when a new paper position is opened."""
    chain_short = "SOL" if sig.chain == "solana" else "BSC"
    lines = [
        f"[PAPER OPEN] {sig.token_symbol or sig.token_address[:8]} ({chain_short})",
        f"Type:     {sig.signal_type}  |  {sig.strength.upper()}",
        f"Entry:    ${sig.price_usd:.8g}",
        f"Size:     ${pos.size_usd:.2f}",
        f"Stop:     {pos.hard_stop_pct*100:.0f}%  |  DEX: {pos.dex_id or 'n/a'}",
        f"Score:    {sig.composite_score:.2f}  (safety={sig.safety_score:.2f} mom={sig.momentum_score:.2f})",
    ]
    if sig.notes:
        lines.append(f"Note:     {sig.notes}")
    if getattr(sig, "dexscreener_url", ""):
        lines.append(sig.dexscreener_url)
    return _send("\n".join(lines))


def alert_live_buy(pos, tx_sig: str, sol_spent: float) -> bool:
    """Fired when a real on-chain buy is confirmed."""
    chain_short = "SOL" if pos.chain == "solana" else "BSC"
    lines = [
        f"🟢 [LIVE BUY] {pos.token_symbol} ({chain_short})",
        f"Entry:    ${pos.entry_price:.8g}",
        f"Size:     ${pos.size_usd:.2f}  ({sol_spent:.4f} SOL spent)",
        f"Stop:     {pos.hard_stop_pct*100:.0f}%  Trail: {pos.trailing_stop_pct*100:.0f}%",
        f"Tx:       {tx_sig[:20]}...",
    ]
    if getattr(pos, "dexscreener_url", ""):
        lines.append(pos.dexscreener_url)
    return _send("\n".join(lines))


def alert_live_sell(pos, sol_received: float, tx_sig: str) -> bool:
    """Fired when a real on-chain sell is confirmed."""
    pnl_pct = pos.pnl_pct * 100
    pnl_usd = pos.pnl_usd
    sign    = "+" if pnl_usd >= 0 else ""
    emoji   = "🟢" if pnl_usd >= 0 else "🔴"
    chain_short = "SOL" if pos.chain == "solana" else "BSC"
    lines = [
        f"{emoji} [LIVE SELL] {pos.token_symbol} ({chain_short})",
        f"Reason:   {pos.exit_reason}",
        f"PnL:      {sign}{pnl_pct:.1f}%  ({sign}${pnl_usd:.2f})",
        f"Entry:    ${pos.entry_price:.8g}  ->  Exit: ${pos.exit_price:.8g}",
        f"SOL rcvd: {sol_received:.4f}",
        f"Tx:       {tx_sig[:20]}...",
    ]
    return _send("\n".join(lines))


def alert_position_close(pos) -> bool:
    """Fired when a paper position is closed."""
    # Skip paper close alert if this was a live position — alert_live_sell handles it
    if pos.notes and "live|tx:" in pos.notes:
        return False
    pnl_pct = pos.pnl_pct * 100
    pnl_usd = pos.pnl_usd
    sign    = "+" if pnl_usd >= 0 else ""
    chain_short = "SOL" if pos.chain == "solana" else "BSC"
    lines = [
        f"[PAPER CLOSE] {pos.token_symbol} ({chain_short})",
        f"Reason:   {pos.exit_reason}",
        f"PnL:      {sign}{pnl_pct:.1f}%  ({sign}${pnl_usd:.2f})",
        f"Entry:    ${pos.entry_price:.8g}  ->  Exit: ${pos.exit_price:.8g}",
        f"Peak:     ${pos.peak_price:.8g}",
    ]
    return _send("\n".join(lines))


def alert_tp_hit(pos, tp_pct: float, locked_usd: float) -> bool:
    """Fired when a take-profit level is hit."""
    pnl_now = pos.pnl_pct * 100
    chain_short = "SOL" if pos.chain == "solana" else "BSC"
    lines = [
        f"[TP HIT] {pos.token_symbol} ({chain_short})",
        f"Level:    +{tp_pct*100:.0f}%",
        f"Locked:   +${locked_usd:.2f}",
        f"Current PnL: +{pnl_now:.1f}%",
        f"Remaining: {pos.remaining_fraction*100:.0f}% of position",
    ]
    return _send("\n".join(lines))


# ---------------------------------------------------------------------------
# Setup helper — run once to find your chat_id
# ---------------------------------------------------------------------------

def get_chat_id(bot_token: str) -> None:
    """
    Print recent updates so you can find your chat_id.
    Usage: python -c "from app.alerts import get_chat_id; get_chat_id('YOUR_TOKEN')"
    """
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    r = requests.get(url, timeout=10)
    data = r.json()
    if not data.get("ok"):
        print("Error:", data)
        return
    for update in data.get("result", []):
        msg = update.get("message", {})
        chat = msg.get("chat", {})
        print(f"chat_id: {chat.get('id')}  from: {chat.get('first_name', '')} @{chat.get('username', '')}")


if __name__ == "__main__":
    print("Run get_chat_id('YOUR_BOT_TOKEN') to find your chat ID")
