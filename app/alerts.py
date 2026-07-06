"""Alert helpers for the quant bot.

Sends Telegram messages to the user's personal chat via a bot.
Credentials are read from env vars:
  TELEGRAM_BOT_TOKEN — from @BotFather
  TELEGRAM_CHAT_ID   — your personal chat ID (run get_chat_id() to find it)

Supported bot commands (send to the bot in Telegram):
  /sells_off              — disable all live on-chain sells (positions keep tracking)
  /sells_on               — re-enable live sells
  /buys_off               — disable live buys (same as kill switch)
  /buys_on                — re-enable live buys
  /manual_sold SYMBOL     — close an open live position without an on-chain sell
  /manual_sold SYMBOL 0.000045  — same, with explicit exit price
  /status                 — show open positions + kill switch state
"""

import logging
import os
import threading
import time

import requests

log = logging.getLogger(__name__)

_BOT_TOKEN = ""
_CHAT_ID   = ""
_cmd_listener_started = False


def init():
    """Load credentials from environment. Call once at startup."""
    global _BOT_TOKEN, _CHAT_ID
    _BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    _CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
    if _BOT_TOKEN and _CHAT_ID:
        log.info("Telegram alerts enabled (chat_id=%s)", _CHAT_ID)
        _start_command_listener()
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
    """Fired when a new paper position is opened. Only notifies for social_alert."""
    if sig.signal_type != "social_alert":
        return False
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
    """Fired when a paper position is closed. Only notifies for social_alert."""
    if pos.signal_type != "social_alert":
        return False
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
    """Fired when a take-profit level is hit. Only notifies for social_alert."""
    if pos.signal_type != "social_alert":
        return False
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
# Telegram bot command listener
# ---------------------------------------------------------------------------

def _dispatch_command(text: str) -> str:
    """
    Parse and execute one bot command. Returns a reply string.
    Accepted commands (case-insensitive):
      /sells_off, /sells_on, /buys_off, /buys_on
      /manual_sold SYMBOL [price]
      /status
    """
    text = text.strip()
    parts = text.split()
    if not parts:
        return "Empty command."
    cmd = parts[0].lower().lstrip("/")
    args = parts[1:]

    if cmd == "sells_off":
        try:
            from memecoin.kill_switch import disable_live_sells
            disable_live_sells("telegram_command")
        except Exception as e:
            return f"Error: {e}"
        return "🔴 Live sells DISABLED. Positions still track. /sells_on to re-enable."

    elif cmd == "sells_on":
        try:
            from memecoin.kill_switch import enable_live_sells
            enable_live_sells("telegram_command")
        except Exception as e:
            return f"Error: {e}"
        return "✅ Live sells RE-ENABLED."

    elif cmd == "buys_off":
        try:
            from memecoin.kill_switch import disable_live_buys
            disable_live_buys("telegram_command")
        except Exception as e:
            return f"Error: {e}"
        return "🔴 Live buys DISABLED. /buys_on to re-enable."

    elif cmd == "buys_on":
        try:
            from memecoin.kill_switch import enable_live_buys
            enable_live_buys("telegram_command")
        except Exception as e:
            return f"Error: {e}"
        return "✅ Live buys RE-ENABLED."

    elif cmd == "manual_sold":
        if not args:
            return "Usage: /manual_sold SYMBOL [price]\nExample: /manual_sold ESCAPE 0.000045"
        symbol = args[0]
        price = 0.0
        if len(args) >= 2:
            try:
                price = float(args[1])
            except ValueError:
                return f"Invalid price '{args[1]}'. Usage: /manual_sold SYMBOL [price]"
        try:
            from memecoin.scanner import portfolio as _pf
            result = _pf.manual_close_live(symbol, price)
        except Exception as e:
            return f"Error closing {symbol}: {e}"
        if result.get("ok"):
            return (
                f"✅ {result['symbol']} closed as manual_sell\n"
                f"pos_id: {result['pos_id']}\n"
                f"Journaled — bot will no longer retry sell for this position."
            )
        else:
            return f"❌ {result.get('msg', 'Unknown error')}"

    elif cmd == "status":
        try:
            from memecoin.scanner import portfolio as _pf
            from memecoin.kill_switch import live_buys_enabled, live_sells_enabled
            open_live = [p for p in _pf._positions.values()
                         if p.status in ("open", "sell_stuck") and p.notes and "live|tx:" in p.notes]
            lines = [
                f"Buys: {'ON' if live_buys_enabled() else 'OFF'}  "
                f"Sells: {'ON' if live_sells_enabled() else 'OFF'}",
                f"Open live positions: {len(open_live)}",
            ]
            for p in open_live[:5]:
                pnl = (p.current_price / p.entry_price - 1) * 100 if p.entry_price else 0
                lines.append(f"  {p.token_symbol} {pnl:+.1f}% [{p.status}]")
            return "\n".join(lines)
        except Exception as e:
            return f"Status error: {e}"

    else:
        return (
            "Unknown command. Available:\n"
            "/sells_off  /sells_on\n"
            "/buys_off   /buys_on\n"
            "/manual_sold SYMBOL [price]\n"
            "/status"
        )


def _start_command_listener() -> None:
    """Start background thread that polls Telegram for bot commands."""
    global _cmd_listener_started
    if _cmd_listener_started:
        return
    _cmd_listener_started = True

    def _listener():
        offset = 0
        log.info("Telegram command listener started")
        while True:
            try:
                url  = f"https://api.telegram.org/bot{_BOT_TOKEN}/getUpdates"
                resp = requests.get(
                    url,
                    params={"offset": offset, "timeout": 30, "allowed_updates": ["message"]},
                    timeout=35,
                )
                if resp.status_code != 200:
                    time.sleep(5)
                    continue
                data = resp.json()
                for update in data.get("result", []):
                    offset = update["update_id"] + 1
                    msg    = update.get("message", {})
                    chat   = msg.get("chat", {})
                    text   = msg.get("text", "")
                    # Only accept commands from the configured chat (security)
                    if str(chat.get("id")) != str(_CHAT_ID):
                        continue
                    if not text.startswith("/"):
                        continue
                    log.info("TG command: %r", text)
                    reply = _dispatch_command(text)
                    _send(reply)
            except Exception as e:
                log.debug("Command listener error: %s", e)
                time.sleep(5)

    t = threading.Thread(target=_listener, daemon=True, name="tg-cmd-listener")
    t.start()


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
