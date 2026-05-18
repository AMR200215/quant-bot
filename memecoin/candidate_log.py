"""
Two-tier signal logging:
  signal_candidates.csv  — every signal that passes the filter (all types, all strengths)
  winners_journal.csv    — only profitable closed positions, promoted at exit
"""

import csv
import logging
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

from memecoin.config import CANDIDATES_FILE, WINNERS_FILE, REJECTIONS_FILE

# Captured at signal time — full market snapshot
_SIGNAL_FIELDS = [
    "signal_id", "signal_time", "chain", "token_address", "token_symbol",
    "signal_type", "strength",
    # scores
    "safety_score", "momentum_score", "composite_score",
    # whale context (copy_trade only; blank for others)
    "whale_count", "whale_tiers",
    # price at signal time
    "price_usd",
    # price action
    "price_change_5m", "price_change_1h", "price_change_6h",
    # buy pressure
    "buys_5m", "sells_5m", "buys_h1", "sells_h1",
    "buy_sell_ratio_5m", "buy_sell_ratio_h1",
    # volume
    "volume_5m", "volume_h1", "volume_h6",
    # market
    "liquidity_usd", "mcap_usd", "fdv", "age_minutes",
    # token info
    "dex_id", "dexscreener_url",
    "has_twitter", "has_telegram", "has_website",
    "rugcheck_score", "mint_disabled", "freeze_disabled", "buy_tax", "sell_tax",
    "notes",
]

# Winners journal: signal snapshot + exit outcome
_WINNERS_FIELDS = _SIGNAL_FIELDS + [
    "position_id", "entry_price", "entry_time", "size_usd",
    "exit_price", "exit_time", "exit_reason",
    "pnl_usd", "pnl_pct", "peak_price",
]


def _sig_to_row(sig) -> dict:
    return {
        "signal_id":           sig.id,
        "signal_time":         time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(sig.timestamp)),
        "chain":               sig.chain,
        "token_address":       sig.token_address,
        "token_symbol":        sig.token_symbol,
        "signal_type":         sig.signal_type,
        "strength":            sig.strength,
        "safety_score":        sig.safety_score,
        "momentum_score":      sig.momentum_score,
        "composite_score":     sig.composite_score,
        "whale_count":         getattr(sig, "whale_count", 0),
        "whale_tiers":         ",".join(str(t) for t in getattr(sig, "whale_tiers", [])),
        "price_usd":           sig.price_usd,
        "price_change_5m":     getattr(sig, "price_change_5m", 0.0),
        "price_change_1h":     getattr(sig, "price_change_1h", 0.0),
        "price_change_6h":     getattr(sig, "price_change_6h", 0.0),
        "buys_5m":             getattr(sig, "buys_5m", 0),
        "sells_5m":            getattr(sig, "sells_5m", 0),
        "buys_h1":             getattr(sig, "buys_h1", 0),
        "sells_h1":            getattr(sig, "sells_h1", 0),
        "buy_sell_ratio_5m":   getattr(sig, "buy_sell_ratio_5m", 0.0),
        "buy_sell_ratio_h1":   getattr(sig, "buy_sell_ratio_h1", 0.0),
        "volume_5m":           getattr(sig, "volume_5m", 0.0),
        "volume_h1":           sig.volume_h1,
        "volume_h6":           getattr(sig, "volume_h6", 0.0),
        "liquidity_usd":       sig.liquidity_usd,
        "mcap_usd":            sig.mcap_usd,
        "fdv":                 getattr(sig, "fdv", 0.0),
        "age_minutes":         round(sig.age_minutes, 1),
        "dex_id":              getattr(sig, "dex_id", ""),
        "dexscreener_url":     getattr(sig, "dexscreener_url", ""),
        "has_twitter":         getattr(sig, "has_twitter", False),
        "has_telegram":        getattr(sig, "has_telegram", False),
        "has_website":         getattr(sig, "has_website", False),
        "rugcheck_score":      getattr(sig, "rugcheck_score", ""),
        "mint_disabled":       getattr(sig, "mint_disabled", ""),
        "freeze_disabled":     getattr(sig, "freeze_disabled", ""),
        "buy_tax":             getattr(sig, "buy_tax", ""),
        "sell_tax":            getattr(sig, "sell_tax", ""),
        "notes":               sig.notes,
    }


def _append_csv(path: Path, fields: list[str], row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def log_signal_candidate(sig):
    """Log every signal that passes the filter — all types, all strengths."""
    _append_csv(CANDIDATES_FILE, _SIGNAL_FIELDS, _sig_to_row(sig))


def _register_dev_bg(chain, token_address, token_symbol, pnl_pct, pnl_usd, signal_id):
    try:
        from memecoin.dev_tracker import register_winner_dev
        register_winner_dev(chain, token_address, token_symbol, pnl_pct, pnl_usd, signal_id)
    except Exception as e:
        log.debug("register_winner_dev failed: %s", e)


def promote_to_winners(pos):
    """Promote a closed, profitable position to the permanent winners journal."""
    if pos.pnl_usd <= 0:
        return
    # Reconstruct signal-level row from position fields
    row = {
        "signal_id":           pos.signal_id,
        "signal_time":         time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(pos.entry_time)),
        "chain":               pos.chain,
        "token_address":       pos.token_address,
        "token_symbol":        pos.token_symbol,
        "signal_type":         pos.signal_type,
        "strength":            pos.strength,
        "safety_score":        pos.safety_score,
        "momentum_score":      pos.momentum_score,
        "composite_score":     pos.composite_score,
        "whale_count":         pos.whale_count,
        "whale_tiers":         ",".join(str(t) for t in pos.whale_tiers),
        "price_usd":           pos.entry_price,
        "price_change_5m":     pos.price_change_5m,
        "price_change_1h":     pos.price_change_1h,
        "price_change_6h":     pos.price_change_6h,
        "buys_5m":             pos.buys_5m,
        "sells_5m":            pos.sells_5m,
        "buys_h1":             pos.buys_h1,
        "sells_h1":            pos.sells_h1,
        "buy_sell_ratio_5m":   pos.buy_sell_ratio_5m,
        "buy_sell_ratio_h1":   pos.buy_sell_ratio_h1,
        "volume_5m":           pos.volume_5m,
        "volume_h1":           pos.volume_h1,
        "volume_h6":           pos.volume_h6,
        "liquidity_usd":       pos.liquidity_usd,
        "mcap_usd":            pos.mcap_usd,
        "fdv":                 pos.fdv,
        "age_minutes":         round(pos.age_minutes, 1),
        "dex_id":              pos.dex_id,
        "dexscreener_url":     pos.dexscreener_url,
        "has_twitter":         pos.has_twitter,
        "has_telegram":        pos.has_telegram,
        "has_website":         pos.has_website,
        "rugcheck_score":      pos.rugcheck_score,
        "buy_tax":             pos.buy_tax,
        "sell_tax":            pos.sell_tax,
        "notes":               pos.notes,
        # exit outcome
        "position_id":         pos.id,
        "entry_price":         pos.entry_price,
        "entry_time":          time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(pos.entry_time)),
        "size_usd":            pos.size_usd,
        "exit_price":          pos.exit_price,
        "exit_time":           time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(pos.exit_time)) if pos.exit_time else "",
        "exit_reason":         pos.exit_reason,
        "pnl_usd":             round(pos.pnl_usd, 4),
        "pnl_pct":             round(pos.pnl_pct * 100, 2),
        "peak_price":          pos.peak_price,
    }
    _append_csv(WINNERS_FILE, _WINNERS_FIELDS, row)
    # find and register the dev in the background — involves RPC calls
    threading.Thread(
        target=_register_dev_bg,
        args=(pos.chain, pos.token_address, pos.token_symbol,
              pos.pnl_pct * 100, pos.pnl_usd, pos.signal_id),
        daemon=True,
    ).start()


# ---------------------------------------------------------------------------
# New-launch rejection log
# ---------------------------------------------------------------------------

_REJECTION_FIELDS = [
    "timestamp", "chain", "token_address",
    "price_change_5m", "liquidity_usd", "mcap_usd", "age_minutes",
    "dex_id", "buy_sell_ratio_5m", "volume_5m",
    "rejection_reason",
]


def log_new_launch_rejection(chain: str, token_address: str, screen: dict, reason: str = "5m_momentum_below_20"):
    """
    Log a new_launch signal that was rejected by an entry filter.
    Non-blocking — any failure is swallowed so the signal pipeline is never interrupted.
    """
    try:
        row = {
            "timestamp":         time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "chain":             chain,
            "token_address":     token_address,
            "price_change_5m":   screen.get("price_change_5m", ""),
            "liquidity_usd":     screen.get("liquidity_usd", ""),
            "mcap_usd":          screen.get("mcap_usd", ""),
            "age_minutes":       round(screen.get("age_minutes", 0), 1) if screen.get("age_minutes") is not None else "",
            "dex_id":            screen.get("dex_id", ""),
            "buy_sell_ratio_5m": screen.get("buy_sell_ratio_5m", ""),
            "volume_5m":         screen.get("volume_5m", ""),
            "rejection_reason":  reason,
        }
        _append_csv(REJECTIONS_FILE, _REJECTION_FIELDS, row)
    except Exception as e:
        log.debug("log_new_launch_rejection failed: %s", e)
