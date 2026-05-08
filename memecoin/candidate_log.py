"""
Two-tier signal logging:
  signal_candidates.csv  — every trade that passes the filter, logged at entry
  winners_journal.csv    — only profitable closes, promoted from candidates
"""

import csv
import time
from pathlib import Path

from memecoin.config import CANDIDATES_FILE, WINNERS_FILE

# Fields captured at entry time (snapshot of market conditions)
_ENTRY_FIELDS = [
    "id", "signal_id", "chain", "token_address", "token_symbol",
    "signal_type", "strength",
    "entry_price", "entry_time", "size_usd",
    # scores
    "safety_score", "momentum_score", "composite_score",
    # whale context
    "whale_count", "whale_tiers",
    # price action at entry
    "price_change_5m", "price_change_1h", "price_change_6h",
    # buy pressure at entry
    "buys_5m", "sells_5m", "buys_h1", "sells_h1",
    "buy_sell_ratio_5m", "buy_sell_ratio_h1",
    # volume at entry
    "volume_5m", "volume_h1", "volume_h6",
    # market at entry
    "liquidity_usd", "mcap_usd", "fdv", "age_minutes",
    # token info
    "dex_id", "dexscreener_url",
    "has_twitter", "has_telegram", "has_website",
    "rugcheck_score", "buy_tax", "sell_tax",
    "notes",
]

# Winners journal adds exit outcome columns
_WINNERS_FIELDS = _ENTRY_FIELDS + [
    "exit_price", "exit_time", "exit_reason",
    "pnl_usd", "pnl_pct", "peak_price",
]


def _pos_to_entry_row(pos) -> dict:
    return {
        "id":                  pos.id,
        "signal_id":           pos.signal_id,
        "chain":               pos.chain,
        "token_address":       pos.token_address,
        "token_symbol":        pos.token_symbol,
        "signal_type":         pos.signal_type,
        "strength":            pos.strength,
        "entry_price":         pos.entry_price,
        "entry_time":          time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(pos.entry_time)),
        "size_usd":            pos.size_usd,
        "safety_score":        pos.safety_score,
        "momentum_score":      pos.momentum_score,
        "composite_score":     pos.composite_score,
        "whale_count":         pos.whale_count,
        "whale_tiers":         ",".join(str(t) for t in pos.whale_tiers),
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
    }


def _append_csv(path: Path, fields: list[str], row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def log_candidate(pos):
    """Log every position that passes the filter at the moment it opens."""
    _append_csv(CANDIDATES_FILE, _ENTRY_FIELDS, _pos_to_entry_row(pos))


def promote_to_winners(pos):
    """Promote a closed, profitable position to the permanent winners journal."""
    if pos.pnl_usd <= 0:
        return
    row = _pos_to_entry_row(pos)
    row["exit_price"]  = pos.exit_price
    row["exit_time"]   = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(pos.exit_time)) if pos.exit_time else ""
    row["exit_reason"] = pos.exit_reason
    row["pnl_usd"]     = round(pos.pnl_usd, 4)
    row["pnl_pct"]     = round(pos.pnl_pct * 100, 2)
    row["peak_price"]  = pos.peak_price
    _append_csv(WINNERS_FILE, _WINNERS_FIELDS, row)
