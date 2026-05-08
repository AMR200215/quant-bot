"""Memecoin module configuration — all tunable constants in one place."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODULE_DIR   = Path(__file__).parent
DATA_DIR     = MODULE_DIR / "data"
LOGS_DIR     = Path(__file__).parent.parent / "logs"

SOL_WALLETS_FILE  = DATA_DIR / "whale_wallets_sol.json"
BNB_WALLETS_FILE  = DATA_DIR / "whale_wallets_bnb.json"
SIGNALS_FILE      = DATA_DIR / "memecoin_signals.json"
POSITIONS_FILE    = DATA_DIR / "memecoin_positions.json"
WHALE_STATS_FILE  = DATA_DIR / "whale_stats.json"
JOURNAL_FILE           = LOGS_DIR / "memecoin_journal.csv"
CANDIDATES_FILE        = LOGS_DIR / "signal_candidates.csv"
WINNERS_FILE           = LOGS_DIR / "winners_journal.csv"

# ---------------------------------------------------------------------------
# Trade sizing  (change this one number to adjust all trade sizes)
# ---------------------------------------------------------------------------
TRADE_SIZE_USD = 30          # $20–50 range, hardcoded for now

# ---------------------------------------------------------------------------
# Safety filters — token must pass ALL of these
# ---------------------------------------------------------------------------
MIN_LIQUIDITY_USD   = 8_000   # minimum pool liquidity
MAX_MCAP_USD        = 8_000_000  # cap at $8M to leave room for 10-50x
MIN_HOLDERS         = 30      # minimum unique holders (lenient for new launches)
MAX_AGE_MINUTES_NEW = 60      # "new launch" window

# ---------------------------------------------------------------------------
# Signal thresholds
# ---------------------------------------------------------------------------
VOLUME_SPIKE_MULTIPLIER = 4.0   # volume must be 4x the 1h average
MIN_WHALE_TIER_FOR_ALERT = 2    # tier 1 or 2 triggers alert; tier 3 = log only

# Whale tier cutoffs (by win-rate rank across all wallets)
TIER1_TOP_N = 50    # top 50 wallets = Tier 1
TIER2_TOP_N = 150   # next 100 = Tier 2, rest = Tier 3

# Confluence thresholds → signal strength
CONFLUENCE_WEAK   = 1   # 1 tier-3 wallet
CONFLUENCE_MEDIUM = 1   # 1 tier-1 OR 2+ tier-2
CONFLUENCE_STRONG = 2   # 2+ tier-1 OR 5+ any tier

# ---------------------------------------------------------------------------
# Exit logic
# ---------------------------------------------------------------------------
HARD_STOP_PCT        = -0.35   # -35% from entry → immediate exit
TRAILING_STOP_PCT    = -0.40   # -40% from peak once in profit
TRAIL_ACTIVATES_PCT  =  1.00   # trailing stop activates at +100%
TIME_STOP_MINUTES    =  45     # exit if flat >45 min with < +30% gain
TIME_STOP_MIN_GAIN   =  0.30   # if gain > 30% don't apply time stop

# Take-profit ladder (only when no whale exit signal)
TP_LEVELS = [
    (1.00, 0.30),   # at +100%, sell 30% of position
    (3.00, 0.30),   # at +300%, sell 30%
    # remaining 40% rides with trailing stop
]

# ---------------------------------------------------------------------------
# Per-signal-type settings
# Each entry overrides the global defaults above for that signal type.
# Keys must match Signal.signal_type values.
# ---------------------------------------------------------------------------
SIGNAL_SETTINGS: dict[str, dict] = {
    # Global defaults — used when no per-type override is set
    "__default__": {
        "trade_size_usd":      TRADE_SIZE_USD,
        "hard_stop_pct":       HARD_STOP_PCT,
        "trailing_stop_pct":   TRAILING_STOP_PCT,
        "trail_activates_pct": TRAIL_ACTIVATES_PCT,
        "time_stop_minutes":   TIME_STOP_MINUTES,
    },
    "copy_trade": {
        "trade_size_usd":      50,
        "hard_stop_pct":       -0.25,
        "trailing_stop_pct":   -0.30,
        "trail_activates_pct": 0.75,
        "time_stop_minutes":   60,
    },
    "volume_breakout": {
        "trade_size_usd":      20,
        "hard_stop_pct":       -0.40,
        "trailing_stop_pct":   -0.45,
        "trail_activates_pct": 1.00,
        "time_stop_minutes":   30,
    },
    "new_launch": {
        "trade_size_usd":      10,
        "hard_stop_pct":       -0.50,
        "trailing_stop_pct":   -0.50,
        "trail_activates_pct": 1.50,
        "time_stop_minutes":   20,
    },
    "manual": {
        "trade_size_usd":      TRADE_SIZE_USD,
        "hard_stop_pct":       HARD_STOP_PCT,
        "trailing_stop_pct":   TRAILING_STOP_PCT,
        "trail_activates_pct": TRAIL_ACTIVATES_PCT,
        "time_stop_minutes":   TIME_STOP_MINUTES,
    },
}


def get_signal_settings(signal_type: str) -> dict:
    """Return effective settings for a signal type, falling back to __default__."""
    defaults = SIGNAL_SETTINGS["__default__"]
    overrides = SIGNAL_SETTINGS.get(signal_type, {})
    return {**defaults, **overrides}

# ---------------------------------------------------------------------------
# Polling intervals
# ---------------------------------------------------------------------------
SOL_WALLET_POLL_SEC  = 30
BNB_WALLET_POLL_SEC  = 60
DEXSCREENER_POLL_SEC = 120   # scan for volume breakouts + new launches

# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------
CHAINS = {
    "solana": {
        "id": "solana",
        "rpc": "https://api.mainnet-beta.solana.com",
        "native_symbol": "SOL",
        "rugcheck_url": "https://api.rugcheck.xyz/v1/tokens/{address}/report/summary",
        "dex_label": "pump.fun / Raydium",
    },
    "bsc": {
        "id": "bsc",
        "native_symbol": "BNB",
        "bscscan_api": "https://api.bscscan.com/api",
        "honeypot_url": "https://api.honeypot.is/v2/IsHoneypot?address={address}&chainID=56",
        "dex_label": "PancakeSwap",
    },
}

# ---------------------------------------------------------------------------
# API base URLs
# ---------------------------------------------------------------------------
DEXSCREENER_BASE = "https://api.dexscreener.com"
GMGN_BASE        = "https://gmgn.ai/defi/quotation/v1"
RUGCHECK_BASE    = "https://api.rugcheck.xyz/v1"
HONEYPOT_BASE    = "https://api.honeypot.is/v2"
