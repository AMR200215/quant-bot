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
DEV_WALLETS_FILE       = DATA_DIR / "dev_wallets.json"
DEV_LAST_SEEN_FILE     = DATA_DIR / "dev_last_seen.json"

# ---------------------------------------------------------------------------
# Capital & trade sizing
#
# Set CAPITAL_USD to your current trading capital.
# Trade sizes are computed as a % of capital and capped per stage:
#
#   Stage 1  $100  – $999   copy=$5   breakout=$3   launch=$2
#   Stage 2  $1K   – $4999  copy=$30  breakout=$15  launch=$8
#   Stage 3  $5K+            copy=$150 breakout=$75  launch=$25
#
# Rule: never risk more than 3-5% of capital on a single trade.
# Change ONLY this one number — everything else auto-adjusts.
# ---------------------------------------------------------------------------
CAPITAL_USD = 100   # ← set this to your actual capital

def _stage(capital: float) -> int:
    if capital < 1_000:  return 1
    if capital < 5_000:  return 2
    return 3

def _trade_sizes(capital: float) -> dict[str, float]:
    """Return per-signal trade sizes scaled to capital stage."""
    stage = _stage(capital)
    if stage == 1:
        return {
            "copy_trade":       max(2, round(capital * 0.05)),   # 5% of capital
            "volume_breakout":  max(2, round(capital * 0.03)),   # 3%
            "new_launch":       max(1, round(capital * 0.02)),   # 2%
            "manual":           max(2, round(capital * 0.03)),   # 3%
        }
    if stage == 2:
        return {
            "copy_trade":       round(capital * 0.04),   # 4%
            "volume_breakout":  round(capital * 0.02),   # 2%
            "new_launch":       round(capital * 0.01),   # 1%
            "manual":           round(capital * 0.02),   # 2%
        }
    # stage 3
    return {
        "copy_trade":       min(250, round(capital * 0.03)),  # 3%, cap $250
        "volume_breakout":  min(100, round(capital * 0.015)), # 1.5%, cap $100
        "new_launch":       min(50,  round(capital * 0.005)), # 0.5%, cap $50
        "manual":           min(150, round(capital * 0.02)),  # 2%, cap $150
    }

_SIZES = _trade_sizes(CAPITAL_USD)
TRADE_SIZE_USD = _SIZES["copy_trade"]   # default fallback (rarely used directly)

# ---------------------------------------------------------------------------
# Safety filters — token must pass ALL of these
# ---------------------------------------------------------------------------
MIN_LIQUIDITY_USD   = 8_000   # minimum pool liquidity (global)
MAX_MCAP_USD        = 8_000_000  # cap at $8M to leave room for 10-50x
MIN_HOLDERS         = 30      # minimum unique holders (lenient for new launches)
MAX_AGE_MINUTES_NEW = 60      # "new launch" window

# new_launch-specific entry filters (tighter than global — data-derived)
MIN_LIQUIDITY_NEW_LAUNCH   = 25_000   # $25K+ liq: win avg goes from -15% → ~-5%
MAX_PRICE_CHANGE_1H_NEW_LAUNCH = 150  # skip already-pumped tokens (winners avg +140% vs losers +407%)
MIN_COMPOSITE_NEW_LAUNCH   = 0.50     # skip the bottom-tier signals

# ---------------------------------------------------------------------------
# Signal thresholds
# ---------------------------------------------------------------------------
VOLUME_SPIKE_MULTIPLIER = 4.0   # volume must be 4x the 1h average
MIN_WHALE_TIER_FOR_ALERT = 2    # tier 1 or 2 triggers alert; tier 3 = log only

# Whale tier cutoffs (by win-rate rank across all wallets)
TIER1_TOP_N = 15    # top 15 wallets = Tier 1 (elite — 100% win rate from ranking)
TIER2_TOP_N = 50    # next 35 = Tier 2, rest = Tier 3

# Tier 1 copy trades get a larger size multiplier (elite wallets = higher conviction)
TIER1_COPY_MULTIPLIER = 3   # $15 at $100 capital vs $5 for regular copy_trade

# Confluence thresholds → signal strength
CONFLUENCE_WEAK   = 1   # 1 tier-3 wallet
CONFLUENCE_MEDIUM = 1   # 1 tier-1 OR 2+ tier-2
CONFLUENCE_STRONG = 2   # 2+ tier-1 OR 5+ any tier

# ---------------------------------------------------------------------------
# Exit logic
# ---------------------------------------------------------------------------
HARD_STOP_PCT        = -0.35   # -35% from entry → immediate exit
TRAILING_STOP_PCT    = -0.40   # -40% from peak once in profit
TRAIL_ACTIVATES_PCT  =  0.75   # trailing stop activates at +75% (was +100%)
TIME_STOP_MINUTES    =  90     # exit if flat >90 min with < +30% gain (was 45)
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
        "trade_size_usd":      _SIZES["copy_trade"],
        "hard_stop_pct":       -0.25,
        "trailing_stop_pct":   -0.30,
        "trail_activates_pct": 0.75,
        "time_stop_minutes":   60,
    },
    "volume_breakout": {
        "trade_size_usd":      _SIZES["volume_breakout"],
        "hard_stop_pct":       -0.40,
        "trailing_stop_pct":   -0.45,
        "trail_activates_pct": 1.00,
        "time_stop_minutes":   30,
    },
    "new_launch": {
        "trade_size_usd":      _SIZES["new_launch"],
        "hard_stop_pct":       -0.30,
        "trailing_stop_pct":   -0.40,
        "trail_activates_pct": 1.00,
        "time_stop_minutes":   20,
    },
    "dev_launch": {
        "trade_size_usd":      40,
        "hard_stop_pct":       -0.35,
        "trailing_stop_pct":   -0.35,
        "trail_activates_pct": 0.75,
        "time_stop_minutes":   45,
    },
    "manual": {
        "trade_size_usd":      _SIZES["manual"],
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
