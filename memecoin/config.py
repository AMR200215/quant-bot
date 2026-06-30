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
JOURNAL_FILE           = LOGS_DIR / "memecoin_journal.csv"         # non-social signals
SOCIAL_JOURNAL_FILE    = LOGS_DIR / "memecoin_social_journal.csv"  # social_alert only (2a)
LIVE_JOURNAL_FILE      = LOGS_DIR / "memecoin_live_journal.csv"
CANDIDATES_FILE        = LOGS_DIR / "signal_candidates.csv"
WINNERS_FILE           = LOGS_DIR / "winners_journal.csv"
REJECTIONS_FILE        = LOGS_DIR / "new_launch_rejections.csv"
TRAJECTORY_FILE        = LOGS_DIR / "signal_trajectory.csv"   # T+30s / T+60s price snapshots
NEAR_MISS_FILE         = DATA_DIR / "near_miss_tracking.json"
PRICE_PATHS_DIR        = LOGS_DIR / "price_paths"     # per-position tick logs (<mint>.csv)
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
CAPITAL_USD = 8     # ← actual wallet balance (update as it grows)

# ---------------------------------------------------------------------------
# Live trading flag
# Flip to True when wallet is funded and SOLANA_PRIVATE_KEY env var is set.
# False = pure paper trade, nothing changes in existing behaviour.
# ---------------------------------------------------------------------------
LIVE_TRADING = False   # PAUSED 2026-06-30 — investigating BST graduated sell failure

# Focus mode: when True, only the social_alert (TG) path runs.
# Disables wallet tracker, market scanner, pumpfun_listener, near-miss poller.
# Zero Helius credits consumed by those subsystems.
# Only active threads: portfolio monitor, PP monitor, TG monitor, reconciler.
SOCIAL_ALERT_ONLY = True

# Daily live loss limit — circuit breaker stops all live trades for the day
DAILY_LOSS_LIMIT = -5.0   # was -$15; tighter to limit exposure while validating executor

# Shadow-live mode: full live path traversal, tx built but NOT sent.
# Every gate decision logged with DRY_RUN prefix.
# Run for 24h, then read logs to get the funnel report.
# Set to False only when going truly live.
LIVE_DRY_RUN = False

# ---------------------------------------------------------------------------
# Realtime price feed
#
# When True:  signal_price baseline = PumpPortal live price captured after
#             screening (~T+1-2s).  Eliminates DexScreener indexer lag
#             (~15-30%) from PnL calculation and stop anchor.
#             Preflight gate tightened to SLIPPAGE_GATE_RT_PCT (20%) since
#             we're now measuring real movement only, not indexer artifact.
#
# When False: signal_price = stale DexScreener price (legacy behaviour).
#             Preflight gate uses SLIPPAGE_GATE_DEX_PCT (15%).
#
# Falls back to DexScreener automatically if PP has no price yet — zero
# behaviour change on miss.
# ---------------------------------------------------------------------------
REALTIME_PRICE_FEED   = True
SLIPPAGE_GATE_RT_PCT  = 0.30   # 30% gate vs PP signal price — aligns with abort_tripwire
                               # (fill > signal*1.30 → abort). Tokens 0-30% above signal
                               # are held normally. Tokens >30% above are reverts, not
                               # fill-then-abort round trips. Prior 35% created a dead-zone
                               # (passes gate but immediately hits abort) losing fee on both legs.
SLIPPAGE_GATE_DEX_PCT = 0.50   # 50% gate vs DexScreener baseline — pump.fun tokens move fast

# ---------------------------------------------------------------------------
# Executor backend
# "pumpportal" — POST pumpportal.fun/api/trade-local → sign → Helius RPC
#               Purpose-built for pump.fun; no Jito, no routing complexity.
# "jupiter"    — Jupiter quote + swap API (legacy fallback for A/B testing)
# ---------------------------------------------------------------------------
EXECUTOR_BACKEND = "pumpportal"

# Sell-stuck retry cooldown: seconds before retrying the full sell ladder
# on a position that exhausted all sell attempts.
SELL_STUCK_RETRY_SEC = 60

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
MAX_PRICE_CHANGE_1H_NEW_LAUNCH = 250  # reverted — 150% filter proven harmful in stress test
MIN_COMPOSITE_NEW_LAUNCH   = 0.50     # skip the bottom-tier signals

# social_alert-specific entry filters (data-derived from v5+v6, 192 trades)
# Stress-tested: 81% WR, avg $2.83, ex-top $28.33 across 32 trades
MIN_BUY_SELL_RATIO_SOCIAL  = 0.55    # buy pressure gate (user chose 0.55 over 0.65)
MIN_VOL_5M_SOCIAL          = 2_000   # low vol = not enough interest
MAX_VOL_5M_SOCIAL          = 50_000  # raised from 15K: research shows winners avg ~$22K; wide zone to capture data
MAX_VOL_H1_SOCIAL          = 100_000  # raised from 20K: match wider 5m ceiling
MAX_PRICE_CHANGE_5M_SOCIAL = 500     # >500% in 5m = blow-off top risk
MAX_MCAP_SOCIAL            = 60_000  # pump.fun graduates at ~$69K — skip near-graduation tokens
MAX_AGE_SOCIAL_LIVE        = 30      # Cat-2 live age gate: TG-alerted BC tokens > 30 min are stale

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
# pump.fun bonding curve graduation threshold (pre-graduation exit)
#
# Calibration (Jun 2026):
#   pump.fun virtual initial SOL: 30 SOL
#   Real SOL to fill bonding curve: 85 SOL
#   vSolInBondingCurve at graduation: 30 + 85 = 115 SOL
#
# Verified by back-solving price formula at known graduation mcap ~$69K:
#   price = (vSol^2 * 1e6 / k) * sol_price, k = 30 * 1.073e15
#   At vSol=115: price = $0.0000678 → mcap = 1e9 × $0.0000678 = $67,800 ≈ $69K ✓
#
# Entry prices for recent graduated tokens back-solved to vSol:
#   WHEN  ($0.0000172900) → vSol ≈ 58.1 SOL (50.5% of graduation)
#   Diego ($0.0000173000) → vSol ≈ 58.1 SOL (50.5% of graduation)
#   3PGWpRRh ($0.0000165500) → vSol ≈ 56.8 SOL (49.4% of graduation)
#
# Trigger at 85% = 97.75 SOL fires 17.25 SOL BEFORE graduation (15% buffer).
# At trigger price ≈ $0.0000490 → +183% vs typical entry price — captures the
# runner leg while still on the bonding curve (PP trade-local sells cleanly).
# ---------------------------------------------------------------------------
GRAD_SOL_UI        = 115.0   # vSolInBondingCurve (SOL) at bonding curve completion
PREGRAD_TRIGGER_PCT = 0.85   # exit when vSol/GRAD_SOL_UI >= this (default 85%)

# ---------------------------------------------------------------------------
# Exit logic
# ---------------------------------------------------------------------------
HARD_STOP_PCT        = -0.35   # -35% from entry → immediate exit
TRAILING_STOP_PCT    = -0.40   # -40% from peak once in profit
TRAIL_ACTIVATES_PCT  =  0.75   # trailing stop activates at +75% (was +100%)
TIME_STOP_MINUTES    =  90     # exit if flat >90 min with < +30% gain (was 45)
TIME_STOP_MIN_GAIN   =  0.30   # if gain > 30% don't apply time stop

# Take-profit ladder (only when no whale exit signal)
# Fractions are of remaining position at each level.
# Path: sell 30% at +30%, 25% of remaining at +60%, 20% of remaining at +120%.
# Trailing stop carries the final ~42% (data: 85% of tokens hit +30%).
TP_LEVELS = [
    (0.30, 0.30),   # at +30%,  sell 30% of remaining
    (0.60, 0.25),   # at +60%,  sell 25% of remaining
    (1.20, 0.20),   # at +120%, sell 20% of remaining
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
        "trail_activates_pct": 0.40,   # was 1.00 — tokens peaking 1.4x then dumping now exit near breakeven
        "time_stop_minutes":   20,
    },
    "dev_launch": {
        "trade_size_usd":      40,
        "hard_stop_pct":       -0.35,
        "trailing_stop_pct":   -0.35,
        "trail_activates_pct": 0.75,
        "time_stop_minutes":   45,
    },
    "social_alert": {
        "trade_size_usd":      3,       # $3/trade — capital reset to $8
        "live_trade_size_usd": 3,       # scale up when wallet recovers
        "hard_stop_pct":       -0.35,
        "time_stop_minutes":   90,
        # ATH-anchored trail tiers (replaces single trailing_stop_pct / trail_activates_pct).
        # Tier selected by peak_gain achieved so far; trail anchors to peak_price.
        # Breakeven floor: peak_gain ≥ 40% → trail_stop ≥ entry * 1.02 (enforced in portfolio).
        # Time stop: only fires while peak_gain < 30% (never interrupts a runner mid-leg).
        # FIX 2: trail arms at +30% (was never used before TP1 at +100%).
        # trail_pct=0.25 at tier-0 captures more of the peak vs old 0.35.
        # Confirm final values via FIX 3 replay before locking in.
        "trail_tiers": [
            {"activates_at": 0.30, "trail_pct": 0.25},  # early: -25% from peak (was 0.35)
            {"activates_at": 1.00, "trail_pct": 0.25},  # +100%: same tightness
            {"activates_at": 3.00, "trail_pct": 0.15},  # +300%: protect the moonshot
        ],
        # profit_lock: exit if gain in [40%, 100%] and peak stalled for N sec
        # FIX 2: 60s stall (was 300s) — fast tokens dump long before 5 min (e.g. Solax)
        "profit_lock_min_gain":   0.40,
        "profit_lock_max_gain":   1.00,
        "profit_lock_stall_sec":  60,   # was 300 — fast tokens dump long before 5 min
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
SOL_WALLET_POLL_SEC  = 86400   # halted — Helius quota; resume when upgraded
BNB_WALLET_POLL_SEC  = 86400   # halted — same
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

# ── T22 buy policy ──────────────────────────────────────────────────────────
ALLOW_T22_LIVE_NORMAL        = False   # live full-size buys of non-hook T22 tokens
ALLOW_T22_CANARY             = True    # canary-size buys of T22 tokens ($3 max)
BLOCK_T22_TRANSFER_HOOK      = True    # paper-only if token has transfer hook
BLOCK_T22_UNKNOWN_EXTENSIONS = True    # paper-only if token has unknown T22 extensions

# ── Exit / sell safety ──────────────────────────────────────────────────────
MAX_JUPITER_EXIT_PRICE_IMPACT_PCT = 35    # block Jupiter exit if impact > 35%
ALLOW_JUPITER_PANIC_EXIT          = False # override impact guard for last-resort

# ── Jupiter retry / backoff (quote + swap build) ──────────────────────────────
JUPITER_MAX_RETRIES        = 4    # max attempts on 429 or timeout (1 initial + 3 retries)
JUPITER_BACKOFF_BASE_MS    = 250  # initial backoff after first 429
JUPITER_BACKOFF_MAX_MS     = 3000 # backoff ceiling (exponential, capped here)
JUPITER_BACKOFF_JITTER_MS  = 150  # uniform random jitter added to each backoff

# ── Jupiter request governor ────────────────────────────────────────────────
# Token-bucket rate limiter with three independent priority tiers.
# BACKGROUND is consumed by scanner/warming. EXIT and EMERGENCY are reserved
# exclusively for sell-path callers. See memecoin/jupiter_governor.py.
JUPITER_GOVERNOR_ENABLED  = True  # set False to disable (pass-through mode)
JUPITER_BACKGROUND_RPM    = 20    # tokens/min for warming / non-urgent calls
JUPITER_EXIT_RPM          = 15    # tokens/min reserved for live exit signals
JUPITER_EMERGENCY_RPM     = 6     # tokens/min reserved for emergency sell retries
# Note: JUPITER_MAX_RETRIES / JUPITER_BACKOFF_* above are shared with the governor.
# JUPITER_JITTER_MS is the governor's jitter parameter (executor uses JUPITER_BACKOFF_JITTER_MS).
JUPITER_JITTER_MS         = 100   # governor jitter (separate from executor backoff jitter)

# Phase-gate: keep False until real-wallet simulation passes for T22 exits
JUPITER_T22_GRAD_PRIMARY_ENABLED = False

# ── Live buy kill switch ─────────────────────────────────────────────────────
LIVE_BUYS_ENABLED              = True    # master switch — auto-disabled on unknown sell failure
AUTO_DISABLE_ON_UNKNOWN_SELL_FAILURE = True  # disable live buys when error_class=unknown_sell_failure

# ── Canary / validation mode ──────────────────────────────────────────────────
LIVE_CANARY_MODE         = True    # cap live buys to MAX_CANARY_TRADE_USD until validated
MAX_CANARY_TRADE_USD     = 3       # max single trade USD in canary mode
EXIT_SYSTEM_VALIDATED    = False   # set True after: 1 T22 BC live sell + 1 T22 PS live sell + 10 clean exits
CANARY_T22_PROBE_ONLY    = False   # T22-only gate disabled — T22 tokens never appear in practice

# ---------------------------------------------------------------------------
# PumpSwap local exit layer (Level 3)
# ---------------------------------------------------------------------------
EXIT_ROUTER_ENABLED          = True    # master switch for exit state classification
PUMPSWAP_LOCAL_SELL_ENABLED  = False   # flip True only after sim validation passes
PUMPSWAP_LOCAL_SIM_ONLY      = True    # simulate, log result, then fall through to executor
PUMPSWAP_LOCAL_REQUIRE_SIM_OK = True   # if sim fails, do not send (always respected)
ALLOW_ZERO_MIN_OUT_EMERGENCY   = False  # if True, skip min_sol_out check (last resort only)
LOCAL_PUMPSWAP_MAX_SLIPPAGE_PCT = 35    # max slippage for min_sol_out computation (35%)
