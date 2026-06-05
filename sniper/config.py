"""Sniper module configuration."""

from pathlib import Path
from memecoin.config import CAPITAL_USD, _trade_sizes

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODULE_DIR      = Path(__file__).parent
DATA_DIR        = MODULE_DIR / "data"
LOGS_DIR        = Path(__file__).parent.parent / "logs"
JOURNAL_FILE    = LOGS_DIR / "sniper_journal.csv"
POSITIONS_FILE  = DATA_DIR / "sniper_positions.json"

# ---------------------------------------------------------------------------
# Pump.fun WebSocket
# ---------------------------------------------------------------------------
PUMPFUN_WS_URL  = "wss://pumpportal.fun/api/data"

# ---------------------------------------------------------------------------
# Snipe strategies
# ---------------------------------------------------------------------------
# "launch"    — buy the moment a new token is created on Pump.fun bonding curve
#               highest risk (~85% rug rate), highest upside (100x+ possible)
#
# "migration" — buy when a token graduates Pump.fun → Raydium (~$69K mcap proof)
#               lower risk (~25% rug rate), upside 3x–30x typical
#               DEFAULT: safer, no infrastructure advantage needed
#
ACTIVE_STRATEGY = "migration"   # ← change to "launch" for full degen mode

# ---------------------------------------------------------------------------
# Trade sizing — separate from memecoin capital (sniper is higher risk)
# Uses the same CAPITAL_USD but with more conservative percentages
# ---------------------------------------------------------------------------
_sniper_sizes = _trade_sizes(CAPITAL_USD)

SNIPE_SIZE_LAUNCH    = max(1, round(CAPITAL_USD * 0.01))   # 1% per launch snipe
SNIPE_SIZE_MIGRATION = max(2, round(CAPITAL_USD * 0.02))   # 2% per migration snipe

# ---------------------------------------------------------------------------
# Fast pre-filter thresholds (applied in < 1 second, before full screener)
# ---------------------------------------------------------------------------
MIN_INITIAL_SOL       = 0.1     # deployer must put in at least 0.1 SOL at launch
MAX_DEPLOYER_TOKENS   = 5       # skip if same wallet created > 5 tokens (serial rugger)
MAX_INITIAL_BUY_PCT   = 0.40    # skip if initial buy > 40% of bonding curve supply
RUG_NAME_KEYWORDS     = [       # obvious rug/scam name patterns to skip
    "elon", "trump", "biden", "rug", "scam", "test", "fake",
    "honeypot", "drain", "steal",
]

# Migration-specific: minimum mcap at graduation to consider
MIN_MIGRATION_MCAP_SOL = 60     # ~$60K SOL mcap at graduation (standard is ~69K)

# ---------------------------------------------------------------------------
# Exit logic — tighter than memecoin (snipes are short-duration plays)
# ---------------------------------------------------------------------------
HARD_STOP_PCT        = -0.40    # -40% → exit immediately
TRAILING_STOP_PCT    = -0.35    # -35% from peak once trailing activates
TRAIL_ACTIVATES_PCT  =  1.00    # trailing stop activates at +100% (2x)
TIME_STOP_MINUTES    =  15      # exit if flat > 15 min with < +50% gain
TIME_STOP_MIN_GAIN   =  0.50    # if gain > 50% don't apply time stop

# Take-profit ladder
TP_LEVELS = [
    (1.00, 0.50),   # at +100% (2x): sell 50% — lock in profit fast
    (3.00, 0.25),   # at +300% (4x): sell 25%
    # remaining 25% rides with trailing stop — the moonshot slice
]

# ---------------------------------------------------------------------------
# Dedup cooldown — don't snipe same token twice
# ---------------------------------------------------------------------------
SNIPE_COOLDOWN_SEC = 3600   # 1 hour
