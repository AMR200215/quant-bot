"""
Signal generation for the memecoin module.

Signal types:
  copy_trade      — one or more whale wallets bought this token
  volume_breakout — sudden volume spike vs 1h average
  new_launch      — fresh token passing safety + momentum filters
"""

import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

from memecoin.config import (
    VOLUME_SPIKE_MULTIPLIER, TIER1_TOP_N, TIER2_TOP_N,
    CONFLUENCE_STRONG, CONFLUENCE_MEDIUM,
    MIN_LIQUIDITY_NEW_LAUNCH, MAX_PRICE_CHANGE_1H_NEW_LAUNCH, MIN_COMPOSITE_NEW_LAUNCH,
    TIER1_COPY_MULTIPLIER, get_signal_settings,
)
from memecoin.screener import compute_safety_score

log = logging.getLogger(__name__)


@dataclass
class Signal:
    id: str
    timestamp: float                 # unix seconds
    chain: str                       # "solana" | "bsc"
    token_address: str
    token_name: str
    token_symbol: str
    signal_type: str                 # copy_trade | volume_breakout | new_launch
    strength: str                    # weak | medium | strong
    price_usd: float
    liquidity_usd: float
    mcap_usd: float
    volume_h1: float
    volume_h24: float
    age_minutes: float
    whales_involved: list = field(default_factory=list)
    whale_count: int = 0
    whale_tiers: list = field(default_factory=list)   # [1,2,3,...]
    safety_score: float = 0.0
    momentum_score: float = 0.0
    composite_score: float = 0.0
    notes: str = ""
    # --- enriched fields for model training ---
    price_change_5m: float = 0.0
    price_change_1h: float = 0.0
    price_change_6h: float = 0.0
    buys_5m: int = 0
    sells_5m: int = 0
    buys_h1: int = 0
    sells_h1: int = 0
    buy_sell_ratio_5m: float = 0.0
    buy_sell_ratio_h1: float = 0.0
    volume_5m: float = 0.0
    volume_h6: float = 0.0
    fdv: float = 0.0
    dex_id: str = ""
    dexscreener_url: str = ""
    has_twitter: bool = False
    has_telegram: bool = False
    has_website: bool = False
    rugcheck_score: float = 0.0
    buy_tax: float = 0.0
    sell_tax: float = 0.0
    # paper-trade tracking
    paper_entry_price: float = 0.0
    paper_entry_time: float = 0.0
    acted_on: bool = False           # True once a paper position is opened

    def to_dict(self) -> dict:
        return asdict(self)


def _tier_for_rank(rank: int) -> int:
    if rank < TIER1_TOP_N:
        return 1
    if rank < TIER1_TOP_N + TIER2_TOP_N:
        return 2
    return 3


def _confluence_strength(tiers: list[int]) -> str:
    tier1 = tiers.count(1)
    total = len(tiers)
    if tier1 >= 2 or total >= 5:
        return "strong"
    if tier1 >= 1 or tiers.count(2) >= 2:
        return "medium"
    return "weak"


def _momentum_score(screen: dict) -> float:
    """0–1 momentum score based on volume relative to liquidity."""
    liq  = screen["liquidity_usd"] or 1
    v1h  = screen["volume_h1"]
    v24h = screen["volume_h24"]
    vol_liq_ratio = v1h / liq
    score = min(vol_liq_ratio / 2.0, 0.6)  # caps at 0.6
    # bonus for very fresh token with rising volume
    if screen["age_minutes"] < 20 and v1h > 0:
        score += 0.2
    return min(score, 1.0)


def _enrich_signal(sig: Signal, screen: dict):
    """Copy all enriched screener fields onto a signal."""
    sig.price_change_5m    = screen.get("price_change_5m", 0.0)
    sig.price_change_1h    = screen.get("price_change_1h", 0.0)
    sig.price_change_6h    = screen.get("price_change_6h", 0.0)
    sig.buys_5m            = screen.get("buys_5m", 0)
    sig.sells_5m           = screen.get("sells_5m", 0)
    sig.buys_h1            = screen.get("buys_h1", 0)
    sig.sells_h1           = screen.get("sells_h1", 0)
    sig.buy_sell_ratio_5m  = screen.get("buy_sell_ratio_5m", 0.0)
    sig.buy_sell_ratio_h1  = screen.get("buy_sell_ratio_h1", 0.0)
    sig.volume_5m          = screen.get("volume_5m", 0.0)
    sig.volume_h6          = screen.get("volume_h6", 0.0)
    sig.fdv                = screen.get("fdv", 0.0)
    sig.dex_id             = screen.get("dex_id", "")
    sig.dexscreener_url    = screen.get("dexscreener_url", "")
    sig.has_twitter        = screen.get("has_twitter", False)
    sig.has_telegram       = screen.get("has_telegram", False)
    sig.has_website        = screen.get("has_website", False)
    sig.rugcheck_score     = screen.get("rugcheck_score") or 0.0
    sig.buy_tax            = screen.get("buy_tax") or 0.0
    sig.sell_tax           = screen.get("sell_tax") or 0.0


def make_copy_trade_signal(
    chain: str,
    token_address: str,
    screen: dict,
    whale_wallets: list[str],
    wallet_ranks: dict[str, int],   # wallet_address → rank (0-based)
) -> Optional[Signal]:
    """
    Build a copy-trade signal.
    Returns None if safety check failed or signal is too weak to log.
    """
    if not screen["passed"]:
        return None

    tiers = [_tier_for_rank(wallet_ranks.get(w, 999)) for w in whale_wallets]
    strength = _confluence_strength(tiers)
    is_tier1 = min(tiers) == 1 if tiers else False

    pair = screen["pair"] or {}
    base = pair.get("baseToken") or {}

    safety   = compute_safety_score(screen)
    momentum = _momentum_score(screen)
    composite = round(0.4 * safety + 0.3 * momentum + 0.3 * (
        1.0 if strength == "strong" else 0.6 if strength == "medium" else 0.2
    ), 3)

    sig = Signal(
        id=str(uuid.uuid4())[:8],
        timestamp=time.time(),
        chain=chain,
        token_address=token_address,
        token_name=base.get("name", ""),
        token_symbol=base.get("symbol", ""),
        signal_type="copy_trade",
        strength=strength,
        price_usd=screen["price_usd"],
        liquidity_usd=screen["liquidity_usd"],
        mcap_usd=screen["mcap_usd"],
        volume_h1=screen["volume_h1"],
        volume_h24=screen["volume_h24"],
        age_minutes=screen["age_minutes"],
        whales_involved=whale_wallets,
        whale_count=len(whale_wallets),
        whale_tiers=tiers,
        safety_score=round(safety, 3),
        momentum_score=round(momentum, 3),
        composite_score=composite,
        notes=f"{len(whale_wallets)} whale(s) tier={min(tiers)}"
              + (" [TIER1-3x]" if is_tier1 else ""),
    )
    sig.paper_entry_price = screen["price_usd"]
    sig.paper_entry_time  = sig.timestamp
    _enrich_signal(sig, screen)

    # Tier 1 wallets get 3x trade size — higher conviction, proven win rate
    if is_tier1:
        base_size = get_signal_settings("copy_trade")["trade_size_usd"]
        sig.notes = sig.notes  # already set above
        # Store multiplied size in notes for portfolio to pick up
        sig._tier1_size = round(base_size * TIER1_COPY_MULTIPLIER, 2)

    return sig


def make_volume_breakout_signal(
    chain: str,
    token_address: str,
    screen: dict,
    baseline_volume: float,
) -> Optional[Signal]:
    """Build a volume-breakout signal."""
    if not screen["passed"]:
        return None

    v1h = screen["volume_h1"]
    if baseline_volume <= 0 or v1h < baseline_volume * VOLUME_SPIKE_MULTIPLIER:
        return None

    spike_x = round(v1h / max(baseline_volume, 1), 1)
    pair = screen["pair"] or {}
    base = pair.get("baseToken") or {}

    safety   = compute_safety_score(screen)
    momentum = _momentum_score(screen)
    composite = round(0.35 * safety + 0.65 * momentum, 3)
    strength = "strong" if spike_x >= 8 else "medium" if spike_x >= 4 else "weak"

    sig = Signal(
        id=str(uuid.uuid4())[:8],
        timestamp=time.time(),
        chain=chain,
        token_address=token_address,
        token_name=base.get("name", ""),
        token_symbol=base.get("symbol", ""),
        signal_type="volume_breakout",
        strength=strength,
        price_usd=screen["price_usd"],
        liquidity_usd=screen["liquidity_usd"],
        mcap_usd=screen["mcap_usd"],
        volume_h1=screen["volume_h1"],
        volume_h24=screen["volume_h24"],
        age_minutes=screen["age_minutes"],
        safety_score=round(safety, 3),
        momentum_score=round(momentum, 3),
        composite_score=composite,
        notes=f"volume spike {spike_x}x vs baseline",
    )
    sig.paper_entry_price = screen["price_usd"]
    sig.paper_entry_time  = sig.timestamp
    _enrich_signal(sig, screen)
    return sig


def make_new_launch_signal(
    chain: str,
    token_address: str,
    screen: dict,
) -> Optional[Signal]:
    """Build a new-launch signal (very fresh token with momentum)."""
    if not screen["passed"]:
        return None
    if screen["age_minutes"] > 60:
        return None
    if screen["volume_h1"] <= 0:
        return None

    # --- Data-derived entry filters (see journal analysis 2026-05-05) ---
    # Liquidity < $25K: avg PnL -15%, above $25K significantly better
    if screen["liquidity_usd"] < MIN_LIQUIDITY_NEW_LAUNCH:
        log.debug("new_launch skipped %s: low liquidity $%.0f", token_address, screen["liquidity_usd"])
        return None
    # Already pumped 250%+ in last hour → likely late entry, dumps shortly after
    if screen["price_change_1h"] > MAX_PRICE_CHANGE_1H_NEW_LAUNCH:
        log.debug("new_launch skipped %s: already pumped %.0f%% in 1h", token_address, screen["price_change_1h"])
        return None
    # Falling in last 5 min → momentum gone, avg PnL -27% vs -5% when positive
    if screen["price_change_5m"] <= 0:
        log.debug("new_launch skipped %s: negative 5m price change %.1f%%", token_address, screen["price_change_5m"])
        return None

    pair = screen["pair"] or {}
    base = pair.get("baseToken") or {}

    safety   = compute_safety_score(screen)
    momentum = _momentum_score(screen)
    composite = round(0.3 * safety + 0.7 * momentum, 3)

    if composite < MIN_COMPOSITE_NEW_LAUNCH:
        log.debug("new_launch skipped %s: composite score %.2f below minimum", token_address, composite)
        return None

    strength = "strong" if composite >= 0.65 else "medium" if composite >= 0.45 else "weak"

    sig = Signal(
        id=str(uuid.uuid4())[:8],
        timestamp=time.time(),
        chain=chain,
        token_address=token_address,
        token_name=base.get("name", ""),
        token_symbol=base.get("symbol", ""),
        signal_type="new_launch",
        strength=strength,
        price_usd=screen["price_usd"],
        liquidity_usd=screen["liquidity_usd"],
        mcap_usd=screen["mcap_usd"],
        volume_h1=screen["volume_h1"],
        volume_h24=screen["volume_h24"],
        age_minutes=round(screen["age_minutes"], 1),
        safety_score=round(safety, 3),
        momentum_score=round(momentum, 3),
        composite_score=composite,
        notes=f"age {screen['age_minutes']:.0f}min liq ${screen['liquidity_usd']:.0f}",
    )
    sig.paper_entry_price = screen["price_usd"]
    sig.paper_entry_time  = sig.timestamp
    _enrich_signal(sig, screen)
    return sig


def make_dev_launch_signal(
    chain: str,
    token_address: str,
    screen: dict,
    dev_address: str,
    dev_entry: dict,
    strength: str,
) -> Optional[Signal]:
    """
    Signal fired when a tracked winner dev deploys a new token.
    Strength is pre-computed by dev_tracker.dev_signal_strength() based on
    that dev's historical win count and average PnL.
    """
    if not screen["passed"]:
        return None

    pair = screen["pair"] or {}
    base = pair.get("baseToken") or {}

    safety    = compute_safety_score(screen)
    momentum  = _momentum_score(screen)
    dev_score = dev_entry.get("score", 0.0)
    composite = round(0.25 * safety + 0.25 * momentum + 0.50 * dev_score, 3)

    sig = Signal(
        id=str(uuid.uuid4())[:8],
        timestamp=time.time(),
        chain=chain,
        token_address=token_address,
        token_name=base.get("name", ""),
        token_symbol=base.get("symbol", ""),
        signal_type="dev_launch",
        strength=strength,
        price_usd=screen["price_usd"],
        liquidity_usd=screen["liquidity_usd"],
        mcap_usd=screen["mcap_usd"],
        volume_h1=screen["volume_h1"],
        volume_h24=screen["volume_h24"],
        age_minutes=round(screen["age_minutes"], 1),
        safety_score=round(safety, 3),
        momentum_score=round(momentum, 3),
        composite_score=composite,
        notes=(
            f"dev:{dev_address[:8]}"
            f" wins={dev_entry['win_count']}"
            f" avg_pnl={dev_entry['avg_pnl_pct']:.0f}%"
            f" score={dev_score:.2f}"
        ),
    )
    sig.paper_entry_price = screen["price_usd"]
    sig.paper_entry_time  = sig.timestamp
    _enrich_signal(sig, screen)
    return sig
