"""
Advanced rug pull detection — goes beyond basic honeypot/rugcheck checks.

Thinking like an experienced rug puller, these are the plays that slip past
standard scanners. This module detects each one.

Each check returns a RiskFlag with a severity level:
  CRITICAL  → almost certainly a rug, block the trade
  HIGH      → strong red flag, skip unless other signals are overwhelming
  MEDIUM    → yellow flag, reduce position size
  LOW       → worth noting, monitor closely
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

CRITICAL = "CRITICAL"
HIGH     = "HIGH"
MEDIUM   = "MEDIUM"
LOW      = "LOW"


@dataclass
class RiskFlag:
    code: str           # machine-readable ID
    severity: str       # CRITICAL / HIGH / MEDIUM / LOW
    detail: str         # human-readable explanation


@dataclass
class RugReport:
    token_address: str
    chain: str
    flags: list[RiskFlag] = field(default_factory=list)
    safe_to_trade: bool = True   # flipped False if any CRITICAL or 2+ HIGH

    # Intermediate data stored for re-use by screener
    top_holders: list[dict] = field(default_factory=list)
    deployer_address: Optional[str] = None
    lp_locked_pct: Optional[float] = None

    def add(self, flag: RiskFlag) -> None:
        self.flags.append(flag)
        if flag.severity == CRITICAL:
            self.safe_to_trade = False
        elif flag.severity == HIGH:
            highs = sum(1 for f in self.flags if f.severity == HIGH)
            if highs >= 2:
                self.safe_to_trade = False

    def summary(self) -> str:
        if not self.flags:
            return "CLEAN"
        parts = [f"[{f.severity}] {f.code}: {f.detail}" for f in self.flags]
        status = "BLOCKED" if not self.safe_to_trade else "WARN"
        return f"{status} | " + " | ".join(parts)


# ---------------------------------------------------------------------------
# Helper: Solana token largest accounts (holder concentration)
# ---------------------------------------------------------------------------

def _sol_largest_accounts(token_address: str, rpc_url: str) -> list[dict]:
    """Return top 20 token holders via Solana RPC getTokenLargestAccounts."""
    import requests
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenLargestAccounts",
        "params": [token_address],
    }
    try:
        r = requests.post(rpc_url, json=payload, timeout=10)
        r.raise_for_status()
        return r.json().get("result", {}).get("value", [])
    except Exception as e:
        log.debug("_sol_largest_accounts failed: %s", e)
        return []


def _sol_token_supply(token_address: str, rpc_url: str) -> float:
    """Return total token supply as a float."""
    import requests
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenSupply",
        "params": [token_address],
    }
    try:
        r = requests.post(rpc_url, json=payload, timeout=10)
        r.raise_for_status()
        val = r.json().get("result", {}).get("value", {})
        return float(val.get("uiAmount") or 0)
    except Exception as e:
        log.debug("_sol_token_supply failed: %s", e)
        return 0.0


def _sol_account_tx_count(wallet: str, rpc_url: str) -> int:
    """Approximate activity level of a wallet by counting recent signatures."""
    import requests
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getSignaturesForAddress",
        "params": [wallet, {"limit": 10}],
    }
    try:
        r = requests.post(rpc_url, json=payload, timeout=8)
        r.raise_for_status()
        sigs = r.json().get("result", [])
        return len(sigs)
    except Exception:
        return 10   # assume active on failure (don't false-positive block)


# ---------------------------------------------------------------------------
# Check 1 — Sybil holder inflation
#
# Rug play: airdrop tokens to 50+ freshly created wallets right at launch
# so MIN_HOLDERS check passes. Each wallet has never done anything else.
# ---------------------------------------------------------------------------

def check_sybil_holders(report: RugReport, rpc_url: str,
                         zombie_threshold: float = 0.60) -> None:
    """
    Flag if the majority of top holders are zero-activity (zombie) wallets.
    zombie_threshold: fraction of holders that must look active before we pass.
    """
    holders = report.top_holders
    if not holders:
        return

    sample = holders[:10]   # checking all 20 is too slow, top 10 is enough
    zombie_count = 0
    for h in sample:
        owner = h.get("address", "")
        if not owner:
            continue
        tx_count = _sol_account_tx_count(owner, rpc_url)
        if tx_count <= 2:          # wallet has done almost nothing ever
            zombie_count += 1
        time.sleep(0.05)           # be gentle with public RPC

    zombie_ratio = zombie_count / len(sample) if sample else 0
    if zombie_ratio >= zombie_threshold:
        report.add(RiskFlag(
            code="SYBIL_HOLDERS",
            severity=HIGH,
            detail=f"{zombie_count}/{len(sample)} top holders are zero-activity wallets "
                   f"({zombie_ratio:.0%}) — likely airdrop inflation",
        ))
    elif zombie_ratio >= 0.40:
        report.add(RiskFlag(
            code="SYBIL_HOLDERS",
            severity=MEDIUM,
            detail=f"{zombie_count}/{len(sample)} top holders look inactive ({zombie_ratio:.0%})",
        ))


# ---------------------------------------------------------------------------
# Check 2 — Top holder concentration (bundled snipe / insider accumulation)
#
# Rug play: use Jito bundles or MEV at block 0 to snipe 20-40% of supply
# across 5-10 wallets so no single address looks suspicious.
# ---------------------------------------------------------------------------

def check_holder_concentration(report: RugReport, token_address: str,
                                rpc_url: str) -> None:
    """
    Flag if top 1 / top 5 / top 10 holders own too much of the supply.
    Thresholds based on observed rug patterns (not LP accounts).
    """
    holders = _sol_largest_accounts(token_address, rpc_url)
    if not holders:
        return

    report.top_holders = holders   # store for sybil check re-use

    total_supply = _sol_token_supply(token_address, rpc_url)
    if total_supply <= 0:
        return

    amounts = [float(h.get("uiAmount") or 0) for h in holders]
    top1_pct  = amounts[0] / total_supply if amounts else 0
    top5_pct  = sum(amounts[:5])  / total_supply if len(amounts) >= 5 else 0
    top10_pct = sum(amounts[:10]) / total_supply if len(amounts) >= 10 else 0

    if top1_pct >= 0.30:
        report.add(RiskFlag(
            code="WHALE_CONCENTRATION",
            severity=CRITICAL,
            detail=f"Top holder owns {top1_pct:.1%} of supply — single entity can dump at will",
        ))
    elif top1_pct >= 0.20:
        report.add(RiskFlag(
            code="WHALE_CONCENTRATION",
            severity=HIGH,
            detail=f"Top holder owns {top1_pct:.1%} of supply",
        ))

    if top5_pct >= 0.60:
        report.add(RiskFlag(
            code="INSIDER_CONCENTRATION",
            severity=HIGH,
            detail=f"Top 5 holders own {top5_pct:.1%} — likely coordinated insider wallets",
        ))
    elif top5_pct >= 0.45:
        report.add(RiskFlag(
            code="INSIDER_CONCENTRATION",
            severity=MEDIUM,
            detail=f"Top 5 holders own {top5_pct:.1%}",
        ))


# ---------------------------------------------------------------------------
# Check 3 — Gradual liquidity drain (slow rug)
#
# Rug play: drain 10-15% of LP every few hours so no single event triggers
# alerts. Over 24 hours the pool is empty.
# ---------------------------------------------------------------------------

# In-memory liquidity history: {token_address: [(timestamp, liquidity_usd), ...]}
_liq_history: dict[str, list[tuple[float, float]]] = {}
_LIQ_WINDOW_SEC   = 3 * 3600    # look back 3 hours
_LIQ_DROP_WARN    = 0.15        # 15% drop → MEDIUM
_LIQ_DROP_HIGH    = 0.30        # 30% drop → HIGH
_LIQ_DROP_CRIT    = 0.50        # 50% drop → CRITICAL


def record_liquidity_snapshot(token_address: str, liquidity_usd: float) -> None:
    """Call this every scan cycle to build a history for drain detection."""
    history = _liq_history.setdefault(token_address, [])
    history.append((time.time(), liquidity_usd))
    # keep only the last 6 hours of snapshots
    cutoff = time.time() - 6 * 3600
    _liq_history[token_address] = [(t, l) for t, l in history if t >= cutoff]


def check_liquidity_drain(report: RugReport, token_address: str,
                           current_liq: float) -> None:
    """Compare current liquidity against the oldest snapshot in the window."""
    record_liquidity_snapshot(token_address, current_liq)
    history = _liq_history.get(token_address, [])
    if len(history) < 3:
        return   # not enough data yet

    cutoff = time.time() - _LIQ_WINDOW_SEC
    old_snapshots = [(t, l) for t, l in history if t <= cutoff]
    if not old_snapshots:
        return

    oldest_liq = old_snapshots[0][1]
    if oldest_liq <= 0:
        return

    drop_pct = (oldest_liq - current_liq) / oldest_liq

    if drop_pct >= _LIQ_DROP_CRIT:
        report.add(RiskFlag(
            code="LIQUIDITY_DRAIN",
            severity=CRITICAL,
            detail=f"Liquidity dropped {drop_pct:.1%} in the last 3h (${oldest_liq:,.0f} → ${current_liq:,.0f}) — active slow rug",
        ))
    elif drop_pct >= _LIQ_DROP_HIGH:
        report.add(RiskFlag(
            code="LIQUIDITY_DRAIN",
            severity=HIGH,
            detail=f"Liquidity dropped {drop_pct:.1%} in 3h",
        ))
    elif drop_pct >= _LIQ_DROP_WARN:
        report.add(RiskFlag(
            code="LIQUIDITY_DRAIN",
            severity=MEDIUM,
            detail=f"Liquidity dropped {drop_pct:.1%} in 3h",
        ))


# ---------------------------------------------------------------------------
# Check 4 — Wash trading detection
#
# Rug play: buy and sell between own wallets to fake organic volume and
# inflate buy/sell ratio so bots like ours generate a BUY signal.
# ---------------------------------------------------------------------------

def check_wash_trading(report: RugReport, screen: dict) -> None:
    """
    Detect wash trading via statistical anomalies in volume vs price change.
    High volume + near-zero price movement = circular trading.
    Also flags suspiciously perfect buy/sell ratios.
    """
    volume_h1 = screen.get("volume_h1", 0)
    price_change_1h = abs(screen.get("price_change_1h", 0))
    buys_h1  = screen.get("buys_h1", 0)
    sells_h1 = screen.get("sells_h1", 0)
    liq      = screen.get("liquidity_usd", 1)

    # Volume >> liquidity with tiny price movement = wash trading
    vol_to_liq = volume_h1 / liq if liq > 0 else 0
    if vol_to_liq > 3.0 and price_change_1h < 5:
        report.add(RiskFlag(
            code="WASH_TRADING",
            severity=HIGH,
            detail=f"1h volume is {vol_to_liq:.1f}x liquidity but price moved only {price_change_1h:.1f}% — circular trading likely",
        ))
    elif vol_to_liq > 1.5 and price_change_1h < 3:
        report.add(RiskFlag(
            code="WASH_TRADING",
            severity=HIGH,
            detail=f"Volume/liquidity ratio {vol_to_liq:.1f}x with {price_change_1h:.1f}% price change",
        ))

    # Suspiciously equal buy/sell counts (bots matching each other)
    total = buys_h1 + sells_h1
    if total > 20:
        balance = abs(buys_h1 - sells_h1) / total
        if balance < 0.05:    # within 5% of perfect 50/50
            report.add(RiskFlag(
                code="WASH_TRADING",
                severity=HIGH,
                detail=f"Buy/sell txn counts are suspiciously equal: {buys_h1}B / {sells_h1}S — possible bot ping-pong",
            ))


# ---------------------------------------------------------------------------
# Check 5 — Delayed honeypot risk (upgradeable tax)
#
# Rug play: launch with 0% tax (passes honeypot check), then flip contract
# variable after enough buyers accumulate.
# On Solana this manifests as a mutable freeze authority.
# On BSC: low initial tax + contract has setTax / setFee functions.
# ---------------------------------------------------------------------------

def check_delayed_honeypot(report: RugReport, screen: dict, chain: str) -> None:
    """
    Flag conditions that make a delayed-activation honeypot possible.
    """
    safety = screen.get("safety") or {}

    if chain == "bsc":
        sell_tax = screen.get("sell_tax")
        if sell_tax is not None and sell_tax == 0:
            # 0% tax on BSC is often a setup for a delayed flip
            report.add(RiskFlag(
                code="ZERO_TAX_SUSPICIOUS",
                severity=LOW,
                detail="0% sell tax on BSC — unusually clean, verify contract has no setTax function",
            ))

    if chain == "solana":
        risks = safety.get("risks", [])
        # Rugcheck flags freeze authority as a risk
        freeze_risks = [r for r in risks if "freeze" in r.lower() or "mint" in r.lower()]
        if freeze_risks:
            report.add(RiskFlag(
                code="FREEZE_AUTHORITY_ACTIVE",
                severity=CRITICAL,
                detail=f"Token has active freeze/mint authority: {freeze_risks} — dev can lock your tokens at any time",
            ))


# ---------------------------------------------------------------------------
# Check 6 — Age vs volume anomaly (bundled snipe / stealth launch)
#
# Rug play: token is < 5 minutes old but already has massive volume because
# the dev sniped 80% of supply in the first few blocks using Jito bundles.
# By the time you see it, the exit is already staged.
# ---------------------------------------------------------------------------

def check_stealth_launch(report: RugReport, screen: dict) -> None:
    """
    Flag tokens that are very new but already show extreme volume/price action.
    """
    age_min = screen.get("age_minutes", 9999)
    volume_h1 = screen.get("volume_h1", 0)
    price_change_5m = screen.get("price_change_5m", 0)
    liq = screen.get("liquidity_usd", 1)

    if age_min > 30:
        return   # not a new launch concern

    # Volume many times larger than liquidity at age < 10 min = insider snipe
    vol_to_liq = volume_h1 / liq if liq > 0 else 0
    if age_min < 5 and vol_to_liq > 5:
        report.add(RiskFlag(
            code="BUNDLED_SNIPE",
            severity=CRITICAL,
            detail=f"Token is {age_min:.1f} min old, 1h volume is {vol_to_liq:.1f}x liquidity — likely insider bundled snipe",
        ))
    elif age_min < 10 and price_change_5m > 200:
        report.add(RiskFlag(
            code="STEALTH_PUMP",
            severity=MEDIUM,
            detail=f"+{price_change_5m:.0f}% in 5m on a {age_min:.0f}-min-old token — wait for organic buyers to appear",
        ))


# ---------------------------------------------------------------------------
# Check 7 — Fake social legitimacy
#
# Rug play: buy 5k Twitter followers, create empty Telegram.
# Our current check just verifies existence, not quality.
# ---------------------------------------------------------------------------

def check_fake_socials(report: RugReport, screen: dict) -> None:
    """
    If a token claims socials but the age is < 10 min, it's suspicious —
    legitimate projects don't launch with all socials pre-built unless
    they planned the token launch (which is fine for legit projects but
    also the playbook for organized rug teams).
    We flag this as LOW unless combined with other signals.
    """
    age_min = screen.get("age_minutes", 9999)
    has_all_three = (
        screen.get("has_twitter") and
        screen.get("has_telegram") and
        screen.get("has_website")
    )
    if age_min < 10 and has_all_three:
        report.add(RiskFlag(
            code="PREBUILT_SOCIALS",
            severity=LOW,
            detail=f"Token has all 3 socials at {age_min:.0f} min old — could be organized rug team (or legit launch; verify content quality)",
        ))


# ---------------------------------------------------------------------------
# Check 8 — Liquidity-to-mcap ratio
#
# Rug play: create the illusion of a healthy token with inflated mcap (low
# float / large total supply priced near zero) but tiny actual liquidity.
# When you try to sell even a small position, slippage destroys you.
# ---------------------------------------------------------------------------

def check_liq_mcap_ratio(report: RugReport, screen: dict) -> None:
    """
    Liquidity/mcap below 2% means you can't exit without massive slippage.
    """
    liq  = screen.get("liquidity_usd", 0)
    mcap = screen.get("mcap_usd", 0)

    if mcap <= 0:
        return

    ratio = liq / mcap
    if ratio < 0.01:
        report.add(RiskFlag(
            code="LOW_LIQ_RATIO",
            severity=CRITICAL,
            detail=f"Liquidity/mcap = {ratio:.2%} — exit slippage will be catastrophic on any size sell",
        ))
    elif ratio < 0.02:
        report.add(RiskFlag(
            code="LOW_LIQ_RATIO",
            severity=HIGH,
            detail=f"Liquidity/mcap = {ratio:.2%} — thin exit liquidity, keep position small",
        ))


# ---------------------------------------------------------------------------
# Check 9 — Sell pressure spike (active rug in progress)
#
# Rug play: dev has already positioned exit wallets and starts dumping.
# First sign is a sudden sell-side dominance that precedes the liquidity pull.
# ---------------------------------------------------------------------------

def check_sell_pressure_spike(report: RugReport, screen: dict) -> None:
    """
    Spike in sell transactions with declining price = distribution phase.
    """
    buys_5m  = screen.get("buys_5m", 0)
    sells_5m = screen.get("sells_5m", 0)
    price_5m = screen.get("price_change_5m", 0)

    total_5m = buys_5m + sells_5m
    if total_5m < 5:
        return   # not enough data

    sell_ratio = sells_5m / total_5m
    if sell_ratio >= 0.80 and price_5m < -10:
        report.add(RiskFlag(
            code="ACTIVE_DUMP",
            severity=CRITICAL,
            detail=f"{sell_ratio:.0%} of 5m txns are sells, price down {price_5m:.1f}% — dumping in progress",
        ))
    elif sell_ratio >= 0.70 and price_5m < -5:
        report.add(RiskFlag(
            code="SELL_PRESSURE_SPIKE",
            severity=HIGH,
            detail=f"{sell_ratio:.0%} sell txns in 5m with {price_5m:.1f}% price drop",
        ))


# ---------------------------------------------------------------------------
# Master function — run all checks
# ---------------------------------------------------------------------------

def run_rug_checks(
    screen: dict,
    chain: str,
    token_address: str,
    rpc_url: str = "https://api.mainnet-beta.solana.com",
    check_holders: bool = True,
) -> RugReport:
    """
    Run all advanced rug checks on a token that has already passed basic
    screener filters. Returns a RugReport with flags and a safe_to_trade bool.

    check_holders=True makes 10+ RPC calls and adds ~1-2s latency.
    Set to False for fast rescans of already-vetted tokens.
    """
    report = RugReport(token_address=token_address, chain=chain)

    # Fast checks (no extra API calls)
    check_wash_trading(report, screen)
    check_delayed_honeypot(report, screen, chain)
    check_stealth_launch(report, screen)
    check_fake_socials(report, screen)
    check_liq_mcap_ratio(report, screen)
    check_sell_pressure_spike(report, screen)

    # Liquidity drain (uses in-memory history, no API call)
    check_liquidity_drain(report, token_address, screen.get("liquidity_usd", 0))

    # Slow checks — Solana RPC calls for holder data
    if check_holders and chain == "solana":
        check_holder_concentration(report, token_address, rpc_url)
        if report.top_holders:
            check_sybil_holders(report, rpc_url)

    return report
