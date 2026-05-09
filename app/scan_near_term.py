"""Live scan for Polymarket markets across configurable time windows.

Usage:
  python -m app.scan_near_term              # default: 2–5 days
  python -m app.scan_near_term 2 60         # 2 days to 2 months
  python -m app.scan_near_term 1 7 0.005    # 1–7 days, threshold 0.005

Fetches live markets, filters by days-to-resolution window, runs the full
model pipeline, and ranks opportunities by signal strength.
Results are saved to the journal automatically.
"""

import sys

from app.data_client import enrich_with_momentum, fetch_markets_by_days
from app.edge import estimate_edge
from app.external_signals import get_external_consensus, _detect_sport
from app.market_classifier import get_topic_category
from app.market_journal import append_journal_record, is_already_logged
from app.model import estimate_probability
from app.portfolio import get_status, is_halted
from app.state import settings

DEFAULT_MIN_DAYS = 1
DEFAULT_MAX_DAYS = 60
DEFAULT_THRESHOLD = 0.005
FETCH_LIMIT = 500

# How many markets to log per run and how many per topic category
LOG_LIMIT = 75
CATEGORY_CAP = 5

# Sports categories subject to the aggregate cap
_SPORTS_CATEGORIES = {"tennis", "soccer", "basketball", "baseball", "hockey",
                      "cricket", "esports", "sports"}
# Max sports markets across all sports categories combined (40% of LOG_LIMIT)
SPORTS_CAP = int(LOG_LIMIT * 0.40)  # 30

# Max GPT calls per scan run (controls cost — ~$0.03-0.05 each)
# GPT is now called inside get_external_consensus() for top candidates only.
GPT_CAP = 10

# Paper trading logs everything above this edge — lower than MIN_EV so we
# collect data even when the model isn't confident enough to signal a real trade
PAPER_TRADE_MIN_EDGE = 0.01

# Minimum adjusted edge to treat a candidate as a real (non-paper) signal.
# Historical data shows peak accuracy in the 1-3% adj_edge band — large edges
# tend to indicate the model is fighting the market (and losing).
# Set at 1% to capture that band; the paper threshold stays at 1% too.
MIN_REAL_SIGNAL_EDGE = 0.01

# Leagues/keywords with elevated match-fixing risk
# Sources: FIFA/TI fixing reports, known scandal history
_HIGH_RISK_PATTERNS = [
    # Chinese football
    "shandong", "qingdao", "zhejiang", "yunnan", "chongqing", "chengdu",
    "beijing guoan", "dalian", "shanghai", "tianjin", "wuhan", "hebei",
    "guangzhou", "shenzhen", "changchun", "jiangsu", "liaoning",
    "fc seoul", "korean", "k league",
    # South/Central American lower tiers
    "pumas", "cienciano", "san lorenzo", "busan",
    # Lower-tier tennis (Challenger/ITF circuits)
    "challenger", "itf", "wuning", "futures",
    # Other flagged leagues
    "albanian", "moldovan", "kosovo", "macedonian", "bosnian",
]


def is_high_risk(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in _HIGH_RISK_PATTERNS)


def describe_edge(value: float) -> str:
    if value < 0.02:
        return "weak"
    if value < 0.05:
        return "moderate"
    return "strong"


def describe_confidence(value: float) -> str:
    if value < 0.30:
        return "low"
    if value < 0.60:
        return "medium"
    return "high"


def describe_risk(value: float) -> str:
    if value < 0.30:
        return "low"
    if value < 0.60:
        return "medium"
    return "high"


def main(
    min_days: float = DEFAULT_MIN_DAYS,
    max_days: float = DEFAULT_MAX_DAYS,
    threshold: float = DEFAULT_THRESHOLD,
) -> None:
    # --- Circuit breaker ---
    if is_halted(settings.max_drawdown):
        status = get_status()
        print("=" * 60)
        print("CIRCUIT BREAKER ACTIVE — no new signals emitted.")
        print(f"  Drawdown: {status['drawdown']:.1%}  |  Limit: {settings.max_drawdown:.1%}")
        print("=" * 60)
        return

    print("=" * 60)
    print(f"MARKET SCAN  ({min_days:.0f}–{max_days:.0f} days to close)")
    print("=" * 60)
    print(f"Fetching markets closing in {min_days:.0f}–{max_days:.0f} days (no junk)...")

    # Fetch two windows separately — short-term sports/events flood the API
    # and fill up the limit before longer-term markets are ever returned
    mid = 7.0
    short = fetch_markets_by_days(min_days=min_days, max_days=mid, limit=FETCH_LIMIT // 2)
    long_ = fetch_markets_by_days(min_days=mid, max_days=max_days, limit=FETCH_LIMIT // 2)
    # FETCH_LIMIT is split evenly so neither window monopolises the API page

    # Merge, deduplicate by market_id
    seen = set()
    markets = []
    for m in short + long_:
        if m.market_id not in seen:
            markets.append(m)
            seen.add(m.market_id)

    # Enrich with 7-day price momentum
    enrich_with_momentum(markets)

    candidates = []

    for market in markets:
        days = market.days_to_resolution
        if days is None:
            continue
        if market.yes_price < 0.05 or market.yes_price > 0.95:
            continue
        if market.volume < 1000:
            continue

        estimate = estimate_probability(market)
        posterior = estimate.posterior

        # Blend with external consensus when available.
        # GPT is called inside get_external_consensus() only for top candidates
        # once the gpt_calls budget allows — handled below after sorting.
        external = get_external_consensus(
            market.question,
            twitter_bearer_token=settings.twitter_bearer_token,
            odds_api_key=settings.odds_api_key,
            kalshi_api_key=settings.kalshi_api_key,
            kalshi_key_id=settings.kalshi_key_id,
            yes_price=market.yes_price,
            # openai_api_key passed later (post-sort) to respect GPT_CAP
        )

        has_sharp    = external.get("sportsbook_p") is not None or external.get("kalshi_p") is not None
        has_external = external.get("consensus_p") is not None
        ext_sources  = external.get("sources", 0)

        if has_external:
            # External signals present — blend, trusting sharp signals heavily.
            # Sharp (sportsbook/kalshi): they are professionally priced, trust more.
            # Multiple soft sources: corroboration adds confidence.
            # Single soft source: modest pull only.
            if has_sharp:
                weight = 0.45
            elif ext_sources >= 2:
                weight = 0.35
            else:
                weight = 0.20
            posterior = round(weight * external["consensus_p"] + (1 - weight) * posterior, 4)
        else:
            # No external signals — model is the only input.
            # Shrink posterior 65% toward the market price to prevent the model
            # from making aggressive contrarian calls it can't back up.
            # Data shows solo contrarian calls are right only 32.6% of the time.
            posterior = round(0.35 * posterior + 0.65 * market.yes_price, 4)

        posterior = max(0.02, min(0.98, posterior))

        edge = estimate_edge(market, posterior, logit=estimate.logit)

        if edge.preferred_side == "buy_yes":
            signal_edge = edge.final_signal_yes
            adjusted_edge = edge.adjusted_edge_yes
        else:
            signal_edge = edge.final_signal_no
            adjusted_edge = edge.adjusted_edge_no

        # Contrarian detection: model disagrees with market price direction.
        # buy_no on a yes_price > 0.5 market, or buy_yes on a yes_price < 0.5 market.
        market_says_yes  = market.yes_price >= 0.5
        model_says_yes   = edge.preferred_side == "buy_yes"
        is_contrarian    = market_says_yes != model_says_yes

        # Paper trade threshold — log everything above 1% edge
        # regardless of whether it passes the strict real-trade filter
        cat = get_topic_category(market.question)
        is_real_signal = (
            adjusted_edge >= max(threshold, settings.min_ev, MIN_REAL_SIGNAL_EDGE)
            and not (
                market.momentum_7d is not None
                and (
                    (edge.preferred_side == "buy_yes" and market.momentum_7d < -0.05)
                    or (edge.preferred_side == "buy_no" and market.momentum_7d > 0.05)
                )
            )
            # Contrarian calls require a sharp external signal to back them up.
            # Without sportsbook or kalshi confirmation, contrarian = paper only.
            and not (is_contrarian and not has_sharp)
            # Extreme probability markets (yes_price <20% or >80%) have 35.3%
            # accuracy — below coin flip. Skip as real signals entirely.
            and not (market.yes_price < 0.20 or market.yes_price > 0.80)
            # Sports markets in the 20-35% zone without sportsbook data are 50-65%
            # accurate — not reliable enough for real signals without sharp data.
            and not (
                cat in _SPORTS_CATEGORIES
                and not has_sharp
                and (0.20 <= market.yes_price <= 0.35 or 0.65 <= market.yes_price <= 0.80)
            )
        )

        if adjusted_edge < PAPER_TRADE_MIN_EDGE:
            continue

        candidates.append(
            {
                "market": market,
                "posterior": posterior,
                "estimate": estimate,
                "edge": edge,
                "signal_edge": signal_edge,
                "adjusted_edge": adjusted_edge,
                "days": days,
                "external": external,
                "is_real_signal": is_real_signal,
                "high_risk": is_high_risk(market.question),
                "category": cat,
            }
        )

    candidates.sort(key=lambda x: x["signal_edge"], reverse=True)

    # Apply diversity cap: walk ranked candidates and keep at most
    # CATEGORY_CAP entries per topic category, up to LOG_LIMIT total.
    # Additionally, sports categories combined are capped at SPORTS_CAP (40%
    # of LOG_LIMIT) so non-sports markets get fair representation.
    category_counts: dict[str, int] = {}
    sports_total = 0
    selected: list[dict] = []
    for c in candidates:
        cat = c["category"]
        if category_counts.get(cat, 0) >= CATEGORY_CAP:
            continue
        if cat in _SPORTS_CATEGORIES and sports_total >= SPORTS_CAP:
            continue
        category_counts[cat] = category_counts.get(cat, 0) + 1
        if cat in _SPORTS_CATEGORIES:
            sports_total += 1
        selected.append(c)
        if len(selected) >= LOG_LIMIT:
            break

    total_scanned = sum(
        1 for m in markets
        if 0.05 <= m.yes_price <= 0.95 and m.volume >= 1000
    )
    real_signals  = [c for c in selected if c["is_real_signal"]]
    paper_only    = [c for c in selected if not c["is_real_signal"]]

    # Category breakdown for transparency
    cat_summary = ", ".join(
        f"{cat}:{cnt}" for cat, cnt in sorted(category_counts.items())
    )
    print(
        f"Markets in window: {total_scanned}  |  "
        f"Selected: {len(selected)} (cap {CATEGORY_CAP}/category, sports cap {sports_total}/{SPORTS_CAP})  |  "
        f"Real signals: {len(real_signals)}  |  "
        f"Paper-only: {len(paper_only)}"
    )
    print(f"Category breakdown: {cat_summary}\n")

    if not selected:
        print("No opportunities found above paper trade threshold (1%).")
        return

    # Second pass: run GPT for top candidates (respects GPT_CAP budget).
    # GPT is called here, post-sort, so credits go to the highest-edge markets.
    # get_external_consensus() caches by question, so re-calling with the
    # openai_api_key just fills in the gpt_p / gpt_verdict fields.
    openai_key = settings.openai_api_key if hasattr(settings, "openai_api_key") else ""
    import os as _os
    if not openai_key:
        openai_key = _os.getenv("OPENAI_API_KEY", "")

    # Prioritize non-sports for GPT — they get full probability estimates vs
    # sports which only get news_check. Sort so non-sports go first within
    # the cap, then sports by edge descending.
    gpt_candidates = [
        c for c in selected
        if not is_already_logged(c["market"].market_id)
        and c["adjusted_edge"] >= 0.02
    ]
    gpt_candidates.sort(key=lambda c: (
        1 if _detect_sport(c["market"].question) else 0,  # non-sports first
        -c["adjusted_edge"]                                # then by edge desc
    ))

    gpt_calls = 0
    for c in gpt_candidates:
        if gpt_calls >= GPT_CAP:
            break
        # Re-call consensus with openai_api_key — cache miss on the new key
        # fills gpt_p / gpt_verdict; everything else comes from cache.
        enriched = get_external_consensus(
            c["market"].question,
            twitter_bearer_token=settings.twitter_bearer_token,
            odds_api_key=settings.odds_api_key,
            kalshi_api_key=settings.kalshi_api_key,
            kalshi_key_id=settings.kalshi_key_id,
            openai_api_key=openai_key,
            yes_price=c["market"].yes_price,
        )
        c["external"] = enriched
        # Also re-blend posterior with updated consensus if gpt_p contributed
        if enriched.get("consensus_p") is not None:
            has_sharp = enriched.get("sportsbook_p") is not None or enriched.get("kalshi_p") is not None
            ext_sources = enriched.get("sources", 1)
            if has_sharp:
                weight = 0.45
            elif ext_sources >= 2:
                weight = 0.35
            else:
                weight = 0.20
            c["posterior"] = round(weight * enriched["consensus_p"] + (1 - weight) * c["posterior"], 4)
            c["posterior"] = max(0.02, min(0.98, c["posterior"]))
        if enriched.get("gpt_verdict") not in ("skipped", ""):
            gpt_calls += 1

    newly_logged = 0
    for rank, c in enumerate(selected, start=1):
        market = c["market"]
        edge   = c["edge"]
        label  = "REAL SIGNAL" if c["is_real_signal"] else "paper trade"
        risk_tag = " | HIGH RISK: match-fixing" if c["high_risk"] else ""

        if is_already_logged(market.market_id):
            continue

        ext        = c["external"]
        gpt_verdict   = ext.get("gpt_verdict", "skipped") or "skipped"
        gpt_reasoning = ext.get("gpt_reasoning", "") or ""

        written = append_journal_record(
            market_id=market.market_id,
            question=market.question,
            yes_price=market.yes_price,
            posterior=c["posterior"],
            preferred_side=edge.preferred_side,
            adjusted_edge=c["adjusted_edge"],
            final_signal=c["signal_edge"],
            confidence=edge.confidence,
            risk_score=edge.risk_score,
            risk_multiplier=edge.risk_multiplier,
            days_to_resolution=c["days"],
            maturity_score=edge.maturity_score,
            resolution_quality_score=edge.resolution_quality_score,
            gpt_verdict=gpt_verdict,
            gpt_reasoning=gpt_reasoning,
            sportsbook_p=str(round(ext["sportsbook_p"], 4)) if ext.get("sportsbook_p") is not None else "",
            kalshi_p=str(round(ext["kalshi_p"], 4)) if ext.get("kalshi_p") is not None else "",
            notes=f"{label} | scan: {min_days:.0f}-{max_days:.0f}d window{risk_tag}",
        )
        if written:
            newly_logged += 1

        print("-" * 60)
        print(f"Rank #{rank}  [{label}]{' ⚠ HIGH RISK: match-fixing' if c['high_risk'] else ''}")
        print(f"Market:    {market.question}")
        print(f"ID:        {market.market_id}")
        print(f"Closes in: {c['days']:.1f} days")
        print()
        print(f"Side:      {edge.preferred_side}")
        print(f"Signal:    {c['signal_edge']:.4f}  [{describe_edge(c['adjusted_edge'])}]")
        print(f"Adj Edge:  {c['adjusted_edge']:.4f}")
        print(
            f"Posterior: {c['posterior']:.4f}  "
            f"(market: {market.yes_price:.4f}  |  "
            f"gap: {c['posterior'] - market.yes_price:+.4f})"
        )
        print(f"Rationale: {c['estimate'].rationale}")
        print(
            f"Confidence:{edge.confidence:.4f}  [{describe_confidence(edge.confidence)}]  |  "
            f"Risk: {edge.risk_score:.4f}  [{describe_risk(edge.risk_score)}]"
        )
        print(f"Volume:    ${market.volume:,.0f}  |  Liquidity: ${market.liquidity_depth:,.0f}")
        if market.momentum_7d is not None:
            direction = "up" if market.momentum_7d > 0 else "down"
            print(f"Momentum (7d): {market.momentum_7d:+.4f}  [{direction}]")
        if ext.get("consensus_p") is not None:
            parts = [f"consensus={ext['consensus_p']:.4f}"]
            if ext.get("sportsbook_p") is not None:
                parts.append(f"Sportsbook={ext['sportsbook_p']:.4f}")
            if ext.get("kalshi_p") is not None:
                parts.append(f"Kalshi={ext['kalshi_p']:.4f}")
            if ext.get("gpt_p") is not None:
                parts.append(f"GPT={ext['gpt_p']:.4f}")
            if ext.get("manifold_p") is not None:
                parts.append(f"Manifold={ext['manifold_p']:.4f}")
            if ext.get("metaculus_p") is not None:
                parts.append(f"Metaculus={ext['metaculus_p']:.4f}")
            if ext.get("x_sentiment") is not None:
                parts.append(f"X={ext['x_sentiment']:.4f}")
            print(f"External:  {'  '.join(parts)}")
        gv = ext.get("gpt_verdict", "skipped") or "skipped"
        gr = ext.get("gpt_reasoning", "") or ""
        if gv not in ("skipped", "error", ""):
            if gv == "news_alert" or (gv == "reject" and ext.get("sportsbook_p") is not None):
                print(f"GPT News:  [! ALERT] {gr}")
            else:
                icon = "✓" if "confirm" in gv else ("~" if gv == "neutral" else "✗")
                print(f"GPT:       [{icon} {gv.upper()}] {gr}")
        print()

    already_known = len(selected) - newly_logged
    print(f"Newly logged: {newly_logged}  |  Already in journal: {already_known}  |  GPT calls used: {gpt_calls}/{GPT_CAP}")
    print("=" * 60)


if __name__ == "__main__":
    args = sys.argv[1:]
    min_d = float(args[0]) if len(args) > 0 else DEFAULT_MIN_DAYS
    max_d = float(args[1]) if len(args) > 1 else DEFAULT_MAX_DAYS
    thr = float(args[2]) if len(args) > 2 else DEFAULT_THRESHOLD
    main(min_d, max_d, thr)
