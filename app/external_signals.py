"""Fetch external consensus probabilities from Manifold, Metaculus, X,
Kalshi, and The Odds API (sportsbook lines).

Signal weights by sharpness:
  sportsbook  3.0  — professional bookmaker lines (Pinnacle etc.)
  kalshi      2.0  — regulated US real-money prediction market
  manifold    1.0  — play-money but large forecasting community
  metaculus   1.0  — structured forecasting community
  x_sentiment 0.5  — noisy directional signal, dampened toward 0.5

Match quality is measured by string similarity against the market question.
A match below MATCH_THRESHOLD is discarded — a wrong signal is worse than none.

Caching: per-process caches prevent hammering external APIs in batch scans.
Sport-level odds are cached at the sport level (one API call per sport per
scan process) to stay within The Odds API free tier (500 req/month).
"""

import difflib
import hashlib
import re
from typing import Optional

import requests

MANIFOLD_SEARCH_URL  = "https://api.manifold.markets/v0/search-markets"
METACULUS_SEARCH_URL = "https://www.metaculus.com/api2/questions/"
TWITTER_SEARCH_URL   = "https://api.twitter.com/2/tweets/search/recent"
ODDS_API_BASE        = "https://api.the-odds-api.com/v4"
KALSHI_API_BASE      = "https://trading-api.kalshi.com/trade-api/v2"

# Minimum text similarity (0–1) to accept an external match
MATCH_THRESHOLD = 0.55

# Per-request timeout in seconds
REQUEST_TIMEOUT = 6

# Signal weights — higher = more trusted
_SIGNAL_WEIGHTS = {
    "sportsbook": 3.0,
    "kalshi":     2.0,
    "manifold":   1.0,
    "metaculus":  1.0,
    "x":          0.5,
}

# Sentiment keywords for X signal
_BULLISH_WORDS = {"yes", "will", "likely", "expect", "confirm", "happen",
                  "true", "bullish", "surge", "win", "pass", "approved"}
_BEARISH_WORDS = {"no", "won't", "wont", "unlikely", "doubt", "bearish",
                  "false", "fail", "reject", "denied", "lose", "drop"}

# ---------------------------------------------------------------------------
# Sport detection — keyword → Odds API sport key
# ---------------------------------------------------------------------------

_SPORT_KEYWORDS: dict[str, list[str]] = {
    "baseball_mlb": [
        "yankees", "red sox", "dodgers", "cubs", "cardinals", "braves", "mets",
        "phillies", "rays", "marlins", "padres", "rockies", "diamondbacks",
        "brewers", "pirates", "reds", "astros", "angels", "rangers", "mariners",
        "athletics", "royals", "tigers", "guardians", "white sox", "twins",
        "orioles", "blue jays", "nationals", "giants" , "sf giants",
    ],
    "basketball_nba": [
        "lakers", "celtics", "warriors", "bucks", "nuggets", "heat", "nets",
        "knicks", "bulls", "suns", "clippers", "rockets", "thunder", "spurs",
        "raptors", "pelicans", "jazz", "grizzlies", "kings", "mavericks",
        "mavs", "hawks", "pacers", "magic", "hornets", "cavaliers", "cavs",
        "wizards", "pistons", "blazers", "trail blazers",
    ],
    "icehockey_nhl": [
        "penguins", "capitals", "bruins", "maple leafs", "blackhawks",
        "canadiens", "flyers", "red wings", "blues", "golden knights",
        "lightning", "avalanche", "stars", "wild", "hurricanes", "oilers",
        "flames", "canucks", "jets", "kraken", "sabres", "senators",
    ],
    "americanfootball_nfl": [
        "patriots", "chiefs", "49ers", "cowboys", "eagles", "rams", "bills",
        "ravens", "bengals", "steelers", "browns", "titans", "colts",
        "texans", "jaguars", "broncos", "raiders", "chargers", "seahawks",
        "falcons", "saints", "panthers", "buccaneers", "packers",
        "bears", "vikings", "lions",
    ],
    "soccer_epl": [
        "arsenal", "chelsea", "liverpool", "manchester city", "manchester united",
        "tottenham", "newcastle", "everton", "aston villa", "nottingham forest",
        "brighton", "bournemouth", "crystal palace", "fulham", "wolves",
        "brentford", "west ham", "leicester", "southampton", "ipswich",
    ],
    "soccer_uefa_champs_league": [
        "champions league", "ucl final", "ucl semi", "champions league final",
    ],
    "soccer_uefa_europa_league": [
        "europa league", "uel final", "europa league final",
    ],
    "tennis_atp": [
        "atp", "madrid open", "roland garros", "wimbledon", "us open",
        "australian open", "challenger",
    ],
}

# In-process caches
_CACHE: dict[str, dict]  = {}           # full consensus results
_ODDS_CACHE: dict[str, list] = {}       # sport_key → list of event dicts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _question_hash(question: str) -> str:
    return hashlib.md5(question.lower().strip().encode()).hexdigest()


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _clamp(p: float) -> float:
    return round(max(0.02, min(0.98, p)), 4)


def _detect_sport(question: str) -> Optional[str]:
    """Return The Odds API sport key for a question, or None if not a sports market."""
    q = question.lower()
    for sport_key, keywords in _SPORT_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return sport_key
    return None


def _extract_yes_team(question: str) -> Optional[str]:
    """Extract the team that resolves the market YES.

    Handles:
      'Tampa Bay Rays vs. Cleveland Guardians'  → 'Tampa Bay Rays'
      'Will Liverpool FC win on 2026-05-03?'    → 'Liverpool FC'
    """
    # "X vs Y" pattern — YES team is X (first named)
    vs_match = re.search(r"^(.+?)\s+vs\.?\s+", question, re.IGNORECASE)
    if vs_match:
        return vs_match.group(1).strip()
    # "Will X win" pattern
    win_match = re.search(r"Will (.+?) win", question, re.IGNORECASE)
    if win_match:
        return win_match.group(1).strip()
    return None


def _implied_prob(event: dict, yes_team: str) -> Optional[float]:
    """Extract normalised implied probability for yes_team from an odds event."""
    yes_odds: Optional[float] = None
    no_odds:  Optional[float] = None

    bookmakers = event.get("bookmakers", [])
    # Prefer Pinnacle (sharpest); fall back to first available bookmaker
    preferred = next((b for b in bookmakers if b["key"] == "pinnacle"), None)
    source_books = [preferred] if preferred else bookmakers[:3]

    for book in source_books:
        for market in book.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                name = outcome.get("name", "")
                price = outcome.get("price")
                if price and _similarity(name, yes_team) > 0.6:
                    yes_odds = float(price)
                elif price:
                    no_odds = float(price)

    if yes_odds and no_odds:
        raw_yes = 1.0 / yes_odds
        raw_no  = 1.0 / no_odds
        # Normalise to strip bookmaker overround
        return _clamp(raw_yes / (raw_yes + raw_no))

    return None


# ---------------------------------------------------------------------------
# Manifold Markets
# ---------------------------------------------------------------------------

def fetch_manifold_probability(question: str) -> Optional[float]:
    """Return the probability of the best-matching open Manifold market."""
    try:
        resp = requests.get(
            MANIFOLD_SEARCH_URL,
            params={"term": question[:120], "limit": 5},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        results = resp.json()
    except (requests.RequestException, ValueError):
        return None

    best_score, best_p = 0.0, None
    for market in results:
        if market.get("isResolved") or market.get("isClosed"):
            continue
        score = _similarity(question, market.get("question", ""))
        p = market.get("probability")
        if score > best_score and p is not None:
            best_score = score
            best_p = float(p)

    return _clamp(best_p) if best_score >= MATCH_THRESHOLD and best_p is not None else None


# ---------------------------------------------------------------------------
# Metaculus
# ---------------------------------------------------------------------------

def fetch_metaculus_probability(question: str) -> Optional[float]:
    """Return the community median forecast of the best-matching Metaculus question."""
    try:
        resp = requests.get(
            METACULUS_SEARCH_URL,
            params={"search": question[:120], "limit": 5, "format": "json"},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None

    results = data.get("results", [])
    best_score, best_p = 0.0, None
    for item in results:
        if item.get("resolution") is not None:
            continue
        score = _similarity(question, item.get("title", ""))
        prediction = item.get("community_prediction") or {}
        p = (prediction.get("full") or {}).get("q2")
        if score > best_score and p is not None:
            best_score = score
            best_p = float(p)

    return _clamp(best_p) if best_score >= MATCH_THRESHOLD and best_p is not None else None


# ---------------------------------------------------------------------------
# The Odds API (sportsbook lines)
# ---------------------------------------------------------------------------

def fetch_odds_probability(question: str, api_key: str) -> Optional[float]:
    """Return a normalised implied probability from sharp sportsbook lines.

    Fetches events for the detected sport once per scan process (cached).
    Matches the question's YES team against home/away team names.
    Uses Pinnacle odds when available (sharpest market), otherwise averages.
    """
    if not api_key:
        return None

    sport = _detect_sport(question)
    if not sport:
        return None

    # Fetch and cache all events for this sport
    if sport not in _ODDS_CACHE:
        try:
            resp = requests.get(
                f"{ODDS_API_BASE}/sports/{sport}/odds/",
                params={
                    "apiKey":      api_key,
                    "regions":     "us,uk",
                    "markets":     "h2h",
                    "oddsFormat":  "decimal",
                },
                timeout=REQUEST_TIMEOUT,
            )
            _ODDS_CACHE[sport] = resp.json() if resp.status_code == 200 else []
        except (requests.RequestException, ValueError):
            _ODDS_CACHE[sport] = []

    events = _ODDS_CACHE[sport]
    if not events:
        return None

    yes_team = _extract_yes_team(question)
    if not yes_team:
        return None

    # Find the best-matching event
    best_score, best_event = 0.0, None
    for event in events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        score = max(
            _similarity(yes_team, home),
            _similarity(yes_team, away),
            _similarity(question.lower(), f"{home} vs {away}".lower()),
        )
        if score > best_score:
            best_score = score
            best_event = event

    if best_score < MATCH_THRESHOLD or best_event is None:
        return None

    return _implied_prob(best_event, yes_team)


# ---------------------------------------------------------------------------
# Kalshi
# ---------------------------------------------------------------------------

def fetch_kalshi_probability(question: str, api_key: str) -> Optional[float]:
    """Return the YES probability of the best-matching open Kalshi market.

    Kalshi is a regulated US prediction market — real-money prices are sharper
    than Manifold, especially for politics, macro, and earnings events.
    """
    if not api_key:
        return None

    try:
        resp = requests.get(
            f"{KALSHI_API_BASE}/markets",
            headers={"Authorization": f"Bearer {api_key}"},
            params={"limit": 10, "status": "open"},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None

    markets = data.get("markets", [])
    best_score, best_p = 0.0, None

    for market in markets:
        title = market.get("title", "") or market.get("question", "")
        score = _similarity(question, title)
        # Kalshi reports yes_bid/yes_ask — use midpoint as probability
        yes_bid = market.get("yes_bid")
        yes_ask = market.get("yes_ask")
        if yes_bid is not None and yes_ask is not None:
            p = (float(yes_bid) + float(yes_ask)) / 2.0 / 100.0  # cents → 0-1
        elif market.get("last_price") is not None:
            p = float(market["last_price"]) / 100.0
        else:
            continue

        if score > best_score:
            best_score = score
            best_p = p

    return _clamp(best_p) if best_score >= MATCH_THRESHOLD and best_p is not None else None


# ---------------------------------------------------------------------------
# X (Twitter) signal
# ---------------------------------------------------------------------------

_X_ACCESS_DENIED = False


def fetch_x_sentiment(question: str, bearer_token: str) -> Optional[float]:
    """Return a sentiment-derived probability from recent X posts."""
    query_text = re.sub(r"[^\w\s]", "", question[:60]).strip()
    query = f"{query_text} -is:retweet lang:en"

    global _X_ACCESS_DENIED
    if _X_ACCESS_DENIED:
        return None

    try:
        resp = requests.get(
            TWITTER_SEARCH_URL,
            headers={"Authorization": f"Bearer {bearer_token}"},
            params={"query": query, "max_results": 100, "tweet.fields": "text"},
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 403:
            _X_ACCESS_DENIED = True
            return None
        if resp.status_code != 200:
            return None
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None

    tweets = data.get("data", [])
    if not tweets:
        return None

    bullish = bearish = 0
    for tweet in tweets:
        words = set(tweet.get("text", "").lower().split())
        bullish += len(words & _BULLISH_WORDS)
        bearish += len(words & _BEARISH_WORDS)

    total = bullish + bearish
    if total < 5:
        return None

    return _clamp(bullish / total)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_external_consensus(
    question: str,
    twitter_bearer_token: str = "",
    odds_api_key: str = "",
    kalshi_api_key: str = "",
) -> dict:
    """Return a weighted external probability consensus for a question.

    Result dict keys:
      manifold_p    — Manifold probability (float or None)
      metaculus_p   — Metaculus community median (float or None)
      sportsbook_p  — Implied probability from sportsbook odds (float or None)
      kalshi_p      — Kalshi market midpoint probability (float or None)
      x_sentiment   — X/Twitter sentiment probability (float or None)
      consensus_p   — Weighted average of available signals (float or None)
      sources       — number of sources that returned a valid probability
    """
    cache_key = _question_hash(question + odds_api_key[:8] + kalshi_api_key[:8])
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    manifold_p   = fetch_manifold_probability(question)
    metaculus_p  = fetch_metaculus_probability(question)
    sportsbook_p = fetch_odds_probability(question, odds_api_key)
    kalshi_p     = fetch_kalshi_probability(question, kalshi_api_key)
    x_sentiment  = (
        fetch_x_sentiment(question, twitter_bearer_token)
        if twitter_bearer_token else None
    )

    # Weighted consensus — each signal contributes weight × probability
    weighted_sum   = 0.0
    total_weight   = 0.0
    source_count   = 0

    def _add(p: Optional[float], source: str) -> None:
        nonlocal weighted_sum, total_weight, source_count
        if p is None:
            return
        w = _SIGNAL_WEIGHTS[source]
        # X sentiment: dampen toward 0.5 before weighting
        val = p * 0.5 + 0.25 if source == "x" else p
        weighted_sum  += w * val
        total_weight  += w
        source_count  += 1

    _add(sportsbook_p, "sportsbook")
    _add(kalshi_p,     "kalshi")
    _add(manifold_p,   "manifold")
    _add(metaculus_p,  "metaculus")
    _add(x_sentiment,  "x")

    consensus_p = _clamp(weighted_sum / total_weight) if total_weight > 0 else None

    result = {
        "manifold_p":   manifold_p,
        "metaculus_p":  metaculus_p,
        "sportsbook_p": sportsbook_p,
        "kalshi_p":     kalshi_p,
        "x_sentiment":  x_sentiment,
        "consensus_p":  consensus_p,
        "sources":      source_count,
    }
    _CACHE[cache_key] = result
    return result


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    test_questions = [
        "Tampa Bay Rays vs. Cleveland Guardians",
        "Will Liverpool FC win on 2026-05-03?",
        "Will the Federal Reserve cut interest rates in 2025?",
    ]
    for q in test_questions:
        print(f"\nQuestion: {q}")
        r = get_external_consensus(
            q,
            odds_api_key=os.getenv("ODDS_API_KEY", ""),
            kalshi_api_key=os.getenv("KALSHI_API_KEY", ""),
        )
        for k, v in r.items():
            if v is not None and k != "sources":
                print(f"  {k:<14} {v}")
        print(f"  sources        {r['sources']}")
