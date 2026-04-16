"""Fetch external consensus probabilities from Manifold, Metaculus, and X (Twitter).

These provide genuinely independent information the Polymarket price may not
have fully priced in.  The consensus_p returned here is blended with the
trained model posterior in the main scan pipelines.

Match quality is measured by string similarity against the market question.
A match below MATCH_THRESHOLD is discarded rather than used — a wrong signal
is worse than no signal.

Caching: results are stored per process in _CACHE so a batch scan does not
hammer external APIs with duplicate questions.

X / Twitter setup
-----------------
Set TWITTER_BEARER_TOKEN in your .env file.  Without it, the X signal is
skipped gracefully.  The signal uses recent-tweet sentiment: the ratio of
bullish to bearish keyword mentions around the question topic.
"""

import difflib
import hashlib
from typing import Optional

import requests

MANIFOLD_SEARCH_URL = "https://api.manifold.markets/v0/search-markets"
METACULUS_SEARCH_URL = "https://www.metaculus.com/api2/questions/"
TWITTER_SEARCH_URL   = "https://api.twitter.com/2/tweets/search/recent"

# Minimum text similarity (0–1) to accept an external match
MATCH_THRESHOLD = 0.55

# Per-request timeout in seconds
REQUEST_TIMEOUT = 5

# Sentiment keywords for X signal
_BULLISH_WORDS = {"yes", "will", "likely", "expect", "confirm", "happen",
                  "true", "bullish", "surge", "win", "pass", "approved"}
_BEARISH_WORDS = {"no", "won't", "wont", "unlikely", "doubt", "bearish",
                  "false", "fail", "reject", "denied", "lose", "drop"}

# In-process cache: question_hash → result dict
_CACHE: dict[str, dict] = {}


def _question_hash(question: str) -> str:
    return hashlib.md5(question.lower().strip().encode()).hexdigest()


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _clamp(p: float) -> float:
    return round(max(0.02, min(0.98, p)), 4)


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

    if best_score >= MATCH_THRESHOLD and best_p is not None:
        return _clamp(best_p)
    return None


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
        full = prediction.get("full") or {}
        p = full.get("q2")  # community median
        if score > best_score and p is not None:
            best_score = score
            best_p = float(p)

    if best_score >= MATCH_THRESHOLD and best_p is not None:
        return _clamp(best_p)
    return None


# ---------------------------------------------------------------------------
# X (Twitter) signal
# ---------------------------------------------------------------------------

_X_ACCESS_DENIED = False  # flipped to True on first 403 so we stop retrying


def fetch_x_sentiment(question: str, bearer_token: str) -> Optional[float]:
    """Return a sentiment-derived probability from recent X posts.

    Searches for the last 100 tweets about the question topic and computes
    a bullish ratio from keyword counts.  Returns None if there are not enough
    relevant tweets (< 5) to form a reliable signal.

    This is a rough directional signal, not a calibrated probability.
    """
    # Extract a compact search query from the question (first ~60 chars, no punctuation)
    import re
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
            _X_ACCESS_DENIED = True  # stop retrying for this process
            return None
        if resp.status_code != 200:
            return None
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None

    tweets = data.get("data", [])
    if not tweets:
        return None

    bullish = 0
    bearish = 0
    for tweet in tweets:
        text_lower = tweet.get("text", "").lower()
        words = set(text_lower.split())
        bullish += len(words & _BULLISH_WORDS)
        bearish += len(words & _BEARISH_WORDS)

    total = bullish + bearish
    if total < 5:
        return None  # not enough signal

    sentiment_p = _clamp(bullish / total)
    return sentiment_p


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_external_consensus(question: str, twitter_bearer_token: str = "") -> dict:
    """Return a blended external probability estimate for a question.

    Result dict keys:
      manifold_p   — Manifold probability (float or None)
      metaculus_p  — Metaculus community median (float or None)
      x_sentiment  — X/Twitter sentiment probability (float or None)
      consensus_p  — Weighted average of available signals (float or None)
      sources      — number of sources that returned a valid probability
    """
    key = _question_hash(question + twitter_bearer_token[:8])
    if key in _CACHE:
        return _CACHE[key]

    manifold_p  = fetch_manifold_probability(question)
    metaculus_p = fetch_metaculus_probability(question)
    x_sentiment = (
        fetch_x_sentiment(question, twitter_bearer_token)
        if twitter_bearer_token
        else None
    )

    # Weight X sentiment at half strength vs structured forecasts
    weighted: list[float] = []
    if manifold_p  is not None: weighted.append(manifold_p)
    if metaculus_p is not None: weighted.append(metaculus_p)
    if x_sentiment is not None: weighted.append(x_sentiment * 0.5 + 0.25)  # dampen toward 0.5

    consensus_p = _clamp(sum(weighted) / len(weighted)) if weighted else None

    result = {
        "manifold_p":  manifold_p,
        "metaculus_p": metaculus_p,
        "x_sentiment": x_sentiment,
        "consensus_p": consensus_p,
        "sources":     len(weighted),
    }
    _CACHE[key] = result
    return result


if __name__ == "__main__":
    test_question = "Will the Federal Reserve cut interest rates in 2025?"
    print(f"Question: {test_question}")
    result = get_external_consensus(test_question)
    print(f"  Manifold:   {result['manifold_p']}")
    print(f"  Metaculus:  {result['metaculus_p']}")
    print(f"  Consensus:  {result['consensus_p']}")
    print(f"  Sources:    {result['sources']}")
