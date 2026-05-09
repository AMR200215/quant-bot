"""GPT-4o web search analyst for prediction market signals.

Two modes:

  estimate   — Used for non-sports markets (politics, finance, general) where
               no sharp sportsbook signal exists. GPT searches the web and
               returns its own probability estimate for YES, blind to the bot's
               view. That probability is fed into the external consensus as a
               real signal (weight 2.0).

  news_check — Used for sports markets where Odds API already provides a sharp
               probability. GPT checks for breaking news (injury, suspension,
               lineup change) in the last 48h that hasn't been priced in yet.
               Returns an alert flag, not a probability.

In both modes GPT never sees the bot's posterior, preferred side, or edge.
The comparison between GPT's view and the bot's view happens in code, not
inside the prompt. This keeps GPT genuinely independent.

Cost: ~$0.03-0.05 per call. Capped at GPT_CAP calls per scan run.
"""

import hashlib
import os

from dotenv import load_dotenv
load_dotenv()

GPT_MIN_EDGE = 0.02
GPT_MODEL    = "gpt-4o-search-preview"

_SYSTEM = (
    "You are a prediction market analyst with access to real-time web search. "
    "Your job is to find current, factual information relevant to a market question "
    "and output a structured verdict using the exact format requested."
)

_CACHE: dict[str, dict] = {}
_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return None
        _client = OpenAI(api_key=api_key)
        return _client
    except ImportError:
        return None


def estimate_probability(question: str, yes_price: float) -> dict:
    """Ask GPT to estimate YES probability independently.

    GPT sees only the question and the current market price.
    It does NOT see the bot's posterior, edge, or preferred side.
    Returns:
        gpt_p     — float probability (0-1) or None
        evidence  — key fact found
        verdict   — 'confirm' | 'reject' | 'neutral' | 'error' (derived in code)
        reasoning — evidence summary
    """
    client = _get_client()
    if client is None:
        return {"gpt_p": None, "verdict": "skipped", "reasoning": "OPENAI_API_KEY not set"}

    cache_key = hashlib.md5(("estimate:" + question.lower().strip()).encode()).hexdigest()
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    prompt = (
        f'Prediction market question: "{question}"\n'
        f"Current market YES price: {yes_price:.0%}\n\n"
        "Search for the most current, factual information about this question.\n"
        "Based only on what you find, estimate the probability this resolves YES.\n\n"
        "Reply in EXACTLY this 3-line format — no extra text, no markdown:\n"
        "EVIDENCE: <one sentence: the key fact you found>\n"
        "YES_PROBABILITY: <a number between 0 and 100, e.g. 65>\n"
        "REASON: <one sentence explaining your estimate>\n\n"
        "If you cannot find useful information, set YES_PROBABILITY: 50 and say so in REASON."
    )

    try:
        response = _get_client().chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=200,
        )
        text = response.choices[0].message.content.strip()
        result = _parse_estimate(text, yes_price)
    except Exception as exc:
        result = {"gpt_p": None, "verdict": "error", "reasoning": str(exc)[:200]}

    _CACHE[cache_key] = result
    return result


def news_check(question: str) -> dict:
    """Check for breaking news that could affect a sports market outcome.

    Used when Odds API already provides a sharp probability — GPT's job here
    is not to reprice the market but to flag information that hasn't been
    priced in yet (injury, suspension, lineup scratch, weather).

    Returns:
        alert     — True if significant breaking news found
        reasoning — description of the news or 'no breaking news found'
        verdict   — 'reject' if alert else 'neutral'
    """
    client = _get_client()
    if client is None:
        return {"alert": False, "verdict": "neutral", "reasoning": "OPENAI_API_KEY not set"}

    cache_key = hashlib.md5(("news:" + question.lower().strip()).encode()).hexdigest()
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    prompt = (
        f'Prediction market question: "{question}"\n\n'
        "Search for breaking news in the last 48 hours that could significantly "
        "affect the outcome of this event — e.g. key player injury, suspension, "
        "lineup change, weather cancellation, or other major development.\n\n"
        "Reply in EXACTLY this 2-line format — no extra text, no markdown:\n"
        "ALERT: yes / no\n"
        "NEWS: <one sentence describing the news, or 'no significant breaking news found'>\n"
    )

    try:
        response = _get_client().chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=150,
        )
        text = response.choices[0].message.content.strip()
        result = _parse_news(text)
    except Exception as exc:
        result = {"alert": False, "verdict": "neutral", "reasoning": str(exc)[:200]}

    _CACHE[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_estimate(text: str, yes_price: float) -> dict:
    """Parse GPT probability estimate response."""
    gpt_p    = None
    evidence = ""
    reason   = text[:300]

    for line in text.splitlines():
        line  = line.strip()
        upper = line.upper()
        if upper.startswith("YES_PROBABILITY:"):
            raw = line.split(":", 1)[1].strip()
            try:
                val = float(raw.replace("%", ""))
                # Accept either 0-100 or 0-1 scale
                gpt_p = val / 100.0 if val > 1 else val
                gpt_p = max(0.02, min(0.98, gpt_p))
            except ValueError:
                pass
        elif upper.startswith("EVIDENCE:"):
            evidence = line.split(":", 1)[1].strip()
        elif upper.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    reasoning = f"{evidence} → {reason}" if evidence and reason and evidence != reason else (reason or evidence)

    # Derive verdict from GPT probability vs market price — no anchoring
    if gpt_p is None:
        verdict = "neutral"
    elif gpt_p > yes_price + 0.05:
        verdict = "confirm_yes"   # GPT thinks YES is underpriced
    elif gpt_p < yes_price - 0.05:
        verdict = "confirm_no"    # GPT thinks NO is underpriced
    else:
        verdict = "neutral"       # GPT roughly agrees with market

    return {"gpt_p": gpt_p, "verdict": verdict, "reasoning": reasoning}


def _parse_news(text: str) -> dict:
    alert  = False
    news   = "no significant breaking news found"

    for line in text.splitlines():
        line  = line.strip()
        upper = line.upper()
        if upper.startswith("ALERT:"):
            alert = "yes" in line.lower()
        elif upper.startswith("NEWS:"):
            news = line.split(":", 1)[1].strip()

    verdict = "reject" if alert else "neutral"
    return {"alert": alert, "verdict": verdict, "reasoning": news}


# ---------------------------------------------------------------------------
# Legacy shim — keeps backfill_gpt.py working without changes
# ---------------------------------------------------------------------------

def analyze(
    question: str,
    yes_price: float,
    posterior: float,
    preferred_side: str,
    adjusted_edge: float,
) -> dict:
    """Backward-compatible wrapper used by backfill_gpt.py.

    Routes to estimate_probability() and maps the result back to the
    old confirm/reject/neutral/skipped schema so existing journal rows
    are not affected.
    """
    if adjusted_edge < GPT_MIN_EDGE:
        return {"verdict": "skipped", "reasoning": "edge below GPT threshold"}

    result = estimate_probability(question, yes_price)
    gpt_p  = result.get("gpt_p")

    if gpt_p is None:
        verdict = result.get("verdict", "neutral")
        if verdict not in ("skipped", "error"):
            verdict = "neutral"
    else:
        # Map back to old schema using preferred_side
        if preferred_side == "buy_yes":
            verdict = "confirm" if gpt_p > yes_price + 0.05 else ("reject" if gpt_p < yes_price - 0.05 else "neutral")
        else:
            verdict = "confirm" if gpt_p < yes_price - 0.05 else ("reject" if gpt_p > yes_price + 0.05 else "neutral")

    return {"verdict": verdict, "reasoning": result.get("reasoning", "")}
