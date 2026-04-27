"""GPT-4o web search analyst for prediction market signals.

Calls gpt-4o-search-preview to find current real-world context for a market
question and returns a verdict: confirm | reject | neutral.

Only called on candidates with adjusted_edge >= GPT_MIN_EDGE to control cost.
Results are cached per scan process so the same question is never called twice.

Cost: ~$0.03-0.05 per call (search + tokens). Capped at GPT_CAP calls per scan
run in scan_near_term.py.
"""

import hashlib
import os
from typing import Optional

GPT_MIN_EDGE = 0.02          # only call GPT on markets with >= 2% adjusted edge
GPT_MODEL    = "gpt-4o-search-preview"

_SYSTEM = (
    "You are a prediction market analyst with access to real-time web search. "
    "Your job is to find current, factual information relevant to a market question "
    "and judge whether the signal direction looks correct. Be brief and factual."
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


def analyze(
    question: str,
    yes_price: float,
    posterior: float,
    preferred_side: str,
    adjusted_edge: float,
) -> dict:
    """Return a GPT web-search verdict for a market signal.

    Args:
        question:       Polymarket question text
        yes_price:      Current market YES price (0-1)
        posterior:      Model's estimated probability (0-1)
        preferred_side: 'buy_yes' or 'buy_no'
        adjusted_edge:  Model's adjusted edge (0-1)

    Returns dict with:
        verdict   — 'confirm' | 'reject' | 'neutral' | 'skipped' | 'error'
        reasoning — one-sentence explanation
    """
    if adjusted_edge < GPT_MIN_EDGE:
        return {"verdict": "skipped", "reasoning": "edge below GPT threshold"}

    client = _get_client()
    if client is None:
        return {"verdict": "skipped", "reasoning": "OPENAI_API_KEY not set"}

    cache_key = hashlib.md5(question.lower().strip().encode()).hexdigest()
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    side_label = "YES" if preferred_side == "buy_yes" else "NO"
    prompt = (
        f"Prediction market question: {question}\n"
        f"Current market price: {yes_price:.0%} YES\n"
        f"Model estimate: {posterior:.0%} YES  →  signal is buy {side_label} "
        f"(edge: {adjusted_edge:.1%})\n\n"
        "Search for current information relevant to this question — news, "
        "recent results, injuries, context, or anything that affects the outcome.\n\n"
        "Reply in EXACTLY this format with no other text:\n"
        "VERDICT: confirm\n"
        "REASON: <one sentence>\n\n"
        "Rules:\n"
        "- 'confirm' if evidence supports buying " + side_label + "\n"
        "- 'reject' if evidence contradicts buying " + side_label + "\n"
        "- 'neutral' if you found nothing useful or the evidence is mixed\n"
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
        verdict, reasoning = _parse_response(text)
    except Exception as exc:
        result = {"verdict": "error", "reasoning": str(exc)[:120]}
        _CACHE[cache_key] = result
        return result

    result = {"verdict": verdict, "reasoning": reasoning}
    _CACHE[cache_key] = result
    return result


def _parse_response(text: str) -> tuple[str, str]:
    """Extract verdict and reasoning from GPT response."""
    verdict   = "neutral"
    reasoning = text[:200]  # fallback: raw response truncated

    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            raw = line.split(":", 1)[1].strip().lower()
            if "confirm" in raw:
                verdict = "confirm"
            elif "reject" in raw:
                verdict = "reject"
            else:
                verdict = "neutral"
        elif line.upper().startswith("REASON:"):
            reasoning = line.split(":", 1)[1].strip()

    return verdict, reasoning
