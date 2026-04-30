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

from dotenv import load_dotenv
load_dotenv()

GPT_MIN_EDGE = 0.02          # only call GPT on markets with >= 2% adjusted edge
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
        reasoning — evidence summary + reason
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

    # Pre-compute the explicit mapping so GPT doesn't have to infer it.
    # This is the key fix for direction confusion: GPT states YES_LIKELY first,
    # then the rules map that unambiguously to confirm/reject.
    if preferred_side == "buy_yes":
        higher_maps_to = "confirm"
        lower_maps_to  = "reject"
    else:
        higher_maps_to = "reject"
        lower_maps_to  = "confirm"

    prompt = (
        f'Prediction market question: "{question}"\n'
        f"This market resolves YES if the stated outcome occurs, NO otherwise.\n"
        f"Current market YES price: {yes_price:.0%}  |  Model estimate: {posterior:.0%} YES\n"
        f"Signal: buy {side_label} (edge {adjusted_edge:.1%})\n\n"
        "Search for the most current, factual information about this question.\n\n"
        "Reply in EXACTLY this 4-line format — no extra text, no markdown:\n"
        "EVIDENCE: <one sentence: the key fact you found>\n"
        "YES_LIKELY: higher / lower / unchanged\n"
        "VERDICT: confirm / reject / neutral\n"
        "REASON: <one sentence explaining the verdict>\n\n"
        "Mapping rules (follow these exactly):\n"
        f"  YES_LIKELY=higher   → VERDICT={higher_maps_to}\n"
        f"  YES_LIKELY=lower    → VERDICT={lower_maps_to}\n"
        f"  YES_LIKELY=unchanged → VERDICT=neutral\n"
        "  If you found nothing useful, set YES_LIKELY=unchanged and VERDICT=neutral.\n"
    )

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=250,
        )
        text = response.choices[0].message.content.strip()
        verdict, reasoning = _parse_response(text)
    except Exception as exc:
        result = {"verdict": "error", "reasoning": str(exc)[:200]}
        _CACHE[cache_key] = result
        return result

    result = {"verdict": verdict, "reasoning": reasoning}
    _CACHE[cache_key] = result
    return result


def _parse_response(text: str) -> tuple[str, str]:
    """Extract verdict and reasoning from GPT response.

    Combines EVIDENCE + REASON into the reasoning field so the journal
    captures both the raw finding and the verdict explanation.
    """
    verdict  = "neutral"
    evidence = ""
    reason   = text[:300]  # fallback: raw response truncated

    for line in text.splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("VERDICT:"):
            raw = line.split(":", 1)[1].strip().lower()
            if "confirm" in raw:
                verdict = "confirm"
            elif "reject" in raw:
                verdict = "reject"
            else:
                verdict = "neutral"
        elif upper.startswith("EVIDENCE:"):
            evidence = line.split(":", 1)[1].strip()
        elif upper.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    # Combine evidence + reason for richer journal logging
    if evidence and reason and evidence != reason:
        reasoning = f"{evidence} → {reason}"
    else:
        reasoning = reason or evidence

    return verdict, reasoning
