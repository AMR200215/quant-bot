"""Model layer for DMDO probability estimation.

Two-tier design:

1. **Trained model** (primary): Logistic regression fit on historical closed markets.
   Loads coefficients from data/model_weights.json at import time.
   Produces a calibrated P(YES) posterior for each market.

2. **Heuristic fallback**: Volume-weighted Bayesian amplification used when
   the weights file doesn't exist (e.g., first run before training).

The trained model uses: yes_price, yes_price², log(volume), |yes_price − 0.5|,
and one-hot category dummies.  It learns which direction markets tend to resolve
given those features — generating genuine YES *and* NO signals based on data,
not hardcoded logic.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from app.bayes import bayes_update
from app.calibration_correction import apply_calibration
from app.data_client import Market

_WEIGHTS_PATH = Path("data/model_weights.json")
_weights: dict | None = None


def _load_weights() -> dict | None:
    """Load model weights once at first call."""
    global _weights
    if _weights is not None:
        return _weights
    if _WEIGHTS_PATH.exists():
        with _WEIGHTS_PATH.open() as f:
            _weights = json.load(f)
    return _weights


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _trained_posterior(market: Market, w: dict) -> tuple[float, str, float]:
    """Compute P(YES) using the trained logistic regression weights.

    Returns (calibrated_posterior, rationale, logit).
    The logit (pre-sigmoid dot product) is exposed so callers can use its
    magnitude as a model conviction signal.
    """
    yp  = market.yes_price
    vol = market.volume
    cat = getattr(market, "category", None) or "other"

    known_cats    = w["known_categories"]
    n_cont        = w["n_continuous"]
    mean          = w["scaler_mean"]
    scale         = w["scaler_scale"]
    coef          = w["coef"]
    intercept     = w["intercept"]
    feature_names = w.get("feature_names", [])
    has_momentum  = "momentum_7d" in feature_names

    # Build raw feature vector (same order as train_model.py)
    momentum = getattr(market, "momentum_7d", None) or 0.0
    base = [
        yp,
        yp ** 2,
        math.log(vol + 1),
        abs(yp - 0.5),
    ]
    if has_momentum:
        base.append(momentum)

    raw = base + [1.0 if cat == c else 0.0 for c in known_cats]

    # Scale continuous features
    feats = list(raw)
    for i in range(n_cont):
        feats[i] = (raw[i] - mean[i]) / scale[i]

    # Logistic regression prediction
    dot = sum(c * f for c, f in zip(coef, feats)) + intercept
    raw_posterior = _sigmoid(dot)
    raw_posterior = round(max(0.02, min(0.98, raw_posterior)), 4)

    # Apply calibration correction (fixes systematic overconfidence in 20-50% range)
    posterior = apply_calibration(raw_posterior)

    direction = "YES" if posterior > yp else "NO"
    auc_note  = f"CV AUC {w.get('cv_auc_mean', '?')}"
    rationale = (
        f"Trained model ({auc_note}): posterior={posterior:.3f} "
        f"(raw={raw_posterior:.3f}) vs market={yp:.3f}  "
        f"→ edge on {direction}  |  "
        f"vol={vol:,.0f}  cat={cat}"
    )
    return posterior, rationale, dot


@dataclass
class ModelEstimate:
    market_id: str
    prior: float
    likelihood_true: float
    likelihood_false: float
    posterior: float
    rationale: str
    logit: float = field(default=0.0)


# ---------------------------------------------------------------------------
# Heuristic fallback (used when weights file is missing)
# ---------------------------------------------------------------------------

def _heuristic_evidence_weights(market: Market) -> tuple[float, float, str]:
    """Volume-weighted Bayesian amplification (no training required)."""
    if market.volume >= 500_000:
        trust_factor = 0.14
    elif market.volume >= 150_000:
        trust_factor = 0.12
    elif market.volume >= 50_000:
        trust_factor = 0.10
    elif market.volume >= 10_000:
        trust_factor = 0.07
    else:
        trust_factor = 0.04

    if market.liquidity_depth >= 50_000:
        trust_factor = min(trust_factor + 0.03, 0.18)
    elif market.liquidity_depth < 3_000:
        trust_factor = max(trust_factor - 0.02, 0.01)

    if market.yes_price >= 0.5:
        lt, lf = 0.5 + trust_factor, 0.5 - trust_factor
        direction = "YES (heuristic)"
    else:
        lt, lf = 0.5 - trust_factor, 0.5 + trust_factor
        direction = "NO (heuristic)"

    rationale = (
        f"Heuristic fallback — {direction};  "
        f"trust={trust_factor:.3f}  vol={market.volume:,.0f}  "
        f"liq={market.liquidity_depth:,.0f}.  "
        f"Run 'python -m app.train_model' to enable data-driven predictions."
    )
    return lt, lf, rationale


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_probability(market: Market) -> ModelEstimate:
    """Estimate a market probability.

    Uses the trained logistic regression if weights exist, otherwise falls back
    to the heuristic Bayesian amplification.
    """
    prior = market.yes_price
    w     = _load_weights()

    logit = 0.0
    if w is not None:
        posterior, rationale, logit = _trained_posterior(market, w)
        # Derive pseudo likelihood-true/false for downstream compatibility
        # (edge.py uses these only to pass through; the posterior is what matters)
        delta = posterior - 0.5
        likelihood_true  = round(0.5 + delta, 4)
        likelihood_false = round(0.5 - delta, 4)
    else:
        lt, lf, rationale = _heuristic_evidence_weights(market)
        posterior          = bayes_update(prior, lt, lf)
        likelihood_true    = lt
        likelihood_false   = lf

    return ModelEstimate(
        market_id=market.market_id,
        prior=prior,
        likelihood_true=likelihood_true,
        likelihood_false=likelihood_false,
        posterior=posterior,
        rationale=rationale,
        logit=logit,
    )


if __name__ == "__main__":
    from types import SimpleNamespace
    sample = SimpleNamespace(
        market_id="btc-90k",
        question="Will BTC close above 90k?",
        yes_price=0.58,
        no_price=0.42,
        volume=185_000.0,
        liquidity_depth=25_000.0,
        category="crypto",
    )
    est = estimate_probability(sample)
    print(est)
