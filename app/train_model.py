"""Train a logistic regression model on historical market data.

Learns which features predict whether the market's implied direction is correct,
then serialises the fitted parameters to data/model_weights.json so model.py
can load them at runtime.

Run:
    python -m app.train_model

Output:
    data/model_weights.json  — coefficients, intercept, scaler params, feature list
    (prints a quick evaluation summary to stdout)
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

DATASET = Path("data/historical_dataset.csv")
WEIGHTS  = Path("data/model_weights.json")

# All categories seen in the dataset.  "other" is the implicit baseline.
KNOWN_CATEGORIES = ["crypto", "esports", "politics", "sports", "stocks", "weather"]


def build_features(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert CSV rows into an (X, y, feature_names) triple for logistic regression.

    Features per row
    ----------------
    yes_price            – raw market probability (0.05–0.95); midlife snapshot
                           after running fetch_historical + build_dataset
    yes_price_sq         – quadratic term to capture non-linearity around extremes
    log_volume           – log(volume+1), captures order-of-magnitude effects
    dist_from_half       – |yes_price - 0.5|, measures market conviction
    momentum_7d          – yes_price change over 7 days (optional; 0.0 if missing)
    cat_<name>           – one-hot category indicators (vs baseline "other")

    Target
    ------
    y = 1 if actual_outcome == "yes", 0 otherwise.
    """
    # Detect whether the dataset contains momentum data
    has_momentum = any("momentum_7d" in row and row["momentum_7d"] not in ("", None) for row in rows)

    X, y = [], []
    for row in rows:
        yp  = float(row["yes_price"])
        vol = float(row["volume"])
        cat = row.get("category", "other") or "other"

        feats = [
            yp,
            yp ** 2,
            math.log(vol + 1),
            abs(yp - 0.5),
        ]

        if has_momentum:
            raw_mom = row.get("momentum_7d", "")
            try:
                feats.append(float(raw_mom) if raw_mom not in ("", None) else 0.0)
            except (TypeError, ValueError):
                feats.append(0.0)

        # category one-hot (omit "other" → baseline)
        for c in KNOWN_CATEGORIES:
            feats.append(1.0 if cat == c else 0.0)

        X.append(feats)
        y.append(1 if row["actual_outcome"] == "yes" else 0)

    cont_names = ["yes_price", "yes_price_sq", "log_volume", "dist_from_half"]
    if has_momentum:
        cont_names.append("momentum_7d")
    feature_names = cont_names + [f"cat_{c}" for c in KNOWN_CATEGORIES]

    return np.array(X), np.array(y), feature_names


def train(verbose: bool = True) -> dict:
    """Fit the model and return a serialisable weights dict."""
    if not DATASET.exists():
        raise FileNotFoundError(
            f"{DATASET} not found — run fetch_historical then build_dataset first."
        )

    with DATASET.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) < 50:
        raise ValueError(
            f"Only {len(rows)} rows — need at least 50 for meaningful training."
        )

    X, y, feature_names = build_features(rows)

    # Number of continuous features to scale; one-hot cols left untouched.
    # 4 base + 1 optional momentum_7d
    N_CONT = 5 if "momentum_7d" in feature_names else 4
    scaler = StandardScaler()
    X[:, :N_CONT] = scaler.fit_transform(X[:, :N_CONT])

    clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=42)
    clf.fit(X, y)

    # --- Evaluation ---
    proba     = clf.predict_proba(X)[:, 1]
    train_auc = roc_auc_score(y, proba)
    train_ll  = log_loss(y, proba)
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")

    if verbose:
        print(f"\nTraining set : {len(rows)} markets  |  "
              f"YES={y.sum()}  NO={len(y)-y.sum()}")
        print(f"Momentum feature included: {'momentum_7d' in feature_names}")
        print(f"Train AUC    : {train_auc:.4f}")
        print(f"Train log-loss: {train_ll:.4f}")
        print(f"CV AUC (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print("\nFeature coefficients:")
        for name, coef in zip(feature_names, clf.coef_[0]):
            print(f"  {name:20s}  {coef:+.4f}")
        print(f"  {'intercept':20s}  {clf.intercept_[0]:+.4f}")

    # --- Serialise ---
    weights = {
        "n_samples": len(rows),
        "feature_names": feature_names,
        "n_continuous": N_CONT,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "train_auc": round(train_auc, 4),
        "cv_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_auc_std": round(float(cv_scores.std()), 4),
        "known_categories": KNOWN_CATEGORIES,
    }

    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    with WEIGHTS.open("w") as f:
        json.dump(weights, f, indent=2)

    if verbose:
        print(f"\nWeights saved → {WEIGHTS}")

    return weights


if __name__ == "__main__":
    train()
