"""Flask web UI for the Quant Bot.

Run with:
    python -m app.web

Then open http://localhost:5000
"""

import math
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for

from app.data_client import fetch_markets_by_days
from app.edge import estimate_edge
from app.market_journal import load_journal_records, update_journal_outcome
from app.model import estimate_probability

# Flask must find templates one level up from app/
TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
app.secret_key = "quant-bot-dev"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edge_label(v: float) -> str:
    if v < 0.02:  return "weak"
    if v < 0.05:  return "moderate"
    return "strong"

def _conf_label(v: float) -> str:
    if v < 0.30:  return "low"
    if v < 0.60:  return "medium"
    return "high"

def _risk_label(v: float) -> str:
    if v < 0.30:  return "low"
    if v < 0.60:  return "medium"
    return "high"


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return render_template(
        "scan.html",
        active="scan",
        results=None,
        params={"min_days": 2, "max_days": 60, "threshold": 0.005},
    )


@app.route("/scan", methods=["POST"])
def scan():
    try:
        min_days   = float(request.form.get("min_days", 2))
        max_days   = float(request.form.get("max_days", 60))
        threshold  = float(request.form.get("threshold", 0.005))
    except ValueError:
        flash("Invalid parameters.", "error")
        return redirect(url_for("index"))

    markets = fetch_markets_by_days(min_days=min_days, max_days=max_days, limit=100)

    results = []
    for market in markets:
        if market.yes_price < 0.05 or market.yes_price > 0.95:
            continue
        if market.volume < 1000:
            continue

        estimate = estimate_probability(market)
        posterior = estimate.posterior
        edge = estimate_edge(market, posterior)

        if edge.preferred_side == "buy_yes":
            signal_edge   = edge.final_signal_yes
            adjusted_edge = edge.adjusted_edge_yes
        else:
            signal_edge   = edge.final_signal_no
            adjusted_edge = edge.adjusted_edge_no

        if adjusted_edge < threshold:
            continue

        results.append({
            "market_id":       market.market_id,
            "question":        market.question,
            "days":            market.days_to_resolution or 0,
            "side":            edge.preferred_side,
            "signal_edge":     signal_edge,
            "signal_label":    _edge_label(adjusted_edge),
            "adjusted_edge":   adjusted_edge,
            "posterior":       posterior,
            "yes_price":       market.yes_price,
            "confidence":      edge.confidence,
            "confidence_label":_conf_label(edge.confidence),
            "risk_score":      edge.risk_score,
            "risk_label":      _risk_label(edge.risk_score),
            "volume":          market.volume,
            "liquidity":       market.liquidity_depth,
            "rationale":       estimate.rationale,
        })

    results.sort(key=lambda x: x["signal_edge"], reverse=True)

    return render_template(
        "scan.html",
        active="scan",
        results=results,
        params={"min_days": min_days, "max_days": max_days, "threshold": threshold},
    )


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------

@app.route("/journal")
def journal():
    records = load_journal_records()
    return render_template("journal.html", active="journal", records=records)


@app.route("/update_outcome", methods=["POST"])
def update_outcome():
    market_id = request.form.get("market_id", "").strip()
    outcome   = request.form.get("outcome", "").strip().lower()
    notes     = request.form.get("notes", "").strip()

    if not market_id or outcome not in ("yes", "no"):
        flash("Market ID and a valid outcome (yes/no) are required.", "error")
        return redirect(url_for("journal"))

    updated = update_journal_outcome(market_id, outcome, notes)
    if updated:
        flash(f"Outcome for {market_id} saved as '{outcome}'.", "success")
    else:
        flash(f"No journal entry found for market ID '{market_id}'.", "error")

    return redirect(url_for("journal"))


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

def _run_analytics():
    """Run the analytics pipeline and return structured data for the template."""
    import csv
    from types import SimpleNamespace
    from app.state import settings

    dataset_path = Path("data/historical_dataset.csv")
    if not dataset_path.exists():
        return None

    THRESHOLDS = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050]
    START = float(settings.bankroll)

    with dataset_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    # Score every row
    scored = []
    for row in rows:
        yp = float(row["yes_price"])
        market = SimpleNamespace(
            market_id=row["market_id"], question=row["question"],
            yes_price=yp, no_price=round(1 - yp, 4),
            volume=float(row["volume"]), liquidity_depth=float(row["liquidity_depth"]),
            days_to_resolution=None,
        )
        est  = estimate_probability(market)
        edge = estimate_edge(market, est.posterior)

        if edge.preferred_side == "buy_yes":
            sig   = edge.adjusted_edge_yes
            price = market.yes_price
            wins  = row["actual_outcome"] == "yes"
        else:
            sig   = edge.adjusted_edge_no
            price = market.no_price
            wins  = row["actual_outcome"] == "no"

        scored.append({
            "category": row.get("category", "other"),
            "preferred_side": edge.preferred_side,
            "signal_edge": sig,
            "trade_price": price,
            "posterior": est.posterior,
            "yes_price": yp,
            "bot_wins": wins,
            "risk_multiplier": edge.risk_multiplier,
            "confidence": edge.confidence,
        })

    # Calibration
    cal_buckets: dict = {}
    for rec in scored:
        p = rec["posterior"]
        pred_p = p if rec["preferred_side"] == "buy_yes" else 1 - p
        label = f"{int(pred_p * 10) * 10}–{int(pred_p * 10) * 10 + 10}%"
        cal_buckets.setdefault(label, []).append(rec["bot_wins"])

    cal_rows = []
    total_gap = 0.0
    for label in sorted(cal_buckets, key=lambda x: int(x.split("–")[0])):
        wins_list = cal_buckets[label]
        pred_mid = (int(label.split("–")[0]) + 5) / 100
        actual   = sum(wins_list) / len(wins_list) if wins_list else 0
        gap      = actual - pred_mid
        total_gap += abs(gap)
        cal_rows.append({
            "bucket": label, "actual": f"{actual:.1%}",
            "count": len(wins_list), "gap": f"{gap:+.1%}", "gap_f": gap,
        })
    cal_mae = f"{total_gap / len(cal_buckets):.1%}" if cal_buckets else "—"

    # Threshold sweep
    thr_rows = []
    best_roi_f = float("-inf")
    best_thr = THRESHOLDS[0]
    for thr in THRESHOLDS:
        bankroll = START
        peak = START
        pnls, wins_t = [], 0
        for rec in scored:
            if rec["signal_edge"] < thr:  continue
            if not (0.05 <= rec["yes_price"] <= 0.95): continue
            size  = 100 * rec["risk_multiplier"]
            price = rec["trade_price"]
            pnl   = size * ((1 / price) - 1) if rec["bot_wins"] else -size
            bankroll += pnl
            peak = max(peak, bankroll)
            pnls.append(pnl)
            if rec["bot_wins"]: wins_t += 1

        n = len(pnls)
        if n == 0:
            thr_rows.append({"threshold": f"{thr:.3f}", "trades": "—",
                             "win_rate": "—", "win_rate_f": 0,
                             "roi": "—", "roi_f": 0, "sharpe": "—", "best": False})
            continue

        wr   = wins_t / n
        roi  = (bankroll - START) / START
        avg  = sum(pnls) / n
        std  = math.sqrt(sum((p - avg) ** 2 for p in pnls) / n) if n > 1 else 0
        shr  = avg / std if std > 0 else 0

        is_best = roi > best_roi_f and n >= 5
        if is_best:
            best_roi_f = roi
            best_thr = thr

        thr_rows.append({
            "threshold": f"{thr:.3f}", "trades": n,
            "win_rate": f"{wr:.1%}", "win_rate_f": wr,
            "roi": f"{roi:+.1%}", "roi_f": roi,
            "sharpe": f"{shr:.2f}", "best": False,
        })

    for row in thr_rows:
        row["best"] = row["threshold"] == f"{best_thr:.3f}"

    # Side bias
    side_rows = []
    for side in ("buy_yes", "buy_no"):
        trades = [r for r in scored if r["preferred_side"] == side]
        wr = sum(1 for r in trades if r["bot_wins"]) / len(trades) if trades else 0
        side_rows.append({"side": side, "count": len(trades),
                          "win_rate": f"{wr:.1%}", "win_rate_f": wr})

    # Category
    cats: dict = {}
    for rec in scored:
        c = rec["category"] or "other"
        cats.setdefault(c, {"n": 0, "w": 0})
        cats[c]["n"] += 1
        if rec["bot_wins"]: cats[c]["w"] += 1
    cat_rows = sorted(
        [{"category": c, "count": v["n"],
          "win_rate": f"{v['w']/v['n']:.1%}" if v["n"] else "—",
          "win_rate_f": v["w"] / v["n"] if v["n"] else 0}
         for c, v in cats.items()],
        key=lambda x: x["count"], reverse=True
    )

    # Overall
    all_wins = sum(1 for r in scored if r["bot_wins"])
    ov_wr    = f"{all_wins / len(scored):.1%}" if scored else "—"

    # Drawdown at best threshold
    bankroll = START
    peak = START
    max_dd = 0.0
    for rec in scored:
        if rec["signal_edge"] < best_thr: continue
        size  = 100 * rec["risk_multiplier"]
        price = rec["trade_price"]
        pnl   = size * ((1 / price) - 1) if rec["bot_wins"] else -size
        bankroll += pnl
        peak = max(peak, bankroll)
        dd = (peak - bankroll) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return {
        "dataset_size":   len(rows),
        "overall_win_rate": ov_wr,
        "best_roi":       f"{best_roi_f:+.1%}" if best_roi_f != float('-inf') else "—",
        "best_roi_f":     best_roi_f if best_roi_f != float('-inf') else 0,
        "best_threshold": f"{best_thr:.3f}",
        "max_drawdown":   f"{max_dd:.1%}",
        "threshold_rows": thr_rows,
        "calibration_rows": cal_rows,
        "calibration_mae":  cal_mae,
        "side_rows":      side_rows,
        "category_rows":  cat_rows,
    }


@app.route("/analytics")
def analytics():
    from pathlib import Path as _P
    import json as _json
    weights_path = _P("data/model_weights.json")
    model_meta = None
    if weights_path.exists():
        with weights_path.open() as f:
            w = _json.load(f)
        model_meta = {
            "n_samples":   w.get("n_samples", "?"),
            "cv_auc_mean": w.get("cv_auc_mean", "?"),
            "cv_auc_std":  w.get("cv_auc_std", "?"),
        }
    data = _run_analytics()
    if data is None:
        return render_template("analytics.html", active="analytics",
                               dataset_size=0, model_meta=model_meta)
    return render_template("analytics.html", active="analytics",
                           model_meta=model_meta, **data)


@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        from app.train_model import train
        w = train(verbose=False)
        # Invalidate cached weights so model.py reloads on next request
        import app.model as _m
        _m._weights = None
        flash(
            f"Model retrained on {w['n_samples']} markets  |  "
            f"CV AUC {w['cv_auc_mean']:.3f} ± {w['cv_auc_std']:.3f}",
            "success",
        )
    except Exception as exc:
        flash(f"Retrain failed: {exc}", "error")
    return redirect(url_for("analytics"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("Starting Quant Bot UI at http://localhost:8080")
    app.run(debug=False, port=8080)


if __name__ == "__main__":
    main()
