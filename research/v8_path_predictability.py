"""
research/v8_path_predictability.py — YD-BATCH item YD1.

Read-only analysis. Does NOT modify research/v8_collection_yield.py's
gating logic, does NOT edit research/v8_candidate_registry.py's frozen
candidates, does NOT touch any live threshold. Composes the SAME
already-tested primitives compute_collection_yield() itself uses
(trustworthy_collection_era_start, load_admission_log_by_mint,
find_ambiguous_mints, resolve_entry_alignment, assess_path_integrity)
independently, at per-mint granularity, joined to alert-time features --
rather than modifying that already-deployed, already-gating module.

Question: over the funded-era admitted population, does anything
knowable AT ALERT TIME (before any trade data exists) predict whether a
token goes on to produce a usable path (admitted_with_valid_usable_path,
same definition v8_collection_yield.py uses)?

Reference candidate for entry-alignment/usability resolution: V8-P0 --
"no numerical progress cutoff beyond valid CURVE_ACTIVE venue", the
broadest of the four frozen candidates, T0-only decision delay (zero
extra T0+capture features). Chosen because it makes the usability check
as general as possible rather than tied to a narrower candidate's own
timing assumptions -- this is a data-availability question, not a
per-candidate profitability question.

Holdout note: this is NOT an entry-candidate EV/profitability question.
It never reads pct_change_peak or any outcome/profit field for ANY row,
holdout or otherwise -- it only asks whether alert-time-known features
predict a DATA-COLLECTION fact (did a usable price trail get captured).
No train/validation/holdout split is applied to this population; the
funded-era admitted population is used in full.

Feature-enforcement discipline: every feature is checked against
research/v8_feature_enforcement.py's allowed_for_entry=true gate before
being offered as a "pre-registerable condition". A feature can be
reported on for its raw correlation (informational only) without being
eligible for the draft-candidate step -- e.g. top10_holder_pct and
creator_holds_pct are both allowed_for_entry=false in the registry and
will never be proposed as a candidate condition here, regardless of
what their raw numbers show. hour_of_day is not a registered feature at
all (ad hoc, computed here) -- also never eligible to be proposed
without first being added to v8_feature_registry.yaml.

Run:
    python -m research.v8_path_predictability
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from research.v8_collection_yield import trustworthy_collection_era_start, load_admission_log_by_mint
from research.v8_entry_alignment import find_ambiguous_mints, resolve_entry_alignment, EntryAlignmentExclusion
from research.v8_path_integrity import assess_path_integrity, PathIntegrityStatus
from research.path_schema import load_path_file
from research.v8_candidate_registry import CANDIDATES
from research.v8_feature_enforcement import check_features_allowed

_REFERENCE_CANDIDATE = next(c for c in CANDIDATES if c["candidate_id"] == "V8-P0")

# Matches research/v8_readiness_engine.py's MIN_SPLIT_BUCKET_N -- the
# same "don't trust a cell this small" convention already used elsewhere
# in this project, not a new number invented for this analysis.
MIN_CELL_N = 20

# "Material lift" bar for offering a condition as a candidate draft --
# stated explicitly, not silently chosen after seeing the numbers.
MATERIAL_LIFT_ABS_PP = 15.0     # +15 percentage points over baseline
MATERIAL_LIFT_MIN_N = 30        # the lifted subpopulation must itself be a real sample

_PROGRESS_BUCKETS = [(0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 0.90), (0.90, 2.0)]
_PROGRESS_LABELS = ["0-25%", "25-50%", "50-75%", "75-90%", "90%+"]

_VSOL_BUCKETS = [(0, 10), (10, 30), (30, 60), (60, 100), (100, 1e9)]
_VSOL_LABELS = ["<10", "10-30", "30-60", "60-100", "100+"]

_VELOCITY_BUCKETS = [(0, 1), (1, 3), (3, 6), (6, 1e9)]
_VELOCITY_LABELS = ["0", "1-2", "3-5", "6+"]

_REALERT_BUCKETS = [(0, 1), (1, 2), (2, 1e9)]
_REALERT_LABELS = ["0", "1", "2+"]

_HOUR_BUCKETS = [(0, 6), (6, 12), (12, 18), (18, 24)]
_HOUR_LABELS = ["00-06 UTC", "06-12 UTC", "12-18 UTC", "18-24 UTC"]


def _bucket_label(value: float, buckets: list, labels: list) -> str:
    for (lo, hi), label in zip(buckets, labels):
        if lo <= value < hi:
            return label
    return labels[-1]


@dataclass(frozen=True)
class MintRecord:
    token_address: str
    alert_time: str
    usable: bool
    features: dict


def _read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _fetch_clean_events(sb) -> list:
    """Same superset query v8_forward_readiness_report.py uses for
    ambiguity detection -- chain=solana, progress_data_ok=True, no
    progress-value filter."""
    rows, offset, batch = [], 0, 1000
    cols = ("event_id,token_address,alert_time,progress_at_signal,vsol_at_signal,"
            "venue_state_at_signal,path_file,channel_velocity_5m,realert_times,"
            "top10_holder_pct,creator_holds_pct")
    while True:
        resp = (sb.table("research_tokens")
                .select(cols)
                .eq("chain", "solana").eq("progress_data_ok", True)
                .range(offset, offset + batch - 1).execute())
        chunk = resp.data or []
        rows.extend(chunk)
        if len(chunk) < batch:
            break
        offset += batch
    return rows


def _resolve_usability(event: dict, ambiguous_mints: set, repo_root: Path) -> Optional[bool]:
    """True/False if resolvable, None if the mint is ambiguous (excluded
    entirely, same as v8_collection_yield.py's treatment)."""
    mint = event.get("token_address")
    if mint in ambiguous_mints:
        return None

    path_file = event.get("path_file")
    if not path_file:
        return False
    full_path = repo_root / path_file
    if not full_path.exists():
        gz_path = full_path.with_suffix(full_path.suffix + ".gz")
        full_path = gz_path if gz_path.exists() else None
    if full_path is None:
        return False

    rows, _warnings = load_path_file(full_path)
    if not rows:
        return False
    has_real_tick = any(r.get("source") == "live_pp" and r.get("backfilled") != "true" for r in rows)
    if not has_real_tick:
        return False

    typed_rows = []
    for r in rows:
        try:
            typed = dict(r)
            typed["ts_ms"] = int(r["ts_ms"])
            typed["price_usd"] = float(r["price_usd"])
            typed_rows.append(typed)
        except (KeyError, ValueError, TypeError):
            continue
    if not typed_rows:
        return False

    alignment = resolve_entry_alignment(event, typed_rows, _REFERENCE_CANDIDATE, ambiguous_mints)
    if isinstance(alignment, EntryAlignmentExclusion):
        return False
    integrity = assess_path_integrity(rows)
    return integrity.status == PathIntegrityStatus.VALID.value


def _realert_count_safe(realert_times: Optional[list], alert_time: str) -> int:
    """T0-safe reconstruction per v8_feature_registry.yaml's explicit
    instruction: count only realert_times entries strictly before this
    row's own alert_time -- never trust the raw stored realert_count
    field, which reflects future realerts as of query time."""
    if not realert_times or not alert_time:
        return 0
    return sum(1 for t in realert_times if isinstance(t, str) and t < alert_time)


def build_records(sb, repo_root: Path) -> list[MintRecord]:
    all_events = _fetch_clean_events(sb)
    ambiguous_mints = find_ambiguous_mints(all_events)

    era_start = trustworthy_collection_era_start(repo_root)
    if era_start is None:
        return []
    era_epoch = era_start.timestamp()

    admission_by_mint = load_admission_log_by_mint(repo_root)

    venue_qualified = [r for r in all_events if r.get("venue_state_at_signal") == "CURVE_ACTIVE"]

    by_mint = {}
    for r in venue_qualified:
        mint = r.get("token_address")
        if mint in ambiguous_mints:
            continue
        at = r.get("alert_time")
        if not at:
            continue
        try:
            dt = datetime.fromisoformat(at.replace("Z", "+00:00"))
        except ValueError:
            continue
        if dt.timestamp() < era_epoch:
            continue
        if mint not in by_mint:
            by_mint[mint] = r

    records = []
    for mint, event in by_mint.items():
        rows = admission_by_mint.get(mint, [])
        admit_row = next((rw for rw in rows if rw.get("path_admitted")), None)
        if admit_row is None:
            continue

        usable = _resolve_usability(event, ambiguous_mints, repo_root)
        if usable is None:
            continue

        alert_time = event.get("alert_time", "")
        try:
            hour = datetime.fromisoformat(alert_time.replace("Z", "+00:00")).hour
        except ValueError:
            hour = None

        features = {
            "progress_at_signal": event.get("progress_at_signal"),
            "vsol_at_signal": event.get("vsol_at_signal"),
            "channel_velocity_5m": event.get("channel_velocity_5m"),
            "realert_count_safe": _realert_count_safe(event.get("realert_times"), alert_time),
            "hour_of_day": hour,
            "top10_holder_pct": event.get("top10_holder_pct"),
            "creator_holds_pct": event.get("creator_holds_pct"),
        }
        records.append(MintRecord(token_address=mint, alert_time=alert_time, usable=usable, features=features))

    return records


# ── Per-feature bucketed usable-rate tables ─────────────────────────────────

def _report_bucketed(records: list[MintRecord], feature: str, buckets: list, labels: list) -> dict:
    cells: dict = {label: [] for label in labels}
    missing = 0
    for r in records:
        v = r.features.get(feature)
        if v is None:
            missing += 1
            continue
        cells[_bucket_label(v, buckets, labels)].append(r.usable)

    print(f"\n  {feature}  (missing={missing}/{len(records)})")
    print(f"    {'bucket':<14} {'n':>5}  {'usable_rate':>12}")
    result = {}
    for label in labels:
        vals = cells[label]
        n = len(vals)
        rate = round(100 * sum(vals) / n, 1) if n else None
        result[label] = {"n": n, "rate": rate}
        if n < MIN_CELL_N:
            print(f"    {label:<14} {n:>5}  INSUFFICIENT (need >={MIN_CELL_N})")
        else:
            print(f"    {label:<14} {n:>5}  {rate:>11.1f}%")
    return result


def _report_coverage_only(records: list[MintRecord], feature: str, known_zero_coverage_reason: str = "") -> None:
    n_present = sum(1 for r in records if r.features.get(feature) is not None)
    print(f"\n  {feature}  present={n_present}/{len(records)}"
          + (f"  ({known_zero_coverage_reason})" if known_zero_coverage_reason and n_present == 0 else ""))


# ── Simple logistic model + AUC ─────────────────────────────────────────────

def fit_logistic_and_auc(records: list[MintRecord]) -> Optional[dict]:
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        return None

    feature_names = ["progress_at_signal", "vsol_at_signal", "channel_velocity_5m",
                      "realert_count_safe", "hour_of_day"]
    X, y = [], []
    for r in records:
        vals = [r.features.get(f) for f in feature_names]
        if any(v is None for v in vals):
            continue
        X.append(vals)
        y.append(1 if r.usable else 0)

    n = len(y)
    if n < 2 * MIN_CELL_N or len(set(y)) < 2:
        return {"n": n, "auc": None, "reason": f"n={n} too small or single-class after dropping missing rows"}

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if min((y == 0).sum(), (y == 1).sum()) >= 2 else None,
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    if len(set(y_test)) < 2:
        return {"n": n, "auc": None, "reason": "test split ended up single-class -- n too small to hold out fairly"}
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    coefs = dict(zip(feature_names, model.coef_[0].tolist()))
    return {"n": n, "n_train": len(y_train), "n_test": len(y_test), "auc": round(auc, 4), "coefs": coefs}


# ── Best single pre-registerable condition ──────────────────────────────────

def find_best_condition(records: list[MintRecord]) -> Optional[dict]:
    baseline_n = len(records)
    baseline_rate = round(100 * sum(r.usable for r in records) / baseline_n, 2) if baseline_n else 0.0

    candidates_by_feature = {
        "progress_at_signal": (_PROGRESS_BUCKETS, _PROGRESS_LABELS),
        "vsol_at_signal": (_VSOL_BUCKETS, _VSOL_LABELS),
        "channel_velocity_5m": (_VELOCITY_BUCKETS, _VELOCITY_LABELS),
        "realert_count_safe": (_REALERT_BUCKETS, _REALERT_LABELS),
    }

    best = None
    for feature, (buckets, labels) in candidates_by_feature.items():
        violations = check_features_allowed([feature] if feature != "realert_count_safe" else ["realert_times"], "entry")
        if violations:
            continue  # not eligible to be proposed as a candidate condition at all

        for (lo, hi), label in zip(buckets, labels):
            subset = [r for r in records if r.features.get(feature) is not None and lo <= r.features[feature] < hi]
            n = len(subset)
            if n < MATERIAL_LIFT_MIN_N:
                continue
            rate = 100 * sum(r.usable for r in subset) / n
            lift = rate - baseline_rate
            if best is None or lift > best["lift_pp"]:
                best = {
                    "feature": feature, "bucket_label": label, "range": (lo, hi),
                    "n": n, "rate_pct": round(rate, 2), "baseline_rate_pct": baseline_rate,
                    "lift_pp": round(lift, 2),
                }

    return best


def print_report(records: list[MintRecord]) -> dict:
    print(f"\n{'=' * 72}")
    print(f"  V8 PATH PREDICTABILITY (YD1) — {datetime.now(timezone.utc).isoformat()}")
    print(f"{'=' * 72}")
    print(f"  funded-era admitted population n={len(records)}")
    if records:
        overall_rate = round(100 * sum(r.usable for r in records) / len(records), 2)
        print(f"  overall usable-path rate: {overall_rate}%")

    print(f"\n{'-' * 72}\n  Per-feature bucketed usable-path rate\n{'-' * 72}")
    _report_bucketed(records, "progress_at_signal", _PROGRESS_BUCKETS, _PROGRESS_LABELS)
    _report_bucketed(records, "vsol_at_signal", _VSOL_BUCKETS, _VSOL_LABELS)
    _report_bucketed(records, "channel_velocity_5m", _VELOCITY_BUCKETS, _VELOCITY_LABELS)
    _report_bucketed(records, "realert_count_safe", _REALERT_BUCKETS, _REALERT_LABELS)
    _report_bucketed(records, "hour_of_day", _HOUR_BUCKETS, _HOUR_LABELS)
    _report_coverage_only(records, "top10_holder_pct")
    _report_coverage_only(records, "creator_holds_pct",
                           known_zero_coverage_reason="known gap, v8_feature_registry.yaml: never populated in production")

    print(f"\n{'-' * 72}\n  Simple logistic model (progress/vsol/velocity/realert/hour)\n{'-' * 72}")
    logit = fit_logistic_and_auc(records)
    if logit is None:
        print("  numpy/scikit-learn not available -- skipped")
    elif logit.get("auc") is None:
        print(f"  n={logit['n']}: {logit['reason']}")
    else:
        print(f"  n={logit['n']} (train={logit['n_train']}, test={logit['n_test']})  AUC={logit['auc']}")
        print(f"  coefficients: {logit['coefs']}")

    print(f"\n{'-' * 72}\n  Best single pre-registerable condition\n{'-' * 72}")
    best = find_best_condition(records)
    if best is None:
        print(f"  no allowed_for_entry=true feature/bucket cleared n>={MATERIAL_LIFT_MIN_N}")
    else:
        print(f"  {best['feature']} in bucket {best['bucket_label']!r} (range {best['range']}):")
        print(f"    n={best['n']}  usable_rate={best['rate_pct']}%  "
              f"(baseline={best['baseline_rate_pct']}%, lift={best['lift_pp']:+.2f}pp)")
        material = best["lift_pp"] >= MATERIAL_LIFT_ABS_PP and best["n"] >= MATERIAL_LIFT_MIN_N
        print(f"    material (>= {MATERIAL_LIFT_ABS_PP}pp lift, n>={MATERIAL_LIFT_MIN_N})? {material}")
        if material:
            print(f"\n  DRAFT candidate-registry entry (NOT applied -- for user sign-off only):")
            print(f"    candidate_id: V8-P4 (next free slot)")
            print(f"    condition: {best['feature']} in {best['bucket_label']} "
                  f"(i.e. {best['range'][0]} <= {best['feature']} < {best['range'][1]})")
            print(f"    combined with existing venue_state_at_signal == CURVE_ACTIVE gate")
            print(f"    measured on this run: yield_if_applied={best['rate_pct']}%  "
                  f"admitted_n_if_applied={best['n']}")

    print(f"\n{'=' * 72}\n")
    return {"n": len(records), "best_condition": best, "logit": logit}


def main():
    from supabase import create_client
    from research.config import SUPABASE_URL, SUPABASE_KEY
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    root = Path(__file__).parent.parent
    records = build_records(sb, root)
    print_report(records)


if __name__ == "__main__":
    main()
