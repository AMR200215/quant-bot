"""
research/analysis/data_quality.py — Era-aware data quality layer.

Produces:
  - Per-day coverage table
  - Era boundary detection → research/analysis/era_boundaries.json
  - Integrity checks (A–E) with PASS/WARN/FAIL
  - Clean cohort metadata → research/analysis/clean_cohort.json
  - Collection health summary (--summary flag)

No strategy recommendations. No V8 rules. Data quality reporting only.

Run:
    python -m research.analysis.data_quality
    python -m research.analysis.data_quality --summary
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

_OUT_DIR = Path(__file__).parent

# ── Field groups ──────────────────────────────────────────────────────────────

BASE_IDENTITY = ["token_address", "symbol", "alert_time", "category"]

ENTRY_FEATURES = [
    "price_usd", "mcap_usd", "liquidity_usd", "age_minutes",
    "volume_5m", "buy_sell_ratio_5m", "dex_id",
    "pp_vsol", "channel_velocity_5m",
    # W2 additions — coverage expected low initially; era boundary auto-detected
    "progress_at_signal", "top10_holder_pct", "creator_holds_pct", "smart_money_hit",
]

OUTCOME_FEATURES = [
    "outcome_complete", "pct_change_peak", "peak_interval",
    "pct_change_t5m", "pct_change_t10m", "pct_change_t20m", "pct_change_t30m",
    "data_partial",
]

TICK_PEAK = ["price_peak_3m", "pct_change_peak_3m", "t_peak_3m_s"]

ALL_IMPORTANT = BASE_IDENTITY + ENTRY_FEATURES + OUTCOME_FEATURES + TICK_PEAK

# Fields where at least one must be non-null for a completed row to count as having
# real outcome data (outcome_complete=True but all of these NULL → excluded from cohorts)
OUTCOME_PEAK_FIELDS = [
    "pct_change_peak", "pct_change_peak_3m",
    "pct_change_t1m", "pct_change_t3m", "pct_change_t5m",
    "pct_change_t10m", "pct_change_t20m", "pct_change_t30m",
]


# ── Supabase loader ───────────────────────────────────────────────────────────

# Use select("*") so schema drift (missing columns) doesn't break the load.
# Missing columns will be absent from row dicts → reported as 0% coverage.
_FULL_COLS = "*"

_MINIMAL_COLS = "id,token_address,symbol,alert_time,category,snapshot_ok,price_usd,outcome_complete,created_at"


def _load_rows(limit: int = 20000, allow_partial: bool = False) -> tuple:
    """
    Returns (rows, load_meta).

    load_meta fields:
      partial_schema_load      bool  — True if full select failed and fell back
      missing_selected_columns list  — columns mentioned in the error (best-effort)

    Default (allow_partial=False):
      Full-select failure raises RuntimeError immediately.  The caller exits nonzero.
      This ensures schema drift is caught rather than silently degraded.

    --allow-partial mode:
      Falls back to minimal select.  load_meta marks partial_schema_load=True.
      build_clean_cohort() returns zero-count invalid cohorts when partial.
    """
    from research.config import SUPABASE_URL, SUPABASE_KEY
    from supabase import create_client
    sb   = create_client(SUPABASE_URL, SUPABASE_KEY)
    load_meta = {"partial_schema_load": False, "missing_selected_columns": []}

    def _paginate() -> list:
        rows   = []
        offset = 0
        batch  = 1000
        while len(rows) < limit:
            resp = sb.table("research_tokens") \
                .select(_FULL_COLS) \
                .order("alert_time", desc=False) \
                .range(offset, offset + batch - 1) \
                .execute()
            chunk = resp.data or []
            rows.extend(chunk)
            if len(chunk) < batch:
                break
            offset += batch
        return rows

    # select("*") always works — schema drift shows as missing keys in row dicts
    # (coverage will report 0% for unmigrated columns, not an error)
    return _paginate(), load_meta


# ── Coverage helpers ──────────────────────────────────────────────────────────

def _date(row) -> str:
    ts = row.get("alert_time") or row.get("created_at") or ""
    return ts[:10]


def _is_nonnull(row, col) -> bool:
    v = row.get(col)
    return v is not None and v != ""


def _is_zero(row, col) -> bool:
    v = row.get(col)
    return isinstance(v, (int, float)) and v == 0.0


# ── Per-day coverage table ────────────────────────────────────────────────────

def per_day_coverage(rows: list) -> list:
    by_day: dict = defaultdict(list)
    for r in rows:
        by_day[_date(r)].append(r)

    result = []
    for day in sorted(by_day):
        day_rows = by_day[day]
        n        = len(day_rows)
        entry    = {"date": day, "row_count": n}
        for col in ALL_IMPORTANT:
            nn  = sum(1 for r in day_rows if _is_nonnull(r, col))
            pct = round(nn / n * 100, 1) if n else 0
            entry[f"{col}_nn"] = nn
            entry[f"{col}_pct"] = pct
        entry["outcome_complete_count"] = sum(
            1 for r in day_rows if r.get("outcome_complete")
        )
        result.append(entry)
    return result


# ── Era boundary detection ────────────────────────────────────────────────────

def detect_era_boundaries(daily: list) -> dict:
    """
    For each important column, find the first date where coverage jumps from
    ~0% to >80%, indicating the column was added or a bug was fixed.
    """
    boundaries = {}
    check_cols = ENTRY_FEATURES + OUTCOME_FEATURES + TICK_PEAK

    for col in check_cols:
        pct_key = f"{col}_pct"
        low_seen  = False
        for i, day in enumerate(daily):
            pct = day.get(pct_key, 0)
            if pct < 10:
                low_seen = True
            if low_seen and pct >= 80:
                # Find previous coverage
                prev_pct = daily[i - 1].get(pct_key, 0) if i > 0 else 0
                boundaries[col] = {
                    "first_reliable_date": day["date"],
                    "coverage_after":      round(pct / 100, 3),
                    "coverage_before":     round(prev_pct / 100, 3),
                    "reason":              f"coverage jumped from {prev_pct:.0f}% to {pct:.0f}%",
                }
                break

    return boundaries


# ── Integrity checks ──────────────────────────────────────────────────────────

def _recent_rows(rows: list, days: int = 3) -> list:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    return [r for r in rows if _date(r) >= cutoff]


def integrity_checks(rows: list, daily: list) -> list:
    results = []
    recent  = _recent_rows(rows, days=3)

    # A. No price field equals 0.0 unless unknown
    zero_price = [r for r in rows if _is_zero(r, "price_usd")]
    level = "FAIL" if any(r for r in recent if _is_zero(r, "price_usd")) else "WARN" if zero_price else "PASS"
    results.append({
        "check": "A_zero_price_usd",
        "level": level,
        "count": len(zero_price),
        "note":  "price_usd=0.0 is invalid; should be NULL when unknown",
        "sample": [r.get("token_address", "")[:12] for r in zero_price[:3]],
    })

    # B. outcome_complete=True but all peak/outcome fields NULL
    def _outcome_empty(r):
        if not r.get("outcome_complete"):
            return False
        outcome_cols = ["pct_change_peak", "pct_change_t5m", "pct_change_t10m",
                        "pct_change_t20m", "pct_change_t30m"]
        return all(r.get(c) is None for c in outcome_cols)

    empty_complete = [r for r in rows if _outcome_empty(r)]
    level = "FAIL" if any(r for r in recent if _outcome_empty(r)) else \
            "WARN" if empty_complete else "PASS"
    results.append({
        "check": "B_complete_but_no_outcomes",
        "level": level,
        "count": len(empty_complete),
        "note":  "outcome_complete=True but all pct_change columns NULL",
        "sample": [r.get("token_address", "")[:12] for r in empty_complete[:3]],
    })

    # C. Duplicate token identity (same token_address + same date)
    seen: dict = {}
    dupes = []
    for r in rows:
        key = (r.get("token_address", ""), _date(r))
        if key in seen:
            dupes.append(r)
        else:
            seen[key] = True
    level = "FAIL" if any(r for r in recent if
                          (r.get("token_address"), _date(r)) in
                          {(d.get("token_address"), _date(d)) for d in dupes}) \
            else "WARN" if dupes else "PASS"
    results.append({
        "check": "C_duplicate_token_day",
        "level": level,
        "count": len(dupes),
        "note":  "Multiple rows for same (token_address, alert_date) — unique index should prevent this",
        "sample": [r.get("token_address", "")[:12] for r in dupes[:3]],
    })

    # D. Category consistency — graduated tokens should have graduated dex_id or pp_vsol
    grad_cats = {"social_alert_grad"}
    grad_dex  = {"pumpswap", "raydium", "orca"}
    def _grad_inconsistent(r):
        if r.get("category") not in grad_cats:
            return False
        dex   = (r.get("dex_id") or "").lower()
        vsol  = r.get("pp_vsol") or 0.0
        return dex not in grad_dex and vsol < 79
    grad_bad = [r for r in rows if _grad_inconsistent(r)]
    level = "WARN" if grad_bad else "PASS"
    results.append({
        "check": "D_category_consistency",
        "level": level,
        "count": len(grad_bad),
        "note":  "social_alert_grad rows with no graduated dex_id and pp_vsol<79",
        "sample": [r.get("token_address", "")[:12] for r in grad_bad[:3]],
    })

    # E. Poller lateness — recent era only
    # Check what % of recent outcome_complete rows have price_usd (baseline) non-null
    recent_complete = [r for r in recent if r.get("outcome_complete")]
    if recent_complete:
        with_price = [r for r in recent_complete if r.get("price_usd")]
        pct = len(with_price) / len(recent_complete) * 100
        level = "FAIL" if pct < 20 else "WARN" if pct < 60 else "PASS"
        results.append({
            "check": "E_recent_baseline_price_coverage",
            "level": level,
            "count": len(recent_complete),
            "note":  f"{pct:.1f}% of recent outcome_complete rows have entry price_usd",
            "sample": [],
        })
    else:
        results.append({
            "check": "E_recent_baseline_price_coverage",
            "level": "WARN",
            "count": 0,
            "note":  "No recent completed rows to check",
            "sample": [],
        })

    return results


# ── Clean cohort metadata ─────────────────────────────────────────────────────

def build_clean_cohort(rows: list, era_boundaries: dict,
                       partial_schema: bool = False) -> dict:
    """
    Build clean cohort metadata.

    Invariants:
    - outcomes_only and entry_features are built ONLY from outcome_complete=True rows.
    - A completed row with all OUTCOME_PEAK_FIELDS null is excluded (no_outcome_data).
    - tick_peaks requires pct_change_peak_3m AND t_peak_3m_s non-null.
    - When partial_schema=True (--allow-partial fallback), all cohorts return
      row_count=0 and trusted=False so callers cannot use them.
    """
    if partial_schema:
        def _invalid(label):
            return {
                "label": label, "trusted": False,
                "start_date": None, "end_date": None,
                "row_count": 0, "excluded_rows_count": len(rows),
                "required_fields": [], "coverage_by_field": {},
                "exclusion_breakdown": {"partial_schema_load": len(rows)},
            }
        return {
            "generated_at":       datetime.now(timezone.utc).isoformat(),
            "partial_schema_load": True,
            "outcomes_only":       _invalid("outcomes_only"),
            "entry_features":      _invalid("entry_features"),
            "tick_peaks":          _invalid("tick_peaks"),
        }

    # Era-aware start dates
    outcome_dates = [
        era_boundaries.get(f, {}).get("first_reliable_date", "2026-01-01")
        for f in ["pct_change_peak", "price_usd"]
    ]
    entry_dates = [
        era_boundaries.get(f, {}).get("first_reliable_date", "2026-01-01")
        for f in ["mcap_usd", "liquidity_usd", "buy_sell_ratio_5m"]
    ]
    tick_dates = [
        era_boundaries.get(f, {}).get("first_reliable_date", "2026-01-01")
        for f in ["pct_change_peak_3m"]
    ]

    outcomes_start = max(outcome_dates)  if outcome_dates  else "2026-01-01"
    entry_start    = max(entry_dates + outcome_dates) if entry_dates else outcomes_start
    tick_start     = max(tick_dates)     if tick_dates     else "2026-01-01"

    # outcome_complete=True rows only — non-completed rows never enter outcome cohorts
    completed          = [r for r in rows if r.get("outcome_complete") is True]
    not_complete_count = len(rows) - len(completed)

    def _has_outcome_data(r) -> bool:
        """At least one pct/peak field must be non-null."""
        return any(_is_nonnull(r, f) for f in OUTCOME_PEAK_FIELDS)

    def _cohort(pool, required_cols, start_date, label, need_outcome_data=False):
        """
        pool          — pre-filtered input (e.g. completed rows only)
        required_cols — all must be non-null
        need_outcome_data — additionally require ≥1 OUTCOME_PEAK_FIELD non-null
        """
        breakdown: dict = {}
        in_era    = [r for r in pool if _date(r) >= start_date]

        # Step 1: outcome-data gate (for outcomes_only + entry_features)
        if need_outcome_data:
            has_data, no_data = [], []
            for r in in_era:
                (has_data if _has_outcome_data(r) else no_data).append(r)
            if no_data:
                breakdown["no_outcome_data"] = len(no_data)
            in_era = has_data

        # Step 2: required-field gate
        cohort_rows, missing_fields = [], []
        for r in in_era:
            if all(_is_nonnull(r, c) for c in required_cols):
                cohort_rows.append(r)
            else:
                missing_fields.append(r)
        if missing_fields:
            breakdown["required_fields_null"] = len(missing_fields)

        excluded_total = len(pool) - len(cohort_rows)

        if not cohort_rows:
            return {
                "label": label, "trusted": True,
                "start_date": start_date, "end_date": None,
                "row_count": 0, "excluded_rows_count": excluded_total,
                "required_fields": required_cols, "coverage_by_field": {},
                "exclusion_breakdown": breakdown,
            }

        dates = sorted(_date(r) for r in cohort_rows)
        cov   = {c: round(sum(1 for r in cohort_rows if _is_nonnull(r, c))
                          / len(cohort_rows) * 100, 1)
                 for c in required_cols}
        return {
            "label":               label,
            "trusted":             True,
            "start_date":          dates[0],
            "end_date":            dates[-1],
            "row_count":           len(cohort_rows),
            "excluded_rows_count": excluded_total,
            "required_fields":     required_cols,
            "coverage_by_field":   cov,
            "exclusion_breakdown": breakdown,
        }

    # outcomes_only: completed rows with ≥1 real outcome field
    outcomes_only = _cohort(
        completed,
        ["token_address", "alert_time", "category"],
        outcomes_start, "outcomes_only",
        need_outcome_data=True,
    )

    # entry_features: completed rows with full entry snapshot + ≥1 outcome field
    entry_features = _cohort(
        completed,
        ["token_address", "alert_time", "category",
         "price_usd", "mcap_usd", "liquidity_usd", "volume_5m",
         "buy_sell_ratio_5m", "age_minutes"],
        entry_start, "entry_features",
        need_outcome_data=True,
    )

    # tick_peaks: any row (outcome_complete not required) with both tick-peak fields
    tick_peaks = _cohort(
        rows,
        ["token_address", "alert_time", "pct_change_peak_3m", "t_peak_3m_s"],
        tick_start, "tick_peaks",
        need_outcome_data=False,
    )

    return {
        "generated_at":         datetime.now(timezone.utc).isoformat(),
        "partial_schema_load":  False,
        "total_rows":           len(rows),
        "outcome_complete_total": len(completed),
        "not_complete_excluded":  not_complete_count,
        "outcomes_only":        outcomes_only,
        "entry_features":       entry_features,
        "tick_peaks":           tick_peaks,
    }


# ── Health summary ────────────────────────────────────────────────────────────

def health_summary(rows: list, checks: list, cohort: dict):
    from research.spool.writer import dropped_field_count, failed_insert_count

    now        = datetime.now(timezone.utc)
    since_24h  = (now - timedelta(hours=24)).strftime("%Y-%m-%d")
    since_7d   = (now - timedelta(days=7)).strftime("%Y-%m-%d")
    recent_24h = [r for r in rows if _date(r) >= since_24h]
    recent_7d  = [r for r in rows if _date(r) >= since_7d]

    dropped_24h = dropped_field_count(since_24h)
    failed_24h  = failed_insert_count(since_24h)

    print(f"\n{'='*60}")
    print("COLLECTION HEALTH SUMMARY")
    print(f"{'='*60}")
    print(f"\n  Rows total             : {len(rows):,}")
    print(f"  Rows last 24h          : {len(recent_24h):,}")
    print(f"  Rows last 7d           : {len(recent_7d):,}")
    print(f"  Dropped fields (24h)   : {dropped_24h}")
    print(f"  Failed inserts (24h)   : {failed_24h}")

    print(f"\n  Clean cohorts:")
    for key in ["outcomes_only", "entry_features", "tick_peaks"]:
        c = cohort.get(key, {})
        n = c.get("row_count", 0)
        s = c.get("start_date", "n/a")
        e = c.get("end_date",   "n/a")
        print(f"    {key:<20} {n:>5} rows  {s} → {e}")

    # Fields with <80% coverage in recent era (last 7d)
    if recent_7d:
        low_cov = []
        for col in ENTRY_FEATURES + OUTCOME_FEATURES:
            nn  = sum(1 for r in recent_7d if _is_nonnull(r, col))
            pct = nn / len(recent_7d) * 100
            if pct < 80:
                low_cov.append((col, pct))
        if low_cov:
            print(f"\n  Fields <80% coverage (last 7d):")
            for col, pct in sorted(low_cov, key=lambda x: x[1]):
                print(f"    {col:<30} {pct:.1f}%")

    # Status
    fails  = [c for c in checks if c["level"] == "FAIL"]
    warns  = [c for c in checks if c["level"] == "WARN"]
    status = "FAIL" if fails or failed_24h > 0 else "WARN" if warns or dropped_24h > 0 else "PASS"
    print(f"\n  Overall status: {status}")
    if fails:
        for c in fails:
            print(f"    FAIL  {c['check']}: {c['note']}")
    if warns:
        for c in warns:
            print(f"    WARN  {c['check']}: {c['note']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true",
                        help="Print health summary only (fast)")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Fall back to minimal columns if full select fails. "
                             "Marks partial_schema_load=true and disables clean cohorts.")
    args = parser.parse_args()

    print("Loading rows from Supabase...")
    try:
        rows, load_meta = _load_rows(allow_partial=args.allow_partial)
    except RuntimeError as e:
        print(f"\nFAIL — {e}")
        sys.exit(1)
    except Exception as e:
        print(f"FAIL — Cannot load data from Supabase: {e}")
        sys.exit(1)

    partial = load_meta["partial_schema_load"]
    if partial:
        print(f"\nWARN: partial_schema_load=true")
        print(f"      missing_selected_columns={load_meta['missing_selected_columns']}")
        print(f"      Clean cohorts are DISABLED — apply schema migration first.\n")

    print(f"Loaded {len(rows):,} rows")

    daily   = per_day_coverage(rows)
    eras    = detect_era_boundaries(daily)
    checks  = integrity_checks(rows, daily)
    cohort  = build_clean_cohort(rows, eras, partial_schema=partial)

    # Save outputs
    (_OUT_DIR / "era_boundaries.json").write_text(json.dumps(eras, indent=2))
    (_OUT_DIR / "clean_cohort.json").write_text(json.dumps(cohort, indent=2))

    if args.summary:
        health_summary(rows, checks, cohort)
        return

    # ── Per-day coverage ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PER-DAY COVERAGE")
    print(f"{'='*60}")
    key_cols = ["price_usd", "mcap_usd", "buy_sell_ratio_5m", "outcome_complete",
                "pct_change_peak", "pct_change_peak_3m"]
    header = f"{'Date':<12} {'Rows':>5}"
    for c in key_cols:
        header += f"  {c[:14]:>14}"
    print(header)
    for d in daily:
        row_str = f"{d['date']:<12} {d['row_count']:>5}"
        for c in key_cols:
            pct = d.get(f"{c}_pct", 0)
            row_str += f"  {pct:>13.0f}%"
        print(row_str)

    # ── Era boundaries ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ERA BOUNDARIES")
    print(f"{'='*60}")
    if eras:
        for col, info in sorted(eras.items()):
            print(f"  {col:<30} first_reliable={info['first_reliable_date']}"
                  f"  cov_after={info['coverage_after']*100:.0f}%"
                  f"  ({info['reason']})")
    else:
        print("  No era boundaries detected (all columns stable or always low)")

    # ── Integrity checks ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("INTEGRITY CHECKS")
    print(f"{'='*60}")
    all_pass = True
    for c in checks:
        sym = {"PASS": "✓", "WARN": "~", "FAIL": "✗"}.get(c["level"], "?")
        print(f"  [{c['level']:4}] {sym} {c['check']} — {c['note']} (n={c['count']})")
        if c["level"] == "FAIL":
            all_pass = False
        if c.get("sample"):
            print(f"          sample: {c['sample']}")

    # ── Clean cohort summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("CLEAN COHORT METADATA")
    print(f"{'='*60}")
    if cohort.get("partial_schema_load"):
        print("  DISABLED — partial_schema_load=true (apply schema migration first)")
    else:
        print(f"  outcome_complete total : {cohort.get('outcome_complete_total', 0):,}")
        print(f"  not_complete excluded  : {cohort.get('not_complete_excluded', 0):,}")
    for key in ["outcomes_only", "entry_features", "tick_peaks"]:
        c = cohort.get(key, {})
        trusted = c.get("trusted", True)
        print(f"\n  {key}:{' [UNTRUSTED — partial schema]' if not trusted else ''}")
        print(f"    rows       : {c.get('row_count', 0):,}")
        print(f"    date range : {c.get('start_date')} → {c.get('end_date')}")
        print(f"    excluded   : {c.get('excluded_rows_count', 0):,}")
        bd = c.get("exclusion_breakdown", {})
        if bd:
            for reason, n in sorted(bd.items()):
                print(f"      {reason:<30} {n:,} rows")
        cov = c.get("coverage_by_field", {})
        if cov:
            for f, pct in cov.items():
                print(f"    {f:<30} {pct:.1f}%")

    # ── Historical warnings ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("HISTORICAL WARNINGS")
    print(f"{'='*60}")
    for col, info in sorted(eras.items()):
        date = info["first_reliable_date"]
        before_pct = info["coverage_before"] * 100
        print(f"  WARN  {col}: coverage <{before_pct:.0f}% before {date} "
              f"(old-era data unreliable for this field)")

    # ── Health summary ────────────────────────────────────────────────────────
    health_summary(rows, checks, cohort)

    print(f"\nOutputs written:")
    print(f"  {_OUT_DIR}/era_boundaries.json")
    print(f"  {_OUT_DIR}/clean_cohort.json")

    # Exit nonzero if current/recent data has failures
    recent = _recent_rows(rows, days=3)
    hard_fails = [c for c in checks if c["level"] == "FAIL"]
    if hard_fails:
        print(f"\nEXIT 1 — {len(hard_fails)} integrity check(s) FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
