"""
RC3 — RF1 before/after NULL-rate coverage check.

Queries research_tokens for outcome_complete rows and compares the price NULL
rate at T1m / T3m / T10m between:
  - preRF1 rows (price_source_t1m IS NULL AND price_status_t1m IS NULL)
  - postRF1 rows (price_source_t1m IS NOT NULL OR price_status_t1m IS NOT NULL)

Split further by category to isolate the BC-token measurement bias that RF1 was
built to fix.

If postRF1 NULL rates haven't collapsed vs preRF1 for BC tokens, that is a bug.

Usage:
    python -m research.scripts.rf1_coverage_check
    python -m research.scripts.rf1_coverage_check --all   # include non-complete rows
"""

import argparse
import sys
from datetime import datetime, timezone

from research.config import SUPABASE_URL, SUPABASE_KEY


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fetch(sb, complete_only: bool) -> list[dict]:
    rows, offset, batch = [], 0, 1000
    while True:
        q = (sb.table("research_tokens")
               .select("id,category,alert_time,"
                       "price_t1m,price_t3m,price_t10m,"
                       "price_source_t1m,price_status_t1m,"
                       "price_source_t3m,price_status_t3m,"
                       "price_source_t10m,price_status_t10m"))
        if complete_only:
            q = q.eq("outcome_complete", True)
        chunk = q.range(offset, offset + batch - 1).execute().data or []
        rows.extend(chunk)
        if len(chunk) < batch:
            break
        offset += batch
    return rows


def _is_clean(row: dict) -> bool:
    """True if RF1 provenance was written for this row."""
    for col in ("price_source_t1m", "price_status_t1m",
                "price_source_t3m", "price_status_t3m",
                "price_source_t10m", "price_status_t10m"):
        if row.get(col) is not None:
            return True
    return False


def _null_pct(rows: list, col: str) -> str:
    if not rows:
        return "   n/a"
    n = sum(1 for r in rows if r.get(col) is None)
    return f"{n/len(rows)*100:5.1f}%"


def _print_table(era_label: str, subset: list):
    cats = [None, "social_alert_bc", "social_alert_grad"]
    print(f"\n  Era: {era_label}  (n={len(subset)})")
    print(f"  {'Category':<22}  {'n':>5}  {'null_t1m':>9}  {'null_t3m':>9}  {'null_t10m':>10}")
    print(f"  {'-'*22}  {'-'*5}  {'-'*9}  {'-'*9}  {'-'*10}")
    for cat in cats:
        if cat is None:
            label = "ALL"
            cat_rows = subset
        else:
            label = cat
            cat_rows = [r for r in subset if r.get("category") == cat]
        if not cat_rows:
            continue
        print(f"  {label:<22}  {len(cat_rows):>5}  "
              f"{_null_pct(cat_rows, 'price_t1m'):>9}  "
              f"{_null_pct(cat_rows, 'price_t3m'):>9}  "
              f"{_null_pct(cat_rows, 'price_t10m'):>10}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RC3 RF1 before/after NULL rate check")
    parser.add_argument("--all", action="store_true",
                        help="Include non-outcome_complete rows")
    args = parser.parse_args()

    try:
        from supabase import create_client
    except ImportError:
        print("supabase-py not installed"); sys.exit(1)

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("SUPABASE_URL / SUPABASE_KEY not set"); sys.exit(1)

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    complete_only = not args.all
    print(f"Querying research_tokens "
          f"({'outcome_complete only' if complete_only else 'ALL rows'})…")
    rows = _fetch(sb, complete_only)
    print(f"  {len(rows)} rows fetched\n")

    clean   = [r for r in rows if _is_clean(r)]
    pre_rf1 = [r for r in rows if not _is_clean(r)]

    sep = "=" * 70
    print(sep)
    print("RC3 — RF1 BEFORE/AFTER NULL-RATE COVERAGE")
    print(f"  Run at: {datetime.now(timezone.utc).isoformat()}")
    print(sep)

    _print_table("dex_conditioned_preRF1  [expected: high NULL rate at T1m/T3m for BC]", pre_rf1)
    _print_table("clean (postRF1)         [expected: low NULL rate — curve-first polling]", clean)

    print()
    print(sep)
    if clean:
        bc_clean = [r for r in clean if r.get("category") == "social_alert_bc"]
        bc_pre   = [r for r in pre_rf1 if r.get("category") == "social_alert_bc"]
        if bc_clean and bc_pre:
            t1m_pre   = sum(1 for r in bc_pre   if r.get("price_t1m") is None) / len(bc_pre)
            t1m_clean = sum(1 for r in bc_clean if r.get("price_t1m") is None) / len(bc_clean)
            delta = t1m_pre - t1m_clean
            verdict = "OK — RF1 reduced NULL rate" if delta > 0.10 else \
                      "WARN — NULL rate unchanged; check curve_oracle logs" if delta <= 0 else \
                      f"PARTIAL — {delta*100:.0f}pp improvement"
            print(f"  BC T1m NULL-rate drop: {t1m_pre*100:.1f}% → {t1m_clean*100:.1f}%  "
                  f"(Δ={delta*100:+.1f}pp)  [{verdict}]")
    else:
        print("  No clean-era rows yet — RF1 has not polled any completed tokens.")
        print("  Re-run once tokens complete their 30min outcome window after Jul 28.")
    print(sep)


if __name__ == "__main__":
    main()
