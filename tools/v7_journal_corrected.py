#!/usr/bin/env python3
"""
tools/v7_journal_corrected.py — one-time history repair + accounting epoch backfill.

Reads logs/memecoin_journal.csv (original — never mutated).
Writes logs/v7_journal_corrected.csv with:
  1. Corrected pnl_usd for rows where partial TPs were hit (inferred from pnl_pct vs TP thresholds)
  2. realized_partial_usd and remaining_fraction columns populated
  3. accounting_epoch backfilled by exit_time vs commit deploy times
  4. tp_levels_hit populated where inferable

Correction rule (conservative):
  - If final pnl_pct >= +100%: TP1 was definitely hit (sold 30% at +100%)
  - If final pnl_pct >= +300%: TP2 was also hit (sold 30% of remaining at +300%)
  - If final pnl_pct < +100%: no correction (TP may have been hit + price crashed back,
    but we can't prove it from journal alone — corrected file notes this as uncorrectable)

Run once after deploying the portfolio.py accounting fix, then v7_report.py reads the output.

Usage:
  python tools/v7_journal_corrected.py [--dry-run]
"""

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
JOURNAL_FILE    = ROOT / "logs" / "memecoin_journal.csv"
CORRECTED_FILE  = ROOT / "logs" / "v7_journal_corrected.csv"

# TP ladder — must match memecoin/config.py TP_LEVELS
TP_LEVELS = [
    (1.00, 0.30),   # at +100%, sell 30% of remaining
    (3.00, 0.30),   # at +300%, sell 30% of remaining
]

# Epoch boundaries (UTC unix timestamps) derived from git commit times:
#   81de8da  2026-06-11 01:20:37 +0200  →  2026-06-10 23:20:37 UTC
#   9a2a332  2026-06-11 02:07:27 +0200  →  2026-06-11 00:07:27 UTC
EPOCH_BOUNDARIES = [
    (datetime(2026, 6, 10, 23, 20, 37, tzinfo=timezone.utc).timestamp(), "e2_pp_exits"),
    (datetime(2026, 6, 11,  0,  7, 27, tzinfo=timezone.utc).timestamp(), "e3_pp_entries_anchored_stops"),
]


def _parse_exit_ts(exit_time_str: str) -> float:
    """Parse 'YYYY-MM-DD HH:MM:SS' (UTC) → unix timestamp. Returns 0 on failure."""
    if not exit_time_str:
        return 0.0
    try:
        dt = datetime.strptime(exit_time_str.strip(), "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return 0.0


def _assign_epoch(exit_ts: float) -> str:
    epoch = "e1_baseline"
    for boundary_ts, epoch_name in EPOCH_BOUNDARIES:
        if exit_ts >= boundary_ts:
            epoch = epoch_name
    return epoch


def _infer_tp_correction(pnl_pct_raw: float, size_usd: float):
    """
    Given final exit pnl_pct (as a ratio, e.g. 2.0 = +200%) and size_usd,
    return (corrected_pnl_usd, realized_partial_usd, remaining_fraction, tp_levels_hit_str).

    Conservative: only corrects rows where final pnl_pct is above a TP threshold,
    meaning the TP was definitely hit (not just possibly hit then price crashed).
    """
    remaining = 1.0
    realized = 0.0
    tp_hit = []

    for tp_pct, tp_fraction in TP_LEVELS:
        if pnl_pct_raw >= tp_pct:
            sell_frac = tp_fraction * remaining
            realized += sell_frac * size_usd * tp_pct
            remaining -= sell_frac
            tp_hit.append(f"tp_{int(tp_pct * 100)}")

    # Remaining fraction exits at the final pnl_pct
    corrected_pnl = realized + remaining * size_usd * pnl_pct_raw
    return corrected_pnl, realized, remaining, ",".join(tp_hit)


def run(dry_run: bool = False):
    if not JOURNAL_FILE.exists():
        print(f"[ERROR] Journal not found: {JOURNAL_FILE}")
        sys.exit(1)

    with open(JOURNAL_FILE, newline="") as f:
        original_rows = list(csv.DictReader(f))

    if not original_rows:
        print("[WARN] Journal is empty — nothing to correct.")
        return

    existing_fields = list(original_rows[0].keys())

    # Ensure output has all original fields plus the new accounting columns
    new_fields = ["tp_levels_hit", "realized_partial_usd", "remaining_fraction", "accounting_epoch"]
    out_fields = existing_fields.copy()
    for f in new_fields:
        if f not in out_fields:
            out_fields.append(f)

    corrected_rows = []
    old_v7_total   = 0.0
    new_v7_total   = 0.0
    n_corrected    = 0
    n_uncorrectable = 0

    for row in original_rows:
        r = dict(row)

        # ── Accounting epoch backfill ────────────────────────────────────────
        if not r.get("accounting_epoch"):
            exit_ts = _parse_exit_ts(r.get("exit_time", ""))
            r["accounting_epoch"] = _assign_epoch(exit_ts)

        # ── TP correction ────────────────────────────────────────────────────
        try:
            pnl_pct_raw = float(r.get("pnl_pct") or 0) / 100.0   # journal stores as %, e.g. 200.0
            size_usd    = float(r.get("size_usd") or 0)
            old_pnl     = float(r.get("pnl_usd") or 0)
        except ValueError:
            corrected_rows.append(r)
            continue

        # Only attempt correction if the row doesn't already have the new fields
        already_corrected = bool(r.get("realized_partial_usd"))

        if already_corrected:
            # Row was written by the new accounting code — no correction needed
            if not r.get("tp_levels_hit"):
                r["tp_levels_hit"] = ""
            corrected_rows.append(r)
            continue

        if pnl_pct_raw >= TP_LEVELS[0][0]:
            # At least one TP was definitely hit — apply correction
            new_pnl, realized, remaining, tp_str = _infer_tp_correction(pnl_pct_raw, size_usd)
            r["pnl_usd"]            = round(new_pnl, 4)
            r["realized_partial_usd"] = round(realized, 4)
            r["remaining_fraction"] = round(remaining, 4)
            r["tp_levels_hit"]      = tp_str
            n_corrected += 1
            delta = new_pnl - old_pnl
            if r.get("config_tag", "").startswith("v7"):
                old_v7_total += old_pnl
                new_v7_total += new_pnl
        else:
            # Cannot infer TP state from final pnl alone — leave pnl_usd as-is.
            # Could be: no TP hit, or TP hit then crashed (undercorrected — real P&L better).
            r["realized_partial_usd"] = 0.0
            r["remaining_fraction"]   = 1.0
            r["tp_levels_hit"]        = ""
            if pnl_pct_raw < 0 and r.get("config_tag", "").startswith("v7"):
                # These are potentially undercorrected (TP could have been hit before crash)
                # but we can't know without the original tp_levels_hit data.
                n_uncorrectable += 1
            if r.get("config_tag", "").startswith("v7"):
                old_v7_total += old_pnl
                new_v7_total += old_pnl  # unchanged

        corrected_rows.append(r)

    # ── Write output ─────────────────────────────────────────────────────────
    if not dry_run:
        with open(CORRECTED_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(corrected_rows)
        print(f"Written: {CORRECTED_FILE}  ({len(corrected_rows)} rows)")
    else:
        print(f"[dry-run] Would write {len(corrected_rows)} rows to {CORRECTED_FILE}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  CORRECTION SUMMARY")
    print("=" * 60)
    print(f"  Total rows processed : {len(original_rows)}")
    print(f"  TP corrections applied: {n_corrected}  (pnl_pct >= +100%, exact TP price known)")
    print(f"  Uncorrectable losses  : {n_uncorrectable}  (pnl_pct < 0 — TP state unknown)")
    print()
    v7_rows = [r for r in corrected_rows if r.get("config_tag", "").startswith("v7")]
    print(f"  V7 rows (config_tag v7*): {len(v7_rows)}")
    print(f"  V7 total PnL (old) : ${old_v7_total:.4f}")
    print(f"  V7 total PnL (new) : ${new_v7_total:.4f}")
    print(f"  V7 delta           : ${new_v7_total - old_v7_total:+.4f}")
    print()
    if n_uncorrectable:
        print(f"  NOTE: {n_uncorrectable} losing v7 rows may be undercorrected.")
        print("  If TP was hit before the crash, real PnL is better than recorded.")
        print("  These rows will be correctly accounted from e3 onwards.")
    print("=" * 60)


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    run(dry_run=dry_run)
