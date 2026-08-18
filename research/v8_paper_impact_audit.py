"""
research/v8_paper_impact_audit.py — V8 DATA RECOVERY batch, item 3:
classifies existing V8 paper journal rows for impact from the
now-fixed live_pp reserve-pricing bug (memecoin/pumpfun_reserve_pricing.py).

The bug (`vsol / (vtok / 1e6)`) applied a CONSTANT multiplicative
factor (always exactly 1e6x too high, independent of vsol/vtok's actual
values) to every price computed from bonding-curve reserves. A constant
multiplicative factor cancels EXACTLY in any ratio -- so if a trade's
entry AND exit prices both came from the same (buggy) reserve-based
formula, its realized pnl_pct is mathematically unaffected even though
its absolute prices are wrong. This module verifies that claim per row
by direct arithmetic, rather than assuming it.

Classifications (never inferred without checking the actual numbers):
  UNAFFECTED                          -- absolute prices are already
      plausible (not corrupted) -- nothing to flag.
  PCT_PNL_PRESERVED_ABSOLUTE_PRICE_BAD -- absolute prices are implausibly
      high (corrupted) BUT entry_price/exit_price's implied ratio
      matches the row's own recorded pnl_pct within tolerance -- the
      percentage return and dollar PnL are trustworthy; only the
      absolute price fields (entry_price/exit_price/peak_price) are not.
  PCT_PNL_CORRUPTED                   -- absolute prices are corrupted
      AND the entry/exit ratio does NOT match the recorded pnl_pct
      (e.g. entry and exit used different formula branches, or a
      venue-state transition happened mid-trade) -- neither the
      absolute prices nor the pnl_pct can be trusted.
  UNKNOWN                             -- not enough fields present to
      check either condition.

No row is ever deleted or silently rewritten. This module only reads
the journal and writes a SEPARATE, new provenance artifact.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Real pump.fun per-token prices, pre-graduation, are always tiny
# fractions of a cent (see memecoin/pumpfun_reserve_pricing.py's
# theoretical-ceiling reasoning) -- ANY entry/exit price at or above
# $0.01/token on a sub-graduation token is implausible on its face,
# independent of the exact curve-ceiling formula (which needs vsol/vtok
# this journal schema doesn't carry). A conservative, clearly-labeled
# sanity floor, not a precise formula, since the journal lacks the raw
# reserve fields needed for the precise one.
IMPLAUSIBLE_ABSOLUTE_PRICE_USD_FLOOR = 0.01

PNL_RATIO_TOLERANCE_PCT = 0.5  # allow half a percentage point of rounding slack


class ImpactClassification:
    UNAFFECTED = "UNAFFECTED"
    PCT_PNL_PRESERVED_ABSOLUTE_PRICE_BAD = "PCT_PNL_PRESERVED_ABSOLUTE_PRICE_BAD"
    PCT_PNL_CORRUPTED = "PCT_PNL_CORRUPTED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RowImpact:
    row_id: str
    token_address: str
    entry_price: Optional[float]
    exit_price: Optional[float]
    recorded_pnl_pct: Optional[float]
    implied_pnl_pct_from_prices: Optional[float]
    absolute_price_plausible: Optional[bool]
    classification: str
    note: str


def _f(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def classify_journal_row(row: dict) -> RowImpact:
    row_id = row.get("id", "")
    token_address = row.get("token_address", "")
    entry_price = _f(row.get("entry_price"))
    exit_price = _f(row.get("exit_price"))
    recorded_pnl_pct = _f(row.get("pnl_pct"))

    if entry_price is None or exit_price is None or recorded_pnl_pct is None or entry_price <= 0:
        return RowImpact(row_id, token_address, entry_price, exit_price, recorded_pnl_pct,
                          None, None, ImpactClassification.UNKNOWN,
                          "missing entry_price/exit_price/pnl_pct -- cannot check")

    implied_pct = (exit_price / entry_price - 1) * 100
    ratio_matches = abs(implied_pct - recorded_pnl_pct) <= PNL_RATIO_TOLERANCE_PCT

    plausible = entry_price < IMPLAUSIBLE_ABSOLUTE_PRICE_USD_FLOOR and exit_price < IMPLAUSIBLE_ABSOLUTE_PRICE_USD_FLOOR

    if plausible:
        classification = ImpactClassification.UNAFFECTED
        note = "absolute prices already plausible -- not corrupted"
    elif ratio_matches:
        classification = ImpactClassification.PCT_PNL_PRESERVED_ABSOLUTE_PRICE_BAD
        note = (f"absolute prices implausible (entry=${entry_price:.4f}) but implied "
                f"pnl_pct ({implied_pct:.2f}%) matches recorded pnl_pct ({recorded_pnl_pct:.2f}%) "
                "within tolerance -- entry and exit used the same constant-scaled formula, "
                "pnl_pct/pnl_usd are trustworthy, absolute price fields are not")
    else:
        classification = ImpactClassification.PCT_PNL_CORRUPTED
        note = (f"absolute prices implausible AND implied pnl_pct ({implied_pct:.2f}%) does NOT "
                f"match recorded pnl_pct ({recorded_pnl_pct:.2f}%) -- entry/exit likely used "
                "different formula branches or crossed a venue-state transition; neither "
                "absolute prices nor pnl_pct can be trusted for this row")

    return RowImpact(row_id, token_address, entry_price, exit_price, recorded_pnl_pct,
                      round(implied_pct, 4), plausible, classification, note)


def audit_journal(journal_path: Path) -> list:
    if not journal_path.exists():
        return []
    with open(journal_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [classify_journal_row(r) for r in rows]


def write_audit_artifact(impacts: list, out_dir: Path) -> Path:
    """Writes a NEW, separate provenance artifact -- never touches the
    original journal CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"v8_paper_impact_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root_cause": "memecoin/pumpfun_reserve_pricing.py -- erroneous /1e6 on vTokensInBondingCurve, fixed 2026-08-19",
        "n_rows": len(impacts),
        "rows": [asdict(i) for i in impacts],
    }, indent=2))
    return out_path
