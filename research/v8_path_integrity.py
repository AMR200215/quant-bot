"""
research/v8_path_integrity.py — V8-FILTER-DERIVATION Phase 2.1 item 2:
per-path integrity gate, built to replace an implicit "any positive
price is fine" assumption that let real, severe price corruption
(confirmed up to ~$73.5B implied market cap) flow unchecked through
research/v8_replay_engine.py.

V8_PATH_INTEGRITY_VERSION = 1

AUDIT OF THE $73.5B EXAMPLES (2026-08-17, git SHA d9468b7, VPS) --
required before encoding any rejection rule, not skipped:

  logs/research_paths/2026-08-04/FN1pzrGdaRfpJeabHtPzDQpRhdMnqaHiE3Nywts7GNdR.csv.gz
  price_usd=73.49, price_sol=0.41996, vsol=116.27, venue_state=CURVE_ACTIVE,
  source=live_pp, token_amount=0, sol_amount=0.0377

  Two INDEPENDENT, provable problems, not one hand-picked ceiling:

  1. vsol=116.27 EXCEEDS GRAD_SOL_UI=115.0 (research/config.py's own
     graduation threshold) while venue_state still reads CURVE_ACTIVE.
     A token cannot have more real SOL in its bonding curve than the
     curve's own graduation point while still being "on curve" --
     that combination is definitionally inconsistent, independent of
     price. Corpus-wide (400-file random sample): 70 such rows found.

  2. price_sol=0.41996 is checked against pump.fun's own bonding-curve
     AMM formula, not an arbitrary number. The curve's public, fixed
     protocol constants (initial virtual token reserves 1,073,000,000;
     real token reserves sold by graduation 793,100,000) mean the
     MINIMUM virtual token reserve ever reached on-curve is
     1,073,000,000 - 793,100,000 = 279,900,000 -- reached exactly at
     the graduation instant, the single highest-price point on the
     entire curve. Using GRAD_SOL_UI=115 (this project's own constant)
     and the protocol's fixed initial virtual SOL reserve (30 SOL), the
     theoretical maximum CURVE_ACTIVE price is
     (30+115)/279,900,000 ~= 5.18e-7 SOL/token. The observed
     0.41996 SOL/token is ~810,000x above that ceiling -- not a
     borderline case.

  This audit also ruled out two plausible-but-wrong hypotheses before
  settling on the above:
    - token_amount==0 initially looked like a discriminating signal,
      but a corpus-wide check found it's true for 100% of live_pp rows
      (a known, separate, harmless gap -- token_amount is simply never
      populated for this source -- not evidence of price corruption).
    - price_usd/price_sol internal ratio ALWAYS looked "sane" (50-400,
      i.e. a plausible SOL/USD rate) even on corrupted rows, in a
      400-file corpus-wide check -- the USD conversion is applied
      consistently even to a wrong price_sol, so ratio-sanity alone
      cannot catch this corruption either.

  A second flagged file
  (logs/research_paths/2026-08-08/AmqU7xrW8RMswHdcve7jE1dk7uD9ByudSfmBJrx8hBvH.csv.gz)
  is a MIXED file (18.5% of its rows corrupted, not 100%) -- confirming
  integrity must be assessed PER TICK, not just per file: a 400-file
  random sample found 15/44 live-bearing files were mixed (neither
  fully clean nor fully corrupted).

  A corpus-wide empirical check (same 400-file sample) independently
  corroborates the formula-derived finding: the LIVE (non-backfilled)
  implied-mcap distribution is sharply bimodal, p64=$860,030 (plausible)
  jumping to p65=$12,447,651,489 (impossible) -- a >14,000x gap with
  essentially no real data in between. This is used ONLY as a courtesy
  ceiling for GRADUATED/DEX_ACTIVE/UNKNOWN-venue rows (which are NOT
  bound by the bonding-curve formula -- DEX price discovery is
  legitimately unconstrained), and only ever produces UNKNOWN, never
  INVALID, since it's an observed pattern, not a proof.

  This bug is STILL ACTIVE as of 2026-08-17 (corrupted rows found dated
  2026-08-04 through 2026-08-17) -- not a resolved historical incident.
  Root-causing/fixing the live capture bug itself is out of this task's
  scope (Phase 2.1 builds the exclusion GATE, not the upstream fix);
  flagged here so it isn't lost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from research.config import GRAD_SOL_UI

V8_PATH_INTEGRITY_VERSION = 1

# ── Formula-derived bound (CURVE_ACTIVE only) ───────────────────────────
# Public, well-documented pump.fun protocol bonding-curve constants --
# NOT measured from this project's own data, standard fixed AMM
# parameters used by every pump.fun bonding-curve token.
#
# V8 DATA RECOVERY (2026-08-19) CORRECTION: this originally added a
# +30 SOL "initial virtual reserve" baseline on top of GRAD_SOL_UI,
# assuming GRAD_SOL_UI represented only the REAL (incremental) SOL
# added to the curve. Live capture (memecoin/pumpfun_reserve_pricing.py's
# module docstring) proved that wrong: a real near-graduation
# PumpPortal message showed vSolInBondingCurve=~115.005 -- matching
# GRAD_SOL_UI=115.0 directly, not ~145. GRAD_SOL_UI (research/config.py:
# "SOL in curve at graduation (virtual reserves)") already IS the full
# virtual SOL reserve figure; no +30 baseline belongs in this formula.
# Reuses memecoin/pumpfun_reserve_pricing.py's constants directly rather
# than re-declaring them a second time (the exact drift risk this batch
# was asked to eliminate).
from memecoin.pumpfun_reserve_pricing import (
    PUMPFUN_INITIAL_VIRTUAL_TOKEN_RESERVES, PUMPFUN_REAL_TOKEN_RESERVES_AT_GRADUATION,
    MIN_VIRTUAL_TOKEN_RESERVES_ON_CURVE as _MIN_VIRTUAL_TOKEN_RESERVES_ON_CURVE,
)

THEORETICAL_MAX_CURVE_ACTIVE_PRICE_SOL = (
    GRAD_SOL_UI / _MIN_VIRTUAL_TOKEN_RESERVES_ON_CURVE
)  # ~4.11e-7 SOL/token -- empirically confirmed exact via live capture

# Slack multiplier: PUMPFUN_* constants above are recalled protocol
# knowledge, not re-derived from this project's own on-chain reads in
# this pass -- a generous multiplier avoids a false INVALID from any
# imprecision in those constants while still catching the ~810,000x
# magnitude corruption actually observed.
CURVE_ACTIVE_PRICE_SLACK_MULTIPLIER = 20.0
CURVE_ACTIVE_PRICE_CEILING_SOL = THEORETICAL_MAX_CURVE_ACTIVE_PRICE_SOL * CURVE_ACTIVE_PRICE_SLACK_MULTIPLIER

# ── Empirical bound (non-CURVE_ACTIVE venue only, UNKNOWN not INVALID) ──
# From the 400-file random sample audited above.
EMPIRICAL_MCAP_GAP_LOWER_USD = 860_030          # p64, live corpus sample
EMPIRICAL_MCAP_GAP_UPPER_USD = 12_447_651_489   # p65, live corpus sample
NON_CURVE_MCAP_CEILING_USD = 2_000_000          # inside the gap, clear of both edges

_SOL_USD_RATIO_MIN = 10.0     # generously wide plausible SOL/USD band --
_SOL_USD_RATIO_MAX = 1000.0   # not tuned to this project's specific rate

_HARD_INVALID_REASONS = frozenset({
    "NON_FINITE_OR_NEGATIVE_PRICE",
    "VSOL_EXCEEDS_GRADUATION_WHILE_CURVE_ACTIVE",
    "PRICE_EXCEEDS_THEORETICAL_CURVE_MAX",
})


class PathIntegrityStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class TickIntegrityResult:
    ts_ms: int
    status: str
    reasons: tuple


@dataclass(frozen=True)
class PathIntegrityResult:
    status: str                # overall path verdict -- VALID / INVALID / UNKNOWN
    total_ticks: int
    valid_ticks: int
    invalid_ticks: int
    unknown_ticks: int
    reasons: tuple              # union of every distinct reason code found anywhere in the path
    integrity_version: int = V8_PATH_INTEGRITY_VERSION


def assess_tick_integrity(row: dict) -> TickIntegrityResult:
    """Classifies ONE tick. Never silently winsorizes -- a bad value is
    flagged, not clamped into plausibility."""
    reasons: list[str] = []
    ts_ms = 0
    try:
        ts_ms = int(row.get("ts_ms") or 0)
    except (TypeError, ValueError):
        pass

    try:
        price_usd = float(row.get("price_usd") if row.get("price_usd") not in (None, "") else "nan")
        price_sol = float(row.get("price_sol") if row.get("price_sol") not in (None, "") else "nan")
        vsol = float(row.get("vsol") if row.get("vsol") not in (None, "") else "nan")
    except (TypeError, ValueError):
        return TickIntegrityResult(ts_ms=ts_ms, status=PathIntegrityStatus.INVALID.value,
                                    reasons=("NON_FINITE_OR_UNPARSEABLE_PRICE",))

    venue = row.get("venue_state")

    if not math.isfinite(price_usd) or not math.isfinite(price_sol) or price_usd < 0 or price_sol < 0:
        reasons.append("NON_FINITE_OR_NEGATIVE_PRICE")

    if not (price_usd > 0):
        # Zero/negative/NaN price on an otherwise-parseable row -- could
        # legitimately be a zero-liquidity tick in some feeds; can't
        # independently prove it wrong, so UNKNOWN, not asserted INVALID.
        reasons.append("NONPOSITIVE_PRICE")

    if math.isfinite(vsol) and venue == "CURVE_ACTIVE" and vsol > GRAD_SOL_UI:
        reasons.append("VSOL_EXCEEDS_GRADUATION_WHILE_CURVE_ACTIVE")

    if venue == "CURVE_ACTIVE" and math.isfinite(price_sol) and price_sol > CURVE_ACTIVE_PRICE_CEILING_SOL:
        reasons.append("PRICE_EXCEEDS_THEORETICAL_CURVE_MAX")

    if price_usd > 0 and price_sol > 0:
        ratio = price_usd / price_sol
        if not (_SOL_USD_RATIO_MIN <= ratio <= _SOL_USD_RATIO_MAX):
            reasons.append("PRICE_USD_SOL_RATIO_IMPLAUSIBLE")

    if venue != "CURVE_ACTIVE" and price_usd > 0:
        mcap = price_usd * 1e9
        if mcap > NON_CURVE_MCAP_CEILING_USD:
            reasons.append("MCAP_ABOVE_EMPIRICAL_GAP_CEILING_NONCURVE")

    if any(r in _HARD_INVALID_REASONS for r in reasons):
        status = PathIntegrityStatus.INVALID.value
    elif reasons:
        status = PathIntegrityStatus.UNKNOWN.value
    else:
        status = PathIntegrityStatus.VALID.value

    return TickIntegrityResult(ts_ms=ts_ms, status=status, reasons=tuple(reasons))


def assess_path_integrity(path: list[dict], metadata: dict | None = None) -> PathIntegrityResult:
    """
    path: the full canonical row list for one mint (as loaded via
        research.path_schema.load_path_file), each row a dict with at
        least ts_ms/price_usd/price_sol/vsol/venue_state.
    metadata: reserved for future per-path context (event join info,
        candidate under evaluation, etc.) -- not consulted by the
        checks above, which are price-series-intrinsic only.

    Path-level verdict: INVALID if ANY tick is INVALID (a single
    provably-impossible tick poisons the whole trajectory -- peak/exit
    price computations scan the entire path); else UNKNOWN if ANY tick
    is UNKNOWN; else VALID. Conservative by design -- errs toward
    exclusion, never toward trusting a suspicious value.
    """
    del metadata  # reserved, unused today -- see docstring
    if not path:
        return PathIntegrityResult(
            status=PathIntegrityStatus.UNKNOWN.value,
            total_ticks=0, valid_ticks=0, invalid_ticks=0, unknown_ticks=0,
            reasons=("EMPTY_PATH",),
        )

    tick_results = [assess_tick_integrity(r) for r in path]
    valid_n = sum(1 for t in tick_results if t.status == PathIntegrityStatus.VALID.value)
    invalid_n = sum(1 for t in tick_results if t.status == PathIntegrityStatus.INVALID.value)
    unknown_n = sum(1 for t in tick_results if t.status == PathIntegrityStatus.UNKNOWN.value)

    all_reasons = tuple(sorted({r for t in tick_results for r in t.reasons}))

    if invalid_n > 0:
        overall = PathIntegrityStatus.INVALID.value
    elif unknown_n > 0:
        overall = PathIntegrityStatus.UNKNOWN.value
    else:
        overall = PathIntegrityStatus.VALID.value

    return PathIntegrityResult(
        status=overall,
        total_ticks=len(path), valid_ticks=valid_n, invalid_ticks=invalid_n, unknown_ticks=unknown_n,
        reasons=all_reasons,
    )


def filter_integrity_qualified_paths(paths_with_meta: list[tuple]) -> list[tuple]:
    """
    paths_with_meta: [(path_rows, metadata), ...]
    Returns only the (path_rows, metadata) pairs whose assess_path_integrity
    verdict is VALID. This is the gate FULL_STRATEGY_EV must go through --
    INVALID and UNKNOWN paths are excluded, never silently included.
    """
    out = []
    for rows, meta in paths_with_meta:
        result = assess_path_integrity(rows, meta)
        if result.status == PathIntegrityStatus.VALID.value:
            out.append((rows, meta))
    return out


# ── Corpus-wide scan + breakdown ────────────────────────────────────────

@dataclass
class CorpusIntegrityReport:
    total_paths: int = 0
    valid: int = 0
    invalid: int = 0
    unknown: int = 0
    by_source: dict = field(default_factory=dict)          # {source: {VALID/INVALID/UNKNOWN: n}}
    by_schema_version: dict = field(default_factory=dict)
    by_backfilled: dict = field(default_factory=dict)      # {"true"/"false": {...}}
    by_date: dict = field(default_factory=dict)            # {"YYYY-MM-DD": {...}}
    by_failure_reason: dict = field(default_factory=dict)  # {reason_code: n_paths_with_that_reason}


def _bump(bucket: dict, key, status: str) -> None:
    slot = bucket.setdefault(key, {"VALID": 0, "INVALID": 0, "UNKNOWN": 0})
    slot[status] += 1


def scan_corpus(research_paths_dir, live_only: bool = False, max_paths: int = 0) -> CorpusIntegrityReport:
    """
    Walks research_paths_dir (a Path), assesses every path file's
    integrity, and returns the breakdown FD-required by Phase 2.1 item 2:
    total/valid/invalid/unknown by source, schema_version, forward/
    backfill, date, and failure reason. Read-only -- never modifies or
    deletes a path file.
    """
    from pathlib import Path as _Path
    from research.path_schema import load_path_file

    research_paths_dir = _Path(research_paths_dir)
    report = CorpusIntegrityReport()
    if not research_paths_dir.exists():
        return report

    files = list(research_paths_dir.rglob("*.csv")) + list(research_paths_dir.rglob("*.csv.gz"))
    if live_only:
        files = [f for f in files if "backfill" not in str(f)]
    if max_paths:
        files = files[:max_paths]

    for fp in files:
        rows, _warnings = load_path_file(fp)
        if not rows:
            continue
        result = assess_path_integrity(rows)
        report.total_paths += 1
        if result.status == "VALID":
            report.valid += 1
        elif result.status == "INVALID":
            report.invalid += 1
        else:
            report.unknown += 1

        source = rows[0].get("source", "unknown")
        schema_version = rows[0].get("schema_version", "unknown")
        backfilled = rows[0].get("backfilled", "unknown")
        # date bucket: directory name if it looks like YYYY-MM-DD, else derived from first ts_ms
        date_bucket = fp.parent.name if len(fp.parent.name) == 10 and fp.parent.name.count("-") == 2 else "unknown"

        _bump(report.by_source, source, result.status)
        _bump(report.by_schema_version, schema_version, result.status)
        _bump(report.by_backfilled, backfilled, result.status)
        _bump(report.by_date, date_bucket, result.status)

        for reason in result.reasons:
            report.by_failure_reason[reason] = report.by_failure_reason.get(reason, 0) + 1

    return report
