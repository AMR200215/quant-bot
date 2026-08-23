"""
research/v8_collection_yield.py — READINESS DENOMINATOR AUDIT/CORRECTION
batch: separates population counts from collector-yield counts.

The bug this fixes: research/v8_candidate_path_coverage.py and
research/v8_forward_readiness_report.py's execution-proxy coverage both
used the ENTIRE historical venue-qualified population as the readiness
denominator. Most of that population:
  - predates reliable path collection (pre v3 schema / pre price-fix),
  - predates PumpPortal funding (subscribeTokenTrade was rejected
    server-side for the whole prior era -- confirmed via a live
    "Minimum balance not met" rejection, root-caused and fixed by
    funding the account, see docs/RECEIPTS.md),
  - or was never admitted by the probabilistic path sampler (P16-3's
    budget-paced admission controller intentionally admits less than
    100% of eligible events -- that is by design, not a failure).

None of those rows could ever have produced a path or execution-proxy
observation, REGARDLESS of current collector health. Counting them as
permanent "missing" observations understates readiness for reasons that
have nothing to do with whether the pipeline works today.

This module does NOT change the sampling policy, the candidate/exit
registries, or any absolute evidence threshold (MIN_PATH_N,
MIN_PATH_COVERAGE_PCT, EXECUTION_PROXY_MIN_N,
EXECUTION_PROXY_MIN_COVERAGE_PCT are all untouched, imported as-is from
their existing modules). It only changes what population those
thresholds are evaluated against, deriving that population from real,
objective, non-outcome-based evidence:

  - PRICE_CORRECTION_DEPLOY_UTC: a fixed, documented constant sourced
    from docs/RECEIPTS.md's "V8 DATA RECOVERY + FORWARD READINESS
    BATCH" entry (git SHA b62a1c3, 2026-08-19) -- the exact deploy that
    fixed the live_pp price-corruption bug and bumped the path schema
    to v3. Path data from before this deploy is real but from a
    different (corrupted-price, v2-schema) era and must not be silently
    blended into "the collector is working now" evidence.
  - The PumpPortal-funded era start is NOT a hardcoded guess -- it is
    read directly from the earliest real observation the funded
    collector has ever produced (logs/research_execution_proxy/
    execution_proxy_log.jsonl's own earliest `observed_at`, or absent
    that, the earliest real (non-backfilled, source=live_pp) tick in
    any v3-schema path file). This is a deployment/mechanism fact (the
    first moment the funded key produced ANY output), not an
    outcome-derived fit to a specific candidate's performance -- the
    same boundary applies identically to every candidate.
  - admission_log.jsonl (research/peak_tracker.py's P16-3/P16-4
    provenance log) is the authoritative record of which mints the
    sampler actually admitted, and with what recorded inclusion
    probability -- used both for path_admitted_n and for IPW-weighted
    effective sample size.

Units discipline: admission_log.jsonl and execution_proxy_log.jsonl are
both keyed by token_address only (no event_id carried on either -- a
real, separately-noted limitation, not fixed here since that is a
collection-code change, out of this audit's scope). To avoid mixing a
mint-count numerator against an event-count denominator (the bug this
audit was asked to check for), every quantity in this module downstream
of ambiguous-mint exclusion is computed at MINT granularity, consistently,
on both sides of every ratio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from research.v8_entry_alignment import find_ambiguous_mints, resolve_entry_alignment, EntryAlignmentExclusion
from research.v8_path_integrity import assess_path_integrity, PathIntegrityStatus
from research.path_schema import load_path_file
from research.v8_statistical_selection import effective_n_after_ipw

# Sourced from docs/RECEIPTS.md, "V8 DATA RECOVERY + FORWARD READINESS
# BATCH", git SHA b62a1c3 -- the deploy that fixed the live_pp
# /1e6 price-corruption bug and bumped PATH_SCHEMA_VERSION 2 -> 3.
PRICE_CORRECTION_DEPLOY_UTC = datetime(2026, 8, 19, 0, 0, 0, tzinfo=timezone.utc)

ERA_BOUNDARY_PROVENANCE = {
    "PRICE_CORRECTION_DEPLOY_UTC": "docs/RECEIPTS.md 'V8 DATA RECOVERY + FORWARD READINESS BATCH', "
                                    "git SHA b62a1c3, 2026-08-19 -- live_pp price-fix + path schema v3 deploy",
    "pp_funded_era_start": "derived live from the earliest real observation the funded collector has ever "
                            "produced (execution_proxy_log.jsonl's earliest observed_at, falling back to the "
                            "earliest real live_pp tick in any v3+ path file) -- a mechanism/deployment fact, "
                            "not fit to any candidate's outcomes",
}


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


def load_admission_log_by_mint(repo_root: Path) -> dict:
    rows = _read_jsonl(repo_root / "logs" / "research_admission" / "admission_log.jsonl")
    by_mint: dict = {}
    for r in rows:
        mint = r.get("token_address")
        if mint:
            by_mint.setdefault(mint, []).append(r)
    return by_mint


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _earliest_real_tick_ts(repo_root: Path) -> Optional[datetime]:
    """Fallback path-file scan for pp_funded_era_start when
    execution_proxy_log.jsonl doesn't exist yet or is empty. Only scans
    the last few UTC day directories -- real (funded-era) ticks cannot
    predate the price-correction deploy, so older directories cannot
    contain evidence this function needs."""
    paths_dir = repo_root / "logs" / "research_paths"
    if not paths_dir.exists():
        return None
    earliest = None
    day_dirs = sorted((d for d in paths_dir.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True)
    for day_dir in day_dirs[:5]:
        for f in list(day_dir.glob("*.csv")) + list(day_dir.glob("*.csv.gz")):
            rows, _warnings = load_path_file(f)
            for r in rows:
                if r.get("source") != "live_pp" or r.get("backfilled") == "true":
                    continue
                try:
                    ts_ms = int(r.get("ts_ms"))
                except (TypeError, ValueError):
                    continue
                ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                if earliest is None or ts < earliest:
                    earliest = ts
    return earliest


def pp_funded_era_start(repo_root: Path, execution_proxy_rows: Optional[list] = None) -> Optional[datetime]:
    if execution_proxy_rows is None:
        execution_proxy_rows = _read_jsonl(repo_root / "logs" / "research_execution_proxy" / "execution_proxy_log.jsonl")
    candidates = [_parse_iso(r.get("observed_at", "")) for r in execution_proxy_rows]
    candidates = [c for c in candidates if c is not None]
    if candidates:
        return min(candidates)
    return _earliest_real_tick_ts(repo_root)


def trustworthy_collection_era_start(repo_root: Path, execution_proxy_rows: Optional[list] = None) -> Optional[datetime]:
    """The binding constraint on collector-yield eligibility: the LATER
    of (a) the price-correction/v3-schema deploy and (b) the moment the
    PumpPortal-funded feed first produced real output. Returns None if
    neither the funded collector nor any real tick has ever been
    observed -- callers must treat that as "not yet determinable", not
    silently substitute a guess."""
    funded_start = pp_funded_era_start(repo_root, execution_proxy_rows)
    if funded_start is None:
        return None
    return max(PRICE_CORRECTION_DEPLOY_UTC, funded_start)


@dataclass(frozen=True)
class CollectionYield:
    candidate_venue_qualified_n: int          # unchanged population count (event-level, all history)
    ambiguous_excluded_mints_n: int
    no_admission_record_n: int                # in-era, non-ambiguous mints with zero admission_log rows
    path_collection_eligible_n: int           # in-era, non-ambiguous mints (the collector-yield denominator)
    path_admitted_n: int                      # of those, admitted by the probabilistic sampler
    admitted_with_tick_n: int                 # of admitted, got >=1 real (non-backfilled) tick
    admitted_with_valid_usable_path_n: int    # of those, passed alignment + integrity
    admitted_path_yield_pct: float            # admitted_with_valid_usable_path_n / path_admitted_n
    ipw_effective_n: float                    # Kish ESS of path_admitted_n under 1/probability weights
    execution_proxy_collection_eligible_n: int  # == path_admitted_n (same admitted, in-era population)
    execution_proxy_observed_n: int           # mint-level, consistent units with the eligible count
    execution_proxy_coverage_pct: float
    unique_forward_days: int                  # unique calendar days among admitted mints
    era_start: Optional[str]                  # ISO string, or None if undetermined
    era_undetermined: bool


def _empty_collection_yield(candidate_venue_qualified_n: int, ambiguous_excluded_mints_n: int) -> CollectionYield:
    return CollectionYield(
        candidate_venue_qualified_n=candidate_venue_qualified_n,
        ambiguous_excluded_mints_n=ambiguous_excluded_mints_n,
        no_admission_record_n=0, path_collection_eligible_n=0, path_admitted_n=0,
        admitted_with_tick_n=0, admitted_with_valid_usable_path_n=0, admitted_path_yield_pct=0.0,
        ipw_effective_n=0.0, execution_proxy_collection_eligible_n=0, execution_proxy_observed_n=0,
        execution_proxy_coverage_pct=0.0, unique_forward_days=0, era_start=None, era_undetermined=True,
    )


def compute_collection_yield(
    candidate_events: list,
    all_events_for_ambiguity: list,
    candidate: dict,
    repo_root: Path,
    admission_log_by_mint: Optional[dict] = None,
    execution_proxy_rows: Optional[list] = None,
) -> CollectionYield:
    """
    candidate_events: this candidate's venue-qualified events (same
        population v8_candidate_path_coverage.compute_candidate_path_coverage
        receives) -- event-level, full history, unchanged.
    all_events_for_ambiguity: the FULL clean-cohort event universe, for
        find_ambiguous_mints() -- same convention as the path-coverage module.
    """
    if admission_log_by_mint is None:
        admission_log_by_mint = load_admission_log_by_mint(repo_root)
    if execution_proxy_rows is None:
        execution_proxy_rows = _read_jsonl(repo_root / "logs" / "research_execution_proxy" / "execution_proxy_log.jsonl")

    candidate_venue_qualified_n = len(candidate_events)
    ambiguous_mints = find_ambiguous_mints(all_events_for_ambiguity)

    all_mints = {r.get("token_address") for r in candidate_events if r.get("token_address")}
    non_ambiguous_mints = all_mints - ambiguous_mints
    ambiguous_excluded_mints_n = len(all_mints) - len(non_ambiguous_mints)

    # One representative event per non-ambiguous mint. Safe: ambiguous
    # (multi-alert) mints are already excluded, so every remaining mint
    # maps to exactly one event in candidate_events.
    by_mint = {}
    for r in candidate_events:
        mint = r.get("token_address")
        if mint in non_ambiguous_mints and mint not in by_mint:
            by_mint[mint] = r

    era_start = trustworthy_collection_era_start(repo_root, execution_proxy_rows)
    if era_start is None:
        return _empty_collection_yield(candidate_venue_qualified_n, ambiguous_excluded_mints_n)
    era_start_epoch = era_start.timestamp()

    eligible_mints = {}
    for mint, r in by_mint.items():
        at = r.get("alert_time")
        if not at:
            continue
        dt = _parse_iso(at)
        if dt is None or dt.timestamp() < era_start_epoch:
            continue
        eligible_mints[mint] = r

    path_collection_eligible_n = len(eligible_mints)

    no_admission_record_n = 0
    admitted_mints = {}
    for mint in eligible_mints:
        rows = admission_log_by_mint.get(mint, [])
        if not rows:
            no_admission_record_n += 1
            continue
        admit_row = next((rw for rw in rows if rw.get("path_admitted")), None)
        if admit_row is not None:
            admitted_mints[mint] = admit_row

    path_admitted_n = len(admitted_mints)

    admitted_with_tick_n = 0
    admitted_with_valid_usable_path_n = 0
    ipw_weights = []
    admitted_days = set()

    for mint, admit_row in admitted_mints.items():
        event = eligible_mints[mint]
        at = event.get("alert_time")
        if at:
            admitted_days.add(at[:10])

        prob = admit_row.get("path_sampling_probability")
        weight = 1.0 / prob if prob and prob > 0 else 1.0
        ipw_weights.append(weight)

        path_file = event.get("path_file")
        if not path_file:
            continue
        full_path = repo_root / path_file
        if not full_path.exists():
            gz_path = full_path.with_suffix(full_path.suffix + ".gz")
            full_path = gz_path if gz_path.exists() else None
        if full_path is None:
            continue

        rows, _warnings = load_path_file(full_path)
        if not rows:
            continue
        has_real_tick = any(r.get("source") == "live_pp" and r.get("backfilled") != "true" for r in rows)
        if not has_real_tick:
            continue
        admitted_with_tick_n += 1

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
            continue

        alignment = resolve_entry_alignment(event, typed_rows, candidate, ambiguous_mints)
        if isinstance(alignment, EntryAlignmentExclusion):
            continue
        integrity = assess_path_integrity(rows)
        if integrity.status != PathIntegrityStatus.VALID.value:
            continue
        admitted_with_valid_usable_path_n += 1

    admitted_path_yield_pct = (
        round(admitted_with_valid_usable_path_n / path_admitted_n * 100, 2) if path_admitted_n else 0.0
    )
    ipw_effective_n = effective_n_after_ipw(ipw_weights)

    proxy_ok_mints = {r.get("token_address") for r in execution_proxy_rows
                       if r.get("status") == "OK" and r.get("token_address") in admitted_mints}
    execution_proxy_collection_eligible_n = path_admitted_n
    execution_proxy_observed_n = len(proxy_ok_mints)
    execution_proxy_coverage_pct = (
        round(execution_proxy_observed_n / execution_proxy_collection_eligible_n * 100, 2)
        if execution_proxy_collection_eligible_n else 0.0
    )

    return CollectionYield(
        candidate_venue_qualified_n=candidate_venue_qualified_n,
        ambiguous_excluded_mints_n=ambiguous_excluded_mints_n,
        no_admission_record_n=no_admission_record_n,
        path_collection_eligible_n=path_collection_eligible_n,
        path_admitted_n=path_admitted_n,
        admitted_with_tick_n=admitted_with_tick_n,
        admitted_with_valid_usable_path_n=admitted_with_valid_usable_path_n,
        admitted_path_yield_pct=admitted_path_yield_pct,
        ipw_effective_n=ipw_effective_n,
        execution_proxy_collection_eligible_n=execution_proxy_collection_eligible_n,
        execution_proxy_observed_n=execution_proxy_observed_n,
        execution_proxy_coverage_pct=execution_proxy_coverage_pct,
        unique_forward_days=len(admitted_days),
        era_start=era_start.isoformat(),
        era_undetermined=False,
    )
