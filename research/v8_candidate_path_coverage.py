"""
research/v8_candidate_path_coverage.py — V8 READINESS MONITOR
CORRECTION batch, item 2: candidate-specific representative path
coverage.

Fixes a real bug: research/v8_forward_readiness_report.py previously
used ALL VALID paths in the WHOLE corpus as every candidate's
representative_path_n. Wrong -- different candidates have different
eligible populations, and a path's VALID status says nothing about
whether it belongs to a given candidate's events at all.

Joins candidate-eligible events to their path files via
research_tokens' own path_file column (the direct, authoritative link
-- never a mint-only guess), respecting:
  - ambiguous-mint exclusion (research.v8_entry_alignment.find_ambiguous_mints)
  - entry-time alignment (research.v8_entry_alignment.resolve_entry_alignment)
  - path integrity == VALID (research.v8_path_integrity.assess_path_integrity)

path_file is written per (mint, day) in research/peak_tracker.py
(_open_csv opens "{addr}.csv" inside today's date directory) -- a mint
that re-alerts on the same day would have every one of those alerts'
research_tokens rows pointing at the SAME physical file, which is
exactly the ambiguity find_ambiguous_mints() already exists to catch.
Ambiguity is a property of the mint's FULL alert history, not just this
candidate's eligible subset -- callers must pass the full clean-cohort
event universe for ambiguity detection, not just candidate_events.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from research.v8_entry_alignment import find_ambiguous_mints, resolve_entry_alignment, EntryAlignmentExclusion
from research.v8_path_integrity import assess_path_integrity, PathIntegrityStatus
from research.path_schema import load_path_file


def _typed_rows(raw_rows: list) -> list:
    """load_path_file returns canonical STRING-typed CSV rows (for
    round-trip fidelity); resolve_entry_alignment needs numeric ts_ms
    for comparison -- same conversion pattern already established in
    research/analysis/replay_exits.py and path_stats.py's own
    _load_path helpers. assess_path_integrity tolerates strings
    directly (casts internally), so this conversion is only needed
    for the alignment step."""
    out = []
    for r in raw_rows:
        try:
            typed = dict(r)
            typed["ts_ms"] = int(r["ts_ms"])
            typed["price_usd"] = float(r["price_usd"])
            out.append(typed)
        except (KeyError, ValueError, TypeError):
            continue
    return out


@dataclass(frozen=True)
class CandidatePathCoverage:
    eligible_n: int
    usable_n: int                        # events with a VALID, alignment-resolved representative path
    coverage_pct: float
    ambiguous_excluded_n: int
    no_path_file_n: int
    path_file_missing_or_empty_n: int
    alignment_excluded_n: int
    integrity_invalid_or_unknown_n: int


def compute_candidate_path_coverage(
    candidate_events: list,
    all_events_for_ambiguity: list,
    candidate: dict,
    repo_root: Path,
) -> CandidatePathCoverage:
    """
    candidate_events: this candidate's eligible events -- dicts with at
        least event_id, token_address, alert_time, path_file, and
        (optionally) progress_capture_lag_ms.
    all_events_for_ambiguity: the FULL clean-cohort event universe
        (every candidate is drawn from the same superset) -- used only
        to compute find_ambiguous_mints() correctly, never filtered to
        this candidate first.
    candidate: the frozen candidate registry dict (for required_features,
        used by resolve_entry_alignment's decision-delay logic).
    """
    ambiguous_mints = find_ambiguous_mints(all_events_for_ambiguity)

    eligible_n = len(candidate_events)
    usable_n = 0
    ambiguous_excluded = 0
    no_path_file = 0
    missing_or_empty = 0
    alignment_excluded = 0
    integrity_bad = 0

    for event in candidate_events:
        token_address = event.get("token_address")
        if token_address in ambiguous_mints:
            ambiguous_excluded += 1
            continue

        path_file = event.get("path_file")
        if not path_file:
            no_path_file += 1
            continue

        full_path = repo_root / path_file
        if not full_path.exists():
            # research/peak_tracker.py gzips the previous UTC day's path
            # files in place (daily rotation, by design) without updating
            # research_tokens.path_file -- a path recorded as "*.csv" may
            # now only exist on disk as "*.csv.gz". Check that sibling
            # before concluding the file is genuinely missing.
            gz_path = full_path.with_suffix(full_path.suffix + ".gz")
            if gz_path.exists():
                full_path = gz_path
            else:
                missing_or_empty += 1
                continue

        rows, _warnings = load_path_file(full_path)
        if not rows:
            missing_or_empty += 1
            continue
        typed_rows = _typed_rows(rows)
        if not typed_rows:
            missing_or_empty += 1
            continue

        alignment = resolve_entry_alignment(event, typed_rows, candidate, ambiguous_mints)
        if isinstance(alignment, EntryAlignmentExclusion):
            alignment_excluded += 1
            continue

        integrity = assess_path_integrity(rows)
        if integrity.status != PathIntegrityStatus.VALID.value:
            integrity_bad += 1
            continue

        usable_n += 1

    coverage_pct = round(usable_n / eligible_n * 100, 2) if eligible_n else 0.0

    return CandidatePathCoverage(
        eligible_n=eligible_n, usable_n=usable_n, coverage_pct=coverage_pct,
        ambiguous_excluded_n=ambiguous_excluded, no_path_file_n=no_path_file,
        path_file_missing_or_empty_n=missing_or_empty, alignment_excluded_n=alignment_excluded,
        integrity_invalid_or_unknown_n=integrity_bad,
    )
