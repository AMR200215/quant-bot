"""
research/v8_entry_alignment.py — V8-FILTER-DERIVATION Phase 2 (P2-7):
real entry-time alignment for replaying a candidate against a path file.

Every prior research tool in this repo (research/analysis/replay_exits.py,
research/analysis/path_stats.py) implicitly assumes `rows[0]` IS the entry
tick. That's wrong for two independent reasons, both fixed here:

  1. A path file's first recorded tick is whenever PeakTracker/backfill
     started capturing that mint -- not necessarily the moment of the
     alert that made it a candidate, and DEFINITELY not the moment a
     T0+capture feature (like progress_at_signal) actually became known.
     A live bot cannot act on a feature before it has that feature.
  2. A mint that re-alerts (FD5: ~22.6% of rows share a token_address
     with another row) has ONE continuous path file across MULTIPLE
     alert events. Attaching any one of those alerts' entry decision to
     that shared file is ambiguous unless there's only one alert for
     that mint -- silently picking the "nearest" one would be a guess,
     not a join.

This module resolves both, producing an EntryAlignment (or a typed
exclusion reason) per (event, candidate) pair -- never guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from research.v8_feature_enforcement import _feature_by_name

# Nominal fallback delay (ms) used only when an event has no real observed
# per-row capture lag -- e.g. historical rows written before
# progress_capture_lag_ms existed. Sourced from the feature registry's own
# availability_delay_ms, never invented here.
_DEFAULT_T0_CAPTURE_DELAY_MS = 500


@dataclass(frozen=True)
class EntryAlignment:
    event_id: str
    token_address: str
    alert_ts_ms: int
    decision_available_ts: int   # earliest ts a live bot could have the candidate's features
    entry_target_ts: int         # == decision_available_ts (kept distinct for FD14/FD20 readability)
    entry_ts_ms: int             # ts_ms of the first executable tick >= entry_target_ts
    entry_price: float
    entry_lag_ms: int            # entry_ts_ms - entry_target_ts (>= 0)
    entry_source: str            # "T0" | "T0+capture:real_lag" | "T0+capture:nominal_fallback"


@dataclass(frozen=True)
class EntryAlignmentExclusion:
    event_id: str
    token_address: str
    reason: str                  # "AMBIGUOUS_PATH_EVENT_JOIN" | "NO_EXECUTABLE_TICK_AFTER_TARGET" | "NO_PATH_ROWS" | "BAD_ALERT_TIME"


def _parse_ts_ms(alert_time: str) -> Optional[int]:
    try:
        dt = datetime.fromisoformat(alert_time.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except (ValueError, AttributeError, TypeError):
        return None


def _feature_delay_ms(feature_name: str) -> int:
    f = _feature_by_name(feature_name)
    if f is None:
        return _DEFAULT_T0_CAPTURE_DELAY_MS
    return int(f.get("availability_delay_ms", _DEFAULT_T0_CAPTURE_DELAY_MS))


def find_ambiguous_mints(events: list[dict]) -> set[str]:
    """FD5/P2-7: any mint (token_address) with more than one event row in
    `events` shares a single continuous path file across multiple alerts --
    attaching a path-based entry decision to any one of them is a guess.
    Returns the set of token_addresses to exclude via AMBIGUOUS_PATH_EVENT_JOIN.
    """
    counts: dict[str, int] = {}
    for e in events:
        addr = e.get("token_address")
        if addr:
            counts[addr] = counts.get(addr, 0) + 1
    return {addr for addr, n in counts.items() if n > 1}


def resolve_entry_alignment(
    event: dict,
    path_rows: list[dict],
    candidate: dict,
    ambiguous_mints: set[str],
) -> "EntryAlignment | EntryAlignmentExclusion":
    """
    event: a research_tokens row -- must have event_id, token_address,
        alert_time, and (optionally) progress_capture_lag_ms.
    path_rows: this mint's canonical path rows (ts_ms ascending, typed).
    candidate: a research/v8_candidate_registry.py candidate dict --
        used for its required_features to determine the real decision
        delay (T0 vs T0+capture), never assumed uniform across candidates.
    ambiguous_mints: precomputed via find_ambiguous_mints() over the
        FULL event set being evaluated -- passed in rather than
        recomputed per-call so the ambiguity determination is made once,
        consistently, across the whole run.
    """
    event_id = event.get("event_id", "")
    token_address = event.get("token_address", "")

    if token_address in ambiguous_mints:
        return EntryAlignmentExclusion(event_id, token_address, "AMBIGUOUS_PATH_EVENT_JOIN")

    if not path_rows:
        return EntryAlignmentExclusion(event_id, token_address, "NO_PATH_ROWS")

    alert_ts_ms = _parse_ts_ms(event.get("alert_time", ""))
    if alert_ts_ms is None:
        return EntryAlignmentExclusion(event_id, token_address, "BAD_ALERT_TIME")

    # Determine decision delay from the candidate's OWN required features
    # (never a single global constant -- P0 has zero T0+capture features,
    # P1/P2/P3/BASELINE-0 all require progress_at_signal + venue_state_at_signal).
    t0_capture_features = [
        f for f in candidate.get("required_features", [])
        if f in ("progress_at_signal", "vsol_at_signal", "venue_state_at_signal")
    ]

    if not t0_capture_features:
        decision_available_ts = alert_ts_ms
        entry_source = "T0"
    else:
        real_lag = event.get("progress_capture_lag_ms")
        if real_lag is not None:
            decision_available_ts = alert_ts_ms + int(real_lag)
            entry_source = "T0+capture:real_lag"
        else:
            nominal_delay = max(_feature_delay_ms(f) for f in t0_capture_features)
            decision_available_ts = alert_ts_ms + nominal_delay
            entry_source = "T0+capture:nominal_fallback"

    entry_target_ts = decision_available_ts

    executable = [r for r in path_rows if r["ts_ms"] >= entry_target_ts]
    if not executable:
        return EntryAlignmentExclusion(event_id, token_address, "NO_EXECUTABLE_TICK_AFTER_TARGET")

    first_tick = executable[0]
    entry_lag_ms = first_tick["ts_ms"] - entry_target_ts

    return EntryAlignment(
        event_id=event_id,
        token_address=token_address,
        alert_ts_ms=alert_ts_ms,
        decision_available_ts=decision_available_ts,
        entry_target_ts=entry_target_ts,
        entry_ts_ms=first_tick["ts_ms"],
        entry_price=first_tick["price_usd"],
        entry_lag_ms=entry_lag_ms,
        entry_source=entry_source,
    )
