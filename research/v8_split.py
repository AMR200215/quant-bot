"""
research/v8_split.py — V8-FILTER-DERIVATION Phase 2 (FD10/FD11):
chronological, mint-grouped train/validation/holdout splitting.

Two requirements that would silently conflict if handled separately:
  1. FD10: split chronologically BY TIME, not by row count -- memecoin
     regimes change over time, a random or row-count split would let
     "future" rows leak into training.
  2. FD5 (confirmed empirically, Phase 1: 22.5% of a 1000-row sample had
     the same token_address across multiple independent rows): the same
     mint must never appear in more than one split, or a model can
     effectively "see" a token's later behavior via an earlier row of
     the same mint sitting in a different split.

Resolution: a token's FIRST-SEEN timestamp (not its last, not each row
individually) decides which split its entire row group belongs to.
This matches how a real prospective decision actually works -- once
you've observed a token for the first time and assigned it to a period,
every later re-alert of the same token is downstream of that same
original decision, not a fresh, independent one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class SplitResult:
    train: list
    validation: list
    holdout: list
    train_cutoff: float
    validation_cutoff: float
    group_count: int
    ambiguous_groups: int   # groups whose rows span a cutoff -- see docstring; count only, not silently dropped


def grouped_chronological_split(
    rows: list[T],
    group_key: Callable[[T], str],
    time_key: Callable[[T], float],
    train_frac: float = 0.6,
    validation_frac: float = 0.2,
) -> SplitResult:
    """
    Splits `rows` into (train, validation, holdout) such that:
      - cutoffs are computed over the TIME RANGE (min/max time_key),
        not row counts;
      - every row sharing the same group_key(row) (e.g. token_address)
        lands in exactly one split, decided by that group's EARLIEST
        time_key value.

    train_frac + validation_frac must be < 1.0 (the remainder is holdout).
    Raises ValueError on empty input or invalid fractions.
    """
    if not rows:
        raise ValueError("grouped_chronological_split: rows must not be empty")
    if not (0 < train_frac < 1) or not (0 < validation_frac < 1) or train_frac + validation_frac >= 1:
        raise ValueError(f"invalid fractions: train_frac={train_frac}, validation_frac={validation_frac}")

    times = [time_key(r) for r in rows]
    t_min, t_max = min(times), max(times)
    span = t_max - t_min
    train_cutoff = t_min + train_frac * span
    validation_cutoff = t_min + (train_frac + validation_frac) * span

    # First-seen time per group.
    first_seen: dict[str, float] = {}
    group_spans_cutoff: dict[str, set] = {}
    for r in rows:
        g = group_key(r)
        t = time_key(r)
        if g not in first_seen or t < first_seen[g]:
            first_seen[g] = t
        # Track which "natural" (unsnapped) bucket each of this group's
        # own rows would fall into, purely to count (not silently hide)
        # how often grouping actually overrides the naive per-row split.
        bucket = "train" if t < train_cutoff else ("validation" if t < validation_cutoff else "holdout")
        group_spans_cutoff.setdefault(g, set()).add(bucket)

    ambiguous_groups = sum(1 for buckets in group_spans_cutoff.values() if len(buckets) > 1)

    def _group_split(g: str) -> str:
        t = first_seen[g]
        if t < train_cutoff:
            return "train"
        if t < validation_cutoff:
            return "validation"
        return "holdout"

    train, validation, holdout = [], [], []
    for r in rows:
        split = _group_split(group_key(r))
        (train if split == "train" else validation if split == "validation" else holdout).append(r)

    return SplitResult(
        train=train, validation=validation, holdout=holdout,
        train_cutoff=train_cutoff, validation_cutoff=validation_cutoff,
        group_count=len(first_seen), ambiguous_groups=ambiguous_groups,
    )
