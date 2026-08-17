"""
research/v8_split.py — V8-FILTER-DERIVATION Phase 2 (FD10/FD11):
chronological, mint-grouped train/validation/holdout splitting.

P2-1 CORRECTION (2026-08-17): the original version of this module
decided a token's ENTIRE row group by its FIRST-SEEN timestamp. That
policy prevents mint overlap across splits, but it does so by moving a
token's LATER rows backward into an earlier split -- e.g. a mint alerted
once in the train era and again in the holdout era would have BOTH rows
placed in train, because train contained its first-seen timestamp. That
later row's outcome is real future information leaking into training.
This is temporal leakage, not resolved mint-overlap prevention.

Corrected policy (no exceptions, no "closest bucket" heuristics):
  1. Each ROW's own natural bucket is decided by its own time_key alone
     (train / validation / holdout, from the global time-range cutoffs).
  2. Rows are grouped by group_key (e.g. token_address).
  3. If every row in a group shares the same natural bucket, the whole
     group is assigned there (this is the common case -- ~77% of mints
     per the Phase 1 FD5 audit only ever appear once).
  4. If a group's rows span MORE THAN ONE natural bucket, the ENTIRE
     group is PURGED -- excluded from all three splits, never moved
     forward or backward. This is counted (boundary_spanning_groups /
     purged_rows), never silently dropped.

This guarantees, simultaneously:
  - strict per-split time ordering: max(train.t) < train_cutoff,
    min(validation.t) >= train_cutoff, max(validation.t) < validation_cutoff,
    min(holdout.t) >= validation_cutoff
  - zero mint overlap between any two splits (a purged group is in
    none of them, so it can't overlap either)
Both are asserted below, not just documented.

FD5 (confirmed empirically, Phase 1: 22.5% of a 1000-row sample had the
same token_address across multiple independent rows) is why grouping is
required at all -- without it a model could "see" a token's later
behavior via an earlier row of the same mint sitting in a different
split.
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
    group_count: int              # distinct groups seen (including purged ones)
    boundary_spanning_groups: int  # groups purged for spanning >1 natural bucket
    purged_rows: int              # total rows excluded because their group was purged


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
      - every row's OWN time_key decides its natural bucket -- never a
        group-level first-seen/last-seen proxy;
      - a group (all rows sharing group_key(row)) is kept only if ALL
        its rows share one natural bucket; otherwise the whole group is
        purged from every split (boundary_spanning_groups / purged_rows).

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

    def _natural_bucket(t: float) -> str:
        if t < train_cutoff:
            return "train"
        if t < validation_cutoff:
            return "validation"
        return "holdout"

    group_buckets: dict[str, set] = {}
    for r in rows:
        g = group_key(r)
        group_buckets.setdefault(g, set()).add(_natural_bucket(time_key(r)))

    boundary_spanning = {g for g, buckets in group_buckets.items() if len(buckets) > 1}
    group_count = len(group_buckets)
    boundary_spanning_groups = len(boundary_spanning)

    train, validation, holdout = [], [], []
    purged_rows = 0
    for r in rows:
        g = group_key(r)
        if g in boundary_spanning:
            purged_rows += 1
            continue
        bucket = _natural_bucket(time_key(r))
        (train if bucket == "train" else validation if bucket == "validation" else holdout).append(r)

    # Invariant 1: strict per-split time ordering.
    if train:
        assert max(time_key(r) for r in train) < train_cutoff
    if validation:
        assert min(time_key(r) for r in validation) >= train_cutoff
        assert max(time_key(r) for r in validation) < validation_cutoff
    if holdout:
        assert min(time_key(r) for r in holdout) >= validation_cutoff

    # Invariant 2: zero mint overlap between any two splits.
    train_groups = {group_key(r) for r in train}
    validation_groups = {group_key(r) for r in validation}
    holdout_groups = {group_key(r) for r in holdout}
    assert not (train_groups & validation_groups)
    assert not (train_groups & holdout_groups)
    assert not (validation_groups & holdout_groups)

    assert len(train) + len(validation) + len(holdout) + purged_rows == len(rows)

    return SplitResult(
        train=train, validation=validation, holdout=holdout,
        train_cutoff=train_cutoff, validation_cutoff=validation_cutoff,
        group_count=group_count, boundary_spanning_groups=boundary_spanning_groups,
        purged_rows=purged_rows,
    )
