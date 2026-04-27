"""Apply calibration correction to raw model posteriors.

The logistic regression is systematically overconfident in the 20-50% range
(mean absolute calibration error ~5.1%).  This module corrects that bias using
the empirical bucket data in data/calibration.json, interpolating linearly
between bucket centres.

Usage:
    from app.calibration_correction import apply_calibration
    calibrated = apply_calibration(raw_posterior)
"""

from __future__ import annotations

import json
from pathlib import Path

_CALIBRATION_PATH = Path("data/calibration.json")
_calibration_points: list[tuple[float, float]] | None = None


def _load_points() -> list[tuple[float, float]]:
    """Load (predicted, actual) pairs from calibration.json, cached after first read."""
    global _calibration_points
    if _calibration_points is not None:
        return _calibration_points

    if not _CALIBRATION_PATH.exists():
        _calibration_points = []
        return _calibration_points

    with _CALIBRATION_PATH.open() as f:
        data = json.load(f)

    points = sorted((b["predicted"], b["actual"]) for b in data.get("buckets", []))
    _calibration_points = points
    return _calibration_points


def apply_calibration(raw: float) -> float:
    """Return a calibration-corrected probability for a raw model posterior.

    Uses linear interpolation between empirical bucket centres.
    Falls back to the raw value when calibration data is unavailable.
    """
    points = _load_points()
    if not points:
        return raw

    if raw <= points[0][0]:
        return round(max(0.02, min(0.98, points[0][1])), 4)
    if raw >= points[-1][0]:
        return round(max(0.02, min(0.98, points[-1][1])), 4)

    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        if x0 <= raw <= x1:
            t = (raw - x0) / (x1 - x0)
            calibrated = y0 + t * (y1 - y0)
            return round(max(0.02, min(0.98, calibrated)), 4)

    return raw
