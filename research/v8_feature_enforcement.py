"""
research/v8_feature_enforcement.py — V8-FILTER-DERIVATION Phase 2
(FD3 code-enforcement): the feature registry (research/v8_feature_
registry.yaml) is data; this module is the guard that actually makes it
load-bearing.

Without this, "allowed_for_entry: false" in the YAML is a document a
future candidate definition could just ignore. With it, any code
building a candidate's feature set MUST go through
assert_features_allowed() (or check_features_allowed() for a non-raising
variant) before that candidate can be evaluated at all -- the exact
FD3 requirement ("Add deterministic tests that fail if forbidden
future/outcome fields are used as strategy inputs") extended from a
test-only guarantee to a runtime one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

_REPO_ROOT = Path(__file__).parent.parent
_REGISTRY_PATH = _REPO_ROOT / "research" / "v8_feature_registry.yaml"

_STAGE_KEY = {
    "entry": "allowed_for_entry",
    "midtrade": "allowed_for_midtrade",
    "exit": "allowed_for_exit",
}

_cache: Optional[dict] = None


def _load_registry() -> dict:
    global _cache
    if _cache is None:
        _cache = yaml.safe_load(_REGISTRY_PATH.read_text())
    return _cache


def reload_registry() -> None:
    """Force a fresh read from disk (tests only, normally)."""
    global _cache
    _cache = None


def _feature_by_name(name: str) -> Optional[dict]:
    for f in _load_registry().get("features", []):
        if f["name"] == name:
            return f
    return None


class FeatureNotRegisteredError(ValueError):
    """A feature name has no entry in v8_feature_registry.yaml at all --
    fail closed, never assume a new/unregistered field is safe."""


class FeatureNotAllowedError(ValueError):
    """A feature is registered but explicitly not allowed for the
    requested stage (entry/midtrade/exit) -- e.g. a POST-TRADE outcome
    field, or a leakage-disqualified field like smart_money_hit."""


def check_features_allowed(feature_names: list[str], stage: str) -> list[str]:
    """Returns the list of violation messages (empty if all clear).
    Non-raising -- use assert_features_allowed() for the raising form."""
    if stage not in _STAGE_KEY:
        raise ValueError(f"unknown stage {stage!r}, must be one of {list(_STAGE_KEY)}")
    key = _STAGE_KEY[stage]
    violations = []
    for name in feature_names:
        f = _feature_by_name(name)
        if f is None:
            violations.append(f"{name!r} has no entry in v8_feature_registry.yaml "
                               f"(fail closed -- unregistered features are never assumed safe)")
            continue
        if not f.get(key, False):
            violations.append(f"{name!r} is not {key}=true "
                               f"(availability_class={f.get('availability_class')!r}, "
                               f"leakage_risk={str(f.get('leakage_risk'))[:120]!r})")
    return violations


def assert_features_allowed(feature_names: list[str], stage: str) -> None:
    """Raises FeatureNotRegisteredError/FeatureNotAllowedError on the
    first violation category found; call this at candidate-definition
    time, before any evaluation, not just in a test."""
    violations = check_features_allowed(feature_names, stage)
    if not violations:
        return
    unregistered = [v for v in violations if "no entry in" in v]
    if unregistered:
        raise FeatureNotRegisteredError("; ".join(unregistered))
    raise FeatureNotAllowedError("; ".join(violations))
