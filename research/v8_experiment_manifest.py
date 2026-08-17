"""
research/v8_experiment_manifest.py — V8-FILTER-DERIVATION Phase 2
(FD10/FD29/FD37, hardened P2-12): the reproducibility record for one
filter-derivation experiment run, PLUS the enforceable holdout lock.

Every real run (once SELECTION_DATA_READY, see research/v8_clean_cohort.py
P15-9) must produce one of these, frozen BEFORE holdout evaluation
starts (FD11) and written alongside the run's output artifacts
(FD29 -- logs/research_reports/v8_filter_selection/<run_id>/manifest.json).
This module defines the shape, a builder, AND (P2-12) the software
mechanism that makes "holdout stays locked until Phase 3" a runtime
guarantee rather than a convention:

  1. holdout_evaluated defaults to False and CANNOT be set True by
     assigning the attribute in normal Phase-2 code with any expectation
     of it being honored -- unlock_holdout_for_phase3() is the only
     sanctioned setter, and it requires typing out the literal
     PHASE3_HOLDOUT_UNLOCK_SENTINEL string. This doesn't make direct
     attribute assignment impossible (Python has no private fields) but
     it does make "I unlocked holdout by accident inside Phase 2 code"
     structurally implausible -- nobody writes
     `manifest.holdout_evaluated = True` by accident when the sanctioned,
     documented, tested path requires a 34-character sentinel argument.
  2. assert_holdout_not_evaluated(manifest) is the guard Phase-2
     reporting code calls before printing/returning any results table --
     raises HoldoutStillLockedViolation if holdout_evaluated is True on
     a manifest a Phase-2 code path is holding.
  3. check_v1_not_invalidated() scans every manifest ever written to
     disk; for any with holdout_evaluated=True, it re-hashes the LIVE
     candidate/exit registries and compares against what that manifest
     recorded. A mismatch means the registry changed after holdout was
     evaluated -- raises ExperimentV1InvalidatedError. This is the
     concrete mechanism behind "changing either registry after first
     holdout evaluation must invalidate experiment v1."
"""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

EXPERIMENT_MANIFEST_VERSION = 1

PHASE3_HOLDOUT_UNLOCK_SENTINEL = "PHASE_3_EXPLICIT_HOLDOUT_UNLOCK"


def _git_sha() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            timeout=5, cwd=Path(__file__).parent.parent)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


@dataclass
class ExperimentManifest:
    manifest_version: int = EXPERIMENT_MANIFEST_VERSION
    run_id: str = field(default_factory=lambda: f"run-{uuid.uuid4().hex[:12]}")
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_sha: str = field(default_factory=_git_sha)

    # Data provenance -- must match the live registries at run time, not
    # be silently re-derived or hand-typed here.
    clean_cohort_version: int = 0
    feature_registry_version: int = 0
    feature_registry_hash: str = ""      # P2-12: hash, not just version -- catches non-version-bumped edits
    candidate_registry_hash: str = ""
    exit_registry_hash: Optional[str] = None
    execution_cost_model_version: Optional[int] = None
    path_schema_version: Optional[int] = None
    smart_money_version: Optional[int] = None          # must stay None -- FD6/P15-8, never reused

    # Split boundaries (research/v8_split.py output), frozen once computed.
    data_cutoff: Optional[str] = None
    train_start: Optional[str] = None
    train_cutoff: Optional[str] = None
    validation_cutoff: Optional[str] = None
    holdout_evaluated: bool = False   # FD11/P2-12 -- see module docstring for the enforcement mechanism

    # FD29 output location.
    output_dir: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def write(self, repo_root: Optional[Path] = None) -> Path:
        root = repo_root or Path(__file__).parent.parent
        out_dir = root / "logs" / "research_reports" / "v8_filter_selection" / self.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "manifest.json"
        out_path.write_text(json.dumps(self.to_dict(), indent=2))
        return out_path


def build_manifest_from_current_state() -> ExperimentManifest:
    """Populates the provenance fields from the actual current registries
    -- never hand-typed, so the manifest can't silently drift from what
    was really used."""
    from research.v8_clean_cohort import V8_CLEAN_COHORT_VERSION
    from research.v8_candidate_registry import registry_hash as candidate_registry_hash
    from research.v8_exit_registry import registry_hash as exit_registry_hash
    from research.v8_execution_cost_model import EXECUTION_COST_MODEL_VERSION
    from research.v8_feature_enforcement import feature_registry_hash

    m = ExperimentManifest()
    m.clean_cohort_version = V8_CLEAN_COHORT_VERSION
    m.candidate_registry_hash = candidate_registry_hash()
    m.exit_registry_hash = exit_registry_hash()
    m.execution_cost_model_version = EXECUTION_COST_MODEL_VERSION
    m.feature_registry_hash = feature_registry_hash()
    try:
        import yaml
        reg = yaml.safe_load((Path(__file__).parent / "v8_feature_registry.yaml").read_text())
        m.feature_registry_version = reg.get("schema_version", 0)
    except Exception:
        pass
    return m


# ── P2-12: the enforceable holdout lock ─────────────────────────────────

class HoldoutStillLockedViolation(RuntimeError):
    """Raised when Phase-2 reporting code tries to proceed on a manifest
    whose holdout has (somehow) already been marked evaluated -- Phase-2
    code must never print/return holdout performance."""


class ExperimentV1InvalidatedError(RuntimeError):
    """Raised when a registry has changed after a manifest recorded
    holdout_evaluated=True against the OLD registry hash -- experiment v1
    is no longer a valid, reproducible record."""


def assert_holdout_not_evaluated(manifest: ExperimentManifest) -> None:
    """Call this before any Phase-2 code path prints/returns a results
    table. Phase-2 must never observe/report holdout performance."""
    if manifest.holdout_evaluated:
        raise HoldoutStillLockedViolation(
            f"manifest {manifest.run_id} has holdout_evaluated=True -- Phase-2 code "
            "must never print or return holdout results. This manifest belongs to "
            "Phase 3, not Phase 2."
        )


def unlock_holdout_for_phase3(manifest: ExperimentManifest, confirmation: str) -> None:
    """The ONLY sanctioned way to flip holdout_evaluated to True. Requires
    the literal PHASE3_HOLDOUT_UNLOCK_SENTINEL string -- a real Phase-3
    entrypoint types this out deliberately; Phase-2 code has no reason to
    ever hold that string. Raises on a wrong/missing confirmation or if
    this manifest's holdout was already unlocked (re-unlocking the same
    manifest is not a supported operation -- start a new run/manifest)."""
    if confirmation != PHASE3_HOLDOUT_UNLOCK_SENTINEL:
        raise ValueError(
            "unlock_holdout_for_phase3 requires the exact PHASE3_HOLDOUT_UNLOCK_SENTINEL "
            "as `confirmation` -- refusing to unlock holdout without it."
        )
    if manifest.holdout_evaluated:
        raise RuntimeError(
            f"manifest {manifest.run_id} already has holdout_evaluated=True -- "
            "start a new manifest/run rather than re-unlocking this one."
        )
    manifest.holdout_evaluated = True


def _iter_written_manifests(repo_root: Optional[Path] = None) -> list[dict]:
    root = repo_root or Path(__file__).parent.parent
    base = root / "logs" / "research_reports" / "v8_filter_selection"
    if not base.exists():
        return []
    out = []
    for p in base.glob("*/manifest.json"):
        try:
            out.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def check_v1_not_invalidated(repo_root: Optional[Path] = None) -> None:
    """Scans every manifest ever written. For any with
    holdout_evaluated=True, re-hashes the LIVE candidate/exit registries
    and compares against what that manifest recorded at holdout-unlock
    time. A mismatch means the registry changed after holdout was
    evaluated against it -- raises ExperimentV1InvalidatedError naming
    the run_id and which registry diverged."""
    from research.v8_candidate_registry import registry_hash as live_candidate_hash
    from research.v8_exit_registry import registry_hash as live_exit_hash

    for m in _iter_written_manifests(repo_root):
        if not m.get("holdout_evaluated"):
            continue
        run_id = m.get("run_id", "<unknown>")
        recorded_candidate_hash = m.get("candidate_registry_hash")
        recorded_exit_hash = m.get("exit_registry_hash")
        if recorded_candidate_hash and recorded_candidate_hash != live_candidate_hash():
            raise ExperimentV1InvalidatedError(
                f"run {run_id}: candidate registry changed after holdout was evaluated "
                f"(recorded={recorded_candidate_hash}, live={live_candidate_hash()}) -- "
                "experiment v1 is invalidated."
            )
        if recorded_exit_hash and recorded_exit_hash != live_exit_hash():
            raise ExperimentV1InvalidatedError(
                f"run {run_id}: exit registry changed after holdout was evaluated "
                f"(recorded={recorded_exit_hash}, live={live_exit_hash()}) -- "
                "experiment v1 is invalidated."
            )
