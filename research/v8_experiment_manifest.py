"""
research/v8_experiment_manifest.py — V8-FILTER-DERIVATION Phase 2
(FD10/FD29/FD37): the reproducibility record for one filter-derivation
experiment run.

Every real run (once SELECTION_DATA_READY, see research/v8_clean_cohort.py
P15-9) must produce one of these, frozen BEFORE holdout evaluation
starts (FD11) and written alongside the run's output artifacts
(FD29 -- logs/research_reports/v8_filter_selection/<run_id>/manifest.json).
This module only defines the shape and a builder; it does not run an
experiment itself.
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

    # Data provenance -- must match research/v8_clean_cohort.py at run time,
    # not be silently re-derived here.
    clean_cohort_version: int = 0
    feature_registry_version: int = 0
    candidate_registry_hash: str = ""
    exit_registry_hash: Optional[str] = None          # None until FD19's registry exists
    execution_cost_model_version: Optional[str] = None  # None until FD20's model exists
    path_schema_version: Optional[int] = None
    smart_money_version: Optional[int] = None          # must stay None -- FD6/P15-8, never reused

    # Split boundaries (research/v8_split.py output), frozen once computed.
    data_cutoff: Optional[str] = None
    train_start: Optional[str] = None
    train_cutoff: Optional[str] = None
    validation_cutoff: Optional[str] = None
    holdout_evaluated: bool = False   # FD11 -- must stay False until the run explicitly opens holdout

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
    from research.v8_candidate_registry import registry_hash

    m = ExperimentManifest()
    m.clean_cohort_version = V8_CLEAN_COHORT_VERSION
    m.candidate_registry_hash = registry_hash()
    try:
        import yaml
        reg = yaml.safe_load((Path(__file__).parent / "v8_feature_registry.yaml").read_text())
        m.feature_registry_version = reg.get("schema_version", 0)
    except Exception:
        pass
    return m
