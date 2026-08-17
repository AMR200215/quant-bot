"""
research/v8_candidate_registry.py — V8-FILTER-DERIVATION Phase 2
(FD8): bounded, pre-registered candidate search space.

Frozen and hashed BEFORE any holdout is ever touched (FD11). This is a
registry, not a ranking and not a selection -- research/v8_clean_cohort.py
already documents (P15-3/P15-9) that progress<0.70 is candidate-0's
scaffold value, not a derived one, and that P0-P3 are the pre-registered
progress policies Phase 2 must compare.

Every candidate here is built ONLY from features research/
v8_feature_registry.yaml marks allowed_for_entry=true (enforced at
evaluation time via research/v8_feature_enforcement.py, not just by
convention here) -- currently: event_id, alert_time, chain,
tg_message_text, channel_velocity_5m, progress_at_signal,
vsol_at_signal, venue_state_at_signal, realert_times (conditional, see
its own registry entry).

Complexity capped at 3 conditions per candidate (FD8's simplicity
prior). No candidate here was chosen by looking at outcomes -- each has
either (a) a direct research-bin rationale (the P0-P3 progress policies,
already named in v8_clean_cohort.py before this registry existed) or
(b) it's the baseline/scaffold itself, kept as the mandatory
control every real candidate must beat.

P2-2 CORRECTION (2026-08-17): BASELINE-0 (progress<0.70 AND
venue_state_at_signal==CURVE_ACTIVE) and the P2 progress policy from
v8_clean_cohort.py's PROGRESS_POLICY_CANDIDATES (also progress<0.70,
also combined with the same venue_state gate) were the exact same rule,
counted as two separate candidate identities. There is exactly ONE
evaluated identity for this rule now: BASELINE-0. The loop below
explicitly SKIPS generating a "V8-P2" entry (rather than silently
producing a differently-named duplicate) and records why, so the skip
is visible in code, not just in this comment.
"""

from __future__ import annotations

import hashlib
import json

from research.v8_clean_cohort import PROGRESS_POLICY_CANDIDATES

CANDIDATE_REGISTRY_VERSION = 1

# BASELINE-0: the running scaffold. Always included as the control every
# other candidate is compared against (FD27) -- never removed even if it
# performs poorly, since "how much better than doing nothing new" is
# itself part of the answer.
_BASELINE = {
    "candidate_id": "BASELINE-0",
    "human_rule": "progress_at_signal < 0.70 AND venue_state_at_signal == CURVE_ACTIVE",
    "conditions": [
        {"feature": "progress_at_signal", "op": "<", "value": 0.70},
        {"feature": "venue_state_at_signal", "op": "==", "value": "CURVE_ACTIVE"},
    ],
    "decision_delay_class": "T0+capture",
    "rationale": "current running V8 candidate-0 (memecoin/v8_paper.py) -- "
                 "the mandatory control, not a derived candidate",
    "required_features": ["progress_at_signal", "venue_state_at_signal"],
}

# P0-P3: the pre-registered progress policies from v8_clean_cohort.py,
# each combined with the same venue_state gate BASELINE-0 already uses
# (on-curve is a hard requirement of what V8 even means, not a tunable
# knob) -- reused here, not redefined, so the two files can't drift.
#
# P2-2: P2 is deliberately SKIPPED here -- progress<0.70 AND CURVE_ACTIVE
# is already BASELINE-0. Generating a second "V8-P2" entry for the exact
# same rule would double-count one candidate identity as two.
_PROGRESS_CANDIDATES = []
for _p in PROGRESS_POLICY_CANDIDATES:
    if _p["id"] == "P2":
        continue  # identical to BASELINE-0 -- see P2-2 correction above
    _conditions = [{"feature": "venue_state_at_signal", "op": "==", "value": "CURVE_ACTIVE"}]
    if _p["id"] != "P0":
        _threshold = float(_p["rule"].split("<")[1].strip())
        _conditions.insert(0, {"feature": "progress_at_signal", "op": "<", "value": _threshold})
    _PROGRESS_CANDIDATES.append({
        "candidate_id": f"V8-{_p['id']}",
        "human_rule": _p["rule"] + " AND venue_state_at_signal == CURVE_ACTIVE",
        "conditions": _conditions,
        "decision_delay_class": "T0+capture",
        "rationale": "pre-registered progress policy, research/v8_clean_cohort.py P15-3",
        "required_features": [c["feature"] for c in _conditions],
    })

# P2-3 AUDIT (2026-08-17, git SHA 60f404a): the original v1 registry
# included two "LOWVEL" extensions gated on channel_velocity_5m <= 5,
# with a code comment claiming 5 was "the channel's own median order of
# magnitude observed in Phase 1 data." That claim was never actually
# verified against data -- it was asserted. Verified now, purely from
# the feature-only population (chain=='solana' AND progress_data_ok==True,
# zero outcome/win-loss/holdout fields touched, n=1249 live rows):
#     min=0, p25=0, median=0, p75=1, mean=0.74, max=67
# The real median is 0, not ~5 -- nowhere near the claimed order of
# magnitude. The threshold fails the reproducibility audit. Per FD8/P2-3,
# a failed threshold is REMOVED, never replaced with a better-fitting
# number (that would be post-hoc tuning dressed up as a fix). No
# channel_velocity_5m-gated candidate exists in v1 as a result. A future
# velocity-based candidate is legitimate only if grounded in a freshly
# and correctly computed feature-only distribution, registered as its
# own dated audit, not a silent edit of this one.

CANDIDATES = [_BASELINE] + _PROGRESS_CANDIDATES

# Every candidate here has <= 3 conditions -- structurally guaranteed,
# not just asserted in prose (see test_v8_candidate_registry.py).
MAX_CONDITIONS_PER_CANDIDATE = 3


def registry_hash() -> str:
    """Deterministic hash of the frozen candidate set -- must be computed
    and recorded (in the experiment manifest) BEFORE holdout evaluation,
    per FD11. Changing any candidate after that point produces a
    different hash, which is the whole point: a silently-edited
    registry is externally detectable."""
    canonical = json.dumps(CANDIDATES, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


# P2-4 FREEZE (2026-08-17, git SHA 60f404a): this is the corrected v1
# entry-candidate registry -- exactly 4 candidates, each a single
# identity, no duplicates (P2-2), no unaudited feature thresholds
# (P2-3). This is the entire v1 set:
#   BASELINE-0  progress<0.70 AND CURVE_ACTIVE     (control)
#   V8-P0       no progress cutoff, CURVE_ACTIVE required
#   V8-P1       progress<0.50 AND CURVE_ACTIVE
#   V8-P3       progress<0.85 AND CURVE_ACTIVE
# No entry-candidate edits are permitted against this frozen set. Adding,
# removing, or changing a condition on any candidate (including
# reinstating a velocity-gated candidate after a future, separately
# dated audit) is "experiment v2" -- bump CANDIDATE_REGISTRY_VERSION,
# do not edit v1's frozen hash below in place.
CANDIDATE_REGISTRY_FROZEN_COUNT = 4
CANDIDATE_REGISTRY_FROZEN_SHA256 = "56990d2757a63930efadba001d838c2845359361209728d24a1c85af9ffe8251"
CANDIDATE_REGISTRY_FROZEN_AT = "2026-08-17T00:00:00+00:00"
CANDIDATE_REGISTRY_FROZEN_GIT_SHA = "60f404a"


def assert_registry_frozen() -> None:
    """Raises if the live registry has drifted from the frozen v1 hash --
    the enforceable half of the freeze, not just a comment. Call this
    anywhere Phase 2/3 code depends on v1 being exactly what was frozen."""
    if len(CANDIDATES) != CANDIDATE_REGISTRY_FROZEN_COUNT or registry_hash() != CANDIDATE_REGISTRY_FROZEN_SHA256:
        raise RuntimeError(
            "v8_candidate_registry.CANDIDATES has drifted from the frozen v1 set "
            f"(frozen hash={CANDIDATE_REGISTRY_FROZEN_SHA256}, live hash={registry_hash()}). "
            "This requires an explicit experiment v2, not an in-place edit."
        )
