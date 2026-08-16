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
_PROGRESS_CANDIDATES = []
for _p in PROGRESS_POLICY_CANDIDATES:
    _conditions = [{"feature": "venue_state_at_signal", "op": "==", "value": "CURVE_ACTIVE"}]
    if _p["id"] != "P0":
        _threshold = float(_p["rule"].split("<")[1].strip())
        _conditions.insert(0, {"feature": "progress_at_signal", "op": "<", "value": _threshold})
    _PROGRESS_CANDIDATES.append({
        "candidate_id": f"V8-{_p['id']}",
        "human_rule": _p["rule"] + " AND venue_state_at_signal == CURVE_ACTIVE",
        "conditions": _conditions,
        "decision_delay_class": "T0+capture",
        "rationale": "pre-registered progress policy, research/v8_clean_cohort.py "
                     "P15-3" + (" (current candidate-0's own threshold)" if _p["id"] == "P2" else ""),
        "required_features": [c["feature"] for c in _conditions],
    })

# Two bounded 3-condition extensions -- channel_velocity_5m is the only
# other T0 feature with any plausible entry-signal rationale (crowding/
# hype proxy, named explicitly in the original FD2 field list); capped
# at exactly these two thresholds (below/above the channel's own median
# order of magnitude observed in Phase 1 data) rather than a grid, per
# FD8's explicit "do not create arbitrary 0.01-step grid searches" rule.
_VELOCITY_EXTENSIONS = [
    {
        "candidate_id": "V8-P2-LOWVEL",
        "human_rule": "progress_at_signal < 0.70 AND venue_state_at_signal == CURVE_ACTIVE "
                       "AND channel_velocity_5m <= 5",
        "conditions": [
            {"feature": "progress_at_signal", "op": "<", "value": 0.70},
            {"feature": "venue_state_at_signal", "op": "==", "value": "CURVE_ACTIVE"},
            {"feature": "channel_velocity_5m", "op": "<=", "value": 5},
        ],
        "decision_delay_class": "T0+capture",
        "rationale": "candidate-0 restricted to low-crowding alerts (channel not "
                     "actively bursting) -- bounded extension, not a grid search",
        "required_features": ["progress_at_signal", "venue_state_at_signal", "channel_velocity_5m"],
    },
    {
        "candidate_id": "V8-P3-LOWVEL",
        "human_rule": "progress_at_signal < 0.85 AND venue_state_at_signal == CURVE_ACTIVE "
                       "AND channel_velocity_5m <= 5",
        "conditions": [
            {"feature": "progress_at_signal", "op": "<", "value": 0.85},
            {"feature": "venue_state_at_signal", "op": "==", "value": "CURVE_ACTIVE"},
            {"feature": "channel_velocity_5m", "op": "<=", "value": 5},
        ],
        "decision_delay_class": "T0+capture",
        "rationale": "wider progress band (P3) restricted to low-crowding alerts",
        "required_features": ["progress_at_signal", "venue_state_at_signal", "channel_velocity_5m"],
    },
]

CANDIDATES = [_BASELINE] + _PROGRESS_CANDIDATES + _VELOCITY_EXTENSIONS

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
