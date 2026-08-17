"""
research/v8_exit_registry.py — V8-FILTER-DERIVATION Phase 2 (P2-5/FD19):
bounded, pre-registered exit + mid-trade candidate registry.

Frozen and hashed BEFORE any holdout is ever touched (FD11), same
discipline as research/v8_candidate_registry.py (the entry side).

AUDIT (2026-08-17, git SHA 60f404a) — prior research consulted before
adding anything new, per the explicit P2-5 instruction:

  research/analysis/replay_exits.py already defines three pre-registered,
  non-outcome-tuned exit specs (E0/E1/E2 below) and a tick-resolution
  replay engine (now research/v8_replay_engine.py, refactored out in
  P2-6). These are reused here, not redefined.

  research/analysis/path_stats.py's analyses A-H were run live against
  the full current path corpus (4262 files) to check for any
  ADDITIONAL, evidence-supported mid-trade rule before considering one:

    A (shakeout depth):        mostly INSUFFICIENT outside the 75-90%
        progress bucket (data is concentrated there, consistent with
        v8_clean_cohort.py's documented base rate). No new rule -- not
        enough cross-bucket evidence to condition an exit on.
    B (post-peak decay):       ret@1m/3m/5m computed as exactly 0.000
        across n=148 (75-90%) and n=52 (90%+) -- NOT a real retention
        finding. Root cause found live: price_usd contains extreme
        outlier spikes (e.g. price_usd=73.49 implying a $73.5 BILLION
        mcap -- physically impossible for a memecoin; confirmed across
        multiple path files: FN1pzrGdaRfpJeabHtPzDQpRhdMnqaHiE3Nywts7GNdR,
        AmqU7xrW8RMswHdcve7jE1dk7uD9ByudSfmBJrx8hBvH, and others). When
        peak_price is one of these corrupted spikes, every retention
        ratio collapses toward 0. DISQUALIFIED as evidence until a
        dedicated price-outlier-cleaning pass exists -- not done here
        (out of P2-5's scope; a data-cleaning task, not a rule-selection
        one).
    E (peak-mcap distribution):  same root cause as B -- "median
        $71,904,029,421" in the live run is the same corrupted-price
        artifact, not a real finding. DISQUALIFIED, same reason as B.
    F (conditional continuation): also mcap-zone-bucketed (price_usd *
        1e9), so at risk of the same contamination -- not used as
        evidence for a new rule until the same cleaning pass exists.
    C (pre-dump order flow):    n=1185 dump windows, Cohen's d=-0.378,
        directionally correct (sell pressure precedes dumps), verdict
        TRUE. This does NOT depend on the mcap conversion (raw
        sol_amount/side flow + relative price-drop detection), so it
        isn't disqualified by the same outlier bug. It IS a real,
        currently-unactionable finding: no leakage-safe, feature-only
        threshold for "how much net sell flow, over what window, should
        trigger an exit" has been derived. Inventing one now (e.g. "exit
        if net 10s SOL flow < -2.0") would repeat exactly the mistake
        P2-3 just caught and fixed on the entry side -- an assumed
        threshold presented as data-derived. NOT added to v1.
    D (graduation velocity), G/H (buyer velocity, sniper density): real,
        uncorrupted findings (vsol- and trader_pk-based, not price*1e9),
        but describe ENTRY-time conditions (how a token got here), not a
        mid-trade EXIT decision -- out of scope for an exit rule.

  CONCLUSION: NO_MIDTRADE_RULE_SUPPORTED for v1. This is an explicitly
  valid outcome per the P2-5 instruction, not a failure to look. E0/E1/E2
  (already-existing, non-outcome-tuned exit specs) are the entire v1
  exit registry. A future mid-trade candidate (e.g. an order-flow-based
  early exit) is legitimate only after (a) a dedicated price-outlier
  cleaning pass on logs/research_paths/, and (b) a P2-3-style
  feature-only threshold derivation for the flow signal -- both
  explicitly deferred, not silently dropped.
"""

from __future__ import annotations

import hashlib
import json

from research.analysis.replay_exits import _V7_SPEC, _ALT1_SPEC, _ALT2_SPEC

EXIT_REGISTRY_VERSION = 1

# E0/E1/E2: reused verbatim from replay_exits.py, not redefined -- so the
# two files can't drift. dict() copies so registry mutation here (there
# is none) can never affect the source module's module-level specs.
_E0_CURRENT = {
    "exit_id": "E0",
    "spec": dict(_V7_SPEC),
    "rationale": "current running v7 exit rule (memecoin/config.py social_alert "
                 "settings) -- the mandatory control every exit candidate is "
                 "compared against, not a derived candidate",
}
_E1_ALT1 = {
    "exit_id": "E1",
    "spec": dict(_ALT1_SPEC),
    "rationale": "existing pre-registered alternative (research/analysis/"
                 "replay_exits.py Spec B) -- earlier-arming, tighter trail, "
                 "shorter time stop",
}
_E2_ALT2 = {
    "exit_id": "E2",
    "spec": dict(_ALT2_SPEC),
    "rationale": "existing pre-registered alternative (research/analysis/"
                 "replay_exits.py Spec C) -- wider hard stop, longer time stop",
}

EXIT_CANDIDATES = [_E0_CURRENT, _E1_ALT1, _E2_ALT2]

# P2-5: no new mid-trade rule survived the audit above -- an explicit,
# valid terminal status, not an omission.
MIDTRADE_CANDIDATES: list = []
MIDTRADE_STATUS = "NO_MIDTRADE_RULE_SUPPORTED"
MIDTRADE_STATUS_REASON = (
    "path_stats.py analyses B/E/F are contaminated by confirmed price_usd "
    "outliers (up to ~$73.5B implied mcap); the one uncontaminated "
    "directional finding (C, pre-dump order flow, Cohen's d=-0.378, "
    "n=1185, verdict TRUE) has no leakage-safe derived threshold and "
    "adding one now would repeat the P2-3 velocity-threshold mistake."
)


def registry_hash() -> str:
    """Deterministic hash of the frozen exit + mid-trade candidate set.
    Mirrors research/v8_candidate_registry.py's registry_hash()."""
    canonical = json.dumps(
        {"exit_candidates": EXIT_CANDIDATES, "midtrade_candidates": MIDTRADE_CANDIDATES},
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


# P2-5 FREEZE (2026-08-17, git SHA 60f404a): the corrected v1 exit +
# mid-trade registry. Exactly 3 exit candidates (E0/E1/E2), 0 mid-trade
# candidates (NO_MIDTRADE_RULE_SUPPORTED). No edits permitted against
# this frozen set without an explicit "experiment v2" (mirrors
# research/v8_candidate_registry.py's CANDIDATE_REGISTRY_FROZEN_* /
# assert_registry_frozen pattern).
EXIT_REGISTRY_FROZEN_EXIT_COUNT = 3
EXIT_REGISTRY_FROZEN_MIDTRADE_COUNT = 0
# Hardcoded snapshot (NOT registry_hash() -- that would recompute from the
# live set every time and could never detect a drift). Computed once at
# freeze time; any future edit to EXIT_CANDIDATES/MIDTRADE_CANDIDATES will
# make registry_hash() diverge from this literal, which is the point.
EXIT_REGISTRY_FROZEN_SHA256 = "bb64c2ec6b6efba39f6a1c6c309add08fbe452212a28960175397f6a64c72b40"
EXIT_REGISTRY_FROZEN_AT = "2026-08-17T00:00:00+00:00"
EXIT_REGISTRY_FROZEN_GIT_SHA = "60f404a"


def assert_registry_frozen() -> None:
    """Raises if the live registry has drifted from the frozen v1 hash."""
    if (len(EXIT_CANDIDATES) != EXIT_REGISTRY_FROZEN_EXIT_COUNT
            or len(MIDTRADE_CANDIDATES) != EXIT_REGISTRY_FROZEN_MIDTRADE_COUNT
            or registry_hash() != EXIT_REGISTRY_FROZEN_SHA256):
        raise RuntimeError(
            "v8_exit_registry has drifted from the frozen v1 set "
            f"(frozen hash={EXIT_REGISTRY_FROZEN_SHA256}, live hash={registry_hash()}). "
            "This requires an explicit experiment v2, not an in-place edit."
        )
