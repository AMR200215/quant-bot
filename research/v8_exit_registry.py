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

EXPERIMENT V2 (2026-08-26) -- E3 added, an explicit registry change per
this module's own "requires an explicit experiment v2, not an in-place
edit" rule (see assert_registry_frozen below).

  Item (a) from the CONCLUSION above -- the price-outlier-cleaning pass
  -- now exists: research/analysis/path_stats.py --valid-only excludes
  any path research/v8_path_integrity.py doesn't classify VALID before
  any analysis runs (out of 4221 total path files, 3847 were excluded --
  the pre-2026-08-19/pre-funding corpus).

  On the clean 374-token corpus, research/analysis/path_stats.py's
  _shakeout_depth_for_target(rows, 30) and _time_to_target_minutes(rows,
  30) were run against every token that ever reached +30% (n=106),
  using the SAME +30% threshold E0/E1/E2 already use as
  time_stop_min_gain -- not a new number invented for this derivation:

    shakeout depth to +30%:  p50=3.82%  p75=21.07%  p90=81.74%  max=99.99%
    time to +30% (minutes):  p50=0.06   p75=1.02    p90=7.29    max=49.87

  The depth p90 (81.74%) is NOT used -- it is wider than E2's existing
  -50% hard stop and almost certainly dominated by a handful of
  extremely thin-liquidity tokens where one trade can gap price 80%+
  (the same failure mode already flagged for the <$10k mcap band in
  Analysis F). Freezing a hard_stop from it would be picking a
  plausible-sounding number out of outlier noise, not deriving one --
  exactly what this registry's own discipline exists to prevent. No
  data-derived hard_stop is added in this experiment; E3 keeps E0's
  hard_stop (-35%) and trail_tiers verbatim, isolating the one
  parameter this derivation actually supports.

  The time-to-target p90 (7.29 min, rounded to 7) does NOT show the
  same outlier-domination problem (p50/p75/p90 are all consistent with
  a fast, front-loaded distribution -- most winners prove themselves in
  under a minute), so it is used: E3's time_stop_min = 7 (vs E0's 90,
  E1's 45, E2's 120) -- give a position roughly the time it took 90% of
  real, clean, funded-era winners to reach +30%, not an inherited or
  guessed number. This is a large, deliberate departure from the
  existing specs' timeouts -- flagged here explicitly, not smuggled in
  as an incremental tweak.

  n=106 (reaching +30%) and the underlying n=374 clean corpus are both
  still small (this is ~1 week of funded-era collection). E3's
  time_stop_min is a first derivation, not a mature one -- it will get
  evaluated against E0/E1/E2 by the same readiness/replay pipeline as
  everything else, on train/validation only, holdout untouched.
"""

from __future__ import annotations

import hashlib
import json

from research.analysis.replay_exits import _V7_SPEC, _ALT1_SPEC, _ALT2_SPEC

EXIT_REGISTRY_VERSION = 2

# E3: NEW in experiment v2 (2026-08-26), not reused from replay_exits.py
# like E0/E1/E2 -- it is genuinely derived here, so (unlike E0/E1/E2) it
# is defined in this file, not smuggled into replay_exits.py as if it
# already existed there. Everything except time_stop_min is copied
# verbatim from _V7_SPEC (E0) -- see the EXPERIMENT V2 docstring above
# for exactly why only the timeout, not the hard stop, was changed.
_E3_SPEC = dict(_V7_SPEC)
_E3_SPEC["name"] = "e3_data_derived_timeout"
_E3_SPEC["time_stop_min"] = 7   # p90 time-to-+30% on the clean funded-era corpus (7.29min), rounded

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
_E3_DATA_DERIVED = {
    "exit_id": "E3",
    "spec": dict(_E3_SPEC),
    "rationale": "experiment v2 (2026-08-26) -- time_stop_min derived from the p90 "
                 "time-to-+30%-gain on the clean, integrity-filtered, funded-era path "
                 "corpus (n=106 tokens reaching target); hard_stop/trail_tiers kept "
                 "identical to E0 because the equivalent depth-based derivation was "
                 "dominated by thin-liquidity outliers and not used -- see this "
                 "module's EXPERIMENT V2 docstring for the full derivation and numbers",
}

EXIT_CANDIDATES = [_E0_CURRENT, _E1_ALT1, _E2_ALT2, _E3_DATA_DERIVED]

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


# P2-5 FREEZE v1 (2026-08-17, git SHA 60f404a): 3 exit candidates
# (E0/E1/E2), 0 mid-trade candidates. Superseded by EXPERIMENT V2 below
# -- kept here only as the historical record of what v1 was.
_V1_FROZEN_EXIT_COUNT = 3
_V1_FROZEN_SHA256 = "bb64c2ec6b6efba39f6a1c6c309add08fbe452212a28960175397f6a64c72b40"
_V1_FROZEN_AT = "2026-08-17T00:00:00+00:00"
_V1_FROZEN_GIT_SHA = "60f404a"

# EXPERIMENT V2 FREEZE (2026-08-26): E3 added (see the EXPERIMENT V2
# docstring above for the full derivation). No edits permitted against
# this frozen set without a further explicit "experiment v3" (mirrors
# research/v8_candidate_registry.py's CANDIDATE_REGISTRY_FROZEN_* /
# assert_registry_frozen pattern).
EXIT_REGISTRY_FROZEN_EXIT_COUNT = 4
EXIT_REGISTRY_FROZEN_MIDTRADE_COUNT = 0
# Hardcoded snapshot (NOT registry_hash() -- that would recompute from the
# live set every time and could never detect a drift). Computed once at
# freeze time; any future edit to EXIT_CANDIDATES/MIDTRADE_CANDIDATES will
# make registry_hash() diverge from this literal, which is the point.
EXIT_REGISTRY_FROZEN_SHA256 = "fe369e04b4ce068c188d2ce5772383122839a71cab69b5760ebd6beceadd163a"
EXIT_REGISTRY_FROZEN_AT = "2026-08-26T00:00:00+00:00"
EXIT_REGISTRY_FROZEN_GIT_SHA = "333dbfb"


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
