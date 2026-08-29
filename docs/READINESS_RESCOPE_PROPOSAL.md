# Readiness Rescope Proposal (YD-BATCH item YD2)

**Status: PROPOSAL ONLY. No code has been changed to implement anything
in this document.** Every number below is real and live-verified as of
2026-08-29 (sources cited inline). This document lays out a design
change for the user to read and decide on. If approved, implementation
is a separate, subsequent, explicitly-authorized task.

## The problem this proposal addresses

`SELECTION_DATA_READY` has been stuck at `False` for the whole life of
this readiness system, and the reported blocker has consistently been
**path coverage** (`admitted_path_yield_pct` / `path_coverage_pct`,
floor 50%) — currently ~16% for V8-P0 and trending down, not up, as the
sample grows (docs/RECEIPTS.md, 2026-08-25 → 2026-08-29: 51.22% → 15.93%
→ 15.82%).

But path coverage was never actually needed to judge whether an ENTRY
candidate is profitable. It's needed to build/tune an EXIT rule with
tick-level precision (stop-loss timing, trailing-stop behavior). A
completely separate, already-existing data stream — the outcome poller
(`research/outcome_poller.py`) — polls price at fixed T+1m/3m/5m/10m/20m
offsets via Helius curve-account reads / DexScreener, **independent of
whether the token ever trades on PumpPortal at all**. That's why its
coverage is dramatically higher (see below): it doesn't depend on the
same thin-liquidity bottleneck path coverage does.

**One gate has been measuring the wrong thing for the wrong purpose.**
This proposal splits it into two.

## Real numbers, with provenance

### Path coverage vs poll-outcome coverage on the SAME population

Computed live, 2026-08-29, funded-era admitted population (era boundary
= `research/v8_collection_yield.py`'s `trustworthy_collection_era_start`,
currently 2026-08-22T23:48:17Z), n=427:

| | n | coverage |
|---|---|---|
| `admitted_with_valid_usable_path` (path-based) | 427 | **~16%** |
| `outcome_complete=True` (poll-based) | 426 | **99.8%** |
| `pct_change_peak IS NOT NULL` (poll-based, usable) | 414 | **97.0%** |

Same 427 tokens. Two different data sources. One is bottlenecked by
whether the token actually traded on PumpPortal; the other polls price
directly regardless.

### Per-candidate poll-outcome coverage, train+validation ONLY (holdout untouched)

Computed live, 2026-08-29, via the same `grouped_chronological_split`
every other readiness computation in this project uses, reading only
`result.train + result.validation` — `result.holdout` is never touched,
matching the same discipline `research/v8_forward_readiness_report.py`'s
`_compute_diagnostics_feasibility` already enforces:

| Candidate | train+val n | `pct_change_peak` present |
|---|---|---|
| BASELINE-0 | 15 | 12 (80.0%) |
| **V8-P0** | **994** | **768 (77.3%)** |
| V8-P1 | 11 | 8 (72.7%) |
| **V8-P3** | **405** | **305 (75.3%)** |

For comparison, path-based `representative_path_n` for V8-P0 was 65 (as
of 2026-08-29's readiness report) — poll-based usable-outcome n is
**768**, nearly 12x larger, on the exact same underlying candidate.

---

## (a) SELECTION (entry-EV) readiness — remove path coverage from this gate

**Proposed new gate**, replacing the current `path_data_ready` component
of `full_eval_ready` for the SELECTION question only:

- `historical_entry_n >= MIN_ENTRY_N` (100) — **unchanged**, already cited
  (`research/analysis/path_stats.py --min-n` default).
- `unique_mints >= MIN_UNIQUE_MINTS` (50) — **unchanged**, already cited.
- `unique_days >= MIN_UNIQUE_DAYS` (14) — **unchanged**, already cited.
- **NEW**: poll-outcome coverage — `pct_change_peak`-present count (train+
  validation only) `>= 100` AND coverage `>= 50%` of the same
  train+validation population. Deliberately reuses the EXACT SAME
  absolute-n and percentage floors already established for path coverage
  (`MIN_PATH_N=100`, `MIN_PATH_COVERAGE_PCT=50%`,
  `research/v8_readiness_engine.py`) — no new number invented, the floor
  is carried over unchanged, only the *numerator/denominator it's
  measured on* changes.
- Path coverage is **removed** from this gate entirely — paths are never
  read to compute entry EV (`pct_change_peak` is populated independently
  of any path file).

**Under this proposed gate, computed on real current data (not yet
adopted, not yet live):** V8-P0 — 994 train+val n (>=100 ✓), 768
poll-outcome-present (>=100 ✓, 77.3% >=50% ✓). V8-P3 — 405 train+val n
(✓), 305 poll-outcome-present (✓, 75.3% ✓). **Both would clear this
specific gate if adopted**, pending whatever other gates
(`full_entry_rule_ready`, split-bucket-size, diagnostics feasibility)
are already independently satisfied or not — this document does not
claim `SELECTION_DATA_READY` is `True`; that determination only happens
if/when this proposal is implemented and the full gate is re-run.

## (b) EXIT-derivation readiness — path n floor + representativeness + imputation bounds

Exit-rule tuning (stop-loss depth, trailing-stop timing) genuinely needs
tick-level path data — poll-based T+1m/3m/5m snapshots are too coarse to
back out "what price would a -35% hard stop have triggered at, and when."
This gate is **not proposed to change its core requirement**
(`representative_path_n >= 100`) — but two things are added:

### Representativeness check (path-havers vs full population)

Reusing `research/v8_path_predictability.py`'s (YD1) live output,
2026-08-29, n=411 funded-era admitted tokens — this IS the
representativeness divergence, already measured:

| Feature | Lower bucket | Higher bucket | Usable-path rate divergence |
|---|---|---|---|
| `progress_at_signal` | 75-90% (n=293) | 90%+ (n=112) | 17.7% vs 9.8% |
| `vsol_at_signal` | 60-100 (n=212) | 100+ (n=194) | 21.7% vs 9.3% |
| `channel_velocity_5m` | 0 (n=292) | 1-2 (n=117) | 19.9% vs 6.0% |

**Honest disclosure, not hidden:** path-havers are NOT a random sample.
Lower-progress, lower-vsol, and lower-channel-velocity tokens are
meaningfully more likely to end up with usable path data. A logistic
model on these features gets AUC=0.763 (n=411, 70/30 train/test split) —
a real, moderate-to-good predictive signal, meaning any EV number
computed only from path-havers is measuring a population skewed toward
these characteristics, not the full admitted population. This is the
exact survivorship-bias risk already discussed with the user directly.
**No exit-EV table may be presented without disclosing this divergence
table alongside it.**

### Pessimistic-imputation sensitivity column

Every exit-rule EV table must carry an additional column bounding the
no-path mass (the ~84% without a usable tick-level path), not silently
drop it. Two bounds, both cited from data/parameters that already exist
— no new number invented for this proposal:

1. **Poll-outcome-with-slippage-haircut bound**: use the coarser
   `pct_change_peak` (poll-based, ~75-97% coverage depending on
   population) for the no-path tokens, haircut by the round-trip cost
   already measured from real, live execution-proxy observations:
   **-1.99%** (`logs/research_execution_proxy/execution_proxy_log.jsonl`,
   real observations, `round_trip_pct` field; matches
   `research/v8_execution_proxy.py`'s `PUMPFUN_TRADING_FEE_RATE=0.01`
   model, ~1% fee each way).
2. **Stop-loss-floor bound**: assume every no-path token hit the exit
   spec's own `hard_stop` — E0/E1/E3: -35%, E2: -50%
   (`research/v8_exit_registry.py`, frozen values, no new number).

Reporting both bounds alongside the path-haver-only EV number turns "we
don't know what happened to 84% of tokens" from a silently-ignored gap
into an honestly-bounded range.

## (c) The 50%-coverage floor — retired only where it provably doesn't consume paths

- **Retired** for SELECTION (entry-EV) readiness — replaced by the
  poll-outcome coverage floor in (a), same numeric bar, different
  denominator, because that gate provably never reads a path file.
- **Retained, unchanged, as-is** for EXIT-derivation readiness (b) — any
  path-consuming statistic (exit-rule EV, stop-loss calibration,
  trailing-stop timing) still needs `representative_path_n >= 100` AND
  is now required to carry the imputation sensitivity bounds above. A
  path-consuming number presented WITHOUT its sensitivity bound remains
  blocked, same as today.

---

## Context: why path/proxy coverage has been unstable (YD3 finding)

Recomputing execution-proxy coverage on the funded-era-only denominator
and reporting the day-by-day trajectory (`research/v8_execution_proxy_
trend.py`) shows real coverage held at 97-98% from 2026-08-23 through
08-26, then collapsed: 72.76% (08-27) → 47.58% (08-28) → 45.50% (08-29).
`cumulative_observed_n` has been frozen at exactly 187 since 2026-08-27
15:19 UTC while `cumulative_admitted_n` kept climbing (257→411).

**Root-caused live, 2026-08-29:** direct WebSocket test against
PumpPortal returned the exact same rejection as before the account was
originally funded — `"Minimum balance not met for PumpSwap websocket
data."` The funded PumpPortal account ran out of balance again from
real metered usage sometime around 2026-08-27 15:19-15:30 UTC. This is
**not a code defect** — same external, account-funding root cause as
the original incident, not something this proposal or any code change
fixes. Flagged to the user directly; account needs another top-up. This
also means YD1's 15.82% overall usable-path-rate figure blends a
well-funded period (Aug 22-27) with a dead period (Aug 27-onward) — a
real caveat on that number, not a flaw in the analysis method.

## What this proposal does NOT do

- Does not open holdout. Every number above is train+validation-only or
  a population-level (non-outcome) count.
- Does not lower any absolute floor (`MIN_ENTRY_N=100`,
  `MIN_PATH_N=100`, the 50% coverage bar) — only changes which
  denominator the coverage bar is measured against, and only for the
  gate that provably doesn't need path data.
- Does not touch `research/v8_candidate_registry.py` or
  `research/v8_exit_registry.py` — no candidate is added, removed, or
  re-scored.
- Does not implement anything. `research/v8_readiness_engine.py`,
  `research/v8_forward_readiness_report.py`, and
  `research/v8_final_state.py` are all unmodified as of this document.

## If approved, what changes

1. `research/v8_readiness_engine.py`: split `path_data_ready` into
   `selection_data_ready` (poll-outcome-keyed, per (a)) and
   `exit_derivation_data_ready` (path-keyed + representativeness +
   imputation bounds, per (b)) — two named gates instead of one
   overloaded one.
2. `research/v8_forward_readiness_report.py`: report both gates
   separately per candidate, plus the representativeness table and
   imputation-bound columns for any exit-EV number.
3. New tests proving: (i) the SELECTION gate never reads a path file,
   (ii) the EXIT gate never presents an EV number without its
   sensitivity bounds, (iii) holdout is never touched by either.

None of this is done yet. Awaiting sign-off.
