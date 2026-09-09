# Exit Evidence Hierarchy

THR-BATCH T4. Maps every exit-design question this project needs answered
to its actual data source, the real n behind it as of 2026-09-09, and a
pessimistic-imputation sensitivity column per the YD2 framework
(`docs/RECEIPTS.md`'s YD2 entry) for whatever mass the source doesn't
cover. Nothing here is invented — every number below is cited to a live
query or an existing, already-committed constant.

## Why two tiers of evidence exist

- **Poll curves** (`research/outcome_poller.py`): fixed-offset price
  reads (T1m/T3m/T5m/T10m/T20m) independent of trade-tick data. High
  coverage (thousands of rows), but only tells you the price AT those
  specific clock offsets — no continuous path, so no shakeout depth, no
  reversal timing, no trail-width calibration.
- **Tick paths** (forward-collected + reconstructed, T1-T3): a
  continuous trade-by-trade price series. Needed for anything that
  depends on *path shape*, not just endpoints. Much lower n historically
  (6-9% yield from forward collection alone) — this is what THR-BATCH
  exists to fix.

| Exit-design question | Source | Real n (2026-09-09) | Pessimistic-imputation sensitivity |
|---|---|---|---|
| Time-stop calibration (does holding past N minutes typically pay off?) | Poll curves | V8-P0: 1,145 · V8-P3: 458 (train+val, `research/v8_entry_ev_report.py`, holdout-safe) | Not needed — poll coverage is near-complete (77-99%), doesn't consume paths (YD2(c): floor retired for gates that provably don't consume paths). |
| Decay curves (how does median/percentile return move from T1m→T20m?) | Poll curves | Same as above, per-offset (price_t1m/t3m/t5m/t10m/t20m all populated per token) | Same — poll-outcome-keyed, not path-consuming. |
| TP hit-rates (what fraction of tokens clear +X% at some poll mark?) | Poll curves | Same as above | Same. |
| Shakeout depth (how far does price dip before a real winner continues?) | Tick paths | V8-P0: 258 combined (90 forward + 178 reconstructed, 10 overlap) · V8-P3: 125 combined (52 forward + 78 reconstructed, 5 overlap) | **Path-consuming — floor retained per YD2(c).** No-path mass imputed at: (a) poll-outcome value with the real measured **-1.99% round-trip execution cost** haircut applied (`docs/RECEIPTS.md`, live execution-proxy observations), and (b) the frozen exit specs' own `hard_stop` floor (E0/E1/E3: **-35%**, E2: **-50%**, `research/analysis/replay_exits.py`) as the pessimistic bound. |
| Trail width (what `trail_pct`/`activates_at` combination captures upside without giving back too much?) | Tick paths | Same as above | Same. |
| Reversal timing (how long after peak does a real reversal typically take?) | Tick paths | Same as above | Same. |

## The readiness-floor nuance (must travel with any exit-EV number below)

`research/v8_readiness_engine.py`'s `exit_derivation_ready` gate requires
**both** `representative_path_n >= MIN_PATH_N` (100) **and**
`path_coverage_pct >= MIN_PATH_COVERAGE_PCT` (50%). T3's combined corpus
changes only one of the two:

| Candidate | Combined valid-path n | vs MIN_PATH_N=100 | Coverage (combined n / admission-eligible n) | vs MIN_PATH_COVERAGE_PCT=50% |
|---|---|---|---|---|
| V8-P0 | 258 (90 forward + 178 reconstructed − 10 overlap) | **CLEARS** (2.6x) | 258/1,541 = 16.7% | **does not clear** |
| V8-P3 | 125 (52 forward + 78 reconstructed − 5 overlap) | **CLEARS** (1.25x) | 125/621 = 20.1% | **does not clear** |
| BASELINE-0 | 12 (12 forward + 0 reconstructed) | does not clear | — | does not clear |
| V8-P1 | 12 (12 forward + 0 reconstructed) | does not clear | — | does not clear |

This is the first time in the project `MIN_PATH_N` has ever been cleared
for V8-P0 or V8-P3. The coverage floor uses the existing admission-
eligible-population denominator (`path_collection_eligible_n`), which is
about the P16-3 budget controller's own funnel — a reconstruction-
augmented corpus samples from a broader population that never goes
through that funnel at all, so whether this specific denominator is even
the right one to apply here is a real, open, unresolved question, not
something T4 decides unilaterally. Per the standing rule (no threshold
moved without written provenance, and YD2(c)'s "floor stays unless paired
with its sensitivity bound"), `exit_derivation_data_ready` stays reported
as **not fully met** under the existing formula. T5 proceeds anyway
because the absolute-n floor — the one that actually gates whether a
statistic is trustworthy at all — is now cleared for the first time, and
that is real, useful evidence; T5's results are labeled accordingly
(below the formal coverage-% bar, past the raw sample-size bar).

## Representativeness caveat that applies to every tick-path number above

T3's own representativeness check (`docs/RECEIPTS.md`) found the
reconstructed subset skews toward **smaller** `pct_change_peak` (median
+58.9% vs the full population's +86.2%) — reconstruction only captures
on-curve trades, and the biggest, fastest winners graduate off-curve
soonest. Any shakeout/trail/reversal conclusion drawn from the combined
corpus likely **understates** extreme-winner tail behavior, not
overstates it. `progress_at_signal` and `vsol_at_signal` are essentially
unbiased (see the same RECEIPTS entry).
