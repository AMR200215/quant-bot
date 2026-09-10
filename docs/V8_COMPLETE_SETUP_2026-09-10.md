# V8 Complete Setup — Status, Gaps, and What's Actually Needed

Written 2026-09-10. Supersedes `docs/V8_HANDOFF_2026-08-29.md` (that
document predates the entire THR-BATCH reconstruction effort and the
entry-EV report — most of what it describes as blocked is now resolved
or superseded below). This is meant to be a complete, standalone
reference — no other document should need to be read first to
understand where V8 stands or what's left.

**Read this whole document before touching anything.** It covers entry,
every exit component (hard stop, time stop, trail stop, profit lock, TP),
data readiness, what's completely unaddressed, a stress test of what
stays unresolved even after the highest-priority fix (§8), and the
standing rules that must not be violated. The goal is that after reading
this, nothing further needs to be asked — if something is unclear or
looks stale, verify it live (SSH, direct query) rather than guess; every
number below
is cited to a file or a live query at write time, but this project's own
data changes daily (new forward collection runs continuously).

---

## 1. What V8 is, in one paragraph

A social-alert memecoin trading strategy for Solana pump.fun tokens. Two
independent halves: **entry** (which alert-time signals to trade,
`research/v8_candidate_registry.py`) and **exit** (when to sell,
`research/v8_exit_registry.py`). `LIVE_TRADING=false` on the VPS — the
whole system is currently paper/research only. Nothing in this document
changes that; going live is a separate, explicit decision the user makes,
not something to infer from data looking good.

---

## 2. Entry (SELECTION side) — validated, one decision outstanding

Four frozen candidates: `BASELINE-0`, `V8-P0`, `V8-P1`, `V8-P3`
(`research/v8_candidate_registry.py`, hash-frozen before holdout was ever
touched). Only two clear any real floor:

| Candidate | n (train+val, holdout-safe) | win rate | mean peak | median peak |
|---|---|---|---|---|
| **V8-P0** | 1,145 | 67.1% | +178.0% | +82.3% |
| **V8-P3** | 458 | 62.2% | +158.7% | +71.2% |
| BASELINE-0 | 17 | — | — | below floor, informational only |
| V8-P1 | 8 | — | — | below floor, informational only |

Source: `research/v8_entry_ev_report.py`, `docs/RECEIPTS.md` (search
"V8 entry-EV"). `SELECTION_DATA_READY = True` for V8-P0/V8-P3 in
`research/v8_readiness_engine.py` — the entry side has real, sufficient,
holdout-safe data behind it. This is genuinely solid.

**Caveat that must travel with these numbers:** "peak" is the best price
*reached*, not a realized exit. No execution cost, no exit timing. This
is exactly why the exit-side work (below) exists — entry-EV alone cannot
tell you what a real trade would have made.

**Outstanding decision, not yet made:** V8-P0 (broader — no progress
condition beyond `CURVE_ACTIVE` venue, more signals/day, slightly higher
win rate) vs V8-P3 (narrower progress filter, still strong). Nothing in
this project has picked one over the other for actual deployment.

---

## 3. Exit — four components, three settled, one is the real leak

Four frozen specs, `research/v8_exit_registry.py`: E0 (current v7 rule,
control), E1 (earlier/tighter trail), E2 (wider hard stop), E3 (data-
derived 7-min time stop, otherwise = E0). Every spec has: `hard_stop`,
`trail_tiers`, `tp_levels`, `time_stop_min` (+`time_stop_min_gain=0.30`
floor), `profit_lock_min_gain=0.40`/`max_gain=1.00`/`stall_sec=60`.

All four validated below via `research/thr_winner_exit_analysis.py`
against real winners: forward-collected tokens (V8-P0 n=16, V8-P3 n=9)
that actually reached the +50% winner threshold (`WINNER_THRESHOLD_PCT`,
reused from `research/analysis/path_stats.py`) at some point after entry.
Full breakdown: `research/thr_winner_exit_results.json`.

### 3a. Hard stop (-35% E0/E1/E3, -50% E2) — VALIDATED, DO NOT RETUNE

93.75% of V8-P0's real winners (15/16) and 100% of V8-P3's (9/9) survive
a -35% stop — their worst dip before winning was shallower than -35%.
The one exception dipped -72.4% before recovering; widening the stop to
save that single case would mean tolerating far deeper losses across the
much larger population of genuine losers (median max-drawdown across ALL
forward-collected tokens is -76% to -78%) — not a good trade on this
evidence. `docs/RECEIPTS.md`, "CORRECTION (2026-09-10)" section — an
earlier, wrong conclusion in this same file claimed the opposite; that
correction explains exactly why it was wrong and should be read once for
the methodology (checking "did the stop actually kill real winners," not
just "how often does it fire").

### 3b. Time stop (90/45/120/7 min, E0-E3) — VALIDATED, DO NOT RETUNE

**Zero real winners, across all four specs, were ever exited via
`time_stop`.** Not a single one. No evidence this costs anything, at any
of the four tested durations including E3's aggressive 7-minute version.

### 3c. Trail stop — NOT VALIDATED, THE REAL LEAK, HIGHEST-PRIORITY NEXT STEP

Trail stop is the *dominant* real-winner exit mechanism (8-14 of 16 for
V8-P0 depending on spec; 5-9 of 9 for V8-P3) and captures only **13-31%
of the eventual peak gain** the token actually went on to reach:

| Candidate | Spec | trail_stop n | mean realized | mean eventual peak | captured_fraction |
|---|---|---|---|---|---|
| V8-P0 | E0/E2/E3 | 8 | +53.0% | +207.7% | **0.20** |
| V8-P0 | E1 (tighter trail) | 14 | +28.2% | +173.1% | **0.19** |
| V8-P3 | E0/E2/E3 | 5 | +81.7% | +277.2% | **0.31** |
| V8-P3 | E1 | 9 | +25.0% | +203.3% | **0.13** |

E1's tighter trail (arms at +20% gain, 20% trail) captures *less* than
E0/E2/E3's default (arms at +30%, 25% trail) — tighter/earlier arming
locks in profit sooner at the cost of more upside, which is intuitive
once measured but was not previously quantified anywhere in this project.

**No retuned trail_tiers candidate has been drafted or applied.** This
is the single most actionable, best-evidenced next step in the entire
V8 project right now. A wider/later-arming trail could plausibly move
real captured returns from the current ~20-31% of peak toward something
much higher — but this needs to be *derived* from the real per-winner
drawdown-before-continuing data (same rigor as the hard_stop check),
proposed as an explicit new `E4`-style registry entry (see
`research/v8_exit_registry.py`'s own "EXPERIMENT V2" pattern for how E3
was added without touching E0-E2), and re-validated via
`research/thr_run_t5.py` + `research/thr_winner_exit_analysis.py` against
the same combined corpus before it's trusted. Not done. Start here.

### 3d. Profit lock (stall detector, gain 40-100%, 60s stall) — leaky, lower priority

Captures ~49-57% of eventual peak — meaningfully better than trail_stop
but still leaves roughly half on the table. Same "propose + re-validate"
process applies if retuned; secondary priority to trail_stop.

### 3e. TP ladder — DOES NOT EXIST, never designed

`tp_levels=[]` in every one of E0-E3. No partial-profit-taking ladder has
ever been designed or tested for V8. Given how much trail_stop leaves on
the table, a TP ladder (lock in a fraction at, say, +50%/+100%, let the
remainder ride under a looser trail) is a plausible complementary fix —
completely unstarted, not even a draft.

### One caveat on `path_end` exits in the table above

`path_end` shows the *highest* apparent captured_fraction (0.66-0.77) —
likely an artifact of the recorded forward path simply ending near its
own high point (collection window or graduation cutoff), not evidence
that holding longer would help. Don't read it as "holding is best" — we
don't know what happened after the path stopped recording for those
tokens.

---

## 4. Data readiness — still not formally met, independent of any of the above

`research/v8_readiness_engine.py`'s `exit_derivation_ready` needs BOTH:

| Candidate | Combined valid-path n | vs `MIN_PATH_N=100` | Coverage (combined n / admission-eligible n) | vs `MIN_PATH_COVERAGE_PCT=50%` |
|---|---|---|---|---|
| V8-P0 | 258 (90 forward + 178 reconstructed − 10 overlap) | **clears** | 16.7% | **does not clear** |
| V8-P3 | 125 (52 forward + 78 reconstructed − 5 overlap) | **clears** | 20.1% | **does not clear** |

The absolute-n floor cleared for the first time in the project's history
via the THR-BATCH reconstruction effort (`docs/RECEIPTS.md`, "THR-BATCH
T2/T3"). The coverage floor has not, and it is a genuinely open question
— not decided anywhere — whether the existing admission-funnel-based
denominator (`path_collection_eligible_n`, from the live P16-3 budget
controller) is even the right yardstick for a corpus that includes
chain-reconstructed data that never went through that controller at all.
`docs/EXIT_EVIDENCE.md` flags this explicitly. Resolve this — or keep
accumulating combined-corpus n via periodic re-runs of
`research/thr_sample_t2.py` + `research/thr_run_t3.py` — before treating
`exit_derivation_data_ready` as formally True.

**Representativeness caveat that must travel with every exit-side
number in this document:** the reconstructed-path corpus underrepresents
the fastest/biggest winners (they graduate off pump.fun's bonding curve
before enough on-curve history accumulates to reconstruct) — median
`pct_change_peak` in the valid-reconstructed subset is +58.9% vs +86.2%
in the full population (`docs/RECEIPTS.md`, "THR-BATCH T2/T3"). Any
exit-timing conclusion above likely *understates* true upside tail
behavior, not overstates it.

---

## 5. Holdout — still completely locked, never touched, correctly so

No number anywhere in this entire project — not one — has ever come from
the holdout split (`research/v8_split.py`'s `grouped_chronological_split`,
`train_frac=0.6, validation_frac=0.2`, remainder holdout). This is
correct discipline and must not change until a specific final
candidate+exit combination is fully settled and pre-registered for a
**single, one-time** holdout evaluation — the actual go/no-go gate. That
point has not been reached: trail_stop tuning (§3c) is still open, and
the data-readiness coverage question (§4) is unresolved. Do not read
holdout — not even row counts beyond what's already established as safe
patterns — until explicitly instructed to run the final gate.

---

## 6. What's completely unaddressed (not V8 filter/exit research, but required for "complete setup")

None of these have been touched by any work described in this document
or in `docs/RECEIPTS.md`'s V8 sections:

- **Position sizing.** Current live config is a flat $3-5/trade
  (`CLAUDE.md`). No rule here ties size to conviction, volatility, or
  anything in this research. Undesigned.
- **Real slippage/execution-cost model.** The -1.99% round-trip figure
  used throughout §3 is a flat, real, *measured* number
  (`docs/RECEIPTS.md`) — but `research/v8_execution_cost_model.py`
  itself documents its own limitation:
  `LINEAR_SIZE_PROJECTION_ONLY (no curve-depth model for slippage/impact)`.
  No model exists for how cost scales with trade size or curve depth at
  entry.
- **Mid-trade signal rules.** `research/v8_exit_registry.py`:
  `MIDTRADE_CANDIDATES = []`, `MIDTRADE_STATUS =
  "NO_MIDTRADE_RULE_SUPPORTED"` — an explicit, deliberate finding from
  Phase 2 (P2-5), not revisited since. No mid-trade adjustment logic
  (e.g., reacting to a realert) exists.
- **T22 / execution-path infrastructure.** Separate track, `CLAUDE.md`'s
  "Known Issues" section: `pumpswap_local`'s T22 sell path only
  simulates (`PUMPSWAP_LOCAL_SELL_ENABLED=False`), root cause of the
  simulation failure still not found. Orthogonal to V8's signal research
  but blocks reliably *executing* whatever V8 eventually decides,
  specifically for Token-2022 tokens.
- **Realert-based features beyond what's already in the frozen entry
  registry.** `realert_count` exists as a conditionally-allowed feature;
  nothing beyond the current candidate registry's use of it has been
  explored as an entry or mid-trade signal.

---

## 7. Recommended order of work from here

1. **Derive and propose a retuned trail_tiers candidate** (§3c) from the
   real per-winner drawdown-before-continuing data already sitting in
   `research/thr_winner_exit_results.json`. Highest leverage, best
   evidenced, most actionable single next step.
2. Re-validate via `research/thr_run_t5.py` +
   `research/thr_winner_exit_analysis.py` against the same combined
   corpus (no new data collection needed for this iteration).
3. Consider a TP ladder (§3e) as a complementary design, informed by the
   same data — genuinely unstarted.
4. Resolve the `MIN_PATH_COVERAGE_PCT` denominator question (§4), or
   keep growing combined-corpus n via periodic THR reconstruction runs.
5. Decide V8-P0 vs V8-P3 (§2) as the one deployed entry candidate.
6. Design position sizing (§6) — currently flat, unexamined.
7. **Only after 1-6 are settled**: pre-register the final
   candidate+exit combination and run the one-time holdout evaluation
   (§5) — the actual go/no-go gate. Not before.
8. Separately, on the execution-infrastructure track (not V8 research):
   resolve the T22 `pumpswap_local` simulation failure.

---

## 8. Stress test: what's still unresolved even after trail_stop is fixed

Fixing trail_stop (§3c, §7 step 1) is not a finish line. Stress-tested
this explicitly, 2026-09-10, before it gets mistaken for one:

- **The fix itself risks being invalid before it's tried.** Any retuned
  trail is derived from only 16 (V8-P0) + 9 (V8-P3) real winners. If
  "validated" by checking improvement on that *same* set, that's
  curve-fitting to a sample smaller than one day of signals, not real
  validation. It needs checking against winners it wasn't derived from
  — fresh forward collection, not a re-run on the identical corpus.
- **Likely measurement trap: widening the trail will look better than
  it is.** Forward-collected paths have a finite recording window. A
  wider/later-arming trail holds positions longer — some will run off
  the end of recorded data and land in `path_end`, which §3c already
  flagged as an inflated bucket (0.66-0.77 captured_fraction, an
  artifact of the window ending near a high point). Widening the trail
  could mechanically shift exits from `trail_stop` into `path_end` and
  look like an improvement purely because measurement stopped, not
  because the strategy improved. Must be checked explicitly.
- **Trail doesn't act alone.** Exit rules fire in priority order:
  `hard_stop` → `trail_stop` → `profit_lock` → `time_stop`. A trail that
  arms later exposes more tokens to `hard_stop` for longer before trail
  gets a chance to help. Retuning trail without re-checking hard_stop's
  hit rate against the *new* trail could quietly erode what §3a already
  validated as fine.
- **Structural blind spot: graduation isn't in this data at all.**
  Everything in §3-§4 is on-curve only. The biggest, fastest winners
  graduate to PumpSwap — different venue, different execution plumbing
  (`exit_router.py`). No backtesting here covers what happens to an open
  position that graduates mid-hold; live behavior there is unvalidated
  against any of this exit logic.
- **Fixing trail doesn't touch anything in §4 or §6.** Not the 50%
  coverage floor, not position sizing, not the flat execution-cost
  model, not mid-trade rules, not the holdout gate. A better trail spec
  doesn't move any of these — worth stating explicitly so "trail is
  fixed" doesn't get read as "V8 is done."
- **Never examined at all: portfolio-level risk.** Every check in this
  document is per-trade. Real trading holds multiple positions
  concurrently; nothing here has looked at correlated drawdowns (a
  broad memecoin risk-off moment hitting many open positions' hard
  stops at once). Per-trade EV being positive doesn't guarantee that's
  survivable at the portfolio level.
- **No drift detection.** Every number in §2-§3 is calibrated on a
  specific ~August-September 2026 window. Memecoin market structure
  shifts. Nothing re-validates this periodically or flags when it stops
  matching reality.
- **The optimization target itself may be wrong to chase.** Mean
  `captured_fraction` (§3c) doesn't account for variance — a wider
  trail could capture more on real winners while giving back more on
  tokens that fake a breakout and reverse. And no amount of exit tuning
  changes the ~17-18% base rate of tokens that ever become winners at
  all (§2's entry-EV numbers) — that ceiling is set on the entry side,
  not reachable from exit work.

None of this blocks doing the trail_stop work in §7 step 1 — it's the
best-evidenced next step regardless. It does mean: budget for a second
validation pass with fresh data before trusting a retuned candidate, and
don't treat any single fix in §3 as closing out the project.

---

## 9. Standing rules — do not violate these

- **Holdout is never read for outcome values.** Row counts are fine;
  `pct_change_peak` or any outcome value from a holdout row is not, ever,
  until §5's final gate.
- **No frozen registry edited in place.** `v8_candidate_registry.py` and
  `v8_exit_registry.py` only change via an explicit, visible new
  `experiment-vN` entry (see how E3 was added — `_E3_SPEC = dict(_V7_SPEC)`
  with one field changed and full written rationale in the same file).
  Never overwrite E0-E3 or the four entry candidates.
- **No threshold moved without written provenance** citing real,
  measured numbers — see the pattern throughout `docs/RECEIPTS.md`.
- **`LIVE_TRADING` stays `false`** until the user explicitly says
  go-live. Nothing in this document is authorization for that.
- **Prove, don't infer.** Verify live (SSH query, direct check) before
  acting on any specific number in this document — it will drift as
  forward collection continues daily. Directional conclusions (trail
  stop is leaky, hard stop is fine) are much more durable than exact
  percentages.
- **Deploy flow**: commit locally → push → SSH → stash the 6 known
  live-drift files (`docs/RECEIPTS.md`, `docs/V8_INPUTS.md`,
  `logs/memecoin_social_journal.csv`,
  `logs/trade_telemetry_summary.csv`,
  `memecoin/data/memecoin_positions.json` — needs a double-snapshot,
  see any recent deploy in `docs/RECEIPTS.md` for the exact race-
  condition workaround — and `memecoin/data/memecoin_signals.json`) →
  pull --rebase → pop → restore the positions snapshot → confirm stash
  count returns to baseline (50 as of this writing).
- **This project has cross-session state ONLY via what's committed to
  the repo.** A different Claude Code session (mobile, another desktop
  instance, Fable 5) has zero visibility into anything not pushed to
  `main`. If you learn something important, commit it — to
  `docs/RECEIPTS.md` if it's a finding, to a registry file via
  `experiment-vN` if it's a proposed threshold, never only in
  conversation.

---

## 10. Where everything actually lives (read these directly, don't re-derive)

| File | What it is |
|---|---|
| `docs/RECEIPTS.md` | Full chronological evidence trail (~4,600+ lines). Search "THR-BATCH" and "CORRECTION" for the most recent, most relevant sections. |
| `docs/EXIT_EVIDENCE.md` | Evidence-hierarchy table + the coverage-floor-denominator open question. |
| `research/v8_candidate_registry.py` | Frozen entry candidates (BASELINE-0, V8-P0, V8-P1, V8-P3). |
| `research/v8_exit_registry.py` | Frozen exit specs (E0-E3) + `MIDTRADE_STATUS`. |
| `research/v8_entry_ev_report.py` | Real entry-side win-rate/peak numbers (§2). |
| `research/v8_readiness_engine.py` + `v8_forward_readiness_report.py` | The formal `SELECTION_DATA_READY` / `exit_derivation_ready` gate logic. |
| `research/thr_reconstruct_paths.py` | Exact on-curve price reconstruction from chain history (T1). |
| `research/thr_sample_t2.py` + `research/thr_t2_sample.json` | Pre-registered 500-token sample (T2), frozen before reconstruction ran. |
| `research/thr_run_t3.py` + `research/thr_t3_results.json` | The reconstruction run: 178 valid usable paths (T3). |
| `research/thr_run_t5.py` + `research/thr_t5_results.json` | E0-E3 replayed on the combined forward+reconstructed corpus (T5), split by path source. |
| `research/thr_winner_exit_analysis.py` + `research/thr_winner_exit_results.json` | **The most actionable file for next steps** — per-real-winner, per-exit-spec breakdown of what each exit rule actually captured vs the eventual peak. |
| `research/v8_execution_cost_model.py` | Documents its own `LINEAR_SIZE_PROJECTION_ONLY` limitation (§6). |
| `logs/research_paths/reconstructed/` | The reconstructed path CSVs themselves (VPS-disk-resident, deliberately untracked-but-not-gitignored, same convention as `logs/research_paths/` generally). |
