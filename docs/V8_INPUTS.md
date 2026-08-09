# V8 Design Inputs — Four Analysis Artifacts

Generated: 2026-07-29

---

## N4(a) — Forward-Validation Table (v8_pass vs v7_pass vs population)

**Window**: most recent 13 days of data (Jul 16–Jul 29 2026)
**Era filter**: CLEAN ERA ONLY (rows where any price_source_*/price_status_* is non-NULL)
**Source**: report.py run 2026-07-29

Clean era has n=7 rows total — RF1 deployed Jul 28, only 41h old. Full-dataset smart_money used as proxy.

| Cohort | era | n | priced | win% | >=+30% | >=+50% | >=+100% | SM_v1_hit% | notes |
|---|---|---|---|---|---|---|---|---|---|
| v8_pass (SM_hit=True) | full | 91 | 91 | 92.3% | n/a | n/a | n/a | 100% | clean-era n too small; full dataset proxy |
| v7_pass (screener) | full | 205 | 196 | 42.3% | n/a | 17.9% | 4.6% | n/a | |
| population | full | 18282 | 1093 | 39.9% | n/a | 16.3% | 3.8% | n/a | |
| clean era (ALL) | clean | 7 | 7 | 0% | 0% | 0% | 0% | n/a | RF1 only 41h old; outcomes not yet resolved |

**Note**: clean era will grow ~500 rows/day. Re-run in 7 days for valid forward-validation.

Smart-money v1 pinned: smart_money_hit=True cohort n=91, win=92% across full dataset.

---

## N4(b) — Era-Split Re-Read: Progress Buckets + v7 Counterfactual

**Source**: report.py run 2026-07-29 (18,282 rows, 38 days)

### Win Rates

- social_alert_bc ALL (preRF1): n=14,393, priced=287 (2%), win=43.6%, >50%=20.2%, max=+1118%
- social_alert_grad ALL (preRF1): n=807, priced=771 (96%), win=38.7%, >50%=14.4%, max=+1064%
- V7 traded: n=753, win=37.3% vs NOT V7 traded: n=17,529, win=41.0%

### Survivorship-Bias Delta (clean vs preRF1)

- preRF1 BC: 2% priced (DexScreener misses BC tokens) — win rate 43.6% is computed on the 2% that graduated and got priced
- clean era: n=7, all priced (0% win — outcomes not yet resolved at 1m/3m intervals)
- Bias note: the preRF1 43.6% BC win rate is a survivor bias artifact — only tokens that hit DexScreener (graduated) are in the priced set

### Progress_at_signal Buckets

0 rows with pp_vsol data (field not yet flowing from outcome_poller to Supabase)

### Missed Winners (screener-rejected >=+50%)

| Filter | missed | med peak | max peak | >=+100% |
|---|---|---|---|---|
| liq<8k | 81 | +131.4% | +1118.4% | 47 |
| bsr<0.55 | 39 | +92.5% | +655.9% | 18 |
| rug>500 | 15 | +86.0% | +957.3% | 6 |
| vol>50k | 7 | +70.2% | +444.1% | 2 |
| vol<2k | 1 | +558.3% | +558.3% | 1 |

Single-filter removable: 71/143 (50%)

---

## N4(c) — path_stats Full Output

> **STATUS 2026-08-03: STILL BLOCKED — do not treat as ready for the V8 freeze session.** PC2 backfill ran to completion (400/400 tokens, 0 script errors), but see the two root-cause findings below. Net usable data: **67 path files, all backfill-sourced, 0 forward-collected.** Every cell below needs n≥100; none clear that bar.

**Finding 1 — PC2 backfill silent low yield (17%).** `research.backfill_paths --winners 200 --losers 200 --parse-mode std_rpc` processed all 400 tokens with zero fetch/parse errors, but only 67 produced any tradeable rows (`_extract_rows_std` in `research/backfill_paths.py:336`, gated by `if token_amount == 0 or sol_amount == 0: continue`). The other 333 silently returned 0 rows — logged only at `log.debug` (`backfill_paths.py:499`, invisible at the script's own `logging.INFO` level, `backfill_paths.py:48`) so the run *looked* clean end-to-end. The `sol_amount`/`token_amount` heuristic (fee-payer's raw pre/post balance deltas, largest per-mint token delta across all touched ATAs) likely misses most real swaps — probably wrong for relayed/aggregator-routed txs where the fee payer isn't the trader, or multi-instruction txs where the largest token delta isn't the swap leg. **Not yet root-caused to a fix** — would need to compare a failing tx's raw `getTransaction` body against what the heuristic expects.

**Finding 2 — forward (PC1/live) collection is at 0% tick yield, not fixed by N7(a) as previously believed. ROOT CAUSE NOW CONFIRMED, and it is a hard blocker, not a code bug.** VPS `quantbot-research.service` day reports show `path_files` opened matching `tokens_scheduled` every day (Jul29: 1459, Jul30: 1218, Jul31: 1082, Aug1: 667, Aug2: 420) — but every tracked token's session ended with `ticks=0`. Investigation found three stacked issues, in the order uncovered:

1. **Wrong PumpPortal subscribe message format** (`peak_tracker.py` sent `{"action":..., "tokenAddress": addr}` instead of the `{"method": "subscribeTokenTrade", "keys": [...]}` format used by the working `memecoin/pumpportal_monitor.py`). Fixed, commit `e36a867`, deployed 2026-08-03 00:36 UTC.
2. **`FileQueueListener` (`research/tg_listener.py`) stuck on a persisted byte-offset (48MB) past the current file's EOF (4.4MB, file had been truncated)** — silently blocked 100% of new alerts from reaching the research pipeline regardless of fix #1. Fixed (reset-on-shrink), commit `7ccb518`, deployed 01:03 UTC. Unblocked a 6,326-alert / 10.9h backlog draining at ~1.2/min (~88h to catch up) — manually skipped by advancing the offset pointer to EOF at 01:27 UTC so live tracking wasn't stuck behind days of dead tokens.
3. **PumpPortal now rejects `subscribeTokenTrade`/`subscribeAccountTrade` without a funded API key.** After fixes #1 and #2, genuinely fresh real-time sessions (`dur≈967-1349s`, confirmed *not* stale backlog) still showed `ticks=0`. Direct standalone test (bypassing the bot entirely — plain `websockets` client, corrected `{"method": "subscribeTokenTrade", "keys": [...]}` payload, no API key) got an explicit rejection from PumpPortal: `"'subscribeTokenTrade' and 'subscribeAccountTrade' methods are only available when connecting with an API key funded with at least 0.02 SOL."` Neither `research/peak_tracker.py` nor `memecoin/pumpportal_monitor.py` pass any API key on connect (`wss://pumpportal.fun/api/data`, no query param, no `.env` var for it on local or VPS). **This is now the confirmed blocker — fix #1 was a real, independent bug worth having fixed, but was never going to produce ticks on its own; the account needs a funded PumpPortal API key wired into the connection to unblock forward tick collection at all.**

> **Separate, higher-priority flag for the live trading bot (outside this session's scope, but found while investigating this):** `memecoin/pumpportal_monitor.py` uses the identical unauthenticated `subscribeTokenTrade` call for real-time position-price monitoring (documented as "Price Source Priority #1, ~1s latency" in CLAUDE.md). VPS logs show **zero** real-time PumpPortal price/tick lines in the last 24h (only the one-time "monitor started" line) — strongly suggesting the live bot has been silently running on its DexScreener/Jupiter fallback (~2s poll) this whole time instead of the intended ~1s real-time feed, with no error ever surfaced (same silent-`mint`-missing-drop pattern as everywhere else in this investigation). Worth a dedicated look independent of the Aug 5 research deadline.
>
> Historical day-folders (2026-07-29 → 2026-08-01, ~4,400 gzipped files per the day-report counts) are **no longer present on the VPS disk** — cause not identified (ruled out: git/gitignore, logrotate, tmpfiles.d, cron, disk-space eviction). Moot for N4(c)/(d) purposes since per Finding 2 those files would have been header-only anyway (same root cause, no funded key existed then either).

**Bottom line on Finding 2: forward collection cannot resume until a funded (≥0.02 SOL) PumpPortal API key is obtained and wired into the WS connection URL in both `research/peak_tracker.py` and (separately, for the live bot) `memecoin/pumpportal_monitor.py`. This is an access/funding decision, not something fixable in code alone.**

**Source**: path_stats run 2026-08-03 (post-PC2-backfill) — `python3 -m research.analysis.path_stats`, log at `logs/path_stats_20260803.log`

67 path files found (all under `logs/research_paths/backfill/`), 0 with `progress_at_signal` metadata (field not yet flowing from outcome_poller, unchanged from N4b note).

All cells: **INSUFFICIENT** (max n=67 pre-split; every sub-bucket splits further, so effectively n=0-57 per cell, need ≥100):

| Section | n | status |
|---|---|---|
| A — shakeout depth by drawdown bucket (×3 targets) | 0 per bucket | INSUFFICIENT |
| B — post-peak retention by drawdown bucket | 0 per bucket | INSUFFICIENT |
| C — pre-dump order flow | 19 | INSUFFICIENT (need ≥100) |
| D — graduation velocity | 0 | INSUFFICIENT (backfill paths excluded, vsol=0 in std_rpc history) |
| E — peak-mcap distribution | 67 overall, 0 per progress bucket | INSUFFICIENT |
| F — conditional continuation | 57 (qualifying trough) | INSUFFICIENT |
| G — unique-buyer velocity | 67 | INSUFFICIENT |
| H — sniper density | 67 | INSUFFICIENT |

### Paths Sanity Line

| field | value |
|---|---|
| File count | 67 (.csv.gz, all backfill) |
| Date range | backfill covers alert_time back to whenever the 200th winner/loser by recency falls (recent weeks) |
| Winner-path count | 67 with real tick data (17% yield off 400 attempted) |
| Backfill set | RAN 2026-08-02 17:22 → 2026-08-03 01:59 (400/400, 0 errors, 17% usable yield) |

---

## N4(d) — replay_exits

> **STATUS 2026-08-03: numbers below are NOT reliable — same 67-path dataset as N4(c), plus a likely data-quality artifact (see caveat).** Do not use for TP/stop calibration yet.

**Source**: replay_exits run 2026-08-03 — `python3 -m research.analysis.replay_exits`, log at `logs/replay_exits_20260803.log`

n=64 (3 of 67 paths skipped — too short/empty). All three specs ran clean, no errors.

| Spec | n | win_rate | mean_pnl | median_pnl | p25 | p75 | p90 | exit reasons |
|---|---|---|---|---|---|---|---|---|
| A (v7 current) | 64 | 54.7% | +72,529.6% | +2.4% | -94.4% | +75.5% | +462,376.5% | hard_stop:37 trail_stop:15 path_end:12 |
| B (early-TP-heavy) | 64 | 54.7% | +72,530.4% | +2.4% | -94.4% | +76.6% | +462,376.5% | hard_stop:37 trail_stop:16 path_end:11 |
| C (wide-stop/small-size) | 64 | 54.7% | +72,530.5% | +2.4% | -80.2% | +75.5% | +462,376.5% | hard_stop:36 trail_stop:15 path_end:13 |

Winner by median PnL: Spec A (ties on median with B/C; A vs B delta +0.8-0.9pp mean_pnl, essentially noise at this n).

**Caveat — mean_pnl and p90 are not credible.** +72,530% mean and +462,377% p90 on a memecoin dataset almost certainly reflect the same std_rpc price-derivation heuristic from Finding 1 above producing a handful of garbage micro-price outliers (e.g., a tx where the "largest token delta across all touched ATAs" isn't the actual swap leg, producing a wildly wrong implied price). Median (+2.4%) is far more plausible and is probably the only usable number here, and even that is on n=64 — well under the ≥100 target and drawn from the same 17%-yield, unvalidated extraction path.

### Three Comparison Configs

| Spec | hard_stop | trail_tiers | time_stop | notes |
|---|---|---|---|---|
| A (v7 current) | -35% | [+30%/−25%, +100%/−25%, +300%/−15%] | 90min | current production config |
| B (early-TP-heavy) | default | [+20%/−20%, +60%/−20%] | 45min | tighter exit |
| C (wide-stop/small-size) | -50% | default | 120min | size=0.5x |

---

## Bottom line for the 2026-08-05 V8 freeze session

**Not ready, and will not be ready by 2026-08-05 for N4(c)/(d) specifically — this is now a known, bounded gap rather than an open mystery.** N4(a) is on track (3,257 clean-era outcome_complete rows since Jul28, growing ~450-1,250/day — the 7-day re-run will have a real sample by Aug5). N4(b) is populated and usable as-is.

**N4(c) and N4(d) are not ready and cannot become ready before Aug 5 without a funding decision.** Real usable sample is 67 backfill-sourced tokens (need ≥100 per cell, ~8-12 cells), and the replay_exits mean/p90 numbers show signs of corruption from the backfill's price-derivation heuristic (Finding 1). Forward collection — the only path to a clean, large sample — is confirmed blocked on PumpPortal requiring a funded (≥0.02 SOL) API key for `subscribeTokenTrade` (Finding 2, root-caused 2026-08-03; two independent code bugs were also found and fixed along the way — wrong subscribe format `e36a867`, and a stuck alert-queue offset `7ccb518` — but neither was the actual blocker). Options for the Aug 5 session: (a) fund a PumpPortal API key today and accept only ~1-2 days of forward data by Aug5 (thin but real, and compounding daily after); (b) root-cause Finding 1's 17%-yield backfill heuristic to widen the backfill-only sample; (c) go into the Aug5 session with N4(a)/(b) only and treat exit-level/TP-stop calibration (N4c/d) as a follow-up once (a) or (b) lands.

---

## [K5 nightly] 2026-08-09

FREEZE GATE (K5): clean_n=7276 (target 2500, MET)  |  path_stats INSUFFICIENT cells=32 (still blocking)  |  NOT READY

7. PROGRESS_AT_SIGNAL BUCKETS  [RC1: clean era only]  (pp_vsol / 115)
======================================================================
  Rows with pp_vsol data: 145 total  (145 clean / 0 preRF1 excluded)

  Bucket         n    med peak   >=+30%   >=+50%   p25/p50/p75 TTP (min)
  ----------  -----  ----------  -------  -------  ----------------------
  <50%            6      +305.9%        6        5                   1/3/5
  50-70%          1      -100.0%        0        0                   -/1/-
  70-85%         55       +65.1%       35       29                  1/3/10
  85%+           83       +91.5%       71       59                  1/3/10

======================================================================

8. READINESS VERDICTS (clean-n + days-to-n≥300 for candidate V8 rules)
======================================================================
  Collection span:  49 days  (450 alerts/day)
  Complete rows:    22041  (22041 non-partial)

  ALL complete non-partial                 n=22041  med=   +0.8%  wr=   51%  [READY]
  social_alert_bc only                     n=18127  med=  +23.2%  wr=   67%  [READY]
  snapshot_ok=True (DexScreener data)      n= 3388  med=   +0.8%  wr=   51%  [READY]
  pp_vsol available (BC real-time)         n=    2  med= +100.4%  wr=   50%  [1d to go]
  progress_at_signal < 0.5 (early BC)      n=    7  med= +305.9%  wr=  100%  [1d to go]
  progress_at_signal 0.5-0.70              n=    1  med= -100.0%  wr=    0%  [1d to go]
  progress_at_signal 0.70-0.85             n=   69  med=  +65.1%  wr=   95%  [1d to go]
  screener_passed (v7 filter)              n=  301  med=   +0.0%  wr=   40%  [READY]
  smart_money_hit=True                     n=  130  med= +132.7%  wr=   76%  [0d to go]
  top10_holder_pct available               n= 1104  med=  +16.0%  wr=   62%  [READY]
  creator_holds_pct available              n=    0  med=     n/a  wr=   n/a  [1d to go]

======================================================================
