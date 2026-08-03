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

**Finding 2 — forward (PC1/live) collection is at 0% tick yield, not fixed by N7(a) as previously believed.** VPS `quantbot-research.service` day reports show `path_files` opened matching `tokens_scheduled` every day (Jul29: 1459, Jul30: 1218, Jul31: 1082, Aug1: 667, Aug2: 420) — but `journalctl -u quantbot-research --since "24 hours ago" | grep -oE "ticks=[0-9]+" | sort | uniq -c` returns **`420 ticks=0`** — literally every one of the last 420 tracked tokens ended its ~15–25min tracking window with zero ticks written. Prime suspect: `_price_from_msg()` (`research/peak_tracker.py:210`) derives price only from `vSolInBondingCurve`/`vTokensInBondingCurve` on incoming PumpPortal `subscribeTokenTrade` messages and returns `None` (dropping the tick entirely, `peak_tracker.py:397-411`, wrapped in a bare `except: pass` at line 444/446) if either is 0/absent. If PumpPortal's live payload no longer carries those fields as expected (schema drift) or they're legitimately empty for a chunk of message types, every tick silently vanishes — matching the observed 100% failure rate exactly. **Not yet confirmed** — would need one raw WS message dumped to verify field presence. This means the "N7(a) fixed forward collection" note below (written 2026-07-29) is **incorrect as of today** — forward collection has produced effectively zero usable ticks since at least Jul 30, most likely obscured because the same "header-only" symptom looks identical whether the cause is the old NameError or this new price-derivation gap.
>
> Historical day-folders (2026-07-29 → 2026-08-01, ~4,400 gzipped files per the day-report counts) are **no longer present on the VPS disk** (`find /root/quant-bot/logs/research_paths` only returns the 2026-08-02 and backfill/ dirs, 14 files total) — cause not identified (ruled out: git/gitignore since `research_paths/` was never tracked, logrotate, tmpfiles.d, cron, disk-space eviction). Moot for N4(c)/(d) purposes since per Finding 2 those files would have been header-only anyway, but flagging as an unexplained data-loss event worth a separate look.

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

**Not ready.** N4(a) is on track (3,257 clean-era outcome_complete rows since Jul28, growing ~450-1,250/day — the 7-day re-run will have a real sample by Aug5). N4(b) is populated and usable as-is. **N4(c) and N4(d) are not** — real usable sample is 67 tokens (need ≥100 per cell, and there are ~8-12 cells), and the numbers that did compute (replay_exits mean/p90) show signs of being corrupted by the backfill's price-derivation heuristic. Two independent bugs need root-causing before this section can support a TP/stop decision: (1) why 83% of std_rpc backfill attempts silently extract 0 rows, and (2) why forward PumpPortal tick collection has been at 0% yield for at least 4 days despite looking healthy in file-open/gzip counts.
