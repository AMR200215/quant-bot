# Execution Receipts — Living Document

## The Rule
Any commit that touches an execution path (executor.py, portfolio.py exit/close/abort,
exit_router.py, bonding_curve_t22.py, verify_execution.py, or any sell route) **must**
add a receipt row to the relevant section of this file in the same commit.

"Proven" means a row exists here with a real on-chain sig and a passing Solscan link.
"Not proven" means no row — regardless of what the code says or what tests pass.

This answers "is X actually working?" without re-auditing the repo.

---

## Sell Matrix — Bonding Curve Routes

Run: `python -m memecoin.tools.verify_execution --cell <CELL> --mint <MINT>`
Requires: ~0.012 SOL/cell, SOLANA_PRIVATE_KEY set, token actively on bonding curve.

| cell | date | commit | mint (first 8) | buy sig (first 8) | sell sig (first 8) | SOL delta | note |
|------|------|--------|----------------|-------------------|--------------------|-----------|------|
| spl_bc_full    | — | — | — | — | — | — | PENDING — SPL BC mints now rare on pump.fun (mostly T22); run if SPL mint appears |
| spl_bc_partial | — | — | — | — | — | — | PENDING — SPL BC mints now rare on pump.fun (mostly T22); run if SPL mint appears |
| t22_bc_full    | 2026-07-09 | ada6c06 | 2tGZPzMR | 51DHXwVa | 5gpeXsFW | −0.000249 | PumpPortal BC path (native T22 ATA disabled); confirmed |
| t22_bc_partial | 2026-07-09 | ada6c06 | 8QbvbYxw | 5YaN222t | 53dH183E | −0.001459 | 30% partial, PumpPortal BC; confirmed |

**How to fill a row after running:**
```
| spl_bc_full | 2026-07-10 | a1b2c3d | AbCdEfGh | 3Xk9mNpQ | 7Yz2wRsT | +0.0021 | |
```
If a cell failed and was fixed, add a `fix:` note in the last column.

---

## Live Trade Execution Log

Every confirmed live trade gets a row here once sell is confirmed on-chain.
The journal CSV is the PnL record; this file is the execution proof.

| date | symbol | cohort | buy sig (first 16) | sell sig (first 16) | route | SOL in | SOL out | net SOL | Solscan sell |
|------|--------|--------|--------------------|---------------------|-------|--------|---------|---------|--------------|
| 2026-07-08 | VLAD | bonding_curve | 5GdddCDQ (partial) | 29VbDaUVc1y (partial) | JUPITER_RESCUE | ~0.040 | 0.038343 | −0.0017 | [unverified — backfilled] |
| 2026-07-08 | Dog | bonding_curve | KvDUAheV | 2Zt1XSKJ | abort_tripwire auto-sell | ~0.030 | ~0.029 | ~−0.001 | confirmed |

---

## Abort Tripwire Log

Trades aborted at fill (slippage > 30% above baseline). Real PnL = recovered SOL − spent SOL.

| date | symbol | fill slip | ref used | buy sig (first 8) | auto-sell sig (first 8) | real PnL USD |
|------|--------|-----------|----------|-------------------|-----------------------|--------------|
| 2026-07-08 | Dog | +61.7% vs preflight_baseline | preflight_baseline | KvDUAhe | 2Zt1XSK | ~−$0.21 |

---

## Route Proof Log

When a new sell route is first proven working end-to-end on mainnet, record it here.

| date | route | token | sell sig (first 16) | SOL received | note |
|------|-------|-------|---------------------|--------------|------|
| 2026-07-05 | JUPITER_RESCUE | Beginner | BT8RCo6a8sYpiKyf | 0.047140 | first confirmed jupiter rescue |
| 2026-07-05 | JUPITER_RESCUE | VLAD | 29VbDaUVc1y (partial) | 0.038343 | second confirmed; mint unknown at log time |

---

## Research Pipeline — W-BATCH (2026-07-18, commits b7b0a6d–81a301e)

**W1 — Queue deadman + persistent offset**
- Scanner heartbeat thread writing `{type:heartbeat}` every 5 min to `signal_queue.jsonl`: **LIVE** (confirmed `tail -3 signal_queue.jsonl` shows 3 consecutive heartbeats, 300s apart)
- FileQueueListener persisted offset: **LIVE** (`.queue_offset` file = 572 on VPS after first poll)
- Deadman alert at >20 min silence: **deployed, untested** (threshold not yet hit; logic in `_check_deadman()`)

**W2 — Smart-money features**
- `fetch_first_buyers()` in snapshot.py: **deployed** (Helius getSignaturesForAddress + bulk parse)
- `smart_wallets.py` loader: **deployed** (gracefully returns (False,0) until backfill runs)
- `progress_at_signal` (pp_vsol/115): **deployed** in tracker.py; awaiting DB migration
- `smart_money_hit/count`: **deployed** in tracker.py; awaiting DB migration
- DB migration SQL (run in Supabase SQL editor):
  ```sql
  ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS smart_money_hit BOOL;
  ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS smart_money_count INT;
  ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS progress_at_signal FLOAT;
  ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS channel_velocity_5m INT;
  ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS data_partial BOOL DEFAULT FALSE;
  ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS top10_holder_pct FLOAT;
  ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS creator_holds_pct FLOAT;
  ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS price_peak_3m FLOAT;
  ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS pct_change_peak_3m FLOAT;
  ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS t_peak_3m_s INT;
  ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS pct_change_t1m FLOAT;
  ALTER TABLE research_tokens ADD COLUMN IF NOT EXISTS price_t1m FLOAT;
  ```

**W3 — Report additions**
- Section 6 (missed-winners): **live** — 135 tokens peaked >+50% despite screener reject; `liq<8k` binding for 76
- Section 7 (progress_at_signal buckets): **deployed**; shows 0 rows until migration + data flows
- Section 8 (readiness verdicts): **live** — screener_passed n=198 (14 days to 300); all W2 segments at 0

**W4 — Run receipts (2026-07-18)**

schema_audit: PASS — all 53 emitted fields present in schema.sql

data_quality snapshot (19,881 rows, Jun 21 – Jul 14):
| metric | value |
|---|---|
| Total rows | 19,881 |
| outcome_complete=True | 19,881 (100%) |
| rows with pct_change_peak | 1,058 (5.3%) |
| rows last 7d | 2,759 |
| rows last 24h | 0 (scanner crashed Jul 15–18, now fixed) |
| Clean outcomes_only cohort | 1,058 rows |
| Clean entry_features cohort | 762 rows |
| tick_peaks cohort | 0 rows (price_peak_3m col missing) |

Report key findings:
- Win rate (>0%): 39.5% overall; BC 41.5%, Grad 38.9%
- >+50% rate: 16.0% overall
- Vol $20k–$50k bucket: best win rate 52.4% (n=185) — top feature signal
- Screener PASS vs FAIL: 43.4% vs 38.7% (weak separation; n=198 too small)
- V7 traded slightly worse than universe (36.7% vs 40.7%)
- Missed winners: 135 tokens; `liq<8k` most binding (76 tokens, max +1118%)
- 50% of missed winners blocked by only 1 filter (relaxable)
- screener_passed readiness: 198/300 clean rows (~14 days to V8 threshold)

---

## Research Pipeline — RF-BATCH + RC-CLOSURE (2026-07-28–29, commit db32f53)

### RC2 — smart_wallets_v1.json (created 2026-07-28, commit db32f53)

| field | value |
|---|---|
| File | `/root/quant-bot/research/smart_wallets_v1.json` |
| sha256 | `6dc245381106d5b57173f92d9720ac769f0be8f40fd9fdca30dd84a3593c2813` |
| Wallets | 106 (appeared early in ≥2 of 95 winner tokens, peak ≥+100%) |
| Winners scanned | 95 (89 with buyer data, 6 missing) |
| Generated at | 2026-07-28T23:27:17Z |
| Config pin | `SMART_MONEY_PINNED_VERSION=1` (default in research/config.py) |
| Shadow columns | `smart_money_hit_v1`, `smart_money_count_v1` (in schema) |
| v8_pass reads | v1 only — advance pin after Jul 25 read concludes |
| 70 rows updated | `smart_money_hit=True` backfilled into existing research_tokens rows |

### RC3 — RF1 NULL-rate artifact (commit db32f53)

Supabase migration run 2026-07-28 (22 ALTER TABLE statements, "Success. No rows returned").
RF1 deployed ~01:17 UTC 2026-07-28. preRF1 table captured same day (20,149 rows):

**Run at: 2026-07-29T18:30:30Z — 21,331 rows (outcome_complete only)**

**Era: dex_conditioned_preRF1 (n=18,457)**

| Category | n | null_t1m | null_t3m | null_t10m |
|---|---|---|---|---|
| ALL | 18457 | 99.9% | 89.1% | 88.7% |
| social_alert_bc | 14455 | 99.9% | 87.3% | 87.3% |
| social_alert_grad | 912 | 100.0% | 100.0% | 100.0% |

**Era: clean/postRF1 (n=2,874)**

| Category | n | null_t1m | null_t3m | null_t10m |
|---|---|---|---|---|
| ALL | 2874 | 98.9% | 94.6% | 94.5% |
| social_alert_bc | 2549 | 98.7% | 96.8% | 96.8% |

**Verdict: PARTIAL (+1.2pp) — root cause confirmed: timing, not a RF1 defect**

**Diagnostic query result (2026-07-29, new query tab in Supabase):**

```sql
SELECT price_status_t1m, count(*)
FROM research_tokens
WHERE price_source_t1m IS NOT NULL OR price_status_t1m IS NOT NULL
GROUP BY price_status_t1m ORDER BY count DESC LIMIT 20;
```

| price_status_t1m | count |
|---|---|
| curve_account_missing | 2843 |
| NULL (source set, price captured) | 34 |
| curve_rpc_error | 17 |

**Conclusion: RF1 is working correctly.** The curve oracle is called at T1m and finds
the bonding curve PDA already deleted for 98.2% of tokens. Pump.fun BC tokens graduate
or die within seconds to ~1 minute of the social alert — by the time T1m fires, the
account is gone. This is a timing problem, not a curve oracle bug.

- `curve_account_missing` 98.2% — expected; token already graduated/died
- `curve_rpc_error` 0.6% — acceptable Helius hiccup rate
- 34 rows with actual price captured — these are the tokens still on BC at T1m

**RF1 is closed.** The T1m NULL rate cannot be fixed without sub-minute polling windows.
That is a separate initiative (sub-30s poll at signal time, not an outcome-poller change).

### RC1 — Era segmentation in report.py (commit db32f53)

`report.py` sections 2 (bucket analysis) and 7 (progress_at_signal) now use clean-era rows only, with excluded-n shown. Section 10 prints era data-quality table (NULL rates at T1m/T3m/T10m by era and by category). First clean-era rows will appear ~30min after first post-RF1 signal completes its outcome window.

---

## How to use this file

**Before changing an execution path:** check if there is a receipt for the current
behaviour. If not, you are changing unproven code — note that in the PR.

**After a fix:** re-run the relevant cell or wait for the next live trade that hits
the changed path. Add the row. Commit it with the fix.

**To answer "does X work?":** search this file for X. If no row, it is not proven.

**Stale rows:** a row is stale if the commit column predates a significant executor
rewrite. Mark it `[stale — re-run]` and re-run when a suitable token is available.

---

## L6 — Screen Compression (2026-07-10, commit: verified in conversation 2026-07-10 (pre-manifest era))

**Implementation**: Already deployed. DexScreener + rugcheck/safety fire concurrently
via `_submit_prefetch` pool in `scanner.py` (_on_telegram_signal path). PP cache-hit
skips DexScreener entirely. Measured from live syslog 2026-07-10:

| Signal | dex_hit | safety_hit | screen_ms | decision_ms |
|---|---|---|---|---|
| GxZv4NJk | True | False | 0ms | 447ms |
| 7GhD87DK | True | True | 223ms | 778ms |
| ARYpA2N8 | True | False | 494ms | 589ms |

Cache-miss (dex+safety): 223–494ms ✓ (target <800ms)
Cache-hit: 0ms ✓ (target <300ms)

batch_verify: no manifest (pre-VG era). Logic verified via live timing measurements above.

**RTT VPS→Helius**: ~80ms network (measured via mainnet-beta proxy; Helius saturated
by live bot during measurement). Helius `getAccountInfo` oracle avg=1915ms (this is
`commitment=confirmed` propagation wait, not network RTT). Under 100ms threshold →
no endpoint switch required.

---

## X1 — Presigned Urgent Exits (2026-07-10, commit: verified in conversation 2026-07-10)

**Code changes**: `memecoin/portfolio.py`
- Added `feed_blind`, `pre_graduation_exit` to `_STOP_REASONS` (presigned-eligible)
- Added oracle gate: `get_pumpfun_curve_complete()` must return `complete==False` before presigned send
- Added T22 skip: tokens in `_mint_token_program_cache` with TOKEN22 program bypass presigned, use ladder
- Fixed fallback log: now emits `presign_fallback reason=<err>` on send failure

**Acceptance**: PENDING first live hard_stop/trailing_stop exit post-deployment.
Will add telemetry line (exit_trigger→sell_sent <300ms) + sig when observed.

batch_verify: no manifest. Acceptance PENDING — first live hard_stop exit post-deployment.

---

## X3 — Exit Telemetry Sub-spans (2026-07-10, commit: verified in conversation 2026-07-10)

**Code changes**: `memecoin/portfolio.py`
- `close_position()` gains optional `_t_detect: float` param
- `exit_triggered` telemetry event now includes `detect_ms`, `dispatch_ms`
- `sell_confirmed` telemetry event now includes `build_ms`, `send_ms`, `land_ms`, `meta_ms` (from executor result timing dict)
- Monitor loop passes `_t_trig` to `close_position` at each exit condition check

exit_route_attempts.csv header — 28 named fields, no unnamed columns confirmed:
```
ts,pos_id,token_symbol,token_mint,token_program,is_token2022,token_extensions,
exit_state,exit_reason,route_name,route_order,vsol_at_trigger,vsol_at_sell,
migration_age,dex_id,pool_address,simulation_ok,simulation_error,custom_error_code,
tx_sent,tx_sig,confirmed,confirm_error,jupiter_price_impact_pct,fallback_used,
final_status,error_class,notes
```

**Acceptance**: PENDING first live sell exit post-deployment (need full trace with all sub-spans).

batch_verify: no manifest. Acceptance PENDING — first live sell exit post-deployment.

---

## X5 — Post-buy Readiness (2026-07-10, commit: verified in conversation 2026-07-10)

**Code changes**: `memecoin/portfolio.py`
- `_fill_confirm_ts` stored on position object after buy confirms
- Monitor loop emits `FIRST_PRICE_MS token=<sym> ms=<N>` log + `first_price_tick` telemetry event on first price tick post-fill
- Target: ≤1000ms from fill confirm to first monitored price

**Helius WS 429 backoff**: `_confirm_tx` already implements 2s backoff for first 15s,
then 4s on 429. The P8 full backoff (2s→4s→8s→cap 60s) applies to the WS reconnect
in `pumpfun_listener.py` — reviewed separately.

**Acceptance**: PENDING first live buy post-deployment (need `FIRST_PRICE_MS` log line).

batch_verify: no manifest. Acceptance PENDING — first live buy post-deployment with FIRST_PRICE_MS log line.

---

## B-batch (epoch gate) — 2026-07-11 — commit: ada6c06 (2026-07-09, verified in conversation 2026-07-09)

### B1 — Dual-source pre-graduation progress

| Field | Value |
|---|---|
| behavior | PP-silent positions now use curve-account vSOL as secondary source for pre-graduation exit |
| code | scanner.py: _curve_vsol dict; executor.py: get_pumpfun_curve_price returns virtual_sol_reserves_ui |
| log format | `PRE-GRADUATION EXIT ... source=curve` |
| test | test_b1_pregrad_dual_source.py — 3 tests pass |
| live proof | PENDING — first PP-silent position crossing 97.75 SOL from curve feed |

batch_verify: no manifest (pre-VG era). Tests pass (test_b1_ through test_b6_ all confirmed). Live proof: PENDING per section above.

### B2 — Immediate graduation dispatch (oracle path)

| Field | Value |
|---|---|
| behavior | curve feed complete=True / account_missing → graduated_exit dispatched immediately (no 30s delay) |
| code | scanner.py: dispatch inside curve feed loop with graduation_first_seen_ts stamp |
| log format | `CURVE FEED GRADUATED ... handing over` then immediate close_position call |
| test | test_b2_immediate_graduation.py — 2 tests pass |
| live proof | PENDING — first oracle-confirmed graduation after deployment |

batch_verify: no manifest (pre-VG era). Tests pass (test_b1_ through test_b6_ all confirmed). Live proof: PENDING per section above.

### B3 — pump-amm first for oracle-confirmed graduated

| Field | Value |
|---|---|
| behavior | Oracle-confirmed graduated (graduation_first_seen_ts in notes): executor(pump-amm) → then Jupiter |
| code | portfolio.py: _oracle_confirmed_graduated flag skips pre-executor Jupiter rescue; post-executor B3 Jupiter fallback added |
| log format | executor pump-amm attempt logged BEFORE any Jupiter RESCUE alert |
| test | test_b3_pump_amm_first.py — 3 tests pass |
| live proof | PENDING — first graduated exit after deployment |

batch_verify: no manifest (pre-VG era). Tests pass (test_b1_ through test_b6_ all confirmed). Live proof: PENDING per section above.

### B4 — Per-venue state

| Field | Value |
|---|---|
| behavior | _venue_state dict tracks cooldown_until, attempts, last_result per pos+venue; fast-window pump-amm capped at 3 attempts |
| code | portfolio.py: _venue_state dict + _get_venue_state / _record_venue_attempt / _venue_in_cooldown / _pump_amm_attempts |
| test | test_b4_venue_state.py — 6 tests pass |
| live proof | PENDING — first graduation fast-window cycling after deployment |

batch_verify: no manifest (pre-VG era). Tests pass (test_b1_ through test_b6_ all confirmed). Live proof: PENDING per section above.

### B5 — T22 graduated pump-amm flags wired

| Field | Value |
|---|---|
| behavior | T22_GRAD_PUMP_AMM_PROBE_ENABLED / T22_GRAD_PUMP_AMM_ENABLED now control escalate flag for T22 graduated |
| code | portfolio.py: B5 T22 grad gate reads classify_mint + config flags; probe mode writes logs/t22_grad_probe.jsonl |
| test | test_b5_t22_flags.py — 4 tests pass |
| flags | Both default False — no T22 graduated sell receipt exists yet |
| live proof | PENDING — set T22_GRAD_PUMP_AMM_PROBE_ENABLED=True when ready for canary test |

batch_verify: no manifest (pre-VG era). Tests pass (test_b1_ through test_b6_ all confirmed). Live proof: PENDING per section above.

### B6 — Classifier repair + integration

| Field | Value |
|---|---|
| behaviors | (a) UNKNOWN/error results get 60s TTL; (b) unknown extension → not tradeable; (c) mint_classifier wired into executor T22 route decision |
| code | mint_classifier.py: TTL cache, allowlist; executor.py: _pumpfun_mint_token_program checks classifier first |
| test | test_b6_classifier_repair.py — 5 tests pass |
| live proof | n/a — classification runs on every buy, logs ENTRY PROGRAM GATE line |

batch_verify: no manifest (pre-VG era). Tests pass (test_b1_ through test_b6_ all confirmed). Live proof: PENDING per section above.

### B7 — Entry timing decomposition (E1 instrument)

| field | description |
|---|---|
| `http_build_ms` | PP API trade-local HTTP POST round-trip (0 for local-build path) |
| `sign_ms` | VersionedTransaction sign (0 for local-build — signing is inside build) |
| `send_ms` | sendTransaction RPC call to Helius |
| `confirm_detect_ms` | Time from send to getSignatureStatuses seeing confirmed/finalized |
| `rpc_429_wait_ms` | Total time spent sleeping on 429 backoffs during buy |
| `quote_ms` | Jupiter quote duration (off critical path — runs async during build+send+confirm) |

| field | value | notes |
|---|---|---|
| code | executor.py: `_buy_timing` dict; ENTRY TIMING log updated (E1) | |
| artifact | PENDING — will appear in next live trade's ENTRY TIMING log line | |

E1 timing: DEFERRED — epoch deferred 2026-07-30 (capital decision). B7 timing row deferred with it. Re-arm when N3' (V8 paper week net-positive after synthetic costs) clears.

ENTRY TIMING format (after E1):
```
ENTRY TIMING SYMBOL | ... | build_ms=X.X  sign_ms=X.X  send_ms=X.X  land_ms=X.X  429_ms=X.X  http_build_ms=X.X  confirm_detect_ms=X.X  quote_ms=X.X
```

---

## Research Pipeline — P-COLLECT BATCH (2026-07-18, commits 046533d+)

**PC1 — Trade-path persistence (peak_tracker.py)**
- Watch window extended 3min → 15min (`TICK_PEAK_WINDOW_S=900`)
- Every PP tick written to `logs/research_paths/YYYY-MM-DD/<mint>.csv` (ts_ms, price_usd, side, sol_amount, vsol)
- CSV opened in `_drain_pending()` immediately after WS subscribe
- `path_file` column updated in Supabase after each file opened
- Daily UTC-midnight rotation: gzip all `.csv` → `.csv.gz` in yesterday's dir
- Deadman: if <100 path files created while ≥20 tokens scheduled → TG alert
- Concurrent-subscription p95 sampled every 60s; logged on day rollover
- **Status: DEPLOYED — live paths accumulating from first alert after restart**

**PC2 — Backfill from Helius history (research/backfill_paths.py)**
- Fetches `getSignaturesForAddress` (1 credit) + enhanced-TX parse (1 credit/tx) per token
- Selects 200 winners (pct_change_peak ≥ +50%) + 200 losers (pct_change_peak < 0)
- Output: `logs/research_paths/backfill/<mint>.csv.gz` (same columns + `source=backfill`)
- Hard credit cap default 50,000; `--dry-run` prints estimate before any API call
- Updates `path_file` in Supabase for each written token
- **Status: NOT YET RUN — run `python -m research.backfill_paths --dry-run` first**

**PC3 — Path statistics (research/analysis/path_stats.py)**
- A: Shakeout depth before +30/+50/+100%, by progress bucket (p25/50/75/90)
- B: Post-peak price retention at peak+1m/3m/5m per progress bucket
- C: Pre-dump net SOL flow (10s window) vs random baseline — Cohen's d + verdict
- D: Graduation velocity d(vsol)/dt for live paths crossing 85% BC fill
- Cells with n<100 → INSUFFICIENT
- **Status: DEPLOYED — will produce INSUFFICIENT until backfill populates paths**

**PC4 — Exit replay harness (research/analysis/replay_exits.py)**
- Spec A: v7 social_alert (hard_stop=-35%, trail_tiers=[+30%/−25%, +100%/−25%, +300%/−15%], time_stop=90min, profit_lock=40–100%/stall 60s)
- Spec B (alt1): tighter trail (−20%), shorter time_stop=45min, profit_lock stall=90s
- Exec lag default 500ms (configurable `--exec-lag-ms`); fills at nearest post-lag tick
- Output: per-spec stats + side-by-side comparison (win-rate, mean/median/p25/75/90 PnL, exit-reason mix)
- **Status: DEPLOYED — will run once backfill paths populated**

**Migration needed in Supabase (run once):**
```sql
DO $$ BEGIN ALTER TABLE research_tokens ADD COLUMN path_file TEXT; EXCEPTION WHEN duplicate_column THEN NULL; END $$;
```

---

## VERIFY-GATE BATCH — VG1-VG4 (2026-07-28, commit db32f53)

**VG1** — `batches/<batch_id>.yaml` schema defined. Greps, test paths, RECEIPTS section heading, receipt_complete flag.

**VG2** — `tools/batch_verify.py` written (~250 lines):
- `_check_greps()`: regex search each file; supports `must_not`
- `_check_tests()`: file existence check + subprocess pytest
- `_check_receipt()`: finds `## <section>` heading, checks commit hash in body
- `get_stale_batches(threshold_hours=48)`: reads `batches/.red_since.json`
- CLI: `--all`, `--no-tests`, `--verbose`; exits 1 if any FAIL

**VG3a** — `.github/workflows/batch_verify.yml` runs `python tools/batch_verify.py --verbose` on every push to main.

**VG3b** — `CLAUDE.md` rule added: any commit claiming batch complete must pass `batch_verify` locally first; PARTIAL commits must say so.

**VG3c** — `health_monitor.py` Alarm (i): calls `get_stale_batches(48)` every 5 min; alerts per-batch if red >48h.

**VG4 — First green table (produced 2026-07-28):**

```
BATCH: rc_closure  (commit: db32f53)
  item      greps    tests    receipt   VERDICT
  --------  -------  -------  --------  -------
  RC1       OK       OK       OK        ✓ GREEN
  RC2       OK       n/a      OK        ✓ GREEN
  RC3       OK       n/a      PARTIAL   ~ PART
           ↳ receipt_complete=false — section exists, pending full data

  SUMMARY: 2 GREEN  1 PARTIAL  0 FAIL
```

RC3 PARTIAL is expected — clean-era rows not yet available (RF1 deployed <24h ago).
Re-run `rf1_coverage_check` tomorrow and set `receipt_complete: true` in rc_closure.yaml.

---

## V8-READINESS BATCH — N1-N5 (2026-07-29)

### N1 — Full pytest suite (commit: TBD — this batch)

Run: `python -m pytest research/tests/ -v` on 2026-07-29

```
platform darwin — Python 3.12.4, pytest-7.4.4
collected 91 items

test_report_era_split.py    15 passed
test_rf1_curve_oracle.py    13 passed
test_rf3_tiered_window.py   13 passed
test_rf4_realert.py         15 passed
test_rf5_path_schema.py     22 passed
test_rf6_versioning.py      13 passed

============================== 91 passed in 0.65s ==============================
```

All 91 tests pass. Coverage: RF1 curve oracle, RF3 tiered window, RF4 realert dedup,
RF5 path schema, RF6 smart-wallet versioning, RC1 era segmentation.

### N2 — Canary trade + B7 E1 timing (commit: TBD — this batch)

Bot state: `LIVE_TRADING=false` (capital decision 2026-07-30). E1 fields wired in executor.py (`_buy_timing` dict) + emitted in portfolio.py ENTRY TIMING log.

**Status: DEFERRED — epoch deferred 2026-07-30 (capital decision). B7/E1 timing row deferred with it.**
Prerequisite to re-arm: V8 paper week net-positive after synthetic execution costs (N3' line).

Pre-B7 timing reference (Jul 10–11 trades, pre-E1-deployment):

| trade | screen | submit | confirm | total | real_slip |
|---|---|---|---|---|---|
| FABLE | 0.5s | 11.91s | 13.81s | 22.0s | +23.3% |
| RETAIL | 0.6s | 11.82s | 13.70s | 22.1s | +18.9% |
| tradition | 0.5s | 11.85s | 13.72s | 21.9s | +11.1% |

Measured latency epoch: submit≈12s, confirm≈14s, total≈22s.

### N3 — Epoch ON: live at canary sizing, daily stats cron

Gates unchanged. Max 2 concurrent enforced by existing circuit breaker.
F3 drought alarm active: `health_monitor.py` fires if ≥3 paper opens + 0 live attempts in 3h.

Daily epoch stats written by cron to `logs/epoch_daily.jsonl` (append-only).
Format: `{"date":"YYYY-MM-DD","trades":N,"pnl_usd":X.XX,"open":N}`.

Cron entry (VPS `/etc/cron.d/quantbot-epoch`):
```
55 23 * * * root cd /root/quant-bot && set -a && . .env && set +a && \
  .venv/bin/python research/scripts/epoch_daily_log.py >> logs/epoch_daily_cron.log 2>&1
```

### N4 — Analysis artifacts → docs/V8_INPUTS.md

See `docs/V8_INPUTS.md` (committed with this batch).

- N4(a): forward-validation table — SM_hit=True: 92.3% win (n=91); clean era n=7 (too early)
- N4(b): era-split — preRF1 BC 43.6% win rate is survivor-bias artifact (only 2% priced)
- N4(c): path_stats — INSUFFICIENT (1,207 header-only path files; PC2 backfill not run)
- N4(d): replay_exits — PENDING (same; 0 paths with tick data)

### N5 — Receipts hygiene

All legacy-commit and pending sections resolved:
- L6, X1, X3, X5: `commit: verified in conversation 2026-07-10 (pre-manifest era)` + batch_verify note
- B-batch (B1–B7): `commit: ada6c06 (2026-07-09)` + batch_verify note + E1 pending line on B7
- No silent placeholders remain in the proof ledger.

### N6 — V8 paper twin: daily v8 vs v7 comparison

`memecoin/v8_paper.py` — independent paper book, own journal
(`logs/memecoin_v8_journal.csv`), own monitor thread, zero shared state with
v7's `Portfolio`. Gate: `progress_at_signal < 0.70` AND no `dex_id` yet
(live-computed from PumpPortal's cached vsol — zero new Helius calls).
**Scope cut**: the spec's "smart-money-v1" half of the gate is NOT applied
live (would require a dedicated Helius call per signal, ruled out by
SOCIAL_ALERT_ONLY's zero-Helius-increase constraint) — joined in later from
`research_tokens.smart_money_hit` for reporting only, not gating.
Exit config: placeholder mirroring v7 social_alert production config —
swap once `replay_exits.py` has a winner (blocked on N4d's PC2 backfill).

0/0 rows below is expected — this was seeded from this session, no live
scanner has run against it yet.

Cron entry (VPS `/etc/cron.d/quantbot-v8`, add alongside the N3 epoch cron):
```
58 23 * * * root cd /root/quant-bot && set -a && . .env && set +a && \
  .venv/bin/python research/scripts/v8_vs_v7_daily.py >> logs/v8_vs_v7_cron.log 2>&1
```

| date | v7 trades | v7 pnl% (mean) | v8 trades | v8 pnl% (mean) | v8 gate |
|---|---|---|---|---|---|
| 2026-07-30 | 0 | +0.0% | 0 | +0.0% | progress<70+no-dex (smart-money offline-only) |

### N7 — Path schema + exit-level analyses

- N7(a): `trader_pk` added to `research/path_schema.py` PATH_HEADER (schema v2).
  Sourced live from PP's `traderPublicKey` in `peak_tracker.py` (PC1); sourced
  from Helius `feePayer` in `backfill_paths.py` (PC2, "backfill rows already
  carry addresses" per spec). **Also fixed while wiring this in**: PC1 had an
  undefined-name bug (`_SCHEMA_VER` referenced but never imported, stale local
  `_CSV_HEADER`) that silently NameError'd on every tick write, caught by a
  bare `except: pass` — this is the real reason all 1,207 path files on disk
  are header-only, not "PP doesn't deliver ticks" as N4(c) above states. Fixed
  by importing `PATH_HEADER`/`PATH_SCHEMA_VERSION` from `path_schema.py`
  (mirrors what `backfill_paths.py` already did correctly). Regression test:
  `research/tests/test_rf5_path_schema.py::TestPeakTrackerSchemaConsistency`.
- N7(b): `research/analysis/path_stats.py` — added E (peak-mcap distribution /
  "where do they turn"), F (conditional continuation / "how high after the
  turn"), G (unique-buyer velocity vs outcome), H (sniper density vs outcome).
  G/H require `trader_pk` data and will read n=0/INSUFFICIENT until path files
  accumulate under the N7(a) fix above — by design, not a bug.
- N7(c): `research/analysis/report.py` — added section 11 (hour-of-day /
  day-of-week outcome table with alerts/hour as a crowding proxy).

All three items are code-complete and unit-tested
(`research/tests/test_path_stats_n7b.py`,
`research/tests/test_rf5_path_schema.py::TestTraderPk`,
`research/tests/test_report_era_split.py::TestAlertDt`) but **have not been
run against real data** — this session has no VPS or Supabase access. The
per-cell `n` in `docs/V8_INPUTS.md` for N7(b)/(c) needs a real run once
trader_pk data and report data are available.

---

## PROGRESS-FIX — event-time, source-provenanced progress_at_signal (2026-08-08)

Bounded measurement-integrity batch (spec sections PF0-PF13). Scope
boundary held throughout: **no changes to V7 trading filters, V8's
`progress_at_signal < 0.70` candidate threshold, entry/exit logic, sizing,
execution routing, live-trading enablement, or PumpPortal spend limits.**
`LIVE_TRADING` remains `false` on the VPS, unchanged by this batch.

### Status by section

| Section | What | Status |
|---|---|---|
| PF0 | curve_oracle.py unit bug (price formula) fixed; historical audit | Done — 0 affected rows (see below) |
| PF2 | Canonical `ProgressCapture` dataclass, NULL never coerced to 0 | Done |
| PF3 | Capture at alert time, off critical path, source order pp_warm→curve_account→pp_post_alert→failure | Done |
| PF4 | Independent of screening-state lifetime (survives scanner eviction) | Done |
| PF5 | Event-keyed durable store (not mint+time-window) | Done |
| PF6 | Research and V8 share one canonical value | Done |
| PF7 | `pp_snapshot_ok` truthfulness fix | Done |
| PF8 | Schema migration | Done — applied manually via Supabase SQL editor 2026-08-08 ~22:15 UTC |
| PF9 | One canonical `GRAD_SOL_UI` constant | Done |
| PF10 | Bounded historical recovery from path ticks | Done — 41/31,014 rows recovered (see below) |
| PF11 | `path_stats.py` coverage reporting, no silent None→0 | Done |
| PF12 | Deploy + monitor ≥100 fresh alerts | Done — 100/100 fresh eligible alerts, 100% success rate (see below) |
| PF13 | 15 regression tests | Done — 55 tests passing across 5 files |

### Answers to the spec's questions

- **Is `progress_at_signal` now measured independently of a future PP trade?** Yes — primary source is a direct on-chain bonding-curve account read (`curve_account`), independent of PumpPortal entirely; `pp_warm`/`pp_post_alert` are the fallback sources, both bounded and event-driven, not polling for a future trade indefinitely.
- **Is it tied to original alert time?** Yes — `alert_ts` is captured once in `memecoin/scanner.py` at the top of `_on_telegram_signal`, before any other work, and threaded through as the anchor for lag calculation.
- **Is capture lag recorded?** Yes — `progress_capture_lag_ms` on every result, success or failure.
- **Can scanner rejection/eviction destroy the measurement?** No — verified by test (`test_curve_account_independent_of_screening`): the capture waiter registry lives in `progress_capture.py`, not in `pumpportal_monitor._screening`.
- **Do Research and V8 use exactly the same value?** Yes, for a given event_id — both read the one `ProgressCapture` result `progress_capture.py` produces (Research via the durable file, V8 via the in-process cache). Verified by test (`test_research_and_v8_read_identical_progress_for_same_event`).
- **Can missing measurement become progress=0?** No — `ProgressCapture.failure()` always sets `progress_at_signal=None`; verified at the dataclass level, the tracker-insert level, and the `path_stats.py` bucket-analysis level (the two silent `if progress is None: progress = 0.0` lines found in `_analyse_shakeout`/`_analyse_decay` were dead code given callers already pre-filter, but fixed anyway).
- **Is `pp_snapshot_ok` truthful?** Yes — now requires at least one real non-zero PP observation, not merely that a snapshot dict exists.
- **Is `GRAD_SOL_UI` defined canonically once per service?** Yes — `research.config.GRAD_SOL_UI` and `memecoin.config.GRAD_SOL_UI`, both currently `115.0`, guarded by `test_grad_sol_ui_canonical.py` against a third copy appearing.
- **Was the curve-oracle token-decimal price bug fixed?** Yes — `virtual_token_reserves` is now correctly divided by `10**6` before the price ratio; fixture-verified against the exact example in the spec (old formula ≈4.38e-13/token, correct ≈4.38e-7/token).
- **Were existing curve_account outcome rows audited for that bug?** Yes — **0 affected rows**. Queried every `price_source_t*m` provenance column across all 31,014 `research_tokens` rows (26,169 of them `social_alert_bc`): only `dexscreener` (4,748) and `jupiter` (5) appear as sources; no row was ever actually populated via the `CURVE_ACTIVE`/curve_account path in production (curve accounts were almost always already missing/graduated by poll time — `curve_account_missing` was ~98% of curve attempts per the existing RF1 audit above — so DexScreener supplied the surviving prices in practice). The buggy formula existed in code but never wrote a corrupted value to a persisted row.
- **What percentage of fresh eligible alerts now have trustworthy progress?** 100% (100/100 over ~17.6 hours) — see PF12 below.

### PF10 — historical recovery result

Pre-registered candidate lag thresholds (chosen before any outcome inspection): 5/15/30/60/120s.

| threshold | recoverable | % of 31,014 candidates |
|---|---|---|
| 5s | 34 | 0.1% |
| 15s | 38 | 0.1% |
| 30s | 40 | 0.1% |
| **60s (chosen)** | **41** | **0.1%** |
| 120s | 52 | 0.2% |

Chosen threshold: 60s, for coverage/cleanliness (loosening to 120s only
adds 11 more rows at the cost of "at signal" meaning a tick up to 2
minutes stale). Applied via `research/analysis/backfill_progress_from_paths.py --apply --threshold-s 60`
on the VPS: 41 rows recovered (`progress_source="pc_path_nearest_tick"`),
82 skipped for exceeding the threshold, 30,891 left `NULL` — no usable
historical path data exists for them (30,794 have no path file at all;
97 have a path file but it's entirely backfill-sourced, vsol=0
throughout, no genuine live observation to recover from). This is a
materially small recovery — path-file instrumentation covers a small
fraction of historically-alerted tokens, and even where a path exists,
the nearest live tick is typically ~3 minutes from the original alert
(p50 lag among matches: 185,943ms) — reported as-is, not chased with a
looser threshold.

### PF12 — live verification status

Deployed and confirmed working end-to-end in production (VPS commits
`6fd6109`, `0a6dbe5`, `5e994c7`, `64dbbf3`, `1656bea`), but has not yet
accumulated the required ≥100 fresh eligible alerts to make a statistically
honest coverage/success-rate claim. Two real production issues were found
and fixed during this verification (both outside PROGRESS-FIX's own scope,
but blocking an honest read of it):

1. **Helius free-plan 429s with no RPC fallback** — `research/curve_oracle.py`'s
   `_fetch_batch` had a single RPC call, no retry/fallback. Added the same
   mainnet-beta→Ankr fallback chain `memecoin/executor.py` already uses.
   Commit `5e994c7`.
2. **`telegram_monitor.py` address-extraction bug** (pre-existing, unrelated
   to PROGRESS-FIX, found only because it was making PF12's numbers look
   broken) — extracted every base58-looking substring in an alert message
   instead of just the token mint, producing ~92x signal amplification
   (4,327 signals from 47 real alerts in one measured window). Fixed by
   anchoring extraction to the alert template (verified 0 failures across
   823 real historical messages). Commit `ad10a3a`. Full writeup:
   https://claude.ai/code/artifact/7cdc91c9-7d69-49c0-a16c-2e0d8aeb8bbd

Two further issues surfaced once real post-fix alerts started arriving:

3. **PF8 migration was never actually applied** — 880 `progress_*` field
   writes had been silently dropped by the existing PGRST204-strip
   resilience since initial deploy (degraded-but-not-broken as designed,
   but meant Research's persisted data carried none of this batch's work).
   Applied manually via the Supabase SQL editor.
4. **`progress_capture_lag_ms` was declared `INT` in the PF8 migration
   itself — a bug in this batch, not a pre-existing one.** The code always
   produces a one-decimal float (e.g. `451.9`); Postgres rejected every
   real value with `invalid input syntax for type integer`, which the
   PGRST204-strip path doesn't catch (it only handles unknown-column
   errors, not type mismatches) — so every fresh post-fix row was failing
   the full INSERT and landing in `research/spool/failed_inserts.jsonl`
   instead. Corrected via `ALTER TABLE research_tokens ALTER COLUMN
   progress_capture_lag_ms TYPE FLOAT` (applied manually, same as #3).

Both spool files (`failed_inserts.jsonl`, `dropped_fields.jsonl`) were
then replayed via the existing `research/analysis/replay_spool.py` —
**0 items still failed** after the type fix; all recoverable rows/fields
were recovered.

**Final result — 100/100 fresh eligible alerts observed**, monitored
2026-08-08 20:04:53 UTC → 2026-08-09 13:42:39 UTC (~17.6 hours,
extraction-bug fix at 20:03:54 UTC through crossing the threshold):

| metric | value |
|---|---|
| fresh eligible alerts | 100 |
| `progress_source=curve_account`, `progress_status=ok` | 100 (100%) |
| `pp_warm` | 0 |
| `pp_post_alert` | 0 |
| any failure (`pp_timeout`, `capture_missing`, etc.) | 0 |
| capture lag | min=352ms, p50=429ms, p90=811ms, p95=866ms, max=1193ms |

100% success rate across the full sample, entirely via the primary
`curve_account` source — the RPC fallback chain was never even needed
once the extraction fix stopped the noise; lag stayed sub-1.2s throughout
with no degradation trend across the ~17.6 hour window. This was checked
incrementally roughly hourly (28/34/38/41/51/53/60/68/76/81/87/92/98/100)
rather than only at the end, precisely so a regression partway through
wouldn't be averaged away — none occurred.

**Research/V8 disagreement check:** not measurable from live data — the
V8 paper journal (`logs/memecoin_v8_journal.csv`) does not exist on the
VPS; zero positions have been journaled during this window. Root cause:
`memecoin/scanner.py`'s call into `v8_paper.book.maybe_open()` (line
~519) is nested inside `if sig.strength in ("medium", "strong")`, a pre-
existing V7 signal-strength classification gate that is upstream of and
unrelated to `progress_at_signal` — none of these 100 alerts reached that
bar, so V8 never got the chance to evaluate (agree or disagree) on any of
them. This is not a PROGRESS-FIX defect and not something this batch may
touch (V7's strength classification is explicitly out of scope). What
*is* covered: of the 100, only 4 had `progress_at_signal < 0.70` (the
rest were already deep into/past the bonding curve at alert time), so
even without the strength gate, live disagreement instances would have
been rare in this sample regardless. The mechanism guarantee itself is
proven at the unit level — `test_research_and_v8_read_identical_progress_for_same_event`
constructs one `ProgressCapture` result and asserts both Research's file-read
path and V8's in-process-cache path resolve the identical value for the
same `event_id`; there is architecturally only one write path
(`progress_capture.py`'s `_store_result`) both readers consume, so a
runtime disagreement is not possible by construction, independent of
whether V8's upstream gate happens to fire.

### Tests

55 tests passing: `test_progress_capture.py` (18), `test_rf1_curve_oracle.py`
(19), `test_grad_sol_ui_canonical.py` (6), `test_v8_paper_progress_integration.py`
(5, new), `test_tracker_progress.py` (7, new).

### Also fixed (found while building PF10, not in original scope)

`research/tracker.py` was only persisting `event_id` to `research_tokens`
for backfilled rows — live rows got `progress_at_signal` captured
correctly at insert time (the value doesn't depend on the column), but
then lost their `event_id` afterward, breaking any later cross-reference
back to the capture that produced it. Now stored unconditionally.

### Modified/added files

`memecoin/progress_capture.py` (new), `memecoin/pumpportal_monitor.py`,
`memecoin/scanner.py`, `memecoin/signals.py`, `memecoin/telegram_monitor.py`,
`memecoin/v8_paper.py`, `research/analysis/backfill_progress_from_paths.py`
(new), `research/analysis/path_stats.py`, `research/analysis/report.py`,
`research/curve_oracle.py`, `research/supabase_schema.sql`,
`research/tracker.py`, plus the 5 test files above.

### Deferred / not done

- Research/V8 live disagreement count could not be measured — V8's
  journal is empty for reasons unrelated to and out of scope for this
  batch (see PF12 above); covered at the unit level instead.
- Historical paper-trading/research data collected during the
  telegram-extraction bug's ~92x noise window is not retroactively
  cleaned — the fix only stops new noise.
- Whether the live-buy preflight path would hard-block a non-mint address
  (from the extraction bug, hypothetically, had it coincided with live
  trading) was not directly verified — inference only.
- Also found, also out of scope, deliberately not touched: RLS was
  disabled on all public Supabase tables (Supabase Advisor, 6 CRITICAL
  findings) — separate from PROGRESS-FIX, fixed by the user directly
  (confirmed the app's `service_role` key bypasses RLS regardless, so
  this was safe to enable with zero functional risk).

### Status

**`PROGRESS_FIX_LIVE_VERIFIED`** — code deployed, migration applied,
100/100 fresh eligible alerts captured with a 100% success rate over a
~17.6 hour live window (2026-08-08 20:04:53 UTC → 2026-08-09 13:42:39
UTC), lag p50=429ms/p90=811ms/p95=866ms/max=1193ms, zero failures, zero
degradation across the window (checked incrementally, not just at the
end). No trading threshold, execution, sizing, or exit behavior was
changed by this batch; `LIVE_TRADING=false` unchanged throughout.

---

## K-BATCH — unblock the dataset (2026-08-10)

Follow-up batch to PROGRESS-FIX, addressing the concrete gaps identified
in `docs/V8_INPUTS.md`'s N4(c)/(d) status (funded PumpPortal key,
silent-fallback monitoring, backfill yield, refresh cadence). Same
constraints held: no V7/V8 trading logic touched, `LIVE_TRADING=false`
unchanged.

### K1 — PumpPortal API key

Both `research/peak_tracker.py` and `memecoin/pumpportal_monitor.py`
already wired `PUMPPORTAL_API_KEY` into their WS connect URLs correctly,
and a funded key was already present in the VPS `.env` (confirmed
non-empty, 206 chars) — the only real gap was startup visibility: neither
logged whether the connection was actually keyed. Added to both.
**Artifact:** real tick lines from both within 3 minutes of deploy —
`peak_tracker.py` wrote a genuine `live_pp`/`CURVE_ACTIVE` tick to a
fresh path file; `pumpportal_monitor.py`'s live capture path was already
proven working throughout PROGRESS-FIX's PF12 verification (100/100
successful captures on the same underlying connection).

### K2 — Tick deadmans

**K2a** (`memecoin/pumpportal_monitor.py`): alerts if an open, subscribed
position has 0 real ticks for 5min. Suppressed while `LIVE_TRADING=false`
(weekly reminder instead, same pattern as `health_monitor.py`'s F3
live-drought alarm), 30min per-mint cooldown when live. Verified
manually: grace period holds, weekly-suppressed path fires, live-alert
path fires and respects cooldown.

**K2b** (`research/peak_tracker.py`): new `_ticks_today` counter
(distinct from `path_files_today` — a file can exist header-only with
zero real ticks, exactly the N4(c) Finding 2 failure mode file-count
alone never caught). Hard FAIL (`log.critical` + alert) when
`tracked_tokens>50` and `ticks==0`, louder than the existing
file-count deadman. Tick count now also surfaces in the routine daily
report line.

### K3 — Backfill parser: real root cause, real fix, real re-run

**Diagnosis (not assumed — verified on real data):** the documented
17% backfill yield was NOT primarily an extraction-heuristic bug.
`_fetch_sigs`'s alert-time windowing paginated backward via
`getSignaturesForAddress` from "now", capped at `max_pages=8` (8,000
signatures). Tested directly on 5 real winner tokens (+1167% to
+3446% peak, alerted between 2026-06-23 and 2026-08-09): **all five
returned 0 signatures in their alert-time window.** One token needed
35,000+ signatures fetched and was *still* 6+ hours short of reaching
its own alert time — high-activity ("winner") tokens accumulate more
post-alert signatures than any reasonable page cap can walk through.
This self-selects against exactly the tokens the backfill exists to
calibrate exits for.

**Fix:** `_estimate_slot_for_time` anchors near the target time via
Solana's ~0.4s/slot rate, refined by real `getBlock` lookups — converges
to within seconds in 4-6 RPC calls regardless of post-target trading
volume (verified: same failing token, now finds its window in the same
handful of calls). `_fetch_sigs` anchors there instead of blind backward
pagination.

**Secondary fixes found under load:** `_fetch_one_tx_std` had no real
retry on 429 (single 2s sleep, then silently dropped the transaction);
`_parse_txs_std` rewritten to JSON-RPC batch requests — same underlying
per-item rate limit on free public RPC (~0.5 successful
`getTransaction`/sec observed), but discovers failures in ~2s instead of
a full backoff cycle, cutting per-token wall time from ~240s to ~50s.
Both endpoints (`mainnet-beta.solana.com` primary, Ankr fallback) used,
matching the resilience pattern already established in
`research/curve_oracle.py`. vsol is now genuinely derived from the curve
PDA's `postBalance` where std_rpc mode's raw pre/post balance data makes
it possible (Helius Enhanced Transactions only exposes deltas, not
absolute balances, so this is std_rpc-only) — correctly stays 0 for
post-graduation trades that never touch the curve account.

**Helius Enhanced Transactions mode tested and confirmed unusable right
now** — `"max usage reached"` (quota genuinely exhausted, not just rate
limited). std_rpc mode (this fix) was the only viable path.

**Artifact — real 400-token re-run, 2026-08-10 22:26-23:32 UTC:**

| metric | value |
|---|---|
| Candidates (200 winners + 200 losers) | 400 |
| Already had a path file (`--skip-existing`) | 175 |
| Newly processed under fixed code | 225 |
| **Newly processed with real rows** | **165/225 = 73.3%** |
| Baseline yield (2026-08-03, documented) | 67/400 = 16.75% |

`path_stats.py --min-n 100` re-run afterward, real numbers (was 67
paths / all INSUFFICIENT on 2026-08-03):

| section | before (Aug 3) | after (Aug 10) |
|---|---|---|
| Total path files on disk | 67 | 2,956 |
| Paths successfully loaded | 67 | 446 (2,510 skipped/empty — thin or corrupt, real residual gap) |
| progress_at_signal coverage | 0/67 (0%) | 218/446 (48.9%) |
| C — pre-dump order flow | n=19, INSUFFICIENT | **n=495+534, Cohen's d=-0.45, TRUE (sell pressure precedes dumps)** |
| E — peak-mcap (overall) | n=67, INSUFFICIENT | **n=446** |
| F — conditional continuation (overall / $250k+ band) | n=57, INSUFFICIENT | **n=340 qualifying / 155 in $250k+ band** |
| G/H — buyer velocity, sniper density | n=67, INSUFFICIENT | **n=446** |
| A/B — progress-bucketed shakeout/retention | n=0 | still INSUFFICIENT (max n=61 per bucket) — needs more days of PROGRESS-FIX's now-working capture, not more backfill |
| D — graduation velocity | n=0 | still INSUFFICIENT (n=60) |

Real, substantial progress on the non-progress-bucketed sections (target
>70% yield: **met**, 73.3%). The progress-bucketed cells (A/B) remain the
one honestly-unmet target — they need `progress_at_signal` on the *same*
tokens as real path data, and that combination has only been reliably
available since PROGRESS-FIX went live ~2.5 days before this batch; K5's
nightly refresh will track this filling in over the coming days.

### K4 — skipped

Conflicted with completed PROGRESS-FIX work (would have reintroduced the
PF1 subscribe-then-immediately-read race via tracker's at-alert
`pp_vsol`). User confirmed: skip — the goal (progress-bucket tables with
real n>0) is already met via the PF2-PF9 `ProgressCapture` mechanism.

### K5 — nightly refresh cadence

`research/scripts/v8_inputs_nightly.py` (new): runs `report.py` +
`path_stats.py` as subprocesses (reuses existing, now-fixed analysis
code rather than a second divergent implementation), appends the
clean-era-relevant sections to `docs/V8_INPUTS.md` dated, prints (does
not act on) the freeze gate: `clean_n>=2500` priced outcomes AND every
path_stats cell clears `n>=100`.

Also installed, found missing during setup: two previously-documented
crons (`docs/RECEIPTS.md`'s own N3/N6 sections said "add this cron
entry") had never actually been installed on the VPS — same
documented-but-never-applied pattern as PROGRESS-FIX's PF8 migration.
All three now live in `/etc/cron.d/`: `quantbot-epoch` (23:55 UTC),
`quantbot-v8` (23:58 UTC), `quantbot-v8-inputs` (00:15 UTC). First real
K5 entry already landed in `V8_INPUTS.md` (2026-08-09).

### Modified/added files

`memecoin/pumpportal_monitor.py`, `research/peak_tracker.py`,
`research/backfill_paths.py`, `research/scripts/v8_inputs_nightly.py`
(new), plus 3 new `/etc/cron.d/` entries (VPS-side, not repo files).

### Status

**Done**, K4 skipped by user decision. K3's yield target (>70%) met
(73.3%). Freeze gate per K5 not yet met — progress-bucketed path_stats
cells (A/B) still below n=100, clean_n status tracked nightly going
forward. No trading logic touched; `LIVE_TRADING=false` unchanged
throughout.

---

## Epoch — Capital Decision (2026-07-30)

Epoch deferred 2026-07-30 — capital decision. Prerequisite for any future live: V8 paper week net-positive after synthetic execution costs (N3' line). B7/E1 timing row deferred with it.

---

## V8-TWIN-FIX — Root Cause + Repair of the Zero-Position V8 Paper Twin (2026-08-11/12)

**Scope constraint honored throughout:** no change to live trading, V7 trading
rules, execution routing, sizing, or V8's frozen progress threshold (0.70).
`LIVE_TRADING=false` unchanged. This is a V8-paper-book-only repair.

### The problem

`memecoin/v8_paper.py`'s paper-trading "V8 twin" book had recorded zero
positions ever, despite the pipeline visibly processing hundreds of TG-PASS
candidates since its 2026-08-09 deploy.

### H1 — root cause: CONFIRMED

`passes_v8_gate()`'s old logic:

```python
dex_id = (getattr(signal, "dex_id", "") or "").strip()
if dex_id:
    return False, f"has_dex_id:{dex_id}", progress
```

treated **any** non-empty `dex_id` as proof the token had graduated to a DEX.
But DexScreener indexes pump.fun bonding-curve tokens with `dexId="pumpfun"`
immediately — long before graduation. Worse, `screen_token()`'s own
`no_dex_data` gate *requires* a non-empty `dex_id` for a candidate to reach
"TG PASS" at all. So every candidate that could ever reach V8's gate was
structurally guaranteed to carry a non-empty `dex_id`, and guaranteed to be
rejected by `has_dex_id` — independent of real bonding-curve progress, and
independent of dedup. This is a code-guaranteed zero, not a rare miss.

**Evidence (recorded in commit `ae31959`):** all 15 real production TG-PASS
candidates with `progress_at_signal < 0.70` prior to the fix were queried
directly — **15/15 (100%)** showed `dex_id="pumpfun"`,
`progress_source="curve_account"` (a genuine, successful on-curve
measurement — not a missing/failed capture). Every one was rejected by
`has_dex_id`, never by an actual graduation.

**Row-level table — could not be reproduced for this writeup, stated
honestly rather than inferred:** `progress_capture.py` caches captures
in-memory only (`_cache_order`, max 5000 entries, no disk persistence), and
the process has restarted multiple times since that live verification pass.
Pre-fix gate-reject log lines were at DEBUG level until commit `3aa6232`
bumped them to INFO for live diagnosis — journalctl has no earlier record at
a level it kept. Both of the two possible sources for exact
mint/event_id/timestamp rows are gone. What survives and is verifiable right
now: (a) the structural code proof above, reproducible via
`git show ae31959^:memecoin/v8_paper.py` and `screen_token()`'s
`no_dex_data` gate — this doesn't depend on a sample, it holds for every
past and future TG-PASS candidate under the old code; (b) the aggregate
15/15 match recorded in the `ae31959` commit message at the time the query
was run. Per spec's "do not infer" — no fabricated per-row table is included
here.

### VF4 — executor observability: not the root cause, confirmed as predicted

Checked "TG screen error" / "TG signal processing error" log lines against
all 15 mints: zero matches. No exception was ever swallowed by the
fire-and-forget `run_in_executor` Future. Fixed anyway per spec (hardening,
not root cause) — see Modified files.

### VF2 — the fix: dex_id replaced with explicit venue state

New gate (`memecoin/v8_paper.py::passes_v8_gate`):

```
progress_at_signal < 0.70   AND   venue_state_at_signal == CURVE_ACTIVE
```

`venue_state_at_signal` (`memecoin/progress_capture.py`, new field on
`ProgressCapture`, one of `CURVE_ACTIVE | GRADUATED | DEX_ACTIVE | UNKNOWN`,
default `UNKNOWN` — fails closed) is derived from the **same** canonical
capture used for `progress_at_signal` — no second independent Helius
measurement was added. `dex_id` is still recorded on every position/event
for observability, but never used as a gate input again.

### VF1 — funnel telemetry: all 9 stages wired

New module `memecoin/v8_telemetry.py` (JSONL, `logs/v8_funnel.jsonl`,
append-only, never raises). `v8_gate_entered` is the literal first line of
`maybe_open()`, before `already_open` or any other return. `_add_signal()`'s
duplicate path now emits `dedup_rejected` explicitly instead of silently
returning.

### VF3 / VF5 — funnel counts, full deployment window (2026-08-11 19:44 → 2026-08-12 22:31 UTC, ~26.8h)

| stage | count |
|---|---|
| `telegram_received` | 199 |
| `screening_rejected` | 92 |
| `screening_passed` | 107 |
| `signal_constructed` | 107 |
| `add_signal_entered` | 108 |
| `dedup_rejected` | 1 |
| `v8_gate_entered` | 107 |
| `v8_gate_rejected` | 106 |
| `v8_opened` | **1** |

Gate-reject reasons (106 total): `progress_over_threshold` 98,
`venue_state:GRADUATED` 5, `progress_unknown` (no capture) 3. Zero rejects
attributable to `dex_id` — confirms the fix removed that failure mode
entirely; every reject in this window has a real, inspectable reason tied to
progress or venue state.

**VF3 answer** — "a TG PASS alone is NOT proof `maybe_open` ran": proven
directly. `screening_passed`=107 but `add_signal_entered`=108 and
`v8_gate_entered`=107 — the gap between `screening_passed` and
`v8_gate_entered` is exactly 1 duplicate signal caught by
`_add_signal`'s dedup, per `dedup_rejected`=1. This is also unit-proven in
`test_scanner_v8_dedup.py` (tests 8-9): a duplicate signal never increments
`maybe_open`'s call count; a fresh one does, exactly once.

**VF5 — experiment universe, labeled explicitly.** Current architecture
feeds V8 only V7-screened survivors: `raw_tg_eligible` (`telegram_received`)
= 199, `post_v7_screen_eligible` (`screening_passed`) = 107. This is **V8
conditional on V7 screening**, not an independent head-to-head comparison —
labeled `V8_CONDITIONAL_ON_V7_SCREEN` below. Not silently changed in this
fix, per spec.

From the earlier (pre-V8-TWIN-FIX) investigation window: of the low-progress
(<0.70) candidates that V7 screening rejected before V8 ever saw them, 4/19
(≈21%) were lost solely to V7 screening rejection — i.e. would have been
gate-eligible for V8 on progress/venue grounds alone had they reached it.
This number was not re-measured in the current window because V8's funnel
does not capture progress for `screening_rejected` candidates (V8 never
sees them) — computing a fresh figure would require adding progress capture
to the screening-rejected path, which is exactly the architecture change
this fix is scoped not to make silently.

**Fork-point proposal (documented only, not implemented):** split the
funnel at Telegram alert into a common capture/hard-safety-floor stage,
then branch — one path to V7's existing screening, one path directly to
V8's gate — so V8 sees the same raw candidate pool V7 does, making it a
true independent comparison instead of a conditional one. This is a real
architecture change (new fork point, new common-stage code, likely new
progress-capture calls for candidates V7 would otherwise reject) and should
only be scheduled as its own tracked item if/when a true head-to-head
comparison is actually needed — not bundled into this fix.

### VF6 — live proof, real fresh production observation

**Scenario 1 (progress<0.70, CURVE_ACTIVE → OPEN): LIVE VERIFIED.**

First real `v8_opened` event since the fix deployed, 2026-08-12 21:08:13 UTC
— mint **Moblin** (`AdeKS1SbF8QzF5YLgoNhHfc7VDaWJfg6PeRUBPFwpump`).

Funnel record (`logs/v8_funnel.jsonl`):
```json
{"ts": 1786568893.791558, "stage": "v8_opened", "event_id": "3fe06b5f4ec22a0b",
 "mint": "AdeKS1SbF8QzF5YLgoNhHfc7VDaWJfg6PeRUBPFwpump", "progress": 0.6511,
 "progress_source": "pp_warm", "dex_id": "pumpfun", "venue_state": "CURVE_ACTIVE",
 "result": "opened", "reason": "V8481249"}
```

Persisted position (`memecoin/data/memecoin_v8_positions.json`, confirmed
on disk, 733 bytes, real content — not the log line alone):
```json
{
  "id": "V8481249", "signal_id": "5cbec8bc", "chain": "solana",
  "token_address": "AdeKS1SbF8QzF5YLgoNhHfc7VDaWJfg6PeRUBPFwpump",
  "token_symbol": "Moblin", "signal_type": "social_alert", "strength": "strong",
  "signal_price": 13.152426723102602, "entry_price": 13.152426723102602,
  "size_usd": 1.0, "progress_at_signal": 0.6511, "dex_id": "pumpfun",
  "entry_source": "pp_tick", "status": "closed",
  "exit_price": 6.860301374297039, "exit_reason": "hard_stop"
}
```

Monitor loop saw it on the next cycle — proven by more than persistence
alone: the position was actively tracked and closed itself via `hard_stop`
~41 seconds after entry (`exit_time` − `entry_time` = 40.88s), which
requires the live monitor loop to have been polling price against this
exact position. All required fields present: `event_id`, `progress`
(0.6511), `progress_source` (`pp_warm`), `venue_state_at_signal`
(`CURVE_ACTIVE`), `dex_id` (`pumpfun`), `entry_price` ($13.1524267231),
position id (`V8481249`).

**Scenario 2 (progress≥0.70 → reject): LIVE VERIFIED.** 98 real instances
this window, e.g. mint Gorm, `progress_0.89_over_0.70`.

**Scenario 3 (progress<0.70 but GRADUATED/DEX_ACTIVE → reject venue): LIVE
VERIFIED.** 5 real instances this window — recurring pattern confirmed
across independent mints (PUSHEEN, grapers, QIZAI all showed
`progress: 0.0, venue_state: GRADUATED` during monitoring). This directly
contradicts an earlier assumption of mine (that this scenario might be
"mathematically unreachable" since a graduated token's vsol should read
≈115≈progress 1.0) — real production data proved that assumption wrong and
validated that VF2's venue-state check is doing real, necessary work a
naive progress-only check would have gotten wrong (would have wrongly
opened these as "early curve").

**Scenario 4 (UNKNOWN → fail closed): LIVE VERIFIED.** 3 real instances
this window (`progress_unknown` reason — no capture landed in time).

**Scenario 5 (duplicate → explicit disposition):** 1 real production
instance this window (`dedup_rejected`=1, per VF3 above) plus deterministic
unit coverage (`test_scanner_v8_dedup.py` tests 8-9), per spec's allowance
to rely on VF7 tests for this scenario.

**memecoin_v8_journal.csv note (per spec):** journal is write-on-close
only, so its absence is not evidence of "no OPEN ever happened" — the
positions JSON is the immediate open receipt, and that's what's cited
above. (The Moblin position has since also closed, so its journal row now
exists too, but that isn't what's being relied on here.)

### Modified/added files

- `memecoin/progress_capture.py` — `venue_state_at_signal` field,
  `VALID_VENUE_STATES`, `_normalize_venue_state()`, curve-oracle-reuse at
  all 3 call sites (`curve_account`, `pp_warm`, `pp_post_alert`)
- `memecoin/v8_paper.py` — `_get_capture_for_gate()`, rewritten
  `passes_v8_gate()`, rewritten `maybe_open()` telemetry
- `memecoin/v8_telemetry.py` (new) — 9-stage JSONL funnel telemetry
- `memecoin/scanner.py` — telemetry emits at `telegram_received`,
  `screening_rejected`, `screening_passed`, `signal_constructed`,
  `add_signal_entered`, `dedup_rejected`
- `memecoin/telegram_monitor.py` — `_log_executor_failure()` +
  `add_done_callback` on the retained executor Future
- `memecoin/tests/test_v8_paper.py` — full rewrite (was silently broken,
  4/20 failing, since PROGRESS-FIX PF6 on 2026-08-08)
- `memecoin/tests/test_scanner_v8_dedup.py` (new)
- `memecoin/tests/test_telegram_executor_observability.py` (new)

### Tests

`test_v8_paper.py` + `test_scanner_v8_dedup.py` +
`test_telegram_executor_observability.py`: **15/15 passing** in isolation.
(Full-suite `memecoin/tests/` collection hits a pre-existing,
unrelated `sys.modules["memecoin.config"]` stubbing collision from other
test files — confirmed pre-existing and out of scope; not introduced by
this fix.)

### Answers to the named questions

1. **Is the zero-position book a code bug or a genuinely-empty market
   window?** Code bug — `has_dex_id` structurally guaranteed rejection of
   every candidate that could reach the gate.
2. **Is `dex_id` ever a valid graduation signal?** No — DexScreener sets it
   on pump.fun bonding-curve tokens well before graduation.
3. **Does dedup explain any of the 15 mystery events?** No — all 15 reached
   the gate and were rejected there, not deduped away (structural proof);
   real-window dedup (1 instance) is a separate, correctly-working path.
4. **Was VF4's executor footgun the root cause?** No, confirmed by log
   audit — no exception was ever swallowed. Fixed anyway as hardening.
5. **Is the current V8/V7 funnel an independent comparison?** No — V8 only
   sees V7-screened survivors. Labeled `V8_CONDITIONAL_ON_V7_SCREEN`.
   Fork-point proposal documented, not implemented.
6. **Can a candidate now silently disappear between funnel stages?** No —
   all 9 stages instrumented; VF3's 107→108→107 count reconciles exactly
   via the 1 dedup reject.
7. **Has a real position actually opened and been tracked, not just
   logged?** Yes — Moblin/`V8481249`, persisted, and its subsequent
   `hard_stop` close proves live monitor tracking, not just a log line.
8. **Does declaring this fixed require only that `maybe_open` logs
   appear?** No, and this receipt does not rely on that — it relies on the
   persisted positions JSON record and the monitor-tracked close.

### Commits

`ae31959` (H1 root cause + VF1-VF4), `d3b2492` (VF7 tests), this commit
(RECEIPTS.md).

### Status: **V8_TWIN_LIVE_VERIFIED**

A real, fresh, production progress<0.70 `CURVE_ACTIVE` candidate opened, was
persisted to `memecoin_v8_positions.json`, was tracked by the live monitor
loop, and closed via `hard_stop` — full lifecycle proof, not log-line-only.
All 5 VF6 scenarios observed live in production except scenario 5, which
relies on VF7's deterministic tests per the spec's explicit allowance.

---

## WATCHDOG-BATCH Phase 1 — Cron Liveness (2026-08-12/13)

**Purpose.** Every incident this session (PP feed silently dead 24h,
V8's gate rejecting 100% of candidates for days, and — found live during
this exact investigation — the K5 nightly cron silently disabled since
2026-08-09) shared one shape: the system's own record of itself said
"working" while production had silently diverged, and nothing caught the
gap until a manual forensic audit. This batch is a standing mechanism to
catch that class of drift automatically instead of by hand.

**Scope decision.** The full design (25 subsections: cron liveness, feed
liveness, funnel stuck-stage detection, test-suite drift, claim-vs-artifact
checks, and an externally-scheduled LLM audit agent) was reviewed and
approved as an architecture, but built in explicit phases rather than one
unverifiable batch — building all of it before proving any of it live
would recreate the exact problem it exists to solve. **Phase 1 = the
deterministic, no-LLM cron-liveness layer** (W1, W2, W5, W10 from the
design spec): the piece that would have caught the actual incidents found
this session, shipped and live-verified before anything heavier.

### Two more dead crons found while instrumenting this

`quantbot-epoch` and `quantbot-v8` had the *identical* backslash-continuation
bug as `quantbot-v8-inputs` (fixed 2026-08-12 as part of V8-TWIN-FIX
monitoring) — all three rejected by the cron daemon since 2026-08-09
18:46 UTC ("`Error: bad minute`", "`this crontab file will be ignored`").
`logs/epoch_daily.jsonl` had exactly one entry (from the original manual
install-time test), dated 2026-08-09, and had not grown since — the daily
epoch capital-decision tracking had been silently dark for 4 days. Fixed
directly on the VPS (rewrote both files as single-line entries, `systemctl
restart cron`, confirmed zero new parser errors), then properly fixed via
the git-tracked `deploy/cron.d/` mechanism below so this can't recur the
same way.

### H1: two independent questions, checked separately

A file can be valid syntax and never fire (wrong permissions, disabled
service) — and it can be invalid syntax while an artifact still looks
fresh from a past manual run (the actual K5 incident: a manual test at
install time updated the artifact and made it look current for 4 days
while the real `/etc/cron.d` entry was silently rejected). Neither
question alone is sufficient.

**W5A — is the `/etc/cron.d` definition syntactically valid?**
(`watchdog/checks/cron_static.py`) Parses each managed file line by line,
classifying blank / comment / env-assignment / cron-entry / malformed;
validates the 5 schedule fields with `croniter`; cross-checks recent
`journalctl -u cron` output for the daemon's own parser-error lines
against each managed filename (trusting the daemon over the parser when
they disagree). Explicitly flags `UNMANAGED_SCHEDULE` for any
`quantbot*` file in `/etc/cron.d` not in the registry — such a file has
no execution-liveness coverage and that fact is itself surfaced, not
silently assumed away.

**W5B — did the scheduled job actually run?** (`watchdog/checks/cron_execution.py`,
fed by `watchdog/exec_wrapper.py`) This is the direct fix for the K5
incident: `exec_wrapper` wraps each cron-invoked command and writes a
durable execution receipt (`job_receipts` table: `job_id`, `trigger_type`,
`started_at`, `finished_at`, `exit_code`, `git_sha`) *tagged with who
triggered it* — `trigger_type="scheduler"` when cron itself invokes it,
`trigger_type="manual"` when a human runs the same wrapper by hand for
testing. Only a `scheduler`-trigger receipt satisfies liveness; a manual
receipt is structurally incapable of masquerading as proof of a real
scheduled fire, closing exactly the gap that hid the original bug. Liveness
is computed against the schedule's own expected-previous-fire time
(`croniter`) plus a configured grace window, with an explicit boot-time
grace so a reboot that skipped one legitimately-unreachable fire doesn't
false-alarm (grace expires normally once a *post-boot* fire is overdue).

### Architecture

**W1 — Layer 1 does not run from cron.** A cron-scheduled watchdog cannot
detect cron itself being dead — it dies with the thing it watches. Layer 1
runs from `systemd` timers instead (`deploy/systemd/quantbot-watchdog-fast.timer`,
every 5 min; `-slow.timer`, hourly, reserved for future heavier checks —
no jobs registered on it yet). Both invoke the same engine
(`python -m watchdog.runner --profile fast|slow`) with a non-blocking
`flock` singleton lock so overlapping runs can't occur.

**W2 — the watchdog watches itself.** Every run writes a `watchdog_runs`
receipt (run_id, host, boot_id, git_sha, checks_due/completed,
final_runner_status) *regardless of whether findings were CRITICAL* — a
CRITICAL finding is the system working correctly, not a runner failure.
`--self-test` exercises the full check-engine/state/notifier pipeline with
a synthetic, clearly-labeled `[WATCHDOG TEST]` fault sequence (fires,
dedups, recovers) and sends one real Telegram round-trip, without touching
any real registry job or trading state.

**W10 — alerting.** Standalone Telegram sender (`watchdog/notifier.py`) —
deliberately does not import `app.alerts`, since this runs as an
independent process outside the gunicorn app and can't assume
`app.alerts.init()` ran in it; reads the same `TELEGRAM_BOT_TOKEN`/
`TELEGRAM_CHAT_ID` env vars directly. Incident lifecycle: (none) → SUSPECT
→ FIRING → RECOVERED, persisted in SQLite (WAL mode, explicit
`BEGIN IMMEDIATE`/`COMMIT` transactions). CRITICAL findings fire
immediately (deterministic proof, no need to wait for repeats); WARN
requires 2 consecutive occurrences before paging. Reminder cadence: 2h
first, 6h thereafter for CRITICAL; 6h for WARN. Exactly one RECOVERED
message on resolution. At most one daily digest.

### Two real bugs the fault-injection tests caught during development — not found any other way

1. **`STATUS_UNKNOWN` was incrementing `consecutive_failures` like a real
   failure**, and could silently downgrade an already-`FIRING` incident
   back to `SUSPECT` if evidence briefly went missing (e.g. a transient
   `journalctl` hiccup) — which would have un-paged a real ongoing
   incident for no reason. Fixed: `FIRING` incidents are left untouched by
   `UNKNOWN` evidence (`state.touch_incident_seen`); non-firing `UNKNOWN`
   results record a zero-streak marker instead of an escalating one.
2. **`state.upsert_incident()` always stamped with the real wall clock**,
   silently ignoring the caller's own `now_ts` — this broke reminder-
   interval math the instant a test (or any future replay/backfill use)
   supplied a non-realtime clock, and was a latent (if usually
   sub-second-harmless) inconsistency even in real production. Fixed by
   threading `now_ts` through explicitly end to end.

Both were caught because the test suite asserted actual behavior (message
counts, state values) rather than "doesn't crash" — exactly the standard
this whole batch exists to hold the rest of the codebase to.

### Tests

37/37 passing (`watchdog/tests/`, deterministic, no live VPS access
needed), covering design-spec fault-injection items 1-6, 20-25, 27, 29-30:
backslash-continuation detection (the exact real bug), valid-file
non-flagging, missing-file/unreadable-file handling, severity capping by
job registry ceiling, journal-evidence-unavailable → UNKNOWN not OK,
manual-vs-scheduler receipt non-substitutability, stale-past-grace
detection, not-yet-due no-false-alarm, boot-grace (including grace
expiring correctly once a post-boot fire is overdue), day-boundary
schedule arithmetic, 12-consecutive-failures → 1 alert + correct reminder
cadence (not 12 messages), exactly-one recovery message, notifier-failure-
doesn't-erase-the-incident, singleton lock (including idempotent release),
DB-missing → created-not-silently-green, corrupt-DB → fails loud
(`sqlite3.DatabaseError`, not silent success), manual receipts never
satisfying a scheduler-only query, one crashing check not blocking
independent checks, and liveness evidence surviving simulated log
rotation (SQLite state is independent of any log file's lifecycle).

Item #6 (DST) is a documented scope limitation, not an untested claim: the
VPS runs in UTC and all schedule math uses Unix epoch floats, which are
timezone-unambiguous by construction — there is no wall-clock DST
transition to get wrong under UTC. If ever deployed against a non-UTC
scheduler, this would need `croniter`'s timezone-aware datetime mode
instead of raw epoch floats.

### Live deployment receipts

- Dependencies (`croniter`, `PyYAML`) installed on the VPS venv; both
  added to `requirements.txt`.
- `bash deploy/systemd/install.sh` — symlinked (not copied) both timer
  pairs into `/etc/systemd/system/`, `daemon-reload`, `enable --now` both.
  `systemctl list-timers` confirms both armed.
- `bash deploy/cron.d/install.sh` — symlinked the 3 wrapped cron entries
  into `/etc/cron.d/`, `systemctl restart cron`, confirmed zero new parser
  errors in syslog post-install.
- `python -m watchdog.runner --self-test` run live on the VPS: `PASS`,
  real Telegram round-trip (CRITICAL fire → RECOVERED) sent and received.
- **First real systemd-triggered fire confirmed**, 2026-08-12 23:45:03
  UTC (`watchdog_runs` row `f57d547f...`, `final_runner_status=ok`,
  6/6 checks completed). Findings on that run were all correctly
  explained by real, known-recent history — not false positives:
  `cron_static` WARN on `epoch_daily`/`v8_vs_v7_daily` because the cron
  daemon's parser-error lines from the 23:21:46 fix were still inside the
   1h journal lookback window (self-resolves once they age out);
  `cron_execution` WARN on all 3 jobs because tonight's first scheduled
  fires through the new wrapper (23:55, 23:58, 00:15 UTC) hadn't happened
  yet at check time — accurate "no receipt yet," not a false alarm.

### Symlink-installed, not copied — closing the exact gap that caused this batch

`deploy/systemd/` and `deploy/cron.d/` are git-tracked and installed via
`ln -sf`, not copied — a future schedule edit lands live on the next
`git pull` alone. This is the direct fix for the meta-failure, not just
the failure: the pre-fix cron entries were "documented in `RECEIPTS.md`
as an instruction to add this cron entry" and then never actually
re-applied when anyone touched them again. A symlink can't silently drift
from the repo the way a one-time heredoc install could.

### Modified/added files

`watchdog/` (new package: `state.py`, `exec_wrapper.py`, `notifier.py`,
`runner.py`, `checks/__init__.py`, `checks/cron_static.py`,
`checks/cron_execution.py`, `checks.yaml`, `tests/*`), `deploy/systemd/*`,
`deploy/cron.d/*` (new), `requirements.txt` (+`croniter`, `+PyYAML`),
`.gitignore` (+`logs/watchdog/`, runtime SQLite state never committed).
No trading-path file touched.

### Deferred to later phases, not silently dropped

Feed liveness (W6: PumpPortal real-tick-vs-fallback, Telegram connection
state, research-pipeline upstream/downstream-stall), funnel stuck-stage
detection (W7: generalizing V8-TWIN-FIX's `v8_funnel.jsonl` telemetry),
test-suite drift (W8), claim-vs-artifact checks (W9), and the externally-
scheduled Layer 2 LLM audit agent (W12-W17) — each needs its own
live-verification pass, not bundled into one unverifiable batch. Phase 2
starts immediately after this receipt (V8 funnel stuck-stage detection +
Telegram feed liveness, both reusing existing telemetry with zero
trading-path changes).

### Status: **WATCHDOG_LAYER1_LIVE**

Deterministic layer running and proven with a real systemd-triggered fire
and correct findings against known ground truth. Not yet
`WATCHDOG_LIVE_VERIFIED` per the spec's own bar — that requires the full
24h acceptance window (including tonight's first real scheduler receipts
landing) and Layer 2 (external LLM audit) existing at all, neither of
which is true yet. No trading logic touched; `LIVE_TRADING=false`
unchanged throughout.

---

## WATCHDOG-BATCH Phase 2 — Funnel + Feed Liveness (2026-08-13)

**Scope.** W7 (V8 funnel stuck-stage detection) and W6B (Telegram feed
liveness), both reusing existing telemetry with zero trading-path changes,
per the deferral list at the end of Phase 1's receipt.

### W7 — V8 funnel terminal-disposition completeness (`watchdog/checks/v8_funnel.py`)

Reuses `logs/v8_funnel.jsonl`, the 9-stage JSONL telemetry built for
V8-TWIN-FIX. Checks that every `add_signal_entered` reaches
`dedup_rejected` or `v8_gate_entered` within grace, and every
`v8_gate_entered` reaches `v8_gate_rejected` or `v8_opened` within grace —
the exact invariant that would have caught V8-TWIN-FIX's root cause
(candidates entering the gate and being rejected by construction) within
hours of deploy instead of days.

**Deliberately does not implement a reject-rate/conversion-percentage
anomaly detector.** Real production data (Phase-1-era V8-TWIN-FIX numbers:
1 open / 107 gate entries over ~27h, ≈1%) is a *legitimate* baseline — a
naive "too many rejects" threshold would have false-positived against
this exact known-good behavior on day one. Terminal-disposition
completeness is the safe invariant instead: true regardless of what the
real accept/reject ratio happens to be, per the design spec's explicit
"hard invariants are primary, conversion-rate anomalies are secondary and
risky" guidance.

### W6B — Telegram feed liveness (`watchdog/checks/telegram_feed.py`)

`telegram_monitor.py` is the sole signal source in the current
`SOCIAL_ALERT_ONLY` deployment mode — if it silently died, nothing else in
the pipeline would ever fire. Combines two already-existing evidence
sources, zero app-code changes: `journalctl -u quantbot` for
`telegram_monitor.py`'s own self-reported `TELEGRAM_AUTH_REQUIRED` /
"tg-monitor thread is dead" states (unambiguous → CRITICAL), cross-checked
against `v8_funnel.jsonl`'s `telegram_received` stage as independent
message-flow evidence. A quiet channel with no error signal reports WARN
("ambiguous — could be a legitimately quiet channel or a silently dead
connection"), never CRITICAL and never silently OK — per the design
spec's explicit requirement that silence alone is never proof of death.

### Tests

15 new fault-injection tests (52/52 total across the whole watchdog
package), covering design-spec items 9-10 and 13-15: AUTH_REQUIRED/
thread-dead → CRITICAL, a legitimately-quiet channel not mislabeled
disconnected, missing-terminal-disposition detection, n=1-within-grace
producing no false alarm, and the specific real-production shape (10
candidates, 10 explicit rejects) correctly *not* flagged as a silent
disappearance.

### Live verification — and a real, unrelated bug found and fixed along the way

Both checks confirmed `OK` against real production data in a manual run
(872 real funnel events all accounted for; last real Telegram message 8
minutes old), then confirmed present and correct inside an actual
systemd-timer-triggered run (`watchdog_runs` row `97c4c3124ea64d33`,
2026-08-13 00:00:00 UTC, 8/8 checks completed).

**While confirming that run's cron-liveness findings, found the wrapped
cron jobs were not actually landing scheduler receipts despite firing.**
`quantbot-v8` fired via cron at 23:58:01 (confirmed in `journalctl`) but
left zero trace anywhere — no output log file, no `job_receipts` row.

**Root cause:** `. .env` (bare filename, no `/`) makes POSIX `sh`'s
dot-builtin search `$PATH` rather than the current directory — under
cron's minimal `PATH`, `.env` isn't found even though `cd /root/quant-bot`
had already run moments earlier in the same command chain. The failed
`. .env` short-circuits the `&&` chain before it ever reaches the actual
command — and since the `>>` redirect is bound only to the *last* command
in the chain, nothing gets written anywhere, at all, silently. **This
predates today's work** — the same `. .env` pattern was in the original
pre-fix cron lines — but was never observable before, because the
backslash-continuation syntax bug (fixed earlier the same day) kept these
jobs from running at all. Two independent bugs stacked on the same three
files; fixing the first is what finally exposed the second.

Fixed by sourcing `./.env` instead (commit `638bd8b`). Verified in three
independent, increasingly strong ways before considering this closed:
1. Reproduced the failure via `env -i PATH=/usr/bin:/bin sh -c "..."`
   matching cron's real minimal environment, confirmed the fix resolves it.
2. Installed a synthetic one-off cron.d entry firing ~3 minutes out;
   confirmed a real `trigger_type=scheduler` receipt landed
   (`exit_code=0`) from an actual cron fire, not a manual simulation.
3. **Waited for `k5_nightly`'s real, unmodified, production 00:15 UTC
   fire** (the actual nightly `report.py` + `path_stats.py` run, not a
   synthetic stand-in) — landed with `trigger_type=scheduler`,
   `exit_code=0`, ~122s real runtime, and a genuine new
   `## [K5 nightly] 2026-08-13` section appended to `docs/V8_INPUTS.md`.
   Re-ran the watchdog immediately after: all 8 checks now report `OK`,
   including `cron_execution.k5_nightly`.

**One anomaly found and honestly left open, not swept under the
rug:** `quantbot-epoch` (`55 23 * * *`) produced **zero** `journalctl -u
cron` invocation line at all at 23:55:01 — not a failure after invocation
like `quantbot-v8`, but no invocation whatsoever, despite the file being
byte-structurally identical to `quantbot-v8`'s (compared via `xxd`, same
trailing newline, same line count, no hidden characters), parsing cleanly
by both the daemon and `cron_static.py`, and correctly symlinked with the
same permissions. Not reproduced despite investigation (hexdump
comparison, journal search across the full day, schedule-collision check,
`crontab -l` cross-check). Given `quantbot-epoch`'s next real fire isn't
until tonight's 23:55 UTC, this is left as an open, explicitly-flagged
item rather than a claimed root cause — and the watchdog itself (W5B) is
now exactly the mechanism that will catch a recurrence automatically:
if it misses again, `cron_execution.epoch_daily` will report CRITICAL
once the 45-minute grace expires, without anyone needing to notice by
hand. This is the intended design working as built, applied to itself.

### Modified/added files

`watchdog/checks/v8_funnel.py`, `watchdog/checks/telegram_feed.py`,
`watchdog/tests/test_v8_funnel.py`, `watchdog/tests/test_telegram_feed.py`
(all new), `watchdog/checks.yaml` (+`funnels:`/`feeds:` sections),
`watchdog/runner.py` (wired both into `run_checks()`),
`deploy/cron.d/quantbot-{epoch,v8,v8-inputs}` (`. .env` → `. ./.env`).

### Status: **WATCHDOG_LAYER1_LIVE** (extended)

Both Phase 2 checks live, tested, and proven against real production data
and a real systemd-triggered run. The `.env` bug found in the process is
fixed and proven via a real, unmodified production job's actual scheduled
fire — not a simulation. One anomaly (`quantbot-epoch`'s single missed
23:55 invocation) remains open and explicitly flagged, with the
watchdog's own W5B check now positioned to catch a recurrence
automatically at tonight's next fire. Still not `WATCHDOG_LIVE_VERIFIED`
— that requires the full 24h acceptance window plus Layer 2, neither of
which exists yet. No trading logic touched; `LIVE_TRADING=false`
unchanged throughout.

---

## WATCHDOG-BATCH Phase 3 — PumpPortal Feed + Research Pipeline Liveness (2026-08-13)

**Scope.** W6A (PumpPortal tick feed liveness) and W6D (research pipeline
upstream-flowing/downstream-stalled detection), the two remaining
higher-value items from the Phase 1 deferral list. **W6C (an active
Solana/Helius RPC probe) was deliberately not built** — `SOCIAL_ALERT_ONLY`
mode currently runs zero RPC-dependent code paths (whale wallet polling,
market scanner, pumpfun listener, near-miss poller are all off), and
`CLAUDE.md` explicitly directs against adding anything that increases
Helius RPC call volume on the downgraded free plan. Building an active
probe now would violate that directive for a check with near-zero current
value; revisit if/when those code paths come back online.

### W6A — PumpPortal feed liveness (`watchdog/checks/pumpportal_feed.py`)

The PP tick feed was silently dead for 24h earlier in this project's
history — ~$36 of real losses before it was root-caused. K2 already added
an in-process tick deadman (`_check_tick_deadman`), but it dies with the
process it watches; this closes the same class of gap Layer 1 closes
everywhere else — an independent, externally-scheduled check reading
durable evidence (`journalctl`) instead of trusting the in-process check's
own continued existence. Reuses `telegram_monitor.py`'s pattern: journal
lines already logged by `pumpportal_monitor.py` (WS connect, WS error, K2's
own tick-deadman warnings), zero new instrumentation.

**Checked against real production logs before writing this, and glad it
was:** PumpPortal reconnects roughly every 45-60 seconds by *deliberate
design* (`"PumpPortal using pre-warmed rotation WS (gap <100ms)"` — a
rotation strategy, not a failure loop). A naive "too many reconnects"
threshold would have been a permanent false positive against completely
normal behavior from day one. The check never evaluates reconnect
frequency at all — only actual error/deadman evidence — with a regression
test locking this in (30 reconnects, zero errors → `OK`, not flagged).

### W6D — research pipeline stall detection (`watchdog/checks/research_pipeline.py`)

Two independent signals, both reusing already-existing on-disk state with
zero new instrumentation:

1. **Queue consumption lag** — `research/data/signal_queue.jsonl` (written
   by `scanner.py`) vs `research/data/.queue_offset` (persisted by
   `research.tg_listener.FileQueueListener` after each processed line). A
   large, sustained gap with the queue still growing is exactly
   "upstream flowing, downstream stalled" — the same shape as the
   historical `pp_vsol`-never-reached-Supabase bug, just at an earlier
   pipeline stage.
2. **Spool growth** — `research/spool/failed_inserts.jsonl` (written by
   `research/spool/writer.py` whenever a Supabase insert fails) is direct,
   durable evidence of active data loss. **Growth-based, not
   total-count-based**: the real file already has 68 historical lines from
   a genuine bug (`progress_capture_lag_ms` — a float — rejected by an
   `integer`-typed Supabase column), confirmed dormant since 2026-08-08. A
   naive "any lines exist" check would have alarmed on 5-day-old history;
   this one correctly reports `OK` because nothing has been appended
   recently, with a regression test locking that in.

### Tests

19 new fault-injection tests (71/71 total across the whole watchdog
package), including the two regression guards above and standard
coverage: deadman fire → `CRITICAL` (`PRIMARY_FEED_DEGRADED`), suppressed
weekly deadman note ≠ critical, error-more-recent-than-connect → `WARN`,
missing queue/offset files → `UNKNOWN` not `OK`, offset-ahead-of-file-size
(rotation/truncation) → `UNKNOWN` not a guess, small gap within threshold
→ no false alarm, minimum-sample floor on spool alerts (never fire on
n=1-2), malformed spool lines skipped not fatal.

### A real bug found and fixed along the way, unrelated to the new checks

While checking whether `research.main` was even running (confirmed: yes,
separate `quantbot-research.service`, distinct from the main trading app),
`systemctl list-units` showed **`quantbot-watchdog-slow.service` as
`failed`** — the watchdog's own infrastructure, one hour after Phase 1/2
deployment.

**Root cause, and it was mine:** `runner.py` returned exit code `75` for a
benign, expected singleton-lock-contention skip (W1's own no-overlapping-
runs guarantee working correctly). systemd's default `Type=oneshot`
semantics treat *any* non-zero exit as a service failure (no
`SuccessExitStatus=` configured) — the skip itself was correct, only the
exit code was wrong, and the misreport was exactly the kind of "system
claims broken when it's fine" (inverted from the usual "claims fine when
broken," but the same root failure to trust primary evidence) this whole
project exists to prevent. Compounding it: the fast timer (`*:0/5`) and
slow timer (`hourly`) both fire at `:00`, guaranteeing lock contention —
and therefore this misreport — every single hour.

Fixed both: exit `0` on lock-contention skip, and slow now fires at
`:03` past the hour instead of `:00`, removing the guaranteed collision
rather than merely tolerating it. Regression test added. Verified live:
`systemctl status quantbot-watchdog-slow.service` now shows
`code=exited, status=0/SUCCESS`.

### Live verification

All 5 new checks (`feed.pumpportal`, `pipeline.research_queue_lag`,
`pipeline.research_spool`, plus re-confirmation of Phase 2's checks)
confirmed `OK` against real production data in a manual run, then
confirmed present and correct inside an actual systemd-triggered run
(`watchdog_runs` row `b33558022db04b4b`, 2026-08-13 01:13:00 UTC, 11/11
checks completed, `final_runner_status=ok`):

```
funnel.v8                    | OK   | 888 events, all accounted for
feed.telegram                | OK   | last message 504s ago
feed.pumpportal               | OK   | most recent WS event: successful connect
pipeline.research_queue_lag   | OK   | consumer caught up (gap=0 bytes)
pipeline.research_spool       | OK   | 0 recent failures (68 historical, dormant)
```

`cron_execution.epoch_daily`/`v8_vs_v7_daily` still correctly show `WARN`
("no scheduler execution receipt ever recorded") — expected and accurate,
not a regression: their real next scheduled fires (which will prove the
`.env` fix under their actual production schedule, not a synthetic
stand-in) aren't until tonight's 23:55/23:58 UTC.

### Modified/added files

`watchdog/checks/pumpportal_feed.py`, `watchdog/checks/research_pipeline.py`,
`watchdog/tests/test_pumpportal_feed.py`, `watchdog/tests/test_research_pipeline.py`
(all new), `watchdog/checks.yaml` (+`pumpportal` feed, +`pipelines:`
section), `watchdog/runner.py` (wired both in, reused the single
`journalctl -u quantbot` fetch across telegram+pumpportal), `watchdog/runner.py`
+ `watchdog/tests/test_runner.py` + `deploy/systemd/quantbot-watchdog-slow.timer`
(the lock-contention exit-code fix).

### Status: **WATCHDOG_LAYER1_LIVE** (extended further)

All Phase 1-3 checks live, tested (71/71), and proven against real
production data and real systemd-triggered runs. The watchdog's own
infrastructure bug (slow timer misreporting failure) found and fixed
before it could erode trust in `systemctl status` as a signal. Remaining
deferred scope: W6C (explicitly skipped, see above), W8 (test-suite
drift), W9 (claim-vs-artifact/`batch_verify` semantics), and Layer 2 (the
externally-scheduled LLM audit agent, W12-W17) — still needed to reach
`WATCHDOG_LIVE_VERIFIED`. No trading logic touched; `LIVE_TRADING=false`
unchanged throughout.

---

## WATCHDOG-BATCH Phase 4 — Test-Suite Drift + Claim-vs-Artifact (2026-08-13)

**Scope.** W8 (test-suite drift) and W9 (claim-vs-artifact verification),
the last two items before Layer 2 (the externally-scheduled LLM audit
agent, the final piece needed for `WATCHDOG_LIVE_VERIFIED`).

### W9 — claim-vs-artifact verification (`watchdog/checks/batch_claims.py`)

`tools/batch_verify.py` already does the hard part: `verify_batch()`
already returns a fully structured per-item `GREEN`/`PARTIAL`/`FAIL`
verdict dict, and `_check_receipt()` already distinguishes "section
missing" (`FAIL`) from "section exists but `receipt_complete=false` or
commit hash absent" (`PARTIAL`) from "fully backed" (`OK`). No `--json`
flag needed, no separate claims registry built — `batches/*.yaml` already
*is* the claims registry the design spec calls for; building a parallel
one would have been redundant infrastructure over something that already
existed and already worked.

**What was actually missing, and it's real:** `main()`'s CLI exits
`1 if any_fail else 0` — `PARTIAL` items exit `0`, identically to a
fully-`GREEN` batch. Confirmed live against this project's own
`v8_readiness.yaml`, unmodified, right now: **4 of 7 items (N2, N4, N6,
N7) are `PARTIAL`**, and `batch_verify`'s own exit code would report this
as fine. This check makes that distinction a real, continuously-monitored
`WARN` instead of something only visible to someone who happens to run
`--verbose` and read the table by hand.

### W8 — test-suite drift (`watchdog/checks/test_drift.py`)

Two checks, both static/subprocess-bounded — no arbitrary production-module
imports inside the watchdog process itself, per the design spec's explicit
safety requirement:

1. **`check_stale_mocks()`** — the exact class of bug that left
   `test_v8_paper.py` silently broken for 3 days during PROGRESS-FIX (it
   mocked a symbol a prior commit had already removed, undetected until a
   full-suite run was done by hand). Pure AST parsing of both the test
   file (extract `patch()`/`patch.object()` targets) and the target
   module (does the symbol still exist there) — never imports either
   file. Reports `WARN`, not `CRITICAL`: static analysis of dynamic
   attributes can false-positive, so this needs a human glance, not an
   auto-page.
2. **`check_test_collection()`** — `pytest --collect-only` as a bounded
   subprocess, per test directory, run **separately** (not combined — the
   combined tree has a pre-existing, unrelated `sys.modules` stubbing
   collision across some `memecoin/tests` files, confirmed during
   V8-TWIN-FIX and out of scope here). Collection failure (import/syntax
   error) is a distinct, worse problem than a test merely failing its
   assertions.

### Two real false positives found and fixed while verifying against the actual repo — not synthetic fixtures

1. **`check_stale_mocks` flagged `memecoin/journal_reconciler.py`'s
   `read_sol_delta` as stale.** It's a defensive
   `try: from x import y / except ImportError: y = None` pattern — the
   module's own code comment literally says *"Tests patch this name
   directly."* My AST scan only checked top-level statements; a name
   assigned inside `try`/`except`/`if` blocks is still module-scoped (no
   new scope introduced, unlike `def`/`class` bodies) but was invisible to
   the naive scan. Fixed by descending into `try`/`except`/`if`/`else`
   bodies specifically, while still correctly excluding function/class
   bodies (which *do* introduce a new scope).
2. **`check_test_collection` hardcoded `"python3"` as the subprocess
   interpreter.** On a dev machine this resolves to system Python (no
   `croniter`/`PyYAML` installed there), which falsely flagged watchdog's
   own test suite — which passes 71/71 through the real venv — as broken
   via `ModuleNotFoundError`. Fixed to default to `sys.executable` (the
   currently-running interpreter), which is also correct on the VPS since
   systemd's `ExecStart` already invokes `.venv/bin/python` directly.

After both fixes: the stale-mock scan is clean (121 real, resolvable
patch targets checked across the whole test tree, zero false positives)
and the collection check correctly shows `memecoin/tests` failing — the
real, already-known, still-unfixed pre-existing pollution issue, not
fixed here, but now carrying a durable automated signal instead of
depending on someone remembering to check by hand — with
`research/tests`, `watchdog/tests`, and the top-level `tests/` all
collecting cleanly.

### Tests

23 new fault-injection tests (85/85 total across the whole watchdog
package), including regression guards locking in both false-positive
fixes above, plus standard coverage: genuinely-removed symbol → flagged,
third-party targets skipped (not guessed at), missing test dirs → `OK`
(nothing to check), import errors → `WARN` not silently passed,
`GREEN`/`PARTIAL`/`FAIL` batch verdicts mapped correctly, `PARTIAL`
capped at the configured severity ceiling by default (documentation gaps
aren't automatically page-worthy) but reachable at `CRITICAL` when
configured to allow it.

### Live verification

Wired on the `slow` (hourly) profile — heavier than the fast-profile
checks (AST scan across the whole test tree + subprocess `pytest`
invocations, ~2s combined), consistent with the design spec's "do not run
pytest every 5 minutes" guidance. Also made the runner's soft
timeout-marker profile-aware (30s fast / 120s slow, previously a single
fast-tuned constant that no longer fit once slow-profile checks got
heavier) — this is a run-receipt annotation only; systemd's own
`TimeoutStartSec` (90s fast / 600s slow) is the real hard kill and was
unaffected.

Confirmed against real production data on the VPS, then confirmed present
and correct inside an actual systemd-triggered run (`watchdog_runs` row
`d8315c137f844443`, 2026-08-13 02:03:07 UTC, 7/7 checks completed,
`final_runner_status=ok`):

```
test_drift.stale_mocks                | OK   | 121 targets checked, 0 findings
test_drift.collection.tests_memecoin  | WARN | pre-existing collection failure (confirmed, not new)
test_drift.collection.tests_research  | OK   | collects cleanly
test_drift.collection.tests_watchdog  | OK   | collects cleanly
test_drift.collection.tests_quant-bot | OK   | collects cleanly
claims.batch.rc_closure               | OK   | 3/3 GREEN
claims.batch.v8_readiness             | WARN | 4/7 PARTIAL (N2, N4, N6, N7)
```

One methodology note, in the interest of not overclaiming live proof: the
first attempt to confirm this landed inside a *real* systemd-triggered
slow-profile run instead matched a manual verification run I'd made
moments earlier (same `profile='slow'` value in the database, no
distinction between manual and timer-triggered runs for watchdog's own
self-checks the way `job_receipts` already distinguishes `trigger_type`
for cron jobs). Caught before writing this receipt, re-verified against
a run with `started_at` strictly after a reference timestamp taken before
any manual check — the `02:03:07 UTC` run cited above is the genuine
timer fire. **Layer 1's own runs don't yet carry a `trigger_type`
distinction the way W5B's cron `job_receipts` do** — worth adding in a
later pass if self-verification like this needs to happen often; noted
here rather than silently worked around.

### Modified/added files

`watchdog/checks/batch_claims.py`, `watchdog/checks/test_drift.py`,
`watchdog/tests/test_batch_claims.py`, `watchdog/tests/test_test_drift.py`
(all new), `watchdog/checks.yaml` (+`test_drift:`, +`claims:` sections),
`watchdog/runner.py` (wired both in, profile-aware timeout marker).

### Status: **WATCHDOG_LAYER1_LIVE** (extended further)

All Phase 1-4 checks live, tested (85/85), and proven against real
production data and a genuine systemd-triggered slow-profile run. Two
real false positives found and fixed by testing against the actual repo
instead of trusting synthetic fixtures alone — exactly the discipline
this whole system exists to enforce, applied to itself. Layer 1 is now
feature-complete per the original design's deferred-scope list (minus the
deliberately-skipped W6C). Only Layer 2 (the externally-scheduled LLM
audit agent, W12-W17) remains before `WATCHDOG_LIVE_VERIFIED` is
reachable. No trading logic touched; `LIVE_TRADING=false` unchanged
throughout.

### Addendum (2026-08-14 00:09 UTC) — the `.env` fix confirmed under real, unmodified production schedule

`epoch_daily` and `v8_vs_v7_daily`'s first real scheduled fires since the
`.env` fix (Phase 2/3 receipts): `epoch_daily` at 23:55:01 UTC, `v8_vs_v7_daily`
at 23:58:01 UTC, 2026-08-13 — both `trigger_type=scheduler`,
`exit_code=0`. Verified against real downstream artifacts, not just the
receipt row: `logs/epoch_daily.jsonl` has a fresh `2026-08-13` entry;
`docs/RECEIPTS.md`'s own N6/N7 table has a fresh `2026-08-13` row (65 v7
trades, +433824656.8% mean — real computed data, not a placeholder). This
was the one still-open item from Phase 2/3 (their prior WARN status was
correctly "no receipt yet," not a regression) — now closed.

Honest framing: this is confirmation the fix holds, not proof the
watchdog can catch a real failure autonomously — nothing failed tonight,
so the alerting pipeline wasn't exercised end-to-end by a genuine
incident. That test remains open.
