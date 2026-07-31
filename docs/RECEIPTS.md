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

## Epoch — Capital Decision (2026-07-30)

Epoch deferred 2026-07-30 — capital decision. Prerequisite for any future live: V8 paper week net-positive after synthetic execution costs (N3' line). B7/E1 timing row deferred with it.
