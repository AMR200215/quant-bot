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

## How to use this file

**Before changing an execution path:** check if there is a receipt for the current
behaviour. If not, you are changing unproven code — note that in the PR.

**After a fix:** re-run the relevant cell or wait for the next live trade that hits
the changed path. Add the row. Commit it with the fix.

**To answer "does X work?":** search this file for X. If no row, it is not proven.

**Stale rows:** a row is stale if the commit column predates a significant executor
rewrite. Mark it `[stale — re-run]` and re-run when a suitable token is available.

---

## L6 — Screen Compression (2026-07-10, commit TBD)

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

**RTT VPS→Helius**: ~80ms network (measured via mainnet-beta proxy; Helius saturated
by live bot during measurement). Helius `getAccountInfo` oracle avg=1915ms (this is
`commitment=confirmed` propagation wait, not network RTT). Under 100ms threshold →
no endpoint switch required.

---

## X1 — Presigned Urgent Exits (2026-07-10, commit TBD)

**Code changes**: `memecoin/portfolio.py`
- Added `feed_blind`, `pre_graduation_exit` to `_STOP_REASONS` (presigned-eligible)
- Added oracle gate: `get_pumpfun_curve_complete()` must return `complete==False` before presigned send
- Added T22 skip: tokens in `_mint_token_program_cache` with TOKEN22 program bypass presigned, use ladder
- Fixed fallback log: now emits `presign_fallback reason=<err>` on send failure

**Acceptance**: PENDING first live hard_stop/trailing_stop exit post-deployment.
Will add telemetry line (exit_trigger→sell_sent <300ms) + sig when observed.

---

## X3 — Exit Telemetry Sub-spans (2026-07-10, commit TBD)

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

---

## X5 — Post-buy Readiness (2026-07-10, commit TBD)

**Code changes**: `memecoin/portfolio.py`
- `_fill_confirm_ts` stored on position object after buy confirms
- Monitor loop emits `FIRST_PRICE_MS token=<sym> ms=<N>` log + `first_price_tick` telemetry event on first price tick post-fill
- Target: ≤1000ms from fill confirm to first monitored price

**Helius WS 429 backoff**: `_confirm_tx` already implements 2s backoff for first 15s,
then 4s on 429. The P8 full backoff (2s→4s→8s→cap 60s) applies to the WS reconnect
in `pumpfun_listener.py` — reviewed separately.

**Acceptance**: PENDING first live buy post-deployment (need `FIRST_PRICE_MS` log line).

---

## B-batch (epoch gate) — 2026-07-11 — commit PLACEHOLDER

### B1 — Dual-source pre-graduation progress

| Field | Value |
|---|---|
| behavior | PP-silent positions now use curve-account vSOL as secondary source for pre-graduation exit |
| code | scanner.py: _curve_vsol dict; executor.py: get_pumpfun_curve_price returns virtual_sol_reserves_ui |
| log format | `PRE-GRADUATION EXIT ... source=curve` |
| test | test_b1_pregrad_dual_source.py — 3 tests pass |
| live proof | PENDING — first PP-silent position crossing 97.75 SOL from curve feed |

### B2 — Immediate graduation dispatch (oracle path)

| Field | Value |
|---|---|
| behavior | curve feed complete=True / account_missing → graduated_exit dispatched immediately (no 30s delay) |
| code | scanner.py: dispatch inside curve feed loop with graduation_first_seen_ts stamp |
| log format | `CURVE FEED GRADUATED ... handing over` then immediate close_position call |
| test | test_b2_immediate_graduation.py — 2 tests pass |
| live proof | PENDING — first oracle-confirmed graduation after deployment |

### B3 — pump-amm first for oracle-confirmed graduated

| Field | Value |
|---|---|
| behavior | Oracle-confirmed graduated (graduation_first_seen_ts in notes): executor(pump-amm) → then Jupiter |
| code | portfolio.py: _oracle_confirmed_graduated flag skips pre-executor Jupiter rescue; post-executor B3 Jupiter fallback added |
| log format | executor pump-amm attempt logged BEFORE any Jupiter RESCUE alert |
| test | test_b3_pump_amm_first.py — 3 tests pass |
| live proof | PENDING — first graduated exit after deployment |

### B4 — Per-venue state

| Field | Value |
|---|---|
| behavior | _venue_state dict tracks cooldown_until, attempts, last_result per pos+venue; fast-window pump-amm capped at 3 attempts |
| code | portfolio.py: _venue_state dict + _get_venue_state / _record_venue_attempt / _venue_in_cooldown / _pump_amm_attempts |
| test | test_b4_venue_state.py — 6 tests pass |
| live proof | PENDING — first graduation fast-window cycling after deployment |

### B5 — T22 graduated pump-amm flags wired

| Field | Value |
|---|---|
| behavior | T22_GRAD_PUMP_AMM_PROBE_ENABLED / T22_GRAD_PUMP_AMM_ENABLED now control escalate flag for T22 graduated |
| code | portfolio.py: B5 T22 grad gate reads classify_mint + config flags; probe mode writes logs/t22_grad_probe.jsonl |
| test | test_b5_t22_flags.py — 4 tests pass |
| flags | Both default False — no T22 graduated sell receipt exists yet |
| live proof | PENDING — set T22_GRAD_PUMP_AMM_PROBE_ENABLED=True when ready for canary test |

### B6 — Classifier repair + integration

| Field | Value |
|---|---|
| behaviors | (a) UNKNOWN/error results get 60s TTL; (b) unknown extension → not tradeable; (c) mint_classifier wired into executor T22 route decision |
| code | mint_classifier.py: TTL cache, allowlist; executor.py: _pumpfun_mint_token_program checks classifier first |
| test | test_b6_classifier_repair.py — 5 tests pass |
| live proof | n/a — classification runs on every buy, logs ENTRY PROGRAM GATE line |

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

ENTRY TIMING format (after E1):
```
ENTRY TIMING SYMBOL | ... | build_ms=X.X  sign_ms=X.X  send_ms=X.X  land_ms=X.X  429_ms=X.X  http_build_ms=X.X  confirm_detect_ms=X.X  quote_ms=X.X
```
