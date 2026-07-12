# Exit Workflow Map

> Last updated: 2026-07-12 (C8 — post C0–C6/C4 patch, MU retry ladder + G-batch + B-batch)

## Exit Trigger → On-Chain Sell

```
portfolio monitor (scanner.py background thread, polls every 1s)
        │
        ▼
close_position(pos_id, reason, price)
    ├── Kill switch check: live_sells_enabled() → False → skip sell, log warning
    ├── Skip sell: reason in ("reconciled_gone", "manual_sell") → no tokens to sell
    └── Live position sell flow:
               │
               ├─① ORACLE STATE CHECK
               │     get_pumpfun_curve_complete() → is bonding curve graduated?
               │     graduation_first_seen_ts stamped in notes on first complete=True (B2)
               │
               ├─② EXIT ROUTE DECISION
               │     Pre-graduation (complete=False):
               │       executor.sell(escalate=False) — bonding-curve route
               │     Post-graduation (complete=True, oracle-confirmed):
               │       _oracle_confirmed_graduated = True
               │       → FAST WINDOW (B4/G-batch): within 60s of grad_first_seen_ts
               │           retry every 5s, up to 3 pump-amm attempts
               │       → After fast window: MU retry ladder
               │
               ├─③ FAST WINDOW (B4) — first 60s after graduation
               │     up to 3 pump-amm attempts at 5s cadence
               │     _venue_state tracks attempts per venue per position
               │     if pump-amm fails all 3: escalate to MU ladder
               │
               ├─④ executor.sell(escalate=True, skip_pump_amm=<from orchestrator>)
               │     C4: skip_pump_amm passed explicitly — no internal T22 cache lookup
               │     pump-amm PRIMARY (SPL tokens): PumpPortal pool="pump-amm"
               │     pump-amm SKIPPED (T22 or orchestrator skip_pump_amm=True):
               │       → _pamm_err = "pump_amm_skipped_orchestrator"
               │       → falls through to Jupiter fallback
               │
               ├─⑤ B3 POST-EXECUTOR JUPITER FALLBACK
               │     Only fires if executor pump-amm failed AND oracle-confirmed graduated
               │     force_jupiter_rescue_sell(pos, reason)
               │     → route:JUPITER_RESCUE_B3 in notes
               │     NOTE: Pre-executor rescue blocked for oracle-confirmed (B3 gate)
               │
               ├─⑥ MU RETRY LADDER (commits 538132f, deployed 2026-07-07)
               │     Attempts 1-3: oracle-gated retry (BC still open → retry sell)
               │     Attempts 4-7: Jupiter rescue escalation (complete=True / acct_missing)
               │     Attempt 8: FINAL GATE
               │       sig sweep (check if any pending sig confirmed)
               │       token balance check
               │       outcome: recovered | reconciled_gone | manual_required
               │     After attempt 8: status="manual_required", no more auto-retries
               │     Total duration: bounded ~8 minutes
               │
               ├─⑦ SELL STUCK
               │     All attempts fail → status="sell_stuck"
               │     _sell_stuck_until[pos_id] = now + SELL_STUCK_RETRY_SEC (60s)
               │     Next reconciler cycle retries
               │
               └─⑧ RECONCILER (every 60s)
                     on-chain balance = 0 but position open
                     → close_position("reconciled_gone")  ← skips on-chain sell
                     _skip_chain_sell gate (ESCAPE incident fix)
```

## Exit Route Priority (post graduation)

```
                    graduation detected (oracle complete=True)
                           │
                    fast window (≤60s)?
                   /              \
                 YES               NO
                  │                │
           pump-amm retry      MU ladder
           (5s cadence, ≤3)    attempt 1-3
                  │            oracle-gated
                  │                │
             success?         attempt 4-7
              /    \          Jupiter rescue
            YES     NO             │
             │       │        attempt 8
           DONE    MU →       final gate
                  ladder
```

## T22 Routing (post C4)

| Flag | escalate | skip_pump_amm | Route |
|---|---|---|---|
| T22_GRAD_PUMP_AMM_PROBE_ENABLED=False, T22_GRAD_PUMP_AMM_ENABLED=False | False | True (orchestrator) | Jupiter only |
| T22_GRAD_PUMP_AMM_PROBE_ENABLED=True | True | False | pump-amm (probe), logs to t22_grad_probe.jsonl |
| T22_GRAD_PUMP_AMM_ENABLED=True | True | False | pump-amm (production) |

**C4 change**: executor.sell() no longer reads `_mint_token_program_cache` to decide T22 routing. The orchestrator (portfolio.py) reads B5 flags and passes `skip_pump_amm=True` explicitly.

## R3/R4/R5 Enforcement (exit_orchestrator.py, C3)

| Rule | Enforcement |
|---|---|
| R3: Jupiter no_route never blocks pump-amm | RouteOutcome.NO_SEND for no_route — does not set pending sig |
| R4: one venue per orchestration step | dispatch() makes exactly one executor_fn call |
| R5: SENT_PENDING blocks all venues | _global_pending_sig set → all dispatch() return NO_SEND |

## Sell-Stuck Sentinel

`_sell_stuck_until[pos_id] = monotonic + SELL_STUCK_RETRY_SEC`

On retry: `_sell_stuck_until.pop(pos_id)` → re-enters close_position.

`/sells_off` Telegram command: disables `live_sells_enabled()`, prevents new sells without closing positions (positions keep tracking).

## Known Failure Modes (historical)

| Token | Issue | Fix |
|---|---|---|
| LEAN (Jul 5) | 20+ PUMPSWAP_LOCAL fails, 28 min loop, -87.82% | MU ladder (commit 538132f) |
| ESCAPE (Jul 2026) | reconciled_gone → infinite sell loop | _skip_chain_sell gate |
| BULL (Jul 2026) | T22 NameError in executor.sell() | executor: replaced `reason` with literal |
| RETAIL (Jul 2026) | _rescue_blocks_executor=True blocked pump-amm for 4m43s | G-batch: always False |
| MetaBacteria | Graduated 42s after signal, Jupiter fired 5min later, -84% | Fast window (B4) |
