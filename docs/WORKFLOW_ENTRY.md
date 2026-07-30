# Entry Workflow Map

> Last updated: 2026-07-12 (C8 — post C0–C6/C4 patch)

## Signal → Live Buy

```
[Telegram pumpdotfunalert]
        │
        ▼
telegram_monitor.py._on_telegram_signal()
        │
        ▼
scanner.py._on_telegram_signal()
    ├── Open paper position immediately (tracks trade regardless of live outcome)
    └── Call _open_live_position() if live_buys_enabled()
               │
               ▼
        portfolio.py._open_live_position()
               │
               ├─① PREFLIGHT — get_pumpfun_curve_snapshot()
               │     complete=True / account_missing → BLOCK (already graduated)
               │     RPC error → 0.5s PP wait fallback
               │     Log: LIVE PREFLIGHT BASELINE token=... baseline=curve|pp_tick
               │
               ├─② ENTRY PROGRAM GATE (C2) — mint_classifier.classify_mint()
               │     → evaluate_live_entry_program_gate(classification, curve_obs)
               │     T22 → BLOCK (token_program="T22", reason="t22_not_supported")
               │     UNKNOWN → BLOCK (reason="unknown_program")
               │     unsupported extensions → BLOCK (is_tradeable=False)
               │     SPL → ALLOW
               │     Log to gate_blocks.csv if blocked
               │     Telemetry: E_ENTRY_GATE_CHECKED / E_ENTRY_GATE_BLOCKED
               │
               ├─③ SIZE NORMALIZATION
               │     position size adjusted by stop distance
               │     wider stop → smaller size
               │
               ├─④ BUY GATE — get_pumpfun_curve_complete()
               │     confirms still on bonding curve (not graduated)
               │
               ├─⑤ TX BUILD + SIGN
               │     _rpc_429_reset() — resets 429 accumulator
               │     EXECUTOR_BACKEND=pumpportal:
               │       local_build (SPL) or PP API fallback (T22 skip handled at gate)
               │     _buy_timing = {build_ms, sign_ms, send_ms, land_ms, rpc_429_wait_ms}
               │
               ├─⑥ TX SEND → BUY tx sent sig=...
               │
               ├─⑦ CONFIRM — _confirm_tx(sig)
               │     _rpc_429_accum() accumulates 429 stalls (C6/D6 fix)
               │     _buy_timing["rpc_429_wait_ms"] += _rpc_429_read()
               │
               ├─⑧ FILL PARSE — fill_from_tx()
               │     actual tokens received + SOL spent from on-chain receipt
               │     execution_receipts.write_receipt() (C5)
               │
               ├─⑨ SLIPPAGE CHECK
               │     fill vs Jupiter quote — if >30% above quote:
               │     → ABORT TRIPWIRE fires
               │       position stays paper; bot auto-sells tokens (recovers SOL)
               │       logged as abort_tripwire in live journal with sell_tx:
               │
               └─⑩ POSITION MARKED LIVE
                     pos.is_live = True (4E: never mutated after this point)
                     notes: live|tx:...|fill:...|slip:...
                     telemetry: E_BUY_FILL_RECORDED
                     Telegram: alert_live_buy()
```

## Timing Breakdown (avg across 11 live trades, pre-C8)

| Phase | Time | Notes |
|---|---|---|
| screen | ~2s | rugcheck + honeypot + buy pressure |
| quote | ~4s | Jupiter quote async |
| **submit** | **~12s** | PP TX build + sign + send ← only improvable bottleneck |
| confirm | ~4s | network-bound |
| **total** | **~22s** | |

## Key Gates

| Gate | Location | Action on Block |
|---|---|---|
| Curve complete (preflight) | portfolio.py:_open_live_position | Return, log PREFLIGHT BLOCK |
| Entry program gate (C2) | portfolio.py:evaluate_live_entry_program_gate | Return, gate_blocks.csv |
| Bonding curve still open | portfolio.py:_open_live_position | Return |
| Abort tripwire (>30% slip) | portfolio.py:_open_live_position | Auto-sell, paper only |
| Kill switch | kill_switch.py:live_buys_enabled | Return early |
| Epoch gate / drawdown | portfolio.py | Suspend live buys |

## C2 Program Gate Detail

```
classify_mint(token_address)
    → MintClassification{token_program: SPL|T22|UNKNOWN, is_tradeable: bool}
    → Cache: SPL/T22 permanent; UNKNOWN 60s TTL

evaluate_live_entry_program_gate(classification, curve_observation)
    → {"allowed": True}
    → {"allowed": False, "reason": str, "token_program": str}
```

Blocked cases: `T22`, `UNKNOWN`, `is_tradeable=False`, `classification is None`, `classification.error set`.
