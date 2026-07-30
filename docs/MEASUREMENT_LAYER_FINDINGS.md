# Measurement Layer Findings — 11 Live Trades (Jul 2026)

> Source: syslog analysis (`/var/log/syslog` on VPS `178.105.94.113`)  
> Telemetry CSV (`logs/trade_telemetry_summary.csv`) deployed 2026-07-08 — all rows currently `pending` (no completed traces yet). Timing data below is from syslog parsing, not from the CSV.

---

## Entry Latency Breakdown (average across 11 trades)

| Phase | Time | % of total |
|---|---|---|
| screen | ~2s | ~9% |
| quote (Jupiter) | ~4s | ~18% |
| **submit (PumpPortal TX)** | **~12s** | **55%** |
| confirm (fill parse) | ~4s | ~18% |
| **total** | **~22s** | 100% |

**Submit is the only improvable bottleneck.** Screen and quote are fast. Confirm is network-bound. Submit (PP TX round-trip) is where latency lives.

---

## PumpPortal Price Signal — Never Warm

- `pp_sig=$0` for **all 11 trades**
- Every entry falls back to DexScreener baseline (`src=dex`)
- PP price only populates after a token has traded volume — by that point the bot has already submitted the TX
- **Implication**: preflight baseline is always DexScreener, never PP tick. Abort tripwire compares fill against DexScreener price, not a real-time signal.

---

## Abort Tripwire Calibration

- Fires when fill > preflight_baseline by >30%
- UBULL: fired at **35.9% above baseline** (just over the gate)
- BULL: fired at **48% above baseline**
- Both were genuine pump events between quote and fill — the tripwire worked correctly
- **Risk**: 30% gate may be too tight for volatile launches. Tokens that pump 35–50% between quote and fill and then continue up are being aborted and missing the move.

---

## Graduation Timing — MetaBacteria

- Token graduated (bonding curve → PumpSwap) in **42 seconds** after signal
- MU retry ladder: retries 1-3 at ~1 min intervals → Jupiter escalation at retry 4
- Jupiter fired **~5 minutes** after graduation → token had already crashed **-84%**
- **Root cause**: MU ladder timing is fixed at ~1 min intervals regardless of graduation speed. Fast-graduating tokens bleed out before Jupiter fires.

---

## Exit Route Attempts (see `logs/exit_route_attempts.csv`)

- 181 sell attempts logged
- Primary failure modes: `pumpswap_bad_pool_layout` (T22 tokens), `MIGRATION_UNCERTAIN` loop (pre-fix), `sell_stuck` timeouts
- LEAN: 20+ failed `PUMPSWAP_LOCAL` attempts over 28 minutes before Jupiter fallback (pre-MU ladder fix)
- Post-fix (commit 538132f): MU ladder bounds total duration to ~8 minutes

---

## Gate Blocks (see `logs/gate_blocks.csv`)

- 119 signals blocked since live trading started
- Most common block reasons: rugcheck fail, buy pressure below threshold, honeypot flag
- No analysis yet on how many blocked tokens would have been profitable (counter-factual gap)

---

## Phase 4 Bugs Fixed (commit 27fbc12, deployed 2026-07-09)

| Bug | Root cause | Fix |
|---|---|---|
| **4B** — Jupiter rescue journal miss | `_finalize_rescue_sell` assigned to read-only `pnl_pct`/`pnl_usd` properties → `AttributeError` → journal never written even though TX confirmed on-chain | Removed property assignments; journal now writes correctly |
| **4C** — TP duplicate dispatch | No inflight guard on TP levels — concurrent price ticks could fire same TP twice | `_tp_inflight` dict with `float("inf")` sentinel + 30s failure cooldown |
| **4D** — MU infinite retry | No exit condition after attempt 8 | `mu_sell_total >= 8` → `status="manual_required"`, pops from `_sell_stuck_until` |
| **4E** — Live journal silent skip | MU retry loop mutated `pos.notes`, stripping `"live|tx:"` prefix → old routing check returned False → live journal write skipped | `pos.is_live` flag (set at buy time, never mutated) replaces notes-based check |
| **4F** — 7 missing journal rows | 3 distinct bugs (4B, 4E, journal migration race) caused rows to never be written | Backfilled manually with on-chain data, tagged `journal_backfilled:4F` |

---

## 7 Backfilled Rows (4F)

All rows now in `logs/memecoin_live_journal.csv` with tag `journal_backfilled:4F` and `bug:` prefix on exit_reason. These were missing due to the above bugs and a journal migration race at 01:33:57 UTC where `_ensure_journal_header` rewrote the file but the subsequent append silently failed.

---

## Notable Trade Outcomes (all 11)

| Token | Outcome | PnL | Exit reason | Notes |
|---|---|---|---|---|
| SAM | +839% (new launch) / +216% (copy trade) | Best trade | manual | April 2026, pre-telemetry |
| LEAN | -87.82% (-$2.63) | Hard loss | sell_stuck→finally sold | 20+ failed sell attempts, 28 min loop, pre-MU-fix |
| ESCAPE | manual sell | n/a | reconciled_gone | Phantom manual sell, `_skip_chain_sell` bug fixed |
| BULL | -$0.58 | abort_tripwire | 48% slippage above baseline, auto-sold | T22 NameError also fixed |
| STOCK BULL | -46% | hard_stop | Correct behaviour, sell confirmed on-chain |
| UBULL | abort_tripwire | - | 35.9% above baseline |
| MetaBacteria | -84% | time/migration | Graduated in 42s, Jupiter 5 min late |
| + 4 others | various | hard_stop / sell_failed | In live journal, pre-telemetry |

---

## What the Telemetry Layer Captures (schema)

Fields in `trade_telemetry_summary.csv`:
- `trace_id`, `pos_id`, `mint`, `symbol`, `live_or_paper`
- Entry phases: `ts_alert`, `ts_screen_done`, `ts_quote_done`, `ts_submit`, `ts_confirm`
- Computed: `screen_ms`, `quote_ms`, `submit_ms`, `confirm_ms`, `total_ms`
- Slippage: `real_slip`, `total_slip`, `artifact` (DexScreener price lag flag)
- Exit phases: `ts_exit_trigger`, `ts_sell_sent`, `ts_sell_confirmed`
- `sell_route`, `sell_attempts`, `exit_reason`

All 2,669 current rows are `pending` — telemetry was instrumented on 2026-07-08 but no completed trades have gone through since deployment.

---

## What to Build Next (open questions for Fable 5)

1. **Submit latency**: Can PP TX submission be parallelised or pre-warmed? 12s is the ceiling to break.
2. **Abort tripwire calibration**: Is 30% the right gate? Backtest against gate_blocks + social_journal to find the PnL-optimal threshold.
3. **Graduation speed detection**: Can we detect fast-graduating tokens and escalate to Jupiter immediately (skip retries 1-3)?
4. **PP price warming**: Is there a way to get a live PP tick before fill, or should we accept DexScreener-only baseline?
5. **Gate block counter-factual**: Of the 119 blocked signals, how many would have been profitable? Is the rugcheck filter over-fitted?
6. **First completed telemetry trace**: Once a trade closes post-July-9, the `trade_telemetry_summary.csv` will have a real row. That's the first ground-truth timing data point.
