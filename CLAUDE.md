# quant-bot — Project Context

## Behavior Rules

- **Never ask "should I do X?" after the user already asked for it.** Execute directly. Only confirm once if the action is destructive (drop DB, force push main, delete files).
- **Never ask the same yes/no question multiple times.** Execute directly for anything already requested.
- **Stress test before suggesting.** Think through failure modes before proposing anything. Never ask "should I build it?" without running the stress test first.
- **Think like management before building.** (1) Does this solve a real gap? (2) Is there a simpler way? (3) What does success look like? (4) What's the cost if it fails?
- **After any new signal/integration, verify it's architecturally independent** — not seeing another signal's output before forming its own view.
- **Server changes via git only.** Write files locally, push, server pulls. Never ask user to paste multiline code into terminal.
- **Before any go-live step, curl every external API from the VPS** — mocked unit tests don't catch Cloudflare blocks or IP restrictions.
- **Validate filters with clean data.** Remove outliers and null values before concluding a filter doesn't work.
- **Trace the full execution flow before every fix or feature** — signal→buy→monitor→TP→stop→sell→journal→wallet. Flag every blocking call and lag failure mode before writing code.
- **Responses should be short and concise.**

---

## What This Is

A research bot with two independent modules:
1. **Prediction markets** (`app/`) — scans Polymarket/Kalshi-style markets, uses Bayesian probability + Kelly sizing. Runs via GitHub Actions twice daily (9AM + 6PM UTC). Commits scan results to `logs/market_journal.csv`.
2. **Memecoin trading** (`memecoin/`) — social alert + whale copy-trade signals on Solana. Runs as a persistent web server (`python -m app.web`). **Live trading is active as of June 2026.**

---

## Infrastructure

- **Server**: Hetzner CX23, 2 vCPU / 4GB RAM, Ubuntu 22.04 — IP `178.105.94.113`
- **Service**: systemd unit `quantbot` (gunicorn, 1 worker, 4 threads, port 8080)
- **Deploy flow**: edit locally → `git push origin main` → `ssh root@178.105.94.113 'cd /root/quant-bot && git pull --rebase origin main && systemctl restart quantbot'`
- **Helius RPC**: PAID plan — never suggest upgrading. If rate-limited, diagnose root cause.
- **Primary branch**: `main`

---

## Current Trading Mode (as of July 2026)

- **`SOCIAL_ALERT_ONLY=True`** — wallet tracker, market scanner, pumpfun listener, near-miss poller are all OFF (zero Helius credits). Only the Telegram social alert feed (`pumpdotfunalert` channel) drives signals.
- **Signal type**: `social_alert` — fires when a token is mentioned in the Telegram channel
- **Live trading**: active. Each qualifying signal attempts a real on-chain buy on Solana.
- **Trade size**: ~$3–5 per trade (size-normalized based on stop distance)
- **Bot wallet**: `8PNHvFWeMT7CqpUvJiAwVgAK545t5KV3uCPd8DUfaTiM` (Solana, confirmed on Solscan)
- **BSC**: not active in current mode
- **Auto-gate**: tracks rolling PnL since epoch date; suspends live buys if drawdown threshold is hit

---

## Telegram Integration (fully live)

### Alerts sent automatically
- Position open (social_alert only)
- Live buy confirmed (with TX sig)
- Live sell confirmed (with PnL)
- TP level hit
- Kill switch triggered

### Bot commands (send to the bot in Telegram)
| Command | Effect |
|---|---|
| `/status` | Open live positions + buys/sells switch state |
| `/sells_off` | Disable all on-chain sells (positions keep tracking) |
| `/sells_on` | Re-enable on-chain sells |
| `/buys_off` | Kill switch for new buys |
| `/buys_on` | Re-enable buys |
| `/manual_sold SYMBOL [price]` | Close a position you sold manually in Phantom — stops retry loops |

**Credentials**: `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` in `/root/quant-bot/.env`
**Command listener**: background thread in `app/alerts.py`, polls `getUpdates` with long-poll timeout=30, only accepts from configured `CHAT_ID`.

---

## Key Files

```
memecoin/
  scanner.py          — background threads: portfolio monitor, reconciler, journal reconciler
  portfolio.py        — position lifecycle: open, TP, hard stop, trailing stop, close
  executor.py         — on-chain buy/sell via PumpPortal + local build; slippage gate; T22 handling
  kill_switch.py      — runtime kill switches: live_buys_enabled / live_sells_enabled
  config.py           — all tunable constants (trade sizes, stop %, slippage gates)
  screener.py         — safety filter: rugcheck, honeypot, buy pressure
  reconcile.py        — checks on-chain wallet vs open positions; closes as reconciled_gone if token gone
  journal_reconciler.py — corrects live journal entries against on-chain fill data
  bonding_curve_t22.py  — T22-native bonding curve sell path
  exit_router.py      — routes sells through correct pool (BC → PumpSwap → Jupiter)
  telegram_monitor.py — monitors pumpdotfunalert Telegram channel for social alerts
  pumpportal_monitor.py — PP subscribeNewToken: pre-screens every pump.fun launch ~200ms after creation
  health_monitor.py   — background alarms: TG silence, zero signals, drawdown, no live fills

app/
  alerts.py           — Telegram send + command listener (sells_off/on, buys_off/on, manual_sold, status)
  web.py              — Flask/gunicorn entry point; starts all background threads

logs/
  memecoin_live_journal.csv   — all live trades (real SOL in/out)
  memecoin_social_journal.csv — paper trades from social_alert signals
  memecoin_journal.csv        — all paper trades (historical)
  exit_route_attempts.csv     — per-attempt sell log (route, slip, result)
  jupiter_rescue_attempts.csv — Jupiter fallback attempts
  gate_blocks.csv             — signals blocked by entry gates with reasons
```

---

## Entry Flow (signal → live buy)

1. **Telegram social alert fires** → `telegram_monitor.py` → `_on_telegram_signal()` in `scanner.py`
2. **Paper position opened** immediately (tracks the trade regardless of live outcome)
3. **Live preflight** (`portfolio.py`):
   - Check PP cache (instant)
   - If miss: fetch curve snapshot via `get_pumpfun_curve_snapshot()` → if `complete=True` or `account_missing` → block (already graduated)
   - If RPC error → 0.5s PP wait fallback
   - Logs: `LIVE PREFLIGHT BASELINE token=... baseline=curve|pp_tick price=... elapsed_ms=...`
4. **Size normalization**: position size adjusted by stop distance (wider stop → smaller size)
5. **Buy gate**: `get_pumpfun_curve_complete()` confirms still on bonding curve (not graduated)
6. **TX sent** via PumpPortal → `BUY tx sent sig=...`
7. **Fill confirmed**: `fill_from_tx` parses on-chain receipt for actual tokens received + SOL spent
8. **Slippage check**: fill vs Jupiter quote — if >30% above quote → **abort tripwire** fires:
   - Position stays paper
   - Bot auto-sells the tokens it bought (recovers most SOL)
   - Logged as `abort_tripwire` in live journal with `sell_tx:`
9. If slippage OK → position marked live (`live|tx:...|fill:...|slip:...` in notes)

---

## Exit Flow (live sell)

1. Portfolio monitor triggers close (hard stop / trailing stop / TP / time stop)
2. `close_position(id, reason, price)` in `portfolio.py`
3. **Kill switch check**: if `live_sells_enabled()` is False → skip sell, log warning
4. **Skip sell entirely** if reason is `reconciled_gone` or `manual_sell` (no tokens to sell)
5. On-chain sell via `executor.sell()` → sell ladder (35% → 60% → 98% slippage)
6. T22 tokens: tries `bonding_curve_t22.run_bc_t22_sell()` first, falls back to PumpPortal
7. If all sell attempts fail → `sell_stuck`, arms `_sell_stuck_until` (retry after backoff)
8. Reconciler (runs every 60s): if on-chain balance = 0 but position open → `close_position("reconciled_gone")` — skips on-chain sell

---

## Known Issues / Active Constraints

### Token-2022 (T22) tokens
Pump.fun tokens using the Token-2022 program have a different pool layout. The bot handles them in "canary mode" — buys work, but sells through `pumpswap_local` fail with `pumpswap_bad_pool_layout`. On graduation/migration, the T22 native sell path (`bonding_curve_t22`) is the primary route.

### Helius rate limits
Helius (paid plan) rate-limits under heavy load. Bot falls back to `mainnet-beta.solana.com`. Public mainnet-beta can return `0` for `getBalance` when itself rate-limited — this is an RPC artifact, not a real zero balance. Check Solscan directly for authoritative wallet balance.

### SOCIAL_ALERT_ONLY mode
With this flag True, only the Telegram social feed drives signals. Whale wallet polling, market scanner, pumpfun listener, and near-miss poller do NOT run. Helius credits are at zero — do not make changes that would increase Helius RPC call volume.

---

## Recent Incidents & Fixes (July 2026)

### ESCAPE token — infinite sell loop (fixed)
**Root cause**: `close_position("reconciled_gone")` was falling into the on-chain sell block. Sell failed every time (T22 bad pool layout) → re-armed `sell_stuck` → reconciler fired again 60s later → infinite loop.
**Fix** (`portfolio.py`): `_skip_chain_sell = reason in ("reconciled_gone", "manual_sell")` — these reasons now bypass the sell block entirely.
**Also added**: `/manual_sold SYMBOL [price]` Telegram command so manually-sold positions can be cleanly closed without triggering the loop.

### BULL token — abort tripwire + T22 NameError (fixed)
**Root cause 1**: Token pumped 48% between Jupiter quote and fill. Abort tripwire (30% gate) fired → position stayed paper, bot auto-sold. Net loss: −$0.58.
**Root cause 2**: T22 native sell path threw `NameError: name 'reason' is not defined` inside `executor.sell()` (which has no `reason` param). Bot caught exception and fell back to PumpPortal (sell succeeded), but T22 path was silently skipped.
**Fix** (`executor.py`, commit `6bcbc95`): replaced `reason` with literal `"auto_sell"` in the T22 sell call.

### Sell kill switch (added)
`kill_switch.py` now has an independent sell switch (`_live_sells_enabled`). `/sells_off` in Telegram disables all on-chain sells without stopping position tracking. `/sells_on` re-enables. Useful if you manually sell in Phantom and want to prevent the bot from also trying to sell.

---

## Notable Trade History

- **SAM** (April 2026) — best trade. New launch +839%, copy trade +216%. Deployer: `CAUbSmiNuj16phNiskMdwWZEAUXCfXaUSamDFyf7pAa6`
- **STOCK BULL** (July 2026) — hard stop at −46%, sell confirmed on-chain. Correct behaviour.
- **ESCAPE** (July 2026) — live buy, manual sell in Phantom, infinite loop bug (now fixed).
- **BULL** (July 2026) — abort tripwire at +48% slippage, auto-sold, −$0.58 net.
- Social alert overall: live trading started June 2026 at $5/trade.

---

## Environment Variables (on VPS in `/root/quant-bot/.env`)

| Var | Purpose |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Bot alerts + command listener |
| `TELEGRAM_CHAT_ID` | Your personal chat ID (security gate) |
| `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` | MTProto client for monitoring `pumpdotfunalert` |
| `HELIUS_API_KEY` | Solana RPC (paid plan — do not increase usage) |
| `SOLANA_PRIVATE_KEY` | Trading wallet private key |
| `LIVE_BUYS_ENABLED` | Master live buy flag in config |
