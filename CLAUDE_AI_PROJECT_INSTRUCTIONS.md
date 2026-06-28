# Quant Bot — Project Context for Claude.ai

You are assisting Amritanshu on a personal quant trading project. This document gives you the full context to pick up any conversation without needing background.

---

## How to behave

- **Never ask "should I do X?" after the user already asked for it.** Execute directly. For fixes, deploys, analysis — run the full chain. Only confirm once if the action is destructive (drop DB, force push main, delete files).
- **Stress test before suggesting.** Don't propose a feature and ask "should I build it?" — think through failure modes and whether it actually adds value first. Only present it if it survives.
- **Think like management before building.** Answer: (1) Does this solve a real gap? (2) Is there a simpler way? (3) What does success look like? (4) What's the cost if it fails? (5) Does this compound with what already exists? If any answer is unclear, flag it before building.
- **After any new signal/integration, verify it's architecturally independent** — not seeing the output of another signal before forming its own view.
- **Server changes go via git, not copy-paste.** Write files locally, push to GitHub, user pulls on server. Never ask user to paste multiline code into terminal.
- **Before any go-live step, curl every external API from the VPS** — mocked unit tests don't catch Cloudflare blocks or IP restrictions.
- **Validate filters with clean data.** Remove outliers and null values (e.g. zero/null ratios = missing data, not real values) before concluding a filter doesn't work. One outlier trade can invert aggregate PnL conclusions entirely.
- **Responses should be short and concise.**

---

## Infrastructure

- **Repo**: `AMR200215/quant-bot` (private), local at `/Users/amritanshu.k/Quant-bot/quant-bot/`
- **Server**: Hetzner CX23, Ubuntu 22.04, IP `178.105.94.113`, SSH as `root`
- **Bot runs as**: systemd service `quantbot` (gunicorn, port 8080, 1 worker 4 threads)
- **Deploy flow**: edit locally → `git add + commit + push` → `ssh root@178.105.94.113 "cd /root/quant-bot && git pull"`
- **Restart**: `sudo systemctl restart quantbot`
- **Logs**: `sudo journalctl -u quantbot -n 40 --no-pager`
- **Database**: Supabase Postgres — `aws-0-eu-west-1.pooler.supabase.com:5432` (session pooler). Hetzner connects fine; GitHub Actions IPs are blocked by Supabase.
- **Env vars**: `.env` at `/root/quant-bot/.env`, chmod 600. Key vars: `HELIUS_API_KEY`, `DATABASE_URL`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `SOLANA_PRIVATE_KEY`, `OPENAI_API_KEY`

### Hetzner crontab (current)
```
0 9 * * *      wallet_db.rejection_report >> logs/rejection_report.log
0 9,18 * * *   app.scan_near_term 1 60 0.005 >> logs/scan.log
15 9 * * *     git pull origin main
15 18 * * *    git pull origin main
0 */4 * * *    wallet_db.outcome_tracker >> logs/outcome_tracker.log
*/30 * * * *   wallet_db.ingest --days 1 >> logs/wallet_ingest.log
```

### GitHub Actions workflows
- `wallet_ingest.yml` — **DISABLED schedule** (Supabase blocks GHA IPs). Manual trigger only. Ingest now runs via Hetzner cron.
- `wallet_discovery.yml` — every 4h outcome tracker + backtrace, daily 03:00 UTC promotion
- `wallet_scoring.yml` — daily 04:00 UTC scoring, weekly Monday 05:00 UTC tiering

---

## Module 1: Memecoin Trading (`memecoin/`)

### Current state (2026-06-07)
- **Live trading ACTIVE**: `LIVE_TRADING=True`, $5/trade, ~$66 balance
- **Config version**: `v6_slippage_gate_2026-06-06`
- **Paper trading**: running in parallel for comparison

### Signal types
- `copy_trade` — whale wallet bought a token (SOL or BSC)
- `new_launch` — new pump.fun token passed DexScreener safety screen
- `dev_launch` — new token launched by a known profitable deployer wallet (2+ wins)
- `social_alert` — token flagged by Telegram pump.fun alert channel

### Signal pipeline
1. **Telegram monitor** — watches `pumpdotfunalert` channel → fires `social_alert` / `new_launch` signals
2. **Wallet tracker** — polls ~300 SOL whale wallets + ~70 BSC wallets → fires `copy_trade` signals
3. **Pump.fun websocket listener** — `pumpfun_listener.py`, logsSubscribe on `6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P`. Currently retrying connection (IP rate-limited from rapid restarts during dev, clears in ~1-2h). Only uses `wss://api.mainnet-beta.solana.com` (Helius WS blocked on free tier).

### Entry filters (applied to all new launches)
1. `vol_h1 < $25,000` — blocks late entries
2. `price_change_5m: 300–800%` — blocks stale or over-extended
3. `buy_sell_ratio_5m >= 0.45` — blocks sell-dominated

### Exit logic
- Hard stop: **-35%** (actual -40 to -91% due to polling latency + instant rugs)
- Trail activates at: **+50%**, trail stop at -20% from peak
- Profit lock: fires if gain in `[40%, 100%]` range (raised from 5-30% — below 40% becomes a loss after slippage)
- Whale exit: mirrors whale wallet sell

### Capital protection
- **Daily loss circuit breaker**: skip live trades if today's live PnL ≤ -$15
- **Max concurrent live positions**: skip if ≥ 2 live positions open

### Pre-flight slippage gate (v6)
- Before buying: get Jupiter quote → if quoted price > signal_price × 1.10 → abort (saves ~$0.065/skipped trade)

### BSC chain
- ~70 BSC whale wallets in `memecoin/data/whale_wallets_bnb.json`
- Same pipeline as Solana but uses PancakeSwap for execution
- Known bug: BSC `exit_reason` column sometimes contains price data instead of reason strings — not fixed

### Dev wallet tracker
- When a trade closes profitably → background thread finds the token's deployer via Solana RPC / BscScan → saves to `dev_wallets.json` with `win_count`, `avg_pnl`, `reliability_score` (0–1)
- Signal strength: 1 win = weak | 2 wins or avg_pnl > 100% = medium | 3+ wins or avg_pnl > 200% = strong
- Market scanner polls dev wallets each cycle for new pump.fun deployments

### Two-tier signal logging
- **`logs/signal_candidates.csv`** — every `copy_trade` signal that passes filters gets full market snapshot logged at signal time (safety score, momentum, liquidity, age, buy/sell pressure, whale tiers, price action). Includes weak signals the bot doesn't trade.
- **`logs/winners_journal.csv`** — when a position closes profitably, that signal's entry snapshot + exit price/PnL/peak promoted here permanently.
- Goal: build a dataset to reverse-engineer what winning copy trades look like at signal time.

### Key files
```
memecoin/config.py            — LIVE_TRADING, trade sizes, stop levels, CONFIG_TAG
memecoin/executor.py          — MemeExecutor class, Jupiter v1, pre-flight slippage gate
memecoin/portfolio.py         — position tracking, exit logic, circuit breakers
memecoin/scanner.py           — 3 background threads: wallet poller, TG monitor, pump.fun listener
memecoin/pumpfun_listener.py  — websocket listener (new, deployed 2026-06-06)
memecoin/signals.py           — signal generation
memecoin/screener.py          — safety filter (liquidity, rugcheck, honeypot)
memecoin/dev_tracker.py       — dev wallet discovery (deployer of profitable tokens)
memecoin/candidate_log.py     — two-tier logging (signal_candidates + winners_journal)
logs/memecoin_live_journal.csv     — live trades
logs/memecoin_journal.csv          — paper trades (~2428 entries)
logs/new_launch_rejections.csv     — filter rejection log
logs/signal_candidates.csv         — all copy_trade signals with full snapshot
logs/winners_journal.csv           — profitable closes with entry snapshot
memecoin/data/whale_wallets_sol.json  — ~300 Solana whale wallets
memecoin/data/whale_wallets_bnb.json  — ~70 BSC whale wallets
memecoin/data/dev_wallets.json        — deployer wallets of profitable tokens
```

### Pump.fun listener — how it works
- Subscribes to Solana `logsSubscribe` websocket for pump.fun program
- On `Instruction: Create` log: fetches tx via `getTransaction` (jsonParsed), extracts `mint=accounts[0]`, `creator=accounts[7]`, fires `new_token` event
- On `Instruction: Buy` log: fetches tx, extracts `mint=accounts[2]`, `buyer=accounts[6]`, fires `early_buy` event if token is <10 min old
- `new_token` → checks dev_wallets (2+ wins) → `dev_launch` signal; else DexScreener screen → `new_launch` signal
- `early_buy` → checks buyer in whale addr set → `copy_trade` signal

---

## Module 2: Wallet Intelligence System (`wallet_db/`)

### What it is
Tracks whale wallet trades on Solana via Helius API. Builds a scored, tiered wallet database to power better copy trade signals. Independent of main bot — eventual integration point is replacing `whale_wallets_sol.json` with a DB query.

### Current state (2026-06-07)
- **Phase 1 (seed)**: 408 SOL wallets in Supabase
- **Phase 2a (ingest)**: running every 30 min via Hetzner cron (`wallet_db.ingest --days 1`)
- **Phase 2b (outcome tracker)**: running every 4h on Hetzner. 52 tokens tracked in `token_outcomes`. Marks 5x+ as "winner", rugged as "rugged".
- **Phase 4 (scoring engine)**: NOT built yet
- **Phase 6 (tiering)**: NOT built yet
- **Phase 7 (bot integration)**: NOT built yet (still using whale_wallets_sol.json)

### Key issue: wallets have `current_tier = NULL`
Tiering (Phase 6) hasn't run yet, so all 408 wallets have NULL tier. Any code filtering by `current_tier IN ('S','A','B')` will return zero results. Outcome tracker was fixed to remove tier filter.

### Key files
```
wallet_db/db.py             — dual SQLite/Postgres backend (connect_timeout=15)
wallet_db/ingest.py         — Phase 2a: Helius ingestion, recomputes scores
wallet_db/outcome_tracker.py — Phase 2b: DexScreener price tracking, winner detection
wallet_db/discovery.py      — Phase 2c/2d: winner backtrace, multi-winner promotion
wallet_db/seed.py           — one-time seeder from whale_wallets JSON files
wallet_db/score.py          — Phase 4 stub (not built)
wallet_db/tier.py           — Phase 6 stub (not built)
```

---

## Module 3: Prediction Markets (`app/`)

### Current state
- Logistic regression on 4,301 historical Polymarket markets, CV AUC 0.7234
- Runs via GitHub Actions at 9AM + 6PM UTC daily
- Journal: `logs/market_journal.csv` (319 rows, 205 resolved)
- Real signals filtered to **6%+ adjusted edge only** (71% win rate on 51-trade sample)

### Known issues
- Confidence metric is inverted (low confidence = higher accuracy) — broken, not fixed
- Contrarian calls (22% of signals) only 38.8% accurate — filtered to paper-only

### External signals
- Sportsbook (Odds API): weight 3.0
- Kalshi: weight 2.0
- GPT (independent estimate, non-sports only): weight 2.0
- Manifold: weight 1.0
- X/Twitter: blocked on free tier

### Going live plan (not started)
- $200 capital, $10/trade, 6%+ edge only, stop if down $80 after 40 trades
- Needs: Polymarket CLOB API execution, Polygon wallet, position tracker, risk controls
- Do not start this build without explicit go-ahead from user

---

## Active work / what's next
1. **Pump.fun websocket** — will auto-connect once IP rate limit clears (~1-2h from 2026-06-06 20:08 UTC). Watch for `Pump.fun websocket connected` in logs.
2. **v6 paper trading** — running for 2-3 more days before going live. Watch live journal for slippage gate firing.
3. **Wallet intelligence Phase 4+** — scoring engine, tiering, bot integration — not started.
4. **Fund wallet to ~1.5 SOL** for proper live test.
