# quant-bot — Project Context

## What This Is
A research bot with two independent modules:
1. **Prediction markets** (`app/`) — scans Polymarket/Kalshi-style markets, uses Bayesian probability + Kelly sizing. Runs via GitHub Actions twice daily (9AM + 6PM UTC). Commits scan results to `logs/market_journal.csv`.
2. **Memecoin trading** (`memecoin/`) — copy trades whale wallets on Solana and BSC, with safety screening, trailing stops, and paper trade tracking. Runs as a persistent web server (`python -m app.web`).

## Infrastructure
- Currently hosted on **Oracle Cloud free tier** (4 vCPU ARM, 16GB RAM) — but processes get OOM-killed under load, making it unreliable for 24/7 operation.
- **Plan: migrate to Hetzner CX23** (2 vCPU Intel/AMD, 4GB RAM, €3.99/month). More than enough for this workload. Use Primary IPv4 (not IPv6 only).
- The memecoin scanner is **currently not running** — last journal entry was April 29 2026. Needs to be started with `nohup python -m app.web > logs/web.log 2>&1 &`.

## Memecoin Module — Key Files
```
memecoin/
  scanner.py        — 3 background threads: wallet poller, market scanner, portfolio monitor
  wallet_tracker.py — polls Solana (~300 wallets) + BSC (~70 wallets) for whale buys/sells
  signals.py        — generates copy_trade / volume_breakout / new_launch / dev_launch signals
  screener.py       — safety filter: liquidity, rugcheck, honeypot, buy pressure
  portfolio.py      — paper trade positions, exit logic (hard stop / trailing stop / whale exit / time stop)
  dev_tracker.py    — NEW: tracks deployer wallets of profitable tokens
  candidate_log.py  — NEW: two-tier logging system (see below)
  config.py         — all tunable constants, trade sizes, stop levels

memecoin/data/
  whale_wallets_sol.json  — ~300 Solana whale wallets
  whale_wallets_bnb.json  — ~70 BSC whale wallets
  dev_wallets.json        — auto-populated: deployers of profitable tokens (starts empty)
  dev_last_seen.json      — polling state for dev wallets

logs/
  memecoin_journal.csv    — all closed trades (584 entries as of last run)
  signal_candidates.csv   — NEW: every copy_trade signal that passes the filter, logged at signal time
  winners_journal.csv     — NEW: only profitable closes, with full entry snapshot + exit outcome
```

## Recently Built Features (May 2026)

### Two-Tier Signal Logging
Goal: build a dataset to reverse-engineer what winning copy trades look like at signal time, so the user can eventually develop their own entry criteria without needing whale wallets.

- **`signal_candidates.csv`** — every copy_trade signal fires → full market snapshot logged (safety score, momentum, liquidity, age, buy/sell pressure, whale tiers, price action). All strengths including weak ones the bot doesn't trade.
- **`winners_journal.csv`** — when a position closes profitably → that signal's entry snapshot + exit price, PnL, peak price promoted here permanently.
- Hook: `scanner.py → _add_signal()` logs candidates; `portfolio.py → close_position()` promotes winners.

### Dev Wallet Tracker
Goal: learn the deployer wallets of profitable tokens so the bot can fire signals when a known winner dev launches a new token — getting in before any whale.

- When a profitable trade closes → background thread finds the token's deployer via Solana RPC / BscScan → saves to `dev_wallets.json` with win count, avg PnL, reliability score (0–1).
- Market scanner polls dev wallets every cycle for new pump.fun deployments → fires `dev_launch` signal.
- Signal strength scales: 1 win = weak | 2 wins or avg PnL > 100% = medium | 3+ wins or avg PnL > 200% = strong.
- New signal type `dev_launch` has its own trade settings in `config.py` (€40 size, -35% hard stop).

## Strategy Intent (User's Goal)
The user wants to develop their own independent trading criteria by studying patterns in `winners_journal.csv` vs `signal_candidates.csv`. Key fields to compare over time: `age_minutes`, `liquidity_usd`, `buy_sell_ratio_5m`, `momentum_score`, `whale_tiers`, `price_change_5m`. The dev tracker adds a second layer: instead of copying whales, eventually follow known profitable devs at deployment.

## Notable Trade History
- **SAM token** (April 23 2026) — best trade in journal. New launch at $0.000112 → +$251 (+839%). Copy trade entry at $0.000334 → +$64 (+216%). Both exited via trailing stop at $0.001054. Deployer wallet: `CAUbSmiNuj16phNiskMdwWZEAUXCfXaUSamDFyf7pAa6` (named "swagg (potion)" in whale list).
- BSC chain: 18 trades logged but PnL not recording correctly for most — exit_reason column has price data instead of reason strings. Needs investigation.
- Copy trade overall: 34 trades, 76.5% win rate, +$22 total PnL on $30 position sizes.

## Telegram Alerts
`app/alerts.py` is scaffolded but not configured. User wants to set this up so signals and exits push to Telegram for passive monitoring from phone.

## Active Branch
All recent development is on `claude/mobile-quant-bot-status-tS0za`. Not yet merged to main.
