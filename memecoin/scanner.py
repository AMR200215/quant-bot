"""
Main scan loop for the memecoin module.

Runs two background threads:
  - wallet_thread   : polls whale wallets every 30s (SOL) / 60s (BNB)
  - market_thread   : scans DexScreener / GMGN every 2 min for breakouts/launches

All discovered signals are stored in a shared signal queue and written
to data/memecoin_signals.json so the web UI can read them.
"""

import csv
import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from datetime import date
from pathlib import Path
from typing import Optional

from memecoin.config import (
    SIGNALS_FILE, DATA_DIR,
    SOL_WALLET_POLL_SEC, BNB_WALLET_POLL_SEC, DEXSCREENER_POLL_SEC,
    MIN_LIQUIDITY_USD, MAX_AGE_MINUTES_NEW,
    MIN_BUY_SELL_RATIO_SOCIAL, MIN_VOL_5M_SOCIAL, MAX_VOL_5M_SOCIAL,
    MAX_VOL_H1_SOCIAL, MAX_PRICE_CHANGE_5M_SOCIAL,
)
from memecoin.data_client import (
    dex_get_new_pairs, dex_get_boosted, gmgn_new_sol, gmgn_trending_sol,
    dex_get_token,
)
from memecoin.screener import screen_token
from memecoin.wallet_tracker import (
    load_all_wallets, build_wallet_ranks, tier_for_wallet,
    poll_sol_wallets_batch, poll_bnb_wallets_batch, WalletEvent,
)
from memecoin.portfolio import Portfolio
from memecoin.candidate_log import (
    log_signal_candidate, log_new_launch_rejection,
    track_near_miss, update_near_miss_check, load_near_miss_data,
)
from memecoin.dev_tracker import poll_all_devs, dev_signal_strength
from memecoin.signals import (
    Signal, make_copy_trade_signal, make_volume_breakout_signal,
    make_new_launch_signal, make_dev_launch_signal, make_social_alert_signal,
)
from app import alerts
from memecoin.pumpportal_monitor import monitor as _pp_monitor


class _NoDexData(Exception):
    """Raised by _on_telegram_signal when DexScreener has no data yet.
    TelegramMonitor catches this and schedules a retry."""
    pass

log = logging.getLogger(__name__)

# Max signals kept in memory / JSON
MAX_SIGNALS = 200

# ---------------------------------------------------------------------------
# Shared state (thread-safe via lock)
# ---------------------------------------------------------------------------

_lock       = threading.Lock()
_signals    = deque(maxlen=MAX_SIGNALS)   # most recent first
_whale_sells: dict[str, list[str]] = defaultdict(list)  # token_addr → wallets that sold

portfolio = Portfolio()

# de-duplicate signals: don't fire same token+type within 15 min
_seen: dict[str, float] = {}   # f"{chain}:{address}:{type}" → timestamp

# per-token-per-day blacklist: once a position is opened on a token today, block all
# re-entries for the rest of the day — prevents re-entering dying tokens after close
_traded_today: dict[str, str] = {}   # f"{chain}:{address}" → ISO date string


# ---------------------------------------------------------------------------
# PumpPortal screening accumulator
# ---------------------------------------------------------------------------

# Seconds after subscribe at which we check entry conditions
_SCREENING_CHECK_TIMES = (30, 60, 120)
# Give up after this many seconds with no entry
_SCREENING_TIMEOUT = 180
# Minimum unique buyers before considering an entry
_SCREENING_MIN_BUYERS = 8
# CSV file for screening outcome diagnostics
_SCREENING_OUTCOMES_FILE = DATA_DIR / "screening_outcomes.csv"
_SCREENING_OUTCOME_FIELDS = [
    "ts", "mint", "chain", "elapsed_s",
    "unique_buyers", "buy_count", "sell_count", "net_sol",
    "creator_sold", "price_first", "price_last", "entry_made",
]

# mint → {ts, chain, checks_done}
_screening_queue: dict[str, dict] = {}
_sq_lock = threading.Lock()


def _screening_conditions_met(state) -> bool:
    """Return True if PumpPortal-screened token meets paper-entry conditions."""
    price_ok = (
        state.first_seen_price <= 0                              # no reference yet
        or state.latest_price >= state.first_seen_price * 0.90  # not >10% below entry
    )
    return (
        state.unique_buyer_count >= _SCREENING_MIN_BUYERS
        and state.net_sol_inflow > 0
        and not state.creator_sold
        and price_ok
    )


def _log_screening_outcome(mint: str, entry: dict, state, entry_made: bool):
    try:
        _SCREENING_OUTCOMES_FILE.parent.mkdir(parents=True, exist_ok=True)
        write_hdr = (
            not _SCREENING_OUTCOMES_FILE.exists()
            or _SCREENING_OUTCOMES_FILE.stat().st_size == 0
        )
        with open(_SCREENING_OUTCOMES_FILE, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_SCREENING_OUTCOME_FIELDS)
            if write_hdr:
                w.writeheader()
            elapsed = time.time() - entry["ts"]
            w.writerow({
                "ts":            time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(entry["ts"])),
                "mint":          mint,
                "chain":         entry["chain"],
                "elapsed_s":     round(elapsed, 1),
                "unique_buyers": state.unique_buyer_count if state else 0,
                "buy_count":     state.buy_count          if state else 0,
                "sell_count":    state.sell_count         if state else 0,
                "net_sol":       round(state.net_sol_inflow, 4) if state else 0,
                "creator_sold":  state.creator_sold       if state else "",
                "price_first":   state.first_seen_price   if state else 0,
                "price_last":    state.latest_price       if state else 0,
                "entry_made":    entry_made,
            })
    except Exception as e:
        log.debug("screening_outcomes log error: %s", e)


def _fire_screening_entry(chain: str, mint: str, state):
    """Open a paper-only position from PumpPortal screening data."""
    price       = state.latest_price or state.first_seen_price
    total_txns  = state.buy_count + state.sell_count
    age_min     = (time.time() - state.first_seen_ts) / 60

    # Minimal screen dict — no DexScreener data available
    screen = {
        "passed":             True,
        "chain":              chain,
        "price_usd":          price,
        "volume_5m":          0,
        "volume_h1":          0,
        "volume_h24":         0,
        "liquidity_usd":      0,
        "mcap_usd":           0,
        "fdv":                0,
        "age_minutes":        round(age_min, 2),
        "buy_sell_ratio_5m":  state.buy_count / max(total_txns, 1),
        "price_change_5m":    0,
        "price_change_1h":    0,
        "price_change_6h":    0,
        "buys_5m":            state.buy_count,
        "sells_5m":           state.sell_count,
        "buys_h1":            0,
        "sells_h1":           0,
        "buy_sell_ratio_h1":  0,
        "dex_id":             "pumpfun",
        "dexscreener_url":    f"https://dexscreener.com/solana/{mint}",
        "has_twitter":        False,
        "has_telegram":       False,
        "has_website":        False,
        "rugcheck_score":     0,
        "buy_tax":            0,
        "sell_tax":           0,
        "pair":               {},
    }
    sig = make_social_alert_signal(chain, mint, screen,
                                   source="pumpportal_screen",
                                   channel="pumpdotfunalert")
    if sig is None:
        return
    # Paper-only gate: override signal_type so live gate in portfolio.py
    # (requires signal_type=="social_alert") does not trigger.
    # These signals are paper-only for the 7-day evaluation window.
    sig.signal_type = "pumpportal_screen"
    sig.notes += (f" | screen: buyers={state.unique_buyer_count}"
                  f" net_sol={state.net_sol_inflow:.3f}")

    _add_signal(sig)
    log.warning(
        "SCREEN ENTRY %s — buyers=%d net_sol=%.3f price=$%.8f",
        mint[:8], state.unique_buyer_count, state.net_sol_inflow, price,
    )


def _run_screening_checks():
    """
    Called every portfolio-poll cycle (~2s).
    Checks PumpPortal-screened tokens at T+30/60/120s; evicts at T+180s.
    """
    now = time.time()
    to_evict = []

    with _sq_lock:
        items = list(_screening_queue.items())

    for mint, entry in items:
        elapsed     = now - entry["ts"]
        checks_done = entry["checks_done"]

        # Run the next scheduled check if it's time
        if checks_done < len(_SCREENING_CHECK_TIMES):
            check_at = _SCREENING_CHECK_TIMES[checks_done]
            if elapsed >= check_at:
                state = _pp_monitor.get_screening_state(mint)
                if state and _screening_conditions_met(state):
                    _fire_screening_entry(entry["chain"], mint, state)
                    _log_screening_outcome(mint, entry, state, entry_made=True)
                    to_evict.append(mint)
                    continue
                # Conditions not met yet — advance check counter
                with _sq_lock:
                    if mint in _screening_queue:
                        _screening_queue[mint]["checks_done"] += 1

        # Hard timeout — give up
        if elapsed >= _SCREENING_TIMEOUT:
            if mint not in to_evict:
                state = _pp_monitor.get_screening_state(mint)
                _log_screening_outcome(mint, entry, state, entry_made=False)
                to_evict.append(mint)

    for mint in to_evict:
        with _sq_lock:
            _screening_queue.pop(mint, None)
        _pp_monitor.evict_screening({mint})


def _mark_traded(chain: str, address: str):
    """Record that we've opened a position on this token today."""
    if address:
        _traded_today[f"{chain}:{address}"] = date.today().isoformat()


def _is_duplicate(chain: str, address: str, sig_type: str) -> bool:
    # Per-type cooldown: same signal type on same token within 15 min
    key = f"{chain}:{address}:{sig_type}"
    now = time.time()
    last = _seen.get(key, 0)
    if now - last < 900:   # 15 min cooldown
        return True
    _seen[key] = now

    # Cross-type dedup: if we already have an open position in this token,
    # skip regardless of signal type. Prevents new_launch + social_alert +
    # copy_trade all opening positions on the same token simultaneously.
    open_addrs = {p.token_address for p in portfolio.open_positions()}
    if address in open_addrs:
        return True

    # Per-token-per-day blacklist: skip tokens we've already traded today,
    # even after the prior position closed. Prevents re-entry on dying tokens.
    today = date.today().isoformat()
    if _traded_today.get(f"{chain}:{address}") == today:
        return True

    return False


# ---------------------------------------------------------------------------
# Signal storage
# ---------------------------------------------------------------------------

def _persist_signals():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _lock:
        data = [s.to_dict() for s in _signals]
    SIGNALS_FILE.write_text(json.dumps(data, indent=2))


def _add_signal(sig: Optional[Signal]):
    if not sig:
        return
    if _is_duplicate(sig.chain, sig.token_address, sig.signal_type):
        return
    with _lock:
        _signals.appendleft(sig)
    # log copy trade signals to temp candidates for strategy research
    if sig.signal_type == "copy_trade":
        try:
            log_signal_candidate(sig)
        except Exception as e:
            log.debug("log_signal_candidate failed: %s", e)
    # auto-open paper position for medium/strong signals only
    if sig.strength in ("medium", "strong"):
        try:
            pos = portfolio.open_position(sig)
            _mark_traded(sig.chain, sig.token_address)
            try:
                alerts.alert_position_open(sig, pos)
            except Exception:
                pass
        except Exception as e:
            log.warning("open_position failed for %s: %s", sig.token_symbol, e)
    _persist_signals()
    log.info(
        "SIGNAL [%s] %s/%s  strength=%s  composite=%.2f  %s",
        sig.signal_type.upper(), sig.chain, sig.token_symbol or sig.token_address[:8],
        sig.strength, sig.composite_score, sig.notes,
    )


# ---------------------------------------------------------------------------
# Wallet thread
# ---------------------------------------------------------------------------

def _process_wallet_event(evt: WalletEvent, ranks: dict):
    """Convert a WalletEvent into a signal or record a whale sell."""
    if evt.action == "sell":
        with _lock:
            _whale_sells[evt.token_address].append(evt.wallet)
        return

    # It's a buy — screen token and generate copy-trade signal
    screen = screen_token(evt.chain, evt.token_address)
    if not screen["passed"]:
        log.debug("Token %s failed safety: %s", evt.token_address[:8], screen["reason"])
        return

    sig = make_copy_trade_signal(
        chain=evt.chain,
        token_address=evt.token_address,
        screen=screen,
        whale_wallets=[evt.wallet],
        wallet_ranks=ranks,
    )
    _add_signal(sig)


def _wallet_thread(wallets: dict, ranks: dict):
    """Background thread: alternates between SOL and BNB polling."""
    sol_wallets = wallets.get("solana", [])
    bnb_wallets = wallets.get("bsc", [])
    bscscan_key = os.getenv("BSCSCAN_API_KEY", "")

    sol_last          = 0.0
    bnb_last          = 0.0
    wallet_reload_ts  = 0.0
    WALLET_RELOAD_SEC = 600   # refresh DB-promoted wallets every 10 min

    log.info("Wallet tracker started — SOL:%d wallets  BNB:%d wallets",
             len(sol_wallets), len(bnb_wallets))

    while True:
        now = time.time()

        # Refresh wallet list + ranks every 10 min so Phase 6/7 DB promotions
        # are picked up without needing a bot restart
        if now - wallet_reload_ts >= WALLET_RELOAD_SEC:
            try:
                new_wallets = load_all_wallets()
                new_ranks   = build_wallet_ranks(new_wallets)
                new_sol     = new_wallets.get("solana", [])
                new_bnb     = new_wallets.get("bsc", [])
                added = len(set(new_sol) - set(sol_wallets))
                if added:
                    log.info("Wallet refresh: +%d new SOL wallets added to polling", added)
                sol_wallets = new_sol
                bnb_wallets = new_bnb
                ranks.clear()
                ranks.update(new_ranks)
                wallet_reload_ts = now
            except Exception as e:
                log.debug("Wallet list refresh error: %s", e)

        if now - sol_last >= SOL_WALLET_POLL_SEC and sol_wallets:
            sol_last = now
            try:
                events = poll_sol_wallets_batch(sol_wallets, ranks)
                for evt in events:
                    _process_wallet_event(evt, ranks)
            except Exception as e:
                log.warning("SOL wallet poll error: %s", e)

        if now - bnb_last >= BNB_WALLET_POLL_SEC and bnb_wallets:
            bnb_last = now
            try:
                events = poll_bnb_wallets_batch(bnb_wallets, ranks, bscscan_key)
                for evt in events:
                    _process_wallet_event(evt, ranks)
            except Exception as e:
                log.warning("BNB wallet poll error: %s", e)

        time.sleep(5)


# ---------------------------------------------------------------------------
# Market scan thread (DexScreener / GMGN)
# ---------------------------------------------------------------------------

# volume baselines: token_address → last h1 volume seen
_vol_baselines: dict[str, float] = {}


_FUNNEL_LOG_MIN_LIQ = 5_000  # only log screener failures for tokens with real liquidity


def _log_rejection(chain: str, addr: str, screen: dict, reason: str):
    """Log a funnel rejection — never raises."""
    try:
        log_new_launch_rejection(chain, addr, screen, reason=reason)
    except Exception as e:
        log.debug("log_rejection failed: %s", e)


def _scan_new_launches(chain: str, candidates: list[dict]):
    """Screen a list of token candidates for new-launch signals."""
    for item in candidates[:30]:
        addr = (
            item.get("tokenAddress")
            or item.get("address")
            or (item.get("baseToken") or {}).get("address")
            or ""
        )
        if not addr:
            continue
        screen = screen_token(chain, addr)
        if not screen["passed"]:
            # Log meaningful screener failures (skip no_dex_data / micro-liq noise)
            if screen.get("liquidity_usd", 0) >= _FUNNEL_LOG_MIN_LIQ:
                _log_rejection(chain, addr, screen, reason=screen["reason"])
            continue
        if screen["age_minutes"] > MAX_AGE_MINUTES_NEW:
            _log_rejection(chain, addr, screen, reason="age_too_old")
            continue
        # Fix 3: 5m momentum filter
        p5m = screen["price_change_5m"]
        if p5m < 20:
            log.debug("new_launch %s skipped: 5m=%.1f%% < 20%%", addr[:8], p5m)
            reason = "5m_momentum_below_20"
            _log_rejection(chain, addr, screen, reason=reason)
            # Near-miss: track tokens at 15–19% for post-rejection outcome analysis
            if 15 <= p5m < 20:
                try:
                    track_near_miss(chain, addr, screen)
                except Exception as e:
                    log.debug("track_near_miss failed: %s", e)
            continue
        # Block meteora — 22 trades, -29.6% avg, 18% win rate
        if screen["dex_id"] == "meteora":
            log.debug("new_launch %s skipped: meteora dex", addr[:8])
            _log_rejection(chain, addr, screen, reason="meteora_block")
            continue
        sig = make_new_launch_signal(chain, addr, screen)
        _add_signal(sig)
        time.sleep(0.2)


def _scan_volume_breakouts(chain: str, candidates: list[dict]):
    """Detect volume spikes across trending pairs."""
    for item in candidates[:40]:
        addr = (
            item.get("tokenAddress")
            or item.get("address")
            or (item.get("baseToken") or {}).get("address")
            or ""
        )
        if not addr:
            continue
        screen = screen_token(chain, addr)
        if not screen["passed"]:
            continue
        baseline = _vol_baselines.get(addr, 0)
        if baseline > 0:
            sig = make_volume_breakout_signal(chain, addr, screen, baseline)
            _add_signal(sig)
        # update baseline with exponential smoothing
        _vol_baselines[addr] = (
            0.7 * screen["volume_h1"] + 0.3 * _vol_baselines.get(addr, screen["volume_h1"])
        )
        time.sleep(0.2)


def _scan_dev_launches():
    """Check all tracked dev wallets for new token deployments."""
    findings = poll_all_devs()
    for f in findings:
        chain   = f["chain"]
        addr    = f["token_address"]
        entry   = f["dev_entry"]
        dev_addr = f["dev_address"]
        screen = screen_token(chain, addr)
        if not screen["passed"]:
            log.debug("Dev launch %s failed safety: %s", addr[:8], screen["reason"])
            continue
        strength = dev_signal_strength(entry)
        sig = make_dev_launch_signal(
            chain=chain,
            token_address=addr,
            screen=screen,
            dev_address=dev_addr,
            dev_entry=entry,
            strength=strength,
        )
        _add_signal(sig)
        time.sleep(0.3)


def _market_thread():
    """Background thread: scans DexScreener + GMGN every 2 min."""
    log.info("Market scanner started (interval=%ds)", DEXSCREENER_POLL_SEC)
    while True:
        try:
            # Solana new launches (GMGN)
            gmgn_new = gmgn_new_sol(limit=30)
            _scan_new_launches("solana", gmgn_new)

            # Solana trending (volume breakout candidates)
            gmgn_trend = gmgn_trending_sol(limit=30)
            _scan_volume_breakouts("solana", gmgn_trend)

            # DexScreener new pairs — both chains
            for chain in ("solana", "bsc"):
                new_pairs = dex_get_new_pairs(chain, limit=30)
                _scan_new_launches(chain, new_pairs)

            # DexScreener boosted (often has volume spikes)
            boosted = dex_get_boosted()
            sol_boosted = [b for b in boosted if b.get("chainId") == "solana"]
            bsc_boosted  = [b for b in boosted if b.get("chainId") == "bsc"]
            _scan_volume_breakouts("solana", sol_boosted)
            _scan_volume_breakouts("bsc",    bsc_boosted)

            # Dev wallet polling — check known devs for new launches
            try:
                _scan_dev_launches()
            except Exception as e:
                log.debug("Dev scan error: %s", e)

        except Exception as e:
            log.warning("Market scan error: %s", e)

        time.sleep(DEXSCREENER_POLL_SEC)


# ---------------------------------------------------------------------------
# Real-time Pump.fun WebSocket thread (replaces polling for new launches)
# ---------------------------------------------------------------------------

def _pumpfun_thread():
    """
    Real-time pump.fun event listener via Solana logsSubscribe websocket.

    Handles two event types:
      new_token  → screen via DexScreener → new_launch signal
                   OR if creator is a known dev → dev_launch signal (fired first)
      early_buy  → if buyer is a tracked whale wallet → copy_trade signal
                   (fires before any Telegram alert sees the token)

    Falls back gracefully if websocket-client is not installed.
    """
    from memecoin.pumpfun_listener import PumpListener, PumpEvent

    log.info("Pump.fun real-time listener starting")
    listener = PumpListener()
    listener.start(daemon=True)

    # Load whale wallet addresses for early-buy detection
    # Refreshed every 5 min so new wallets added to JSON are picked up
    _whale_addrs: set = set()
    _whale_reload_ts = 0.0

    def _reload_whale_addrs():
        nonlocal _whale_addrs, _whale_reload_ts
        try:
            wallets = load_all_wallets()
            # load_all_wallets returns {"solana": [addr_str, ...], "bsc": [...]}
            _whale_addrs = set(wallets.get("solana", []))
            _whale_reload_ts = time.time()
            log.debug("Loaded %d whale addresses for early-buy filter", len(_whale_addrs))
        except Exception as e:
            log.warning("Could not reload whale addresses: %s", e)

    _reload_whale_addrs()

    while True:
        # Refresh whale list every 5 min
        if time.time() - _whale_reload_ts > 300:
            _reload_whale_addrs()

        event: PumpEvent = listener.get(timeout=2.0)
        if not event:
            continue

        try:
            # ── new_token: dev wallet check first, then DexScreener screen ──
            if event.event_type == "new_token":
                addr    = event.mint
                creator = event.creator

                # 1. Dev wallet check — fire dev_launch immediately if known dev
                from memecoin.dev_tracker import load_dev_wallets, dev_signal_strength
                dev_wallets = load_dev_wallets()
                dev_map     = {d["address"]: d for d in dev_wallets if d.get("win_count", 0) >= 2}
                if creator in dev_map:
                    dev_entry = dev_map[creator]
                    strength  = dev_signal_strength(dev_entry)
                    screen    = screen_token("solana", addr)
                    # Dev signals skip most filters — we trust the dev's history
                    screen["passed"] = True
                    sig = make_dev_launch_signal(
                        "solana", addr, screen,
                        dev_address=creator, dev_entry=dev_entry, strength=strength,
                    )
                    if sig:
                        log.info("DEV LAUNCH (realtime)  dev=%s  mint=%s  wins=%d",
                                 creator[:8], addr[:8], dev_entry.get("win_count", 0))
                        _add_signal(sig)
                    continue   # don't double-fire as new_launch

                # 2. Standard new_launch screen (DexScreener — arrives within ~5s)
                # Wait a few seconds for DexScreener to index the token
                time.sleep(5)
                screen = screen_token("solana", addr)
                if not screen["passed"]:
                    if screen.get("liquidity_usd", 0) >= _FUNNEL_LOG_MIN_LIQ:
                        _log_rejection("solana", addr, screen, reason=screen["reason"])
                    continue
                if screen["age_minutes"] > MAX_AGE_MINUTES_NEW:
                    _log_rejection("solana", addr, screen, reason="age_too_old")
                    continue
                p5m = screen.get("price_change_5m", 0)
                if p5m < 20:
                    _log_rejection("solana", addr, screen, reason="5m_momentum_below_20")
                    if 15 <= p5m < 20:
                        try:
                            track_near_miss("solana", addr, screen)
                        except Exception:
                            pass
                    continue
                if screen.get("dex_id") == "meteora":
                    _log_rejection("solana", addr, screen, reason="meteora_block")
                    continue
                sig = make_new_launch_signal("solana", addr, screen)
                _add_signal(sig)

            # ── early_buy: whale bought a recently-created token ──────────
            elif event.event_type == "early_buy":
                buyer = event.buyer
                addr  = event.mint
                if buyer not in _whale_addrs:
                    continue   # not a tracked whale — ignore

                log.info("EARLY WHALE BUY (realtime)  buyer=%s  mint=%s",
                         buyer[:8], addr[:8])

                # Wait for DexScreener to have price data
                time.sleep(5)
                screen = screen_token("solana", addr)
                # Bypass most filters — the whale buy IS the signal
                screen["passed"] = True

                # Build a copy_trade signal with the whale's address
                all_wallets = load_all_wallets()
                ranks = build_wallet_ranks(all_wallets)
                sig   = make_copy_trade_signal(
                    "solana", addr, screen,
                    whale_wallets=[buyer],
                    wallet_ranks=ranks,
                )
                _add_signal(sig)

        except Exception as e:
            log.debug("Pump.fun event error  type=%s  mint=%s  err=%s",
                      event.event_type, event.mint[:8], e)


# ---------------------------------------------------------------------------
# Telegram social signal callback
# ---------------------------------------------------------------------------

def _on_telegram_signal(chain: str, address: str, message_text: str):
    """Called by TelegramMonitor when a token address is found in a channel message."""
    import time as _time
    _t0 = _time.time()
    try:
        screen = screen_token(chain, address)
        reason = screen.get("reason", "")

        # Hard reject: no DexScreener data yet.
        # Subscribe to PumpPortal screening in parallel with the DexScreener retry.
        # scanner._run_screening_checks() will fire a paper-only entry at T+30/60/120s
        # if on-chain demand is strong enough.  The DexScreener retry path (_NoDexData)
        # continues unchanged — two independent entry chances for the same token.
        if reason == "no_dex_data":
            with _sq_lock:
                if address not in _screening_queue:
                    _pp_monitor.subscribe_screening(address)
                    _screening_queue[address] = {
                        "ts":          _t0,
                        "chain":       chain,
                        "checks_done": 0,
                    }
                    log.info("TG SCREEN-QUEUE %s — PumpPortal accumulator started",
                             address[:8])
            log.info("TG REJECT %s — no_dex_data (DexScreener not indexed yet, screen took %.1fs)",
                     address[:8], _time.time() - _t0)
            raise _NoDexData(address)
        if any(r in reason for r in ("rugcheck_fail", "honeypot", "rug_detector")):
            log.info("TG REJECT %s — rug/safety: %s", address[:8], reason)
            return

        # Social alert entry filters (data-derived from v5+v6, 192 trades)
        bs   = screen.get("buy_sell_ratio_5m") or 0
        v5m  = screen.get("volume_5m") or 0
        vh1  = screen.get("volume_h1") or 0
        pc5m = screen.get("price_change_5m") or 0

        if bs < MIN_BUY_SELL_RATIO_SOCIAL:
            log.info("TG REJECT %s — bs=%.2f < %.2f (buy pressure too low)", address[:8], bs, MIN_BUY_SELL_RATIO_SOCIAL)
            return
        if not (MIN_VOL_5M_SOCIAL <= v5m < MAX_VOL_5M_SOCIAL):
            log.info("TG REJECT %s — vol_5m=%.0f not in $%d-$%d", address[:8], v5m, MIN_VOL_5M_SOCIAL, MAX_VOL_5M_SOCIAL)
            return
        if vh1 >= MAX_VOL_H1_SOCIAL:
            log.info("TG REJECT %s — vol_h1=%.0f >= $%d (already pumped)", address[:8], vh1, MAX_VOL_H1_SOCIAL)
            return
        if 0 < pc5m >= MAX_PRICE_CHANGE_5M_SOCIAL:
            log.info("TG REJECT %s — pc5m=%.0f >= %d%% (blow-off top)", address[:8], pc5m, MAX_PRICE_CHANGE_5M_SOCIAL)
            return

        screen["passed"] = True
        _t_screen_end = _time.time()
        _screen_latency = _t_screen_end - _t0
        log.info("TG PASS %s — bs=%.2f vol5m=%.0f vh1=%.0f pc5m=%.0f screen_took=%.1fs",
                 address[:8], bs, v5m, vh1, pc5m, _screen_latency)

        channel = "pumpdotfunalert"
        sig = make_social_alert_signal(chain, address, screen, source="telegram", channel=channel)
        # Attach timing for entry latency instrumentation (Step 2)
        sig._t_tg_receive  = _t0
        sig._t_screen_end  = _t_screen_end
        sig._price_dex     = screen.get("price_usd") or 0  # stale DexScreener baseline
        _add_signal(sig)
    except _NoDexData:
        raise  # propagate to TelegramMonitor for 45s retry
    except Exception as e:
        log.warning("TG signal processing error %s: %s", address[:8], e)


# ---------------------------------------------------------------------------
# Near-miss outcome poller thread
# ---------------------------------------------------------------------------

def _near_miss_poller_thread():
    """
    Check near-miss tokens (5m=15-19%) at 1h and 6h post-rejection.
    Runs every 15 minutes. Fetches current DexScreener price at each interval
    and records the % change from rejection price → determines if we missed
    a profitable trade or if the filter was correct.
    """
    log.info("Near-miss poller started")
    while True:
        try:
            data = load_near_miss_data()
            now  = time.time()
            for key, entry in list(data.items()):
                chain   = entry["chain"]
                addr    = entry["token_address"]
                t0      = entry.get("rejection_time", 0)
                elapsed = now - t0

                needs_1h = not entry.get("check_1h_done") and elapsed >= 3600
                needs_6h = not entry.get("check_6h_done") and elapsed >= 21600

                if not (needs_1h or needs_6h):
                    continue

                pair  = dex_get_token(chain, addr)
                price = float(pair.get("priceUsd") or 0) if pair else 0

                if needs_1h:
                    update_near_miss_check(key, "1h", price)
                    log.debug("near-miss 1h check %s: price=%.8f", addr[:8], price)
                if needs_6h:
                    update_near_miss_check(key, "6h", price)
                    log.debug("near-miss 6h check %s: price=%.8f outcome=%s",
                              addr[:8], price, load_near_miss_data().get(key, {}).get("outcome"))

                time.sleep(0.3)  # be polite to DexScreener
        except Exception as e:
            log.debug("Near-miss poller error: %s", e)

        time.sleep(900)  # run every 15 min


# ---------------------------------------------------------------------------
# Portfolio monitor thread
# ---------------------------------------------------------------------------

_FEED_BLIND_SEC = 20.0   # both feeds silent for this long → market-sell live position


def _portfolio_thread():
    """Update open position prices and check exit conditions every 2s."""
    log.info("Portfolio monitor started")
    _subscribed_mints: set[str] = set()
    # mint → timestamp when we last got a DexScreener price for it
    _dex_last_seen: dict[str, float] = {}

    while True:
        try:
            with _lock:
                whale_sells_snapshot = dict(_whale_sells)
                _whale_sells.clear()

            # ── PumpPortal subscription management ───────────────────────────
            # Diff open pumpfun positions against what we're subscribed to.
            # Subscribe new; unsubscribe closed. Graduated/BSC tokens skip PP.
            open_pumpfun = {
                p.token_address
                for p in portfolio.open_positions()
                if p.chain == "solana"
                and "pump" in (p.dex_id or "").lower()
            }
            new_mints   = open_pumpfun - _subscribed_mints
            stale_mints = _subscribed_mints - open_pumpfun
            if new_mints:
                _pp_monitor.subscribe(new_mints)
                _subscribed_mints |= new_mints
            if stale_mints:
                _pp_monitor.unsubscribe(stale_mints)
                _subscribed_mints -= stale_mints

            # Fresh PumpPortal prices override DexScreener for subscribed tokens
            price_overrides = _pp_monitor.get_prices()

            # ── DexScreener staleness tracking ───────────────────────────────
            # Probe DexScreener for mints not covered by a fresh PP price, but
            # throttled to once per 10s per mint to avoid rate-limit false blinds.
            from memecoin.data_client import dex_get_token as _dex_get
            now = time.time()
            for mint in open_pumpfun:
                if mint not in price_overrides:
                    last_dex = _dex_last_seen.get(mint, 0)
                    if now - last_dex >= 10:   # throttle: 1 probe per 10s max
                        try:
                            pair = _dex_get("solana", mint)
                            if pair and float(pair.get("priceUsd") or 0) > 0:
                                _dex_last_seen[mint] = now
                        except Exception:
                            pass

            exits = portfolio.update_prices(
                whale_sells=whale_sells_snapshot,
                price_overrides=price_overrides,
            )
            if exits:
                for e in exits:
                    log.info("EXIT %s  reason=%s  pnl=%.1f%%",
                             e["token_symbol"], e["reason"], e["pnl_pct"])

            # ── Blind-exit check: both feeds silent >20s for live positions ──
            # Fires a market-sell so we don't hold through an unobservable dump.
            for pos in list(portfolio.open_positions()):
                if not (pos.notes and "live|tx:" in pos.notes):
                    continue   # paper-only — no money at risk
                mint = pos.token_address
                if mint not in open_pumpfun:
                    continue   # graduated / BSC — DexScreener covers it, skip
                pp_age  = _pp_monitor.get_last_seen(mint)
                dex_age = now - _dex_last_seen.get(mint, 0) if _dex_last_seen.get(mint) else float("inf")
                if pp_age > _FEED_BLIND_SEC and dex_age > _FEED_BLIND_SEC:
                    log.error(
                        "FEED BLIND — both PP (%.0fs) and DexScreener (%.0fs) silent "
                        "for live position %s (%s). Triggering market-sell.",
                        pp_age, dex_age, pos.token_symbol, pos.id,
                    )
                    try:
                        from app.alerts import _send
                        _send(
                            f"⚠️ FEED BLIND {pos.token_symbol} — PP silent {pp_age:.0f}s, "
                            f"DEX silent {dex_age:.0f}s. Executing market-sell to protect capital."
                        )
                    except Exception:
                        pass
                    portfolio.close_position(pos.id, "feed_blind", pos.current_price)

            # Check PumpPortal-screened tokens for paper entry conditions
            try:
                _run_screening_checks()
            except Exception as e:
                log.debug("screening checks error: %s", e)

        except Exception as e:
            log.warning("Portfolio monitor error: %s", e)
        time.sleep(2)


# ---------------------------------------------------------------------------
# Public API — start / query
# ---------------------------------------------------------------------------

_started = False


def start(daemon: bool = True):
    """
    Start all background threads.  Call once at app startup.
    """
    global _started
    if _started:
        return
    _started = True

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    wallets = load_all_wallets()
    ranks   = build_wallet_ranks(wallets)

    _pp_monitor.start(daemon=daemon)

    for target, kwargs in [
        (_wallet_thread,       {"wallets": wallets, "ranks": ranks}),
        (_market_thread,       {}),
        (_portfolio_thread,    {}),
        (_pumpfun_thread,      {}),
        (_near_miss_poller_thread, {}),
    ]:
        t = threading.Thread(target=target, kwargs=kwargs, daemon=daemon)
        t.start()

    # Telegram monitor — start if credentials are configured
    tg_api_id   = os.getenv("TELEGRAM_API_ID")
    tg_api_hash = os.getenv("TELEGRAM_API_HASH")
    if tg_api_id and tg_api_hash:
        try:
            from memecoin.telegram_monitor import TelegramMonitor
            tg = TelegramMonitor(
                api_id=int(tg_api_id),
                api_hash=tg_api_hash,
                signal_callback=_on_telegram_signal,
            )
            tg.start(daemon=daemon)
        except Exception as e:
            log.warning("Telegram monitor failed to start: %s", e)
    else:
        log.info("Telegram monitor disabled — set TELEGRAM_API_ID and TELEGRAM_API_HASH to enable")

    alerts.init()
    log.info("Memecoin scanner started.")


def get_signals(limit: int = 50) -> list[dict]:
    with _lock:
        return [s.to_dict() for s in list(_signals)[:limit]]


def get_open_positions() -> list[dict]:
    from dataclasses import asdict
    return [asdict(p) for p in portfolio.open_positions()]


def manual_close(pos_id: str) -> Optional[dict]:
    from dataclasses import asdict
    pos = portfolio.manual_close(pos_id)
    return asdict(pos) if pos else None


def get_journal(limit: int = 200) -> list[dict]:
    return portfolio.load_journal()[-limit:]


def get_summary() -> dict:
    return {
        **portfolio.summary(),
        "signal_count": len(_signals),
    }
