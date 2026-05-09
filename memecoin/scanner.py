"""
Main scan loop for the memecoin module.

Runs two background threads:
  - wallet_thread   : polls whale wallets every 30s (SOL) / 60s (BNB)
  - market_thread   : scans DexScreener / GMGN every 2 min for breakouts/launches

All discovered signals are stored in a shared signal queue and written
to data/memecoin_signals.json so the web UI can read them.
"""

import json
import logging
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

from memecoin.config import (
    SIGNALS_FILE, DATA_DIR,
    SOL_WALLET_POLL_SEC, BNB_WALLET_POLL_SEC, DEXSCREENER_POLL_SEC,
    MIN_LIQUIDITY_USD, MAX_AGE_MINUTES_NEW,
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
from memecoin.candidate_log import log_signal_candidate
from memecoin.dev_tracker import poll_all_devs, dev_signal_strength
from memecoin.signals import (
    Signal, make_copy_trade_signal, make_volume_breakout_signal,
    make_new_launch_signal, make_dev_launch_signal,
)

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


def _is_duplicate(chain: str, address: str, sig_type: str) -> bool:
    key = f"{chain}:{address}:{sig_type}"
    now = time.time()
    last = _seen.get(key, 0)
    if now - last < 900:   # 15 min cooldown
        return True
    _seen[key] = now
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
            portfolio.open_position(sig)
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

    sol_last = 0.0
    bnb_last = 0.0

    log.info("Wallet tracker started — SOL:%d wallets  BNB:%d wallets",
             len(sol_wallets), len(bnb_wallets))

    while True:
        now = time.time()

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
            continue
        if screen["age_minutes"] > MAX_AGE_MINUTES_NEW:
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
    Listens to Pump.fun WebSocket for new token creation events.
    Pushes them directly to the screener — no 2-min poll delay.
    Falls back gracefully if websocket-client is not installed.
    """
    try:
        from sniper.listener import PumpListener, PumpEvent
    except ImportError:
        log.info("sniper module not available — skipping Pump.fun WebSocket")
        return

    log.info("Pump.fun real-time listener started (memecoin new_launch feed)")
    listener = PumpListener(strategies=["launch"])
    listener.start(daemon=True)

    while True:
        event: PumpEvent = listener.get(timeout=2.0)
        if not event:
            continue
        if event.event_type != "new_token":
            continue

        # Quick age gate — only process truly new tokens
        # (event arrives within seconds of creation, so age_minutes ≈ 0)
        try:
            screen = screen_token("solana", event.mint)
            if not screen["passed"]:
                continue
            if screen["age_minutes"] > MAX_AGE_MINUTES_NEW:
                continue
            sig = make_new_launch_signal("solana", event.mint, screen)
            _add_signal(sig)
        except Exception as e:
            log.debug("Pump.fun token screen error %s: %s", event.mint[:8], e)


# ---------------------------------------------------------------------------
# Portfolio monitor thread
# ---------------------------------------------------------------------------

def _portfolio_thread():
    """Update open position prices and check exit conditions every 60s."""
    log.info("Portfolio monitor started")
    while True:
        try:
            with _lock:
                whale_sells_snapshot = dict(_whale_sells)
                _whale_sells.clear()
            exits = portfolio.update_prices(whale_sells=whale_sells_snapshot)
            if exits:
                for e in exits:
                    log.info("EXIT %s  reason=%s  pnl=%.1f%%",
                             e["token_symbol"], e["reason"], e["pnl_pct"])
        except Exception as e:
            log.warning("Portfolio monitor error: %s", e)
        time.sleep(60)


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

    for target, kwargs in [
        (_wallet_thread,    {"wallets": wallets, "ranks": ranks}),
        (_market_thread,    {}),
        (_portfolio_thread, {}),
        (_pumpfun_thread,   {}),
    ]:
        t = threading.Thread(target=target, kwargs=kwargs, daemon=daemon)
        t.start()

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
