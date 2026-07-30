"""
Helius standby price feed — accountSubscribe on bonding curve PDAs.

Purpose
-------
Secondary (standby) feed for pump.fun tokens we hold as live positions.
Activates only when PumpPortal is disconnected OR has been silent for a
specific mint for >5s. Uses Helius accountSubscribe to get the same
vSolReserves / vTokenReserves data as PP, but via a different path.

Policy
------
  Activate  : PP last_seen > ACTIVATE_AFTER_SEC (5s) for a live position mint
  Deactivate: PP healthy for DEACTIVATE_HYSTERESIS_SEC (10s) continuously
  Max concurrent subscriptions: 2 (matches max live position cap)
  Never for: screening tokens, paper-only positions
  Steady-state usage: ZERO. If PP is healthy, no Helius WS connection exists.

Bonding curve PDA
-----------------
  Seeds: [b"bonding-curve", bytes(mint_pubkey)]
  Program: 6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P

Account data layout (after 8-byte discriminator):
  offset  8: virtual_token_reserves  u64 LE
  offset 16: virtual_sol_reserves    u64 LE

  price_usd = (v_sol / 1e9) / (v_token / 1e6) * sol_price_usd

Credit accounting
-----------------
  Every accountNotification = 1 credit on Helius free tier.
  Each activation logs: duration, update_count, mints involved.
  Caller reads this from the log — no steady-state usage to track.
"""

import json
import logging
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

# Pump.fun program ID (mainnet)
PUMP_FUN_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

# Activation / deactivation thresholds
ACTIVATE_AFTER_SEC           = 5.0    # PP silent this long → activate Helius for mint
DEACTIVATE_HYSTERESIS_SEC    = 10.0   # PP healthy this long → deactivate Helius
MAX_CONCURRENT               = 2      # matches live position cap

# Websocket
_WS_TIMEOUT                  = 30
_RECONNECT_DELAY             = 2.0   # base delay (P8: now used as base for exponential backoff)
# P8: exponential backoff + max retry limit
_WS_MAX_RETRIES              = 20    # ~8 minutes total before alerting and stopping
_POST_BUY_WS_QUIET_SEC       = 10.0  # do not reconnect WS for this many seconds after a live buy

# Price is "fresh" if received within this many seconds
PRICE_STALE_SEC              = 15.0


# ---------------------------------------------------------------------------
# Internal slot
# ---------------------------------------------------------------------------

@dataclass
class _Slot:
    mint:              str
    bc_pubkey:         str        # bonding curve PDA (base58)
    subscription_id:   Optional[int] = None
    activated_ts:      float         = field(default_factory=time.time)
    update_count:      int           = 0
    pp_healthy_since:  float         = 0.0    # when PP ticks resumed (0 = not yet)
    price_usd:         float         = 0.0
    price_ts:          float         = 0.0


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class HeliusAccountMonitor:
    """
    Standby Helius accountSubscribe feed for live pump.fun positions.

    Usage (called from scanner._portfolio_thread every cycle):
        helius_monitor.update(live_positions, pp_monitor)
        overrides = helius_monitor.get_prices()  # merge with pp prices
    """

    def __init__(self):
        self._slots: dict[str, _Slot]   = {}    # mint → _Slot
        self._lock                      = threading.Lock()

        self._ws                        = None
        self._ws_lock                   = threading.Lock()
        self._ws_thread: Optional[threading.Thread] = None

        # sub_id → mint (for routing incoming notifications)
        self._sub_id_map: dict[int, str] = {}

        # Pending subscribe/unsubscribe requests for the WS thread
        self._pending_subs:   set[str]  = set()   # bc_pubkeys to subscribe
        self._pending_unsubs: set[int]  = set()   # sub_ids to unsubscribe
        self._pending_lock              = threading.Lock()

        self._sol_price: float          = 170.0
        self._sol_price_lock            = threading.Lock()
        self._next_req_id: int          = 1

        # Daily credit counter (informational log only)
        self._daily_updates:  int       = 0
        self._daily_day:      str       = ""

        # P8: post-buy quiet window — suppress WS reconnect for 10s after live buy
        self._post_buy_quiet_until: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, live_positions: list, pp_monitor) -> None:
        """
        Called every portfolio cycle. Manages Helius subscriptions based on
        PP health per mint. `pp_monitor` is the PumpPortalMonitor singleton.
        `live_positions` is the list of open positions with notes containing 'live|tx:'.
        """
        now = time.time()

        # Only consider live pumpfun positions
        live_pumpfun = {
            p.token_address: p
            for p in live_positions
            if (p.notes and "live|tx:" in p.notes)
            and p.chain == "solana"
            and (getattr(p, "dex_id", "") or "") in ("pumpfun", "pumpswap")
        }

        with self._lock:
            # ── Deactivate slots whose PP is now healthy ──────────────────
            to_close = []
            for mint, slot in list(self._slots.items()):
                pp_age = pp_monitor.get_last_seen(mint)
                if pp_age < PRICE_STALE_SEC:
                    # PP is delivering fresh ticks — start/reset hysteresis timer
                    if slot.pp_healthy_since == 0.0:
                        slot.pp_healthy_since = now
                    elif now - slot.pp_healthy_since >= DEACTIVATE_HYSTERESIS_SEC:
                        to_close.append(mint)
                else:
                    # PP still silent — reset hysteresis
                    slot.pp_healthy_since = 0.0

            for mint in to_close:
                slot = self._slots.pop(mint)
                if slot.subscription_id is not None:
                    with self._pending_lock:
                        self._pending_unsubs.add(slot.subscription_id)
                    self._sub_id_map.pop(slot.subscription_id, None)
                duration = now - slot.activated_ts
                log.info(
                    "Helius standby DEACTIVATED %s — PP healthy for %.0fs  "
                    "duration=%.0fs  updates=%d",
                    mint[:8], DEACTIVATE_HYSTERESIS_SEC, duration, slot.update_count,
                )

            # ── Activate new slots for mints where PP is silent ───────────
            for mint, pos in live_pumpfun.items():
                if mint in self._slots:
                    continue   # already active
                if len(self._slots) >= MAX_CONCURRENT:
                    continue   # cap reached
                pp_age = pp_monitor.get_last_seen(mint)
                if pp_age <= ACTIVATE_AFTER_SEC:
                    continue   # PP still fresh — do not activate

                bc_pubkey = _derive_bonding_curve_pda(mint)
                if not bc_pubkey:
                    log.debug("Helius standby: could not derive bonding curve PDA for %s", mint[:8])
                    continue

                slot = _Slot(mint=mint, bc_pubkey=bc_pubkey)
                self._slots[mint] = slot
                with self._pending_lock:
                    self._pending_subs.add(bc_pubkey)

                log.warning(
                    "Helius standby ACTIVATED %s — PP silent %.1fs  bc_pda=%s",
                    mint[:8], pp_age, bc_pubkey[:8],
                )

            # ── Close mints no longer in live_pumpfun ────────────────────
            stale = set(self._slots) - set(live_pumpfun)
            for mint in stale:
                slot = self._slots.pop(mint)
                if slot.subscription_id is not None:
                    with self._pending_lock:
                        self._pending_unsubs.add(slot.subscription_id)
                    self._sub_id_map.pop(slot.subscription_id, None)
                duration = now - slot.activated_ts
                log.info(
                    "Helius standby closed (position gone) %s  duration=%.0fs  updates=%d",
                    mint[:8], duration, slot.update_count,
                )

        # Start or stop the WS connection based on whether any slots are active
        self._manage_connection()

    def get_prices(self, max_age: float = PRICE_STALE_SEC) -> dict[str, float]:
        """
        Return {mint: price_usd} for Helius-monitored tokens with fresh data.
        Caller should merge these ONLY for mints absent from PP get_prices().
        """
        now = time.time()
        with self._lock:
            return {
                slot.mint: slot.price_usd
                for slot in self._slots.values()
                if slot.price_usd > 0 and now - slot.price_ts <= max_age
            }

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _manage_connection(self):
        """Start WS thread if slots are active; stop it if no slots remain."""
        with self._lock:
            has_slots = bool(self._slots)

        if has_slots:
            if self._ws_thread is None or not self._ws_thread.is_alive():
                self._ws_thread = threading.Thread(
                    target=self._ws_run, daemon=True, name="helius-account-monitor"
                )
                self._ws_thread.start()
                log.info("Helius account monitor WS thread started")
        else:
            # No active slots — close connection if open
            with self._ws_lock:
                ws = self._ws
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass
            with self._ws_lock:
                self._ws = None

    def set_post_buy_quiet(self) -> None:
        """P8: Call this immediately after a live buy TX is submitted.
        Suppresses Helius WS reconnect for _POST_BUY_WS_QUIET_SEC seconds
        to avoid contending with buy TX propagation on the same connection."""
        self._post_buy_quiet_until = time.time() + _POST_BUY_WS_QUIET_SEC
        log.debug("Helius standby WS: post-buy quiet window set (%.0fs)", _POST_BUY_WS_QUIET_SEC)

    def _ws_run(self):
        """WS reconnect loop with exponential backoff (P8).
        Runs only while there are active slots.
        Max retries: _WS_MAX_RETRIES (~8 min). Alerts on exhaustion."""
        attempt = 0
        while True:
            with self._lock:
                if not self._slots:
                    log.info("Helius account monitor: no active slots, WS thread exiting")
                    return

            # P8: post-buy quiet window — don't reconnect while buy TX is propagating
            _quiet_remaining = self._post_buy_quiet_until - time.time()
            if _quiet_remaining > 0:
                log.debug("Helius standby WS: post-buy quiet window (%.1fs remaining)", _quiet_remaining)
                time.sleep(min(_quiet_remaining, 1.0))
                continue

            if attempt >= _WS_MAX_RETRIES:
                log.error(
                    "Helius standby WS: max retries (%d) exhausted — giving up. "
                    "Restart the bot or check Helius API key.", _WS_MAX_RETRIES,
                )
                try:
                    from app.alerts import _send
                    _send(
                        f"🚨 Helius standby WS: {_WS_MAX_RETRIES} reconnect attempts exhausted. "
                        f"Standby feed unavailable — PP feed only. Restart if needed."
                    )
                except Exception:
                    pass
                return

            try:
                _reason = "helius_primary_disconnect" if attempt == 0 else f"helius_retry_{attempt}"
                if attempt > 0:
                    log.info("STANDBY WS ACTIVATE: triggered by %s", _reason)
                self._ws_connect()
                attempt = 0  # reset on successful connection
            except Exception as e:
                delay = min(60, 2 ** attempt)  # exponential backoff: 1, 2, 4, 8, 16, 32, 60, 60...
                attempt += 1
                log.warning(
                    "Helius WS error (attempt %d/%d): %s — retry in %.0fs",
                    attempt, _WS_MAX_RETRIES, e, delay,
                )
                time.sleep(delay)

    def _ws_connect(self):
        api_key = os.getenv("HELIUS_API_KEY", "")
        if not api_key:
            log.warning("HELIUS_API_KEY not set — standby feed unavailable")
            time.sleep(60)
            return

        try:
            import websocket as _ws_mod
        except ImportError:
            log.warning("websocket-client not installed")
            time.sleep(3600)
            return

        url = f"wss://atlas-mainnet.helius-rpc.com/?api-key={api_key}"
        try:
            ws  = _ws_mod.create_connection(url, timeout=_WS_TIMEOUT)
        except Exception as _conn_err:
            _err_str = str(_conn_err)
            if "429" in _err_str or "rate limit" in _err_str.lower():
                log.warning(
                    "STANDBY WS ACTIVATE: triggered by helius_primary_429 — "
                    "Helius returned 429 (rate limited) during WS handshake"
                )
            else:
                log.warning(
                    "STANDBY WS ACTIVATE: triggered by helius_primary_disconnect — "
                    "connection error: %s", _conn_err,
                )
            raise
        log.info("Helius WS connected")

        with self._ws_lock:
            self._ws = ws

        # Subscribe to all pending bc pubkeys
        with self._pending_lock:
            to_sub   = set(self._pending_subs)
            self._pending_subs.clear()
        for bc_pubkey in to_sub:
            self._ws_subscribe(ws, bc_pubkey)

        while True:
            # Process pending ops first
            with self._pending_lock:
                subs   = set(self._pending_subs);   self._pending_subs.clear()
                unsubs = set(self._pending_unsubs);  self._pending_unsubs.clear()
            for bc_pubkey in subs:
                self._ws_subscribe(ws, bc_pubkey)
            for sub_id in unsubs:
                self._ws_unsubscribe(ws, sub_id)

            # Exit loop if no slots remain
            with self._lock:
                if not self._slots:
                    break

            try:
                ws.settimeout(1.0)
                try:
                    raw = ws.recv()
                except Exception:
                    # Timeout or error — loop back to check pending ops
                    continue
            except Exception:
                raise

            if not raw:
                raise ConnectionError("Helius WS: empty recv")

            try:
                msg = json.loads(raw)
            except Exception:
                continue

            self._handle_notification(msg)

        with self._ws_lock:
            self._ws = None
        try:
            ws.close()
        except Exception:
            pass
        log.info("Helius WS disconnected cleanly (no active slots)")

    def _ws_subscribe(self, ws, bc_pubkey: str):
        req_id = self._next_req_id
        self._next_req_id += 1
        msg = json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "accountSubscribe",
            "params": [bc_pubkey, {"encoding": "base64", "commitment": "confirmed"}],
        })
        try:
            ws.send(msg)
            log.debug("Helius accountSubscribe sent for bc_pda=%s (req_id=%d)",
                      bc_pubkey[:8], req_id)
        except Exception as e:
            log.warning("Helius subscribe send failed: %s", e)

    def _ws_unsubscribe(self, ws, sub_id: int):
        req_id = self._next_req_id
        self._next_req_id += 1
        msg = json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "accountUnsubscribe",
            "params": [sub_id],
        })
        try:
            ws.send(msg)
            log.debug("Helius accountUnsubscribe sent sub_id=%d", sub_id)
        except Exception as e:
            log.warning("Helius unsubscribe send failed: %s", e)

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _handle_notification(self, msg: dict):
        method = msg.get("method")

        # Subscription confirmation: {"id": N, "result": <sub_id>}
        if "result" in msg and "method" not in msg and isinstance(msg.get("result"), int):
            sub_id = msg["result"]
            # Match subscription id back to a mint via bc_pubkey lookup
            # (we find which slot is waiting for confirmation)
            with self._lock:
                for slot in self._slots.values():
                    if slot.subscription_id is None:
                        slot.subscription_id = sub_id
                        self._sub_id_map[sub_id] = slot.mint
                        log.debug("Helius sub confirmed: sub_id=%d mint=%s",
                                  sub_id, slot.mint[:8])
                        break
            return

        if method != "accountNotification":
            return

        params = msg.get("params") or {}
        sub_id = params.get("subscription")
        value  = (params.get("result") or {}).get("value") or {}
        data   = (value.get("data") or [None])[0]   # base64 string

        if sub_id is None or not data:
            return

        mint = self._sub_id_map.get(sub_id)
        if not mint:
            return

        price_usd = self._parse_bc_price(data)
        if price_usd <= 0:
            return

        now = time.time()
        today = time.strftime("%Y-%m-%d", time.gmtime(now))

        with self._lock:
            slot = self._slots.get(mint)
            if slot is None:
                return
            slot.price_usd  = price_usd
            slot.price_ts   = now
            slot.update_count += 1
            self._daily_updates += 1
            if today != self._daily_day:
                if self._daily_day:
                    log.info("Helius daily updates: %d (credits used)", self._daily_updates)
                self._daily_updates = 0
                self._daily_day = today

        log.debug("Helius price update %s: $%.10f (update #%d)",
                  mint[:8], price_usd, slot.update_count)

        # Fire PP price callbacks so event-driven stop detection still works
        # (Helius is a standby source, same callback interface)
        from memecoin.pumpportal_monitor import monitor as _pp
        with _pp._cb_lock:
            cbs = list(_pp._price_callbacks)
        for cb in cbs:
            try:
                cb(mint, price_usd)
            except Exception as _e:
                log.debug("Helius price callback error: %s", _e)

    def _parse_bc_price(self, data_b64: str) -> float:
        """
        Parse bonding curve account data → price in USD.
        Layout (after 8-byte discriminator):
          offset  8: virtual_token_reserves (u64 LE)
          offset 16: virtual_sol_reserves   (u64 LE)
        """
        try:
            raw = __import__("base64").b64decode(data_b64)
            if len(raw) < 24:
                return 0.0
            v_token, v_sol = struct.unpack_from("<QQ", raw, 8)
            if v_token == 0:
                return 0.0
            with self._sol_price_lock:
                sol_price = self._sol_price
            return (v_sol / 1e9) / (v_token / 1e6) * sol_price
        except Exception as e:
            log.debug("Helius bc parse error: %s", e)
            return 0.0

    def update_sol_price(self, price: float) -> None:
        """Called by scanner when SOL price updates."""
        with self._sol_price_lock:
            self._sol_price = price


# ---------------------------------------------------------------------------
# PDA derivation helper
# ---------------------------------------------------------------------------

def _derive_bonding_curve_pda(mint: str) -> Optional[str]:
    """
    Derive the pump.fun bonding curve PDA for a given mint address.
    Returns base58 string or None on failure.
    """
    try:
        from solders.pubkey import Pubkey
        program = Pubkey.from_string(PUMP_FUN_PROGRAM_ID)
        mint_pk = Pubkey.from_string(mint)
        pda, _  = Pubkey.find_program_address(
            [b"bonding-curve", bytes(mint_pk)],
            program,
        )
        return str(pda)
    except Exception as e:
        log.debug("PDA derivation failed for %s: %s", mint[:8], e)
        return None


# Module-level singleton
helius_monitor = HeliusAccountMonitor()
