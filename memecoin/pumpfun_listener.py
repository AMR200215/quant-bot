"""
Real-time pump.fun event listener via Solana logsSubscribe websocket.

Subscribes to the pump.fun program and fires two event types:

  new_token  — a new token was just created on pump.fun
               → check if creator is in dev_wallets → dev_launch signal
               → otherwise pass to DexScreener screener → new_launch signal

  early_buy  — a tracked whale wallet bought a token that was created
               in the last 10 minutes (before any Telegram alert fires)
               → fire copy_trade signal immediately

Uses Helius free-tier websocket (logsSubscribe is available on free tier).
Requires:  pip install websocket-client

For each create/buy detected, one getTransaction RPC call is made to resolve
the mint address and creator/buyer wallet from the instruction accounts.

Rate: pump.fun sees ~10–30 new tokens/min and far more buys. To stay within
free RPC limits, buy-transaction fetches are skipped unless the buyer is
suspected to be a tracked whale — we cannot pre-filter without the full tx,
so we use a bounded thread pool (20 workers) and a per-signature dedup cache
to avoid re-fetching the same transaction from parallel log notifications.
"""

import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

log = logging.getLogger(__name__)

PUMPFUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

# Pump.fun instruction account positions (from public IDL)
# create: [mint, mintAuthority, bondingCurve, assocBondingCurve, global,
#          mplTokenMetadata, metadata, user/creator, systemProgram, ...]
_CREATE_MINT_IDX    = 0
_CREATE_CREATOR_IDX = 7

# buy: [global, feeRecipient, mint, bondingCurve, assocBondingCurve,
#       associatedUser, user/buyer, systemProgram, ...]
_BUY_MINT_IDX  = 2
_BUY_BUYER_IDX = 6

# Track tokens created in last N seconds for early-buy detection
_EARLY_BUY_WINDOW_SEC = 600   # 10 minutes

# Websocket endpoints to try in order (rotate on repeated failure)
# Helius free tier: 5 pubsub connections allowed, but connection rate-limited on
# rapid reconnects — needs 60s+ between attempts. Works fine when stable.
# Public mainnet: heavily rate-limited on our IP from prior rapid reconnects.
# Both work fine with appropriate backoff.
_WS_ENDPOINTS = [
    "",   # placeholder — filled by _build_ws_url() using HELIUS_API_KEY if set
    "wss://api.mainnet-beta.solana.com",   # fallback
]
_RECONNECT_BASE_DELAY = 60   # 60s base, doubles per failure — avoids rate limit storms


@dataclass
class PumpEvent:
    event_type: str       # "new_token" | "early_buy"
    mint: str
    creator: str  = ""
    buyer: str    = ""
    timestamp: float = field(default_factory=time.time)


class PumpListener:
    """
    Listens to pump.fun program logs via Solana logsSubscribe websocket.
    Thread-safe: call start(), then get() in a polling loop.

    Compatible with the existing _pumpfun_thread() in scanner.py which
    already calls listener.start(daemon=True) and listener.get(timeout=2.0).
    """

    def __init__(self, strategies=None):
        self._queue: queue.Queue      = queue.Queue(maxsize=1000)
        self._thread: Optional[threading.Thread] = None
        # mint → created_ts, for early-buy window tracking
        self._recent_tokens: dict[str, float] = {}
        self._recent_lock   = threading.Lock()
        # dedup cache: signatures we've already fetched (bounded to 2000 entries)
        self._seen_sigs: set = set()
        self._seen_lock = threading.Lock()
        self._rpc_url = ""
        self._ws_url  = ""
        # Bounded thread pool for getTransaction calls
        self._fetch_sem = threading.Semaphore(20)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, daemon: bool = True):
        self._setup_urls()
        self._thread = threading.Thread(
            target=self._run, daemon=daemon, name="pumpfun-ws"
        )
        self._thread.start()

    def get(self, timeout: float = 2.0) -> Optional[PumpEvent]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_urls(self):
        key = os.getenv("HELIUS_API_KEY", "")
        # Always use public RPC for HTTP calls — reserves Helius quota for live trading
        # (executor.py confirm_tx / token_balance).
        self._rpc_url   = "https://api.mainnet-beta.solana.com"
        self._helius_ws = f"wss://mainnet.helius-rpc.com/?api-key={key}" if key else ""
        log.info("Pump.fun listener: RPC=public (Helius reserved for live trading)")

    def _ws_endpoints(self) -> list[str]:
        """
        Return WS endpoints to try in order.
        Helius free tier blocks websocket (plan limit). Public mainnet works
        but rate-limits rapid reconnects — needs 60s+ between attempts.
        """
        return ["wss://api.mainnet-beta.solana.com"]

    # ------------------------------------------------------------------
    # Main loop with auto-reconnect + endpoint rotation
    # ------------------------------------------------------------------

    def _run(self):
        try:
            import websocket as _ws_mod
        except ImportError:
            log.warning(
                "websocket-client not installed — pump.fun listener disabled. "
                "Run: pip install websocket-client"
            )
            return

        log.info("Pump.fun listener starting")
        fail_count = 0
        endpoints  = self._ws_endpoints()
        ep_idx     = 0

        while True:
            ws_url = endpoints[ep_idx % len(endpoints)]
            try:
                self._connect(_ws_mod, ws_url)
                fail_count = 0   # reset on clean disconnect
            except Exception as e:
                fail_count += 1
                delay = min(_RECONNECT_BASE_DELAY * (2 ** min(fail_count - 1, 3)), 300)
                ep_idx += 1      # rotate endpoint on every failure
                log.warning("Pump.fun WS error (attempt %d, ep=%s): %s — retry in %ds",
                            fail_count, ws_url.split("//")[1].split("/")[0], e, delay)
                time.sleep(delay)

    def _connect(self, ws_mod, ws_url: str):
        ws = ws_mod.create_connection(ws_url, timeout=30)
        log.info("Pump.fun websocket connected")

        ws.send(json.dumps({
            "jsonrpc": "2.0",
            "id":      1,
            "method":  "logsSubscribe",
            "params":  [
                {"mentions": [PUMPFUN_PROGRAM]},
                {"commitment": "confirmed"},
            ],
        }))

        last_clean = time.time()

        while True:
            raw = ws.recv()
            if not raw:
                raise ConnectionError("empty recv — connection dropped")

            try:
                msg = json.loads(raw)
            except Exception:
                continue

            # Subscription confirmation
            if "result" in msg:
                log.info("Pump.fun subscription confirmed (sub_id=%s)", msg["result"])
                continue

            value = (msg.get("params") or {}).get("result", {}).get("value", {})
            if not value or value.get("err"):
                continue

            logs = value.get("logs", [])
            sig  = value.get("signature", "")
            if not sig or not logs:
                continue

            is_create = any("Instruction: Create" in l for l in logs)
            is_buy    = any("Instruction: Buy"    in l for l in logs)

            # Dedup
            with self._seen_lock:
                if sig in self._seen_sigs:
                    continue
                self._seen_sigs.add(sig)
                if len(self._seen_sigs) > 2000:
                    self._seen_sigs.clear()

            if is_create:
                threading.Thread(
                    target=self._handle_create, args=(sig,), daemon=True
                ).start()
            elif is_buy:
                # Only spawn a fetch thread if early-buy window has any tokens
                with self._recent_lock:
                    has_recent = bool(self._recent_tokens)
                if has_recent:
                    threading.Thread(
                        target=self._handle_buy, args=(sig,), daemon=True
                    ).start()

            # Periodic cleanup every 60s
            if time.time() - last_clean > 60:
                self._clean_recent_tokens()
                last_clean = time.time()

    # ------------------------------------------------------------------
    # Transaction fetching + parsing
    # ------------------------------------------------------------------

    def _fetch_tx(self, sig: str) -> Optional[dict]:
        """Fetch parsed transaction. Retries up to 3x on 429."""
        with self._fetch_sem:
            for attempt in range(3):
                try:
                    resp = requests.post(
                        self._rpc_url,
                        json={
                            "jsonrpc": "2.0", "id": 1,
                            "method": "getTransaction",
                            "params": [sig, {
                                "encoding": "jsonParsed",
                                "maxSupportedTransactionVersion": 0,
                                "commitment": "confirmed",
                            }],
                        },
                        timeout=12,
                    )
                    if resp.status_code == 429:
                        time.sleep(2 ** attempt)
                        continue
                    return resp.json().get("result")
                except Exception as e:
                    log.debug("getTransaction %s attempt %d: %s", sig[:12], attempt, e)
                    time.sleep(1)
            return None

    def _get_pumpfun_accounts(self, tx: dict) -> Optional[list[str]]:
        """
        Return the pump.fun instruction's accounts as a flat list of pubkey strings.

        In jsonParsed encoding, for programs Solana cannot natively decode
        (like pump.fun), the instruction 'accounts' field is already a list
        of pubkey strings — NOT integer indices into accountKeys.
        """
        try:
            msg = tx["transaction"]["message"]
            for ix in msg.get("instructions", []):
                if ix.get("programId") == PUMPFUN_PROGRAM:
                    return ix.get("accounts", [])
            for group in (tx.get("meta") or {}).get("innerInstructions", []):
                for ix in group.get("instructions", []):
                    if ix.get("programId") == PUMPFUN_PROGRAM:
                        return ix.get("accounts", [])
        except Exception as e:
            log.debug("Account parse error: %s", e)
        return None

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_create(self, sig: str):
        time.sleep(1)   # wait 1s for tx to be indexable
        tx = self._fetch_tx(sig)
        if not tx:
            return

        accs = self._get_pumpfun_accounts(tx)
        if not accs or len(accs) <= max(_CREATE_MINT_IDX, _CREATE_CREATOR_IDX):
            return

        try:
            # accs is a list of pubkey strings directly (jsonParsed encoding)
            mint    = accs[_CREATE_MINT_IDX]
            creator = accs[_CREATE_CREATOR_IDX]
        except IndexError:
            return

        # Register as recent token for early-buy window
        with self._recent_lock:
            self._recent_tokens[mint] = time.time()

        event = PumpEvent(event_type="new_token", mint=mint, creator=creator)
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            log.debug("Event queue full — dropping create event for %s", mint[:8])

        log.debug("CREATE  mint=%s  creator=%s", mint[:8], creator[:8])

    def _handle_buy(self, sig: str):
        time.sleep(1)
        tx = self._fetch_tx(sig)
        if not tx:
            return

        accs = self._get_pumpfun_accounts(tx)
        if not accs or len(accs) <= max(_BUY_MINT_IDX, _BUY_BUYER_IDX):
            return

        try:
            mint  = accs[_BUY_MINT_IDX]
            buyer = accs[_BUY_BUYER_IDX]
        except IndexError:
            return

        # Only fire if token is still in the early-buy window
        with self._recent_lock:
            created_ts = self._recent_tokens.get(mint)
        if not created_ts or (time.time() - created_ts) > _EARLY_BUY_WINDOW_SEC:
            return

        event = PumpEvent(event_type="early_buy", mint=mint, buyer=buyer)
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            log.debug("Event queue full — dropping buy event for %s", mint[:8])

        log.debug("EARLY_BUY  mint=%s  buyer=%s", mint[:8], buyer[:8])

    def _clean_recent_tokens(self):
        cutoff = time.time() - _EARLY_BUY_WINDOW_SEC
        with self._recent_lock:
            stale = [m for m, ts in self._recent_tokens.items() if ts < cutoff]
            for m in stale:
                del self._recent_tokens[m]
        if stale:
            log.debug("Cleaned %d stale tokens from early-buy window", len(stale))
