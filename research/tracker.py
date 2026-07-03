"""
Tracker thread — consumes the TG alert queue and writes to Supabase.

Flow per token:
  1. Dedup check (seen in last 24h? skip)
  2. Fetch snapshot with retry (DexScreener + rugcheck)
  3. Assign category from dex_id + age
  4. INSERT to research_tokens (synchronous before scheduling polls)
  5. Notify outcome_poller to schedule intervals for this token

Design decisions:
- INSERT is synchronous before polls are scheduled (Issue 5 from stress test)
- Raw fields only — no screener pass/fail boolean stored here.
  Analysis scripts compute the decision at query time using config thresholds.
- If DexScreener still NULL after retries: insert with snapshot_ok=False,
  still schedule outcome polls — token may index in time for T+5m poll.
"""

import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from research.config import (
    SUPABASE_URL, SUPABASE_KEY,
    CATEGORY_INTERVALS, DEDUP_WINDOW_HOURS,
    GRAD_VSOL_THRESHOLD,
)
from research.snapshot import fetch_snapshot_with_retry
from research.spool.writer import spool_dropped_field, spool_failed_insert
from research.tg_listener import TGAlert

log = logging.getLogger(__name__)

# Shared file written by scanner when it processes a TG alert.
# Contains PP price/vsol/buy-pressure at exact alert time — bypasses DexScreener lag.
_PP_SNAPSHOTS_PATH = Path(__file__).parent / "data" / "pp_snapshots.jsonl"


def _read_pp_snapshot(token_address: str, alert_ts: float) -> dict:
    """
    Return PP snapshot written by scanner for this token, or {} if not found.
    Matches within 120s of alert_ts (scanner and research process both see the
    TG alert nearly simultaneously, scanner is usually faster).
    """
    if not _PP_SNAPSHOTS_PATH.exists():
        return {}
    try:
        best: dict = {}
        with open(_PP_SNAPSHOTS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    if e.get("mint") == token_address:
                        if abs(e.get("ts", 0) - alert_ts) < 120:
                            best = e   # take last match (most recent)
                except Exception:
                    pass
        return best
    except Exception:
        return {}


def _enrich_with_pp(snap: dict, pp: dict) -> dict:
    """
    Merge PP snapshot data into a DexScreener snapshot.
    PP data fills gaps when DexScreener hasn't indexed the token yet (90% of cases).

    Priority: DexScreener wins if it returned a value; PP fills missing fields.
    Exception: price_usd — PP price at alert time is more accurate than
    DexScreener's 30-90s-stale price, so always prefer PP price when available.

    PP fields:
      pp_price    — USD price derived from vSol/vTokens (exact alert time)
      pp_vsol     — vSolInBondingCurve (SOL) at alert time (→ market cap)
      pp_buys     — buy tx count since token creation (proxy for buys_5m)
      pp_sells    — sell tx count since token creation
      pp_sol_in   — total SOL bought since creation (proxy for volume)
      sol_price   — SOL/USD rate at alert time (for USD volume conversion)
    """
    pp_price = pp.get("pp_price") or 0.0
    pp_vsol  = pp.get("pp_vsol")  or 0.0
    pp_buys  = pp.get("pp_buys")  or 0
    pp_sells = pp.get("pp_sells") or 0
    pp_sol   = pp.get("pp_sol_in") or 0.0
    sol_px   = pp.get("sol_price") or 0.0

    # PP price is always more accurate at alert time — prefer over DexScreener
    if pp_price > 0:
        snap["price_usd"] = pp_price
        # Pump.fun supply = 1e9 tokens; give mcap even if DexScreener missed it
        if not snap.get("mcap_usd"):
            snap["mcap_usd"] = pp_price * 1_000_000_000

    # vSol → write to pp_vsol field (new column, stored separately from DexScreener data)
    if pp_vsol > 0:
        snap["pp_vsol"] = pp_vsol

    # Buy/sell counts — only fill if DexScreener didn't provide them
    if pp_buys > 0 and not snap.get("buys_5m"):
        snap["buys_5m"] = pp_buys
    if pp_sells > 0 and not snap.get("sells_5m"):
        snap["sells_5m"] = pp_sells

    # BSR from PP counts
    total_txns = pp_buys + pp_sells
    if total_txns > 0 and not snap.get("buy_sell_ratio_5m"):
        snap["buy_sell_ratio_5m"] = round(pp_buys / total_txns, 3)

    # Volume in USD from SOL in × SOL price
    if pp_sol > 0 and sol_px > 0 and not snap.get("volume_5m"):
        snap["volume_5m"] = round(pp_sol * sol_px, 2)

    # Mark that PP data was used — helps analysis distinguish data sources
    snap["pp_snapshot_ok"] = True

    return snap


def _assign_category(snap: dict, chain: str = "solana") -> str:
    """
    Assign token category from snapshot fields.

    Priority order:
    1. pp_vsol near graduation threshold → social_alert_grad
       (PP is real-time; more reliable than stale DexScreener dex_id)
    2. DexScreener dex_id (pumpswap/raydium/orca → grad; pumpfun → bc)
    3. snapshot_ok=False + Solana → social_alert_bc (safe default)
    """
    # FIX 3c: use bonding-curve SOL to detect near-graduation tokens
    # Pump.fun accumulates ~85 SOL before graduation; >=79 SOL = graduating
    pp_vsol = snap.get("pp_vsol") or 0.0
    if chain == "solana" and pp_vsol >= GRAD_VSOL_THRESHOLD:
        return "social_alert_grad"

    if not snap.get("snapshot_ok"):
        return "social_alert_bc" if chain == "solana" else "unknown"

    dex_id = (snap.get("dex_id") or "").lower()
    if dex_id in ("pumpswap", "raydium", "orca"):
        return "social_alert_grad"
    # pumpfun or unrecognised on Solana → bonding curve
    return "social_alert_bc" if chain == "solana" else "unknown"


class Tracker:
    """
    Consumes TGAlert queue, snapshots tokens, writes to Supabase.
    Notifies outcome_poller and peak_tracker after each successful INSERT.
    """

    def __init__(
        self,
        in_queue: queue.Queue,
        poll_schedule_cb: Callable[[str, str, datetime, str], None],
        peak_schedule_cb: Optional[Callable] = None,
    ):
        """
        in_queue:          TGAlert objects from TGListener
        poll_schedule_cb:  cb(token_address, category, alert_time, chain)
        peak_schedule_cb:  cb(token_address, alert_time, entry_price) — optional
        """
        self._q              = in_queue
        self._cb             = poll_schedule_cb
        self._peak_cb        = peak_schedule_cb
        self._sb             = None
        self._thread: Optional[threading.Thread] = None
        self._recent_inserts: list = []   # timestamps for channel_velocity_5m

    def start(self):
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="research-tracker"
        )
        self._thread.start()
        log.info("Tracker thread started")

    def _init_supabase(self):
        try:
            from supabase import create_client
            self._sb = create_client(SUPABASE_URL, SUPABASE_KEY)
            log.info("Supabase client initialised")
        except Exception as e:
            log.error("Supabase init failed: %s", e)
            self._sb = None

    def _is_duplicate(self, token_address: str) -> bool:
        """Return True if token already logged in the last DEDUP_WINDOW_HOURS."""
        if not self._sb:
            return False
        try:
            from datetime import datetime, timezone, timedelta
            cutoff = (
                datetime.now(timezone.utc) - timedelta(hours=DEDUP_WINDOW_HOURS)
            ).isoformat()
            resp = (
                self._sb.table("research_tokens")
                .select("id")
                .eq("token_address", token_address)
                .gte("created_at", cutoff)
                .limit(1)
                .execute()
            )
            return len(resp.data) > 0
        except Exception as e:
            log.debug("Dedup check error for %s: %s", token_address[:8], e)
            return False

    def _get_velocity(self) -> int:
        """Count tokens inserted in the last 5 minutes (before this one)."""
        now = time.time()
        cutoff = now - 300
        self._recent_inserts = [t for t in self._recent_inserts if t > cutoff]
        count = len(self._recent_inserts)
        self._recent_inserts.append(now)
        return count

    def _insert(self, alert: TGAlert, snap: dict, attempts: int) -> Optional[str]:
        """
        INSERT row to research_tokens.
        Returns the inserted row's id, or None on failure.
        """
        if not self._sb:
            return None

        category = _assign_category(snap, alert.chain)

        row = {
            "token_address":      alert.token_address,
            "chain":              alert.chain,
            "alert_time":         alert.alert_time.isoformat(),
            "category":           category,
            # context fields — point-in-time, can't reconstruct later
            "symbol":             snap.get("symbol"),
            "tg_message_text":    alert.raw_text[:500] if alert.raw_text else None,
            "channel_velocity_5m": self._get_velocity(),
            "snapshot_ok":        snap.get("snapshot_ok", False),
            "snapshot_attempts":  attempts,
            # market fields (None if snapshot_ok=False; PP-enriched when DexScreener misses)
            "price_usd":          snap.get("price_usd"),
            "mcap_usd":           snap.get("mcap_usd"),
            "liquidity_usd":      snap.get("liquidity_usd"),
            "fdv":                snap.get("fdv"),
            "age_minutes":        snap.get("age_minutes"),
            "volume_5m":          snap.get("volume_5m"),
            "volume_1h":          snap.get("volume_1h"),
            "buys_5m":            snap.get("buys_5m"),
            "sells_5m":           snap.get("sells_5m"),
            "buy_sell_ratio_5m":  snap.get("buy_sell_ratio_5m"),
            "buys_1h":            snap.get("buys_1h"),
            "sells_1h":           snap.get("sells_1h"),
            "buy_sell_ratio_1h":  snap.get("buy_sell_ratio_1h"),
            "price_change_5m":    snap.get("price_change_5m"),
            "price_change_1h":    snap.get("price_change_1h"),
            "price_change_6h":    snap.get("price_change_6h"),
            "dex_id":             snap.get("dex_id"),
            "has_twitter":        snap.get("has_twitter"),
            "has_telegram":       snap.get("has_telegram"),
            "has_website":        snap.get("has_website"),
            "rugcheck_score":     snap.get("rugcheck_score"),
            "mint_disabled":      snap.get("mint_disabled"),
            "freeze_disabled":    snap.get("freeze_disabled"),
        }

        # Optional fields — added to extras so the retry loop strips them on PGRST204
        # rather than failing the whole INSERT.  Add new columns here as they're added
        # to Supabase; the retry loop handles missing ones automatically.
        _pp_extras: dict = {}
        if snap.get("pp_snapshot_ok"):
            _pp_extras["pp_snapshot_ok"] = True
        if snap.get("pp_vsol"):
            _pp_extras["pp_vsol"] = snap["pp_vsol"]
        if snap.get("top10_holder_pct") is not None:
            _pp_extras["top10_holder_pct"] = snap["top10_holder_pct"]
        if snap.get("creator_holds_pct") is not None:
            _pp_extras["creator_holds_pct"] = snap["creator_holds_pct"]

        def _do_insert(base: dict, extra: dict) -> Optional[str]:
            resp = (
                self._sb.table("research_tokens")
                .insert({**base, **extra}, returning="representation")
                .execute()
            )
            if resp.data:
                log.info(
                    "Logged %s | cat=%s | price=$%.8f | pp=%s | snap_ok=%s",
                    alert.token_address[:12], category,
                    snap.get("price_usd") or 0,
                    "yes" if snap.get("pp_snapshot_ok") else "no",
                    snap.get("snapshot_ok"),
                )
                return resp.data[0]["id"]
            return None

        import re as _re
        _base  = dict(row)
        _extra = dict(_pp_extras)
        _sym   = snap.get("symbol") or ""
        # Retry loop: on PGRST204, strip the offending column, spool it, and retry.
        # Stripped fields are written to spool/dropped_fields.jsonl — never silently lost.
        for _attempt in range(6):
            try:
                return _do_insert(_base, _extra)
            except Exception as e:
                e_str = str(e).lower()
                if "unique" in e_str or "duplicate" in e_str:
                    log.debug("Dedup race for %s — already in DB", alert.token_address[:8])
                    return None
                if "pgrst204" in e_str or "schema cache" in e_str:
                    m       = _re.search(r"'(\w+)'\s+column", str(e))
                    missing = m.group(1) if m else None
                    if missing and missing in _extra:
                        spool_dropped_field(
                            token_address=alert.token_address, symbol=_sym,
                            table="research_tokens", column=missing,
                            value=_extra[missing], source_file="tracker.py",
                            insert_context="base_row",
                        )
                        _extra = {k: v for k, v in _extra.items() if k != missing}
                    elif missing and missing in _base:
                        spool_dropped_field(
                            token_address=alert.token_address, symbol=_sym,
                            table="research_tokens", column=missing,
                            value=_base[missing], source_file="tracker.py",
                            insert_context="base_row",
                        )
                        _base = {k: v for k, v in _base.items() if k != missing}
                    else:
                        log.error("Supabase INSERT schema error (unrecognised) for %s: %s",
                                  alert.token_address[:8], e)
                        spool_failed_insert(
                            token_address=alert.token_address, symbol=_sym,
                            table="research_tokens", row={**_base, **_extra},
                            error=str(e), source_file="tracker.py",
                        )
                        return None
                else:
                    log.error("Supabase INSERT failed for %s: %s", alert.token_address[:8], e)
                    spool_failed_insert(
                        token_address=alert.token_address, symbol=_sym,
                        table="research_tokens", row={**_base, **_extra},
                        error=str(e), source_file="tracker.py",
                    )
                    return None
        log.error("Supabase INSERT gave up after retries for %s", alert.token_address[:8])
        spool_failed_insert(
            token_address=alert.token_address, symbol=_sym,
            table="research_tokens", row={**_base, **_extra},
            error="max_retries_exceeded", source_file="tracker.py",
        )
        return None

    def _process(self, alert: TGAlert):
        # 1. Dedup
        if self._is_duplicate(alert.token_address):
            log.debug("Dedup skip %s", alert.token_address[:8])
            return

        # 2. Fetch snapshot with retry (DexScreener + rugcheck + Jupiter fallback)
        snap, attempts = fetch_snapshot_with_retry(alert.token_address, alert.chain)

        # 2b. Enrich with PP data written by scanner at TG alert time.
        # PP data is available for all Solana tokens regardless of DexScreener lag.
        # Fields: pp_price, pp_vsol, pp_buys, pp_sells, pp_sol_in, sol_price.
        if alert.chain == "solana":
            pp = _read_pp_snapshot(alert.token_address, alert.alert_time.timestamp())
            if pp:
                snap = _enrich_with_pp(snap, pp)

        # 3 + 4. Insert (category assigned inside _insert)
        row_id = self._insert(alert, snap, attempts)
        if not row_id:
            return

        # 5. Notify poller + peak_tracker — must happen AFTER successful INSERT
        category = _assign_category(snap, alert.chain)
        self._cb(alert.token_address, category, alert.alert_time, alert.chain)
        if self._peak_cb:
            try:
                self._peak_cb(
                    alert.token_address,
                    alert.alert_time,
                    snap.get("price_usd"),
                )
            except Exception as _pe:
                log.debug("PeakTracker schedule error: %s", _pe)

    def _run(self):
        self._init_supabase()
        while True:
            try:
                alert = self._q.get(timeout=5)
            except queue.Empty:
                continue
            try:
                self._process(alert)
            except Exception as e:
                log.error("Tracker error for %s: %s", getattr(alert, "token_address", "?")[:8], e, exc_info=True)
            finally:
                self._q.task_done()
