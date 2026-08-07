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
    GRAD_VSOL_THRESHOLD, GRAD_SOL_UI,
)
from research.snapshot import fetch_snapshot_with_retry
from research.spool.writer import spool_dropped_field, spool_failed_insert
from research.tg_listener import TGAlert

log = logging.getLogger(__name__)

# Shared file written by scanner when it processes a TG alert.
# Contains PP price/vsol/buy-pressure at exact alert time — bypasses DexScreener lag.
_PP_SNAPSHOTS_PATH = Path(__file__).parent / "data" / "pp_snapshots.jsonl"

# PROGRESS-FIX PF5: event-keyed progress store, written by
# memecoin/progress_capture.py. Replaces mint+-120s matching (_read_pp_snapshot,
# still used for the OTHER pp_* enrichment fields — price/buy-pressure — which
# this migration doesn't touch) for progress_at_signal specifically.
_PROGRESS_SNAPSHOTS_PATH = Path(__file__).parent / "data" / "progress_snapshots.jsonl"
_PROGRESS_WAIT_TOTAL_S   = 1.5   # bounded — this is the research process, never blocks trading
_PROGRESS_WAIT_STEP_S    = 0.3


def _read_progress_snapshot(event_id: str, max_wait_s: float = _PROGRESS_WAIT_TOTAL_S) -> dict:
    """
    Return the ProgressCapture result for this exact event_id, or {} if it
    never arrives within max_wait_s. Capture runs asynchronously on the
    live-bot side, so a short bounded wait/retry here (safe — this is the
    research process, not the trading path) catches the common case where
    this INSERT runs slightly before capture has finished (curve_account
    capture typically resolves in well under a second).

    Exact event_id match only — no mint+time-window guessing (PF5).
    """
    if not event_id:
        return {}
    deadline = time.time() + max_wait_s
    while True:
        result = _scan_progress_file(event_id)
        if result or time.time() >= deadline:
            return result
        time.sleep(_PROGRESS_WAIT_STEP_S)


def _scan_progress_file(event_id: str) -> dict:
    if not _PROGRESS_SNAPSHOTS_PATH.exists():
        return {}
    try:
        best: dict = {}
        with open(_PROGRESS_SNAPSHOTS_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                    if e.get("event_id") == event_id:
                        best = e   # take last match — a later re-write for the
                                   # same event_id (shouldn't normally happen,
                                   # capture writes once) wins
                except Exception:
                    pass
        return best
    except Exception:
        return {}


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

    # PROGRESS-FIX PF7: pp_snapshot_ok must mean a real PumpPortal observation
    # existed, not merely that a (possibly all-default) snapshot dict was
    # written. Previously set unconditionally here, so a freshly-created
    # ScreeningState read back before any real tick arrived (the exact PF1
    # race) still reported pp_snapshot_ok=True with every pp_* value at its
    # zero default — a silently false "yes, PP data was used".
    snap["pp_snapshot_ok"] = (pp_price > 0 or pp_vsol > 0 or pp_buys > 0
                               or pp_sells > 0 or pp_sol > 0)

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

    def _check_dedup(self, token_address: str) -> tuple:
        """
        Return (is_new_event, existing_id):
          - (True, None)         — no row in last DEDUP_WINDOW_HOURS → new event, insert
          - (False, existing_id) — row exists within window → genuine re-alert, update it
        On DB error: returns (True, None) so the insert path runs (safe default).
        """
        if not self._sb:
            return (True, None)
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
            if resp.data:
                return (False, resp.data[0]["id"])
            return (True, None)
        except Exception as e:
            log.debug("Dedup check error for %s: %s", token_address[:8], e)
            return (True, None)

    def _record_realert(self, existing_id: str, alert: "TGAlert"):
        """Atomically increment realert_count and append to realert_times."""
        try:
            # Read current values first
            resp = self._sb.table("research_tokens") \
                .select("realert_count, realert_times, realert_message_ids") \
                .eq("id", existing_id) \
                .limit(1).execute()
            if not resp.data:
                return
            row = resp.data[0]
            count = (row.get("realert_count") or 0) + 1
            times = list(row.get("realert_times") or [])
            msg_ids = list(row.get("realert_message_ids") or [])
            times.append(alert.alert_time.isoformat())
            msg_ids.append(getattr(alert, "tg_message_id", None))
            self._sb.table("research_tokens").update({
                "realert_count":       count,
                "realert_times":       times,
                "realert_message_ids": [m for m in msg_ids if m is not None],
                "last_realert_time":   alert.alert_time.isoformat(),
            }).eq("id", existing_id).execute()
            log.info("Realert #%d for %s (id=%s)", count, alert.token_address[:8], existing_id[:8])
        except Exception as e:
            log.debug("Realert update failed for %s: %s", existing_id[:8], e)

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
        _pp_extras: dict = {
            # RF4 realert tracking — initialised at zero on first insert
            "realert_count":       0,
            "realert_times":       [],
            "realert_message_ids": [],
            "last_realert_time":   None,
        }
        if snap.get("pp_snapshot_ok"):
            _pp_extras["pp_snapshot_ok"] = True
        pp_vsol = snap.get("pp_vsol")
        if pp_vsol:
            # Retained for compatibility (PF8) — this is the OLD pp_snapshot-
            # derived value, NOT authoritative for progress_at_signal anymore
            # (see PROGRESS-FIX block below). Kept purely as a debugging/
            # cross-check field; do not derive progress_at_signal from it —
            # that path had the exact subscribe-then-immediately-read race
            # (PF1) that motivated this whole fix.
            _pp_extras["pp_vsol"] = pp_vsol

        # PROGRESS-FIX PF2/PF5/PF6/PF8: canonical, source-provenanced
        # progress capture, keyed by event_id — the SAME value memecoin/
        # v8_paper.py uses for the same event (PF6). NEVER coerces a
        # missing/failed measurement to 0 — stays NULL with an explicit
        # progress_status reason (PF2, PF11 depend on this distinction).
        _progress = _read_progress_snapshot(alert.event_id) if alert.event_id else {}
        if _progress:
            _pp_extras["vsol_at_signal"]          = _progress.get("vsol_at_signal")
            _pp_extras["progress_at_signal"]       = _progress.get("progress_at_signal")
            _pp_extras["progress_source"]          = _progress.get("progress_source")
            _obs_at = _progress.get("progress_observed_at")
            if _obs_at:
                _pp_extras["progress_observed_at"] = datetime.fromtimestamp(_obs_at, tz=timezone.utc).isoformat()
            _pp_extras["progress_capture_lag_ms"]  = _progress.get("progress_capture_lag_ms")
            _pp_extras["progress_status"]          = _progress.get("progress_status")
            _pp_extras["progress_data_ok"]         = _progress.get("progress_status") == "ok"
            _pp_extras["progress_schema_version"]  = 1
        else:
            _pp_extras["progress_status"]  = "capture_missing"   # never arrived in time
            _pp_extras["progress_data_ok"] = False
        if snap.get("top10_holder_pct") is not None:
            _pp_extras["top10_holder_pct"] = snap["top10_holder_pct"]
        if snap.get("creator_holds_pct") is not None:
            _pp_extras["creator_holds_pct"] = snap["creator_holds_pct"]

        # Backfill provenance — stored in extras so missing columns degrade gracefully
        if alert.backfilled:
            _pp_extras["backfilled"]        = True
            _pp_extras["source"]            = alert.source
            _pp_extras["event_id"]          = alert.event_id
            if alert.backfill_batch_id:
                _pp_extras["backfill_batch_id"] = alert.backfill_batch_id
        else:
            _pp_extras["backfilled"] = False
            _pp_extras["source"]     = alert.source

        # Smart-money: check if any early buyers are known winning wallets
        if alert.chain == "solana":
            try:
                from research.config import HELIUS_API_KEY as _hk
                from research.snapshot import fetch_first_buyers as _ffb
                from research.smart_wallets import (
                    check_smart_money as _csm,
                    get_loaded_version as _glv,
                    check_smart_money_versioned as _csmv,
                )
                if _hk:
                    _buyers = _ffb(alert.token_address, _hk, n=30)
                    if _buyers:
                        # Pinned version scoring (always)
                        _sm_hit, _sm_count = _csm(_buyers)
                        _pp_extras["smart_money_hit"]   = _sm_hit
                        _pp_extras["smart_money_count"] = _sm_count

                        # RF6: shadow V1 scoring fields (PGRST204-strippable extras)
                        _pinned_v = _glv()
                        if _pinned_v:
                            _pp_extras["smart_money_data_ok_v1"]  = True
                            _pp_extras["smart_money_hit_v1"]      = _sm_hit
                            _pp_extras["smart_money_count_v1"]    = _sm_count
                    else:
                        # Explicitly store False so coverage is trackable
                        _pp_extras["smart_money_hit"]   = False
                        _pp_extras["smart_money_count"] = 0
            except Exception as _sme:
                log.debug("smart_money check failed %s: %s", alert.token_address[:8], _sme)

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
                    _at     = alert.alert_time.isoformat()
                    if missing and missing in _extra:
                        spool_dropped_field(
                            token_address=alert.token_address, symbol=_sym,
                            table="research_tokens", column=missing,
                            value=_extra[missing], source_file="tracker.py",
                            insert_context="base_row", alert_time=_at,
                        )
                        _extra = {k: v for k, v in _extra.items() if k != missing}
                    elif missing and missing in _base:
                        spool_dropped_field(
                            token_address=alert.token_address, symbol=_sym,
                            table="research_tokens", column=missing,
                            value=_base[missing], source_file="tracker.py",
                            insert_context="base_row", alert_time=_at,
                        )
                        _base = {k: v for k, v in _base.items() if k != missing}
                    else:
                        log.error("Supabase INSERT schema error (unrecognised) for %s: %s",
                                  alert.token_address[:8], e)
                        spool_failed_insert(
                            token_address=alert.token_address, symbol=_sym,
                            table="research_tokens", row={**_base, **_extra},
                            error=str(e), source_file="tracker.py", alert_time=_at,
                        )
                        return None
                else:
                    log.error("Supabase INSERT failed for %s: %s", alert.token_address[:8], e)
                    spool_failed_insert(
                        token_address=alert.token_address, symbol=_sym,
                        table="research_tokens", row={**_base, **_extra},
                        error=str(e), source_file="tracker.py",
                        alert_time=alert.alert_time.isoformat(),
                    )
                    return None
        log.error("Supabase INSERT gave up after retries for %s", alert.token_address[:8])
        spool_failed_insert(
            token_address=alert.token_address, symbol=_sym,
            table="research_tokens", row={**_base, **_extra},
            error="max_retries_exceeded", source_file="tracker.py",
            alert_time=alert.alert_time.isoformat(),
        )
        return None

    def _process(self, alert: TGAlert):
        # 1. Dedup / realert check
        is_new, existing_id = self._check_dedup(alert.token_address)
        if not is_new:
            self._record_realert(existing_id, alert)
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

        # Update health timestamp so health_monitor can track research insert activity
        try:
            from memecoin.health_monitor import update_health_timestamp
            update_health_timestamp("_last_research_insert", __import__("time").time())
        except Exception:
            pass

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
