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

import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from research.config import (
    SUPABASE_URL, SUPABASE_KEY,
    CATEGORY_INTERVALS, DEDUP_WINDOW_HOURS,
)
from research.snapshot import fetch_snapshot_with_retry
from research.tg_listener import TGAlert

log = logging.getLogger(__name__)


def _assign_category(snap: dict) -> str:
    """Assign token category from snapshot fields."""
    if not snap.get("snapshot_ok"):
        return "unknown"
    dex_id = (snap.get("dex_id") or "").lower()
    if dex_id == "pumpfun":
        return "social_alert_bc"
    elif dex_id in ("pumpswap", "raydium", "orca"):
        return "social_alert_grad"
    return "unknown"


class Tracker:
    """
    Consumes TGAlert queue, snapshots tokens, writes to Supabase,
    notifies outcome_poller via callback.
    """

    def __init__(
        self,
        in_queue: queue.Queue,
        poll_schedule_cb: Callable[[str, str, datetime, str], None],
    ):
        """
        in_queue:         TGAlert objects from TGListener
        poll_schedule_cb: called as cb(token_address, category, alert_time, chain)
                          after a successful INSERT so poller can schedule intervals
        """
        self._q    = in_queue
        self._cb   = poll_schedule_cb
        self._sb   = None   # Supabase client, initialised in thread
        self._thread: Optional[threading.Thread] = None

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

    def _insert(self, alert: TGAlert, snap: dict, attempts: int) -> Optional[str]:
        """
        INSERT row to research_tokens.
        Returns the inserted row's id, or None on failure.
        """
        if not self._sb:
            return None

        category = _assign_category(snap)

        row = {
            "token_address":      alert.token_address,
            "chain":              alert.chain,
            "alert_time":         alert.alert_time.isoformat(),
            "category":           category,
            "snapshot_ok":        snap.get("snapshot_ok", False),
            "snapshot_attempts":  attempts,
            # market fields (None if snapshot_ok=False)
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

        try:
            resp = (
                self._sb.table("research_tokens")
                .insert(row, returning="representation")
                .execute()
            )
            if resp.data:
                row_id = resp.data[0]["id"]
                log.info(
                    "Logged %s | cat=%s | liq=$%.0f | snap_ok=%s",
                    alert.token_address[:12], category,
                    snap.get("liquidity_usd") or 0,
                    snap.get("snapshot_ok"),
                )
                return row_id
        except Exception as e:
            # Unique constraint violation = already logged today (race condition)
            if "unique" in str(e).lower() or "duplicate" in str(e).lower():
                log.debug("Dedup race for %s — already in DB", alert.token_address[:8])
            else:
                log.error("Supabase INSERT failed for %s: %s", alert.token_address[:8], e)
        return None

    def _process(self, alert: TGAlert):
        # 1. Dedup
        if self._is_duplicate(alert.token_address):
            log.debug("Dedup skip %s", alert.token_address[:8])
            return

        # 2. Fetch snapshot with retry
        snap, attempts = fetch_snapshot_with_retry(alert.token_address, alert.chain)

        # 3 + 4. Insert (category assigned inside _insert)
        row_id = self._insert(alert, snap, attempts)
        if not row_id:
            return

        # 5. Notify poller — must happen AFTER successful INSERT
        category = _assign_category(snap)
        self._cb(alert.token_address, category, alert.alert_time, alert.chain)

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
