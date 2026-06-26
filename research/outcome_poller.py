"""
Outcome poller — polls DexScreener at category-specific intervals and
writes prices back to research_tokens.

Architecture:
- Min-heap of (fire_time_epoch, token_address, interval_label, chain)
- Sleeps until next poll is due, fetches price, updates Supabase
- On restart: rebuilds heap from Supabase (tokens in last POLLER_LOOKBACK_HOURS
  that have outcome_complete=False). Past-due polls execute immediately, flagged late=True.

Issue 3 from stress test: heap is in-memory → must rebuild from Supabase on boot.
Issue 6: peak = max of poll prices, not true tick-level high (documented limitation).
"""

import heapq
import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from research.config import (
    SUPABASE_URL, SUPABASE_KEY,
    CATEGORY_INTERVALS, INTERVAL_MINUTES,
    POLLER_LOOKBACK_HOURS,
)
from research.snapshot import fetch_price

log = logging.getLogger(__name__)

# Interval label → Supabase column name
# NOTE: Postgres lowercases all unquoted identifiers, so price_T3m → price_t3m.
_INTERVAL_COL = {
    "T3m":  "price_t3m",
    "T5m":  "price_t5m",
    "T10m": "price_t10m",
    "T15m": "price_t15m",
    "T20m": "price_t20m",
    "T30m": "price_t30m",
}

# All intervals for a category → their columns that hold prices
_CATEGORY_PRICE_COLS = {
    "social_alert_bc":   ["price_t3m", "price_t5m", "price_t10m", "price_t20m"],
    "social_alert_grad": ["price_t15m", "price_t30m"],
    "unknown":           ["price_t5m", "price_t10m", "price_t20m", "price_t30m"],
}


class OutcomePoller:
    """
    Min-heap based poller.  Thread-safe: schedule_token() can be called
    from the tracker thread; the poller loop runs in its own thread.
    """

    def __init__(self):
        self._heap:   list  = []   # (fire_epoch, token_address, interval_label, chain)
        self._lock            = threading.Lock()
        self._wake            = threading.Event()
        self._sb              = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="research-poller"
        )
        self._thread.start()
        log.info("Outcome poller thread started")

    # ── Public API (called from tracker thread) ───────────────────────────────

    def schedule_token(
        self,
        token_address: str,
        category: str,
        alert_time: datetime,
        chain: str = "solana",
    ):
        """Schedule all outcome polls for a token based on its category."""
        intervals = CATEGORY_INTERVALS.get(category, CATEGORY_INTERVALS["unknown"])
        now = time.time()
        with self._lock:
            for label in intervals:
                offset_min = INTERVAL_MINUTES[label]
                fire_at    = alert_time.timestamp() + offset_min * 60
                # If already past-due (e.g. restart recovery): fire immediately
                fire_at    = max(fire_at, now + 1)
                heapq.heappush(self._heap, (fire_at, token_address, label, chain))
        self._wake.set()   # wake the poller loop if sleeping

    # ── Internal ──────────────────────────────────────────────────────────────

    def _init_supabase(self):
        try:
            from supabase import create_client
            self._sb = create_client(SUPABASE_URL, SUPABASE_KEY)
            log.info("Outcome poller: Supabase client initialised")
        except Exception as e:
            log.error("Outcome poller: Supabase init failed: %s", e)

    def _rebuild_from_db(self):
        """
        On startup: load incomplete tokens from Supabase and rebuild the heap.
        Handles Issue 3 (heap lost on restart).
        Past-due polls are scheduled 1s from now (fire immediately), marked late.
        """
        if not self._sb:
            return
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=POLLER_LOOKBACK_HOURS)).isoformat()
            resp = (
                self._sb.table("research_tokens")
                .select("token_address, category, alert_time, chain")
                .eq("outcome_complete", False)
                .gte("created_at", cutoff)
                .execute()
            )
            rows = resp.data or []
            log.info("Outcome poller: rebuilding heap from %d incomplete tokens", len(rows))
            now = time.time()
            with self._lock:
                for row in rows:
                    addr     = row["token_address"]
                    cat      = row.get("category") or "unknown"
                    chain    = row.get("chain", "solana")
                    try:
                        alert_ts = datetime.fromisoformat(row["alert_time"]).timestamp()
                    except Exception:
                        alert_ts = now
                    intervals = CATEGORY_INTERVALS.get(cat, CATEGORY_INTERVALS["unknown"])
                    for label in intervals:
                        offset   = INTERVAL_MINUTES[label] * 60
                        fire_at  = alert_ts + offset
                        late     = fire_at < now
                        fire_at  = now + 2 if late else fire_at
                        heapq.heappush(self._heap, (fire_at, addr, label, chain))
            if rows:
                self._wake.set()
        except Exception as e:
            log.error("Outcome poller: DB rebuild failed: %s", e)

    def _poll(self, token_address: str, label: str, chain: str, late: bool):
        """Fetch price and update Supabase."""
        price, mcap, liq = fetch_price(token_address)
        polled_at = datetime.now(timezone.utc).isoformat()
        col = _INTERVAL_COL.get(label)

        # Log to outcome_polls table
        try:
            poll_row = {
                "token_address":  token_address,
                "interval_label": label,
                "scheduled_at":   None,   # we don't track exact scheduled_at here
                "polled_at":      polled_at,
                "price_usd":      price,
                "mcap_usd":       mcap,
                "liquidity_usd":  liq,
                "late":           late,
                "error":          None if price else "no_price",
            }
            self._sb.table("research_outcome_polls").insert(poll_row).execute()
        except Exception as e:
            log.debug("outcome_polls insert error: %s", e)

        if not col:
            return

        # Update the price column on research_tokens.
        # Write 0.0 for dead tokens (price=None) so _maybe_finalise can complete
        # the row — NULL means "not polled yet"; 0.0 means "polled, token dead".
        try:
            update = {col: price if price is not None else 0.0}
            self._sb.table("research_tokens") \
                .update(update) \
                .eq("token_address", token_address) \
                .execute()
        except Exception as e:
            log.debug("research_tokens update error for %s/%s: %s", token_address[:8], label, e)

        log.info("Poll %s %s → $%.10f%s",
                 token_address[:12], label, price or 0, " [LATE]" if late else "")

        # Check if all intervals for this token are now complete
        self._maybe_finalise(token_address)

    def _maybe_finalise(self, token_address: str):
        """
        If all expected price columns are filled for this token,
        compute pct_change_* and pct_change_peak, mark outcome_complete=True.
        """
        try:
            resp = (
                self._sb.table("research_tokens")
                .select("category, price_usd, price_t3m, price_t5m, price_t10m, price_t15m, price_t20m, price_t30m")
                .eq("token_address", token_address)
                .eq("outcome_complete", False)
                .limit(1)
                .execute()
            )
            if not resp.data:
                return
            row   = resp.data[0]
            cat   = row.get("category") or "unknown"
            p0    = row.get("price_usd")
            cols  = _CATEGORY_PRICE_COLS.get(cat, _CATEGORY_PRICE_COLS["unknown"])
            # Check all expected cols have a value
            if any(row.get(c) is None for c in cols):
                return   # not all polls done yet

            if not p0 or p0 <= 0:
                # No entry price — mark complete but skip pct calcs
                self._sb.table("research_tokens") \
                    .update({"outcome_complete": True}) \
                    .eq("token_address", token_address) \
                    .execute()
                return

            # Compute pct changes
            # NOTE: Postgres lowercases columns → price_t3m not price_T3m
            label_to_col = {
                "T3m": "price_t3m", "T5m": "price_t5m",
                "T10m": "price_t10m", "T15m": "price_t15m",
                "T20m": "price_t20m", "T30m": "price_t30m",
            }
            pct_updates = {}
            peak_pct    = None
            peak_label  = None

            for label, pcol in label_to_col.items():
                px = row.get(pcol)
                if px and px > 0:
                    pct = (px / p0 - 1) * 100
                    # Map to actual schema column names (lowercase)
                    pct_col_map = {
                        "T3m":  "pct_change_t5m",   # no separate T3m col, reuse T5m
                        "T5m":  "pct_change_t5m",
                        "T10m": "pct_change_t10m",
                        "T20m": "pct_change_t20m",
                        "T30m": "pct_change_t30m",
                    }
                    col_name = pct_col_map.get(label)
                    if col_name:
                        # Don't overwrite with a worse value (T3m shouldn't overwrite T5m)
                        if col_name not in pct_updates or pct > pct_updates[col_name]:
                            pct_updates[col_name] = round(pct, 2)
                    if peak_pct is None or pct > peak_pct:
                        peak_pct   = pct
                        peak_label = label

            update = {
                **pct_updates,
                "pct_change_peak": round(peak_pct, 2) if peak_pct is not None else None,
                "peak_interval":   peak_label,
                "outcome_complete": True,
            }
            self._sb.table("research_tokens") \
                .update(update) \
                .eq("token_address", token_address) \
                .execute()
            log.info("Finalised %s | peak=%.1f%% at %s",
                     token_address[:12], peak_pct or 0, peak_label)

        except Exception as e:
            log.error("Finalise error for %s: %s", token_address[:8], e)

    def _backfill_old_tokens(self):
        """
        One-time backfill on startup: find tokens with outcome_complete=False
        whose entire poll window has elapsed (alert_time older than 35 min).
        The heap rebuild only covers POLLER_LOOKBACK_HOURS — tokens outside that
        window are stranded forever unless we close them here.

        Strategy: mark outcome_complete=True directly.  Most of these tokens have
        price_usd=None (DexScreener never indexed them) so pct computation is
        meaningless.  For the rare ones with real entry + poll prices, pct_change
        columns will remain NULL — acceptable; analysis scripts filter on
        outcome_complete anyway.
        """
        if not self._sb:
            return
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(minutes=35)).isoformat()
            resp = (
                self._sb.table("research_tokens")
                .select("token_address", count="exact")
                .eq("outcome_complete", False)
                .lt("alert_time", cutoff)
                .execute()
            )
            n = resp.count or 0
            if not n:
                log.info("Backfill: no stuck tokens found")
                return
            log.info("Backfill: marking %d past-window tokens outcome_complete=True", n)
            self._sb.table("research_tokens") \
                .update({"outcome_complete": True}) \
                .eq("outcome_complete", False) \
                .lt("alert_time", cutoff) \
                .execute()
            log.info("Backfill complete")
        except Exception as e:
            log.error("Backfill error: %s", e)

    def _run(self):
        self._init_supabase()
        self._rebuild_from_db()
        self._backfill_old_tokens()   # one-time: close tokens past their poll window

        while True:
            self._wake.clear()
            with self._lock:
                next_fire = self._heap[0][0] if self._heap else None

            if next_fire is None:
                # Nothing scheduled — wait up to 60s for new items
                self._wake.wait(timeout=60)
                continue

            sleep_s = next_fire - time.time()
            if sleep_s > 0:
                # Wake early if new item scheduled before next_fire
                self._wake.wait(timeout=sleep_s)
                continue

            # Fire due polls
            now = time.time()
            with self._lock:
                due = []
                while self._heap and self._heap[0][0] <= now:
                    due.append(heapq.heappop(self._heap))

            for fire_at, token_address, label, chain in due:
                late = (now - fire_at) > 120   # >2 min late = restart-recovered
                try:
                    self._poll(token_address, label, chain, late)
                except Exception as e:
                    log.error("Poll error %s/%s: %s", token_address[:8], label, e)
