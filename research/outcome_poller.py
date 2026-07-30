"""
Outcome poller — polls prices at category-specific intervals and writes prices
back to research_tokens.

RF1 change: CURVE_ACTIVE tokens (social_alert_bc with pp_vsol < GRAD_SOL_UI)
now use bonding-curve account data via Helius getMultipleAccounts as the primary
price source.  DexScreener is used as fallback and for GRADUATED tokens.

Per-interval provenance columns (price_source_*, price_status_*,
price_observed_at_*) are written alongside each price poll.

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
import re as _re
import threading
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional

from research.config import (
    SUPABASE_URL, SUPABASE_KEY,
    CATEGORY_INTERVALS, INTERVAL_MINUTES,
    POLLER_LOOKBACK_HOURS,
    HELIUS_API_KEY,
    GRAD_SOL_UI,
)
from research.snapshot import fetch_price
from research.curve_oracle import get_curve_prices_batch, get_sol_usd_cached
from research.spool.writer import spool_dropped_field, spool_failed_insert

log = logging.getLogger(__name__)


# ── VenueState enum ────────────────────────────────────────────────────────────

class VenueState(str, Enum):
    CURVE_ACTIVE = "CURVE_ACTIVE"
    GRADUATED    = "GRADUATED"
    DEX_ACTIVE   = "DEX_ACTIVE"
    UNKNOWN      = "UNKNOWN"


# ── Interval label → column mappings ─────────────────────────────────────────

# Interval label → Supabase column name
# NOTE: Postgres lowercases all unquoted identifiers, so price_T3m → price_t3m.
_INTERVAL_COL = {
    "T1m":  "price_t1m",
    "T3m":  "price_t3m",
    "T5m":  "price_t5m",
    "T10m": "price_t10m",
    "T15m": "price_t15m",
    "T20m": "price_t20m",
    "T30m": "price_t30m",
}

# Per-interval provenance columns
_INTERVAL_SOURCE_COL = {
    "T1m":  "price_source_t1m",
    "T3m":  "price_source_t3m",
    "T5m":  "price_source_t5m",
    "T10m": "price_source_t10m",
    "T15m": "price_source_t15m",
    "T20m": "price_source_t20m",
    "T30m": "price_source_t30m",
}

_INTERVAL_STATUS_COL = {
    "T1m":  "price_status_t1m",
    "T3m":  "price_status_t3m",
    "T5m":  "price_status_t5m",
    "T10m": "price_status_t10m",
    "T15m": "price_status_t15m",
    "T20m": "price_status_t20m",
    "T30m": "price_status_t30m",
}

_INTERVAL_OBSERVED_COL = {
    "T1m":  "price_observed_at_t1m",
    "T3m":  "price_observed_at_t3m",
    "T5m":  "price_observed_at_t5m",
    "T10m": "price_observed_at_t10m",
    "T15m": "price_observed_at_t15m",
    "T20m": "price_observed_at_t20m",
    "T30m": "price_observed_at_t30m",
}

# All intervals for a category → their columns that hold prices
_CATEGORY_PRICE_COLS = {
    "social_alert_bc":   ["price_t1m", "price_t3m", "price_t5m", "price_t10m", "price_t20m"],
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
        alert_time_iso = alert_time.isoformat()
        with self._lock:
            for label in intervals:
                offset_min = INTERVAL_MINUTES[label]
                fire_at    = alert_time.timestamp() + offset_min * 60
                # If already past-due (e.g. restart recovery): fire immediately
                fire_at    = max(fire_at, now + 1)
                heapq.heappush(self._heap, (fire_at, token_address, label, chain, alert_time_iso))
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
            # Paginate: Supabase caps SELECT at 1000 rows by default
            rows = []
            offset = 0
            batch  = 1000
            while True:
                resp = (
                    self._sb.table("research_tokens")
                    .select("token_address, category, alert_time, chain")
                    .eq("outcome_complete", False)
                    .gte("created_at", cutoff)
                    .range(offset, offset + batch - 1)
                    .execute()
                )
                chunk = resp.data or []
                rows.extend(chunk)
                if len(chunk) < batch:
                    break
                offset += batch
            log.info("Outcome poller: rebuilding heap from %d incomplete tokens", len(rows))
            now = time.time()
            with self._lock:
                for row in rows:
                    addr           = row["token_address"]
                    cat            = row.get("category") or "unknown"
                    chain          = row.get("chain", "solana")
                    alert_time_str = row.get("alert_time", "")
                    try:
                        alert_ts = datetime.fromisoformat(alert_time_str).timestamp()
                    except Exception:
                        alert_ts = now
                        alert_time_str = ""
                    intervals = CATEGORY_INTERVALS.get(cat, CATEGORY_INTERVALS["unknown"])
                    for label in intervals:
                        offset_s = INTERVAL_MINUTES[label] * 60
                        fire_at  = alert_ts + offset_s
                        late     = fire_at < now
                        fire_at  = now + 2 if late else fire_at
                        heapq.heappush(self._heap, (fire_at, addr, label, chain, alert_time_str))
            if rows:
                self._wake.set()
        except Exception as e:
            log.error("Outcome poller: DB rebuild failed: %s", e)

    # ── Venue state determination ─────────────────────────────────────────────

    def _get_token_meta(self, token_address: str) -> dict:
        """
        Read category and pp_vsol for a token from DB.
        Returns {"category": str, "pp_vsol": float|None}.
        Returns defaults on any error.
        """
        try:
            resp = (
                self._sb.table("research_tokens")
                .select("category, pp_vsol")
                .eq("token_address", token_address)
                .limit(1)
                .execute()
            )
            if resp.data:
                return {
                    "category": resp.data[0].get("category") or "unknown",
                    "pp_vsol":  resp.data[0].get("pp_vsol"),
                }
        except Exception as e:
            log.debug("_get_token_meta failed for %s: %s", token_address[:8], e)
        return {"category": "unknown", "pp_vsol": None}

    def _determine_venue_state(self, category: str, pp_vsol: Optional[float]) -> VenueState:
        """
        Determine the venue state for a token based on its category and pp_vsol.

        Rules:
        - social_alert_bc with pp_vsol < GRAD_SOL_UI → CURVE_ACTIVE
        - social_alert_bc with pp_vsol >= GRAD_SOL_UI → GRADUATED
        - social_alert_grad → GRADUATED
        - unknown → UNKNOWN (try curve then DEX)
        """
        if category == "social_alert_grad":
            return VenueState.GRADUATED

        if category == "social_alert_bc":
            if pp_vsol is not None and pp_vsol >= GRAD_SOL_UI:
                return VenueState.GRADUATED
            return VenueState.CURVE_ACTIVE   # default for BC tokens

        # "unknown" or anything else
        return VenueState.UNKNOWN

    # ── Price fetch with venue routing ────────────────────────────────────────

    def _fetch_price_with_venue(
        self,
        token_address: str,
        category: str,
        pp_vsol: Optional[float],
    ) -> tuple[Optional[float], str, Optional[str]]:
        """
        Fetch price using the appropriate venue for this token's state.

        Returns: (price_usd, source, failure_reason)
          source: "curve_account" | "dexscreener" | "jupiter" | None
          failure_reason: None (success) | "curve_rpc_error" | "curve_account_missing" |
                          "curve_parse_error" | "curve_layout_unknown" | "sol_usd_stale" |
                          "no_price" (all failed)
        """
        venue_state = self._determine_venue_state(category, pp_vsol)

        # ── CURVE_ACTIVE path ─────────────────────────────────────────────────
        if venue_state in (VenueState.CURVE_ACTIVE, VenueState.UNKNOWN):
            curve_result = self._try_curve_price(token_address)

            if curve_result is not None:
                curve_venue, price, failure = curve_result

                if curve_venue == "CURVE_ACTIVE" and price is not None:
                    return price, "curve_account", None

                if curve_venue == "GRADUATED":
                    # complete=True in account data → fall through to DEX
                    log.info("curve_oracle: %s complete=True → switching to DEX path",
                             token_address[:8])
                    # fall through to DEX below

                elif failure is not None:
                    # RPC/parse error — try DexScreener as fallback but do NOT treat as graduation
                    log.debug("curve_oracle fallback to DEX for %s: %s", token_address[:8], failure)
                    dex_price, dex_source, dex_failure = self._try_dex_price(token_address)
                    if dex_price is not None:
                        return dex_price, dex_source, None
                    # All failed — return the original curve failure reason
                    return None, None, failure

                # CURVE_MISSING or PARSE_ERROR without a specific re-route — try DEX
                if price is None and curve_venue in ("CURVE_MISSING", "PARSE_ERROR"):
                    dex_price, dex_source, dex_failure = self._try_dex_price(token_address)
                    if dex_price is not None:
                        return dex_price, dex_source, None
                    return None, None, failure or dex_failure

        # ── GRADUATED / DEX_ACTIVE path ───────────────────────────────────────
        dex_price, dex_source, dex_failure = self._try_dex_price(token_address)
        if dex_price is not None:
            return dex_price, dex_source, None

        return None, None, dex_failure or "no_price"

    def _try_curve_price(self, token_address: str) -> Optional[tuple]:
        """
        Attempt to get price from bonding-curve account.
        Returns (venue_state_str, price_usd|None, failure_reason|None) or None if skipped.
        """
        if not HELIUS_API_KEY:
            log.debug("curve_oracle: no HELIUS_API_KEY — skipping curve lookup for %s",
                      token_address[:8])
            return None

        sol_usd, sol_age = get_sol_usd_cached()

        results = get_curve_prices_batch(
            mints=[token_address],
            helius_key=HELIUS_API_KEY,
            sol_price_usd=sol_usd,
            sol_price_age_s=sol_age,
        )

        r = results.get(token_address)
        if r is None:
            return None

        return r["venue_state"], r.get("price_usd"), r.get("failure_reason")

    def _try_dex_price(self, token_address: str) -> tuple:
        """
        Try DexScreener, then Jupiter fallback.
        Returns (price_usd|None, source|None, failure_reason|None).
        """
        price, mcap, liq = fetch_price(token_address)
        if price is not None:
            # Determine source — fetch_price tries DexScreener first then Jupiter;
            # we can't easily distinguish them here, so call snapshot to check
            # but fetch_price itself tries dex first, then Jupiter.
            # We'll label based on whether we got mcap too (dex has mcap, jupiter doesn't)
            source = "dexscreener" if mcap is not None else "jupiter"
            return price, source, None
        return None, None, "no_price"

    # ── Poll execution ─────────────────────────────────────────────────────────

    def _poll(self, token_address: str, label: str, chain: str, late: bool,
              alert_time_iso: str = ""):
        """Fetch price with venue routing and update Supabase with provenance columns."""
        polled_at = datetime.now(timezone.utc).isoformat()
        col = _INTERVAL_COL.get(label)

        # Read token meta for venue determination
        meta     = self._get_token_meta(token_address)
        category = meta["category"]
        pp_vsol  = meta["pp_vsol"]

        # Fetch price with curve-first routing
        price, source, failure_reason = self._fetch_price_with_venue(
            token_address, category, pp_vsol
        )

        # Back-compat: also retrieve mcap/liq for outcome_polls log
        # (only available from DexScreener path — accept None from curve path)
        mcap = None
        liq  = None
        if source == "dexscreener":
            _p, mcap, liq = fetch_price.__wrapped__(token_address) \
                if hasattr(fetch_price, "__wrapped__") else (price, None, None)

        # Log to outcome_polls table
        try:
            poll_row = {
                "token_address":  token_address,
                "interval_label": label,
                "scheduled_at":   None,
                "polled_at":      polled_at,
                "price_usd":      price,
                "mcap_usd":       mcap,
                "liquidity_usd":  liq,
                "late":           late,
                "error":          failure_reason if price is None else None,
            }
            self._sb.table("research_outcome_polls").insert(poll_row).execute()
        except Exception as e:
            log.debug("outcome_polls insert error: %s", e)

        if not col:
            return

        # ── Check if interval already populated — skip price write if so ─────
        try:
            existing_resp = (
                self._sb.table("research_tokens")
                .select(col)
                .eq("token_address", token_address)
                .limit(1)
                .execute()
            )
            existing_price = (existing_resp.data or [{}])[0].get(col) if existing_resp.data else None
        except Exception:
            existing_price = None

        # Build provenance columns (always write these even if price already set)
        source_col   = _INTERVAL_SOURCE_COL.get(label)
        status_col   = _INTERVAL_STATUS_COL.get(label)
        observed_col = _INTERVAL_OBSERVED_COL.get(label)

        if existing_price is not None:
            # Price already populated — only write provenance if missing
            prov_upd: dict = {}
            if source_col and source:
                prov_upd[source_col] = source
            if status_col:
                prov_upd[status_col] = failure_reason
            if observed_col:
                prov_upd[observed_col] = polled_at
            if prov_upd:
                self._safe_update(token_address, prov_upd, label, alert_time_iso,
                                  context="provenance_only")
            log.debug("Poll %s %s: price already set (%.10f), skipping price write",
                      token_address[:12], label, existing_price)
        else:
            # Build full update dict: price + provenance
            upd: dict = {}
            if price is not None:
                upd[col] = price
            if source_col:
                upd[source_col] = source
            if status_col:
                upd[status_col] = failure_reason
            if observed_col:
                upd[observed_col] = polled_at

            if upd:
                self._safe_update(token_address, upd, label, alert_time_iso,
                                  context="outcome_update")

        log.info("Poll %s %s → %s [src=%s]%s",
                 token_address[:12], label,
                 f"${price:.10f}" if price else "NULL",
                 source or "none",
                 " [LATE]" if late else "")

        # Check if this token is ready to finalise
        self._maybe_finalise(token_address)

    def _safe_update(
        self,
        token_address: str,
        upd: dict,
        label: str,
        alert_time_iso: str,
        context: str = "outcome_update",
    ):
        """
        Write `upd` to research_tokens with PGRST204 retry-strip pattern.
        Spools any column that causes a schema-cache error.
        """
        _upd = dict(upd)
        for _attempt in range(6):
            try:
                self._sb.table("research_tokens") \
                    .update(_upd) \
                    .eq("token_address", token_address) \
                    .execute()
                return
            except Exception as e:
                _es = str(e).lower()
                if "pgrst204" in _es or "schema cache" in _es:
                    _m    = _re.search(r"'(\w+)'\s+column", str(e))
                    _miss = _m.group(1) if _m else None
                    if _miss and _miss in _upd:
                        spool_dropped_field(
                            token_address=token_address, symbol="",
                            table="research_tokens", column=_miss,
                            value=_upd[_miss], source_file="outcome_poller.py",
                            insert_context=context, alert_time=alert_time_iso,
                        )
                        _upd = {k: v for k, v in _upd.items() if k != _miss}
                        if not _upd:
                            return
                    else:
                        spool_failed_insert(
                            token_address=token_address, symbol="",
                            table="research_tokens",
                            row={**_upd, "interval_label": label},
                            error=str(e)[:200], source_file="outcome_poller.py",
                            insert_context=context, alert_time=alert_time_iso,
                        )
                        return
                else:
                    spool_failed_insert(
                        token_address=token_address, symbol="",
                        table="research_tokens",
                        row={**_upd, "interval_label": label},
                        error=str(e)[:200], source_file="outcome_poller.py",
                        insert_context=context, alert_time=alert_time_iso,
                    )
                    return

    def _maybe_finalise(self, token_address: str):
        """
        Finalise a token when either:
          (a) all expected price columns are non-NULL, OR
          (b) enough time has elapsed that all polls should have fired.

        FIX: NULL means "polled, no price found" — compute pct only for
        non-NULL intervals.  Set data_partial=True if any interval is NULL
        at finalisation time so analysis queries can exclude incomplete rows.
        """
        try:
            try:
                resp = (
                    self._sb.table("research_tokens")
                    .select("category, price_usd, alert_time, "
                            "price_t1m, price_t3m, price_t5m, price_t10m, "
                            "price_t15m, price_t20m, price_t30m")
                    .eq("token_address", token_address)
                    .eq("outcome_complete", False)
                    .limit(1)
                    .execute()
                )
            except Exception as _sel_e:
                if "price_t1m" in str(_sel_e):
                    resp = (
                        self._sb.table("research_tokens")
                        .select("category, price_usd, alert_time, "
                                "price_t3m, price_t5m, price_t10m, "
                                "price_t15m, price_t20m, price_t30m")
                        .eq("token_address", token_address)
                        .eq("outcome_complete", False)
                        .limit(1)
                        .execute()
                    )
                else:
                    raise

            if not resp.data:
                return
            row = resp.data[0]
            cat = row.get("category") or "unknown"
            p0  = row.get("price_usd")

            intervals = CATEGORY_INTERVALS.get(cat, CATEGORY_INTERVALS["unknown"])
            cols      = _CATEGORY_PRICE_COLS.get(cat, _CATEGORY_PRICE_COLS["unknown"])

            # Timer: have all polls had time to fire? (max interval + 2 min grace)
            all_polls_due = False
            alert_time_str = row.get("alert_time")
            if alert_time_str:
                try:
                    alert_ts  = datetime.fromisoformat(alert_time_str)
                    max_min   = max(INTERVAL_MINUTES[l] for l in intervals if l in INTERVAL_MINUTES)
                    elapsed   = (datetime.now(timezone.utc) - alert_ts).total_seconds() / 60
                    all_polls_due = elapsed >= (max_min + 2)
                except Exception:
                    pass

            null_cols = [c for c in cols if row.get(c) is None]

            if null_cols and not all_polls_due:
                return  # still waiting for polls to fire

            data_partial = bool(null_cols)

            if not p0 or p0 <= 0:
                update = {"outcome_complete": True, "data_partial": data_partial}
                try:
                    self._sb.table("research_tokens").update(update) \
                        .eq("token_address", token_address).execute()
                except Exception:
                    self._sb.table("research_tokens") \
                        .update({"outcome_complete": True}) \
                        .eq("token_address", token_address).execute()
                return

            label_to_col = {
                "T1m": "price_t1m",
                "T3m": "price_t3m",  "T5m":  "price_t5m",
                "T10m": "price_t10m", "T15m": "price_t15m",
                "T20m": "price_t20m", "T30m": "price_t30m",
            }
            pct_col_map = {
                "T1m":  "pct_change_t1m",
                "T3m":  "pct_change_t3m",
                "T5m":  "pct_change_t5m",
                "T10m": "pct_change_t10m",
                "T15m": "pct_change_t15m",
                "T20m": "pct_change_t20m",
                "T30m": "pct_change_t30m",
            }
            pct_updates = {}
            peak_pct    = None
            peak_label  = None

            for label, pcol in label_to_col.items():
                px = row.get(pcol)
                # Skip NULL (failed/unpolled) — never compute -100% on missing data
                if px is not None and px > 0:
                    pct = (px / p0 - 1) * 100
                    col_name = pct_col_map.get(label)
                    if col_name:
                        pct_updates[col_name] = round(pct, 2)
                    if peak_pct is None or pct > peak_pct:
                        peak_pct  = pct
                        peak_label = label

            update = {
                **pct_updates,
                "pct_change_peak":  round(peak_pct, 2) if peak_pct is not None else None,
                "peak_interval":    peak_label,
                "outcome_complete": True,
                "data_partial":     data_partial,
            }
            _upd = dict(update)
            for _fa in range(6):
                try:
                    self._sb.table("research_tokens") \
                        .update(_upd) \
                        .eq("token_address", token_address) \
                        .execute()
                    break
                except Exception as _upd_e:
                    _upd_s = str(_upd_e).lower()
                    if "pgrst204" in _upd_s or "schema cache" in _upd_s:
                        _m      = _re.search(r"'(\w+)'\s+column", str(_upd_e))
                        _miss   = _m.group(1) if _m else None
                        if _miss and _miss in _upd:
                            spool_dropped_field(
                                token_address=token_address, symbol="",
                                table="research_tokens", column=_miss,
                                value=_upd[_miss], source_file="outcome_poller.py",
                                insert_context="finalize",
                                alert_time=alert_time_str,
                            )
                            _upd = {k: v for k, v in _upd.items() if k != _miss}
                        else:
                            log.warning("Finalise schema error (unrecognised) for %s: %s",
                                        token_address[:8], _upd_e)
                            break
                    else:
                        raise

            log.info("Finalised %s | peak=%.1f%% at %s | partial=%s",
                     token_address[:12], peak_pct or 0, peak_label, data_partial)

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

    def _sync_v7_traded(self):
        """
        Cross-reference research_tokens with the memecoin journal to flag
        tokens that v7 actually traded (v7_traded=True).
        Reads logs/memecoin_journal.csv and logs/memecoin_social_journal.csv.
        Safe to call repeatedly — only updates rows where v7_traded is still False.
        """
        if not self._sb:
            return
        import csv
        from pathlib import Path

        journal_paths = [
            Path(__file__).parent.parent / "logs" / "memecoin_journal.csv",
            Path(__file__).parent.parent / "logs" / "memecoin_social_journal.csv",
        ]
        addresses: set = set()
        for path in journal_paths:
            if not path.exists():
                continue
            try:
                with open(path, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Try common column names for the token address
                        addr = (
                            row.get("token_address")
                            or row.get("address")
                            or row.get("mint")
                            or row.get("token")
                            or ""
                        ).strip()
                        if addr:
                            addresses.add(addr)
            except Exception as e:
                log.debug("v7_traded sync: could not read %s: %s", path.name, e)

        if not addresses:
            log.debug("v7_traded sync: no journal addresses found")
            return

        log.info("v7_traded sync: flagging up to %d traded tokens in research_tokens", len(addresses))
        updated = 0
        # Supabase .in_() filter — send in batches of 200 to stay within URL length limits
        addr_list = list(addresses)
        for i in range(0, len(addr_list), 200):
            batch = addr_list[i:i + 200]
            try:
                self._sb.table("research_tokens") \
                    .update({"v7_traded": True}) \
                    .in_("token_address", batch) \
                    .eq("v7_traded", False) \
                    .execute()
                updated += len(batch)
            except Exception as e:
                log.debug("v7_traded sync batch error: %s", e)
        log.info("v7_traded sync: done (%d addresses processed)", updated)

    def _run(self):
        self._init_supabase()
        self._rebuild_from_db()
        self._backfill_old_tokens()   # one-time: close tokens past their poll window
        self._sync_v7_traded()        # cross-reference with memecoin journal

        _last_v7_sync = time.time()

        while True:
            self._wake.clear()
            with self._lock:
                next_fire = self._heap[0][0] if self._heap else None

            if next_fire is None:
                # Nothing scheduled — wait up to 60s for new items
                self._wake.wait(timeout=60)
                # Hourly v7_traded sync
                if time.time() - _last_v7_sync > 3600:
                    self._sync_v7_traded()
                    _last_v7_sync = time.time()
                continue

            sleep_s = next_fire - time.time()
            if sleep_s > 0:
                # Wake early if new item scheduled before next_fire
                self._wake.wait(timeout=sleep_s)
                # Hourly v7_traded sync (check while sleeping too)
                if time.time() - _last_v7_sync > 3600:
                    self._sync_v7_traded()
                    _last_v7_sync = time.time()
                continue

            # Fire due polls
            now = time.time()
            with self._lock:
                due = []
                while self._heap and self._heap[0][0] <= now:
                    due.append(heapq.heappop(self._heap))

            for fire_at, token_address, label, chain, alert_time_iso in due:
                late = (now - fire_at) > 120   # >2 min late = restart-recovered
                try:
                    self._poll(token_address, label, chain, late, alert_time_iso)
                except Exception as e:
                    log.error("Poll error %s/%s: %s", token_address[:8], label, e)
