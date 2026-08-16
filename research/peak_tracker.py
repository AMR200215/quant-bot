"""
PeakTracker — subscribes to PumpPortal token-trade stream for newly alerted
tokens and records:
  • The highest price seen in a tiered watch window (15 min base, up to 60 min).
  • Every trade tick to logs/research_paths/YYYY-MM-DD/<mint>.csv for path analysis.

Columns written to Supabase (must exist):
  price_peak_3m      FLOAT  — max USD price seen in window
  pct_change_peak_3m FLOAT  — % above entry price at alert time
  t_peak_3m_s        INT    — seconds after alert when peak occurred
  path_file          TEXT   — relative path of the per-token trade CSV

RF3 tiered-window columns (separate update dict, PGRST204-safe):
  path_extension_count  INT
  path_stop_reason      TEXT
  path_watch_duration_s INT
  path_valid_tick_count INT

CSV path columns: ts_ms, price_usd, side, sol_amount, vsol

Daily rotation: yesterday's directory is gzipped (file-by-file) at UTC midnight.
Deadman: if <PATH_DEADMAN_MIN_FILES path files were created today and ≥20 tokens
were scheduled, a Telegram alert fires (scanner may be broken).

Standalone — no memecoin/ imports.
"""

import asyncio
import csv
import gzip
import json
import logging
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import quantiles
from typing import Optional

import requests

from research.config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    TICK_PEAK_WINDOW_S,
    PP_WS_URL,
    PUMPPORTAL_API_KEY,
    RESEARCH_PATHS_DIR,
    PATH_DEADMAN_MIN_FILES,
    PATH_SUB_SAMPLE_INTERVAL,
    PP_DAILY_MSG_BUDGET,
)
from research.spool.writer import spool_dropped_field
from research.path_schema import PATH_HEADER as _CSV_HEADER, PATH_SCHEMA_VERSION as _SCHEMA_VER

log = logging.getLogger(__name__)

_SOL_MINT = "So11111111111111111111111111111111111111112"
_PEAK_COLS = ("price_peak_3m", "pct_change_peak_3m", "t_peak_3m_s")

# ── RF3 tiered-window constants ───────────────────────────────────────────────
BASE_WINDOW_S       = 900   # 15 min
EXTENSION_INCREMENT_S = 900   # 15 min per extension
HARD_CAP_S          = 3600  # 60 min absolute maximum
_MAX_EXTENSIONS     = (HARD_CAP_S - BASE_WINDOW_S) // EXTENSION_INCREMENT_S  # = 3

# Extension condition: last valid tick must be within this many seconds of now
_RECENT_TICK_WINDOW_S = 180  # 3 minutes


def _should_extend(
    now: float,
    expiry: float,
    extension_count: int,
    last_tick_ts: float,
    last_valid_price: float,
    session_peak_price: float,
) -> bool:
    """
    Pure function — can be tested without asyncio.

    Returns True iff both extension conditions are satisfied AND the hard cap
    has not been reached:
      1. A valid price tick occurred within the last _RECENT_TICK_WINDOW_S seconds.
      2. last_valid_price >= 0.50 * session_peak_price.

    Missing data (last_tick_ts == 0 or last_valid_price == 0) counts as NOT active.
    """
    if extension_count >= _MAX_EXTENSIONS:
        return False
    # Condition 1: recent tick
    if last_tick_ts <= 0 or (now - last_tick_ts) > _RECENT_TICK_WINDOW_S:
        return False
    # Condition 2: price at least 50% of session peak
    if session_peak_price <= 0 or last_valid_price <= 0:
        return False
    if last_valid_price < 0.50 * session_peak_price:
        return False
    return True


# ── V8-FD P16-3: budget-paced admission controller ────────────────────────────
# Replaces "spend until PP_DAILY_MSG_BUDGET is hit, then silently and
# permanently drop every later token for the rest of the UTC day" (the
# confirmed root cause of P15-5's time-of-day selection bias) with a
# pacing scheme that spreads admission chances evenly across all 24 UTC
# hours. A naturalistic research sample must not be concentrated in
# whichever hours happen to fire first each day.
HOURS_PER_DAY = 24


def _admission_probability(
    messages_used_this_hour: float,
    hourly_budget: float,
    messages_used_today: float,
    daily_budget: float,
) -> float:
    """
    Pure function — testable without asyncio, no live state.

    Inputs are ONLY pacing/budget signals (how much of this hour's and
    today's message allowance has been consumed) — deliberately no
    token identity, no progress_at_signal, no V7/V8 pass state, no
    outcome data of any kind (P16-5/P16-6: naturalistic collection must
    stay independent of the eventual strategy label, and admission must
    be decidable before any outcome exists).

    Returns a probability in [0, 1], not an admit/reject decision itself
    — the caller draws its own random number so the probability that
    was actually used is always available to record (P16-4: inclusion-
    probability provenance for future inverse-probability weighting).

    Semantics:
      - Absolute daily ceiling (daily_budget) is a hard stop: 0.0 once
        reached, regardless of hourly pace. This is the real, approved
        cost ceiling (P16-2) — pacing must never cause it to be exceeded
        by an unbounded amount.
      - Within the daily ceiling, each UTC hour gets an equal share of
        the budget (hourly_budget = daily_budget / 24 — deterministic,
        no time-of-day preference of its own). Under that hour's pace:
        admit freely (1.0).
      - Over that hour's pace: probability decays smoothly as
        hourly_budget / messages_used_this_hour rather than cliffing to
        0 — every hour keeps SOME admission chance even under heavy
        load, which is the direct fix for the old behavior. At 2x pace,
        ~50% chance; at 4x, ~25%; approaches but never quite reaches 0
        for any finite overage (the daily hard stop is what actually
        reaches 0).
    """
    if daily_budget <= 0 or messages_used_today >= daily_budget:
        return 0.0
    if hourly_budget <= 0:
        return 0.0
    if messages_used_this_hour < hourly_budget:
        return 1.0
    return max(0.0, min(1.0, hourly_budget / messages_used_this_hour))


class PeakTracker:
    """
    Runs an asyncio loop in its own daemon thread.
    schedule_token() is thread-safe and can be called from the tracker thread.
    """

    def __init__(self):
        self._tracked: dict      = {}   # addr → state dict
        self._lock               = threading.Lock()
        self._pending: list      = []   # addrs awaiting first WS subscription
        self._pending_lock       = threading.Lock()
        self._sb                 = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._sol_price: float   = 175.0

        # PC1 — path persistence
        # All _csv_* accessed only from the asyncio event-loop thread
        self._csv_files: dict    = {}   # addr → {file, writer, path, path_str}

        # Daily stats (reset at UTC midnight)
        self._today_date: str    = ""
        self._tokens_scheduled_today: int = 0
        self._path_files_today: int  = 0
        # K2b: real ticks written today (distinct from path_files_today --
        # a file can exist header-only with zero real ticks, which is
        # exactly the failure mode N4(c) Finding 2 root-caused and file
        # count alone never caught).
        self._ticks_today: int = 0

        # PumpPortal message budget (metered: 0.01 SOL / 10k messages as of 2026-08).
        # New subscriptions pause once hit; already-subscribed tokens finish their
        # current window naturally (max ~15min residual) rather than being force-
        # unsubscribed mid-flight.
        self._pp_messages_today: int = 0
        self._pp_budget_alerted: bool = False
        # V8-FD P15-5/P15-7: tokens whose scheduling was dropped outright
        # because the daily PP message budget was already exhausted --
        # previously untracked, see the _drain_pending() comment.
        self._budget_dropped_today: int = 0

        # V8-FD P16-3/P16-5: hour-paced admission + hourly funnel stats.
        # _hourly_stats persists for the whole UTC day (reset at daily
        # rollover, same as the other _today counters above) -- only
        # _messages_this_hour resets on the hour. Accessed from both the
        # tracker thread (schedule_token, read-only path_eligible count
        # via the admission log, not this dict) and the asyncio loop
        # thread (_recv/_drain_pending/_write_peak) -- kept to simple
        # dict/int mutations under the GIL, same non-locking assumption
        # the pre-existing _tokens_scheduled_today counter already makes.
        self._current_hour: int = -1              # forces first-touch init
        self._messages_this_hour: int = 0
        self._hourly_stats: dict[int, dict] = {}   # hour(0-23) -> counters

        # Concurrent-subscription sampling for p95 report
        self._sub_samples: list  = []   # list of int counts
        self._last_sub_sample: float = 0.0

    def start(self):
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="research-peak"
        )
        self._thread.start()
        log.info("PeakTracker thread started")

    def schedule_token(
        self,
        token_address: str,
        alert_time: datetime,
        entry_price: Optional[float],
        research_event_id: str = "",
        event_id: str = "",
    ):
        """
        Called from tracker thread after a successful INSERT.
        Adds the token to the tiered tick-peak tracking window.

        research_event_id: UUID of the research_tokens row (RF5 — empty string if unknown).
        event_id: deterministic event ID (RF5 — empty string if unknown).
        Both default to "" so existing callers don't break.
        """
        with self._lock:
            if token_address in self._tracked:
                return
            ep = entry_price or 0.0
            now = time.time()
            alert_ts = alert_time.timestamp()
            self._tracked[token_address] = {
                # Core price tracking
                "entry_price":          ep,
                "max_price":            ep,
                "max_ts":               alert_ts,
                "alert_ts":             alert_ts,
                # RF5 path schema IDs
                "research_event_id":    research_event_id,
                "event_id":             event_id,
                # RF3 tiered-window fields
                "expiry":               now + BASE_WINDOW_S,
                "base_expiry":          now + BASE_WINDOW_S,
                "done":                 False,
                "extension_count":      0,
                "last_tick_ts":         0.0,
                "last_valid_price":     0.0,
                "stop_reason":          None,
                "valid_tick_count":     0,
                "disconnection_periods": [],
                "ws_connected":         True,
                # V8-FD P16-5: set once the admission decision is made
                # (_drain_pending) -- attributes this token's eventual
                # tick/usable-path stats to the UTC hour it was SAMPLED
                # in, not whichever hour it happens to finalise in (a
                # session can run up to 60 minutes).
                "admission_hour":       None,
            }
        with self._pending_lock:
            self._pending.append(token_address)
        # bump daily counter (thread-safe, just int assignment — GIL protects)
        self._tokens_scheduled_today += 1
        # Signal the asyncio loop to subscribe this token immediately
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(lambda: None)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _init_supabase(self):
        try:
            from supabase import create_client
            self._sb = create_client(SUPABASE_URL, SUPABASE_KEY)
            log.info("PeakTracker: Supabase client initialised")
        except Exception as e:
            log.error("PeakTracker: Supabase init failed: %s", e)

    def _refresh_sol_price(self):
        try:
            r = requests.get(
                f"https://api.jup.ag/price/v2?ids={_SOL_MINT}", timeout=5
            )
            if r.status_code == 200:
                entry = (r.json().get("data") or {}).get(_SOL_MINT)
                if entry:
                    self._sol_price = float(entry.get("price") or self._sol_price)
        except Exception:
            pass

    def _price_from_msg(self, msg: dict) -> Optional[float]:
        """
        Derive USD price from bonding-curve reserves, falling back to
        per-trade amounts for graduated/pump-amm tokens (no vSol/vTokens
        on those messages — same gap found and fixed in
        memecoin/pumpportal_monitor.py's _compute_price on 2026-08-04).
        tokenAmount/solAmount arrive already in human-readable UI units —
        do NOT apply the /1e6 that vTokensInBondingCurve needs.
        """
        vsol = float(msg.get("vSolInBondingCurve") or 0)
        vtok = float(msg.get("vTokensInBondingCurve") or 0)
        if vsol > 0 and vtok > 0:
            return (vsol / (vtok / 1e6)) * self._sol_price
        sol_amt   = msg.get("solAmount")
        token_amt = msg.get("tokenAmount")
        if sol_amt and token_amt and float(token_amt) > 0:
            return (float(sol_amt) / float(token_amt)) * self._sol_price
        return None

    # ── V8-FD P16-3/P16-4/P16-5: admission pacing + hourly stats ──────────────

    def _hour_bucket(self, hour: int) -> dict:
        """Lazily initialise and return this UTC hour's stat bucket.
        Persists for the whole day (reset only at daily rollover)."""
        b = self._hourly_stats.get(hour)
        if b is None:
            b = {
                "path_eligible": 0, "path_admitted": 0, "subscriptions_started": 0,
                "ticks_ge1": 0, "ticks_ge2": 0, "usable_paths": 0, "budget_messages": 0,
            }
            self._hourly_stats[hour] = b
        return b

    def _maybe_roll_hour(self, now: float) -> int:
        """Resets the hourly message counter on a real UTC hour change.
        Does NOT touch _hourly_stats (that's daily-scoped). Returns the
        current hour. Called from the asyncio thread only."""
        hour = datetime.fromtimestamp(now, tz=timezone.utc).hour
        if hour != self._current_hour:
            self._current_hour = hour
            self._messages_this_hour = 0
        return hour

    def _write_admission_log(self, entry: dict) -> None:
        """V8-FD P16-4: one line per admission decision. Append-only,
        same spool-file pattern as research/spool/*.jsonl -- never
        blocks/raises into the caller."""
        try:
            out_dir = Path(__file__).parent.parent / "logs" / "research_admission"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "admission_log.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            log.debug("PeakTracker: admission log write failed: %s", e)

    def _open_csv(self, addr: str) -> str:
        """
        Open (or reopen) a per-token CSV in today's research_paths directory.
        Returns the relative path string stored in DB.
        Called from asyncio event-loop thread only.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        dir_ = RESEARCH_PATHS_DIR / today
        dir_.mkdir(parents=True, exist_ok=True)
        path = dir_ / f"{addr}.csv"
        is_new = not path.exists() or path.stat().st_size == 0
        f = open(path, "a", newline="", buffering=1)   # line-buffered
        writer = csv.writer(f)
        if is_new:
            writer.writerow(_CSV_HEADER)
            self._path_files_today += 1
        self._csv_files[addr] = {"file": f, "writer": writer, "path": path}
        # relative path for DB storage
        rel = f"logs/research_paths/{today}/{addr}.csv"
        return rel

    def _close_csv(self, addr: str):
        """Flush and close the CSV for a finished token. Asyncio thread only."""
        entry = self._csv_files.pop(addr, None)
        if entry:
            try:
                entry["file"].flush()
                entry["file"].close()
            except Exception:
                pass

    def _gzip_directory(self, day_str: str):
        """
        Compress every .csv file in logs/research_paths/<day_str>/ in-place
        to .csv.gz. Called in a thread-pool executor (not asyncio thread).
        """
        dir_ = RESEARCH_PATHS_DIR / day_str
        if not dir_.exists():
            return
        compressed = 0
        for csv_path in dir_.glob("*.csv"):
            gz_path = csv_path.with_suffix(".csv.gz")
            try:
                with open(csv_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                csv_path.unlink()
                compressed += 1
            except Exception as e:
                log.warning("PeakTracker gzip %s: %s", csv_path.name, e)
        log.info("PeakTracker: gzipped %d files in %s", compressed, day_str)

    def _check_deadman_and_report(self, day_str: str):
        """
        After daily rollover: log p95 concurrent subscriptions + deadman alert.
        Called in a thread-pool executor.
        """
        samples = list(self._sub_samples)
        scheduled = self._tokens_scheduled_today

        # p95 concurrent subscriptions
        if samples:
            p95 = int(quantiles(samples, n=100)[94]) if len(samples) >= 20 else max(samples)
            log.info(
                "PeakTracker DAY REPORT %s | tokens_scheduled=%d path_files=%d "
                "ticks=%d sub_p95=%d sub_peak=%d pp_messages=%d/%d budget_dropped=%d",
                day_str,
                scheduled,
                self._path_files_today,
                self._ticks_today,
                p95,
                max(samples),
                self._pp_messages_today,
                PP_DAILY_MSG_BUDGET,
                self._budget_dropped_today,
            )
            if max(samples) >= 50:
                log.warning(
                    "PeakTracker: concurrent subscriptions hit cap (%d) on %s",
                    max(samples), day_str,
                )
        else:
            log.info("PeakTracker DAY REPORT %s | no sub samples", day_str)

        # V8-FD P15-7: durable, machine-readable daily status for the
        # watchdog to consume (watchdog/checks/path_collection.py) --
        # the send_alert-based deadman/FAIL below is real and already
        # proven working, but isn't visible to the watchdog's own
        # incident/debounce/digest system. This file is the bridge.
        try:
            self._write_daily_status_json(day_str, scheduled, samples)
        except Exception as e:
            log.debug("PeakTracker: daily status JSON write failed: %s", e)

        # Deadman: if scanner was active (≥20 signals) but paths are scarce
        if scheduled >= 20 and self._path_files_today < PATH_DEADMAN_MIN_FILES:
            msg = (
                f"[PeakTracker DEADMAN] {day_str}: only {self._path_files_today} path files "
                f"created ({scheduled} tokens scheduled). "
                f"trade-path collection may be broken."
            )
            log.error(msg)
            try:
                from app.alerts import send_alert
                send_alert(msg)
            except Exception as al_err:
                log.debug("PeakTracker deadman alert failed: %s", al_err)

        # K2b: hard FAIL, distinct from the file-count deadman above. A file
        # can be created (header written) with zero real ticks ever landing
        # in it -- exactly the N4(c) Finding 2 failure mode (path_files
        # matched tokens_scheduled every day, but every session had
        # ticks=0, and file-count alone never caught it). tracked_tokens>50
        # is a much higher bar than the >=20 deadman above -- this is meant
        # to be unmissable, not a routine warning.
        if scheduled > 50 and self._ticks_today == 0:
            fail_msg = (
                f"[PeakTracker FAIL] {day_str}: {scheduled} tokens tracked, "
                f"{self._path_files_today} path files, but 0 real ticks written "
                f"all day. Forward tick collection is completely dead -- check "
                f"PUMPPORTAL_API_KEY / subscribeTokenTrade rejection, not just "
                f"whether files exist."
            )
            log.critical(fail_msg)
            try:
                from app.alerts import send_alert
                send_alert(fail_msg)
            except Exception as al_err:
                log.debug("PeakTracker FAIL alert failed: %s", al_err)

    def _write_daily_status_json(self, day_str: str, scheduled: int, samples: list) -> None:
        """V8-FD P15-7/P16-5: durable status snapshot for
        watchdog/checks/path_collection.py. Overwrites in place (one file,
        most recent completed day) -- historical detail lives in the
        journalctl DAY REPORT lines already, this is just a machine-
        readable mirror plus the P16 hourly breakdown (P16-5's direct
        answer to "is the sample concentrated in a narrow part of the
        day"), which has no other durable home."""
        path_files = self._path_files_today
        yield_pct = round(100.0 * path_files / scheduled, 1) if scheduled > 0 else None

        hourly = []
        for h in range(HOURS_PER_DAY):
            b = self._hourly_stats.get(h, {})
            eligible = b.get("path_eligible", 0)
            admitted = b.get("path_admitted", 0)
            usable = b.get("usable_paths", 0)
            hourly.append({
                "utc_hour": h,
                "path_eligible": eligible,
                "path_admitted": admitted,
                "subscriptions_started": b.get("subscriptions_started", 0),
                "ticks_ge1": b.get("ticks_ge1", 0),
                "ticks_ge2": b.get("ticks_ge2", 0),
                "usable_paths": usable,
                "budget_messages": b.get("budget_messages", 0),
                "admission_rate_pct": round(100.0 * admitted / eligible, 1) if eligible else None,
                "usable_path_yield_pct": round(100.0 * usable / admitted, 1) if admitted else None,
            })

        status = {
            "day": day_str,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tokens_scheduled": scheduled,
            "path_files": path_files,
            "ticks": self._ticks_today,
            "pp_messages": self._pp_messages_today,
            "pp_daily_msg_budget": PP_DAILY_MSG_BUDGET,
            "budget_exceeded": self._pp_messages_today >= PP_DAILY_MSG_BUDGET,
            "budget_dropped_tokens": self._budget_dropped_today,
            "path_yield_pct": yield_pct,
            "sub_peak": max(samples) if samples else 0,
            "hourly": hourly,
        }
        out_path = Path(__file__).parent.parent / "logs" / "watchdog" / "path_collection_daily.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(status, indent=2))
        tmp.replace(out_path)

    def _update_path_file_in_db(self, addr: str, rel_path: str):
        """Write path_file to Supabase. Runs in thread-pool executor."""
        if not self._sb:
            return
        try:
            self._sb.table("research_tokens") \
                .update({"path_file": rel_path}) \
                .eq("token_address", addr) \
                .execute()
        except Exception as e:
            e_str = str(e).lower()
            if "pgrst204" in e_str or "schema cache" in e_str or "path_file" in e_str:
                log.debug("PeakTracker: path_file column not yet in DB for %s", addr[:8])
            else:
                log.debug("PeakTracker: path_file update failed for %s: %s", addr[:8], e)

    def _stop_token(self, addr: str, reason: str):
        """
        Immediately mark a token as done with a given stop_reason.
        Called for out-of-band termination events:
          - curve_account_gone (stub — RF1 hook)
          - graduated_and_stream_migrated (stub)
          - no_valid_venue (stub)
          - websocket_failure (after 5 failed reconnect attempts)
          - process_restart (set on _write_peak if stop_reason is still None)
        """
        with self._lock:
            st = self._tracked.get(addr)
            if st is None or st.get("done"):
                return
            st["done"] = True
            st["stop_reason"] = reason
            snapshot = dict(st)

        self._close_csv(addr)
        loop = self._loop
        if loop and not loop.is_closed():
            loop.run_in_executor(None, self._write_peak, addr, snapshot)
        else:
            self._write_peak(addr, snapshot)

    # ── Async loops ───────────────────────────────────────────────────────────

    async def _ws_loop(self):
        """
        Persistent WebSocket to PumpPortal.
        Subscribes to subscribeTokenTrade for each tracked token.
        Reconnects on any error.
        """
        _reconnect_attempts = 0
        while True:
            try:
                import websockets as _ws_lib
                async with _ws_lib.connect(
                    PP_WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    log.info("PeakTracker: PP WebSocket connected (%s)",
                             "keyed" if PUMPPORTAL_API_KEY else "UNKEYED — subscribeTokenTrade will be rejected")
                    _reconnect_attempts = 0

                    # Mark all tracked tokens as ws_connected after reconnect
                    with self._lock:
                        for st in self._tracked.values():
                            if not st["done"] and not st.get("ws_connected"):
                                st["ws_connected"] = True

                    # Re-subscribe all live tokens after reconnect
                    with self._lock:
                        live_addrs = [a for a, s in self._tracked.items() if not s["done"]]
                    if live_addrs:
                        await ws.send(json.dumps({
                            "method": "subscribeTokenTrade",
                            "keys": live_addrs,
                        }))

                    async def _recv():
                        async for raw in ws:
                            try:
                                msg = json.loads(raw)
                                if msg.get("errors"):
                                    continue
                                mint = msg.get("mint")
                                if not mint:
                                    continue
                                self._pp_messages_today += 1
                                _hr = self._maybe_roll_hour(time.time())
                                self._messages_this_hour += 1
                                self._hour_bucket(_hr)["budget_messages"] += 1
                                if (self._pp_messages_today >= PP_DAILY_MSG_BUDGET
                                        and not self._pp_budget_alerted):
                                    self._pp_budget_alerted = True
                                    log.warning(
                                        "PeakTracker: PumpPortal daily message budget "
                                        "(%d) reached — new admissions stop entirely "
                                        "for the rest of the UTC day (hard ceiling; "
                                        "see V8-FD P16-3 hourly pacing for the softer "
                                        "within-day throttling)", PP_DAILY_MSG_BUDGET,
                                    )
                                    try:
                                        from app.alerts import send_alert as _sa
                                        _sa(
                                            f"PeakTracker: PumpPortal daily message budget "
                                            f"({PP_DAILY_MSG_BUDGET}) reached — new research "
                                            f"admissions stopped until UTC rollover."
                                        )
                                    except Exception:
                                        pass
                                price = self._price_from_msg(msg)
                                now = time.time()
                                with self._lock:
                                    st = self._tracked.get(mint)
                                    if st and not st["done"] and now < st["expiry"]:
                                        if price is not None:
                                            # RF3: update tick tracking fields
                                            st["last_tick_ts"]     = now
                                            st["last_valid_price"] = price
                                            st["valid_tick_count"] = st.get("valid_tick_count", 0) + 1
                                            if price > st["max_price"]:
                                                st["max_price"] = price
                                                st["max_ts"]    = now
                                # Write tick to CSV — RF5 canonical row (outside lock, asyncio thread only)
                                if price is not None:
                                    csv_entry = self._csv_files.get(mint)
                                    if csv_entry:
                                        side       = msg.get("txType", "") or "unknown"
                                        sol_amount = float(msg.get("solAmount") or 0)
                                        vsol       = float(msg.get("vSolInBondingCurve") or 0)
                                        ts_ms      = int(now * 1000)
                                        price_sol  = round(price / self._sol_price, 12) if self._sol_price > 0 else 0.0
                                        # Retrieve IDs from state (set in schedule_token)
                                        with self._lock:
                                            _st = self._tracked.get(mint, {})
                                            _rev_id = _st.get("research_event_id", "")
                                            _ev_id  = _st.get("event_id", "")
                                        # Canonical RF5 row — column order matches PATH_HEADER
                                        trader_pk = msg.get("traderPublicKey", "")  # N7(a)
                                        try:
                                            csv_entry["writer"].writerow([
                                                _SCHEMA_VER,          # schema_version
                                                _rev_id,              # research_event_id
                                                _ev_id,               # event_id
                                                ts_ms,                # ts_ms
                                                round(price, 12),     # price_usd
                                                price_sol,            # price_sol
                                                side,                 # side
                                                0,                    # token_amount (not from PP)
                                                sol_amount,           # sol_amount
                                                vsol,                 # vsol
                                                "live_pp",            # source
                                                "CURVE_ACTIVE",       # venue_state
                                                "false",              # backfilled
                                                "ok",                 # data_status
                                                trader_pk,            # trader_pk (N7a)
                                            ])
                                            self._ticks_today += 1
                                        except Exception:
                                            pass
                            except Exception:
                                pass

                    async def _drain_pending():
                        """Subscribe new tokens as they arrive from schedule_token().

                        V8-FD P16-3: admission is now hour-paced rather than
                        first-come-until-cap. Every pending token gets an
                        admission decision AND a logged record (P16-4) --
                        "budget pressure" now means a lower admission
                        PROBABILITY, sampled rather than a hard drop, except
                        at the absolute daily ceiling (still a real 0%, since
                        that ceiling is the actual approved cost bound)."""
                        import random as _random
                        loop = asyncio.get_event_loop()
                        while True:
                            await asyncio.sleep(0.3)
                            with self._pending_lock:
                                new = list(self._pending)
                                self._pending.clear()
                            if not new:
                                continue

                            now_ts = time.time()
                            hour = self._maybe_roll_hour(now_ts)
                            hourly_budget = PP_DAILY_MSG_BUDGET / HOURS_PER_DAY
                            bucket = self._hour_bucket(hour)

                            admitted, rejected = [], []
                            for addr in new:
                                bucket["path_eligible"] += 1
                                prob = _admission_probability(
                                    messages_used_this_hour=self._messages_this_hour,
                                    hourly_budget=hourly_budget,
                                    messages_used_today=self._pp_messages_today,
                                    daily_budget=PP_DAILY_MSG_BUDGET,
                                )
                                draw = _random.random()
                                is_admitted = draw < prob
                                if self._pp_messages_today >= PP_DAILY_MSG_BUDGET:
                                    reason = "daily_cap_hard_stop"
                                elif prob >= 1.0:
                                    reason = "under_hourly_pace"
                                elif is_admitted:
                                    reason = "sampled_admit"
                                else:
                                    reason = "sampled_reject"
                                self._write_admission_log({
                                    "ts": now_ts, "token_address": addr, "utc_hour": hour,
                                    "path_eligible": True, "path_admitted": is_admitted,
                                    "path_sampling_probability": round(prob, 6),
                                    "admission_reason": reason,
                                    "budget_used": self._pp_messages_today,
                                    "budget_remaining": max(0, PP_DAILY_MSG_BUDGET - self._pp_messages_today),
                                })
                                if is_admitted:
                                    bucket["path_admitted"] += 1
                                    with self._lock:
                                        st = self._tracked.get(addr)
                                        if st is not None:
                                            st["admission_hour"] = hour
                                    admitted.append(addr)
                                else:
                                    rejected.append(addr)

                            # V8-FD P16-3: rejected tokens are a recorded
                            # SAMPLING decision (probability logged above),
                            # not the old silent-and-permanent drop -- but
                            # they still don't get subscribed this cycle.
                            # Distinct counter from the pre-P16 hard-drop
                            # count, which only fires at the absolute daily
                            # ceiling now.
                            if rejected:
                                self._budget_dropped_today += sum(
                                    1 for a in rejected
                                    if self._pp_messages_today >= PP_DAILY_MSG_BUDGET
                                )

                            for addr in admitted:
                                try:
                                    await ws.send(json.dumps({
                                        "method": "subscribeTokenTrade",
                                        "keys": [addr],
                                    }))
                                    bucket["subscriptions_started"] += 1
                                    # Open CSV (asyncio thread) then update DB in executor
                                    rel_path = self._open_csv(addr)
                                    await loop.run_in_executor(
                                        None, self._update_path_file_in_db, addr, rel_path
                                    )
                                except Exception:
                                    # WS probably closed — re-queue for next connect
                                    with self._pending_lock:
                                        self._pending.insert(0, addr)
                                    return

                    await asyncio.gather(_recv(), _drain_pending())

            except Exception as e:
                _reconnect_attempts += 1
                log.warning("PeakTracker WS: %s — reconnect in 3s (attempt %d)",
                            e, _reconnect_attempts)
                # After 5 failed reconnects, mark ws_connected=False on all live tokens
                if _reconnect_attempts >= 5:
                    with self._lock:
                        for addr, st in self._tracked.items():
                            if not st["done"] and st.get("ws_connected", True):
                                st["ws_connected"] = False
                                log.warning(
                                    "PeakTracker: marking %s websocket_failure after 5 reconnect attempts",
                                    addr[:8],
                                )
                await asyncio.sleep(3)

    async def _finalise_loop(self):
        """Every 10s: check extension conditions, write peaks for expired tokens, purge state."""
        _last_sol_refresh = 0.0
        loop = asyncio.get_event_loop()

        while True:
            await asyncio.sleep(10)
            now = time.time()

            # Refresh SOL/USD every 60s
            if now - _last_sol_refresh > 60:
                await loop.run_in_executor(None, self._refresh_sol_price)
                _last_sol_refresh = now

            # Sample concurrent subscription count for p95
            if now - self._last_sub_sample >= PATH_SUB_SAMPLE_INTERVAL:
                with self._lock:
                    active = sum(1 for s in self._tracked.values() if not s["done"])
                self._sub_samples.append(active)
                self._last_sub_sample = now

            # Collect tokens whose current expiry has been reached
            to_extend = []    # (addr, snapshot) — extension conditions met
            to_finalise = []  # (addr, snapshot) — done, write peak

            with self._lock:
                for addr, st in list(self._tracked.items()):
                    if st["done"] or now < st["expiry"]:
                        continue

                    # Expiry reached — check extension conditions
                    should_ext = _should_extend(
                        now=now,
                        expiry=st["expiry"],
                        extension_count=st["extension_count"],
                        last_tick_ts=st["last_tick_ts"],
                        last_valid_price=st["last_valid_price"],
                        session_peak_price=st["max_price"],
                    )

                    if should_ext:
                        # Extend the window
                        st["extension_count"] += 1
                        st["expiry"] += EXTENSION_INCREMENT_S
                        new_expiry_min = (st["expiry"] - st["alert_ts"]) / 60
                        log.info(
                            "PeakTracker EXTEND %s | ext=%d | new_expiry=T+%.0fmin",
                            addr[:8], st["extension_count"], new_expiry_min,
                        )
                        # Don't finalise yet — continue collecting
                    else:
                        # Determine stop reason
                        if st["extension_count"] >= _MAX_EXTENSIONS:
                            stop_reason = "hard_cap_reached"
                        elif st["extension_count"] > 0:
                            stop_reason = "extension_condition_failed"
                        else:
                            stop_reason = "base_window_expired"
                        st["stop_reason"] = stop_reason
                        st["done"] = True
                        to_finalise.append((addr, dict(st)))

                # Purge done entries older than 1 h
                old = [a for a, s in self._tracked.items()
                       if s["done"] and s["expiry"] < now - 3600]
                for a in old:
                    del self._tracked[a]

            # Close CSVs and write peaks for expired tokens
            for addr, st in to_finalise:
                self._close_csv(addr)
                await loop.run_in_executor(None, self._write_peak, addr, st)

    async def _rotation_loop(self):
        """
        Runs every 60s. At UTC midnight rollover:
        - gzip yesterday's research_paths directory
        - fire deadman / p95 report
        - reset daily counters
        """
        loop = asyncio.get_event_loop()
        self._today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        while True:
            await asyncio.sleep(60)
            current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if current_date != self._today_date:
                yesterday = self._today_date
                log.info("PeakTracker: date rollover %s → %s", yesterday, current_date)
                # Run blocking work in executor
                await loop.run_in_executor(
                    None, self._check_deadman_and_report, yesterday
                )
                await loop.run_in_executor(
                    None, self._gzip_directory, yesterday
                )
                # Reset daily counters
                self._sub_samples.clear()
                self._tokens_scheduled_today = 0
                self._path_files_today = 0
                self._ticks_today = 0
                self._pp_messages_today = 0
                self._pp_budget_alerted = False
                self._budget_dropped_today = 0
                self._hourly_stats = {}
                self._messages_this_hour = 0
                self._current_hour = -1
                self._today_date = current_date

    # ── Supabase write ────────────────────────────────────────────────────────

    def _write_peak(self, addr: str, st: dict):
        # V8-FD P16-5: attribute this token's terminal tick/usable-path
        # stats to the UTC hour it was ADMITTED in (not whichever hour
        # finalisation happens to land in -- a session can run up to 60
        # minutes past admission). Pure local bookkeeping, so this runs
        # regardless of Supabase connectivity, unlike the write below.
        # Known limitation, same class as the pre-existing _ticks_today
        # daily reset: a token admitted just before UTC midnight and
        # finalising after rollover has its hourly attribution lost along
        # with that day's _hourly_stats -- not a new regression, the
        # existing daily counters already have this exact edge case.
        admission_hour = st.get("admission_hour")
        if admission_hour is not None:
            bucket = self._hour_bucket(admission_hour)
            ticks = st.get("valid_tick_count", 0)
            if ticks >= 1:
                bucket["ticks_ge1"] += 1
            if ticks >= 2:
                bucket["ticks_ge2"] += 1
                bucket["usable_paths"] += 1   # P15-7's own bar: a header-only or single-tick file doesn't count

        if not self._sb:
            return

        # If stop_reason was never set (e.g. process restart mid-window), label it
        if st.get("stop_reason") is None:
            st["stop_reason"] = "process_restart"

        entry    = st["entry_price"]
        peak     = st["max_price"]
        alert_ts = st["alert_ts"]
        max_ts   = st["max_ts"]

        pct_peak = ((peak / entry - 1) * 100) if (entry > 0 and peak > entry) else None
        t_peak_s = int(max_ts - alert_ts)      if (peak > entry) else None

        # Primary peak update (existing columns)
        update = {
            "price_peak_3m":       round(peak, 12) if peak > 0 else None,
            "pct_change_peak_3m":  round(pct_peak, 2) if pct_peak is not None else None,
            "t_peak_3m_s":         t_peak_s,
        }
        import re as _re
        _alert_time_iso = datetime.fromtimestamp(alert_ts, tz=timezone.utc).isoformat()
        _update = dict(update)
        for _attempt in range(4):
            try:
                self._sb.table("research_tokens") \
                    .update(_update) \
                    .eq("token_address", addr) \
                    .execute()
                log.info("PeakTracker %s | tick_peak=%.2f%% at T+%ds | "
                         "ext=%d stop=%s dur=%ds ticks=%d",
                         addr[:12], pct_peak or 0, t_peak_s or 0,
                         st.get("extension_count", 0),
                         st.get("stop_reason", "?"),
                         int(time.time() - alert_ts),
                         st.get("valid_tick_count", 0))
                break
            except Exception as e:
                e_str = str(e).lower()
                if "pgrst204" in e_str or "schema cache" in e_str:
                    m       = _re.search(r"'(\w+)'\s+column", str(e))
                    missing = m.group(1) if m else None
                    if missing and missing in _update:
                        spool_dropped_field(
                            token_address=addr, symbol="",
                            table="research_tokens", column=missing,
                            value=_update[missing], source_file="peak_tracker.py",
                            insert_context="peak_update",
                            alert_time=_alert_time_iso,
                        )
                        _update = {k: v for k, v in _update.items() if k != missing}
                    else:
                        log.warning("PeakTracker schema error (unrecognised col) for %s: %s",
                                    addr[:8], e)
                        break
                else:
                    log.warning("PeakTracker write error for %s: %s", addr[:8], e)
                    break

        # RF3 extension metadata — separate update dict with PGRST204-safe retry
        watch_duration_s = int(time.time() - alert_ts)
        rf3_update = {
            "path_extension_count":  st.get("extension_count", 0),
            "path_stop_reason":      st.get("stop_reason"),
            "path_watch_duration_s": watch_duration_s,
            "path_valid_tick_count": st.get("valid_tick_count", 0),
        }
        _rf3 = dict(rf3_update)
        for _attempt in range(4):
            try:
                self._sb.table("research_tokens") \
                    .update(_rf3) \
                    .eq("token_address", addr) \
                    .execute()
                break
            except Exception as e:
                e_str = str(e).lower()
                if "pgrst204" in e_str or "schema cache" in e_str:
                    m       = _re.search(r"'(\w+)'\s+column", str(e))
                    missing = m.group(1) if m else None
                    if missing and missing in _rf3:
                        spool_dropped_field(
                            token_address=addr, symbol="",
                            table="research_tokens", column=missing,
                            value=_rf3[missing], source_file="peak_tracker.py",
                            insert_context="rf3_update",
                            alert_time=_alert_time_iso,
                        )
                        _rf3 = {k: v for k, v in _rf3.items() if k != missing}
                    else:
                        log.debug("PeakTracker RF3 schema error for %s: %s", addr[:8], e)
                        break
                else:
                    log.debug("PeakTracker RF3 update failed for %s: %s", addr[:8], e)
                    break

    # ── Thread entry ──────────────────────────────────────────────────────────

    def _run(self):
        self._init_supabase()
        self._refresh_sol_price()
        RESEARCH_PATHS_DIR.mkdir(parents=True, exist_ok=True)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        while True:
            try:
                loop.run_until_complete(
                    asyncio.gather(
                        self._ws_loop(),
                        self._finalise_loop(),
                        self._rotation_loop(),
                    )
                )
            except Exception as e:
                log.error("PeakTracker crashed: %s — restart in 5s", e)
                time.sleep(5)
