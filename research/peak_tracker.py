"""
PeakTracker — subscribes to PumpPortal token-trade stream for newly alerted
tokens and records:
  • The highest price seen in the first TICK_PEAK_WINDOW_S seconds (15 min).
  • Every trade tick to logs/research_paths/YYYY-MM-DD/<mint>.csv for path analysis.

Columns written to Supabase (must exist):
  price_peak_3m      FLOAT  — max USD price seen in window
  pct_change_peak_3m FLOAT  — % above entry price at alert time
  t_peak_3m_s        INT    — seconds after alert when peak occurred
  path_file          TEXT   — relative path of the per-token trade CSV

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
    RESEARCH_PATHS_DIR,
    PATH_DEADMAN_MIN_FILES,
    PATH_SUB_SAMPLE_INTERVAL,
)
from research.spool.writer import spool_dropped_field

log = logging.getLogger(__name__)

_SOL_MINT = "So11111111111111111111111111111111111111112"
_PEAK_COLS = ("price_peak_3m", "pct_change_peak_3m", "t_peak_3m_s")
_CSV_HEADER = ["ts_ms", "price_usd", "side", "sol_amount", "vsol"]


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
    ):
        """
        Called from tracker thread after a successful INSERT.
        Adds the token to the 15-min tick-peak tracking window.
        """
        with self._lock:
            if token_address in self._tracked:
                return
            ep = entry_price or 0.0
            self._tracked[token_address] = {
                "entry_price": ep,
                "max_price":   ep,
                "max_ts":      alert_time.timestamp(),
                "alert_ts":    alert_time.timestamp(),
                "expiry":      time.time() + TICK_PEAK_WINDOW_S,
                "done":        False,
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
        """Derive USD price from bonding-curve reserves."""
        vsol = float(msg.get("vSolInBondingCurve") or 0)
        vtok = float(msg.get("vTokensInBondingCurve") or 0)
        if vsol > 0 and vtok > 0:
            return (vsol / vtok) * self._sol_price
        return None

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
                "sub_p95=%d sub_peak=%d",
                day_str,
                scheduled,
                self._path_files_today,
                p95,
                max(samples),
            )
            if max(samples) >= 50:
                log.warning(
                    "PeakTracker: concurrent subscriptions hit cap (%d) on %s",
                    max(samples), day_str,
                )
        else:
            log.info("PeakTracker DAY REPORT %s | no sub samples", day_str)

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

    # ── Async loops ───────────────────────────────────────────────────────────

    async def _ws_loop(self):
        """
        Persistent WebSocket to PumpPortal.
        Subscribes to subscribeTokenTrade for each tracked token.
        Reconnects on any error.
        """
        while True:
            try:
                import websockets as _ws_lib
                async with _ws_lib.connect(
                    PP_WS_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    log.info("PeakTracker: PP WebSocket connected")

                    # Re-subscribe all live tokens after reconnect
                    with self._lock:
                        live_addrs = [a for a, s in self._tracked.items() if not s["done"]]
                    for addr in live_addrs:
                        await ws.send(json.dumps({
                            "action": "subscribeTokenTrade",
                            "tokenAddress": addr,
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
                                price = self._price_from_msg(msg)
                                now = time.time()
                                with self._lock:
                                    st = self._tracked.get(mint)
                                    if st and not st["done"] and now < st["expiry"]:
                                        if price is not None and price > st["max_price"]:
                                            st["max_price"] = price
                                            st["max_ts"]    = now
                                # Write tick to CSV (outside lock, asyncio thread only)
                                if price is not None:
                                    csv_entry = self._csv_files.get(mint)
                                    if csv_entry:
                                        side       = msg.get("txType", "")
                                        sol_amount = float(msg.get("solAmount") or 0)
                                        vsol       = float(msg.get("vSolInBondingCurve") or 0)
                                        ts_ms      = int(now * 1000)
                                        try:
                                            csv_entry["writer"].writerow(
                                                [ts_ms, round(price, 12), side, sol_amount, vsol]
                                            )
                                        except Exception:
                                            pass
                            except Exception:
                                pass

                    async def _drain_pending():
                        """Subscribe new tokens as they arrive from schedule_token()."""
                        loop = asyncio.get_event_loop()
                        while True:
                            await asyncio.sleep(0.3)
                            with self._pending_lock:
                                new = list(self._pending)
                                self._pending.clear()
                            for addr in new:
                                try:
                                    await ws.send(json.dumps({
                                        "action": "subscribeTokenTrade",
                                        "tokenAddress": addr,
                                    }))
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
                log.warning("PeakTracker WS: %s — reconnect in 3s", e)
                await asyncio.sleep(3)

    async def _finalise_loop(self):
        """Every 10s: write peaks for expired tokens, sample sub counts, purge state."""
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

            # Collect expired tracking windows
            expired = []
            with self._lock:
                for addr, st in list(self._tracked.items()):
                    if not st["done"] and now >= st["expiry"]:
                        st["done"] = True
                        expired.append((addr, dict(st)))
                # Purge done entries older than 1 h
                old = [a for a, s in self._tracked.items()
                       if s["done"] and s["expiry"] < now - 3600]
                for a in old:
                    del self._tracked[a]

            # Close CSVs and write peaks for expired tokens
            for addr, st in expired:
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
                self._today_date = current_date

    # ── Supabase write ────────────────────────────────────────────────────────

    def _write_peak(self, addr: str, st: dict):
        if not self._sb:
            return
        entry    = st["entry_price"]
        peak     = st["max_price"]
        alert_ts = st["alert_ts"]
        max_ts   = st["max_ts"]

        pct_peak = ((peak / entry - 1) * 100) if (entry > 0 and peak > entry) else None
        t_peak_s = int(max_ts - alert_ts)      if (peak > entry) else None

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
                log.info("PeakTracker %s | tick_peak=%.2f%% at T+%ds",
                         addr[:12], pct_peak or 0, t_peak_s or 0)
                return
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
                        return
                else:
                    log.warning("PeakTracker write error for %s: %s", addr[:8], e)
                    return

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
