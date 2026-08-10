"""
Backfill trade-path CSVs from on-chain transaction history.

Fetches on-chain trade activity for:
  • Up to --winners  tokens with pct_change_peak ≥ --win-thresh  (default 200 / +50%)
  • Up to --losers   tokens with pct_change_peak <  0             (default 200)

Two parse modes (auto-detected at startup):
  helius  — Helius enhanced API (api.helius.xyz/v0/transactions); richer labels;
             1 credit/tx.  Used when Helius quota is available.
  std_rpc — Standard JSON-RPC getTransaction via any public RPC
             (https://api.mainnet-beta.solana.com by default).  No Helius quota
             consumed; fetches run in parallel (3 workers).  Auto-selected when
             Helius returns "max usage reached" (429).

Writes per-token CSVs to logs/research_paths/backfill/<mint>.csv.gz in the same
format as live PeakTracker paths.  Updates research_tokens.path_file for each file.

Hard Helius credit cap (--credit-cap, default 50000):
  Only applies in helius mode.  Ignored in std_rpc mode.

Dry-run (--dry-run):
  Prints estimated credit cost and detected parse mode, then exits.

Run:
    python -m research.backfill_paths [--dry-run] [--winners N] [--losers N]
    python -m research.backfill_paths --credit-cap 20000 --win-thresh 100
    python -m research.backfill_paths --parse-mode std_rpc   # force public RPC

CSV columns: ts_ms, price_usd, side, sol_amount, vsol, source, ...
"""

import argparse
import csv
import gzip
import hashlib
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("backfill_paths")

# RF5: use canonical schema header
from research.path_schema import PATH_HEADER as _CSV_HEADER, PATH_SCHEMA_VERSION as _SCHEMA_VER

_SOL_MINT    = "So11111111111111111111111111111111111111112"
_LAMPORTS    = 1_000_000_000
_MAX_SIGS    = 1000   # getSignaturesForAddress limit
_PARSE_BATCH = 100    # Helius enhanced-tx batch size
_RATE_SLEEP      = 0.5   # seconds between tokens (both modes)
_MAX_RETRIES     = 4     # retries on 429 before giving up on a token
_STD_MAX_WORKERS = 5     # K3: was 10 -- reduced after diagnosis showed 10
                          # parallel workers against a single free public
                          # RPC endpoint drove sustained 429s (5 real winner
                          # tokens took >14min instead of the expected
                          # couple minutes)
_STD_TX_CAP      = 300   # max transactions fetched per token in std_rpc mode
_STD_TX_SLEEP    = 0.05  # seconds between individual getTransaction calls (per worker)

_PUBLIC_RPC = "https://api.mainnet-beta.solana.com"
# K3: fallback tier for std_rpc mode when the primary free endpoint 429s --
# std_rpc mode's whole point is 0 Helius credits, so the fallback is
# another free public RPC (same one research/curve_oracle.py already
# falls back to), not Helius.
_STD_RPC_FALLBACK = "https://rpc.ankr.com/solana"

# K3 fix (2026-08-09): slot-time estimation constants, replacing blind
# backward pagination for alert_ts-windowed signature fetches. See
# _estimate_slot_for_time docstring for why.
_SLOT_TIME_S                = 0.4
_SLOT_ESTIMATE_TOLERANCE_S  = 30
_SLOT_ESTIMATE_MAX_ITER     = 6
_SLOT_ESTIMATE_BLOCK_SEARCH = 15   # skipped-slot search radius per iteration

# Average tx count used for credit estimation in dry-run
_DRY_RUN_AVG_TXS = 200


def _get_sol_price() -> float:
    """Fetch current SOL/USD from Jupiter price API."""
    try:
        r = requests.get(
            f"https://api.jup.ag/price/v2?ids={_SOL_MINT}", timeout=8
        )
        if r.status_code == 200:
            entry = (r.json().get("data") or {}).get(_SOL_MINT)
            if entry:
                price = float(entry.get("price") or 0)
                if price > 0:
                    log.info("SOL price: $%.2f", price)
                    return price
    except Exception as e:
        log.warning("SOL price fetch failed: %s — using $170.00", e)
    return 170.0


def _load_tokens(sb, winners: int, losers: int, win_thresh: float) -> tuple:
    """
    Returns (winner_list, loser_list) of research_token rows.
    Both lists capped at the requested counts.
    """
    def _page(query):
        rows, offset, batch = [], 0, 1000
        while True:
            chunk = (
                query
                .range(offset, offset + batch - 1)
                .execute()
                .data or []
            )
            rows.extend(chunk)
            if len(chunk) < batch:
                break
            offset += batch
        return rows

    def _base():
        # Supabase Python client mutates the query builder in place on each
        # chained filter call — rebuild from scratch for every query so winner
        # and loser filters don't accumulate on the same object.
        return (
            sb.table("research_tokens")
            .select("id,token_address,symbol,pct_change_peak,alert_time,path_file")
            .eq("outcome_complete", True)
            .eq("chain", "solana")
        )

    winner_rows = _page(
        _base().gte("pct_change_peak", win_thresh).order("alert_time", desc=True)
    )[:winners]

    loser_rows = _page(
        _base().lt("pct_change_peak", 0).order("alert_time", desc=True)
    )[:losers]

    return winner_rows, loser_rows


def _fetch_sigs_page(mint: str, rpc_url: str, before: str | None) -> list | None:
    """One getSignaturesForAddress page (DESC/newest-first). None = gave up after retries.

    K3: switches to _STD_RPC_FALLBACK partway through retries instead of
    hammering the same rate-limited endpoint for the full retry budget --
    a fallback tier that costs 0 Helius credits (matching std_rpc mode's
    whole point), same free-endpoint pattern research/curve_oracle.py uses.
    """
    backoff = 2.0
    switch_at = _MAX_RETRIES // 2
    for attempt in range(_MAX_RETRIES):
        active_url = _STD_RPC_FALLBACK if attempt >= switch_at else rpc_url
        try:
            params = {"limit": _MAX_SIGS, "commitment": "confirmed"}
            if before:
                params["before"] = before
            r = requests.post(
                active_url,
                json={
                    "jsonrpc": "2.0", "id": 1,
                    "method":  "getSignaturesForAddress",
                    "params":  [mint, params],
                },
                timeout=20,
            )
            if r.status_code == 429:
                wait = backoff * (2 ** attempt)
                log.warning("getSignaturesForAddress 429 for %s on %s — sleeping %.1fs (attempt %d/%d)",
                            mint[:8], active_url, wait, attempt + 1, _MAX_RETRIES)
                time.sleep(wait)
                continue
            return r.json().get("result") or []
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                wait = backoff * (2 ** attempt)
                log.warning("getSignaturesForAddress error for %s: %s — retry in %.1fs",
                            mint[:8], e, wait)
                time.sleep(wait)
                continue
            log.warning("getSignaturesForAddress failed for %s after %d attempts: %s",
                        mint[:8], _MAX_RETRIES, e)
            return None
    log.warning("getSignaturesForAddress gave up for %s after %d attempts", mint[:8], _MAX_RETRIES)
    return None


def _rpc_call(rpc_url: str, method: str, params: list):
    """Minimal retried RPC call — used only by slot estimation below. K3:
    same fallback-tier pattern as _fetch_sigs_page."""
    backoff = 1.0
    switch_at = _MAX_RETRIES // 2
    for attempt in range(_MAX_RETRIES):
        active_url = _STD_RPC_FALLBACK if attempt >= switch_at else rpc_url
        try:
            r = requests.post(
                active_url,
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                timeout=15,
            )
            if r.status_code == 429:
                time.sleep(backoff * (2 ** attempt))
                continue
            return r.json().get("result")
        except Exception:
            time.sleep(backoff * (2 ** attempt))
    return None


def _get_block_near(rpc_url: str, slot: int, search: int = _SLOT_ESTIMATE_BLOCK_SEARCH):
    """Find a real, non-skipped block at or after `slot`. Returns (slot, block) or (None, None)."""
    for offset in range(search):
        blk = _rpc_call(rpc_url, "getBlock", [slot + offset, {
            "maxSupportedTransactionVersion": 0,
            "transactionDetails": "signatures",
            "rewards": False,
        }])
        if blk and blk.get("blockTime") and blk.get("signatures"):
            return slot + offset, blk
    return None, None


def _estimate_slot_for_time(rpc_url: str, target_ts: float):
    """
    K3 fix (2026-08-09): iteratively estimate the slot nearest `target_ts`
    via Solana's ~0.4s/slot rate, refined by real getBlock lookups.

    Replaces blind backward pagination via getSignaturesForAddress, which
    was the actual root cause of the backfill's low yield -- diagnosed on
    real data: a genuine +2148% winner token needed 35,000+ signatures
    fetched via the old max_pages-bounded backward walk and STILL landed
    6+ hours short of its own alert time, because high-activity tokens
    accumulate more post-alert signatures than any reasonable page cap can
    walk through. This converges to within _SLOT_ESTIMATE_TOLERANCE_S
    seconds in ~4-6 getBlock calls regardless of how much trading happened
    after the target time, verified against the same token above.

    Returns (slot, signatures_in_that_block) or None if unreachable
    (RPC failure, or too far in the past for skipped-slot search to land
    on a real block within _SLOT_ESTIMATE_MAX_ITER tries).
    """
    cur_slot = _rpc_call(rpc_url, "getSlot", [{"commitment": "confirmed"}])
    if cur_slot is None:
        return None
    now = time.time()
    est_slot = cur_slot - int((now - target_ts) / _SLOT_TIME_S)
    if est_slot < 0:
        return None

    actual_slot, blk = None, None
    for _ in range(_SLOT_ESTIMATE_MAX_ITER):
        actual_slot, blk = _get_block_near(rpc_url, est_slot)
        if blk is None:
            return None
        delta = blk["blockTime"] - target_ts
        if abs(delta) < _SLOT_ESTIMATE_TOLERANCE_S:
            break
        est_slot = actual_slot - int(delta / _SLOT_TIME_S)
    if blk is None:
        return None
    return actual_slot, (blk.get("signatures") or [])


def _fetch_sigs(mint: str, rpc_url: str, alert_ts: float | None = None,
                 window_s: int = 900, max_pages: int = 10) -> list:
    """
    getSignaturesForAddress → list of sig strings (oldest first, no errors).

    Without alert_ts: legacy single-page behavior (most-recent _MAX_SIGS).

    With alert_ts: anchors near alert_ts+window_s via slot-time estimation
    (_estimate_slot_for_time) instead of paginating backward from "now" —
    see that function's docstring for why the old approach was the real
    root cause of low backfill yield (K3). Any real signature works as the
    `before` anchor for getSignaturesForAddress regardless of which
    address it touched (Solana RPC treats `before` as a pure chronological
    cursor) — from there, only a SHORT bounded pagination is needed since
    we're already within seconds of the target window, not chasing it
    across potentially tens of thousands of newer signatures.
    """
    if alert_ts is None:
        result = _fetch_sigs_page(mint, rpc_url, before=None)
        if not result:
            return []
        result.reverse()
        return [s["signature"] for s in result if not s.get("err")]

    window_start = alert_ts
    window_end   = alert_ts + window_s

    anchor = _estimate_slot_for_time(rpc_url, window_end)
    anchor_sig = None
    if anchor is not None:
        _, block_sigs = anchor
        if block_sigs:
            anchor_sig = block_sigs[0]

    all_sigs: list = []
    before: str | None = anchor_sig
    for _page in range(max_pages):
        result = _fetch_sigs_page(mint, rpc_url, before)
        if not result:
            break
        all_sigs.extend(result)
        oldest = result[-1]
        oldest_bt = oldest.get("blockTime")
        before = oldest["signature"]
        if oldest_bt and oldest_bt <= window_start:
            break
        time.sleep(_STD_TX_SLEEP)

    windowed = [
        s for s in all_sigs
        if s.get("blockTime") and window_start <= s["blockTime"] <= window_end
        and not s.get("err")
    ]
    windowed.reverse()   # oldest-first, matching existing contract
    return [s["signature"] for s in windowed]


def _parse_txs(sigs: list, parse_url: str) -> list:
    """
    Batch-parse up to _PARSE_BATCH sigs via Helius enhanced tx API.
    Returns list of parsed tx dicts.
    """
    try:
        r = requests.post(
            parse_url,
            json={"transactions": sigs[:_PARSE_BATCH]},
            timeout=30,
        )
        parsed = r.json()
        return parsed if isinstance(parsed, list) else []
    except Exception as e:
        log.warning("Enhanced TX parse failed (batch %d): %s", len(sigs), e)
        return []


def _extract_rows(parsed_txs: list, mint: str, sol_price: float,
                  research_event_id: str = "") -> list:
    """
    Convert Helius enhanced tx list → canonical RF5 row dicts.

    Price derivation:
      sol_amount = |native balance change of fee payer| in SOL
      token_amount = tokenTransfers for this mint received/sent by fee payer
      price_usd = (sol_amount / token_amount) * sol_price   [per token]
    vsol is not available from history → stored as 0.

    RF5 notes:
      source      = "backfill_helius"
      backfilled  = "true"
      side        = "unknown"  — direction can't be reliably inferred from Helius
                                  enhanced TX for most cases
      venue_state = "UNKNOWN"
      event_id    = deterministic hash of "backfill:{mint}:{ts_ms}"
    """
    import hashlib as _hashlib
    rows = []
    for tx in parsed_txs:
        if not isinstance(tx, dict):
            continue
        if tx.get("type") not in ("SWAP", "UNKNOWN"):
            continue
        ts    = tx.get("timestamp") or tx.get("blockTime")
        if not ts:
            continue
        ts_ms = int(ts) * 1000
        fee_payer = tx.get("feePayer", "")

        # Determine token amount (side stored as "unknown" per RF5 spec)
        token_amount  = 0.0
        for tt in (tx.get("tokenTransfers") or []):
            if tt.get("mint") != mint:
                continue
            raw_amt = float(tt.get("tokenAmount") or 0)
            if tt.get("toUserAccount") == fee_payer or tt.get("fromUserAccount") == fee_payer:
                token_amount = raw_amt
                break
        if token_amount == 0:
            continue

        # SOL amount — absolute native balance change for fee payer
        sol_amount = 0.0
        for ad in (tx.get("accountData") or []):
            if ad.get("account") == fee_payer:
                change     = float(ad.get("nativeBalanceChange") or 0)
                sol_amount = abs(change) / _LAMPORTS
                break
        # Fallback: sum nativeTransfers involving fee payer
        if sol_amount == 0:
            for nt in (tx.get("nativeTransfers") or []):
                if nt.get("fromUserAccount") == fee_payer or nt.get("toUserAccount") == fee_payer:
                    sol_amount += abs(float(nt.get("amount") or 0)) / _LAMPORTS

        price_usd = (sol_amount / token_amount * sol_price) if token_amount > 0 else 0.0
        price_sol = round(price_usd / sol_price, 12) if sol_price > 0 else 0.0

        # Deterministic event_id
        event_id = _hashlib.sha256(f"backfill:{mint}:{ts_ms}".encode()).hexdigest()[:32]

        # Canonical RF5 dict — fields in PATH_HEADER order
        rows.append({
            "schema_version":    str(_SCHEMA_VER),
            "research_event_id": research_event_id,
            "event_id":          event_id,
            "ts_ms":             ts_ms,
            "price_usd":         round(price_usd, 12),
            "price_sol":         price_sol,
            "side":              "unknown",        # RF5: can't reliably infer from Helius
            "token_amount":      round(token_amount, 0),
            "sol_amount":        round(sol_amount, 9),
            "vsol":              0.0,              # not available in history
            "source":            "backfill_helius",
            "venue_state":       "UNKNOWN",
            "backfilled":        "true",
            "data_status":       "ok",
            "trader_pk":         fee_payer,   # N7(a): Helius tx feePayer IS the trader
        })

    # Sort by timestamp ascending
    rows.sort(key=lambda r: r["ts_ms"])
    return rows


# ---------------------------------------------------------------------------
# Standard RPC path (fallback when Helius quota exhausted)
# ---------------------------------------------------------------------------

def _check_helius_quota(rpc_url: str) -> bool:
    """Return True if Helius reports 'max usage reached' (quota exhausted)."""
    try:
        r = requests.post(rpc_url, json={"jsonrpc": "2.0", "id": 1, "method": "getSlot"},
                          timeout=10)
        if r.status_code == 429:
            body = r.text.lower()
            return "max usage" in body or "max_usage" in body
    except Exception:
        pass
    return False


def _fetch_one_tx_std(sig: str, rpc_url: str) -> dict | None:
    """Fetch a single transaction via standard getTransaction RPC. Returns None on error.

    K3 fix: previously gave up after a single 429 with only a fixed 2s
    sleep and no actual retry -- under _STD_MAX_WORKERS-parallel load
    against a public RPC, 429s are routine, not exceptional, so this was
    silently dropping a meaningful fraction of real transactions rather
    than retrying them. Now retries with backoff and a fallback tier,
    matching _fetch_sigs_page's pattern.
    """
    backoff = 1.0
    switch_at = _MAX_RETRIES // 2
    for attempt in range(_MAX_RETRIES):
        active_url = _STD_RPC_FALLBACK if attempt >= switch_at else rpc_url
        try:
            r = requests.post(
                active_url,
                json={
                    "jsonrpc": "2.0", "id": 1,
                    "method": "getTransaction",
                    "params": [sig, {"encoding": "json", "commitment": "confirmed",
                                     "maxSupportedTransactionVersion": 0}],
                },
                timeout=15,
            )
            if r.status_code == 429:
                time.sleep(backoff * (2 ** attempt))
                continue
            data = r.json()
            return data.get("result")
        except Exception:
            time.sleep(backoff * (2 ** attempt))
    return None


_BATCH_FETCH_SIZE   = 100   # JSON-RPC batch request size for getTransaction
_BATCH_FETCH_ROUNDS = 6     # retry rounds for sigs that failed (429) this token
_BATCH_ROUND_SLEEP  = 1.5   # pause between retry rounds


def _batch_get_transactions(sigs: list, rpc_url: str) -> dict:
    """
    One JSON-RPC batch request (all sigs in one HTTP call) for getTransaction.
    Returns {sig: result_dict} for whichever sigs succeeded this round --
    Solana public RPC rate-limits individual items WITHIN a batch response
    (per-item {"error": {"code": 429, ...}}), not just the HTTP call itself,
    so a batch is not "all or nothing" -- typically only a fraction succeed
    per round, but discovering that takes ~2s instead of the many seconds
    a single failed individual call+backoff used to cost.
    """
    payload = [
        {"jsonrpc": "2.0", "id": i, "method": "getTransaction",
         "params": [sig, {"encoding": "json", "commitment": "confirmed",
                          "maxSupportedTransactionVersion": 0}]}
        for i, sig in enumerate(sigs)
    ]
    try:
        r = requests.post(rpc_url, json=payload, timeout=30)
        if r.status_code != 200:
            return {}
        results = r.json()
        if not isinstance(results, list):
            return {}
        ok = {}
        for x in results:
            idx = x.get("id")
            if idx is None or idx >= len(sigs):
                continue
            if x.get("result") is not None:
                ok[sigs[idx]] = x["result"]
        return ok
    except Exception as e:
        log.debug("batch getTransaction failed: %s", e)
        return {}


def _parse_txs_std(sigs: list, rpc_url: str) -> list:
    """
    Fetch up to _STD_TX_CAP transactions via batched getTransaction.
    Returns list of result dicts.

    K3 fix: previously used _STD_MAX_WORKERS individual parallel requests,
    which under real rate-limit pressure (verified: mainnet-beta.solana.com
    sustains roughly 0.5 successful getTransaction/sec regardless of
    request pattern) meant most requests spent their time in per-request
    exponential backoff, fetching as few as 10/300 in 220+ seconds. Batch
    requests get the SAME per-item rate limit, but discover failures in
    ~2s instead of a full backoff cycle, so retry rounds are dramatically
    faster -- same ceiling throughput, much less wasted wall-clock time
    finding out what failed.
    """
    remaining = sigs[:_STD_TX_CAP]
    all_ok: dict = {}
    for round_i in range(_BATCH_FETCH_ROUNDS):
        if not remaining:
            break
        new_ok = {}
        for i in range(0, len(remaining), _BATCH_FETCH_SIZE):
            chunk = remaining[i:i + _BATCH_FETCH_SIZE]
            active_url = _STD_RPC_FALLBACK if round_i >= _BATCH_FETCH_ROUNDS // 2 else rpc_url
            new_ok.update(_batch_get_transactions(chunk, active_url))
        all_ok.update(new_ok)
        remaining = [s for s in remaining if s not in all_ok]
        if remaining:
            time.sleep(_BATCH_ROUND_SLEEP)

    return list(all_ok.values())


def _extract_rows_std(tx_results: list, mint: str, sol_price: float,
                      research_event_id: str = "", curve_pda: str | None = None) -> list:
    """
    Convert standard getTransaction result dicts → canonical RF5 row dicts.

    Price derivation:
      sol_amount = |pre/post native balance change of fee payer| in SOL
      token_amount = |pre/post uiTokenAmount for the mint held by fee payer's ATA|
      price_usd = (sol_amount / token_amount) * sol_price

    source = "backfill_std_rpc".

    K3: vsol IS derivable here (unlike Helius Enhanced Transactions mode,
    which only exposes balance deltas, not absolute balances) -- if the
    bonding-curve PDA is among this tx's touched accounts (static
    accountKeys or versioned-tx loadedAddresses), its postBalance in
    preBalances/postBalances IS the vsol reading at that tick. Only
    populated for pre-graduation trades that actually touch the curve
    account; post-graduation (PumpSwap/DEX) trades correctly get vsol=0,
    same "missing, never coerced" contract as everywhere else in this repo.
    """
    rows = []
    for result in tx_results:
        if not isinstance(result, dict) or result.get("err"):
            continue

        ts = result.get("blockTime")
        if not ts:
            continue
        ts_ms = int(ts) * 1000

        meta = result.get("meta") or {}
        msg = (result.get("transaction") or {}).get("message") or {}
        account_keys = msg.get("accountKeys") or []
        if not account_keys:
            continue

        fee_payer = account_keys[0]
        pre_bals  = meta.get("preBalances") or []
        post_bals = meta.get("postBalances") or []
        sol_amount = 0.0
        if pre_bals and post_bals:
            sol_amount = abs(post_bals[0] - pre_bals[0]) / _LAMPORTS

        # Token amount: largest delta for this mint across all ATAs
        token_amount = 0.0
        pre_tb  = {t["accountIndex"]: t for t in (meta.get("preTokenBalances") or [])
                   if t.get("mint") == mint}
        post_tb = {t["accountIndex"]: t for t in (meta.get("postTokenBalances") or [])
                   if t.get("mint") == mint}
        for idx in set(pre_tb) | set(post_tb):
            pre_ui  = float((pre_tb.get(idx, {}).get("uiTokenAmount") or {}).get("uiAmount") or 0)
            post_ui = float((post_tb.get(idx, {}).get("uiTokenAmount") or {}).get("uiAmount") or 0)
            delta = abs(post_ui - pre_ui)
            if delta > token_amount:
                token_amount = delta

        if token_amount == 0 or sol_amount == 0:
            continue

        price_usd = (sol_amount / token_amount) * sol_price
        price_sol = round(price_usd / sol_price, 12) if sol_price > 0 else 0.0
        event_id  = hashlib.sha256(f"backfill:{mint}:{ts_ms}".encode()).hexdigest()[:32]

        # K3: vsol from curve PDA's postBalance, where the tx touches it.
        vsol = 0.0
        if curve_pda:
            loaded = meta.get("loadedAddresses") or {}
            all_keys = account_keys + loaded.get("writable", []) + loaded.get("readonly", [])
            if curve_pda in all_keys:
                idx = all_keys.index(curve_pda)
                if idx < len(post_bals):
                    vsol = post_bals[idx] / _LAMPORTS

        rows.append({
            "schema_version":    str(_SCHEMA_VER),
            "research_event_id": research_event_id,
            "event_id":          event_id,
            "ts_ms":             ts_ms,
            "price_usd":         round(price_usd, 12),
            "price_sol":         price_sol,
            "side":              "unknown",
            "token_amount":      round(token_amount, 0),
            "sol_amount":        round(sol_amount, 9),
            "vsol":              round(vsol, 9),
            "source":            "backfill_std_rpc",
            "venue_state":       "CURVE_ACTIVE" if vsol > 0 else "UNKNOWN",
            "backfilled":        "true",
            "data_status":       "ok",
            "trader_pk":         fee_payer,
        })

    rows.sort(key=lambda r: r["ts_ms"])
    return rows


# ---------------------------------------------------------------------------
# Token processor — dispatches to helius or std_rpc path
# ---------------------------------------------------------------------------

def _write_csv(mint: str, rows: list, out_dir: Path) -> str:
    """
    Write canonical RF5 rows to out_dir/<mint>.csv.gz.
    rows is a list of dicts with keys matching PATH_HEADER.
    Returns relative path.
    """
    from research.path_schema import PATH_HEADER
    out_dir.mkdir(parents=True, exist_ok=True)
    gz_path = out_dir / f"{mint}.csv.gz"
    with gzip.open(gz_path, "wt", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(PATH_HEADER)
        for row in rows:
            writer.writerow([row.get(col, "") for col in PATH_HEADER])
    # relative path for DB (callers check .csv or .csv.gz)
    return f"logs/research_paths/backfill/{mint}.csv.gz"


def _update_db(sb, row_id: str, path_str: str):
    try:
        sb.table("research_tokens") \
            .update({"path_file": path_str}) \
            .eq("id", row_id) \
            .execute()
    except Exception as e:
        log.debug("path_file update failed for id %s: %s", row_id, e)


def _process_token(tok: dict, rpc_url: str, parse_url: str, sol_price: float,
                   out_dir: Path, sb, credit_budget: list,
                   parse_mode: str = "helius", std_rpc_url: str = _PUBLIC_RPC) -> int:
    """
    Process one token. Updates credit_budget[0] in place (helius mode only).
    Returns credits consumed (helius) or tx count (std_rpc).
    parse_mode: "helius" (default) or "std_rpc" (public RPC fallback).
    """
    mint   = tok.get("token_address", "")
    sym    = tok.get("symbol") or mint[:8]
    peak   = tok.get("pct_change_peak")
    peak_s = f"{peak:+.0f}%" if peak is not None else "n/a"

    log.info("  %s (%s) peak=%s [%s]", sym, mint[:8], peak_s, parse_mode)

    alert_ts = None
    _alert_raw = tok.get("alert_time")
    if _alert_raw:
        try:
            alert_ts = datetime.fromisoformat(str(_alert_raw).replace("Z", "+00:00")).timestamp()
        except Exception:
            alert_ts = None

    # Fetch signatures (1 credit in helius mode; free in std_rpc mode)
    # Time-targeted at alert_ts when known — see _fetch_sigs docstring for why
    # (blind "most recent" misses the real trading window for many tokens).
    sigs = _fetch_sigs(mint, rpc_url if parse_mode == "helius" else std_rpc_url,
                       alert_ts=alert_ts)
    if parse_mode == "helius":
        credit_budget[0] -= 1
    time.sleep(_RATE_SLEEP)

    if not sigs:
        log.debug("  no sigs for %s", mint[:8])
        return 1

    # --- Helius enhanced path ---
    if parse_mode == "helius":
        all_rows: list = []
        total_parsed = 0
        for i in range(0, len(sigs), _PARSE_BATCH):
            batch = sigs[i:i + _PARSE_BATCH]
            if credit_budget[0] < len(batch):
                log.warning("  credit cap reached mid-token %s — stopping parse", mint[:8])
                break
            parsed = _parse_txs(batch, parse_url)
            credits_used = len(batch)
            credit_budget[0] -= credits_used
            total_parsed += credits_used
            rows = _extract_rows(parsed, mint, sol_price,
                                 research_event_id=tok.get("id", ""))
            all_rows.extend(rows)
            time.sleep(_RATE_SLEEP)
            if credit_budget[0] <= 0:
                break
        credits_consumed = 1 + total_parsed

    # --- Standard RPC path (public mainnet) ---
    else:
        tx_results = _parse_txs_std(sigs, std_rpc_url)
        try:
            from research.curve_oracle import derive_curve_address
            curve_pda = derive_curve_address(mint)
        except Exception:
            curve_pda = None
        all_rows = _extract_rows_std(tx_results, mint, sol_price,
                                     research_event_id=tok.get("id", ""),
                                     curve_pda=curve_pda)
        credits_consumed = len(sigs[:_STD_TX_CAP])  # informational only

    if not all_rows:
        log.debug("  no tradeable rows extracted for %s", mint[:8])
        return credits_consumed

    rel_path = _write_csv(mint, all_rows, out_dir)
    _update_db(sb, tok["id"], rel_path)
    log.info("  → %d rows → %s", len(all_rows), rel_path)
    return credits_consumed


def main():
    parser = argparse.ArgumentParser(
        description="Backfill trade-path CSVs from Helius transaction history"
    )
    parser.add_argument("--dry-run",     action="store_true",
                        help="print estimated credit cost only, no files written")
    parser.add_argument("--winners",     type=int, default=200,
                        help="max winner tokens to backfill (default: 200)")
    parser.add_argument("--losers",      type=int, default=200,
                        help="max loser tokens to backfill (default: 200)")
    parser.add_argument("--win-thresh",  type=float, default=50.0,
                        help="min pct_change_peak for winners (default: 50)")
    parser.add_argument("--credit-cap",  type=int, default=50_000,
                        help="hard Helius credit cap (default: 50000)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip tokens that already have a path_file in DB")
    parser.add_argument("--parse-mode",   type=str, default="auto",
                        choices=["auto", "helius", "std_rpc"],
                        help="parse mode: auto (detect quota), helius, or std_rpc (default: auto)")
    parser.add_argument("--std-rpc-url",  type=str, default=_PUBLIC_RPC,
                        help=f"RPC URL for std_rpc mode (default: {_PUBLIC_RPC})")
    args = parser.parse_args()

    from research.config import SUPABASE_URL, SUPABASE_KEY, HELIUS_API_KEY, RESEARCH_PATHS_DIR

    if not SUPABASE_URL or not SUPABASE_KEY:
        log.error("SUPABASE_URL and SUPABASE_KEY required")
        sys.exit(1)
    if not HELIUS_API_KEY:
        log.error("HELIUS_API_KEY required (even for std_rpc mode — used for dry-run estimate)")
        sys.exit(1)

    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    log.info("Loading tokens from Supabase...")
    winner_rows, loser_rows = _load_tokens(sb, args.winners, args.losers, args.win_thresh)
    all_tokens = winner_rows + loser_rows
    log.info("Loaded %d winners + %d losers = %d tokens",
             len(winner_rows), len(loser_rows), len(all_tokens))

    if args.skip_existing:
        before = len(all_tokens)
        all_tokens = [t for t in all_tokens if not t.get("path_file")]
        log.info("Skipped %d already-pathed tokens → %d remaining",
                 before - len(all_tokens), len(all_tokens))

    if not all_tokens:
        log.warning("No tokens to process")
        sys.exit(0)

    # Dry-run: detect mode then estimate
    if args.dry_run:
        helius_rpc_dr = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"
        dr_mode = args.parse_mode
        if dr_mode == "auto":
            log.info("Dry-run: checking Helius quota...")
            dr_mode = "std_rpc" if _check_helius_quota(helius_rpc_dr) else "helius"

        est_sigs   = len(all_tokens)
        est_parse  = len(all_tokens) * _DRY_RUN_AVG_TXS
        est_total  = est_sigs + est_parse
        pct_of_cap = est_total / args.credit_cap * 100
        print(f"\n=== Dry-run Credit Estimate (mode={dr_mode}) ===")
        print(f"Tokens to process:      {len(all_tokens)}")
        if dr_mode == "helius":
            print(f"  getSignaturesForAddress calls: {est_sigs}   ({est_sigs} credits)")
            print(f"  Enhanced TX parse (avg {_DRY_RUN_AVG_TXS} txns/token): {est_parse} credits")
            print(f"Total estimated credits: {est_total:,}  ({pct_of_cap:.1f}% of --credit-cap {args.credit_cap:,})")
            if est_total > args.credit_cap:
                n_covered = int(args.credit_cap / (1 + _DRY_RUN_AVG_TXS))
                print(f"WARNING: estimate exceeds cap. Would fully cover ~{n_covered} tokens.")
        else:
            est_min = est_sigs * _STD_TX_CAP * _STD_TX_SLEEP / _STD_MAX_WORKERS
            print(f"  std_rpc mode: 0 Helius credits. Public RPC, {_STD_MAX_WORKERS} workers.")
            print(f"  Estimated time: ~{est_min/60:.0f}-{est_min*2/60:.0f} min for {len(all_tokens)} tokens.")
        print(f"\nTo run for real:  python -m research.backfill_paths"
              f" --winners {args.winners} --losers {args.losers}"
              f" --parse-mode {dr_mode}")
        return

    sol_price  = _get_sol_price()
    helius_rpc = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"
    parse_url  = f"https://api.helius.xyz/v0/transactions/?api-key={HELIUS_API_KEY}"
    out_dir    = RESEARCH_PATHS_DIR / "backfill"

    # Determine parse mode
    parse_mode = args.parse_mode
    if parse_mode == "auto":
        log.info("Checking Helius quota...")
        if _check_helius_quota(helius_rpc):
            parse_mode = "std_rpc"
            log.warning("Helius quota exhausted — switching to std_rpc (public mainnet)")
        else:
            parse_mode = "helius"
            log.info("Helius quota OK — using enhanced API")

    std_rpc_url = args.std_rpc_url
    rpc_url     = helius_rpc if parse_mode == "helius" else std_rpc_url

    credit_budget = [args.credit_cap]   # mutable wrapper; only decremented in helius mode
    done = 0
    skipped = 0

    log.info("Starting backfill (mode=%s, credit_cap=%d, SOL=$%.2f)",
             parse_mode, args.credit_cap, sol_price)

    for i, tok in enumerate(all_tokens, 1):
        if credit_budget[0] <= 0:
            log.warning("Credit cap exhausted after %d tokens. Stopping.", done)
            break
        log.info("[%d/%d] Processing %s", i, len(all_tokens),
                 tok.get("symbol") or tok.get("token_address", "")[:8])
        try:
            used = _process_token(
                tok, rpc_url, parse_url, sol_price, out_dir, sb, credit_budget,
                parse_mode=parse_mode, std_rpc_url=std_rpc_url,
            )
            done += 1
            remaining = credit_budget[0]
            log.info("  credits used this token: %d | remaining: %d", used, remaining)
        except Exception as e:
            log.warning("  failed: %s", e)
            skipped += 1

    credits_used = args.credit_cap - credit_budget[0]
    print("\n=== Backfill Path Summary ===")
    print(f"Tokens processed:    {done}")
    print(f"Tokens skipped/err:  {skipped}")
    print(f"Credits used:        {credits_used:,} / {args.credit_cap:,}")
    print(f"Output directory:    {out_dir}")


if __name__ == "__main__":
    main()
