"""
research/thr_reconstruct_paths.py — THR-BATCH T1: exact curve-reserve
reconstruction + cross-validation gate for on-curve trade history.

WHY THIS EXISTS (state of the post-K3 extractor, confirmed 2026-09-08):
research/backfill_paths.py's std_rpc path (_extract_rows_std) prices every
historical trade via a BALANCE-DELTA HEURISTIC: sol_amount / token_amount
from pre/post lamport and token-account balances. That heuristic is the
best available signal from getTransaction's balance fields alone, but it
is not exact -- it can't distinguish a trader's own fee/rent noise from
the actual trade, and (see bug below) its "vsol" field is wrong.

BUG FOUND DURING THIS TASK'S "confirm state" step, not previously known:
_extract_rows_std's vsol comment claims "the bonding-curve PDA's postBalance
IS the vsol reading at that tick." Verified false live, 2026-09-08: for
mint HnAVbEMfF1iLdBsSF2YGf9LHWvurwX63cTGHyKaWpump, curve PDA native lamport
balance = 0.1144 SOL vs decoded virtual_sol_reserves = 30.1126 SOL from the
account's own data field -- a ~30 SOL gap matching pump.fun's known virtual-
reserve design (the account barely holds real lamports; virtual_sol_reserves
is a struct field, not the account's native balance). Every vsol value ever
written by backfill_std_rpc mode is therefore wrong. Not fixed here (that
module's contract stays as-is per this task's scope); this module supplies
the correct exact reconstruction instead for on-curve segments.

EXACT RECONSTRUCTION METHOD, empirically verified (not recalled/assumed):
pump.fun's bonding-curve program (PUMP_PROGRAM, same constant as
research/curve_oracle.py) self-CPI-emits an Anchor "TradeEvent" on every
on-curve trade, visible in getTransaction's meta.logMessages as a
"Program data: <base64>" line. Verified byte layout, live, against TWO
independent real mints (2026-09-08, VPS):

  offset  0-8   discriminator (8 bytes, ignored -- match via mint bytes instead)
  offset  8-40  mint (Pubkey, 32 bytes)          <- anchor: must equal this trade's mint
  offset 40-48  sol_amount            (u64 LE, lamports)
  offset 48-56  token_amount          (u64 LE, raw base units, PUMP_DECIMALS)
  offset 56-57  is_buy                (bool, 1 byte)
  offset 57-89  user                  (Pubkey, 32 bytes, unused here)
  offset 89-97  timestamp             (i64 LE, unix seconds)
  offset 97-105 virtual_sol_reserves  (u64 LE, lamports)   <- THE exact reserve, post-trade
  offset 105-113 virtual_token_reserves (u64 LE, raw base units) <- THE exact reserve, post-trade
  offset 113-121 real_sol_reserves    (u64 LE, lamports)
  offset 121-129 real_token_reserves  (u64 LE, raw base units)
  (further bytes exist in newer program versions -- e.g. a length-prefixed
   "sell"/"buy" tag string -- not needed for pricing, not parsed)

Validation performed: for mint HnAVbEMfF1iLdBsSF2YGf9LHWvurwX63cTGHyKaWpump,
the event decoded from its most recent trade tx gave virtual_sol_reserves=
30.112633589 SOL, virtual_token_reserves=1068986546.118749 -- BOTH exactly
bit-identical to a fresh live getAccountInfo decode of the same curve
account moments later (no intervening trade). Repeated on a second,
independent mint (Fg8f3Q8GGhdzCw4QRjgCioQ3scq2SXcZt3mzZytcpump): again
exactly bit-identical (vsol=61.262072082, vtoken=525447472.570982). The
implied constant-product invariant k = virtual_sol_reserves *
virtual_token_reserves also matches pump.fun's publicly documented initial
curve constants (30 SOL / 1,073,000,000 tokens virtual, see
memecoin/pumpfun_reserve_pricing.py's PUMPFUN_INITIAL_VIRTUAL_TOKEN_RESERVES)
to ~8 significant figures across all three mints sampled.

Price formula applied to the event's own reserves is IDENTICAL to
research/curve_oracle.py's live PF0-fixed formula
(price_sol = vsol_sol / vtoken_ui, price_usd = price_sol * sol_price_usd)
-- just evaluated at the historical trade's own reserves instead of a live
snapshot. This is the "exact curve-reserve math" this task asks for.

SCOPE: only fires for ON-CURVE trades (the event is only emitted by the
bonding-curve program). Post-graduation / DEX trades have no matching
event and fall back to the existing balance-delta heuristic
(backfill_paths._extract_rows_std, reused not duplicated) -- tagged with a
distinct source so exact and heuristic rows are never conflated downstream.

Read-only. Does not touch any frozen registry, does not change any
threshold, does not write to Supabase directly (callers own persistence).
"""

from __future__ import annotations

import base64
import hashlib
import struct
from dataclasses import dataclass
from typing import Optional

from research.config import INTERVAL_MINUTES
from research.curve_oracle import PUMP_PROGRAM, PUMP_DECIMALS
from research.path_schema import PATH_SCHEMA_VERSION as _SCHEMA_VER
from memecoin.pumpfun_reserve_pricing import PUMPFUN_INITIAL_VIRTUAL_TOKEN_RESERVES

THR_RECONSTRUCT_VERSION = 1

# XVAL_TOLERANCE_PCT provenance (2026-09-09, 20-token pilot,
# research/thr_pilot_t1.py, interpolated gate): 77 comparable observations
# across 11 reconstructed tokens. p50=2.106% p75=8.319% p90=72.584%
# mean=18.569%. Distribution is bimodal, not a smooth tail: 9/11 tokens
# cluster under 14%, 2/11 sit at a near-constant ~71%/~89% across every
# one of their rows (not scattered -- a systematic per-token offset).
# Investigated, not assumed benign: both outlier tokens have
# progress_source="curve_account" (the SAME exact on-chain read mechanism
# this module's own reconstruction uses, not a DexScreener/heuristic
# estimate) for vsol_at_signal, and outcome_poller.py uses the same
# curve_oracle mechanism for price_t1m -- both interpolation endpoints are
# independently exact, on-chain, same-class reads. The large diffs are
# consistent with genuine large price moves inside the 60s linear-
# interpolation gap (both tokens show large pct_change_peak swings
# elsewhere in their own record), not a reconstruction or interpolation
# bug. Tolerance set at ~2x p75 (8.3% -> 15%): clears normal
# reconstruction/interpolation noise with margin, while still catching
# the two identified extreme-volatility cases as INVALID_XVAL -- correct,
# conservative behavior (flagging genuine uncertainty), not a false
# positive. Pilot pass rate at this tolerance: 9/11 tokens (82%).
XVAL_TOLERANCE_PCT = 15.0

# pump.fun's publicly documented initial virtual reserves (30 SOL /
# 1,073,000,000 tokens) -- cross-checked live against 3 real curve
# accounts during T1 verification: implied k = vsol*vtoken matched this
# to ~8 significant figures in every case (module docstring above).
# Used ONLY to convert an independently-captured vsol reading (e.g.
# vsol_at_signal, from memecoin/progress_capture.py -- never from this
# module's own reconstruction) into a price, for xval interpolation.
_PUMP_INITIAL_VIRTUAL_SOL_LAMPORTS = 30_000_000_000
_PUMP_K_RAW = _PUMP_INITIAL_VIRTUAL_SOL_LAMPORTS * (PUMPFUN_INITIAL_VIRTUAL_TOKEN_RESERVES * (10 ** PUMP_DECIMALS))

_INDEPENDENT_POLL_OFFSET_LABELS = ("T1m", "T3m", "T5m", "T10m", "T20m")

# Anchor event: 8 (discriminator) + 32 (mint) + 8+8+1+32+8+8+8+8+8 (fields
# through real_token_reserves) = 129 bytes minimum. Verified live, see
# module docstring.
_TRADE_EVENT_MIN_LEN = 129

_XVAL_OFFSET_LABELS = ("T1m", "T3m", "T10m")  # per T1 spec: peak + t1m/t3m/t10m


def _decode_trade_events(log_messages: list, mint_bytes: bytes) -> list[dict]:
    """Scan one tx's logMessages for pump.fun TradeEvent(s) matching this
    mint. See module docstring for the verified byte layout. Silently
    skips any 'Program data:' line that doesn't decode, is too short, or
    belongs to a different mint/program -- those are not on-curve trade
    events for this token."""
    events: list[dict] = []
    for line in (log_messages or []):
        if not line.startswith("Program data: "):
            continue
        try:
            raw = base64.b64decode(line[len("Program data: "):])
        except Exception:
            continue
        if len(raw) < _TRADE_EVENT_MIN_LEN:
            continue
        if raw[8:40] != mint_bytes:
            continue
        try:
            sol_amount, = struct.unpack_from("<Q", raw, 40)
            token_amount, = struct.unpack_from("<Q", raw, 48)
            is_buy = bool(raw[56])
            timestamp, = struct.unpack_from("<q", raw, 89)
            virtual_sol_reserves, = struct.unpack_from("<Q", raw, 97)
            virtual_token_reserves, = struct.unpack_from("<Q", raw, 105)
            real_sol_reserves, = struct.unpack_from("<Q", raw, 113)
            real_token_reserves, = struct.unpack_from("<Q", raw, 121)
        except struct.error:
            continue
        if virtual_token_reserves <= 0 or virtual_sol_reserves <= 0:
            continue
        events.append({
            "sol_amount": sol_amount, "token_amount": token_amount, "is_buy": is_buy,
            "timestamp": timestamp,
            "virtual_sol_reserves": virtual_sol_reserves,
            "virtual_token_reserves": virtual_token_reserves,
            "real_sol_reserves": real_sol_reserves,
            "real_token_reserves": real_token_reserves,
        })
    return events


def curve_event_price_usd(event: dict, sol_price_usd: float) -> float:
    """Exact on-curve price from a decoded TradeEvent's own reserves --
    same formula as research/curve_oracle.py's live _parse_curve_account
    (PF0-fixed), evaluated at the event's historical reserves."""
    vsol_sol = event["virtual_sol_reserves"] / 1e9
    vtoken_ui = event["virtual_token_reserves"] / (10 ** PUMP_DECIMALS)
    price_sol_per_token = vsol_sol / vtoken_ui
    return price_sol_per_token * sol_price_usd


def extract_rows_exact(tx_results: list, mint: str, sol_price: float,
                        research_event_id: str = "") -> tuple[list[dict], list[dict]]:
    """
    Splits tx_results into (exact_rows, heuristic_candidate_txs).

    exact_rows: canonical RF5 rows priced via curve-reserve math, for every
      tx where a matching on-curve TradeEvent was found. source=
      "reconstructed_curve_exact". venue_state="CURVE_ACTIVE" (a matching
      event structurally proves the trade was on-curve).

    heuristic_candidate_txs: the raw tx_results with NO matching event
      (post-graduation trades, or unparseable) -- caller decides whether to
      run these through backfill_paths._extract_rows_std as a fallback and
      how to tag them (kept separate here rather than calling that function
      internally, to avoid this module silently taking on that module's
      heuristic-pricing behavior as its own default).
    """
    import base58

    mint_bytes = base58.b58decode(mint)
    exact_rows: list[dict] = []
    heuristic_candidates: list[dict] = []

    for result in tx_results:
        if not isinstance(result, dict):
            continue
        meta = result.get("meta") or {}
        # NOTE: real getTransaction responses carry the error under
        # meta.err, not a top-level "err" key -- backfill_paths.py's
        # _extract_rows_std checks result.get("err") instead, which is
        # always None (silent no-op), a second pre-existing bug found
        # during this task. Not fixed there (out of scope); fixed here.
        if meta.get("err"):
            continue
        log_messages = meta.get("logMessages") or []
        events = _decode_trade_events(log_messages, mint_bytes)
        if not events:
            heuristic_candidates.append(result)
            continue

        ts = result.get("blockTime")
        if not ts:
            continue
        ts_ms = int(ts) * 1000

        for ev in events:
            price_usd = curve_event_price_usd(ev, sol_price)
            price_sol = round(price_usd / sol_price, 12) if sol_price > 0 else 0.0
            event_id = hashlib.sha256(
                f"reconstructed:{mint}:{ts_ms}:{ev['sol_amount']}:{ev['token_amount']}:{ev['is_buy']}".encode()
            ).hexdigest()[:32]
            exact_rows.append({
                "schema_version":    str(_SCHEMA_VER),
                "research_event_id": research_event_id,
                "event_id":          event_id,
                "ts_ms":             ts_ms,
                "price_usd":         round(price_usd, 12),
                "price_sol":         price_sol,
                "side":              "buy" if ev["is_buy"] else "sell",
                "token_amount":      round(ev["token_amount"] / (10 ** PUMP_DECIMALS), 6),
                "sol_amount":        round(ev["sol_amount"] / 1e9, 9),
                "vsol":              round(ev["virtual_sol_reserves"] / 1e9, 9),
                "vtok":              ev["virtual_token_reserves"],  # v3 schema: raw units, NO /1e6 (see path_schema.py)
                "source":            "reconstructed_curve_exact",
                "venue_state":       "CURVE_ACTIVE",
                "backfilled":        "true",
                "data_status":       "ok",
                "trader_pk":         "",
            })

    exact_rows.sort(key=lambda r: r["ts_ms"])
    return exact_rows, heuristic_candidates


# ── Cross-validation gate ───────────────────────────────────────────────

@dataclass(frozen=True)
class XvalOffsetDiff:
    label: str                  # "T1m" / "T3m" / "T10m" / "peak"
    reconstructed_price: Optional[float]
    poll_price: Optional[float]
    pct_diff: Optional[float]   # (reconstructed - poll) / poll * 100, None if either side missing
    staleness_s: Optional[float] = None  # target offset time minus the reconstructed
                                          # reference row's own ts (seconds). A large
                                          # value means the "reconstructed" price being
                                          # compared is not actually from near the target
                                          # offset (real trading went quiet before then) --
                                          # a comparison confound, not a reconstruction
                                          # error. See T1 pilot finding, docs/RECEIPTS.md.


@dataclass(frozen=True)
class XvalResult:
    diffs: tuple                # tuple[XvalOffsetDiff, ...]
    max_abs_pct_diff: Optional[float]
    status: str                 # "PASS" / "FAIL" / "INSUFFICIENT_DATA"


def _nearest_reconstructed_row_at_or_before(rows: list, target_ms: int) -> Optional[dict]:
    candidates = [r for r in rows if r["ts_ms"] <= target_ms]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["ts_ms"])


def compute_xval_diffs(reconstructed_rows: list, token_row: dict, alert_ts: float) -> list[XvalOffsetDiff]:
    """
    Compares reconstructed prices against this token's OWN independently-
    polled prices (research.outcome_poller's price_t1m/t3m/t10m columns --
    a feed that never reads path/tick data, so this is a genuine
    cross-check between two independent sources, not circular).
    Also compares reconstructed peak vs pct_change_peak-implied peak price
    where both a peak reference price and reconstructed rows exist.

    Every offset comparison also reports staleness_s: how far the
    reconstructed reference row actually is from the target offset. When
    real trading goes quiet shortly after alert (common for thin tokens --
    see T1 pilot finding), the nearest-before row can be minutes stale;
    a large pct_diff paired with large staleness is expected token
    movement over that gap, not evidence the reconstruction math is wrong.
    Callers deriving a tolerance should condition on staleness, not treat
    the raw pooled distribution as apples-to-apples.
    """
    diffs: list[XvalOffsetDiff] = []

    for label in _XVAL_OFFSET_LABELS:
        offset_s = INTERVAL_MINUTES[label] * 60
        target_ms = int((alert_ts + offset_s) * 1000)
        poll_price = token_row.get(f"price_{label.lower()}")
        recon_row = _nearest_reconstructed_row_at_or_before(reconstructed_rows, target_ms)
        recon_price = recon_row["price_usd"] if recon_row else None
        staleness_s = (target_ms - recon_row["ts_ms"]) / 1000.0 if recon_row else None
        pct_diff = None
        if poll_price not in (None, 0) and recon_price is not None:
            pct_diff = (recon_price - poll_price) / poll_price * 100.0
        diffs.append(XvalOffsetDiff(label=label, reconstructed_price=recon_price,
                                     poll_price=poll_price, pct_diff=pct_diff,
                                     staleness_s=staleness_s))

    # Peak: compare reconstructed max price over the reconstructed rows
    # against the poll-derived peak price implied by pct_change_peak +
    # the alert-time entry price (price_t1m is the earliest reliable
    # independent post-alert price available; falls back gracefully if
    # missing, same as the T1m/T3m/T10m checks above).
    pct_change_peak = token_row.get("pct_change_peak")
    entry_ref_price = token_row.get("price_t1m")
    poll_peak_price = None
    if pct_change_peak is not None and entry_ref_price not in (None, 0):
        poll_peak_price = entry_ref_price * (1 + pct_change_peak / 100.0)
    recon_peak_price = max((r["price_usd"] for r in reconstructed_rows), default=None)
    pct_diff_peak = None
    if poll_peak_price not in (None, 0) and recon_peak_price is not None:
        pct_diff_peak = (recon_peak_price - poll_peak_price) / poll_peak_price * 100.0
    diffs.append(XvalOffsetDiff(label="peak", reconstructed_price=recon_peak_price,
                                 poll_price=poll_peak_price, pct_diff=pct_diff_peak))

    return diffs


def classify_xval(diffs: list, tolerance_pct: float) -> XvalResult:
    """tolerance_pct must be supplied by the caller, derived from the
    pilot's own observed mismatch distribution (see run_pilot) -- never
    a value invented ahead of seeing real data."""
    usable = [d for d in diffs if d.pct_diff is not None]
    if not usable:
        return XvalResult(diffs=tuple(diffs), max_abs_pct_diff=None, status="INSUFFICIENT_DATA")
    max_abs = max(abs(d.pct_diff) for d in usable)
    status = "PASS" if max_abs <= tolerance_pct else "FAIL"
    return XvalResult(diffs=tuple(diffs), max_abs_pct_diff=max_abs, status=status)


# ── Interpolated cross-validation (T1 pilot fix) ────────────────────────
# The fixed-offset gate above (compute_xval_diffs) compares a reconstructed
# row's price against a poll mark 60-600s away -- fine when reconstructed
# coverage reaches that far, but most thin tokens stop trading within
# seconds of alert (T1 pilot finding, docs/RECEIPTS.md), so the comparison
# ends up confounded by genuine token movement over the gap rather than
# testing reconstruction accuracy. This section instead interpolates an
# INDEPENDENT reference price at each reconstructed row's OWN timestamp,
# never extrapolating beyond the available reference range -- a same-time
# comparison, not same-fixed-offset.

def price_from_vsol_via_curve_invariant(vsol_ui: Optional[float], sol_price_usd: float) -> Optional[float]:
    """Exact on-curve price from a vsol reading alone, via the verified
    constant-product invariant k = virtual_sol_reserves * virtual_token_
    reserves (module docstring: cross-checked live against 3 real curve
    accounts, matched pump.fun's public initial constants to ~8 sig figs).
    Intended for vsol_at_signal (memecoin/progress_capture.py) -- a live
    capture completely independent of this module's own reconstruction."""
    if vsol_ui is None or vsol_ui <= 0:
        return None
    vsol_lamports = vsol_ui * 1e9
    vtoken_raw = _PUMP_K_RAW / vsol_lamports
    vtoken_ui = vtoken_raw / (10 ** PUMP_DECIMALS)
    price_sol = vsol_ui / vtoken_ui
    return price_sol * sol_price_usd


def independent_reference_points(token_row: dict, alert_ts: float, sol_price_usd: float) -> list[tuple]:
    """Sorted [(ts_ms, price_usd), ...] built ONLY from sources independent
    of this module's reconstruction: vsol_at_signal (progress_capture.py,
    a separate live-capture mechanism, converted via the curve invariant)
    and price_t1m/t3m/t5m/t10m/t20m (outcome_poller.py, never reads
    path/tick data). Neither source ever touches a reconstructed row."""
    points: list[tuple] = []

    vsol_at_signal = token_row.get("vsol_at_signal")
    if vsol_at_signal is not None:
        price = price_from_vsol_via_curve_invariant(vsol_at_signal, sol_price_usd)
        if price is not None:
            lag_ms = token_row.get("progress_capture_lag_ms") or 0
            ts_ms = int(alert_ts * 1000 + lag_ms)
            points.append((ts_ms, price))

    for label in _INDEPENDENT_POLL_OFFSET_LABELS:
        offset_s = INTERVAL_MINUTES[label] * 60
        price = token_row.get(f"price_{label.lower()}")
        if price is not None and price > 0:
            points.append((int((alert_ts + offset_s) * 1000), price))

    points.sort(key=lambda p: p[0])
    return points


def _interpolate_price_at(points: list, target_ms: int) -> Optional[float]:
    """Linear interpolation strictly between two known reference points.
    Returns None if target_ms falls outside the known range -- never
    extrapolates. A single reference point is usable only for an exact
    timestamp match (no second point to bracket with)."""
    if not points or target_ms < points[0][0] or target_ms > points[-1][0]:
        return None
    if len(points) == 1:
        return points[0][1] if target_ms == points[0][0] else None
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        if t0 <= target_ms <= t1:
            if t1 == t0:
                return p0
            frac = (target_ms - t0) / (t1 - t0)
            return p0 + frac * (p1 - p0)
    return None


@dataclass(frozen=True)
class InterpolatedTickDiff:
    ts_ms: int
    reconstructed_price: float
    interpolated_price: float
    pct_diff: float


def compute_interpolated_xval(reconstructed_rows: list, token_row: dict, alert_ts: float,
                               sol_price_usd: float) -> list[InterpolatedTickDiff]:
    """For each reconstructed row, compares against an independent
    reference price interpolated AT THAT ROW'S OWN TIMESTAMP -- removes
    the staleness confound structurally rather than flagging it. Rows
    outside the independent reference range are skipped, not compared."""
    points = independent_reference_points(token_row, alert_ts, sol_price_usd)
    diffs: list[InterpolatedTickDiff] = []
    for row in reconstructed_rows:
        interp = _interpolate_price_at(points, row["ts_ms"])
        if interp is None or interp <= 0:
            continue
        pct_diff = (row["price_usd"] - interp) / interp * 100.0
        diffs.append(InterpolatedTickDiff(ts_ms=row["ts_ms"], reconstructed_price=row["price_usd"],
                                           interpolated_price=interp, pct_diff=pct_diff))
    return diffs


def classify_interpolated_xval(diffs: list, tolerance_pct: float = XVAL_TOLERANCE_PCT) -> XvalResult:
    """Per-token verdict for T3's path INVALID_XVAL gate. Defaults to
    XVAL_TOLERANCE_PCT (see its provenance comment) but accepts an
    override for sensitivity checks. INSUFFICIENT_DATA when no
    reconstructed row fell within the independent reference range at
    all (not the same as PASS -- absence of evidence, not evidence of
    agreement)."""
    if not diffs:
        return XvalResult(diffs=(), max_abs_pct_diff=None, status="INSUFFICIENT_DATA")
    max_abs = max(abs(d.pct_diff) for d in diffs)
    status = "PASS" if max_abs <= tolerance_pct else "FAIL"
    return XvalResult(diffs=tuple(diffs), max_abs_pct_diff=max_abs, status=status)
