"""
Gate-block logger + counterfactual reporter.

Every time the live-trading pipeline blocks a signal at one of the four
admission gates, a row is appended to gate_blocks.csv.  A background poller
(run_followup_polls, called from the near-miss thread) fetches the DexScreener
price at T+1h for each unfollowed block, simulates the trade, and writes the
result to gate_followup.json.  generate_gate_report() then joins these two
sources and returns a single formatted string — one line per gate.

Gates logged:
  preflight_no_price  — PP returned no price in 2s
  preflight_price     — PP price already >15% above signal
  creator             — creator wallet unresolved after 3s
  breaker             — daily-loss or max-concurrent circuit breaker

Counterfactual simulation:
  sim_fill  = pp_price × 1.05  (measured latency offset)
              OR signal_price × 1.10 if pp_price == 0
  entry logic applies the signal-anchored hard stop (HARD_STOP = -0.35)
  T+1h DexScreener price is used as exit:
    — if price < stop_level  → sim_pnl_pct = HARD_STOP
    — otherwise              → sim_pnl_pct = (price - fill) / fill
  Fee drag of 3.4% is subtracted.
  sim_pnl_usd = size_usd × (sim_pnl_pct - FEE_DRAG)

Negative EV across a gate = gate is paying for itself.
Positive EV = gate is blocking profitable trades — tighten or remove it.
"""

import csv
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from memecoin.config import LOGS_DIR, DATA_DIR

log = logging.getLogger(__name__)

GATE_LOG_FILE    = LOGS_DIR / "gate_blocks.csv"
GATE_FOLLOWUP    = DATA_DIR / "gate_followup.json"

GATE_LOG_FIELDS = [
    "timestamp", "week", "gate", "chain",
    "token_address", "token_symbol",
    "pp_price", "signal_price", "size_usd",
]

# Counterfactual simulation constants
HARD_STOP   = -0.35    # replicate live hard-stop; intentionally fixed for comparability
FEE_DRAG    = 0.034    # ~3.4% round-trip (Jupiter spread + sol fees + pump.fun fee)
LATENCY_MUL = 1.05     # estimated fill slippage above PP price at block time
LATENCY_MUL_NOPRICE = 1.10  # coarser estimate when PP price was unavailable

_fw_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_gate_block(
    gate: str,
    chain: str,
    token_address: str,
    token_symbol: str,
    pp_price: float,
    signal_price: float,
    size_usd: float = 5.0,
) -> None:
    """
    Append one gate-block row to gate_blocks.csv.
    Thread-safe; never raises.
    """
    try:
        GATE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        write_hdr = not GATE_LOG_FILE.exists() or GATE_LOG_FILE.stat().st_size == 0
        with open(GATE_LOG_FILE, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=GATE_LOG_FIELDS)
            if write_hdr:
                w.writeheader()
            ts = time.time()
            w.writerow({
                "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
                "week":          time.strftime("%Y-W%V",  time.gmtime(ts)),
                "gate":          gate,
                "chain":         chain,
                "token_address": token_address,
                "token_symbol":  token_symbol,
                "pp_price":      round(pp_price, 12),
                "signal_price":  round(signal_price, 12),
                "size_usd":      round(size_usd, 2),
            })
        log.debug("GATE BLOCK logged  gate=%s  token=%s  pp=%.8f",
                  gate, token_symbol, pp_price)
    except Exception as e:
        log.debug("gate_logger.log_gate_block failed: %s", e)


# ---------------------------------------------------------------------------
# Follow-up poller (T+1h DexScreener price)
# ---------------------------------------------------------------------------

def _load_followup() -> dict:
    if not GATE_FOLLOWUP.exists():
        return {}
    try:
        return json.loads(GATE_FOLLOWUP.read_text())
    except Exception:
        return {}


def _save_followup(data: dict) -> None:
    GATE_FOLLOWUP.parent.mkdir(parents=True, exist_ok=True)
    with _fw_lock:
        GATE_FOLLOWUP.write_text(json.dumps(data))


def _sim_pnl(pp_price: float, signal_price: float,
             followup_price: float, size_usd: float) -> Optional[float]:
    """
    Simulate a trade: enter at estimated fill, exit at T+1h (or hard stop).
    Returns net PnL in USD, or None if inputs are invalid.
    """
    if followup_price <= 0:
        return None
    if pp_price > 0:
        fill = pp_price * LATENCY_MUL
    elif signal_price > 0:
        fill = signal_price * LATENCY_MUL_NOPRICE
    else:
        return None
    if fill <= 0:
        return None

    stop_level = fill * (1 + HARD_STOP)
    if followup_price <= stop_level:
        raw_pnl_pct = HARD_STOP
    else:
        raw_pnl_pct = (followup_price - fill) / fill

    net_pnl_usd = size_usd * (raw_pnl_pct - FEE_DRAG)
    return round(net_pnl_usd, 3)


def run_followup_polls() -> None:
    """
    Fetch DexScreener T+1h prices for gate blocks that are 1–23h old
    and don't yet have a follow-up price.  Updates gate_followup.json.
    Call from a background poller (e.g. the near-miss thread).  Never raises.
    """
    try:
        from memecoin.data_client import dex_get_token as _dex
        rows = _read_recent_blocks(hours=24)
        followup = _load_followup()
        now = time.time()
        updated = False

        for row in rows:
            ts_str  = row.get("timestamp", "")
            key     = f"{row.get('chain')}:{row.get('token_address')}:{ts_str}"
            if key in followup:
                continue   # already fetched

            # Only follow up once the block is >= 1h old
            try:
                ts_epoch = time.mktime(time.strptime(ts_str, "%Y-%m-%dT%H:%M:%SZ"))
            except Exception:
                continue
            if (now - ts_epoch) < 3600:
                continue

            chain   = row.get("chain", "")
            address = row.get("token_address", "")
            if not address:
                continue

            try:
                pair = _dex(chain, address)
                price_1h = float(pair.get("priceUsd") or 0) if pair else 0.0
            except Exception:
                price_1h = 0.0

            sim = _sim_pnl(
                pp_price      = float(row.get("pp_price") or 0),
                signal_price  = float(row.get("signal_price") or 0),
                followup_price = price_1h,
                size_usd      = float(row.get("size_usd") or 5),
            )
            followup[key] = {
                "followup_price": price_1h,
                "followup_ts":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                "sim_pnl_usd":    sim,
            }
            updated = True
            log.debug("Gate followup  gate=%s  token=%s  1h_price=%.8f  sim=$%.2f",
                      row.get("gate"), row.get("token_symbol"), price_1h, sim or 0)
            time.sleep(0.3)   # polite to DexScreener

        if updated:
            _save_followup(followup)
    except Exception as e:
        log.debug("gate_logger.run_followup_polls failed: %s", e)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _read_recent_blocks(hours: int = 168) -> list[dict]:
    """Read gate_blocks.csv rows from the last `hours` hours."""
    rows = []
    cutoff = time.time() - hours * 3600
    try:
        with open(GATE_LOG_FILE, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    ts_epoch = time.mktime(
                        time.strptime(row["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
                    )
                except Exception:
                    continue
                if ts_epoch >= cutoff:
                    rows.append(row)
    except FileNotFoundError:
        pass
    except Exception as e:
        log.debug("gate_logger._read_recent_blocks: %s", e)
    return rows


def generate_gate_report(days: int = 7) -> str:
    """
    Returns a formatted multi-line report.
    One summary line per gate + a totals line.

    Example:
      preflight_no_price   blocked 12   EV/trade $-0.41   total $-4.92  [gate CORRECT]
      preflight_price      blocked  4   EV/trade  $1.23   total  $4.92  [gate WRONG]
      creator              blocked  1   EV/trade  $0.00   total  $0.00  [no data]
      breaker              blocked  2   EV/trade $-0.88   total $-1.76  [gate CORRECT]
      ─────────────────────────────────────────────────────────────────
      TOTAL                blocked 19   EV/trade $-0.12   total $-2.76
    """
    rows     = _read_recent_blocks(hours=days * 24)
    followup = _load_followup()

    # Accumulate per gate
    stats: dict[str, dict] = {}
    for row in rows:
        gate = row.get("gate", "unknown")
        if gate not in stats:
            stats[gate] = {"count": 0, "ev_sum": 0.0, "ev_n": 0}
        stats[gate]["count"] += 1
        ts_str  = row.get("timestamp", "")
        key     = f"{row.get('chain')}:{row.get('token_address')}:{ts_str}"
        fu      = followup.get(key)
        if fu and fu.get("sim_pnl_usd") is not None:
            stats[gate]["ev_sum"] += fu["sim_pnl_usd"]
            stats[gate]["ev_n"]   += 1

    if not stats:
        return f"Gate report ({days}d): no data yet — gate_blocks.csv is empty or all blocks < 1h old"

    lines = [f"Gate counterfactual report (past {days}d):"]
    total_count = total_ev = 0

    gate_order = ["preflight_no_price", "preflight_price", "creator", "breaker"]
    # include any unexpected gates too
    for g in stats:
        if g not in gate_order:
            gate_order.append(g)

    for gate in gate_order:
        if gate not in stats:
            continue
        s = stats[gate]
        n  = s["count"]
        ev_n = s["ev_n"]
        total_count += n
        if ev_n > 0:
            ev_trade = s["ev_sum"] / ev_n
            ev_total = s["ev_sum"]
            total_ev += ev_total
            verdict = "gate CORRECT" if ev_total < 0 else "gate WRONG — consider widening"
            lines.append(
                f"  {gate:<22} blocked {n:>3}   "
                f"EV/trade ${ev_trade:>+6.2f}   total ${ev_total:>+7.2f}  [{verdict}]"
            )
        else:
            lines.append(
                f"  {gate:<22} blocked {n:>3}   "
                f"EV/trade      ?   total       ?  [no followup data yet]"
            )

    lines.append("  " + "─" * 70)
    if total_count > 0:
        ev_per = total_ev / max(sum(s["ev_n"] for s in stats.values()), 1)
        lines.append(
            f"  {'TOTAL':<22} blocked {total_count:>3}   "
            f"EV/trade ${ev_per:>+6.2f}   total ${total_ev:>+7.2f}"
        )

    return "\n".join(lines)
