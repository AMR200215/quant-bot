"""
memecoin/tools/summarize_trade_telemetry.py — Gap attribution tool.

Reads logs/trade_telemetry.jsonl and logs/memecoin_live_journal.csv,
groups events by trace_id, and classifies paper/live PnL gaps.

Usage:
    python -m memecoin.tools.summarize_trade_telemetry [--since DATE] [--pair-id ID] [--output csv]
"""

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

TELEMETRY_FILE  = Path(__file__).parent.parent.parent / "logs" / "trade_telemetry.jsonl"
LIVE_JOURNAL    = Path(__file__).parent.parent.parent / "logs" / "memecoin_live_journal.csv"
PAPER_JOURNAL   = Path(__file__).parent.parent.parent / "logs" / "memecoin_social_journal.csv"
ARTIFACTS_DIR   = Path(__file__).parent.parent.parent / "artifacts" / "telemetry"

OUTPUT_FIELDS = [
    "trace_id", "pair_id", "symbol", "mint", "category", "token_program",
    "route_buy", "route_sell",
    "alert_received_ts", "preflight_start_ts", "preflight_done_ts",
    "buy_build_start_ts", "buy_sent_ts", "buy_confirmed_ts", "fill_recorded_ts",
    "exit_triggered_ts", "sell_build_start_ts", "sell_sent_ts", "sell_confirmed_ts",
    "alert_to_fill_ms", "exit_trigger_to_sell_confirmed_ms",
    "live_size_usd", "paper_size_usd", "hyp_size_025_usd",
    "live_pnl_usd", "paper_pnl_usd", "live_pnl_pct", "paper_pnl_pct",
    "raw_dollar_gap", "pct_gap",
    "gap_reason", "missing_fields",
]


def load_telemetry(since: str | None = None, pair_id: str | None = None) -> dict[str, list[dict]]:
    """Load events grouped by trace_id."""
    traces: dict[str, list[dict]] = defaultdict(list)
    if not TELEMETRY_FILE.exists():
        return traces
    since_dt = None
    if since:
        since_dt = datetime.fromisoformat(since).replace(tzinfo=timezone.utc)
    with open(TELEMETRY_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if pair_id and rec.get("pair_id") != pair_id:
                continue
            if since_dt:
                try:
                    wall = rec.get("timestamp_wall", "")
                    ts = datetime.fromisoformat(wall.rstrip("Z")).replace(tzinfo=timezone.utc)
                    if ts < since_dt:
                        continue
                except (ValueError, TypeError):
                    pass
            tid = rec.get("trace_id", "unknown")
            traces[tid].append(rec)
    return traces


def load_journal() -> dict[str, dict]:
    """Load live journal keyed by pos_id (id column)."""
    rows: dict[str, dict] = {}
    if not LIVE_JOURNAL.exists():
        return rows
    with open(LIVE_JOURNAL) as f:
        for r in csv.DictReader(f):
            pid = r.get("id", "")
            if pid:
                rows[pid] = r
    return rows


def load_paper_journal() -> dict[str, dict]:
    """Load paper (social) journal keyed by pair_id, then id."""
    rows: dict[str, dict] = {}
    if not PAPER_JOURNAL.exists():
        return rows
    with open(PAPER_JOURNAL) as f:
        for r in csv.DictReader(f):
            # Prefer pair_id as key for pairing; fall back to id
            pid = r.get("pair_id", "") or r.get("id", "")
            if pid:
                rows[pid] = r
    return rows


def _get_ts(events: list[dict], event_name: str, field: str = "timestamp_wall") -> float | None:
    """Find first event with given name and return its timestamp as epoch."""
    for e in events:
        if e.get("event_name") == event_name:
            raw = e.get(field)
            if raw is None:
                return None
            if isinstance(raw, (int, float)):
                return float(raw)
            try:
                return datetime.fromisoformat(raw.rstrip("Z")).replace(tzinfo=timezone.utc).timestamp()
            except (ValueError, TypeError):
                return None
    return None


def _get_field(events: list[dict], event_name: str, field: str):
    """Get a specific field from the first matching event."""
    for e in events:
        if e.get("event_name") == event_name:
            return e.get(field)
    return None


def _get_buy_sent_ts(events: list[dict]) -> float | None:
    """Return buy_sent timestamp, falling back to buy_build_done if buy_sent absent."""
    ts = _get_ts(events, "buy_sent")
    if ts is not None:
        return ts
    return _get_ts(events, "buy_build_done")


def _get_token_program(events: list[dict]) -> str:
    """Extract token_program from entry_gate_checked event."""
    val = _get_field(events, "entry_gate_checked", "token_program")
    return val if val is not None else ""


def pair_traces(
    live_traces: dict[str, list[dict]],
    paper_traces: dict[str, list[dict]],
) -> list[dict]:
    """
    Pair live telemetry traces with paper traces.

    Priority:
    1. pair_id match between live trace and paper trace
    2. signal_id + mint match
    3. Legacy: same mint within 60s of each other
    4. MISSING_PAIR if none found

    Returns list of dicts: {live_trace_id, paper_trace_id, match_method}
    """
    results = []

    def _first_field(events: list[dict], field: str):
        for e in events:
            v = e.get(field)
            if v:
                return v
        return None

    def _first_ts(events: list[dict]) -> float | None:
        for e in events:
            raw = e.get("timestamp_wall")
            if raw:
                try:
                    if isinstance(raw, (int, float)):
                        return float(raw)
                    return datetime.fromisoformat(raw.rstrip("Z")).replace(tzinfo=timezone.utc).timestamp()
                except (ValueError, TypeError):
                    pass
        return None

    paper_by_pair_id: dict[str, str] = {}
    paper_by_signal_mint: dict[tuple, str] = {}
    paper_mint_ts: list[tuple[str, str, float]] = []  # (trace_id, mint, ts)

    for ptid, pevents in paper_traces.items():
        ppid = _first_field(pevents, "pair_id")
        if ppid:
            paper_by_pair_id[ppid] = ptid

        sig = _first_field(pevents, "signal_id")
        mint = _first_field(pevents, "mint")
        if sig and mint:
            paper_by_signal_mint[(sig, mint)] = ptid

        pts = _first_ts(pevents)
        if mint and pts:
            paper_mint_ts.append((ptid, mint, pts))

    matched_paper: set[str] = set()

    for ltid, levents in live_traces.items():
        lpid = _first_field(levents, "pair_id")
        lsig = _first_field(levents, "signal_id")
        lmint = _first_field(levents, "mint")
        lts = _first_ts(levents)

        matched_ptid = None
        method = "MISSING_PAIR"

        # Priority 1: pair_id match
        if lpid and lpid in paper_by_pair_id:
            candidate = paper_by_pair_id[lpid]
            if candidate not in matched_paper:
                matched_ptid = candidate
                method = "pair_id"

        # Priority 2: signal_id + mint
        if matched_ptid is None and lsig and lmint:
            key = (lsig, lmint)
            if key in paper_by_signal_mint:
                candidate = paper_by_signal_mint[key]
                if candidate not in matched_paper:
                    matched_ptid = candidate
                    method = "signal_id_mint"

        # Priority 3: same mint within 60s
        if matched_ptid is None and lmint and lts:
            best_delta = None
            best_ptid = None
            for ptid, pmint, pts in paper_mint_ts:
                if pmint == lmint and ptid not in matched_paper:
                    delta = abs(lts - pts)
                    if delta <= 60 and (best_delta is None or delta < best_delta):
                        best_delta = delta
                        best_ptid = ptid
            if best_ptid:
                matched_ptid = best_ptid
                method = "mint_within_60s"

        if matched_ptid:
            matched_paper.add(matched_ptid)

        results.append({
            "live_trace_id": ltid,
            "paper_trace_id": matched_ptid or "MISSING_PAIR",
            "match_method": method,
        })

    return results


def classify_gap(
    live_pnl_pct: float | None,
    paper_pnl_pct: float | None,
    live_size: float | None,
    paper_size: float | None,
    events: list[dict],
    missing: list[str],
) -> str:
    """Classify the paper/live gap reason."""
    event_names = {e.get("event_name") for e in events}

    # Priority 1: tp_partial_failure
    for e in events:
        if e.get("event_name") == "sell_failed" and "tp_partial" in str(e.get("reason", "")):
            return "tp_partial_failure"

    # Priority 2: sell_route_failure
    for e in events:
        if e.get("event_name") == "sell_failed" and e.get("all_steps_reverted"):
            return "sell_route_failure"

    # Priority 3: entry_drift
    drift = _get_field(events, "size_shadow", "drift_pct")
    gap_pp = abs((live_pnl_pct or 0) - (paper_pnl_pct or 0))
    if drift is not None and abs(drift) > 20 and gap_pp > 10:
        return "entry_drift"

    # Priority 4: size_asymmetry_confound
    if live_size and paper_size and live_size != paper_size and gap_pp < 5:
        return "size_asymmetry_confound"

    # Priority 5: exit_latency
    exit_trigger_ts = _get_ts(events, "exit_triggered")
    sell_sent_ts = _get_ts(events, "sell_confirmed")  # approximate
    if exit_trigger_ts and sell_sent_ts and (sell_sent_ts - exit_trigger_ts) * 1000 > 5000:
        return "exit_latency"

    # Priority 6: journal_pending_reconcile
    for e in events:
        if e.get("event_name") == "journal_written" and e.get("fill_estimated"):
            return "journal_pending_reconcile"

    # Priority 7: route_pending_receipt
    for e in events:
        if e.get("event_name") == "sell_confirmed":
            route = e.get("route_used", "")
            if "T22" in str(route).upper():
                return "route_pending_receipt"

    # Priority 8: strategy_loss
    if (live_pnl_pct is not None and live_pnl_pct < 0
            and paper_pnl_pct is not None and paper_pnl_pct < 0):
        return "strategy_loss"

    # Priority 9: missing_data
    if missing:
        return "missing_data"

    return "unknown"


def summarize(since: str | None = None, pair_id: str | None = None) -> list[dict]:
    """Build summary rows."""
    traces = load_telemetry(since=since, pair_id=pair_id)
    journal = load_journal()
    paper_journal = load_paper_journal()
    rows = []

    for tid, events in traces.items():
        if not events:
            continue
        first = events[0]
        pos_id = first.get("pos_id", "")
        trace_pair_id = first.get("pair_id", "")
        jrow = journal.get(pos_id, {})

        # Paper journal: try pair_id first, then pos_id
        pjrow = paper_journal.get(trace_pair_id, {}) or paper_journal.get(pos_id, {})

        # Timing
        timing_fields = {
            "alert_received_ts": _get_ts(events, "alert_received"),
            "preflight_start_ts": _get_ts(events, "preflight_started"),
            "preflight_done_ts": _get_ts(events, "preflight_baseline_selected"),
            "buy_build_start_ts": _get_ts(events, "buy_build_started"),
            "buy_sent_ts": _get_buy_sent_ts(events),  # Bug 2 fix: buy_sent with fallback
            "buy_confirmed_ts": _get_ts(events, "buy_confirmed"),
            "fill_recorded_ts": _get_ts(events, "buy_fill_recorded"),
            "exit_triggered_ts": _get_ts(events, "exit_triggered"),
            "sell_build_start_ts": _get_ts(events, "exit_queued"),
            "sell_sent_ts": _get_ts(events, "sell_confirmed"),
            "sell_confirmed_ts": _get_ts(events, "sell_confirmed"),
        }

        missing = [k for k, v in timing_fields.items() if v is None]

        alert_ts = timing_fields["alert_received_ts"]
        fill_ts = timing_fields["fill_recorded_ts"]
        exit_ts = timing_fields["exit_triggered_ts"]
        sell_ts = timing_fields["sell_confirmed_ts"]

        alert_to_fill = round((fill_ts - alert_ts) * 1000, 1) if alert_ts and fill_ts else None
        exit_to_sell = round((sell_ts - exit_ts) * 1000, 1) if exit_ts and sell_ts else None

        live_size = _get_field(events, "buy_fill_recorded", "live_size_usd")
        paper_size = _get_field(events, "size_shadow", "live_size_usd")
        hyp_size = _get_field(events, "size_shadow", "hyp_size_025_usd")

        live_pnl_usd = None
        live_pnl_pct = None
        paper_pnl_usd = None
        paper_pnl_pct = None
        try:
            live_pnl_usd = float(jrow.get("pnl_usd") or 0) if jrow else None
            live_pnl_pct = float(jrow.get("pnl_pct") or 0) if jrow else None
        except (TypeError, ValueError):
            pass

        # Bug 1 fix: populate paper PnL from paper journal
        try:
            paper_pnl_usd = float(pjrow.get("pnl_usd") or 0) if pjrow else None
            paper_pnl_pct = float(pjrow.get("pnl_pct") or 0) if pjrow else None
        except (TypeError, ValueError):
            pass

        raw_gap = None
        pct_gap = None
        if live_pnl_usd is not None and paper_pnl_usd is not None:
            raw_gap = round(live_pnl_usd - paper_pnl_usd, 4)
        if live_pnl_pct is not None and paper_pnl_pct is not None:
            pct_gap = round(live_pnl_pct - paper_pnl_pct, 2)

        gap_reason = classify_gap(
            live_pnl_pct, paper_pnl_pct, live_size, paper_size, events, missing
        )

        # Bug 4: note partial exits in missing_fields / gap_reason
        n_partial_exits = sum(
            1 for e in events
            if e.get("event_name") in ("tp_sell_confirmed", "tp_partial_confirmed")
        )
        missing_note = list(missing)
        if n_partial_exits > 1:
            missing_note.append(f"partial_exits={n_partial_exits}")

        # Bug 3 fix: extract token_program from entry_gate_checked event
        token_program = _get_token_program(events)

        row = {
            "trace_id": tid,
            "pair_id": trace_pair_id,
            "symbol": first.get("symbol", ""),
            "mint": first.get("mint", ""),
            "category": first.get("live_or_paper", ""),
            "token_program": token_program,
            "route_buy": _get_field(events, "buy_confirmed", "route_used") or "",
            "route_sell": _get_field(events, "sell_confirmed", "route_used") or "",
            **{k: v for k, v in timing_fields.items()},
            "alert_to_fill_ms": alert_to_fill,
            "exit_trigger_to_sell_confirmed_ms": exit_to_sell,
            "live_size_usd": live_size,
            "paper_size_usd": paper_size,
            "hyp_size_025_usd": hyp_size,
            "live_pnl_usd": live_pnl_usd,
            "paper_pnl_usd": paper_pnl_usd,
            "live_pnl_pct": live_pnl_pct,
            "paper_pnl_pct": paper_pnl_pct,
            "raw_dollar_gap": raw_gap,
            "pct_gap": pct_gap,
            "gap_reason": gap_reason,
            "missing_fields": ",".join(missing_note),
        }
        rows.append(row)

    return rows


def _write_csv_files(rows: list[dict]) -> None:
    """Write 6 CSV files to artifacts/telemetry/."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. per_trade.csv
    with open(ARTIFACTS_DIR / "per_trade.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in OUTPUT_FIELDS})

    # 2. missing_data.csv
    missing_rows = [r for r in rows if r.get("gap_reason") == "missing_data"]
    with open(ARTIFACTS_DIR / "missing_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        w.writeheader()
        for r in missing_rows:
            w.writerow({k: r.get(k, "") for k in OUTPUT_FIELDS})

    # 3. timing_summary.csv
    timing_keys = ["alert_to_fill_ms", "exit_trigger_to_sell_confirmed_ms"]
    timing_summary_fields = ["metric", "count", "avg", "p50", "p95"]
    with open(ARTIFACTS_DIR / "timing_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=timing_summary_fields)
        w.writeheader()
        for key in timing_keys:
            vals = [r[key] for r in rows if r.get(key) is not None]
            if vals:
                sorted_vals = sorted(vals)
                n = len(sorted_vals)
                avg = round(sum(sorted_vals) / n, 1)
                p50 = round(statistics.median(sorted_vals), 1)
                p95_idx = max(0, int(n * 0.95) - 1)
                p95 = round(sorted_vals[p95_idx], 1)
            else:
                avg = p50 = p95 = ""
            w.writerow({"metric": key, "count": len(vals), "avg": avg, "p50": p50, "p95": p95})

    # 4. gap_attribution.csv
    gap_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        gap_counts[r.get("gap_reason", "unknown")] += 1
    with open(ARTIFACTS_DIR / "gap_attribution.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gap_reason", "count"])
        w.writeheader()
        for reason, count in sorted(gap_counts.items()):
            w.writerow({"gap_reason": reason, "count": count})

    # 5. token_program_breakdown.csv
    tp_counts: dict[str, int] = defaultdict(int)
    for r in rows:
        tp = r.get("token_program") or "UNKNOWN"
        tp_upper = str(tp).upper()
        if "T22" in tp_upper or "TOKEN2022" in tp_upper or "TOKEN_2022" in tp_upper:
            key = "T22"
        elif "SPL" in tp_upper or "TOKEN" in tp_upper:
            key = "SPL"
        else:
            key = "UNKNOWN"
        tp_counts[key] += 1
    with open(ARTIFACTS_DIR / "token_program_breakdown.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["token_program", "count"])
        w.writeheader()
        for tp, count in sorted(tp_counts.items()):
            w.writerow({"token_program": tp, "count": count})

    # 6. missing_pair.csv
    # Traces where we couldn't find paper journal data (paper_pnl_usd is None)
    missing_pair_rows = [r for r in rows if r.get("paper_pnl_usd") is None]
    with open(ARTIFACTS_DIR / "missing_pair.csv", "w", newline="") as f:
        pair_fields = ["trace_id", "pair_id", "symbol", "mint", "gap_reason"]
        w = csv.DictWriter(f, fieldnames=pair_fields)
        w.writeheader()
        for r in missing_pair_rows:
            w.writerow({k: r.get(k, "") for k in pair_fields})


def main():
    parser = argparse.ArgumentParser(description="Trade telemetry gap attribution")
    parser.add_argument("--since", type=str, default=None, help="ISO date filter")
    parser.add_argument("--pair-id", type=str, default=None, help="Filter by pair_id")
    parser.add_argument("--output", type=str, default="table", choices=["csv", "table"],
                        help="Output format")
    args = parser.parse_args()

    rows = summarize(since=args.since, pair_id=args.pair_id)
    if not rows:
        print("No telemetry data found.")
        return

    if args.output == "csv":
        _write_csv_files(rows)
        # Also write main output to stdout for piping
        w = csv.DictWriter(sys.stdout, fieldnames=OUTPUT_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in OUTPUT_FIELDS})
    else:
        for r in rows:
            print(f"\n--- {r['trace_id']} ({r['symbol']}) ---")
            for k in OUTPUT_FIELDS:
                v = r.get(k, "")
                if v is not None and v != "":
                    print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
