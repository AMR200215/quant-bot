"""
compare_paper_live_execution.py

Compare paper vs live trade execution to identify slippage, route failures,
and execution drag.

Usage:
    python -m memecoin.tools.compare_paper_live_execution

Reads:
    logs/memecoin_live_journal.csv   — live trades
    logs/memecoin_journal.csv        — all trades (paper + social)
    logs/exit_route_attempts.csv     — route attempt log (optional)

Outputs:
    Printed table + logs/paper_vs_live_execution.csv
"""

import csv
import os
import sys
from pathlib import Path

# Allow running as a script
_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    from memecoin.config import LOGS_DIR
except ImportError:
    LOGS_DIR = _REPO_ROOT / "logs"

_LIVE_JOURNAL   = LOGS_DIR / "memecoin_live_journal.csv"
_PAPER_JOURNAL  = LOGS_DIR / "memecoin_journal.csv"
_ROUTE_ATTEMPTS = LOGS_DIR / "exit_route_attempts.csv"
_OUTPUT_CSV     = LOGS_DIR / "paper_vs_live_execution.csv"

_OUTPUT_FIELDS = [
    "token_address", "token_symbol", "chain",
    "signal_time", "paper_entry_price", "live_entry_price",
    "paper_exit_price", "live_exit_price",
    "paper_pnl_pct", "live_pnl_pct", "execution_drag_pct",
    "exit_reason", "route_used", "error_class",
    "divergence_source",
]

_TIME_WINDOW_SEC = 300  # trades within 5 min are considered the same token event


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _float(v, default=0.0) -> float:
    try:
        return float(v) if v not in (None, "", "None") else default
    except (ValueError, TypeError):
        return default


def _classify_divergence(row: dict) -> str:
    """Classify what caused the paper vs live performance gap."""
    entry_drag = abs(_float(row["paper_entry_price"]) - _float(row["live_entry_price"]))
    exit_drag  = abs(_float(row["paper_exit_price"])  - _float(row["live_exit_price"]))
    error_cls  = row.get("error_class", "")
    route      = row.get("route_used", "")

    if error_cls and error_cls not in ("", "None"):
        return "route_failure"
    if "fallback" in route.lower() or "jupiter" in route.lower():
        return "fallback"
    if entry_drag > exit_drag * 2:
        return "entry"
    if exit_drag > entry_drag * 2:
        return "exit"
    return "matched"


def run():
    live_rows  = _load_csv(_LIVE_JOURNAL)
    paper_rows = _load_csv(_PAPER_JOURNAL)
    route_rows = _load_csv(_ROUTE_ATTEMPTS)

    if not live_rows:
        print(f"No live trades found in {_LIVE_JOURNAL}")
        return

    # Index route attempts by token_mint for quick lookup
    route_by_mint: dict[str, dict] = {}
    for rr in route_rows:
        mint = rr.get("token_mint", "")
        if mint:
            route_by_mint[mint] = rr  # last attempt wins

    # Index paper trades by (token_address, approx_signal_time bucket)
    paper_by_token: dict[str, list[dict]] = {}
    for row in paper_rows:
        addr = row.get("token_address", "")
        if addr:
            paper_by_token.setdefault(addr, []).append(row)

    results = []

    for live in live_rows:
        addr        = live.get("token_address", "")
        symbol      = live.get("token_symbol", "")
        chain       = live.get("chain", "solana")
        live_sig    = _float(live.get("signal_time") or live.get("entry_time"), 0)
        live_entry  = _float(live.get("entry_price"))
        live_exit   = _float(live.get("exit_price"))
        exit_reason = live.get("exit_reason", "")

        live_pnl_pct = (
            (live_exit / live_entry - 1) * 100 if live_entry > 0 and live_exit > 0 else 0.0
        )

        # Find matching paper trade (same token, signal time within window)
        paper_match = None
        for paper in paper_by_token.get(addr, []):
            paper_sig = _float(paper.get("signal_time") or paper.get("entry_time"), 0)
            if abs(paper_sig - live_sig) <= _TIME_WINDOW_SEC:
                paper_match = paper
                break

        paper_entry = _float(paper_match.get("entry_price") if paper_match else 0)
        paper_exit  = _float(paper_match.get("exit_price")  if paper_match else 0)
        paper_pnl_pct = (
            (paper_exit / paper_entry - 1) * 100 if paper_entry > 0 and paper_exit > 0 else 0.0
        )

        execution_drag = live_pnl_pct - paper_pnl_pct

        route_info  = route_by_mint.get(addr, {})
        route_used  = route_info.get("route_name", route_info.get("route", ""))
        error_class = route_info.get("error_class", "")

        row = {
            "token_address":      addr,
            "token_symbol":       symbol,
            "chain":              chain,
            "signal_time":        live.get("signal_time", live.get("entry_time", "")),
            "paper_entry_price":  paper_entry,
            "live_entry_price":   live_entry,
            "paper_exit_price":   paper_exit,
            "live_exit_price":    live_exit,
            "paper_pnl_pct":      round(paper_pnl_pct, 2),
            "live_pnl_pct":       round(live_pnl_pct, 2),
            "execution_drag_pct": round(execution_drag, 2),
            "exit_reason":        exit_reason,
            "route_used":         route_used,
            "error_class":        error_class,
            "divergence_source":  "",
        }
        row["divergence_source"] = _classify_divergence(row)
        results.append(row)

    # Sort by execution_drag (worst first)
    results.sort(key=lambda r: _float(r["execution_drag_pct"]))

    # Print table
    print(f"\n{'Token':<12} {'Symbol':<12} {'Paper%':>8} {'Live%':>8} {'Drag%':>8} {'Divergence':<16} {'Route':<24} {'Error'}")
    print("-" * 110)
    for r in results:
        print(
            f"{r['token_address'][:10]:<12} {r['token_symbol'][:10]:<12}"
            f" {r['paper_pnl_pct']:>8.1f} {r['live_pnl_pct']:>8.1f}"
            f" {r['execution_drag_pct']:>8.1f} {r['divergence_source']:<16}"
            f" {str(r['route_used'])[:22]:<24} {r['error_class']}"
        )

    print(f"\nTotal: {len(results)} matched live trades")

    # Summary stats
    if results:
        avg_drag = sum(_float(r["execution_drag_pct"]) for r in results) / len(results)
        route_failures = sum(1 for r in results if r["divergence_source"] == "route_failure")
        fallbacks      = sum(1 for r in results if r["divergence_source"] == "fallback")
        print(f"Avg execution drag:  {avg_drag:.2f}%")
        print(f"Route failures:      {route_failures}")
        print(f"Fallback exits:      {fallbacks}")

    # Save CSV
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {_OUTPUT_CSV}")


if __name__ == "__main__":
    run()
