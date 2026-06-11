#!/usr/bin/env python3
"""
tools/v7_report.py — canonical V7 scoreboard.

This is the SINGLE SOURCE OF TRUTH for all PnL verdicts.
No ad-hoc summations. Every future "how are we doing?" comes from this script.

Reads:
  logs/v7_journal_corrected.csv   — closed trades with corrected PnL + epochs
  memecoin/data/memecoin_positions.json — open positions (unrealized section)

Filters:
  config_tag starts with "v7"

Output:
  - Closed trades split by: epoch × paper/live
  - Gross PnL and net PnL (after 3.4% round-trip fee per trade)
  - Win rate, avg PnL/trade, best/worst trade
  - Separate UNREALIZED section for open positions

Usage:
  python tools/v7_report.py [--live-only] [--paper-only] [--epoch e3_pp_entries_anchored_stops]
"""

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT            = Path(__file__).parent.parent
CORRECTED_FILE  = ROOT / "logs" / "v7_journal_corrected.csv"
POSITIONS_FILE  = ROOT / "memecoin" / "data" / "memecoin_positions.json"

FEE_RT = 0.034   # 3.4% round-trip (1.7% buy + 1.7% sell)

EPOCHS_ORDER = [
    "e1_baseline",
    "e2_pp_exits",
    "e3_pp_entries_anchored_stops",
]

EPOCH_LABELS = {
    "e1_baseline":                    "e1  baseline           (pre PP exits)",
    "e2_pp_exits":                    "e2  PP exits wired     (commit 81de8da)",
    "e3_pp_entries_anchored_stops":   "e3  anchored stops     (commit 9a2a332, current)",
    "":                               "e?  unknown epoch",
}


def _is_live(row: dict) -> bool:
    return "live|tx:" in (row.get("notes") or "")


def _net_pnl(pnl_usd: float, size_usd: float) -> float:
    return pnl_usd - size_usd * FEE_RT


def _fmt_sign(v: float) -> str:
    return f"+${v:.2f}" if v >= 0 else f"-${abs(v):.2f}"


def _stats_block(rows: list[dict]) -> dict:
    if not rows:
        return {}
    n = len(rows)
    wins = sum(1 for r in rows if float(r.get("pnl_usd") or 0) > 0)
    gross = sum(float(r.get("pnl_usd") or 0) for r in rows)
    net   = sum(_net_pnl(float(r.get("pnl_usd") or 0), float(r.get("size_usd") or 0)) for r in rows)
    best  = max(rows, key=lambda r: float(r.get("pnl_usd") or 0))
    worst = min(rows, key=lambda r: float(r.get("pnl_usd") or 0))
    return {
        "n": n,
        "win_rate": wins / n * 100,
        "gross": gross,
        "net": net,
        "avg_gross": gross / n,
        "avg_net": net / n,
        "best_sym":  best.get("token_symbol", "?"),
        "best_pnl":  float(best.get("pnl_usd") or 0),
        "worst_sym": worst.get("token_symbol", "?"),
        "worst_pnl": float(worst.get("pnl_usd") or 0),
    }


def _print_stats(label: str, rows: list[dict]):
    s = _stats_block(rows)
    if not s:
        print(f"  {label}: (no trades)")
        return
    print(f"  {label}: {s['n']} trades  WR {s['win_rate']:.0f}%  "
          f"gross {_fmt_sign(s['gross'])}  net {_fmt_sign(s['net'])}  "
          f"avg/trade {_fmt_sign(s['avg_net'])}  "
          f"best {s['best_sym']} {_fmt_sign(s['best_pnl'])}  "
          f"worst {s['worst_sym']} {_fmt_sign(s['worst_pnl'])}")


def run(args: list[str]):
    live_only  = "--live-only"  in args
    paper_only = "--paper-only" in args
    epoch_filter = None
    if "--epoch" in args:
        idx = args.index("--epoch")
        if idx + 1 < len(args):
            epoch_filter = args[idx + 1]

    if not CORRECTED_FILE.exists():
        print(f"[ERROR] {CORRECTED_FILE} not found.")
        print("  Run:  python tools/v7_journal_corrected.py")
        sys.exit(1)

    with open(CORRECTED_FILE, newline="") as f:
        all_rows = list(csv.DictReader(f))

    # Filter to v7 only, closed only
    v7_rows = [
        r for r in all_rows
        if r.get("config_tag", "").startswith("v7")
        and r.get("exit_reason")   # closed trades have an exit_reason
    ]

    if epoch_filter:
        v7_rows = [r for r in v7_rows if r.get("accounting_epoch") == epoch_filter]
    if live_only:
        v7_rows = [r for r in v7_rows if _is_live(r)]
    if paper_only:
        v7_rows = [r for r in v7_rows if not _is_live(r)]

    live_rows  = [r for r in v7_rows if _is_live(r)]
    paper_rows = [r for r in v7_rows if not _is_live(r)]

    # Group by epoch
    by_epoch: dict[str, list[dict]] = defaultdict(list)
    for r in v7_rows:
        by_epoch[r.get("accounting_epoch") or ""].append(r)

    width = 78
    print("=" * width)
    print("  V7 SCOREBOARD  —  canonical, from v7_journal_corrected.csv")
    print(f"  Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    if epoch_filter:
        print(f"  Epoch filter: {epoch_filter}")
    if live_only:
        print("  Mode: LIVE only")
    elif paper_only:
        print("  Mode: PAPER only")
    print("=" * width)

    # ── Overall ──────────────────────────────────────────────────────────────
    print()
    print("  OVERALL (all epochs)")
    print("-" * width)
    _print_stats("PAPER", paper_rows)
    _print_stats("LIVE ", live_rows)
    _print_stats("ALL  ", v7_rows)

    # ── Per epoch ────────────────────────────────────────────────────────────
    print()
    print("  BY EPOCH")
    print("-" * width)
    for epoch in EPOCHS_ORDER:
        rows = by_epoch.get(epoch, [])
        if not rows:
            continue
        label = EPOCH_LABELS.get(epoch, epoch)
        print(f"\n  [{label}]")
        e_live  = [r for r in rows if _is_live(r)]
        e_paper = [r for r in rows if not _is_live(r)]
        _print_stats("  paper", e_paper)
        _print_stats("  live ", e_live)

    # Handle unknown epoch
    unknown = by_epoch.get("", [])
    if unknown:
        print(f"\n  [e? unknown epoch]")
        _print_stats("  all  ", unknown)

    # ── Exit reason breakdown ────────────────────────────────────────────────
    print()
    print("  EXIT REASONS  (v7 closed, all epochs)")
    print("-" * width)
    reason_map: dict[str, list] = defaultdict(list)
    for r in v7_rows:
        reason_map[r.get("exit_reason") or "unknown"].append(r)
    for reason, rows in sorted(reason_map.items(), key=lambda x: -len(x[1])):
        s = _stats_block(rows)
        print(f"  {reason:<22} n={s['n']:<4} WR={s['win_rate']:.0f}%  "
              f"gross {_fmt_sign(s['gross'])}  net {_fmt_sign(s['net'])}")

    # ── UNREALIZED (open positions) ──────────────────────────────────────────
    print()
    print("  UNREALIZED (open positions)")
    print("-" * width)
    if not POSITIONS_FILE.exists():
        print("  (positions file not found)")
    else:
        try:
            positions = json.loads(POSITIONS_FILE.read_text())
        except Exception as e:
            print(f"  (error reading positions: {e})")
            positions = []

        open_pos = [p for p in positions if p.get("status") == "open"]
        if not open_pos:
            print("  (no open positions)")
        else:
            for p in open_pos:
                ep  = p.get("entry_price") or 0
                cp  = p.get("current_price") or 0
                rf  = float(p.get("remaining_fraction") or 1.0)
                sz  = float(p.get("size_usd") or 0)
                rz  = float(p.get("realized_pnl_usd") or 0)
                pnl_pct = (cp - ep) / ep if ep > 0 else 0
                unreal   = pnl_pct * sz * rf
                total    = rz + unreal
                mode     = "LIVE " if "live|tx:" in (p.get("notes") or "") else "paper"
                ep_tag   = ACCOUNTING_EPOCH_SHORT.get(p.get("accounting_epoch") or "", "")
                print(
                    f"  [{mode}] {p.get('token_symbol','?'):<12} "
                    f"entry=${ep:.8f}  cur=${cp:.8f}  "
                    f"pnl%={pnl_pct*100:+.1f}%  "
                    f"realized=${rz:.2f}  unrealized={_fmt_sign(unreal)}  "
                    f"total={_fmt_sign(total)}  rf={rf:.0%}"
                )

    print()
    print("=" * width)
    print("  Fee assumption: 3.4% round-trip (1.7% each way)")
    print("  Source: logs/v7_journal_corrected.csv  (DO NOT edit directly)")
    print("=" * width)


# short labels for open position display
ACCOUNTING_EPOCH_SHORT = {
    "e1_baseline":                  "e1",
    "e2_pp_exits":                  "e2",
    "e3_pp_entries_anchored_stops": "e3",
}

if __name__ == "__main__":
    run(sys.argv[1:])
