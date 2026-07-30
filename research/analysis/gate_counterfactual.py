"""
gate_counterfactual.py — L5 (post-measurement batch, 2026-07-10)

Joins logs/gate_blocks.csv (tokens blocked by live-entry gates) against
research_tokens outcome-poll data (real tick-level peak tracking, not the
synthetic T+1h DexScreener simulation gate_logger.generate_gate_report()
uses) to answer, per gate: how many tokens did it block, and what fraction
of those would have peaked >=+30% / +50% / +100%, and what was the median
peak.

Analysis only — makes no live trading changes. Output is V8 filter-tuning
input: a gate with a high missed-upside rate and low blocked-count value is
a tightening/removal candidate; a gate whose blocked tokens mostly dumped is
paying for itself.

Usage:
    python -m research.analysis.gate_counterfactual
    python -m research.analysis.gate_counterfactual --output gate_cf.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Optional

from memecoin.gate_logger import GATE_LOG_FILE
from research.config import SUPABASE_URL, SUPABASE_KEY

_THRESHOLDS = (30, 50, 100)


def _load_gate_blocks(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _fetch_research_tokens(sb) -> dict[str, dict]:
    """Paginated fetch of research_tokens, keyed by token_address (last row wins)."""
    by_addr: dict[str, dict] = {}
    offset, batch = 0, 1000
    while True:
        chunk = (
            sb.table("research_tokens")
            .select("token_address,pct_change_peak,pct_change_peak_3m,outcome_complete")
            .range(offset, offset + batch - 1)
            .execute()
            .data
        ) or []
        for row in chunk:
            addr = row.get("token_address")
            if addr:
                by_addr[addr] = row
        if len(chunk) < batch:
            break
        offset += batch
    return by_addr


def _peak_pct(row: dict) -> Optional[float]:
    """Prefer tick-level peak (price_peak_3m-derived) over poll-based peak."""
    p = row.get("pct_change_peak_3m")
    if p is None:
        p = row.get("pct_change_peak")
    return p


def build_table(blocks: list[dict], outcomes: dict[str, dict]) -> dict[str, dict]:
    """Return {gate: {blocked, priced, median_peak, >=30%, >=50%, >=100%}}."""
    by_gate: dict[str, list[dict]] = defaultdict(list)
    for b in blocks:
        by_gate[b.get("gate", "unknown")].append(b)

    table: dict[str, dict] = {}
    for gate, rows in by_gate.items():
        peaks = []
        for b in rows:
            addr = b.get("token_address")
            outcome = outcomes.get(addr)
            if not outcome or not outcome.get("outcome_complete"):
                continue
            p = _peak_pct(outcome)
            if p is not None:
                peaks.append(p)

        entry = {
            "blocked": len(rows),
            "priced":  len(peaks),
            "median_peak_pct": round(median(peaks), 1) if peaks else None,
        }
        for th in _THRESHOLDS:
            entry[f">={th}%"] = (
                round(sum(1 for p in peaks if p >= th) / len(peaks) * 100, 1)
                if peaks else None
            )
        table[gate] = entry
    return table


def print_table(table: dict[str, dict]) -> None:
    header = f"{'gate':<20}{'blocked':>8}{'priced':>8}{'median_peak%':>14}"
    for th in _THRESHOLDS:
        header += f"{'>=' + str(th) + '%':>10}"
    print(header)
    print("-" * len(header))
    for gate in sorted(table, key=lambda g: -table[g]["blocked"]):
        e = table[gate]
        line = f"{gate:<20}{e['blocked']:>8}{e['priced']:>8}"
        med = e["median_peak_pct"]
        med_str = f"{med:+.1f}" if med is not None else "n/a"
        line += f"{med_str:>14}"
        for th in _THRESHOLDS:
            v = e[f">={th}%"]
            v_str = f"{v:.1f}" if v is not None else "n/a"
            line += f"{v_str:>10}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Gate counterfactual — per-filter missed-upside table")
    parser.add_argument("--output", metavar="FILE", help="Write the per-filter table as CSV")
    args = parser.parse_args()

    blocks = _load_gate_blocks(GATE_LOG_FILE)
    if not blocks:
        print(f"No gate blocks found at {GATE_LOG_FILE} — nothing to analyze.")
        return

    try:
        from supabase import create_client
    except ImportError:
        print("supabase-py not installed — run: pip install supabase")
        sys.exit(1)
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("SUPABASE_URL / SUPABASE_KEY not set in .env")
        sys.exit(1)

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    print(f"Loaded {len(blocks)} gate_blocks rows; fetching research_tokens outcomes…")
    outcomes = _fetch_research_tokens(sb)
    print(f"  {len(outcomes)} research_tokens rows loaded\n")

    table = build_table(blocks, outcomes)
    print_table(table)

    if args.output:
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gate", "blocked", "priced", "median_peak_pct"]
                       + [f">={th}%" for th in _THRESHOLDS])
            for gate, e in table.items():
                w.writerow([gate, e["blocked"], e["priced"], e["median_peak_pct"]]
                           + [e[f">={th}%"] for th in _THRESHOLDS])
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
