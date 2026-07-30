"""
research/scripts/v8_vs_v7_daily.py — N6 (V8-READINESS batch): daily v8_paper vs v7_paper comparison.

Run daily by cron. Reads both paper journals, computes today's (UTC) trade
count + net PnL% for each, and appends one row to a dedicated table in
docs/RECEIPTS.md under "### N6 — V8 paper twin: daily v8 vs v7 comparison".
Idempotent — re-running for a date whose row already exists updates that row
in place rather than duplicating it.

v7 comparison book: logs/memecoin_social_journal.csv (social_alert paper —
the same signal cohort v8_paper's gate draws from).
v8 comparison book: logs/memecoin_v8_journal.csv (memecoin/v8_paper.py).

Usage:
  python -m research.scripts.v8_vs_v7_daily
  python research/scripts/v8_vs_v7_daily.py
"""

import csv
import re
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT   = Path(__file__).parent.parent.parent
_V7_JOURNAL  = _REPO_ROOT / "logs" / "memecoin_social_journal.csv"
_V8_JOURNAL  = _REPO_ROOT / "logs" / "memecoin_v8_journal.csv"
_RECEIPTS    = _REPO_ROOT / "docs" / "RECEIPTS.md"

_SECTION_HEADER = "### N6 — V8 paper twin: daily v8 vs v7 comparison"
_TABLE_HEADER   = (
    "| date | v7 trades | v7 pnl% (mean) | v8 trades | v8 pnl% (mean) | v8 gate |\n"
    "|---|---|---|---|---|---|"
)


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _day_stats(journal_path: Path, day: str) -> tuple[int, float]:
    """Return (trade_count, mean_pnl_pct) for positions closed on `day` (UTC)."""
    if not journal_path.exists():
        return 0, 0.0
    n, total_pct = 0, 0.0
    with open(journal_path, newline="") as f:
        for row in csv.DictReader(f):
            exit_time = (row.get("exit_time") or "").strip()
            if not exit_time.startswith(day):
                continue
            n += 1
            try:
                total_pct += float(row.get("pnl_pct") or 0)
            except (ValueError, TypeError):
                pass
    mean_pct = total_pct / n if n else 0.0
    return n, mean_pct


def _upsert_row(text: str, day: str, row_line: str) -> str:
    """Insert row_line into the N6 table, replacing any existing row for `day`."""
    if _SECTION_HEADER not in text:
        block = f"\n{_SECTION_HEADER}\n\n{_TABLE_HEADER}\n{row_line}\n"
        return text.rstrip("\n") + "\n" + block

    # Section exists — find its table and either replace today's row or append it.
    section_start = text.index(_SECTION_HEADER)
    # Find the table header within this section
    rest = text[section_start:]
    if _TABLE_HEADER not in rest:
        # Section exists without a table yet (shouldn't normally happen) — add one.
        insert_at = section_start + len(_SECTION_HEADER)
        return text[:insert_at] + f"\n\n{_TABLE_HEADER}\n{row_line}\n" + text[insert_at:]

    table_start = section_start + rest.index(_TABLE_HEADER) + len(_TABLE_HEADER)
    # Existing rows run until the next blank line or next "###"/"##" header.
    after = text[table_start:]
    m = re.search(r"\n(?=\n|##|\Z)", after)
    rows_block_end = table_start + (m.start() if m else len(after))
    rows_block = text[table_start:rows_block_end]

    existing_lines = [l for l in rows_block.split("\n") if l.strip().startswith("|")]
    existing_lines = [l for l in existing_lines if not l.strip().startswith(f"| {day} ")]
    existing_lines.append(row_line)

    new_rows_block = "\n" + "\n".join(existing_lines)
    return text[:table_start] + new_rows_block + text[rows_block_end:]


def run():
    day = _today_utc()
    v7_n, v7_pct = _day_stats(_V7_JOURNAL, day)
    v8_n, v8_pct = _day_stats(_V8_JOURNAL, day)

    row_line = (
        f"| {day} | {v7_n} | {v7_pct*100:+.1f}% | {v8_n} | {v8_pct*100:+.1f}% | "
        f"progress<70+no-dex (smart-money offline-only) |"
    )

    if not _RECEIPTS.exists():
        print(f"RECEIPTS.md not found at {_RECEIPTS} — nothing to update.")
        return

    text = _RECEIPTS.read_text()
    new_text = _upsert_row(text, day, row_line)
    _RECEIPTS.write_text(new_text)

    print(f"[v8_vs_v7_daily] {day}  v7: n={v7_n} mean_pnl={v7_pct*100:+.1f}%  "
          f"v8: n={v8_n} mean_pnl={v8_pct*100:+.1f}%")


if __name__ == "__main__":
    run()
