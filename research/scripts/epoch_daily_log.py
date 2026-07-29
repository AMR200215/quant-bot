"""
research/scripts/epoch_daily_log.py — Append one epoch-day record to logs/epoch_daily.jsonl.

Run daily by cron at 23:55 UTC.  Reads the live journal CSV, computes:
  - trades today (by exit_time date)
  - net PnL USD today
  - currently open positions (no exit_time)

Output line format (JSON, one per day, append-only):
  {"date": "YYYY-MM-DD", "trades": N, "pnl_usd": X.XX, "open": N}

Usage:
  python -m research.scripts.epoch_daily_log
  python research/scripts/epoch_daily_log.py
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT   = Path(__file__).parent.parent.parent
_JOURNAL     = _REPO_ROOT / "logs" / "memecoin_live_journal.csv"
_OUTPUT      = _REPO_ROOT / "logs" / "epoch_daily.jsonl"


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def run():
    today = _today_utc()

    if not _JOURNAL.exists():
        print(f"Journal not found: {_JOURNAL}")
        return

    trades_today = 0
    pnl_today    = 0.0
    open_count   = 0

    with open(_JOURNAL, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            exit_time = (row.get("exit_time") or "").strip()
            pnl_str   = (row.get("pnl_usd") or "").strip()

            if not exit_time:
                open_count += 1
                continue

            # Closed today
            if exit_time.startswith(today):
                trades_today += 1
                try:
                    pnl_today += float(pnl_str)
                except (ValueError, TypeError):
                    pass

    record = {
        "date":     today,
        "trades":   trades_today,
        "pnl_usd":  round(pnl_today, 4),
        "open":     open_count,
    }

    with open(_OUTPUT, "a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"[epoch_daily_log] {today}  trades={trades_today}  pnl=${pnl_today:.2f}  open={open_count}")


if __name__ == "__main__":
    run()
