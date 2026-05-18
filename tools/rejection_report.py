"""
Daily cron script — sends new_launch rejection log summary to Telegram.
Run via cron on Hetzner:
    0 9 * * * cd /root/quant-bot && .venv/bin/python3 -m tools.rejection_report
"""

import csv
import os
import sys
from collections import Counter
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
LOG_FILE  = Path(__file__).parent.parent / "logs" / "new_launch_rejections.csv"


def send(text: str):
    if not BOT_TOKEN or not CHAT_ID:
        print("No Telegram credentials — printing to stdout instead.")
        print(text)
        return
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, json={
        "chat_id":    CHAT_ID,
        "text":       text,
        "parse_mode": "HTML",
    }, timeout=10)


def main():
    if not LOG_FILE.exists():
        send(
            "🔴 <b>NEW LAUNCH REJECTION REPORT</b>\n"
            "No rejections logged yet — file doesn't exist.\n"
            "Market may be slow or bot restarted recently."
        )
        return

    rows = []
    with open(LOG_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        send(
            "🔴 <b>NEW LAUNCH REJECTION REPORT</b>\n"
            "File exists but 0 rows — market is genuinely slow.\n"
            "5m filter is not the bottleneck."
        )
        return

    # 5m value distribution
    vals = []
    for r in rows:
        try:
            vals.append(float(r["price_change_5m"]))
        except:
            pass

    vals.sort()
    below_10  = sum(1 for v in vals if v < 10)
    below_15  = sum(1 for v in vals if 10 <= v < 15)
    below_20  = sum(1 for v in vals if 15 <= v < 20)
    near_miss = sum(1 for v in vals if 15 <= v < 20)  # closest to threshold

    avg_5m = sum(vals) / len(vals) if vals else 0
    median_5m = vals[len(vals)//2] if vals else 0

    # Dex breakdown
    dex_counts = Counter(r.get("dex_id", "unknown") for r in rows)
    dex_lines = "  ".join(f"{dex}:{cnt}" for dex, cnt in dex_counts.most_common(4))

    # Verdict
    if len(rows) < 5:
        verdict = "Market is slow — filter not the bottleneck."
    elif near_miss >= len(rows) * 0.25:
        verdict = f"⚠️ {near_miss} trades within 15-20% range — consider lowering threshold to 15%."
    else:
        verdict = "Filter is working as intended — most rejections well below threshold."

    msg = (
        f"🔴 <b>NEW LAUNCH REJECTION REPORT</b>\n"
        f"────────────────────\n"
        f"Total rejections: <b>{len(rows)}</b>\n"
        f"Avg 5m change:    <b>{avg_5m:.1f}%</b>\n"
        f"Median 5m change: <b>{median_5m:.1f}%</b>\n\n"
        f"<b>Distribution:</b>\n"
        f"  &lt;10%:   {below_10} trades\n"
        f"  10-15%:  {below_15} trades\n"
        f"  15-20%:  {below_20} trades  ← near-misses\n\n"
        f"<b>Dex breakdown:</b> {dex_lines}\n\n"
        f"<b>Verdict:</b> {verdict}"
    )
    send(msg)


if __name__ == "__main__":
    main()
