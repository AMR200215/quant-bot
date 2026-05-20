"""
Daily cron script — sends new_launch funnel report to Telegram.
Run via cron on Hetzner:
    0 9 * * * cd /root/quant-bot && .venv/bin/python3 -m tools.rejection_report
"""

import csv
import json
import os
from collections import Counter
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
LOG_FILE  = Path(__file__).parent.parent / "logs" / "new_launch_rejections.csv"
NEAR_MISS_FILE = Path(__file__).parent.parent / "memecoin" / "data" / "near_miss_tracking.json"


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


# ── Stage labels for the Telegram message ────────────────────────────────────
_STAGE_LABELS = {
    "low_liquidity":         "Low liquidity",
    "mcap_too_high":         "Mcap too high",
    "rugcheck_fail":         "Rugcheck fail",
    "honeypot":              "Honeypot",
    "rug_detector":          "Rug detector",
    "age_too_old":           "Age > 60 min",
    "meteora_block":         "Meteora dex",
    "5m_momentum_below_20":  "5m < 20% (momentum)",
}

# Screener-level stages (happen before 5m filter)
_SCREENER_STAGES = {"low_liquidity", "mcap_too_high", "rugcheck_fail",
                    "honeypot", "rug_detector"}


def _label(reason: str) -> str:
    for key, label in _STAGE_LABELS.items():
        if reason.startswith(key):
            return label
    return reason


def main():
    if not LOG_FILE.exists():
        send(
            "🔴 <b>NEW LAUNCH FUNNEL REPORT</b>\n"
            "No rejections logged yet — bot may have just started.\n"
            "Market may be slow or filters haven't fired."
        )
        return

    rows = []
    with open(LOG_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        send(
            "🔴 <b>NEW LAUNCH FUNNEL REPORT</b>\n"
            "File exists but 0 rows — market is genuinely slow."
        )
        return

    total = len(rows)

    # ── Funnel breakdown ──────────────────────────────────────────────────────
    reason_counts = Counter()
    for r in rows:
        raw = r.get("rejection_reason", "unknown")
        # normalise: "low_liquidity:1234" → "low_liquidity"
        stage = raw.split(":")[0]
        reason_counts[stage] += 1

    screener_total = sum(v for k, v in reason_counts.items() if k in _SCREENER_STAGES)
    age_total      = reason_counts.get("age_too_old", 0)
    meteora_total  = reason_counts.get("meteora_block", 0)
    momentum_total = reason_counts.get("5m_momentum_below_20", 0)
    passed_filter  = total - screener_total - age_total - meteora_total - momentum_total

    funnel_lines = []
    for stage, label in _STAGE_LABELS.items():
        cnt = sum(v for k, v in reason_counts.items() if k == stage)
        if cnt:
            funnel_lines.append(f"  {label}: {cnt}")

    # ── 5m distribution (momentum-rejected only) ──────────────────────────────
    vals = []
    for r in rows:
        if r.get("rejection_reason", "").startswith("5m_momentum"):
            try:
                vals.append(float(r["price_change_5m"]))
            except (KeyError, ValueError):
                pass

    near_miss_count = sum(1 for v in vals if 15 <= v < 20)
    avg_5m    = sum(vals) / len(vals) if vals else 0
    median_5m = sorted(vals)[len(vals) // 2] if vals else 0

    # ── Near-miss outcomes ────────────────────────────────────────────────────
    nm_outcomes = {"profitable": 0, "dump": 0, "flat": 0, "pending": 0}
    nm_data = {}
    if NEAR_MISS_FILE.exists():
        try:
            nm_data = json.loads(NEAR_MISS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            nm_data = {}

    for entry in nm_data.values():
        outcome = entry.get("outcome")
        if outcome in nm_outcomes:
            nm_outcomes[outcome] += 1
        elif not entry.get("check_6h_done"):
            nm_outcomes["pending"] += 1

    nm_total_resolved = nm_outcomes["profitable"] + nm_outcomes["dump"] + nm_outcomes["flat"]

    # ── Verdict ───────────────────────────────────────────────────────────────
    if total < 5:
        verdict = "Too few rejections — market is slow. No action needed."
    elif screener_total > momentum_total:
        verdict = (
            f"Screener is the main bottleneck ({screener_total} rejections). "
            "5m filter is NOT the problem — tokens are failing safety checks first."
        )
    elif nm_total_resolved >= 5 and nm_outcomes["profitable"] / nm_total_resolved >= 0.4:
        pct = round(nm_outcomes["profitable"] / nm_total_resolved * 100)
        verdict = (
            f"⚠️ {pct}% of near-misses (15-19%) pumped 50%+ after rejection. "
            "Consider lowering 5m threshold to 15%."
        )
    elif nm_total_resolved >= 5 and nm_outcomes["dump"] / nm_total_resolved >= 0.6:
        pct = round(nm_outcomes["dump"] / nm_total_resolved * 100)
        verdict = (
            f"✅ Data confirms filter is correct — {pct}% of near-misses dumped within 6h. "
            "Keep threshold at 20%."
        )
    elif near_miss_count >= momentum_total * 0.25:
        verdict = (
            f"⚠️ {near_miss_count} near-misses ({near_miss_count/max(momentum_total,1)*100:.0f}% of 5m rejects). "
            "Wait for 6h outcome data before deciding on threshold."
        )
    else:
        verdict = "Filter is working as intended — most rejections well below threshold."

    # ── Near-miss outcome lines ───────────────────────────────────────────────
    nm_lines = ""
    if nm_total_resolved > 0 or nm_outcomes["pending"] > 0:
        nm_lines = (
            f"\n<b>Near-miss outcomes (15-19% 5m):</b>\n"
            f"  Pumped 50%+:  {nm_outcomes['profitable']} → filter was too tight\n"
            f"  Dumped:       {nm_outcomes['dump']} → filter was correct\n"
            f"  Flat:         {nm_outcomes['flat']} → borderline\n"
            f"  Pending 6h:   {nm_outcomes['pending']}\n"
        )
    elif near_miss_count > 0:
        nm_lines = (
            f"\n<b>Near-miss tracking:</b> {near_miss_count} tokens being tracked. "
            "Outcomes available after 6h.\n"
        )

    msg = (
        f"🔴 <b>NEW LAUNCH FUNNEL REPORT</b>\n"
        f"────────────────────\n"
        f"<b>Total rejected: {total}</b>\n\n"
        f"<b>Funnel breakdown:</b>\n"
        + "\n".join(funnel_lines) +
        f"\n\n<b>5m momentum rejects ({momentum_total}):</b>\n"
        f"  Avg 5m:    {avg_5m:.1f}%\n"
        f"  Median 5m: {median_5m:.1f}%\n"
        f"  Near-miss (15-19%): {near_miss_count}\n"
        + nm_lines +
        f"\n<b>Verdict:</b> {verdict}"
    )
    send(msg)


if __name__ == "__main__":
    main()
