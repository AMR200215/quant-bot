#!/bin/bash
# watchdog_sync.sh — alert if journal sync is stale >90 min.
# Runs every 15 min via cron (*/15 * * * *).
# F2: journal staleness alarm.

LAST_SYNC_FILE="/root/quant-bot/logs/last_sync_ts"
MAX_AGE_SEC=5400   # 90 min

if [ ! -f "${LAST_SYNC_FILE}" ]; then
    AGE_DESC="unknown (no sync record found)"
    STALE=1
else
    LAST=$(cat "${LAST_SYNC_FILE}" 2>/dev/null || echo "0")
    NOW=$(date +%s)
    AGE_SEC=$((NOW - LAST))
    AGE_MIN=$((AGE_SEC / 60))
    AGE_DESC="${AGE_MIN}min ago"
    if [ "${AGE_SEC}" -gt "${MAX_AGE_SEC}" ]; then
        STALE=1
    else
        STALE=0
    fi
fi

if [ "${STALE}" = "1" ]; then
    # Load .env for TG credentials
    set -a
    source /root/quant-bot/.env 2>/dev/null || true
    set +a

    MSG="⚠️ JOURNAL SYNC STALE: last push ${AGE_DESC} (threshold 90min). Journal data may be lost on next deploy. Check /root/quant-bot/logs/journal_sync.log"

    if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            -d "chat_id=${TELEGRAM_CHAT_ID}" \
            -d "text=${MSG}" > /dev/null
        echo "$(date -u +%Y-%m-%dT%H:%M:%S) STALE ALERT sent: ${AGE_DESC}"
    else
        echo "$(date -u +%Y-%m-%dT%H:%M:%S) STALE (no TG creds): ${MSG}"
    fi
fi
