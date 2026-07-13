#!/bin/bash
# sync_journals.sh — append-only CSV sync to origin.
# Runs every 30 min via cron. Commits and pushes journal CSVs.
# Rebases on origin/main so scan commits never block the push.
# Writes logs/last_sync_ts on success (used by watchdog_sync.sh).

set -uo pipefail
cd /root/quant-bot || exit 1

LOG_PREFIX="$(date -u +%Y-%m-%dT%H:%M:%S)"
LAST_SYNC_FILE="logs/last_sync_ts"

git add logs/memecoin_live_journal.csv \
        logs/memecoin_social_journal.csv \
        logs/memecoin_journal.csv 2>/dev/null || true

if git diff --cached --quiet; then
    echo "${LOG_PREFIX} no changes"
    exit 0
fi

NOW=$(date -u +%Y-%m-%dT%H:%M)
git commit -m "chore: sync journals ${NOW}" \
    --author="quantbot-sync <noreply@quantbot>" -q

# Rebase on top of any origin changes (scan commits, fix deploys, etc.)
# Append-only CSVs never conflict on rebase — new lines just sit on top.
git fetch origin main -q
if ! git rebase origin/main -q 2>/tmp/sync_rebase.log; then
    git rebase --abort 2>/dev/null || true
    echo "${LOG_PREFIX} ERROR: rebase failed — $(head -3 /tmp/sync_rebase.log)"
    exit 1
fi

if git push origin main -q 2>&1; then
    date +%s > "${LAST_SYNC_FILE}"
    echo "${LOG_PREFIX} sync OK"
else
    echo "${LOG_PREFIX} ERROR: push failed"
    exit 1
fi
