#!/usr/bin/env bash
# remove_vps_sync_commit.sh
#
# Run on the VPS to find and disable the cron/service that runs
# "git commit" on journal files.  This script is read-only by default;
# it audits and tells you what to remove.  Pass --apply to disable.
#
# Usage (from dev machine):
#   ! ssh root@178.105.94.113 'bash -s' < scripts/remove_vps_sync_commit.sh
#   ! ssh root@178.105.94.113 'bash -s -- --apply' < scripts/remove_vps_sync_commit.sh

set -euo pipefail

APPLY=false
[[ "${1:-}" == "--apply" ]] && APPLY=true

echo "=== VPS Journal Sync Audit ==="
echo ""

echo "--- crontab (root) ---"
crontab -l 2>/dev/null || echo "(no crontab)"

echo ""
echo "--- crontab (quantbot user, if exists) ---"
crontab -u quantbot -l 2>/dev/null || echo "(no quantbot crontab)"

echo ""
echo "--- /etc/cron.d/ entries referencing git commit ---"
grep -rl "git commit" /etc/cron.d/ 2>/dev/null || echo "(none)"

echo ""
echo "--- systemd timers ---"
systemctl list-timers --all 2>/dev/null | grep -i "sync\|journal\|quant" || echo "(none matching)"

echo ""
echo "--- scripts in repo dir containing 'git commit' ---"
REPO_DIR="${REPO_DIR:-/root/quant-bot}"
grep -rl "git commit" "$REPO_DIR" 2>/dev/null \
    | grep -v ".venv\|__pycache__\|.git" \
    || echo "(none)"

echo ""

if $APPLY; then
    echo "=== Applying: removing git commit from crontab ==="

    # Back up current crontab
    crontab -l 2>/dev/null > /tmp/crontab.bak && echo "Backed up to /tmp/crontab.bak"

    # Remove lines containing 'git commit' from crontab
    BEFORE=$(crontab -l 2>/dev/null | wc -l)
    crontab -l 2>/dev/null | grep -v "git commit" | crontab - || true
    AFTER=$(crontab -l 2>/dev/null | wc -l)
    echo "Crontab lines: $BEFORE → $AFTER"

    # Also check quantbot user
    if id quantbot &>/dev/null; then
        crontab -u quantbot -l 2>/dev/null > /tmp/crontab_quantbot.bak || true
        crontab -u quantbot -l 2>/dev/null | grep -v "git commit" | crontab -u quantbot - || true
        echo "quantbot crontab updated"
    fi

    echo ""
    echo "=== Verify: no git commit in crontab ==="
    crontab -l 2>/dev/null | grep "git commit" && echo "WARNING: entries remain!" || echo "OK — no git commit entries"
else
    echo "=== Dry run — re-run with --apply to remove ==="
    echo "Entries that WOULD be removed from crontab:"
    crontab -l 2>/dev/null | grep "git commit" || echo "(none found in crontab)"
fi

echo ""
echo "Deploy instructions after disabling sync:"
echo "  1. On dev: git push origin main"
echo "  2. On VPS: cd /root/quant-bot && git pull --rebase origin main"
echo "  3. On VPS: systemctl restart quantbot"
