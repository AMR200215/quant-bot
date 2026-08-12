#!/usr/bin/env bash
# deploy/cron.d/install.sh — idempotent install of the wrapped cron
# entries onto the VPS. Symlinked (not copied), same reasoning as
# deploy/systemd/install.sh: `git pull` alone keeps /etc/cron.d in sync
# after this has been run once. This is the fix for the exact class of
# bug that caused K5/epoch/v8_vs_v7 to go silently dead for days: a
# previous ad-hoc heredoc install and a symlinked, git-tracked one behave
# very differently the next time someone edits the schedule and forgets
# to also touch the live file.
#
# Run from the repo root on the VPS:
#   sudo bash deploy/cron.d/install.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_DIR="$REPO_ROOT/deploy/cron.d"
TARGET_DIR="/etc/cron.d"

for f in "$SRC_DIR"/quantbot-*; do
    name="$(basename "$f")"
    ln -sf "$f" "$TARGET_DIR/$name"
    chmod 644 "$f"
    echo "linked $name"
done

systemctl restart cron

echo "--- post-install syslog check (should show no parser errors) ---"
sleep 1
journalctl -u cron --since "-30s" --no-pager | grep -i "quantbot\|error\|syntax" || echo "no errors"
