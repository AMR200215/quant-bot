#!/usr/bin/env bash
# deploy/systemd/install.sh — idempotent install of the watchdog systemd
# timers on the VPS. Symlinks (not copies) so `git pull` alone keeps the
# live unit files in sync without needing to re-run this after every edit
# -- re-run only when a *new* unit file is added.
#
# Run from the repo root on the VPS:
#   sudo bash deploy/systemd/install.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIT_DIR="$REPO_ROOT/deploy/systemd"
TARGET_DIR="/etc/systemd/system"

for f in "$UNIT_DIR"/quantbot-watchdog-*.service "$UNIT_DIR"/quantbot-watchdog-*.timer; do
    name="$(basename "$f")"
    ln -sf "$f" "$TARGET_DIR/$name"
    echo "linked $name"
done

systemctl daemon-reload
systemctl enable --now quantbot-watchdog-fast.timer
systemctl enable --now quantbot-watchdog-slow.timer

echo "--- timers ---"
systemctl list-timers 'quantbot-watchdog-*' --no-pager
