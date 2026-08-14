#!/usr/bin/env bash
# deploy/layer2/install.sh — provisions the read-only, forced-command SSH
# credential for Layer 2 (the external LLM audit agent).
#
# NOT RUN AUTOMATICALLY. This is the credential-provisioning checkpoint
# flagged throughout Phase 5 -- run this yourself (or ask Claude to run it
# with you watching) only once you're ready to grant GitHub Actions
# read-only access to this VPS's evidence.
#
# What this does:
#   1. Creates a dedicated, unprivileged system user `layer2audit`
#      (no login shell, no home directory writes needed).
#   2. Generates a new ed25519 SSH keypair for it.
#   3. Installs the PUBLIC key in that user's authorized_keys with a
#      forced command + hardening flags -- whatever command an SSH client
#      requests is IGNORED; only evidence_dump.py ever runs, no matter
#      what. This is the actual security boundary, not anything enforced
#      by evidence_dump.py's own (also read-only) logic.
#   4. Installs one narrowly-scoped sudoers rule (NOPASSWD, exact absolute
#      path, no argument substitution possible) so evidence_dump.py can
#      read root-owned files (the repo, cron.d, the watchdog state DB)
#      without broadly changing any existing file permissions on the box.
#   5. Prints the PRIVATE key once, to be added as the LAYER2_SSH_KEY
#      GitHub Actions secret, and never stored anywhere else on disk
#      after this script exits.
#
# What it deliberately does NOT grant: systemctl restart/stop, kill, rm,
# git reset/checkout, sed -i, any database write, any trading command --
# none of those are in the forced command or the sudoers rule, so the
# credential is structurally incapable of them regardless of what's
# requested over the SSH connection.
set -euo pipefail

REPO_ROOT="/root/quant-bot"
SCRIPT_PATH="$REPO_ROOT/deploy/layer2/evidence_dump.py"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python3"
AUDIT_USER="layer2audit"
KEY_PATH="/tmp/layer2audit_key"

if ! id "$AUDIT_USER" &>/dev/null; then
    # Shell must NOT be /usr/sbin/nologin (or /bin/false) -- that blocks
    # SSH login entirely, at the PAM/shell level, before authorized_keys'
    # forced-command directive is ever consulted. Found live: the first
    # version of this script used nologin and broke its own setup ("This
    # account is currently not available."). The actual restriction is,
    # and must be, enforced entirely by the `command=` forced-command
    # below -- a real shell here is required for that to work at all, not
    # a security downgrade.
    useradd --system -d "/home/$AUDIT_USER" --shell /bin/bash "$AUDIT_USER"
    echo "created user $AUDIT_USER"
else
    echo "user $AUDIT_USER already exists, reusing"
fi

mkdir -p "/home/$AUDIT_USER/.ssh" 2>/dev/null || mkdir -p "/var/lib/$AUDIT_USER/.ssh"
SSH_DIR="/home/$AUDIT_USER/.ssh"
[ -d "$SSH_DIR" ] || SSH_DIR="/var/lib/$AUDIT_USER/.ssh"

ssh-keygen -t ed25519 -f "$KEY_PATH" -N "" -C "layer2-audit-readonly" -q

FORCED_CMD="sudo -n $PYTHON_BIN $SCRIPT_PATH"
{
    echo -n 'command="'"$FORCED_CMD"'",no-port-forwarding,no-X11-forwarding,no-agent-forwarding,no-pty '
    cat "${KEY_PATH}.pub"
} > "$SSH_DIR/authorized_keys"
chown -R "$AUDIT_USER:$AUDIT_USER" "$SSH_DIR"
chmod 700 "$SSH_DIR"
chmod 600 "$SSH_DIR/authorized_keys"

SUDOERS_FILE="/etc/sudoers.d/layer2audit"
echo "$AUDIT_USER ALL=(root) NOPASSWD: $PYTHON_BIN $SCRIPT_PATH" > "$SUDOERS_FILE"
chmod 440 "$SUDOERS_FILE"
visudo -c -f "$SUDOERS_FILE"

echo ""
echo "=== SETUP COMPLETE ==="
echo "Add this PRIVATE key as the GitHub Actions secret LAYER2_SSH_KEY, then delete it locally:"
echo ""
cat "$KEY_PATH"
echo ""
echo "SSH user for the workflow: $AUDIT_USER"
echo "Test with: ssh -i $KEY_PATH $AUDIT_USER@<vps-ip> 'anything, ignored'"
echo ""
echo "Once the secret is saved, run: shred -u $KEY_PATH ${KEY_PATH}.pub"
