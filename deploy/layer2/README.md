# Layer 2 — Watchdog External Audit Agent

Runs from GitHub Actions, not the VPS (W12): a VPS-wide outage, or Layer 1
itself going down, must not also disable the thing supervising them.
Detect only — never modifies code, config, or production state.

## Architecture

```
GitHub Actions (daily 03:30 UTC + manual dispatch)
      |
      v  SSH, forced-command key (can run ONLY evidence_dump.py)
deploy/layer2/evidence_dump.py on the VPS  -- read-only, one JSON dump
      |
      v
watchdog/layer2/evidence_bundle.py  -- EV-ID assignment, SHA-256 hash,
                                        claims vs evidence split
      |
      v
watchdog/layer2/audit_prompt.py  -- two-call ordering (W15):
   Call 1: evidence only -> model derives ground truth, citing EV-IDs
   Call 2: ground truth + claims (RECEIPTS.md, labeled as assertions,
           not evidence) -> structured findings, each citing EV-IDs
      |
      v
watchdog/layer2/findings_store.py  -- logs/watchdog/audits/<id>/*,
                                        logs/watchdog/layer2_heartbeat.json
      |
      v
git commit + push (GitHub Actions' own built-in token — no new
credential needed for this step)
      |
      v
VPS's existing git pull cron (9am/18:00 UTC) picks up the heartbeat file
-> watchdog/checks/layer2_staleness.py (Layer 1) sees it, flags if stale
```

## What's needed to go live (the credential checkpoint)

Two things, both requiring your action — nothing here is provisioned
automatically:

### 1. Anthropic API key

Create one at console.anthropic.com with billing enabled (separate from
whatever Claude Code session/subscription is being used to work on this
repo — Claude Code usage doesn't carry over as API credits). Add it as
the GitHub Actions secret `ANTHROPIC_API_KEY` on this repo.

**Cost estimate and model choice, worked out in full so this doesn't
depend on any conversation history:**

A daily audit run's two model calls (`watchdog/layer2/audit_prompt.py`)
send roughly 15-30k input tokens (the evidence bundle — bounded by
`evidence_dump.py`'s per-section caps — plus the RECEIPTS.md tail used as
claims text) and produce 1-2k output tokens per call, ~30-60k input /
2-4k output total per day.

| Model | Cost/run | Cost/month (30 runs) | Notes |
|---|---|---|---|
| Haiku | ~$0.02-0.04 | under $1.50 | cheapest, but weaker at the multi-hop cross-referencing (timestamp + file state + doc claim -> one conclusion) this audit actually needs — risk is a missed finding, not a wrong one, and that failure is silent |
| **Sonnet (default, `MODEL` in `run_audit.py`)** | ~$0.10-0.15 | **~$3-5** | recommended: good at exactly the anchoring-resistant, evidence-then-claims reasoning W15 requires, at a cost delta from Haiku too small to be the deciding factor |
| Opus | ~$0.50-0.70 | ~$15-20 | only worth it if Sonnet's real audits turn out to miss things in practice |

GitHub Actions cost is effectively zero at this volume (a few minutes/day,
well inside the free tier).

**If you want to reconsider the model choice**: the empirically honest way
to do it is to run the same evidence bundle through two models on a few
real audits and compare findings directly, not decide from pricing tables
alone — swap `MODEL` in `watchdog/layer2/run_audit.py` for a side-by-side
comparison once real audit history exists.

### 2. Read-only SSH credential

Run `deploy/layer2/install.sh` **on the VPS as root**. It:
- Creates a dedicated, unprivileged system user `layer2audit` (no login
  shell).
- Generates a new ed25519 keypair.
- Installs the public key with a **forced command** — whatever the SSH
  client requests, only `evidence_dump.py` ever runs, no matter what.
  This is the actual security boundary.
- Installs one narrowly-scoped `sudoers` rule (`NOPASSWD`, exact absolute
  path, no argument substitution possible) so the script can read
  root-owned files without changing any existing file permissions on the
  box.
- Prints the private key once, to be added as the GitHub Actions secret
  `LAYER2_SSH_KEY`, then deleted locally (`shred -u`).

What this credential can **never** do, structurally (not by policy —
the forced command and sudoers rule simply don't include these):
`systemctl restart/stop`, `kill`, `rm`, `git reset`/`checkout`, `sed -i`,
any database write, any trading command.

Also needed as repo secrets (reuse existing values, no new credentials):
- `VPS_HOST` — the VPS IP
- `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` — same bot already used by
  Layer 1 and the trading app

## Verifying before trusting it

```
# From your own machine, confirm the forced command actually restricts:
ssh -i /tmp/layer2audit_key layer2audit@<vps-ip> "rm -rf /"
# Must run evidence_dump.py and print JSON, NOT execute the rm command.
```

Then trigger `workflow_dispatch` manually once from the GitHub Actions
tab and inspect `logs/watchdog/audits/<new-id>/` before trusting the
daily schedule.
