#!/usr/bin/env python3
"""deploy/layer2/evidence_dump.py — Layer 2's ONLY entrypoint on the VPS.

This is installed as a forced-command for a dedicated, restricted SSH key
(see deploy/layer2/README.md): the key's authorized_keys entry pins
`command="..../evidence_dump.py"`, so whatever the SSH client requests,
this is the only thing that ever runs — there is no way to use this
credential to run anything else, ever. That is the actual security
boundary, not anything enforced by this script's own logic.

Still, this script's own logic is written as if that boundary might not
hold (defense in depth): every subprocess call here is read-only by
construction (systemctl status/list-*, journalctl, git rev-parse/status/
log, sqlite3 SELECT via Python's sqlite3 module, file reads). Nothing
here writes, restarts, kills, resets, or touches trading state. If you
are adding a new evidence section, it must not need anything beyond read
access -- if it does, it does not belong in this script.

Emits exactly one JSON object to stdout. Never raises past its own
try/except boundaries -- a failed section is recorded as
{"error": "..."} for that section, not a crashed script (an audit that
can see 9/10 sections is far more useful than an audit that sees none
because the 10th section's command isn't available on this box).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

REPO_ROOT = "/root/quant-bot"
STATE_DB = os.path.join(REPO_ROOT, "logs", "watchdog", "state.db")
RECEIPTS_PATH = os.path.join(REPO_ROOT, "docs", "RECEIPTS.md")
CMD_TIMEOUT_S = 15
MAX_TEXT_LEN = 4000  # per-section cap; keeps the whole bundle bounded


def _run(cmd: list[str], timeout: int = CMD_TIMEOUT_S) -> dict:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"exit_code": proc.returncode,
                "stdout": proc.stdout[-MAX_TEXT_LEN:],
                "stderr": proc.stderr[-1000:]}
    except subprocess.TimeoutExpired:
        return {"error": f"timed out after {timeout}s"}
    except Exception as e:
        return {"error": repr(e)}


def _section(name):
    """Decorator-ish helper: run fn(), catch anything, tag with section name."""
    def wrap(fn):
        try:
            return fn()
        except Exception as e:
            return {"error": f"{name} collection raised: {e!r}"}
    return wrap


def collect_host_state() -> dict:
    boot_id = ""
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            boot_id = f.read().strip()
    except Exception:
        pass
    uptime_s = None
    try:
        with open("/proc/uptime") as f:
            uptime_s = float(f.read().split()[0])
    except Exception:
        pass
    return {
        "host_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "boot_id": boot_id,
        "uptime_s": uptime_s,
    }


def collect_git_state() -> dict:
    sha = _run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"])
    status = _run(["git", "-C", REPO_ROOT, "status", "--porcelain"])
    return {
        "head_sha": sha.get("stdout", "").strip() or sha,
        "dirty_tree_lines": status.get("stdout", "").strip().splitlines(),
    }


def collect_systemd_state() -> dict:
    units = ["quantbot.service", "quantbot-research.service",
             "quantbot-watchdog-fast.service", "quantbot-watchdog-slow.service"]
    services = {}
    for u in units:
        r = _run(["systemctl", "show", u, "--property=ActiveState,SubState,Result,ExecMainStatus"])
        services[u] = r.get("stdout", "").strip()
    timers = _run(["systemctl", "list-timers", "quantbot-watchdog-*", "--no-pager"])
    return {"services": services, "timers": timers.get("stdout", "")}


def collect_cron_state() -> dict:
    managed = ["quantbot-epoch", "quantbot-v8", "quantbot-v8-inputs"]
    files = {}
    for name in managed:
        path = f"/etc/cron.d/{name}"
        try:
            st = os.stat(path)
            with open(path) as f:
                content = f.read()
            files[name] = {"mode": oct(st.st_mode), "mtime": st.st_mtime, "content": content}
        except Exception as e:
            files[name] = {"error": repr(e)}
    journal_errors = _run(["journalctl", "-u", "cron", "--since", "-24h", "--no-pager"])
    stdout = journal_errors.get("stdout", "")
    error_lines = [l for l in stdout.splitlines() if "error" in l.lower() or "bad " in l.lower()]
    return {"files": files, "recent_parser_error_lines": error_lines[-20:]}


def collect_watchdog_state() -> dict:
    try:
        import sqlite3
        conn = sqlite3.connect(STATE_DB, timeout=5)
        conn.row_factory = sqlite3.Row
        out = {}
        for profile in ("fast", "slow"):
            row = conn.execute(
                "SELECT * FROM watchdog_runs WHERE profile=? ORDER BY started_at DESC LIMIT 1",
                (profile,),
            ).fetchone()
            out[f"latest_{profile}_run"] = dict(row) if row else None
            if row:
                checks = conn.execute(
                    "SELECT check_id, status, reason, evidence FROM check_results WHERE run_id=?",
                    (row["run_id"],),
                ).fetchall()
                out[f"latest_{profile}_check_results"] = [dict(c) for c in checks]
        incidents = conn.execute(
            "SELECT fingerprint, check_id, subject, severity, state, first_seen, last_seen, "
            "consecutive_failures, last_notified, recovered_at FROM incidents"
        ).fetchall()
        out["all_incidents"] = [dict(i) for i in incidents]
        receipts = conn.execute(
            "SELECT job_id, trigger_type, started_at, exit_code FROM job_receipts "
            "ORDER BY started_at DESC LIMIT 20"
        ).fetchall()
        out["recent_job_receipts"] = [dict(r) for r in receipts]
        conn.close()
        return out
    except Exception as e:
        return {"error": repr(e)}


def collect_research_pipeline_state() -> dict:
    out = {}
    qpath = os.path.join(REPO_ROOT, "research", "data", "signal_queue.jsonl")
    opath = os.path.join(REPO_ROOT, "research", "data", ".queue_offset")
    try:
        out["queue_size"] = os.path.getsize(qpath) if os.path.exists(qpath) else None
        out["queue_offset"] = open(opath).read().strip() if os.path.exists(opath) else None
    except Exception as e:
        out["error"] = repr(e)
    return out


def collect_receipts_tail() -> dict:
    try:
        with open(RECEIPTS_PATH) as f:
            text = f.read()
        return {"mtime": os.path.getmtime(RECEIPTS_PATH), "length": len(text),
                "tail": text[-MAX_TEXT_LEN:]}
    except Exception as e:
        return {"error": repr(e)}


def main() -> int:
    bundle = {
        "collected_at": time.time(),
        "host": _section("host")(collect_host_state),
        "git": _section("git")(collect_git_state),
        "systemd": _section("systemd")(collect_systemd_state),
        "cron": _section("cron")(collect_cron_state),
        "watchdog": _section("watchdog")(collect_watchdog_state),
        "research_pipeline": _section("research_pipeline")(collect_research_pipeline_state),
        "receipts_tail": _section("receipts_tail")(collect_receipts_tail),
    }
    print(json.dumps(bundle, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
