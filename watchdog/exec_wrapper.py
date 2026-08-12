"""watchdog/exec_wrapper.py — wraps a scheduled job invocation and records
a scheduler execution receipt (W5B).

This is the fix for the exact K5 incident: an artifact's mtime being fresh
is not proof cron ran it — a manual test run updates the same artifact.
This wrapper makes the *invocation itself* the evidence, and tags it with
who triggered it, so a manual run can never masquerade as a scheduler firing.

Usage (from a cron line):
    .venv/bin/python -m watchdog.exec_wrapper --job-id k5_nightly \\
        --schedule "15 0 * * *" -- .venv/bin/python -m research.scripts.v8_inputs_nightly

Usage (manual, for testing — explicitly does NOT satisfy scheduler liveness):
    .venv/bin/python -m watchdog.exec_wrapper --job-id k5_nightly --trigger manual \\
        --schedule "15 0 * * *" -- .venv/bin/python -m research.scripts.v8_inputs_nightly

Exit code always mirrors the wrapped command's exit code — receipt-recording
failures are logged to stderr but never mask or block the underlying job.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from watchdog import state as wd_state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Wrap a scheduled command and record its execution receipt.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--trigger", choices=("scheduler", "manual"), default="scheduler")
    parser.add_argument("--schedule", default="", help="cron expression, for receipt context only")
    parser.add_argument("--output-artifact", default="")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("watchdog.exec_wrapper: no command given after '--'", file=sys.stderr)
        return 2

    started_at = wd_state.now()
    try:
        proc = subprocess.run(cmd)
        exit_code = proc.returncode
    except FileNotFoundError as e:
        print(f"watchdog.exec_wrapper: command not found: {e}", file=sys.stderr)
        exit_code = 127
    except Exception as e:
        print(f"watchdog.exec_wrapper: failed to launch command: {e}", file=sys.stderr)
        exit_code = 126
    finished_at = wd_state.now()

    try:
        conn = wd_state.connect(args.db_path)
        wd_state.record_job_receipt(
            conn, job_id=args.job_id, trigger_type=args.trigger,
            started_at=started_at, finished_at=finished_at, exit_code=exit_code,
            output_artifact=args.output_artifact,
        )
        conn.close()
    except Exception as e:
        # Never let receipt-recording failure mask the wrapped command's
        # real exit code — fail loud (stderr) but don't block the job.
        print(f"watchdog.exec_wrapper: failed to record execution receipt: {e}",
              file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
