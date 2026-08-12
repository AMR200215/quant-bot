"""watchdog/checks/cron_execution.py — W5B: did the scheduled job actually
run, not just does its output artifact's mtime look fresh?

This is the direct fix for the K5 incident: a manual test run at install
time updated the artifact and made it look current for days while the
actual `/etc/cron.d` entry was silently rejected by the cron daemon.
Manual executions are recorded with trigger_type="manual" and are
structurally incapable of satisfying this check — only a
trigger_type="scheduler" receipt (written by watchdog.exec_wrapper when
cron itself invokes the job) counts as proof of a real scheduled run.
"""

from __future__ import annotations

import time
from typing import Optional

import croniter

from watchdog import state as wd_state
from watchdog.checks import CheckResult, STATUS_CRITICAL, STATUS_OK, STATUS_WARN, cap_severity


def get_boot_time() -> Optional[float]:
    try:
        with open("/proc/uptime") as f:
            uptime_s = float(f.read().split()[0])
        return time.time() - uptime_s
    except Exception:
        return None


def evaluate_job_liveness(schedule: str, grace_minutes: float, now_ts: float,
                           boot_time_ts: Optional[float],
                           latest_receipt: Optional[dict]) -> tuple[str, str, dict]:
    """Pure function (no I/O) so it's directly unit-testable against
    fabricated clocks/boot times/receipts."""
    itr = croniter.croniter(schedule, now_ts)
    expected_prev = itr.get_prev(float)

    if boot_time_ts is not None and expected_prev < boot_time_ts:
        # The most recently expected fire predates this boot -- the
        # machine was down for it. Don't fault cron for a fire it
        # structurally could not have executed; look at the first fire
        # due *after* boot instead.
        itr2 = croniter.croniter(schedule, boot_time_ts)
        next_after_boot = itr2.get_next(float)
        if next_after_boot > now_ts:
            return (STATUS_OK,
                    f"no scheduled fire has been due since boot ({boot_time_ts:.0f}); "
                    f"next due {next_after_boot:.0f}",
                    {"expected_prev": expected_prev, "boot_time": boot_time_ts,
                     "next_after_boot": next_after_boot})
        expected_prev = next_after_boot

    deadline = expected_prev + grace_minutes * 60
    evidence = {"expected_prev_fire": expected_prev, "deadline": deadline,
                "now": now_ts, "latest_receipt": latest_receipt}

    if now_ts < deadline:
        return (STATUS_OK,
                f"within grace of most recently expected fire ({expected_prev:.0f}), "
                f"not yet overdue (deadline {deadline:.0f})",
                evidence)

    if latest_receipt is None:
        return (STATUS_CRITICAL,
                f"no scheduler execution receipt ever recorded for this job; "
                f"expected fire at {expected_prev:.0f} is past its {grace_minutes:.0f}min grace",
                evidence)

    if latest_receipt["started_at"] < expected_prev:
        return (STATUS_CRITICAL,
                f"latest scheduler receipt ({latest_receipt['started_at']:.0f}) predates "
                f"the most recently expected fire ({expected_prev:.0f}) — job did not run "
                f"on schedule (a manual-trigger receipt, if any exists more recently, does "
                f"NOT count as proof of scheduled execution)",
                evidence)

    if latest_receipt.get("exit_code") not in (0,):
        return (STATUS_WARN,
                f"last scheduler run at {latest_receipt['started_at']:.0f} "
                f"exited {latest_receipt.get('exit_code')} (non-zero)",
                evidence)

    return (STATUS_OK,
            f"last scheduler run at {latest_receipt['started_at']:.0f} succeeded",
            evidence)


def check_cron_execution(jobs: list[dict], conn, now_ts: Optional[float] = None,
                          boot_time_ts: Optional[float] = None) -> list[CheckResult]:
    now_ts = now_ts if now_ts is not None else wd_state.now()
    if boot_time_ts is None:
        boot_time_ts = get_boot_time()

    results: list[CheckResult] = []
    for job in jobs:
        latest = wd_state.get_latest_job_receipt(conn, job["id"], trigger_type="scheduler")
        status, reason, evidence = evaluate_job_liveness(
            job["schedule"], job.get("grace_minutes", 30), now_ts, boot_time_ts, latest,
        )
        capped = cap_severity(status, job.get("severity", "CRITICAL"))
        results.append(CheckResult(
            check_id=f"cron_execution.{job['id']}", status=capped, reason=reason,
            subject=job["id"], severity=job.get("severity", "WARN"), evidence=evidence,
        ))
    return results
