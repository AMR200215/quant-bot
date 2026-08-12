"""watchdog/runner.py — Layer 1 entrypoint (W1/W2/W11).

    python -m watchdog.runner --profile fast
    python -m watchdog.runner --profile slow
    python -m watchdog.runner --self-test

Invoked by systemd timers (quantbot-watchdog-fast.timer /
quantbot-watchdog-slow.timer), never by cron — a cron-scheduled watchdog
cannot detect cron itself being dead, which is exactly the failure class
this exists to catch. See deploy/systemd/.

Each check is isolated: if one raises, it's recorded as UNKNOWN and the
run continues (W11). The runner's own exit code reflects whether IT
completed, not whether findings were CRITICAL — a CRITICAL finding is the
system working correctly, not a runner bug. systemd will surface a
non-zero exit / timeout as its own unit-failure signal, giving external
observability into the watchdog itself (W2).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

from watchdog import notifier as wd_notifier
from watchdog import state as wd_state
from watchdog.checks import CheckResult, STATUS_CRITICAL, STATUS_OK, STATUS_UNKNOWN, STATUS_WARN
from watchdog.checks import cron_execution, cron_static, telegram_feed, v8_funnel

log = logging.getLogger("watchdog.runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_PATH = REPO_ROOT / "watchdog" / "checks.yaml"
RUN_TIMEOUT_SEC = 60  # hard cap; fast profile must stay cheap


def load_registry(path: Optional[Path] = None) -> dict:
    path = path or DEFAULT_REGISTRY_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def _safe_run(fn, check_id_prefix: str, *args, **kwargs) -> list[CheckResult]:
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        log.exception("watchdog: check %s raised", check_id_prefix)
        return [CheckResult(
            check_id=f"{check_id_prefix}.crashed", status=STATUS_UNKNOWN,
            reason=f"check function raised: {e!r}",
        )]


def run_checks(registry: dict, profile: str, conn, db_path: Optional[Path] = None) -> list[CheckResult]:
    since_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(wd_state.now() - 3600))
    results: list[CheckResult] = []

    jobs = [j for j in registry.get("jobs", []) if j.get("profile", "fast") == profile]
    if jobs:
        cron_dir = registry.get("managed_cron_dir", "/etc/cron.d")
        journal_lines = cron_static.fetch_journal_cron_errors(since_iso)
        results += _safe_run(cron_static.check_cron_static, "cron_static", jobs, cron_dir,
                              journal_lines=journal_lines, journal_fetch_failed=journal_lines is None)
        results += _safe_run(cron_execution.check_cron_execution, "cron_execution", jobs, conn)

    for f in registry.get("funnels", []):
        if f.get("profile", "fast") != profile:
            continue
        results += _safe_run(
            v8_funnel.check_v8_funnel, f"funnel.{f['id']}",
            funnel_path=REPO_ROOT / f["funnel_path"], grace_seconds=f.get("grace_seconds", 120),
            severity_ceiling=f.get("severity", "CRITICAL"), check_id=f"funnel.{f['id']}",
        )

    for feed in registry.get("feeds", []):
        if feed.get("profile", "fast") != profile:
            continue
        if feed["id"] == "telegram":
            journal_lines = telegram_feed.fetch_journal_quantbot_lines(since_iso)
            results += _safe_run(
                telegram_feed.check_telegram_feed,
                f"feed.{feed['id']}", journal_lines=journal_lines,
                journal_fetch_failed=journal_lines is None,
                funnel_path=REPO_ROOT / feed["funnel_path"],
                stale_threshold_s=feed.get("stale_threshold_minutes", 120) * 60,
                severity_ceiling=feed.get("severity", "CRITICAL"), check_id=f"feed.{feed['id']}",
            )

    return results


def run_self_test(conn) -> bool:
    """Exercises the full check-engine + state + notifier pipeline with a
    synthetic, clearly-labeled fault sequence (W20). Sends exactly one
    real [WATCHDOG TEST] Telegram message if credentials are configured.
    Never touches trading state or real checks.yaml jobs."""
    fp_subject = "self_test_synthetic"
    ok = True

    # 1. Synthetic CRITICAL fires immediately.
    r1 = CheckResult(check_id="watchdog.self_test", status=STATUS_CRITICAL,
                      reason="[WATCHDOG TEST] synthetic failure — exercising alert pipeline",
                      subject=fp_subject, severity="CRITICAL")
    s1 = wd_notifier.process_results(conn, [r1], send=True)
    inc = wd_state.get_incident(conn, f"{r1.check_id}:{fp_subject}")
    ok = ok and inc is not None and inc["state"] == "FIRING" and len(s1["new_critical"]) == 1

    # 2. Same incident firing again immediately must NOT re-notify (dedup).
    s2 = wd_notifier.process_results(conn, [r1], send=True)
    ok = ok and len(s2["reminders"]) == 0 and len(s2["new_critical"]) == 0

    # 3. Recovery sends exactly one RECOVERED message.
    r2 = CheckResult(check_id="watchdog.self_test", status=STATUS_OK,
                      reason="[WATCHDOG TEST] synthetic recovery", subject=fp_subject)
    s3 = wd_notifier.process_results(conn, [r2], send=True)
    inc2 = wd_state.get_incident(conn, f"{r1.check_id}:{fp_subject}")
    ok = ok and len(s3["recovered"]) == 1 and inc2["state"] == "RECOVERED"

    log.info("watchdog self-test: %s", "PASS" if ok else "FAIL")
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("fast", "slow"), default="fast")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--db-path", default=None)
    parser.add_argument("--lock-path", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--no-send", action="store_true",
                         help="process incident state but never call Telegram (tests)")
    args = parser.parse_args(argv)

    lock = wd_state.SingletonLock(Path(args.lock_path) if args.lock_path else None)
    if not lock.acquire():
        log.warning("watchdog: another instance holds the singleton lock, skipping this run")
        return 75  # EX_TEMPFAIL-ish: expected/benign skip, not a runner failure

    t0 = time.monotonic()
    run_id = wd_state.new_run_id()
    conn = wd_state.connect(Path(args.db_path) if args.db_path else None)
    final_status = "ok"
    checks_completed = 0
    try:
        if args.self_test:
            wd_state.record_run_start(conn, run_id, "self_test", checks_due=3)
            ok = run_self_test(conn)
            checks_completed = 3
            final_status = "ok" if ok else "self_test_failed"
            wd_state.record_run_finish(conn, run_id, checks_completed, final_status)
            return 0 if ok else 1

        registry = load_registry(Path(args.registry) if args.registry else None)
        results = run_checks(registry, args.profile, conn)
        wd_state.record_run_start(conn, run_id, args.profile, checks_due=len(results))

        for r in results:
            wd_state.record_check_result(conn, run_id, r.check_id, r.status, r.reason, r.evidence)
            checks_completed += 1
            if r.status == STATUS_CRITICAL:
                log.error("[%s] %s: %s", r.status, r.check_id, r.reason)
            elif r.status == STATUS_WARN:
                log.warning("[%s] %s: %s", r.status, r.check_id, r.reason)
            else:
                log.info("[%s] %s: %s", r.status, r.check_id, r.reason)

        wd_notifier.process_results(conn, results, send=not args.no_send)

        def _digest_text():
            active = wd_state.list_active_incidents(conn)
            lines = [f"[WATCHDOG] daily digest", f"active incidents: {len(active)}"]
            for inc in active:
                lines.append(f"  - {inc['severity']} {inc['fingerprint'] if 'fingerprint' in inc.keys() else inc['check_id']}: "
                              f"firing since {inc['first_seen']:.0f}")
            lines.append(f"deployed git sha: {wd_state.get_git_sha()}")
            return "\n".join(lines)

        wd_notifier.maybe_send_daily_digest(conn, _digest_text, send=not args.no_send)

        if time.monotonic() - t0 > RUN_TIMEOUT_SEC:
            final_status = "timeout_exceeded"
        wd_state.record_run_finish(conn, run_id, checks_completed, final_status)
        return 0
    except Exception:
        log.exception("watchdog runner crashed")
        try:
            wd_state.record_run_finish(conn, run_id, checks_completed, "crashed")
        except Exception:
            pass
        return 1
    finally:
        conn.close()
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
