"""watchdog/notifier.py — incident lifecycle (W3) + alerting policy (W10).

Deliberately does NOT import app.alerts: this process runs independently
of the gunicorn app (via a systemd timer), so it reads the same Telegram
env vars directly rather than depending on app.alerts.init() having run
in this process. See the design doc: "do not assume importing
app.alerts._send is sufficient."

Incident lifecycle: (no row) -> SUSPECT -> FIRING -> RECOVERED.
CRITICAL findings fire immediately (deterministic proof, no need to wait
for repeats). WARN findings require WARN_MIN_CONSECUTIVE occurrences
before they page, to avoid alerting on a single transient miss.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests

from watchdog import state as wd_state
from watchdog.checks import CheckResult, STATUS_CRITICAL, STATUS_NOT_APPLICABLE, STATUS_OK, STATUS_UNKNOWN, STATUS_WARN

log = logging.getLogger("watchdog.notifier")

CRITICAL_REMINDER_FIRST_SEC = 2 * 3600
CRITICAL_REMINDER_SUBSEQUENT_SEC = 6 * 3600
WARN_MIN_CONSECUTIVE = 2
WARN_REMINDER_SEC = 6 * 3600

_PREFIX = {
    "CRITICAL": "[WATCHDOG CRITICAL]",
    "WARN": "[WATCHDOG WARN]",
    "INFO": "[WATCHDOG INFO]",
    "RECOVERED": "[WATCHDOG RECOVERED]",
}


def _fingerprint(r: CheckResult) -> str:
    return f"{r.check_id}:{r.subject}" if r.subject else r.check_id


def send_telegram(text: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        log.warning("watchdog notifier: Telegram not configured, dropping alert:\n%s", text)
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text}, timeout=10,
        )
        if r.status_code != 200:
            log.error("watchdog notifier: Telegram send failed (%s): %s", r.status_code, r.text[:300])
            return False
        return True
    except Exception as e:
        log.error("watchdog notifier: Telegram send raised: %s", e)
        return False


def _format_alert(kind: str, r: CheckResult, incident: dict) -> str:
    prefix = _PREFIX.get(kind, "[WATCHDOG]")
    lines = [
        f"{prefix} {r.check_id}",
        f"component: {r.subject or r.check_id}",
        f"observed: {r.status} — {r.reason}",
        f"severity: {r.severity}",
        f"first_seen: {incident.get('first_seen')}",
        f"consecutive_failures: {incident.get('consecutive_failures')}",
        f"deployed_git_sha: {wd_state.get_git_sha()}",
        f"fingerprint: {incident.get('fingerprint') or _fingerprint(r)}",
    ]
    return "\n".join(lines)


def _format_recovery(r: CheckResult, incident: dict) -> str:
    return "\n".join([
        f"{_PREFIX['RECOVERED']} {r.check_id}",
        f"component: {r.subject or r.check_id}",
        f"now: {r.status} — {r.reason}",
        f"was firing since: {incident.get('first_seen')}",
        f"fingerprint: {incident.get('fingerprint') or _fingerprint(r)}",
    ])


def process_results(conn, results: list[CheckResult], now_ts: Optional[float] = None,
                     send: bool = True) -> dict:
    """Applies incident-lifecycle policy to a batch of check results from
    one watchdog run. Returns a summary dict; as a side effect, persists
    incident state and sends Telegram alerts per the dedup policy above.

    `send=False` lets callers (tests, --self-test) exercise the full state
    machine without hitting the network.
    """
    now_ts = now_ts if now_ts is not None else wd_state.now()
    summary = {"new_critical": [], "new_warn": [], "reminders": [], "recovered": [], "ok_noop": []}
    to_send: list[str] = []

    for r in results:
        fp = _fingerprint(r)
        existing = wd_state.get_incident(conn, fp)

        if r.status in (STATUS_OK, STATUS_NOT_APPLICABLE):
            if existing and existing["state"] in ("SUSPECT", "FIRING"):
                incident = wd_state.upsert_incident(
                    conn, fp, r.check_id, r.subject, r.severity, "RECOVERED",
                    evidence=r.evidence, reset_consecutive=True, recovered=True, ts=now_ts,
                )
                summary["recovered"].append(fp)
                to_send.append(_format_recovery(r, incident))
            else:
                summary["ok_noop"].append(fp)
            continue

        if r.status == STATUS_UNKNOWN:
            # Evidence unavailable this run. UNKNOWN is not GREEN, so it
            # must never silently recover a FIRING incident just because
            # evidence briefly went missing -- but it's also not a new
            # failure, so it must not extend the consecutive-failure
            # streak or page on its own either. Leave FIRING incidents
            # exactly as they are; for anything else, record a
            # zero-streak SUSPECT marker so it's visible without looking
            # like an escalating failure.
            if existing and existing["state"] == "FIRING":
                wd_state.touch_incident_seen(conn, fp, ts=now_ts)
            else:
                wd_state.upsert_incident(conn, fp, r.check_id, r.subject, "WARN", "SUSPECT",
                                          evidence=r.evidence, reset_consecutive=True, ts=now_ts)
            continue

        # WARN or CRITICAL
        if r.status == STATUS_CRITICAL:
            new_state = "FIRING"  # deterministic proof -> fire immediately, no debounce
        else:
            prior_consecutive = existing["consecutive_failures"] if existing else 0
            new_state = "FIRING" if prior_consecutive + 1 >= WARN_MIN_CONSECUTIVE else "SUSPECT"

        was_firing = bool(existing and existing["state"] == "FIRING")
        incident = wd_state.upsert_incident(
            conn, fp, r.check_id, r.subject, r.severity, new_state, evidence=r.evidence, ts=now_ts,
        )
        incident["fingerprint"] = fp

        if new_state != "FIRING":
            continue  # SUSPECT: recorded, not yet alerted

        if not was_firing:
            # First time this incident reaches FIRING.
            summary["new_critical" if r.status == STATUS_CRITICAL else "new_warn"].append(fp)
            to_send.append(_format_alert(r.status, r, incident))
            wd_state.upsert_incident(conn, fp, r.check_id, r.subject, r.severity, "FIRING",
                                      evidence=r.evidence, touch_notified=True, ts=now_ts)
        else:
            # Still firing -- only send a reminder per the cadence policy.
            last_notified = incident.get("last_notified") or 0
            reminder_interval = (
                CRITICAL_REMINDER_FIRST_SEC if r.status == STATUS_CRITICAL
                and (now_ts - incident["first_seen"]) < (CRITICAL_REMINDER_FIRST_SEC + CRITICAL_REMINDER_SUBSEQUENT_SEC)
                else CRITICAL_REMINDER_SUBSEQUENT_SEC if r.status == STATUS_CRITICAL
                else WARN_REMINDER_SEC
            )
            if now_ts - last_notified >= reminder_interval:
                summary["reminders"].append(fp)
                to_send.append(_format_alert(r.status, r, incident))
                wd_state.upsert_incident(conn, fp, r.check_id, r.subject, r.severity, "FIRING",
                                          evidence=r.evidence, touch_notified=True, ts=now_ts)

    if to_send and send:
        # Aggregate same-run alerts into one message where practical (W10).
        if len(to_send) == 1:
            send_telegram(to_send[0])
        else:
            send_telegram(f"[WATCHDOG] {len(to_send)} findings this run:\n\n" + "\n\n---\n\n".join(to_send))

    summary["_messages"] = to_send
    return summary


def maybe_send_daily_digest(conn, digest_text_fn, now_ts: Optional[float] = None,
                             send: bool = True) -> bool:
    """At most one digest per UTC calendar date. digest_text_fn() -> str is
    called lazily only if a digest is actually due, so callers don't pay
    for building digest content on every run."""
    import datetime

    now_ts = now_ts if now_ts is not None else wd_state.now()
    today = datetime.datetime.fromtimestamp(now_ts, tz=datetime.timezone.utc).date().isoformat()
    last = wd_state.get_digest_marker(conn, "last_digest_date")
    if last == today:
        return False
    wd_state.set_digest_marker(conn, "last_digest_date", today)
    if send:
        send_telegram(digest_text_fn())
    return True
