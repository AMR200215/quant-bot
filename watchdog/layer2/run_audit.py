"""watchdog/layer2/run_audit.py — orchestrates one full Layer 2 audit.
Entrypoint invoked by the GitHub Actions workflow.

    python -m watchdog.layer2.run_audit --host <vps-ip> --ssh-key-path <path>

Flow: SSH to the VPS (forced-command key -- see deploy/layer2/README.md,
can ONLY run deploy/layer2/evidence_dump.py, nothing else) -> build the
evidence bundle -> ground-truth pass (evidence only) -> comparison pass
(ground truth + claims) -> write artifacts + heartbeat -> Telegram summary
if any WARN/CRITICAL findings. Detect only: never modifies code, config,
or production state anywhere in this flow.

The Anthropic client is imported lazily inside main() -- keeps this
module importable (and its logic testable) in environments without the
`anthropic` package installed, consistent with how the rest of this
package avoids hard dependencies where avoidable.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger("watchdog.layer2.run_audit")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SSH_TIMEOUT_S = 30
MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096


def fetch_raw_evidence(host: str, ssh_user: str, ssh_key_path: str) -> dict:
    """Runs the forced-command over SSH. Whatever command is passed here
    is IGNORED by the server if the key is properly configured as a
    forced-command -- evidence_dump.py always runs regardless. Passing a
    command anyway for clarity/local-testing-without-a-forced-key."""
    cmd = [
        "ssh", "-i", ssh_key_path, "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10", "-o", "BatchMode=yes",
        f"{ssh_user}@{host}", "python3 deploy/layer2/evidence_dump.py",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=SSH_TIMEOUT_S)
    if proc.returncode != 0:
        raise RuntimeError(f"evidence collection SSH failed (exit {proc.returncode}): {proc.stderr[-2000:]}")
    return json.loads(proc.stdout)


def make_anthropic_call_model(api_key: str):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    def call_model(prompt: str) -> str:
        response = client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if hasattr(block, "text"))

    return call_model


def run_audit(raw_evidence: dict, call_model, repo_root: Path, send_telegram_fn=None) -> dict:
    """Pure-ish orchestration function, taking raw_evidence and call_model
    as inputs -- fully testable with fakes, no SSH or real API needed."""
    from watchdog.layer2 import audit_prompt, findings_store
    from watchdog.layer2.evidence_bundle import build_evidence_bundle

    bundle = build_evidence_bundle(raw_evidence)
    log.info("built evidence bundle %s (%d items, sha256=%s)",
              bundle.audit_id, len(bundle.items), bundle.raw_bundle_sha256[:12])

    ground_truth = audit_prompt.run_ground_truth_pass(call_model, bundle)
    log.info("ground truth pass complete (%d chars)", len(ground_truth))

    raw_comparison_response = call_model(
        audit_prompt.COMPARISON_SYSTEM_PROMPT.format(
            ground_truth=ground_truth, claims=bundle.to_claims_text())
    )
    findings = audit_prompt.parse_findings(raw_comparison_response, audit_id=bundle.audit_id)
    log.info("comparison pass complete: %d valid finding(s)", len(findings))

    findings_store.write_audit_artifacts(repo_root, bundle, ground_truth, findings,
                                          raw_comparison_response, status="ok")
    findings_store.write_heartbeat(repo_root, bundle, findings, status="ok")

    actionable = [f for f in findings if f.severity in ("WARN", "CRITICAL")]
    if actionable and send_telegram_fn:
        lines = [f"[WATCHDOG LAYER2] {bundle.audit_id}: {len(actionable)} finding(s)"]
        for f in actionable[:5]:
            lines.append(f"\n[{f.severity}] {f.component}: {f.observed_ground_truth[:200]}"
                          f"\n  next: {f.suggested_next_step[:150]}"
                          f"\n  evidence: {', '.join(f.evidence_ids)}")
        send_telegram_fn("\n".join(lines))

    return {"audit_id": bundle.audit_id, "findings_count": len(findings),
            "critical_count": sum(1 for f in findings if f.severity == "CRITICAL"),
            "warn_count": sum(1 for f in findings if f.severity == "WARN")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--ssh-user", default="layer2-audit")
    parser.add_argument("--ssh-key-path", required=True)
    parser.add_argument("--anthropic-api-key", default=None,
                         help="defaults to ANTHROPIC_API_KEY env var")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    args = parser.parse_args(argv)

    import os
    api_key = args.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("no Anthropic API key provided (--anthropic-api-key or ANTHROPIC_API_KEY)")
        return 1

    try:
        raw_evidence = fetch_raw_evidence(args.host, args.ssh_user, args.ssh_key_path)
    except Exception:
        log.exception("evidence collection failed")
        return 1

    call_model = make_anthropic_call_model(api_key)

    send_telegram_fn = None
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if bot_token and chat_id:
        import requests
        def send_telegram_fn(text):
            try:
                requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage",
                               json={"chat_id": chat_id, "text": text}, timeout=10)
            except Exception:
                log.exception("Layer 2 Telegram send failed")

    try:
        result = run_audit(raw_evidence, call_model, Path(args.repo_root), send_telegram_fn)
    except Exception:
        log.exception("audit run failed")
        return 1

    log.info("audit complete: %s", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
