"""
research/scripts/v8_head_to_head.py — V8-REWIRE VR14: per-event V7/V8
disposition matrix.

Reads logs/v8_funnel.jsonl, groups by event_id, and classifies each
resolved event into a 2x2 matrix (V7 pass/fail x V8 pass/fail). This is
the direct, data-grounded answer to the question the whole rewire was
built to let V8 answer honestly: does V8 actually find real candidates
V7 rejects (the fail/pass cell), not just a narrower subset of what V7
already likes.

Restricted to events at/after the V8-REWIRE deploy cutover
(logs/watchdog/v8_rewire_deploy_ts.txt, the same self-bootstrapping stamp
memecoin/v8_paper.py writes) -- pre-rewire rows have no v8_fork_entered
stage at all and would only ever land in a spurious "unresolved" bucket,
not a real disposition.

V7 disposition:
    pass -- screening_passed present
    fail -- screening_rejected present
V8 disposition:
    pass -- v8_opened present
    fail -- v8_gate_rejected, v8_pass_unpriced, or v8_transport_duplicate present
Either side missing both -> unresolved (excluded from the matrix, counted
separately -- most commonly a genuinely in-flight recent event).

Usage:
  python -m research.scripts.v8_head_to_head
  python -m research.scripts.v8_head_to_head --funnel-path logs/v8_funnel.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_FUNNEL_PATH = _REPO_ROOT / "logs" / "v8_funnel.jsonl"
_DEPLOY_STAMP_PATH = _REPO_ROOT / "logs" / "watchdog" / "v8_rewire_deploy_ts.txt"

_V7_PASS_STAGES = {"screening_passed"}
_V7_FAIL_STAGES = {"screening_rejected"}
_V8_PASS_STAGES = {"v8_opened"}
_V8_FAIL_STAGES = {"v8_gate_rejected", "v8_pass_unpriced", "v8_transport_duplicate"}


def _deploy_cutover_ts() -> Optional[float]:
    try:
        return float(_DEPLOY_STAMP_PATH.read_text().strip())
    except Exception:
        return None


def build_matrix(funnel_path: Path, min_ts: Optional[float] = None) -> dict:
    events: dict[str, dict[str, dict]] = {}
    with open(funnel_path) as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            eid = e.get("event_id")
            stage = e.get("stage")
            if not eid or not stage:
                continue
            events.setdefault(eid, {})[stage] = e

    matrix = {"pass_pass": [], "pass_fail": [], "fail_pass": [], "fail_fail": []}
    unresolved = []

    for eid, stages_seen in events.items():
        tr = stages_seen.get("telegram_received")
        if min_ts is not None and (tr is None or tr.get("ts", 0) < min_ts):
            continue
        stage_names = set(stages_seen.keys())

        v7 = "pass" if stage_names & _V7_PASS_STAGES else ("fail" if stage_names & _V7_FAIL_STAGES else None)
        v8 = "pass" if stage_names & _V8_PASS_STAGES else ("fail" if stage_names & _V8_FAIL_STAGES else None)

        if v7 is None or v8 is None:
            unresolved.append({"event_id": eid, "mint": tr.get("mint", "") if tr else "",
                                "v7": v7, "v8": v8, "stages": sorted(stage_names)})
            continue

        matrix[f"{v7}_{v8}"].append({"event_id": eid, "mint": tr.get("mint", "") if tr else ""})

    return {"matrix": matrix, "unresolved": unresolved}


def format_report(result: dict, min_ts: Optional[float]) -> str:
    m = result["matrix"]
    total_resolved = sum(len(v) for v in m.values())
    lines = [
        "V8-REWIRE VR14 — V7/V8 head-to-head disposition matrix",
        f"(era filter: events at/after deploy cutover {min_ts})" if min_ts is not None
        else "(no era filter applied -- min_ts unavailable, includes all history)",
        "",
        f"{'':12}{'V8 pass':>10}{'V8 fail':>10}",
        f"{'V7 pass':12}{len(m['pass_pass']):>10}{len(m['pass_fail']):>10}",
        f"{'V7 fail':12}{len(m['fail_pass']):>10}{len(m['fail_fail']):>10}",
        "",
        f"total resolved: {total_resolved}",
        f"unresolved (in-flight or missing a side): {len(result['unresolved'])}",
        "",
        "The fail_pass cell is the direct evidence for the rewire's whole "
        "premise -- V8 opening on candidates V7 rejected outright:",
    ]
    for ex in m["fail_pass"][:10]:
        lines.append(f"  {ex['event_id']}  {ex['mint']}")
    if not m["fail_pass"]:
        lines.append("  (none yet)")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--funnel-path", default=str(_DEFAULT_FUNNEL_PATH))
    parser.add_argument("--no-era-filter", action="store_true",
                         help="include pre-rewire history too (mostly noise -- see module docstring)")
    args = parser.parse_args(argv)

    path = Path(args.funnel_path)
    if not path.exists():
        print(f"funnel telemetry file does not exist: {path}")
        return 1

    min_ts = None if args.no_era_filter else _deploy_cutover_ts()
    result = build_matrix(path, min_ts=min_ts)
    print(format_report(result, min_ts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
