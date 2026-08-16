"""watchdog/checks/path_collection.py — V8-FD Phase 1.5 (P15-7): expose
research path-collection health (the FD14/FD15/FD17 data full-strategy
EV depends on) to the watchdog's own incident/debounce/digest system,
not just the parallel send_alert-based deadman in
research/peak_tracker.py.

Reads logs/watchdog/path_collection_daily.json, written by
research/peak_tracker.py:_write_daily_status_json() at each UTC
rollover. Root-caused live (V8-FD P15-5): real production days show the
daily PumpPortal message budget (PP_DAILY_MSG_BUDGET) exceeded by
23-48%, with the budget getting hit as early as 04:15 UTC on some days
-- after which every newly-scheduled token for the rest of the day gets
silently dropped (research/peak_tracker.py's _drain_pending(), now
counted via budget_dropped_tokens instead of untracked).

Deliberately WARN, not CRITICAL, ceiling: this is a known, understood,
cost-bounded constraint (raising PP_DAILY_MSG_BUDGET spends real SOL --
a decision for the user, not something this check should page urgently
for) -- not a code bug. The existing send_alert-based deadman/FAIL in
peak_tracker.py already pages for the more severe case (zero ticks all
day); this check's job is visibility (yield % trend, budget-exceeded
frequency) for the eventual filter-derivation decision, not urgent
incident response.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from watchdog.checks import CheckResult, STATUS_OK, STATUS_UNKNOWN, STATUS_WARN, cap_severity

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_STATUS_PATH = REPO_ROOT / "logs" / "watchdog" / "path_collection_daily.json"
DEFAULT_STALE_THRESHOLD_S = 30 * 3600  # 30h -- daily cadence + grace, matching layer2_staleness's pattern

# Below this yield %, flag WARN even if the deadman's own (higher) file-count
# floor didn't trip -- yield is the metric that actually matters for FD14/17,
# not raw file count alone.
LOW_YIELD_WARN_PCT = 50.0


def check_path_collection(now_ts: Optional[float] = None,
                           status_path: Optional[Path] = None,
                           stale_threshold_s: float = DEFAULT_STALE_THRESHOLD_S,
                           severity_ceiling: str = "WARN",
                           check_id: str = "research.path_collection") -> list[CheckResult]:
    now_ts = now_ts if now_ts is not None else time.time()
    path = Path(status_path) if status_path else DEFAULT_STATUS_PATH

    if not path.exists():
        return [CheckResult(
            check_id=check_id, status=STATUS_UNKNOWN,
            reason="no path_collection_daily.json found — either peak_tracker has never "
                   "completed a full UTC day yet, or this is a pre-P15-7 deploy",
            severity=severity_ceiling,
        )]

    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return [CheckResult(check_id=check_id, status=STATUS_UNKNOWN,
                             reason=f"failed to parse path_collection_daily.json: {e!r}",
                             severity=severity_ceiling)]

    generated_at = data.get("generated_at")
    evidence = dict(data)

    # Staleness check first — a fresh-looking OK on stale data is worse than UNKNOWN.
    if generated_at:
        try:
            from datetime import datetime
            gen_ts = datetime.fromisoformat(generated_at.replace("Z", "+00:00")).timestamp()
            age = now_ts - gen_ts
            evidence["age_s"] = age
            if age > stale_threshold_s:
                return [CheckResult(
                    check_id=check_id, status=STATUS_UNKNOWN,
                    reason=f"path_collection_daily.json is {age/3600:.1f}h old "
                           f"(threshold {stale_threshold_s/3600:.0f}h) — peak_tracker's daily "
                           f"rollover may have stopped firing",
                    severity=severity_ceiling, evidence=evidence,
                )]
        except Exception:
            pass

    yield_pct = data.get("path_yield_pct")
    budget_exceeded = data.get("budget_exceeded", False)
    budget_dropped = data.get("budget_dropped_tokens", 0)
    scheduled = data.get("tokens_scheduled", 0)

    reasons = []
    if yield_pct is not None and yield_pct < LOW_YIELD_WARN_PCT:
        reasons.append(f"path yield {yield_pct}% (< {LOW_YIELD_WARN_PCT}% floor)")
    if budget_exceeded:
        reasons.append(f"PP daily message budget exceeded "
                        f"({data.get('pp_messages')}/{data.get('pp_daily_msg_budget')}, "
                        f"{budget_dropped} token(s) dropped as a result)")

    if reasons:
        return [CheckResult(
            check_id=check_id, status=cap_severity(STATUS_WARN, severity_ceiling),
            reason=(f"{data.get('day')}: " + "; ".join(reasons) +
                    f" — known, cost-bounded constraint (raising the budget spends real "
                    f"SOL), not a code bug; see docs/RECEIPTS.md's V8-FD Phase 1.5 section"),
            severity=severity_ceiling, evidence=evidence,
        )]

    return [CheckResult(
        check_id=check_id, status=STATUS_OK,
        reason=f"{data.get('day')}: {scheduled} scheduled, "
               f"{data.get('path_files')} path files (yield {yield_pct}%), "
               f"budget {data.get('pp_messages')}/{data.get('pp_daily_msg_budget')}",
        evidence=evidence,
    )]
