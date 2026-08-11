"""
memecoin/v8_telemetry.py — V8-TWIN-FIX VF1: complete funnel observability.

One structured JSONL line per (event_id, stage) for every Telegram
candidate, so no candidate can silently disappear between pipeline
stages without a trace. Append-only, best-effort (never raises, never
blocks the signal path). Deliberately separate from the general-purpose
entry-trace system (memecoin/telemetry.py) -- this is purpose-built for
one question: where in the funnel did this candidate's V8 disposition
get decided, and why.

Root cause this exists for: 15 real production candidates with
progress_at_signal<0.70 passed V7 screening but produced zero v8_paper
log output -- not even a rejection line -- because the only V8-relevant
logging that existed was debug-level and/or didn't cover every stage.
See docs/RECEIPTS.md's "V8-TWIN-FIX" section for the full investigation.

Stages (a candidate may exit at any stage; that's a valid, observable
outcome -- the requirement is that it's never invisible):
    telegram_received     -- alert text parsed, event_id assigned
    screening_rejected     -- failed a V7 screening filter (no_dex_data,
                               low_buy_pressure, vol_*_out_of_range,
                               price_change_blow_off, mcap_too_high, ...)
    screening_passed       -- cleared all V7 screening filters
    signal_constructed     -- Signal object built (make_social_alert_signal)
    add_signal_entered     -- _add_signal() reached with a non-None Signal
    dedup_rejected          -- _is_duplicate() returned True
    v8_gate_entered         -- v8_paper.book.maybe_open() reached
    v8_gate_rejected        -- passes_v8_gate() returned False
    v8_opened               -- V8 paper position created

Run:
    tail -f logs/v8_funnel.jsonl | python3 -m json.tool
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_PATH = Path(__file__).parent.parent / "logs" / "v8_funnel.jsonl"
_LOCK = threading.Lock()

VALID_STAGES = frozenset({
    "telegram_received", "screening_rejected", "screening_passed",
    "signal_constructed", "add_signal_entered", "dedup_rejected",
    "v8_gate_entered", "v8_gate_rejected", "v8_opened",
})


def emit(stage: str, *, event_id: str = "", mint: str = "",
         progress: Optional[float] = None, progress_source: str = "",
         dex_id: str = "", venue_state: str = "",
         result: str = "", reason: str = "") -> None:
    """Best-effort structured append. Never raises, never blocks the
    caller -- a failure here must never affect the signal path."""
    if stage not in VALID_STAGES:
        log.debug("v8_telemetry: unknown stage %r (emitting anyway)", stage)
    row = {
        "ts":              time.time(),
        "stage":           stage,
        "event_id":        event_id,
        "mint":            mint,
        "progress":        progress,
        "progress_source": progress_source,
        "dex_id":          dex_id,
        "venue_state":     venue_state,
        "result":          result,
        "reason":          reason,
    }
    try:
        line = json.dumps(row)
        with _LOCK:
            _PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_PATH, "a") as f:
                f.write(line + "\n")
    except Exception as e:
        log.debug("v8_telemetry: emit failed: %s", e)
