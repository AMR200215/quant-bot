"""
memecoin/execution_receipts.py — Append-only JSONL execution receipt logger.

Every on-chain buy/sell TX gets a receipt written immediately after the
signature is received (even before confirmation).

Public API:
    write_receipt(receipt: dict) -> None
    read_receipts(mint=None, pos_id=None, since_ts=None) -> list[dict]

Storage: logs/execution_receipts.jsonl (one JSON object per line)

Receipt schema (required: sig, action):
    sig: str              — on-chain TX signature
    action: str           — "buy" | "sell"
    mint: str
    pos_id: str
    symbol: str
    sol_amount: float
    token_amount: float
    price_usd: float
    route: str            — "pumpportal" | "pump-amm" | "jupiter" | "bc_t22" etc.
    status: str           — "sent" | "confirmed" | "reverted" | "failed"
    sent_at_utc: str      — ISO8601 UTC (auto-populated if absent)
    confirmed_at_utc: str | None
    slot: int | None
    error: str | None
    metadata: dict

Security: strip fields named *_key, *_secret, *private*, *seed*, *mnemonic*,
*password*, *keypair* before writing.
"""

import json
import logging
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
try:
    from memecoin.config import EXECUTION_RECEIPTS_FILE  # type: ignore
except (ImportError, AttributeError):
    EXECUTION_RECEIPTS_FILE = "logs/execution_receipts.jsonl"

# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
_write_lock = threading.Lock()

_SECRET_RE = re.compile(
    r"(?i)(private|(?:api|helius|rpc|wallet|auth|solana)_key|secret_key|seed|mnemonic|password|keypair)"
)


def _receipts_path(override: str | None = None) -> Path:
    p = Path(override or EXECUTION_RECEIPTS_FILE)
    if not p.is_absolute():
        p = Path(__file__).parent.parent / p
    return p


def _redact(d: dict) -> dict:
    return {k: v for k, v in d.items() if not _SECRET_RE.search(k)}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_receipt(receipt: dict, _path_override: str | None = None) -> None:
    """
    Append a redacted receipt to execution_receipts.jsonl.
    Adds sent_at_utc automatically if absent. Never raises.
    """
    try:
        if not isinstance(receipt, dict):
            log.debug("write_receipt: non-dict input ignored (%s)", type(receipt))
            return
        safe = dict(receipt)
        safe.setdefault("sent_at_utc", _utc_now_iso())
        safe = _redact(safe)
        line = json.dumps(safe, default=str) + "\n"
        p = _receipts_path(_path_override)
        p.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock:
            with open(p, "a") as f:
                f.write(line)
    except Exception as exc:
        log.debug("write_receipt error: %s", exc)


def read_receipts(
    mint: str | None = None,
    pos_id: str | None = None,
    since_ts: float | None = None,   # unix epoch float
    _path_override: str | None = None,
) -> list[dict]:
    """
    Read receipts, optionally filtered.

    since_ts: unix epoch; only receipts with sent_at_utc >= since_ts are returned.
    Returns [] if file missing or on any error.
    """
    results: list[dict] = []
    try:
        p = _receipts_path(_path_override)
        if not p.exists():
            return results
        with open(p) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if mint is not None and rec.get("mint") != mint:
                    continue
                if pos_id is not None and rec.get("pos_id") != pos_id:
                    continue
                if since_ts is not None:
                    try:
                        wall = rec.get("sent_at_utc", "")
                        ts_str = wall.rstrip("Z")
                        ts = datetime.fromisoformat(ts_str).replace(
                            tzinfo=timezone.utc
                        ).timestamp()
                        if ts < since_ts:
                            continue
                    except (ValueError, TypeError):
                        pass
                results.append(rec)
    except Exception as exc:
        log.debug("read_receipts error: %s", exc)
    return results
