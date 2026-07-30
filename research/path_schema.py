"""
Canonical path-file schema (RF5).

All per-token trade-path CSVs written by PeakTracker (PC1) and backfill_paths (PC2)
must conform to this schema.  Consumers call load_path_file() rather than parsing
CSV fields ad-hoc.

Schema versioning
-----------------
  PATH_SCHEMA_VERSION — increment when fields change.
  Rows written before versioning have no schema_version field.  They are detected
  by absence of that field and treated as schema_version=0 (legacy).

Side rules
----------
  "buy" | "sell" | "unknown"
  Live rows (backfilled=false) with side="unknown" get data_status="partial".
  Backfill rows (backfilled=true) may legitimately have side="unknown".

Unknown future fields
---------------------
  Any field in the file that is not in PATH_HEADER is silently ignored.

Partial rows
------------
  Rows missing a required field are included in the returned list but given
  data_status="partial".  Consumers should filter on data_status != "partial"
  for analysis that requires complete data.
"""

import csv
import gzip
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# ── Schema constants ───────────────────────────────────────────────────────────

# Increment when PATH_HEADER changes
PATH_SCHEMA_VERSION: int = 2

# Fields that must be present and non-empty for a row to be valid
PATH_REQUIRED_FIELDS: list[str] = ["schema_version", "ts_ms", "price_usd", "backfilled"]

# Full canonical column list — order matches what writers produce
PATH_HEADER: list[str] = [
    "schema_version",
    "research_event_id",   # UUID of research_tokens row (empty string if unknown)
    "event_id",            # deterministic event ID
    "ts_ms",               # unix ms (int)
    "price_usd",           # float
    "price_sol",           # float (price_usd / sol_price; 0 if unavailable)
    "side",                # "buy" | "sell" | "unknown"
    "token_amount",        # float (0 if unknown)
    "sol_amount",          # float
    "vsol",                # vSolInBondingCurve (0 if backfill)
    "source",              # "live_pp" | "backfill_helius" | "unknown"
    "venue_state",         # "CURVE_ACTIVE" | "GRADUATED" | "DEX_ACTIVE" | "UNKNOWN"
    "backfilled",          # "true" | "false"
    "data_status",         # "ok" | "partial" | "inferred"
    "trader_pk",           # N7(a): PP traderPublicKey — base58 wallet, "" if unknown/legacy
]

# Allowed values for enum-like fields
_VALID_SIDES      = {"buy", "sell", "unknown"}
_VALID_SOURCES    = {"live_pp", "backfill_helius", "unknown"}
_VALID_VENUES     = {"CURVE_ACTIVE", "GRADUATED", "DEX_ACTIVE", "UNKNOWN"}
_VALID_STATUSES   = {"ok", "partial", "inferred"}
_VALID_BACKFILLED = {"true", "false"}

# Defaults used when normalising legacy (v0) rows
_LEGACY_DEFAULTS: dict = {
    "schema_version":    "0",
    "research_event_id": "",
    "event_id":          "",
    "price_sol":         "0",
    "token_amount":      "0",
    "venue_state":       "UNKNOWN",
    "data_status":       "ok",
}


# ── Public API ─────────────────────────────────────────────────────────────────

def validate_row(row: dict) -> tuple[bool, str]:
    """
    Validate a single row dict against the canonical schema.

    Returns (valid, error_message).
    Empty error_message means the row is valid.

    Rules:
    - All PATH_REQUIRED_FIELDS must be present and non-empty.
    - ts_ms must be a valid integer.
    - price_usd must be a valid float.
    - side must be in _VALID_SIDES.
    - Live rows (backfilled="false") with side="unknown" → invalid.
    - backfilled must be "true" or "false".
    """
    # Required fields present?
    for field in PATH_REQUIRED_FIELDS:
        val = row.get(field)
        if val is None or str(val).strip() == "":
            return False, f"missing required field: {field}"

    # ts_ms must be integer-parseable
    try:
        int(row["ts_ms"])
    except (ValueError, TypeError):
        return False, f"ts_ms not an integer: {row['ts_ms']!r}"

    # price_usd must be float-parseable
    try:
        float(row["price_usd"])
    except (ValueError, TypeError):
        return False, f"price_usd not a float: {row['price_usd']!r}"

    # backfilled must be "true" or "false"
    backfilled_raw = str(row.get("backfilled", "")).strip().lower()
    if backfilled_raw not in _VALID_BACKFILLED:
        return False, f"backfilled must be 'true' or 'false', got: {backfilled_raw!r}"

    # side validation
    side = str(row.get("side", "")).strip().lower()
    if side not in _VALID_SIDES:
        return False, f"side must be one of {_VALID_SIDES}, got: {side!r}"

    # Live rows must not use "unknown" side
    if backfilled_raw == "false" and side == "unknown":
        return False, "live row (backfilled=false) must have side='buy' or 'sell', not 'unknown'"

    return True, ""


def normalise_legacy_row(row: dict) -> dict:
    """
    Take a row from an old CSV (missing new fields) and return a complete canonical
    row with defaults filled in.

    Detection: no 'schema_version' field → schema_version=0, fill with safe defaults.
    Never raises — fills unknown fields with defaults.

    Legacy PC1 columns: ts_ms, price_usd, side, sol_amount, vsol
    Legacy PC2 columns: ts_ms, price_usd, side, sol_amount, vsol, source
    """
    is_legacy = "schema_version" not in row

    # Start with a copy so we never mutate the caller's dict
    out = dict(row)

    if is_legacy:
        out["schema_version"] = "0"

    # Fill missing canonical fields with safe defaults
    for field, default in _LEGACY_DEFAULTS.items():
        if field not in out or str(out[field]).strip() == "":
            out[field] = default

    # Normalise 'source' if missing
    if "source" not in out or not out["source"]:
        # Guess from what we know: vsol > 0 implies live_pp, else unknown
        try:
            vsol = float(out.get("vsol") or 0)
        except (ValueError, TypeError):
            vsol = 0.0
        out["source"] = "live_pp" if vsol > 0 else "unknown"

    # Normalise 'backfilled' if missing
    if "backfilled" not in out or out["backfilled"] not in _VALID_BACKFILLED:
        # Legacy PC2 rows had source="backfill"; PC1 rows had no source field
        src = str(out.get("source", "")).lower()
        out["backfilled"] = "true" if "backfill" in src else "false"

    # Normalise side to lowercase
    side = str(out.get("side", "")).strip().lower()
    if side not in _VALID_SIDES:
        side = "unknown"
    out["side"] = side

    # If live (backfilled=false) and side=unknown after normalisation → data_status=partial
    if out["backfilled"] == "false" and out["side"] == "unknown":
        out["data_status"] = "partial"

    # Ensure all PATH_HEADER fields exist (fill with empty string if truly unknown)
    for col in PATH_HEADER:
        if col not in out:
            out[col] = ""

    return out


def load_path_file(path: Path) -> tuple[list[dict], list[str]]:
    """
    Load a path CSV or CSV.GZ.  Returns (rows, warnings).

    - Normalises legacy rows (no schema_version → v0 treatment).
    - Validates each row.
    - Rows with validation errors are included with data_status='partial'.
    - Unknown future fields (in file but not in PATH_HEADER) are silently stripped.
    - Never raises — returns ([], warnings) on complete failure.
    - Rows are sorted by ts_ms ascending.
    """
    rows: list[dict]    = []
    warnings: list[str] = []

    try:
        p = Path(path)
        if p.suffix == ".gz":
            opener = gzip.open(p, "rt", encoding="utf-8")
        else:
            opener = open(p, "r", encoding="utf-8", newline="")

        with opener as f:
            reader = csv.DictReader(f)
            for line_num, raw_row in enumerate(reader, start=2):  # 1 = header
                try:
                    # Keep only known fields; silently drop unknown future fields
                    known = {k: v for k, v in raw_row.items() if k in PATH_HEADER}

                    # Detect and normalise legacy rows
                    if "schema_version" not in known:
                        known = normalise_legacy_row(known)
                    else:
                        # Still normalise to ensure all fields exist
                        known = normalise_legacy_row(known)

                    # Validate
                    valid, err = validate_row(known)
                    if not valid:
                        known["data_status"] = "partial"
                        warnings.append(f"line {line_num}: {err}")

                    rows.append(known)

                except Exception as row_err:
                    warnings.append(f"line {line_num}: unexpected error: {row_err}")

    except Exception as file_err:
        warnings.append(f"failed to open {path}: {file_err}")
        return [], warnings

    # Sort by ts_ms ascending; rows with unparseable ts_ms sort to end
    def _ts_key(r: dict) -> int:
        try:
            return int(r["ts_ms"])
        except (ValueError, TypeError):
            return 2**63

    rows.sort(key=_ts_key)
    return rows, warnings
