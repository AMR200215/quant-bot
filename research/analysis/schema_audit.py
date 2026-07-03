"""
research/analysis/schema_audit.py — Schema source-of-truth audit.

Compares every field emitted by the research pipeline against the Supabase schema.
Exits nonzero if any active emitted field is missing from the schema.

Run: python -m research.analysis.schema_audit
"""

import re
import sys
from pathlib import Path

# ── Schema parser ─────────────────────────────────────────────────────────────

def _parse_schema(sql_path: Path) -> set:
    """Extract column names from CREATE TABLE research_tokens in the SQL file."""
    text    = sql_path.read_text()
    # Find the CREATE TABLE block
    m = re.search(r"CREATE TABLE IF NOT EXISTS research_tokens\s*\((.+?)\);",
                  text, re.DOTALL | re.IGNORECASE)
    if not m:
        return set()
    block = m.group(1)
    cols  = set()
    for line in block.splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith("--") or line.upper().startswith("PRIMARY") \
                or line.upper().startswith("UNIQUE") or line.upper().startswith("CHECK"):
            continue
        col = re.split(r"\s+", line)[0].strip('"')
        if col:
            cols.add(col.lower())
    return cols


# ── Emitter field extractor ───────────────────────────────────────────────────

def _extract_emitted(src_path: Path, patterns: list) -> set:
    """Extract field names from source file using list of regex patterns."""
    text   = src_path.read_text()
    fields = set()
    for pat in patterns:
        for m in re.finditer(pat, text):
            fields.add(m.group(1).lower())
    return fields


def _audit_tracker(src: Path) -> dict:
    """Fields emitted by tracker.py _insert row dict and _pp_extras."""
    # Match "key": or "key" : in dict literals
    patterns = [
        r'"(\w+)"\s*:',   # double-quoted keys
        r"'(\w+)'\s*:",   # single-quoted keys
    ]
    raw = _extract_emitted(src, patterns)
    # Remove non-field keys (Python keywords, Supabase methods, etc.)
    exclude = {
        "token_address", "returning", "representation",
        "pgrst204", "schema", "cache", "unique", "duplicate",
        "action", "subscribetokentrade", "tokenaddress",
        "errors", "mint",
        # String literals that look like dict keys but aren't schema fields
        "solana", "bsc", "eth", "pumpfun", "pumpswap",
        "minimal", "research_tokens",
    }
    return {"tracker.py": raw - exclude}


def _audit_peak_tracker(src: Path) -> dict:
    fields = set()
    for line in src.read_text().splitlines():
        # Lines like  "price_peak_3m":  or  "pct_change_peak_3m":
        m = re.search(r'"(price_peak_3m|pct_change_peak_3m|t_peak_3m_s)"', line)
        if m:
            fields.add(m.group(1).lower())
    return {"peak_tracker.py": fields}


def _audit_outcome_poller(src: Path) -> dict:
    text   = src.read_text()
    fields = set()
    # _INTERVAL_COL values
    for m in re.finditer(r'"(price_t\w+)"\s*[,}]', text):
        fields.add(m.group(1).lower())
    # pct_col_map values
    for m in re.finditer(r'"(pct_change_t\w+)"\s*[,}]', text):
        fields.add(m.group(1).lower())
    # finalize update keys
    for key in ("pct_change_peak", "peak_interval", "outcome_complete",
                 "data_partial", "time_to_peak_min"):
        fields.add(key)
    return {"outcome_poller.py": fields}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    research_dir = Path(__file__).parent.parent
    schema_path  = research_dir / "supabase_schema.sql"

    schema_cols = _parse_schema(schema_path)

    emitters = {}
    emitters.update(_audit_tracker(research_dir / "tracker.py"))
    emitters.update(_audit_peak_tracker(research_dir / "peak_tracker.py"))
    emitters.update(_audit_outcome_poller(research_dir / "outcome_poller.py"))

    # Union of all emitted fields
    all_emitted: set = set()
    for fields in emitters.values():
        all_emitted |= fields

    missing_in_schema  = all_emitted - schema_cols
    unused_schema_cols = schema_cols - all_emitted - {
        # Schema-only housekeeping columns that no emitter writes
        "id", "created_at", "chain",
        "v7_traded", "v7_traded_at",          # written by outcome_poller sync job
        "code_version",                         # written by tracker when enabled
        "channel_velocity_5m",                  # written by tracker (now fixed)
        "tg_message_text",                      # written by tracker
        "snapshot_ok", "snapshot_attempts",     # written by tracker
        "symbol",                               # written by tracker
        "alert_time", "category",               # written by tracker
        "token_address",                        # written by tracker
        "time_to_peak_min",                     # optional outcome field
    }

    # Fields explicitly deprecated (stripped silently in old code, now spooled)
    deprecated: set = set()   # add field names here when intentionally retiring

    active_missing = missing_in_schema - deprecated

    # ── Print report ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SCHEMA AUDIT — research_tokens")
    print(f"{'='*60}")
    print(f"\nSchema columns : {len(schema_cols)}")
    print(f"Emitted fields : {len(all_emitted)}")
    print(f"Missing in schema (active) : {len(active_missing)}")
    print(f"Unused schema cols         : {len(unused_schema_cols)}")

    print(f"\n── Emitted fields by file {'─'*34}")
    for fname, fields in sorted(emitters.items()):
        print(f"\n  {fname}:")
        for f in sorted(fields):
            flag = " *** MISSING FROM SCHEMA ***" if f in active_missing else ""
            print(f"    {f}{flag}")

    if active_missing:
        print(f"\n{'!'*60}")
        print("FAIL — Active emitted fields missing from schema:")
        for f in sorted(active_missing):
            print(f"  - {f}")
        print(f"{'!'*60}")
        print("\nRun the ALTER TABLE migration block in supabase_schema.sql")
        sys.exit(1)
    else:
        print(f"\n{'='*60}")
        print("PASS — All active emitted fields present in schema")
        print(f"{'='*60}")

    if unused_schema_cols:
        print(f"\nINFO — Schema cols not currently emitted (may be legacy or poller-only):")
        for c in sorted(unused_schema_cols):
            print(f"  {c}")


if __name__ == "__main__":
    main()
