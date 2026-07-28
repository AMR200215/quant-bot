"""
Backfill smart-money wallet registry.

RF6: versioned output.  Each run writes:
  smart_wallets_vN.json          — wallet data (same format as before)
  smart_wallets_vN.metadata.json — provenance metadata
  smart_wallets_latest.json      — copy of the latest version (for zero-config loading)

Never overwrites an existing smart_wallets_vN.json.
Use --version N to force a specific version number (recovery only).

1. Queries Supabase for all outcome-complete tokens where pct_change_peak >= THRESHOLD.
2. For each winner, fetches first N buyers via Helius parsed-tx.
3. Identifies wallets that appear early in >= MIN_WINS distinct winner tokens.
4. Writes versioned files to SMART_WALLETS_BASE_DIR.

Run:
    python -m research.backfill_smart_wallets [--dry-run] [--min-wins N] [--threshold PCT]
    python -m research.backfill_smart_wallets --version 3   # force version (recovery only)

Examples:
    python -m research.backfill_smart_wallets
    python -m research.backfill_smart_wallets --dry-run --threshold 50
"""

import argparse
import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("backfill_smart_wallets")


# ── Version helpers ────────────────────────────────────────────────────────────

def _next_version(base_dir: Path) -> int:
    """
    Determine the next version number from existing smart_wallets_vN.json files.
    Returns 1 if no versioned files exist.
    """
    existing = sorted(base_dir.glob("smart_wallets_v*.json"))
    # Exclude metadata files
    existing = [f for f in existing if ".metadata." not in f.name]
    if not existing:
        return 1
    nums = []
    for f in existing:
        m = re.search(r'v(\d+)', f.name)
        if m:
            nums.append(int(m.group(1)))
    return max(nums) + 1 if nums else 1


def _git_commit_sha() -> str:
    """Return the current git commit SHA, or empty string on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def _sha256_file(path: Path) -> str:
    """Return SHA-256 hex digest of a file's content."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ── Supabase helpers ───────────────────────────────────────────────────────────

def _load_winners(sb, threshold: float) -> list:
    """Load all outcome-complete tokens with pct_change_peak >= threshold."""
    rows   = []
    offset = 0
    batch  = 1000
    while True:
        resp = (
            sb.table("research_tokens")
            .select("id,token_address,symbol,alert_time,pct_change_peak,pct_change_peak_3m,chain")
            .eq("outcome_complete", True)
            .gte("pct_change_peak", threshold)
            .eq("chain", "solana")
            .order("alert_time", desc=False)
            .range(offset, offset + batch - 1)
            .execute()
        )
        chunk = resp.data or []
        rows.extend(chunk)
        if len(chunk) < batch:
            break
        offset += batch
    return rows


def _update_supabase_rows(sb, wallet_set: set, winner_tokens: list, dry_run: bool):
    """
    Back-fill smart_money_hit / smart_money_count on winner tokens that already
    have outcome rows — so historical data reflects who was smart money.
    Best-effort — skipped on dry-run.
    """
    if dry_run:
        log.info("dry-run: skipping Supabase backfill of smart_money columns")
        return

    log.info("Supabase backfill: updating %d winner tokens...", len(winner_tokens))
    updated = 0
    for tok in winner_tokens:
        buyers = tok.get("_buyers", [])
        hits   = [b for b in buyers if b in wallet_set]
        if hits:
            try:
                sb.table("research_tokens").update({
                    "smart_money_hit":   True,
                    "smart_money_count": len(hits),
                }).eq("id", tok["id"]).execute()
                updated += 1
            except Exception as e:
                log.debug("update %s failed: %s", tok.get("token_address", "")[:8], e)
    log.info("Supabase backfill done: %d rows updated with smart_money_hit=True", updated)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build versioned smart_wallets from Supabase winners")
    parser.add_argument("--dry-run",    action="store_true",   help="print stats only, don't write files")
    parser.add_argument("--min-wins",   type=int,   default=2, help="min winner appearances (default: 2)")
    parser.add_argument("--threshold",  type=float, default=100.0, help="min pct_change_peak (default: 100)")
    parser.add_argument("--first-n",    type=int,   default=30,    help="first N buyers per token (default: 30)")
    parser.add_argument("--version",    type=int,   default=0,
                        help="force a specific version number (recovery only; default: auto)")
    args = parser.parse_args()

    from research.config import (
        SUPABASE_URL, SUPABASE_KEY, HELIUS_API_KEY,
        SMART_WALLETS_BASE_DIR, SMART_WALLETS_PATH,
    )
    from research.snapshot import fetch_first_buyers

    if not SUPABASE_URL or not SUPABASE_KEY:
        log.error("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        sys.exit(1)
    if not HELIUS_API_KEY:
        log.error("HELIUS_API_KEY must be set in .env")
        sys.exit(1)

    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # ── Determine version number ───────────────────────────────────────────────
    base_dir = SMART_WALLETS_BASE_DIR
    version = args.version if args.version > 0 else _next_version(base_dir)

    versioned_path   = base_dir / f"smart_wallets_v{version}.json"
    meta_path        = base_dir / f"smart_wallets_v{version}.metadata.json"
    latest_path      = base_dir / "smart_wallets_latest.json"

    if not args.dry_run:
        if versioned_path.exists():
            log.error(
                "smart_wallets_v%d.json already exists at %s — refusing to overwrite. "
                "Use --version N with a higher N to create a new version.",
                version, versioned_path,
            )
            sys.exit(1)

    log.info("RF6 versioned backfill: will write v%d", version)

    # ── Load winners ──────────────────────────────────────────────────────────
    log.info("Loading winner tokens (pct_change_peak >= %.0f%%)...", args.threshold)
    winners = _load_winners(sb, args.threshold)
    log.info("Found %d winner tokens", len(winners))

    if not winners:
        log.warning("No winners found — check threshold or DB contents")
        sys.exit(0)

    # Determine training date range from actual data (not hardcoded)
    alert_times = [w.get("alert_time") for w in winners if w.get("alert_time")]
    training_start  = min(alert_times) if alert_times else ""
    training_cutoff = max(alert_times) if alert_times else ""

    # Source data hash: SHA-256 of sorted winner token addresses
    sorted_addresses = sorted(w.get("token_address", "") for w in winners)
    source_data_hash = _sha256_str("\n".join(sorted_addresses))

    # Config hash: SHA-256 of min_wins + threshold + first_n
    config_str = json.dumps(
        {"min_wins": args.min_wins, "threshold": args.threshold, "first_n": args.first_n},
        sort_keys=True,
    )
    config_hash = _sha256_str(config_str)

    # ── Fetch first N buyers for each winner ──────────────────────────────────
    wallet_token_map: dict = defaultdict(list)   # wallet → [token_address, ...]
    helius_tokens_fetched = 0
    helius_tokens_empty   = 0
    helius_request_count  = 0
    failed = 0

    for i, tok in enumerate(winners):
        addr  = tok.get("token_address", "")
        sym   = tok.get("symbol") or addr[:8]
        peak  = tok.get("pct_change_peak") or 0
        if not addr:
            continue
        log.info("[%d/%d] Fetching buyers for %s (%s, peak=+%.0f%%)",
                 i + 1, len(winners), sym, addr[:8], peak)
        buyers = fetch_first_buyers(addr, HELIUS_API_KEY, n=args.first_n)
        helius_request_count += 1
        tok["_buyers"] = buyers
        if not buyers:
            log.debug("  no buyers found for %s", addr[:8])
            helius_tokens_empty += 1
            failed += 1
        else:
            log.debug("  %d buyers found", len(buyers))
            helius_tokens_fetched += 1
            for w in buyers:
                wallet_token_map[w].append(addr)
        # Rate-limit: Helius paid plan is generous but be courteous
        time.sleep(0.3)

    log.info("Buyer fetch complete: %d/%d tokens had data (%d failed/empty)",
             len(winners) - failed, len(winners), failed)

    # ── Identify smart wallets ────────────────────────────────────────────────
    smart_wallets: dict = {}
    for wallet, tokens in wallet_token_map.items():
        unique_tokens = list(dict.fromkeys(tokens))   # deduplicate, preserve order
        if len(unique_tokens) >= args.min_wins:
            smart_wallets[wallet] = {
                "first_bought_count": len(unique_tokens),
                "tokens": unique_tokens[:20],   # cap stored list at 20 for file size
            }

    log.info(
        "Smart wallets identified: %d wallets appeared early in >= %d winner tokens",
        len(smart_wallets), args.min_wins,
    )

    if smart_wallets:
        counts = sorted([v["first_bought_count"] for v in smart_wallets.values()], reverse=True)
        log.info("Top counts: %s", counts[:10])

    output = {
        "version":          version,
        "generated_at":     datetime.now(timezone.utc).isoformat(),
        "winner_threshold": args.threshold,
        "min_wins":         args.min_wins,
        "first_n":          args.first_n,
        "winner_count":     len(winners),
        "wallets_scanned":  len(wallet_token_map),
        "wallet_count":     len(smart_wallets),
        "wallets":          smart_wallets,
    }

    if args.dry_run:
        log.info("dry-run: would write v%d with %d wallets to %s",
                 version, len(smart_wallets), versioned_path)
        print(json.dumps({k: v for k, v in output.items() if k != "wallets"}, indent=2))
        return

    # ── Write versioned data file ─────────────────────────────────────────────
    versioned_path.write_text(json.dumps(output, indent=2))
    log.info("Written: %s (%d smart wallets)", versioned_path, len(smart_wallets))

    # ── Compute output hash ───────────────────────────────────────────────────
    output_sha256 = _sha256_file(versioned_path)

    # ── Write metadata file ───────────────────────────────────────────────────
    metadata = {
        "version":                   version,
        "created_at":                datetime.now(timezone.utc).isoformat(),
        "training_start":            training_start,
        "training_cutoff":           training_cutoff,
        "eligible_token_count":      len(winners),
        "wallet_count":              len(smart_wallets),
        "minimum_wallet_sample_rule": (
            f"appear in >= {args.min_wins} winner tokens "
            f"(pct_change_peak >= {args.threshold:.0f}%)"
        ),
        "source_data_hash":          source_data_hash,
        "config_hash":               config_hash,
        "output_sha256":             output_sha256,
        "git_commit_sha":            _git_commit_sha(),
        "helius_request_count":      helius_request_count,
        "helius_tokens_fetched":     helius_tokens_fetched,
        "helius_tokens_empty":       helius_tokens_empty,
        "total_candidates_scanned":  len(wallet_token_map),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))
    log.info("Written: %s", meta_path)

    # ── Update latest copy ────────────────────────────────────────────────────
    shutil.copy2(versioned_path, latest_path)
    log.info("Updated: %s → v%d copy", latest_path, version)

    # Also keep the legacy SMART_WALLETS_PATH (smart_wallets.json) in sync
    # so existing code that imports SMART_WALLETS_PATH continues to work
    if SMART_WALLETS_PATH != latest_path and SMART_WALLETS_PATH != versioned_path:
        shutil.copy2(versioned_path, SMART_WALLETS_PATH)
        log.info("Kept legacy smart_wallets.json in sync")

    # ── Backfill Supabase rows ────────────────────────────────────────────────
    smart_wallet_set = set(smart_wallets.keys())
    _update_supabase_rows(sb, smart_wallet_set, winners, dry_run=args.dry_run)

    print("\n=== Smart Wallet Backfill Summary ===")
    print(f"Version:            v{version}")
    print(f"Winners scanned:    {len(winners)}")
    print(f"Winners with buyers:{len(winners) - failed}")
    print(f"Smart wallets found:{len(smart_wallets)}")
    print(f"Output: {versioned_path}")
    print(f"Latest: {latest_path}")


if __name__ == "__main__":
    main()
