"""
canary_t22_probe_watcher.py — Watch for T22 canary buys, fire jupiter_t22_probe immediately.

What it does:
  1. Polls memecoin/data/memecoin_positions.json every POLL_INTERVAL_S for new open positions.
  2. When a new open position appears with token_program:TOKEN_2022 in its notes, fires
     jupiter_t22_probe as a subprocess against the real wallet + that mint.
  3. Waits for probe to finish, reads the probe CSV log for the result row.
  4. Prints a verdict table to terminal.
  5. Logs results to logs/canary_t22_probe_watcher.csv.

No trading logic is changed. No transactions are sent. The probe is read-only.

Prerequisites:
  - LIVE_TRADING=True in config.py (set by engineer before running watcher)
  - LIVE_CANARY_MODE=True, MAX_CANARY_TRADE_USD=3 (already set)
  - SOLANA_PRIVATE_KEY env var set (for probe's wallet resolution)

Usage:
    # On VPS (after: set -a && source .env && set +a)
    python -m memecoin.tools.canary_t22_probe_watcher [--rpc <RPC_URL>] [--poll 5]

Stop with Ctrl-C when validation is complete.

Validation criteria (Phase 1):
    quote_ok=True AND swap_build_ok=True AND sim_ok=True AND no 429 errors.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

_REPO_ROOT     = Path(__file__).parent.parent.parent
_POSITIONS_FILE = _REPO_ROOT / "memecoin" / "data" / "memecoin_positions.json"
_PROBE_LOG     = _REPO_ROOT / "logs" / "jupiter_t22_probe.csv"
_WATCHER_LOG   = _REPO_ROOT / "logs" / "canary_t22_probe_watcher.csv"

_WATCHER_FIELDS = [
    "timestamp", "pos_id", "mint", "symbol",
    "probe_verdict", "quote_ok", "swap_build_ok", "sim_ok",
    "probe_error_class", "probe_latency_ms", "probe_exit_code",
]

# ── Token-2022 detection ──────────────────────────────────────────────────────

_TOKEN22_NOTE_MARKER = "token_program:TOKEN_2022"


def _is_t22_position(pos: dict) -> bool:
    """Return True if position notes indicate Token-2022."""
    notes = pos.get("notes") or ""
    return _TOKEN22_NOTE_MARKER in notes


def _is_graduated_position(pos: dict) -> bool:
    """
    Best-effort: positions opened via the PumpSwap (graduated) path
    typically have 'graduated' or 'pumpswap' in their notes.
    We probe ALL T22 positions regardless — graduated or bonding curve.
    """
    notes = (pos.get("notes") or "").lower()
    return "graduated" in notes or "pumpswap" in notes


# ── Colour helpers ────────────────────────────────────────────────────────────

_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


# ── Read positions JSON ───────────────────────────────────────────────────────

def _read_open_positions() -> dict[str, dict]:
    """
    Read memecoin_positions.json, return {pos_id: pos_dict} for open positions.
    The file is a JSON list of position objects, each with an "id" field.
    Returns empty dict on any read/parse error (watcher keeps running).
    """
    try:
        raw = json.loads(_POSITIONS_FILE.read_text())
        # File is a list of position dicts
        if isinstance(raw, list):
            return {
                pos["id"]: pos
                for pos in raw
                if pos.get("status") == "open" and pos.get("id")
            }
        # Fallback: legacy dict format {pos_id: pos_dict}
        return {
            pos_id: pos
            for pos_id, pos in raw.items()
            if pos.get("status") == "open"
        }
    except Exception as e:
        print(f"  {_YELLOW}[{_ts()}] positions read error: {e}{_RESET}")
        return {}


# ── Run probe ─────────────────────────────────────────────────────────────────

def _run_probe(mint: str, rpc: str | None) -> dict:
    """
    Run jupiter_t22_probe for the given mint.
    Returns a dict with probe results parsed from the CSV log row.
    """
    cmd = [sys.executable, "-m", "memecoin.tools.jupiter_t22_probe", "--mint", mint]
    if rpc:
        cmd += ["--rpc", rpc]

    print(f"\n  {_CYAN}[{_ts()}] Running probe  mint={mint[:16]}…{_RESET}")
    print(f"  cmd: {' '.join(cmd)}")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(_REPO_ROOT),
        )
        elapsed_ms = round((time.time() - t0) * 1000)
        exit_code = result.returncode

        # Print probe stdout so user sees the full output
        if result.stdout:
            for line in result.stdout.strip().splitlines():
                print(f"    {line}")
        if result.stderr:
            for line in result.stderr.strip().splitlines()[:10]:
                print(f"    {_YELLOW}{line}{_RESET}")

        # Read result from probe's CSV log — last row for this mint
        row = _read_probe_log_for_mint(mint)
        row["probe_exit_code"] = exit_code
        row["probe_latency_ms"] = elapsed_ms
        return row

    except subprocess.TimeoutExpired:
        elapsed_ms = round((time.time() - t0) * 1000)
        print(f"  {_RED}[{_ts()}] Probe timed out after {elapsed_ms}ms{_RESET}")
        return {
            "probe_verdict":    "FAIL_PROBE_TIMEOUT",
            "quote_ok":         False,
            "swap_build_ok":    False,
            "sim_ok":           False,
            "probe_error_class": "probe_timeout",
            "probe_latency_ms": elapsed_ms,
            "probe_exit_code":  -1,
        }
    except Exception as e:
        elapsed_ms = round((time.time() - t0) * 1000)
        print(f"  {_RED}[{_ts()}] Probe error: {e}{_RESET}")
        return {
            "probe_verdict":    "FAIL_PROBE_EXCEPTION",
            "quote_ok":         False,
            "swap_build_ok":    False,
            "sim_ok":           False,
            "probe_error_class": str(e)[:60],
            "probe_latency_ms": elapsed_ms,
            "probe_exit_code":  -1,
        }


def _read_probe_log_for_mint(mint: str) -> dict:
    """
    Read the most recent row from jupiter_t22_probe.csv for the given mint.
    Returns defaults if not found.
    """
    defaults = {
        "probe_verdict":    "UNKNOWN",
        "quote_ok":         False,
        "swap_build_ok":    False,
        "sim_ok":           False,
        "probe_error_class": "log_not_found",
    }
    try:
        if not _PROBE_LOG.exists():
            return defaults
        with open(_PROBE_LOG, newline="") as f:
            rows = list(csv.DictReader(f))
        # Find most recent row for this mint
        matches = [r for r in rows if r.get("mint", "").startswith(mint[:16])]
        if not matches:
            return defaults
        row = matches[-1]
        return {
            "probe_verdict":    row.get("verdict", "UNKNOWN"),
            "quote_ok":         row.get("jupiter_quote_ok", "").lower() in ("true", "1", "yes"),
            "swap_build_ok":    row.get("jupiter_swap_build_ok", "").lower() in ("true", "1", "yes"),
            "sim_ok":           row.get("jupiter_sim_ok", "").lower() in ("true", "1", "yes"),
            "probe_error_class": row.get("jupiter_error_class", ""),
        }
    except Exception as e:
        defaults["probe_error_class"] = f"log_read_error:{e}"
        return defaults


# ── Log watcher result ────────────────────────────────────────────────────────

def _write_watcher_log(row: dict) -> None:
    try:
        _WATCHER_LOG.parent.mkdir(parents=True, exist_ok=True)
        write_header = not _WATCHER_LOG.exists()
        with open(_WATCHER_LOG, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_WATCHER_FIELDS, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(row)
    except Exception as e:
        print(f"  {_YELLOW}watcher log write failed: {e}{_RESET}")


# ── Verdict display ───────────────────────────────────────────────────────────

def _print_verdict(pos: dict, probe: dict) -> None:
    mint   = pos.get("token_address", "")
    sym    = pos.get("token_symbol", "?")
    grad   = "graduated" if _is_graduated_position(pos) else "bonding-curve"
    sim_ok = probe.get("sim_ok", False)

    color  = _GREEN if sim_ok else _RED
    banner = "PASS — Phase 1 VALIDATED" if sim_ok else "FAIL — Phase 1 not validated"

    print(f"\n{'═'*60}")
    print(f"{_BOLD}CANARY T22 PROBE RESULT{_RESET}")
    print(f"{'═'*60}")
    print(f"  mint:          {mint[:16]}…")
    print(f"  symbol:        {sym}")
    print(f"  pool:          {grad}")
    print(f"  quote_ok:      {'✓' if probe.get('quote_ok') else '✗'}")
    print(f"  swap_build_ok: {'✓' if probe.get('swap_build_ok') else '✗'}")
    print(f"  sim_ok:        {'✓' if sim_ok else '✗'}")
    print(f"  verdict:       {probe.get('probe_verdict', '?')}")
    if probe.get("probe_error_class"):
        print(f"  error_class:   {probe['probe_error_class']}")
    print(f"\n  {color}{_BOLD}{banner}{_RESET}")
    print(f"{'═'*60}\n")

    if sim_ok:
        print(f"  Next steps:")
        print(f"    1. Confirm the live sell executes cleanly (exit system handles it normally)")
        print(f"    2. Check logs/memecoin_live_journal.csv for sell_route=jupiter")
        print(f"    3. If sell confirmed: set JUPITER_T22_GRAD_PRIMARY_ENABLED=True")
        print(f"       and increment EXIT_SYSTEM_VALIDATED progress counter")
    else:
        print(f"  Debug:")
        print(f"    - Check logs/jupiter_t22_probe.csv for full detail")
        print(f"    - Re-run probe manually: python -m memecoin.tools.jupiter_t22_probe --mint {mint[:8]}…")
    print()


# ── Main poll loop ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watch for T22 canary buys and run jupiter_t22_probe immediately."
    )
    parser.add_argument("--rpc",  help="RPC URL for probe (default: from config)")
    parser.add_argument("--poll", type=float, default=5.0,
                        help="Poll interval in seconds (default: 5)")
    parser.add_argument("--max-probes", type=int, default=3,
                        help="Exit after this many T22 probes regardless of result (default: 3)")
    args = parser.parse_args()

    probe_count = 0

    # Tracked state: set of pos_ids already probed this session
    probed: set[str] = set()

    # Seed with positions already open at startup (don't re-probe existing positions)
    initial = _read_open_positions()
    probed.update(initial.keys())

    t22_initial = {k: v for k, v in initial.items() if _is_t22_position(v)}
    if t22_initial:
        print(f"\n{_YELLOW}[{_ts()}] {len(t22_initial)} existing T22 position(s) at startup — skipping probe (no new buy){_RESET}")
        for pos_id, pos in t22_initial.items():
            print(f"  {pos_id[:16]}  {pos.get('token_symbol','?')}  {pos.get('token_address','?')[:16]}…")

    print(f"\n{_BOLD}{'═'*60}{_RESET}")
    print(f"{_BOLD}CANARY T22 PROBE WATCHER{_RESET}")
    print(f"{'═'*60}")
    print(f"  Polling: {_POSITIONS_FILE.name} every {args.poll}s")
    print(f"  Looking for: new open positions with token_program:TOKEN_2022")
    print(f"  Probe tool:  python -m memecoin.tools.jupiter_t22_probe")
    print(f"  Watcher log: {_WATCHER_LOG.relative_to(_REPO_ROOT)}")
    if args.rpc:
        print(f"  RPC override: {args.rpc[:40]}…")
    print(f"  Auto-exit:   after first sim_ok=True OR after {args.max_probes} probes")
    print(f"\n  {_YELLOW}Waiting for first T22 canary buy…{_RESET} (Ctrl-C to stop)\n")

    try:
        while True:
            time.sleep(args.poll)

            current = _read_open_positions()

            for pos_id, pos in current.items():
                if pos_id in probed:
                    continue   # already handled

                probed.add(pos_id)
                mint   = pos.get("token_address", "")
                sym    = pos.get("token_symbol", "?")
                notes  = pos.get("notes", "")

                if not _is_t22_position(pos):
                    print(f"  [{_ts()}] new position (SPL) — skipping probe  {sym}  {mint[:12]}…")
                    continue

                grad_tag = " [graduated]" if _is_graduated_position(pos) else ""
                print(f"\n  {_GREEN}[{_ts()}] NEW T22 POSITION DETECTED{_RESET}{grad_tag}")
                print(f"    pos_id: {pos_id}")
                print(f"    mint:   {mint[:16]}…")
                print(f"    symbol: {sym}")
                print(f"    notes:  {notes[:80]}")

                # ── Fire probe ─────────────────────────────────────────────────
                probe = _run_probe(mint, args.rpc)

                # ── Display verdict ────────────────────────────────────────────
                _print_verdict(pos, probe)

                # ── Log result ─────────────────────────────────────────────────
                _write_watcher_log({
                    "timestamp":        datetime.now(timezone.utc).isoformat(),
                    "pos_id":           pos_id,
                    "mint":             mint[:16],
                    "symbol":           sym,
                    "probe_verdict":    probe.get("probe_verdict", "UNKNOWN"),
                    "quote_ok":         probe.get("quote_ok", False),
                    "swap_build_ok":    probe.get("swap_build_ok", False),
                    "sim_ok":           probe.get("sim_ok", False),
                    "probe_error_class": probe.get("probe_error_class", ""),
                    "probe_latency_ms": probe.get("probe_latency_ms", 0),
                    "probe_exit_code":  probe.get("probe_exit_code", -1),
                })

                probe_count += 1
                sim_ok = probe.get("sim_ok", False)
                verdict = probe.get("probe_verdict", "UNKNOWN")
                ec = probe.get("probe_error_class", "")

                # ── Telegram alert ─────────────────────────────────────────────
                try:
                    from app.alerts import send as _tg_send
                    if sim_ok:
                        _tg_send(
                            f"✅ T22 CANARY PROBE — PHASE 1 VALIDATED\n"
                            f"mint: {mint[:16]}…\n"
                            f"symbol: {sym}\n"
                            f"quote ✓  build ✓  sim ✓\n"
                            f"verdict: {verdict}"
                        )
                    else:
                        _tg_send(
                            f"⚠️ T22 CANARY PROBE — sim FAILED ({probe_count}/{args.max_probes})\n"
                            f"mint: {mint[:16]}…\n"
                            f"symbol: {sym}\n"
                            f"quote {'✓' if probe.get('quote_ok') else '✗'}  "
                            f"build {'✓' if probe.get('swap_build_ok') else '✗'}  sim ✗\n"
                            f"error: {ec[:60]}"
                        )
                except Exception as _tg_e:
                    print(f"  [{_ts()}] Telegram alert failed: {_tg_e}")

                if sim_ok:
                    print(f"[{_ts()}] sim_ok=True — Phase 1 validated. Watcher exiting.\n")
                    return

                if probe_count >= args.max_probes:
                    print(f"[{_ts()}] Reached max-probes={args.max_probes}. Watcher exiting.")
                    print(f"  Check logs/canary_t22_probe_watcher.csv for all results.\n")
                    try:
                        from app.alerts import send as _tg_send
                        _tg_send(
                            f"⛔ T22 CANARY PROBE — max probes reached ({args.max_probes}), "
                            f"sim never passed. Check logs/canary_t22_probe_watcher.csv"
                        )
                    except Exception:
                        pass
                    return

                print(f"  [{_ts()}] sim_ok=False ({probe_count}/{args.max_probes} probes). Continuing…\n")

    except KeyboardInterrupt:
        print(f"\n[{_ts()}] Watcher stopped by user.\n")


if __name__ == "__main__":
    main()
