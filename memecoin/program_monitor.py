"""program_monitor.py — detects on-chain upgrades to programs the bot depends on.

Polls once per hour (free — one lightweight RPC call per program).
On any program upgrade, logs ERROR and sends a Telegram alert so you
can review whether the TX structure needs updating before the next sell.

Programs watched
────────────────
  PumpSwap AMM      pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA
  pump.fun bonding  6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P
  pump.fun fees     pfeeUxB6jkeY1Hxd7CsFCAjcbHA9rWtchMGdZ6VojVZ
  Jupiter v6        JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4

How it works
────────────
Each upgradeable program has a ProgramData account whose first 12 bytes are:
  [0:4]  discriminant (u32 LE = 3)
  [4:12] last_deploy_slot (u64 LE)

We derive the ProgramData address via BPF Upgradeable Loader PDA, fetch it
once per hour, and compare the deploy slot. If it changes → program upgraded.
State is persisted to logs/program_slots.json so restarts don't re-alert.
"""

import json
import logging
import os
import struct
import threading
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

# ─── BPF Upgradeable Loader ───────────────────────────────────────────────────
_BPF_UPGRADEABLE_LOADER = "BPFLoaderUpgradeab1e11111111111111111111111"

# ─── Programs to monitor ──────────────────────────────────────────────────────
WATCHED_PROGRAMS = {
    "PumpSwap AMM":     "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA",
    "pump.fun bonding": "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P",
    "pump.fun fees":    "pfeeUxB6jkeY1Hxd7CsFCAjcbHA9rWtchMGdZ6VojVZ",
    "Jupiter v6":       "JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",
}

# ─── Config ───────────────────────────────────────────────────────────────────
_POLL_INTERVAL_SEC = 3600          # 1 hour
_STATE_FILE        = Path(__file__).parent.parent / "logs" / "program_slots.json"

# Use public RPC — this is low-frequency read-only polling, no Helius credits needed
_PUBLIC_RPC = "https://api.mainnet-beta.solana.com"


# ─── PDA derivation ───────────────────────────────────────────────────────────
def _derive_program_data_address(program_address: str) -> str:
    """Return the ProgramData PDA for a BPF Upgradeable program."""
    from solders.pubkey import Pubkey
    loader = Pubkey.from_string(_BPF_UPGRADEABLE_LOADER)
    prog   = Pubkey.from_string(program_address)
    pda, _ = Pubkey.find_program_address([bytes(prog)], loader)
    return str(pda)


# ─── RPC helper ───────────────────────────────────────────────────────────────
def _get_account_data_b64(address: str) -> bytes | None:
    """Fetch account data as raw bytes via getAccountInfo.  Returns None on error."""
    payload = {
        "jsonrpc": "2.0", "id": 1,
        "method":  "getAccountInfo",
        "params":  [address, {"encoding": "base64"}],
    }
    try:
        r = requests.post(_PUBLIC_RPC, json=payload, timeout=15)
        r.raise_for_status()
        result = r.json().get("result", {})
        value  = result.get("value")
        if not value:
            return None
        import base64
        data_b64 = value["data"][0]
        return base64.b64decode(data_b64)
    except Exception as e:
        log.warning("program_monitor: getAccountInfo(%s) failed: %s", address[:8], e)
        return None


def _read_deploy_slot(program_data_address: str) -> int | None:
    """Return the last_deploy_slot from a ProgramData account, or None on error."""
    data = _get_account_data_b64(program_data_address)
    if data is None or len(data) < 12:
        return None
    # bytes 4-11: last_deploy_slot (u64 LE)
    slot, = struct.unpack_from("<Q", data, 4)
    return slot


# ─── State persistence ────────────────────────────────────────────────────────
def _load_state() -> dict:
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_state(state: dict) -> None:
    try:
        _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        log.warning("program_monitor: could not save state: %s", e)


# ─── Alert ────────────────────────────────────────────────────────────────────
def _send_alert(name: str, program_addr: str, old_slot: int, new_slot: int) -> None:
    msg = (
        f"⚠️ PROGRAM UPGRADED: {name}\n"
        f"Address: {program_addr}\n"
        f"Old slot: {old_slot:,}\n"
        f"New slot: {new_slot:,}\n"
        f"ACTION: review TX structure in pumpswap_local.py / executor.py "
        f"before next sell."
    )
    log.error("PROGRAM UPGRADED — %s  old_slot=%d  new_slot=%d  addr=%s",
              name, old_slot, new_slot, program_addr)
    try:
        from app.alerts import _send
        _send(msg)
    except Exception:
        pass  # alerts may not be configured — log is enough


# ─── Main poll loop ───────────────────────────────────────────────────────────
def _poll_once(state: dict) -> dict:
    changed = False
    for name, program_addr in WATCHED_PROGRAMS.items():
        try:
            pda_addr = _derive_program_data_address(program_addr)
            slot     = _read_deploy_slot(pda_addr)
            if slot is None:
                log.debug("program_monitor: could not read slot for %s — skipping", name)
                continue

            prev_slot = state.get(program_addr)
            if prev_slot is None:
                # First run — just record the slot
                log.info("program_monitor: %s  slot=%d  (baseline recorded)", name, slot)
                state[program_addr] = slot
                changed = True
            elif slot != prev_slot:
                _send_alert(name, program_addr, prev_slot, slot)
                state[program_addr] = slot
                changed = True
            else:
                log.debug("program_monitor: %s  slot=%d  (no change)", name, slot)

        except Exception as e:
            log.warning("program_monitor: error checking %s: %s", name, e)

    if changed:
        _save_state(state)
    return state


def _monitor_thread() -> None:
    log.info("program_monitor: started — watching %d programs, poll every %ds",
             len(WATCHED_PROGRAMS), _POLL_INTERVAL_SEC)
    state = _load_state()

    while True:
        try:
            state = _poll_once(state)
        except Exception as e:
            log.error("program_monitor: unexpected error: %s", e)
        time.sleep(_POLL_INTERVAL_SEC)


def start(daemon: bool = True) -> None:
    t = threading.Thread(target=_monitor_thread, daemon=daemon, name="program-monitor")
    t.start()
    log.info("program_monitor: background thread started")
