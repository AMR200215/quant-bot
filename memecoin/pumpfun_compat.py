"""
Z8 — pump.fun bonding-curve program upgrade compatibility gate.

Tracks whether the current on-chain pump.fun bonding-curve program interface has
been verified against a known-good deployment for the LOCAL TX BUILDER.

PUMPFUN_COMPATIBILITY states
-----------------------------
VERIFIED   The post-upgrade instruction interface (discriminators, account order,
           signer/writable flags, account count) was confirmed compatible with the
           local builder.  local_build_allowed() → True.

UNKNOWN    Compatibility has not been established for the current deployment.
           Local building is disabled; PumpPortal/PumpSwap/Jupiter safe paths used.

CHANGED    A confirmed interface change was detected (discriminator mismatch,
           account count change, flag change, or new deploy slot).
           Local building is disabled until a new VERIFIED verification is recorded.

Default state: CHANGED (confirmed post-upgrade sell discriminator and account-count
differences were found on 2026-07-15 at slot 433,095,571).

Epoch-gate rule: new live entries use PumpPortal when state != VERIFIED.
Existing positions continue via PumpSwap/Jupiter regardless of state.

Tripwire (Z8.6)
---------------
At process startup and every TRIPWIRE_INTERVAL_S seconds, the live deploy slot is
compared against BASELINE_DEPLOY_SLOT.  A slot change triggers:
  - state = CHANGED immediately (never silently continues with stale assumptions)
  - local_build_allowed() → False
  - layout_graduation_allowed() → False
  - High-priority Telegram alert

BC account validation (Z8.4)
-----------------------------
validate_bc_account(raw, owner) provides structural validation before trusting
byte 48 as the complete flag:
  - owner must be _PUMPFUN_PROGRAM
  - raw[:8] must equal _BC_DISCRIMINATOR
  - len(raw) must be in _BC_VALID_LENGTHS (49 or 151)
  - raw[48] must be exactly 0 or 1

Confirmed post-upgrade evidence (verified 2026-07-15)
------------------------------------------------------
- Upgrade slot:      433,095,571  (confirmed from ProgramData account)
- ProgramData addr:  B5MvUwXdiW1NMM6QFFD3ssPKBujD4zMohncbM73Z2BQu
- ELF SHA-256:       5d65238ffd3513fa5980c01ba1eb2da9cc050091ad9c46b75916ce4a09b02bbf
- Buy discriminator: 66063d1201daebea = sha256("global:buy")[:8]  UNCHANGED
- Buy account count: 18 (V1 was 12) — CHANGED (+6 new accounts, creator/buyback system)
- Sell discriminator:5df6823ce7e940b2 = sha256("global:sell_v2")[:8]  CHANGED
- Sell account count:26 (V1 was 12) — CHANGED
- BC account size:   151 bytes (V1 was 49) — bytes 0-48 unchanged, 102 new bytes appended
- BC discriminator:  17b7f83760d8ac60 — UNCHANGED
- BC complete flag:  byte 48 — UNCHANGED
"""

import base64
import logging
import struct
import threading
import time

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Program identity constants
# ---------------------------------------------------------------------------
_PUMPFUN_PROGRAM  = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
_PROGRAMDATA_ADDR = "B5MvUwXdiW1NMM6QFFD3ssPKBujD4zMohncbM73Z2BQu"

# Baseline snapshot: the upgrade that triggered Z8 (verified 2026-07-15).
# Tripwire compares the live deploy slot against this value.
# If it changes, state immediately → CHANGED.
BASELINE_DEPLOY_SLOT = 433_095_571
BASELINE_EXE_HASH    = (
    "5d65238ffd3513fa5980c01ba1eb2da9cc050091ad9c46b75916ce4a09b02bbf"
)

# ---------------------------------------------------------------------------
# Instruction discriminators (Z8.3 evidence)
# ---------------------------------------------------------------------------
# V1 (pre-upgrade, sha256("global:<name>")[:8])
DISC_BUY_V1  = bytes.fromhex("66063d1201daebea")  # sha256("global:buy")[:8]
DISC_SELL_V1 = bytes.fromhex("33e685a4017f83ad")  # sha256("global:sell")[:8]

# V2 (post-upgrade, confirmed from on-chain transactions at slot 433,122,735)
# Buy discriminator is UNCHANGED in V2.
DISC_BUY_V2  = DISC_BUY_V1                         # same as V1 ← confirmed
DISC_SELL_V2 = bytes.fromhex("5df6823ce7e940b2")   # sha256("global:sell_v2")[:8]

# V1 account counts for buy (12) and sell (12)
V1_BUY_ACCOUNT_COUNT  = 12
V1_SELL_ACCOUNT_COUNT = 12
# V2 account counts (confirmed from post-upgrade transactions)
V2_BUY_ACCOUNT_COUNT  = 18  # +6 new accounts (creator vault + buyback fee system)
V2_SELL_ACCOUNT_COUNT = 26  # +14 new accounts

# ---------------------------------------------------------------------------
# BC account validation constants (Z8.4)
# ---------------------------------------------------------------------------
# sha256("account:BondingCurve")[:8] — CONFIRMED UNCHANGED post-upgrade
_BC_DISCRIMINATOR = bytes.fromhex("17b7f83760d8ac60")
# Valid account sizes: 49 bytes (V1) and 151 bytes (V2, +creator+u64+padding)
_BC_VALID_LENGTHS  = frozenset({49, 151})
_BC_COMPLETE_OFFSET = 48   # byte 48 = bool complete — CONFIRMED UNCHANGED

# ---------------------------------------------------------------------------
# Compatibility state
# ---------------------------------------------------------------------------
VERIFIED = "VERIFIED"
UNKNOWN  = "UNKNOWN"
CHANGED  = "CHANGED"

_state_lock   = threading.Lock()
# Default: CHANGED — confirmed interface differences found on 2026-07-15.
# Must be explicitly set to VERIFIED after full fixture + simulation pass.
_compat_state = CHANGED

# Tripwire background check interval
TRIPWIRE_INTERVAL_S = 1800  # 30 minutes

_SOLANA_RPC_FALLBACK = "https://api.mainnet-beta.solana.com"


# ---------------------------------------------------------------------------
# Public state accessors
# ---------------------------------------------------------------------------

def get_state() -> str:
    """Return the current compatibility state: VERIFIED | UNKNOWN | CHANGED."""
    with _state_lock:
        return _compat_state


def local_build_allowed() -> bool:
    """
    True only when state == VERIFIED.

    CHANGED or UNKNOWN → False → callers must use PumpPortal / safe fallback.
    Covers both buy and sell local builds on the pump.fun bonding-curve program.
    Does NOT affect PumpSwap or Jupiter paths.
    """
    return get_state() == VERIFIED


def layout_graduation_allowed() -> bool:
    """
    True only when state == VERIFIED.

    Controls whether complete=True from the BC account binary parser may trigger
    graduation.  When False, graduation must use independently verified evidence:
      - account_missing (Z1, not layout-dependent)
      - DexScreener pumpswap dex_id detection
      - Confirmed PumpSwap pool discovery

    Does NOT affect account_missing detection or any non-layout graduation path.
    """
    return get_state() == VERIFIED


def mark_interface_verified(deploy_slot: int, exe_hash: str) -> None:
    """
    Record that the current deployment has been verified.

    Called after:
      1. post-upgrade fixture comparison passes (discriminator + account structure)
      2. BC layout verification passes
      3. deterministic fixture tests pass
      4. local TX simulation succeeds with no instruction/account error

    Updates state to VERIFIED.  If deploy_slot != BASELINE_DEPLOY_SLOT, raises
    ValueError — the tripwire must be updated to the new slot first.
    """
    if deploy_slot != BASELINE_DEPLOY_SLOT:
        raise ValueError(
            f"mark_interface_verified: slot {deploy_slot} != baseline {BASELINE_DEPLOY_SLOT}. "
            "Update BASELINE_DEPLOY_SLOT if the program was re-deployed."
        )
    _set_state(VERIFIED, f"explicit_verification slot={deploy_slot} hash={exe_hash[:16]}")
    log.warning(
        "Z8 COMPAT: marked VERIFIED  slot=%d  hash=%s",
        deploy_slot, exe_hash[:16],
    )


# ---------------------------------------------------------------------------
# Internal state mutator
# ---------------------------------------------------------------------------

def _set_state(new_state: str, reason: str) -> None:
    global _compat_state
    with _state_lock:
        old = _compat_state
        _compat_state = new_state
    if old != new_state:
        log.warning("Z8 COMPAT: %s → %s  reason=%s", old, new_state, reason)
        if new_state in (CHANGED, UNKNOWN):
            _alert_compat_change(new_state, reason)


def _alert_compat_change(state: str, reason: str) -> None:
    """Fire a high-priority Telegram alert when compat degrades."""
    try:
        from app.alerts import send_alert as _sa
        _sa(
            f"Z8 COMPAT STATE → {state}\n"
            f"Reason: {reason}\n"
            f"LOCAL PUMP.FUN BUILDS DISABLED. PumpPortal/Jupiter active.\n"
            f"Program: {_PUMPFUN_PROGRAM}"
        )
    except Exception as _e:
        log.warning("Z8 COMPAT alert failed: %s", _e)


# ---------------------------------------------------------------------------
# BC account validation (Z8.4)
# ---------------------------------------------------------------------------

def validate_bc_account(raw: bytes, owner: str) -> tuple[bool, str]:
    """
    Validate a pump.fun bonding-curve account before trusting byte 48.

    Args:
        raw:   raw account data bytes (decoded from base64)
        owner: account owner pubkey as base58 string

    Returns (valid: bool, reason: str).
    reason is "ok" on success, or a short failure code on rejection.

    Validation sequence:
      1. owner == _PUMPFUN_PROGRAM  (rejects wrong-program accounts)
      2. len(raw) >= 49             (rejects truncated/empty accounts)
      3. len(raw) in _BC_VALID_LENGTHS  (rejects unrecognised layouts)
      4. raw[0:8] == _BC_DISCRIMINATOR  (rejects non-BondingCurve accounts)
      5. raw[48] in {0, 1}          (rejects corrupt complete byte)
    """
    if owner != _PUMPFUN_PROGRAM:
        return False, f"wrong_owner:{owner[:16] if owner else 'empty'}"
    if len(raw) < 49:
        return False, f"too_short:{len(raw)}"
    if len(raw) not in _BC_VALID_LENGTHS:
        return False, f"unexpected_length:{len(raw)}"
    if raw[:8] != _BC_DISCRIMINATOR:
        return False, f"wrong_discriminator:{raw[:8].hex()}"
    complete_byte = raw[_BC_COMPLETE_OFFSET]
    if complete_byte not in (0, 1):
        return False, f"invalid_complete_byte:{complete_byte}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Tripwire — deploy slot monitoring (Z8.6)
# ---------------------------------------------------------------------------

def _fetch_current_deploy_slot(rpc_url: str = _SOLANA_RPC_FALLBACK) -> int | None:
    """
    Query the ProgramData account to read the current deploy slot.
    Returns the slot integer, or None on any RPC/parse failure.
    """
    try:
        resp = requests.post(
            rpc_url,
            json={
                "jsonrpc": "2.0",
                "id":      1,
                "method":  "getAccountInfo",
                "params":  [_PROGRAMDATA_ADDR, {"encoding": "base64"}],
            },
            timeout=10,
        )
        resp.raise_for_status()
        result = resp.json().get("result", {})
        value  = result.get("value") if isinstance(result, dict) else None
        if value is None:
            log.warning("Z8 TRIPWIRE: ProgramData account not found")
            return None
        data_b64 = value["data"][0]
        raw = base64.b64decode(data_b64)
        # ProgramData layout: [0:4] discriminant u32 LE = 3, [4:12] slot u64 LE
        if len(raw) < 12:
            log.warning("Z8 TRIPWIRE: ProgramData too short (%d bytes)", len(raw))
            return None
        slot = struct.unpack_from("<Q", raw, 4)[0]
        return slot
    except Exception as e:
        log.warning("Z8 TRIPWIRE: deploy slot fetch failed: %s", e)
        return None


def check_tripwire_once(rpc_url: str = _SOLANA_RPC_FALLBACK) -> None:
    """
    One-shot tripwire check: compare live deploy slot against BASELINE_DEPLOY_SLOT.

    Called at process startup and periodically by the background thread.

    Rules:
    - live_slot != BASELINE_DEPLOY_SLOT → set state CHANGED, fire alert.
    - live_slot == BASELINE_DEPLOY_SLOT → no change to existing state.
      (Preserves CHANGED/UNKNOWN/VERIFIED — does not auto-promote to VERIFIED.)
    - RPC failure → log warning, no state change (fail open on monitoring, not on build).
    """
    current_slot = _fetch_current_deploy_slot(rpc_url)
    if current_slot is None:
        log.warning("Z8 TRIPWIRE: could not read deploy slot — compat state unchanged")
        return

    if current_slot != BASELINE_DEPLOY_SLOT:
        _set_state(
            CHANGED,
            f"deploy_slot_changed:{BASELINE_DEPLOY_SLOT}→{current_slot}",
        )
        log.warning(
            "Z8 TRIPWIRE FIRED: pump.fun UPGRADED  "
            "old_slot=%d  new_slot=%d  local_build=DISABLED",
            BASELINE_DEPLOY_SLOT, current_slot,
        )
    else:
        log.info(
            "Z8 TRIPWIRE: slot unchanged (%d)  compat_state=%s",
            current_slot, get_state(),
        )


def start_tripwire_thread(rpc_url: str = _SOLANA_RPC_FALLBACK) -> None:
    """
    Start the background tripwire thread. Call once at process startup.

    Thread runs a check immediately, then repeats every TRIPWIRE_INTERVAL_S seconds.
    The thread is a daemon so it does not prevent process exit.
    """
    def _loop():
        try:
            check_tripwire_once(rpc_url)
        except Exception as e:
            log.warning("Z8 TRIPWIRE startup check error: %s", e)
        while True:
            time.sleep(TRIPWIRE_INTERVAL_S)
            try:
                check_tripwire_once(rpc_url)
            except Exception as e:
                log.warning("Z8 TRIPWIRE periodic check error: %s", e)

    t = threading.Thread(target=_loop, daemon=True, name="z8-compat-tripwire")
    t.start()
    log.info(
        "Z8 TRIPWIRE: thread started  baseline_slot=%d  interval=%ds",
        BASELINE_DEPLOY_SLOT, TRIPWIRE_INTERVAL_S,
    )
