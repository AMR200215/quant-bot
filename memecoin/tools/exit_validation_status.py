"""
exit_validation_status.py — prints validation status for the local exit system.

Reads logs/exit_route_attempts.csv and reports whether EXIT_SYSTEM_VALIDATED
can be safely set True.

Usage:
    python -m memecoin.tools.exit_validation_status
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict

_LOG_PATH = Path(__file__).parent.parent.parent / "logs" / "exit_route_attempts.csv"


def main():
    if not _LOG_PATH.exists():
        print(f"NOT VALIDATED — {_LOG_PATH} does not exist (no sell attempts logged yet)")
        return

    rows = []
    try:
        with open(_LOG_PATH) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f"NOT VALIDATED — could not read {_LOG_PATH}: {e}")
        return

    print(f"\n{'='*60}")
    print("EXIT SYSTEM VALIDATION STATUS")
    print(f"{'='*60}")
    print(f"Total route attempts logged: {len(rows)}")

    # Counts
    ps_local      = [r for r in rows if r.get("route_name") == "PUMPSWAP_LOCAL"]
    ps_sim_ok     = [r for r in ps_local if r.get("simulation_ok", "").lower() == "true"]
    ps_t22_sim_ok = [r for r in ps_sim_ok if r.get("is_token2022", "").lower() == "true"]
    ps_confirmed  = [r for r in ps_local if r.get("confirmed", "").lower() == "true"]

    bc_t22        = [r for r in rows if r.get("route_name") == "BONDING_CURVE_T22_LOCAL"]
    bc_t22_sim_ok = [r for r in bc_t22 if r.get("simulation_ok", "").lower() == "true"]
    bc_t22_conf   = [r for r in bc_t22 if r.get("confirmed", "").lower() == "true"]

    unknown_fail  = [r for r in rows if r.get("error_class") == "unknown_sell_failure"]

    print(f"\nPumpSwap local (GRADUATED):")
    print(f"  Total attempts:      {len(ps_local)}")
    print(f"  Sim OK:              {len(ps_sim_ok)}")
    print(f"  T22 sim OK:          {len(ps_t22_sim_ok)}")
    print(f"  Confirmed live:      {len(ps_confirmed)}")

    print(f"\nBC T22 local (BONDING_CURVE_T22):")
    print(f"  Total attempts:      {len(bc_t22)}")
    print(f"  Sim OK:              {len(bc_t22_sim_ok)}")
    print(f"  Confirmed live:      {len(bc_t22_conf)}")

    print(f"\nUnknown sell failures: {len(unknown_fail)}")

    # Validation criteria
    print(f"\n{'='*60}")
    print("VALIDATION CRITERIA:")
    crit = [
        (">=1 PUMPSWAP_LOCAL sim_ok",          len(ps_sim_ok) >= 1),
        (">=1 T22 PUMPSWAP_LOCAL sim_ok",       len(ps_t22_sim_ok) >= 1),
        (">=1 PUMPSWAP_LOCAL confirmed live",   len(ps_confirmed) >= 1),
        (">=1 BC_T22_LOCAL sim_ok",             len(bc_t22_sim_ok) >= 1),
        (">=1 BC_T22_LOCAL confirmed live",     len(bc_t22_conf) >= 1),
        ("0 unknown_sell_failures",            len(unknown_fail) == 0),
    ]
    all_pass = True
    for label, passed in crit:
        status = "OK" if passed else "FAIL"
        print(f"  [{status}] {label}")
        if not passed:
            all_pass = False

    print(f"\n{'='*60}")
    if all_pass:
        print("STATUS: VALIDATED — safe to set EXIT_SYSTEM_VALIDATED=True")
    else:
        print("STATUS: NOT VALIDATED — do not set EXIT_SYSTEM_VALIDATED=True yet")
    print(f"{'='*60}\n")

    # Config cross-check
    try:
        from memecoin.config import (
            EXIT_SYSTEM_VALIDATED, PUMPSWAP_LOCAL_SELL_ENABLED,
            PUMPSWAP_LOCAL_SIM_ONLY, LIVE_CANARY_MODE, MAX_CANARY_TRADE_USD,
        )
        print("Current config:")
        print(f"  EXIT_SYSTEM_VALIDATED={EXIT_SYSTEM_VALIDATED}")
        print(f"  PUMPSWAP_LOCAL_SELL_ENABLED={PUMPSWAP_LOCAL_SELL_ENABLED}")
        print(f"  PUMPSWAP_LOCAL_SIM_ONLY={PUMPSWAP_LOCAL_SIM_ONLY}")
        print(f"  LIVE_CANARY_MODE={LIVE_CANARY_MODE}")
        print(f"  MAX_CANARY_TRADE_USD={MAX_CANARY_TRADE_USD}")
        if PUMPSWAP_LOCAL_SELL_ENABLED and not all_pass:
            print("\nWARNING: PUMPSWAP_LOCAL_SELL_ENABLED=True but validation criteria not met!")
    except Exception as e:
        print(f"(could not import config: {e})")


if __name__ == "__main__":
    main()
