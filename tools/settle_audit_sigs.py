"""
settle_audit_sigs.py — One-off settlement script.

Reads SOL delta for three known audit signatures, then runs a full
journal reconciler pass and reports the result.

Usage (from /root/quant-bot):
    python3 tools/settle_audit_sigs.py
"""

import sys
import os

# Ensure project root is on sys.path so memecoin imports work.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

WALLET = "8PNHvFWeMT7CqpUvJjAwVgAK545t5KV3uCPd8DUfaTiM"

AUDIT_SIGS = [
    ("WLA sig 1",
     "4aPtMHxZGRDig9okDN4YU7mEsi7cVD2pPc8YhoNUxntGgWhxn8LD1P1Pw53g2Sm4npPPsZqjR2qXWaUMVXGQgGGa"),
    ("WLA sig 2",
     "3Nor9w2v6Xa9hpngEMghJgGsnA7vGV6gvzDL2a31kiE1AiQ2aG2Y9dQ2ZUat1UbGsvtbv2cb7qev592UMgeVwrTx"),
    ("Beginner sig",
     "BT8RCo6a8sYpiKyfWnmPagqT7C5cDmm3dZ3bBLNhUu2KpFJTar4zb1DKnbGZQn51j5ehkobZ6pMfcgvpRZnucp4"),
]


def main():
    from memecoin.tx_meta import read_sol_delta
    from memecoin.journal_reconciler import run_reconciler_pass

    print("=" * 70)
    print("settle_audit_sigs.py — reading SOL delta for audit signatures")
    print(f"wallet: {WALLET}")
    print("=" * 70)

    for label, sig in AUDIT_SIGS:
        print(f"\n[{label}]")
        print(f"  sig: {sig}")
        result = read_sol_delta(sig, WALLET)
        print(f"  ok:        {result.get('ok')}")
        print(f"  sol_delta: {result.get('sol_delta')}")
        print(f"  source:    {result.get('source')}")
        print(f"  attempts:  {result.get('attempts')}")
        print(f"  reason:    {result.get('reason')}")

    print("\n" + "=" * 70)
    print("Running journal reconciler pass ...")
    print("=" * 70)
    pass_result = run_reconciler_pass(WALLET)
    print(f"\nReconciler result:")
    print(f"  rows_checked:   {pass_result['rows_checked']}")
    print(f"  rows_corrected: {pass_result['rows_corrected']}")
    print(f"  usd_adjustment: {pass_result['usd_adjustment']}")

    print("\nDONE — check journal for updated rows")


if __name__ == "__main__":
    main()
