"""
memecoin/tools/promote_receipt.py — CLI to look up and optionally promote an execution receipt.

Usage:
    python memecoin/tools/promote_receipt.py <sig>
    python memecoin/tools/promote_receipt.py <sig> --confirm

Exit 0 if found, exit 1 if not found.
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from memecoin.execution_receipts import read_receipts, write_receipt


def main():
    parser = argparse.ArgumentParser(description="Look up an execution receipt by TX sig")
    parser.add_argument("sig", help="On-chain TX signature")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Update receipt status to 'confirmed' (appends a new row)",
    )
    args = parser.parse_args()

    receipts = read_receipts()
    match = next((r for r in receipts if r.get("sig") == args.sig), None)

    if match is None:
        print(f"Receipt not found for sig: {args.sig}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(match, indent=2, default=str))

    if args.confirm:
        promoted = dict(match)
        promoted["status"] = "confirmed"
        write_receipt(promoted)
        print(f"\nPromoted: status → confirmed", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
