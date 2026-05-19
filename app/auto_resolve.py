"""Auto-resolve journaled markets by polling the Polymarket Gamma API.

Reads every unresolved entry in logs/market_journal.csv, checks whether the
market has settled on Polymarket, and fills in actual_outcome + bot_correct
automatically.

Run manually:
    python -m app.auto_resolve

Or let the GitHub Actions workflow call it daily — it commits the updated
journal back to the repo so the paper-trading loop runs without any manual
work.
"""

import time
from typing import Optional

import requests

from app.market_journal import load_journal_records, update_journal_outcome
import app.pm_positions as pm_positions

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
OUTCOME_THRESHOLD = 0.99  # price at which we declare a winner
REQUEST_DELAY_S = 0.3     # be polite to the API


def _parse_list(raw) -> list:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        import json
        try:
            return json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            return []
    return []


def check_market_resolution(market_id: str) -> Optional[str]:
    """Return 'yes', 'no', or None if the market hasn't resolved yet."""
    try:
        resp = requests.get(
            f"{GAMMA_BASE_URL}/markets/{market_id}",
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None

    # outcomePrices tells us the settlement price of each outcome
    outcome_prices = _parse_list(data.get("outcomePrices"))
    outcomes       = _parse_list(data.get("outcomes"))

    if len(outcome_prices) < 2 or len(outcomes) < 2:
        return None

    try:
        p0 = float(outcome_prices[0])
        p1 = float(outcome_prices[1])
    except (TypeError, ValueError):
        return None

    if p0 >= OUTCOME_THRESHOLD and p1 < 0.5:
        winner = outcomes[0].lower().strip()
    elif p1 >= OUTCOME_THRESHOLD and p0 < 0.5:
        winner = outcomes[1].lower().strip()
    else:
        return None  # not resolved yet

    # Normalise to yes/no — the first outcome on Polymarket is always YES
    if winner in ("yes",):
        return "yes"
    if winner in ("no",):
        return "no"

    # Some markets use custom labels (e.g. "Trump", "Biden").
    # Treat first outcome as YES, second as NO.
    first  = outcomes[0].lower().strip()
    second = outcomes[1].lower().strip()
    if winner == first:
        return "yes"
    if winner == second:
        return "no"

    return None


def run() -> dict:
    """Check all unresolved journal entries and fill in outcomes.

    Returns a summary dict with counts.
    """
    records = load_journal_records()

    # Collect unique market IDs that still need an outcome
    pending: dict[str, list[int]] = {}  # market_id → list of row indices
    for i, row in enumerate(records):
        if not row.get("actual_outcome", "").strip():
            mid = row.get("market_id", "").strip()
            if mid:
                pending.setdefault(mid, []).append(i)

    total   = len(pending)
    resolved = 0
    skipped  = 0

    print(f"Checking {total} unresolved markets...")

    for market_id in pending:
        outcome = check_market_resolution(market_id)

        if outcome:
            success = update_journal_outcome(market_id, outcome, notes="auto-resolved")
            if success:
                resolved += 1
                print(f"  ✓ {market_id[:20]}...  → {outcome}")

                # Resolve Polymarket position if one was opened for this market
                pos = pm_positions.get_position(market_id)
                if pos and pos.status == "open":
                    won = (
                        (pos.side == "YES" and outcome == "yes") or
                        (pos.side == "NO"  and outcome == "no")
                    )
                    exit_price = 1.0 if won else 0.0
                    closed = pm_positions.resolve_position(
                        market_id,
                        outcome="win" if won else "loss",
                        exit_price=exit_price,
                    )
                    if closed:
                        mode = "DRY-RUN" if pos.dry_run else "LIVE"
                        print(
                            f"    [PM {mode}] {pos.side} → {'WIN' if won else 'LOSS'}"
                            f"  PnL={closed.pnl_usdc:+.2f} USDC"
                        )
            else:
                skipped += 1
        else:
            skipped += 1

        time.sleep(REQUEST_DELAY_S)

    print(f"\nDone. Resolved: {resolved}  |  Still pending: {skipped}")
    return {"total": total, "resolved": resolved, "pending": skipped}


if __name__ == "__main__":
    run()
