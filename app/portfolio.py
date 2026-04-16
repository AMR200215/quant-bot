"""Portfolio equity tracking and drawdown circuit breaker.

State is persisted to data/portfolio.json so it survives process restarts.
The circuit breaker halts all new signal emission when the drawdown from peak
equity exceeds the configured threshold.

Typical usage in a scan pipeline:
    from app.portfolio import is_halted, update_equity
    from app.state import settings

    if is_halted(settings.max_drawdown):
        print("CIRCUIT BREAKER: drawdown limit hit — no new positions.")
        return

    # ... after a trade resolves:
    update_equity(pnl=+45.00, starting_bankroll=settings.bankroll)
"""

import json
from pathlib import Path

STATE_FILE = Path("data/portfolio.json")


def load_state() -> dict:
    """Load portfolio state from disk, or return defaults."""
    if STATE_FILE.exists():
        try:
            with STATE_FILE.open() as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "current_equity": None,
        "peak_equity": None,
        "drawdown": 0.0,
        "halted": False,
        "total_trades": 0,
    }


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("w") as f:
        json.dump(state, f, indent=2)


def update_equity(pnl: float, starting_bankroll: float) -> dict:
    """Record a realised PnL and update equity / drawdown state.

    Args:
        pnl: Profit (positive) or loss (negative) from the closed position.
        starting_bankroll: Used as the initial equity baseline on first call.

    Returns:
        Updated state dict.
    """
    state = load_state()

    current = state.get("current_equity") or starting_bankroll
    peak    = state.get("peak_equity")    or starting_bankroll

    current += pnl
    peak     = max(peak, current)
    dd       = (peak - current) / peak if peak > 0 else 0.0

    state["current_equity"] = round(current, 2)
    state["peak_equity"]    = round(peak, 2)
    state["drawdown"]       = round(dd, 4)
    state["total_trades"]   = state.get("total_trades", 0) + 1

    save_state(state)
    return state


def is_halted(max_drawdown: float = 0.15) -> bool:
    """Return True if the portfolio should stop opening new positions.

    Halts when:
    1. The state file contains halted=True (manual override), or
    2. Drawdown from peak equity exceeds max_drawdown.
    """
    state = load_state()

    if state.get("halted"):
        return True

    current = state.get("current_equity")
    peak    = state.get("peak_equity")
    if current is None or peak is None or peak <= 0:
        return False

    dd = (peak - current) / peak
    return dd >= max_drawdown


def get_status() -> dict:
    """Return a human-readable status dict for display."""
    state = load_state()
    return {
        "current_equity": state.get("current_equity"),
        "peak_equity":    state.get("peak_equity"),
        "drawdown":       state.get("drawdown", 0.0),
        "halted":         state.get("halted", False),
        "total_trades":   state.get("total_trades", 0),
    }


def reset_halt() -> None:
    """Manually clear the halt flag (e.g. after topping up the bankroll)."""
    state = load_state()
    state["halted"] = False
    save_state(state)
    print("Circuit breaker reset.")


if __name__ == "__main__":
    status = get_status()
    print("Portfolio status:")
    for k, v in status.items():
        print(f"  {k}: {v}")
