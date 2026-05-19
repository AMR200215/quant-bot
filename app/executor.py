"""
Polymarket execution layer.

Usage:
    from app.executor import PolyExecutor
    from app.state import settings

    ex = PolyExecutor(settings)
    result = ex.place_order(market, edge)   # dry-run by default

Set LIVE_TRADING=true in .env to send real orders via py_clob_client.

Order flow
──────────
1. Pre-flight checks (daily loss limit, max open positions, min edge)
2. Determine side (YES or NO) and token ID
3. Dry-run → log only; Live → call CLOB API
4. Persist position to pm_positions.json via pm_positions module
5. Return result dict with all metadata

Sell / exit
───────────
For prediction markets, contracts resolve on their own (settle at 1 or 0).
We never need to send a sell order — we just call auto_resolve.py to record
the outcome once the market is settled.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.data_client import Market
from app import pm_positions as positions
from app.pm_positions import PMPosition
from app.state import Settings

log = logging.getLogger(__name__)


@dataclass
class EdgeResult:
    """Minimal interface from scan_near_term — only the fields executor cares about."""
    preferred_side: str    # "buy_yes" or "buy_no"
    adjusted_edge: float   # e.g. 0.08 = 8% edge


@dataclass
class OrderResult:
    success: bool
    dry_run: bool
    market_id: str
    side: str
    entry_price: float
    size_usdc: float
    order_id: str
    reason: str = ""       # rejection reason when success=False


class PolyExecutor:
    """
    Places limit orders on Polymarket CLOB.

    Dry-run mode (default): logs intent, persists position, never touches API.
    Live mode: calls py_clob_client, same persistence path.
    """

    def __init__(self, settings: Settings):
        self._s = settings
        self._client: Optional[Any] = None  # lazy-init

    # ── public ────────────────────────────────────────────────────────────────

    def place_order(self, market: Market, edge: EdgeResult) -> OrderResult:
        """
        Evaluate pre-flight checks then place (or simulate) a limit order.
        Returns an OrderResult describing what happened.
        """
        s = self._s

        # Pre-flight 1: minimum edge
        if edge.adjusted_edge < s.pm_min_edge:
            return self._reject(
                market, edge,
                f"edge {edge.adjusted_edge:.1%} < min {s.pm_min_edge:.1%}"
            )

        # Pre-flight 2: max open positions
        open_n = positions.open_count()
        if open_n >= s.pm_max_positions:
            return self._reject(
                market, edge,
                f"max positions reached ({open_n}/{s.pm_max_positions})"
            )

        # Pre-flight 3: daily loss limit
        dpnl = positions.daily_pnl()
        if dpnl <= -abs(s.pm_daily_loss_limit):
            return self._reject(
                market, edge,
                f"daily loss limit hit (P&L={dpnl:.2f} USDC)"
            )

        # Pre-flight 4: already have a position in this market
        if positions.get_position(market.market_id) is not None:
            return self._reject(
                market, edge,
                "position already open for this market"
            )

        # Determine side + token
        if edge.preferred_side == "buy_yes":
            side = "YES"
            token_id = market.clob_token_id or ""
            entry_price = market.yes_price
        else:
            side = "NO"
            token_id = market.clob_token_id_no or ""
            entry_price = market.no_price

        if not token_id:
            return self._reject(market, edge, "no CLOB token ID available")

        size_usdc = s.pm_position_size
        contracts = round(size_usdc / entry_price, 4) if entry_price > 0 else 0

        if s.live_trading:
            order_id = self._live_order(token_id, side, entry_price, contracts, market)
        else:
            order_id = ""  # dry-run

        pos = PMPosition(
            market_id=market.market_id,
            question=market.question,
            side=side,
            token_id=token_id,
            order_id=order_id,
            entry_price=entry_price,
            size_usdc=size_usdc,
            contracts=contracts,
            dry_run=not s.live_trading,
            adjusted_edge=edge.adjusted_edge,
            end_date=market.end_date,
        )
        positions.open_position(pos)

        mode = "LIVE" if s.live_trading else "DRY-RUN"
        log.info(
            "[%s] %s %s @ %.3f | edge=%.1f%% | size=$%.0f | order_id=%s",
            mode, side, market.question[:60], entry_price,
            edge.adjusted_edge * 100, size_usdc, order_id or "n/a",
        )

        return OrderResult(
            success=True,
            dry_run=not s.live_trading,
            market_id=market.market_id,
            side=side,
            entry_price=entry_price,
            size_usdc=size_usdc,
            order_id=order_id,
        )

    # ── private ───────────────────────────────────────────────────────────────

    def _reject(self, market: Market, edge: EdgeResult, reason: str) -> OrderResult:
        log.debug("Order rejected — %s — market=%s", reason, market.market_id)
        return OrderResult(
            success=False,
            dry_run=not self._s.live_trading,
            market_id=market.market_id,
            side=edge.preferred_side,
            entry_price=0.0,
            size_usdc=0.0,
            order_id="",
            reason=reason,
        )

    def _live_order(
        self,
        token_id: str,
        side: str,
        price: float,
        contracts: float,
        market: Market,
    ) -> str:
        """Call py_clob_client and return order_id. Raises on failure."""
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import OrderArgs, OrderType
        except ImportError as e:
            raise RuntimeError(
                "py-clob-client not installed. Run: pip install py-clob-client==0.34.6"
            ) from e

        s = self._s
        if self._client is None:
            self._client = ClobClient(
                host="https://clob.polymarket.com",
                key=s.poly_private_key,
                chain_id=137,  # Polygon mainnet
                creds={
                    "apiKey":      s.poly_api_key,
                    "secret":      s.poly_api_secret,
                    "passphrase":  s.poly_api_passphrase,
                },
            )

        order_args = OrderArgs(
            token_id=token_id,
            price=round(price, 4),
            size=round(contracts, 4),
            side=side,
        )
        resp = self._client.create_and_post_order(order_args)
        order_id = resp.get("orderID", "") if isinstance(resp, dict) else str(resp)
        log.info("CLOB order placed: %s", order_id)
        return order_id
