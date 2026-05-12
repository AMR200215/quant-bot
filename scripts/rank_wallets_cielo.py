"""
Whale wallet ranker — uses Cielo Finance API for pre-computed wallet PnL.

Much faster than on-chain RPC parsing: ~5 minutes for 300 wallets vs 4 hours.
Cielo provides realized PnL, win rate, trade count — all pre-computed.

Setup:
  1. Sign up free at cielo.finance
  2. Get API key from dashboard
  3. Add to .env:  CIELO_API_KEY=your_key_here

Run:
  python3 scripts/rank_wallets_cielo.py
  python3 scripts/rank_wallets_cielo.py --resume
  python3 scripts/rank_wallets_cielo.py --days 30   (default 30-day window)
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
ROOT     = Path(__file__).parent.parent
WALLETS  = ROOT / "memecoin" / "data" / "whale_wallets_sol.json"
OUT_FILE = ROOT / "memecoin" / "data" / "whale_stats.json"

CIELO_BASE  = "https://feed-api.cielo.finance/v1"
CIELO_KEY   = os.getenv("CIELO_API_KEY", "")
CALL_DELAY  = 0.25   # 4 req/sec — safe under free tier (1000/month)
MIN_TRADES  = 5      # minimum trades to include wallet in ranking


# ---------------------------------------------------------------------------
def _headers() -> dict:
    return {"X-API-KEY": CIELO_KEY, "accept": "application/json"}


def fetch_wallet_pnl(wallet: str, days: int = 30) -> dict | None:
    """
    Fetch PnL summary for a wallet from Cielo Finance.
    Returns parsed stats or None on failure.

    Cielo endpoint: GET /v1/{wallet}/pnl/total-stats
    Query params: chains=solana, period=30d|7d|all
    """
    period = f"{days}d" if days in (7, 30, 60, 90) else "30d"
    url = f"{CIELO_BASE}/{wallet}/pnl/total-stats"
    params = {"chains": "solana", "period": period}

    try:
        r = requests.get(url, headers=_headers(), params=params, timeout=10)
        if r.status_code == 401:
            print("\nERROR: Invalid or missing CIELO_API_KEY — check your .env file")
            raise SystemExit(1)
        if r.status_code == 404:
            return None  # wallet not indexed
        if r.status_code == 429:
            print("  rate-limited, sleeping 10s…")
            time.sleep(10)
            return fetch_wallet_pnl(wallet, days)
        if r.status_code != 200:
            return None

        data = r.json()
        # Cielo response shape:
        # { "data": { "winrate": 0.45, "realized_pnl_usd": 1234.5,
        #             "total_trades": 87, "tokens_traded": 42, ... } }
        stats = data.get("data") or data
        if not stats:
            return None
        return stats

    except requests.exceptions.RequestException:
        return None


def fetch_wallet_trades(wallet: str, days: int = 30, limit: int = 200) -> list[dict]:
    """
    Fetch individual token trades for a wallet to get avg_win/avg_loss breakdown.
    Endpoint: GET /v1/{wallet}/pnl/tokens
    """
    period = f"{days}d" if days in (7, 30, 60, 90) else "30d"
    url = f"{CIELO_BASE}/{wallet}/pnl/tokens"
    params = {"chains": "solana", "period": period, "limit": limit}

    try:
        r = requests.get(url, headers=_headers(), params=params, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        return data.get("data", {}).get("items") or data.get("items") or []
    except Exception:
        return []


def build_stats(wallet: str, name: str, days: int) -> dict | None:
    """Fetch Cielo data and build the same stats format as rank_wallets.py"""
    summary = fetch_wallet_pnl(wallet, days)
    if not summary:
        return None

    total_trades = summary.get("total_trades") or summary.get("trades_count") or 0
    if total_trades < MIN_TRADES:
        return None

    win_rate    = float(summary.get("winrate") or summary.get("win_rate") or 0)
    realized    = float(summary.get("realized_pnl_usd") or 0)
    unrealized  = float(summary.get("unrealized_pnl_usd") or 0)

    # Try to get avg_win / avg_loss from token-level data
    trades = fetch_wallet_trades(wallet, days)
    time.sleep(CALL_DELAY)

    wins   = [t for t in trades if float(t.get("pnl_usd") or 0) > 0]
    losses = [t for t in trades if float(t.get("pnl_usd") or 0) < 0]

    avg_win  = (sum(float(t.get("pnl_usd", 0)) for t in wins)   / len(wins))   if wins   else 0
    avg_loss = (sum(float(t.get("pnl_usd", 0)) for t in losses) / len(losses)) if losses else 0

    # Rough avg_roi_pct from realized PnL relative to total volume
    total_vol = float(summary.get("total_invested_usd") or summary.get("volume_usd") or 0)
    avg_roi   = (realized / total_vol * 100) if total_vol > 0 else 0

    score = win_rate * max(avg_roi, 0)

    return {
        "wallet":           wallet,
        "name":             name,
        "trades":           total_trades,
        "win_rate":         round(win_rate, 4),
        "avg_roi":          round(avg_roi, 2),
        "avg_win_usd":      round(avg_win, 2),
        "avg_loss_usd":     round(avg_loss, 2),
        "realized_pnl_usd": round(realized, 2),
        "unrealized_usd":   round(unrealized, 2),
        "score":            round(score, 4),
        "days_window":      days,
        "source":           "cielo",
    }


# ---------------------------------------------------------------------------
def main():
    if not CIELO_KEY:
        print("ERROR: CIELO_API_KEY not set in .env\n"
              "  1. Sign up free at cielo.finance\n"
              "  2. Add CIELO_API_KEY=your_key to .env")
        raise SystemExit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--days",   type=int, default=30,
                        help="PnL window in days: 7, 30, 60, 90 (default 30)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip wallets already in output file")
    args = parser.parse_args()

    wallets = json.loads(WALLETS.read_text())
    print(f"Loaded {len(wallets)} wallets")
    print(f"PnL window: {args.days} days  |  API key: {CIELO_KEY[:8]}…\n")

    existing: dict[str, dict] = {}
    if args.resume and OUT_FILE.exists():
        for entry in json.loads(OUT_FILE.read_text()):
            existing[entry["wallet"]] = entry
        print(f"Resuming — {len(existing)} wallets already done\n")

    results: list[dict] = list(existing.values())
    total = len(wallets)

    for i, w in enumerate(wallets):
        addr = w.get("trackedWalletAddress") or w.get("address") or ""
        name = w.get("name", addr[:8])
        if not addr:
            continue
        if args.resume and addr in existing:
            continue

        print(f"[{i+1:>3}/{total}] {name:<20} {addr[:12]}…", end="  ", flush=True)

        stats = build_stats(addr, name, args.days)
        time.sleep(CALL_DELAY)

        if stats:
            results.append(stats)
            wr  = stats["win_rate"] * 100
            roi = stats["avg_roi"]
            pnl = stats["realized_pnl_usd"]
            print(f"{stats['trades']} trades  {wr:.0f}% win  "
                  f"avg_roi {roi:+.1f}%  realized ${pnl:+,.0f}")
        else:
            print("not indexed / <5 trades")

        # Save after every wallet
        ranked = _rank(results)
        OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUT_FILE.write_text(json.dumps(ranked, indent=2))

    ranked = _rank(results)
    OUT_FILE.write_text(json.dumps(ranked, indent=2))

    print(f"\n{'='*70}")
    print(f"Done. {len(ranked)} wallets ranked  |  saved to {OUT_FILE.name}\n")
    _print_table(ranked[:20])


def _rank(results: list[dict]) -> list[dict]:
    return sorted(
        [r for r in results if r.get("trades", 0) >= MIN_TRADES],
        key=lambda x: x.get("score", 0),
        reverse=True,
    )


def _print_table(ranked: list[dict]) -> None:
    print(f"{'Rank':<5} {'Name':<20} {'Trades':>6} {'Win%':>6} "
          f"{'AvgROI':>8} {'Realized$':>11} {'Score':>7}")
    print("-" * 70)
    for rank, r in enumerate(ranked, 1):
        print(f"{rank:<5} {r.get('name','?'):<20} {r['trades']:>6} "
              f"{r['win_rate']*100:>5.1f}%  {r['avg_roi']:>+7.1f}%  "
              f"${r['realized_pnl_usd']:>+10,.0f}  {r['score']:>7.4f}")


if __name__ == "__main__":
    main()
