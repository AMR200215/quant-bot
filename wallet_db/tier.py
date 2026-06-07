"""
Phase 6 — Weekly tiering job.

Assigns S/A/B tiers to wallets using stability-adjusted Phase 4 scores
(weighted average of last 4 weekly snapshots). Runs every Monday after
the daily scoring job.

Key differences from Phase 2a ingest tiering:
  - Uses a 4-week stability score, not just the current day's score
  - Requires minimum active weeks to earn a tier (prevents flukes)
  - Logs tier transitions (promotion/demotion events)
  - Sends Telegram alert for S-tier promotions

Tier thresholds:
  S  — stability_score >= 20  AND trades >= 10  AND active_weeks >= 2/4
  A  — stability_score >= 10  AND trades >= 5   AND active_weeks >= 1/4
  B  — stability_score >= 2   AND trades >= 3
  None — below threshold

Dormant: wallet had 0 trades in ALL of the last 4 scoring snapshots.

Run:
    python3 -m wallet_db.tier
    python3 -m wallet_db.tier --dry-run     (compute only, no DB writes)
    python3 -m wallet_db.tier --wallet <addr>
"""

import argparse
import logging
import os
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from wallet_db.db import get_conn, init_db, _USE_POSTGRES

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_PH = "%s" if _USE_POSTGRES else "?"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "")

# Tier thresholds
TIER_S_MIN_SCORE  = 20.0
TIER_A_MIN_SCORE  = 10.0
TIER_B_MIN_SCORE  =  2.0
TIER_S_MIN_TRADES = 10
TIER_A_MIN_TRADES =  5
TIER_B_MIN_TRADES =  3
TIER_S_MIN_ACTIVE =  2    # of last 4 weekly snapshots must have score > 0
TIER_A_MIN_ACTIVE =  1

# Stability score weights: most recent first
STABILITY_WEIGHTS = [0.40, 0.30, 0.20, 0.10]

DORMANT_SNAPSHOTS = 4     # mark dormant if all last N snapshots had 0 trades


# ---------------------------------------------------------------------------
# Stability score
# ---------------------------------------------------------------------------

def _compute_stability(history: list[dict]) -> dict:
    """
    Given up to 4 recent score snapshots (newest first), compute:
      stability_score  — weighted average composite score
      active_weeks     — how many snapshots had trade_count > 0
      avg_trades       — average trade_count across snapshots
    """
    if not history:
        return {"stability_score": 0.0, "active_weeks": 0, "avg_trades": 0}

    # Pad to 4 entries with zeros if fewer snapshots exist
    padded = (history + [{"score": 0.0, "trade_count": 0}] * 4)[:4]

    stability_score = sum(
        STABILITY_WEIGHTS[i] * (padded[i].get("score") or 0.0)
        for i in range(4)
    )
    active_weeks = sum(1 for p in padded if (p.get("trade_count") or 0) > 0)
    avg_trades   = sum((p.get("trade_count") or 0) for p in padded) / 4

    return {
        "stability_score": round(stability_score, 4),
        "active_weeks":    active_weeks,
        "avg_trades":      round(avg_trades, 1),
    }


def _assign_tier(stability_score: float, avg_trades: float, active_weeks: int) -> str | None:
    """Pure tier assignment — no DB side effects."""
    if stability_score >= TIER_S_MIN_SCORE and avg_trades >= TIER_S_MIN_TRADES and active_weeks >= TIER_S_MIN_ACTIVE:
        return "S"
    if stability_score >= TIER_A_MIN_SCORE and avg_trades >= TIER_A_MIN_TRADES and active_weeks >= TIER_A_MIN_ACTIVE:
        return "A"
    if stability_score >= TIER_B_MIN_SCORE and avg_trades >= TIER_B_MIN_TRADES:
        return "B"
    return None


# ---------------------------------------------------------------------------
# Telegram helper
# ---------------------------------------------------------------------------

def _tg(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
    except Exception as e:
        log.debug("Telegram send failed: %s", e)


# ---------------------------------------------------------------------------
# Main tiering job
# ---------------------------------------------------------------------------

def run_tiering(single_wallet: str = "", chain: str = "solana",
                dry_run: bool = False) -> None:
    init_db()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    conn = get_conn()
    if single_wallet:
        rows = conn.execute(
            f"SELECT address, chain, name, current_tier, current_score, status"
            f" FROM wallets WHERE address={_PH} AND chain={_PH}",
            (single_wallet, chain),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT address, chain, name, current_tier, current_score, status"
            f" FROM wallets WHERE chain={_PH} AND status != 'dormant'",
            (chain,),
        ).fetchall()
    conn.close()

    wallets = [dict(r) for r in rows]
    log.info("Phase 6 tiering — %d wallets (chain=%s)%s",
             len(wallets), chain, "  [DRY RUN]" if dry_run else "")

    if not wallets:
        log.warning("No wallets found.")
        return

    results      = []
    promotions   = []
    demotions    = []
    new_dormant  = []

    for w in wallets:
        addr     = w["address"]
        wchain   = w.get("chain") or chain
        name     = w.get("name") or addr[:10]
        old_tier = w.get("current_tier")
        old_status = w.get("status", "active")

        # Pull last 4 daily score snapshots (newest first)
        # composite_score is written by Phase 4 — use it directly for all snapshots
        conn = get_conn()
        snaps = conn.execute(
            f"""
            SELECT score_date, hit_rate, median_return, trade_count,
                   COALESCE(composite_score, 0.0) as composite_score
            FROM wallet_scores_history
            WHERE wallet_address={_PH} AND chain={_PH}
            ORDER BY score_date DESC
            LIMIT 4
            """,
            (addr, wchain),
        ).fetchall()
        conn.close()

        history = [dict(s) for s in snaps]

        scored_history = []
        for h in history:
            scored_history.append({
                "score":       h.get("composite_score") or 0.0,
                "trade_count": h.get("trade_count") or 0,
            })

        stab = _compute_stability(scored_history)
        new_tier = _assign_tier(
            stab["stability_score"],
            stab["avg_trades"],
            stab["active_weeks"],
        )

        # Check for dormant: all 4 snapshots had 0 trades
        all_zero = stab["active_weeks"] == 0 and len(history) >= DORMANT_SNAPSHOTS
        if all_zero and old_status == "active":
            new_dormant.append(name)

        # Detect transitions
        if new_tier != old_tier:
            tier_rank = {"S": 3, "A": 2, "B": 1, None: 0}
            if (tier_rank.get(new_tier, 0) > tier_rank.get(old_tier, 0)):
                promotions.append((name, addr, old_tier, new_tier, stab["stability_score"]))
            elif (tier_rank.get(new_tier, 0) < tier_rank.get(old_tier, 0)):
                demotions.append((name, addr, old_tier, new_tier, stab["stability_score"]))

        results.append({
            "name":             name,
            "address":          addr,
            "old_tier":         old_tier,
            "new_tier":         new_tier,
            "stability_score":  stab["stability_score"],
            "active_weeks":     stab["active_weeks"],
            "avg_trades":       stab["avg_trades"],
            "all_zero":         all_zero,
        })

        log.debug(
            "  %s  old=%s new=%s  stab=%.2f  active_wks=%d  avg_tr=%.1f",
            name[:20], old_tier, new_tier,
            stab["stability_score"], stab["active_weeks"], stab["avg_trades"],
        )

        if not dry_run:
            conn = get_conn()
            if all_zero:
                conn.execute(
                    f"UPDATE wallets SET status='dormant', current_tier=NULL"
                    f" WHERE address={_PH} AND chain={_PH}",
                    (addr, wchain),
                )
            else:
                conn.execute(
                    f"UPDATE wallets SET current_tier={_PH}"
                    f" WHERE address={_PH} AND chain={_PH}",
                    (new_tier, addr, wchain),
                )
            # Update tier in today's history row if it exists
            conn.execute(
                f"""
                UPDATE wallet_scores_history SET tier={_PH}
                WHERE wallet_address={_PH} AND chain={_PH} AND score_date={_PH}
                """,
                (new_tier, addr, wchain, today),
            )
            conn.commit()
            conn.close()

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    results.sort(key=lambda x: x["stability_score"], reverse=True)
    W = 84
    print(f"\n{'='*W}")
    print(f"Phase 6 Tiering — {today} — {len(results)} wallets"
          + ("  [DRY RUN]" if dry_run else ""))
    print(f"{'='*W}")
    print(f"{'Name':<22} {'Old':>4} {'New':>4} {'Stab':>6} {'AvgTr':>6} {'ActWk':>6}  Change")
    print("-" * W)

    for r in results:
        change = ""
        rank = {"S": 3, "A": 2, "B": 1, None: 0}
        if r["new_tier"] != r["old_tier"]:
            if rank.get(r["new_tier"], 0) > rank.get(r["old_tier"], 0):
                change = "↑ PROMOTED"
            elif rank.get(r["new_tier"], 0) < rank.get(r["old_tier"], 0):
                change = "↓ demoted"
        if r["all_zero"]:
            change = "→ DORMANT"

        print(
            f"{r['name']:<22} "
            f"{str(r['old_tier'] or '-'):>4} "
            f"{str(r['new_tier'] or '-'):>4} "
            f"{r['stability_score']:>6.2f} "
            f"{r['avg_trades']:>6.1f} "
            f"{r['active_weeks']:>6d}  "
            f"{change}"
        )

    s_count  = sum(1 for r in results if r["new_tier"] == "S")
    a_count  = sum(1 for r in results if r["new_tier"] == "A")
    b_count  = sum(1 for r in results if r["new_tier"] == "B")
    no_count = sum(1 for r in results if r["new_tier"] is None and not r["all_zero"])
    dm_count = sum(1 for r in results if r["all_zero"])

    print(f"\nS={s_count}  A={a_count}  B={b_count}  Untiered={no_count}  NewDormant={dm_count}")
    if promotions:
        print(f"\nPromotions ({len(promotions)}):")
        for name, addr, old, new, score in promotions:
            print(f"  {name:<22}  {old or '-'} → {new}  (stab={score:.2f})")
    if demotions:
        print(f"\nDemotions ({len(demotions)}):")
        for name, addr, old, new, score in demotions:
            print(f"  {name:<22}  {old} → {new or '-'}  (stab={score:.2f})")
    print(f"{'='*W}\n")

    # ---------------------------------------------------------------------------
    # Telegram alerts for S-tier promotions
    # ---------------------------------------------------------------------------
    if not dry_run and promotions:
        s_promos = [(n, a, o, nw, sc) for n, a, o, nw, sc in promotions if nw == "S"]
        if s_promos:
            lines = ["*Wallet Intelligence — S-Tier Promotions*"]
            for name, addr, old, new, score in s_promos:
                lines.append(f"• {name} `{addr[:8]}…`  {old or '?'} → S  (score={score:.1f})")
            _tg("\n".join(lines))
            log.info("Sent Telegram alert for %d S-tier promotion(s)", len(s_promos))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6 — Weekly tiering job")
    parser.add_argument("--wallet",  type=str,  default="",       help="Single wallet test mode")
    parser.add_argument("--chain",   type=str,  default="solana")
    parser.add_argument("--dry-run", action="store_true",         help="Compute only, no DB writes")
    args = parser.parse_args()
    run_tiering(single_wallet=args.wallet, chain=args.chain, dry_run=args.dry_run)
