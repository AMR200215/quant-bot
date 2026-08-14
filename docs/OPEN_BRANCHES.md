# Open Branches — Cross-Session Tracker

Different Claude Code sessions (desktop, mobile, others) working on this
repo don't share memory or see each other's branches until something is
merged into `main`. This file exists so that gap doesn't cost anyone
another round of `git branch -r` + manual diff archaeology to rediscover
work that's sitting unmerged.

**Rule**: if you push a branch you're not merging immediately, add a row
here in the same push. If you review or merge a branch listed here,
update its row in the same commit — don't leave it stale.

| Branch | Origin session | Base commit (age) | Status | Notes |
|---|---|---|---|---|
| `claude/quant-bot-ec9cm3` | unknown session, ~2026-08-04 | 60 commits behind main as of 2026-08-14 | **MERGED** (`2313000`, 2026-08-14) | Price sanity guard for the SPOTTY incident. Reviewed, merged, deployed, and verified live in the running `quantbot` process — see `docs/RECEIPTS.md`'s "Price Sanity Guard" section. |
| `claude/mobile-quant-bot-status-tS0za` | mobile Claude Code session | forked 2026-06-07 — **892 commits behind main** as of 2026-08-14 | **NOT MERGED — needs re-verification, not a blind merge** | See below. |

## `claude/mobile-quant-bot-status-tS0za` — detailed status

Two commits, four claimed fixes, none merged:

1. **`alert_live_buy`/`alert_live_sell`/`alert_live_close`/`alert_live_skip`
   missing from `app/alerts.py`, silently swallowed by `except: pass`, so
   no live-trade Telegram alert ever fires.**
   **CONFIRMED STALE as of 2026-08-14** — `alert_live_buy` and
   `alert_live_sell` both exist in current `main`'s `app/alerts.py` and
   are correctly imported by `memecoin/portfolio.py`. Something else
   fixed this (or it was added) sometime in the ~2 months since this
   branch forked. Do not merge this part.

2. **`alert_live_buy` unconditionally appends `"Tx: ..."` even when
   `tx_sig=""`** (missing the `if tx_sig:` guard that `alert_live_close`
   already has). **Not re-verified against current `main`** — check
   `app/alerts.py::alert_live_buy` directly before trusting this.

3. **`alert_live_sell` accepts `sol_received` but silently drops it by
   delegating to `alert_live_close`**, losing the only way to verify real
   exit PnL vs DexScreener. **Not re-verified against current `main`.**

4. **`portfolio.py`'s live-position-count gate calls `_count_open_live()`
   twice** (once to check the gate, once for the log message), opening a
   race window for a concurrent thread to open a position between the two
   reads. **Not re-verified against current `main`.**

5. (Second commit) **Live gate's `dex_id` check fails silently for
   graduated pump.fun tokens**, which have `dex_id="raydium"` after
   migration and don't match `"pump" in dex_id.lower()`. **Not
   re-verified against current `main`** — this is the same *class* of
   bug (`dex_id` used as a proxy for something it doesn't reliably
   indicate) that V8-TWIN-FIX root-caused for a completely different
   function back on 2026-08-11 (see `docs/RECEIPTS.md`), so it's
   plausible this is still real even though the specific line numbers
   have likely shifted — worth checking first, not dismissing because
   claim #1 turned out to be stale.

6. (Second commit) **v7 filter rejections logged at `debug` instead of
   `info`**, making them invisible in production logs. Low-risk,
   observability-only — worth checking whether this is still true and
   applying directly if so, independent of the other claims.

**Recommended next step for whoever picks this up** (any session with
read access to `main` can do this — it's pure code reading, no VPS
needed): open `app/alerts.py` and `memecoin/portfolio.py` on current
`main`, check claims 2-6 one at a time against the actual current code,
and for each one that's still real, re-implement it clean against
current `main` rather than attempting to merge the stale branch. Update
this file's table to `MERGED` (with what actually landed) once done, or
note here which claims turned out to also be stale.
