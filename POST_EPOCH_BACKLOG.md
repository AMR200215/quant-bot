# POST_EPOCH_BACKLOG

Items deferred from the Z1-Z7 pre-epoch batch. Address before or after scale-up depending on severity.

---

## 1. Full Position State Machine (high priority, post-epoch)

Replace ad-hoc `notes`-string state tracking with a proper position state machine:

```
open → exit_pending → tx_pending → closed_confirmed
```

- `open`: position active, monitoring price
- `exit_pending`: exit intent recorded (`exit_intent_reason` set), no TX sent yet
- `tx_pending`: TX sent (`pending_signature` set), awaiting on-chain confirmation
- `closed_confirmed`: TX confirmed on-chain, journal row written

**Why deferred**: Z2 adds the serialised fields; full wiring requires refactoring `close_position()` and the retry ladder, which is multi-day scope. Z2 fields are the prerequisite — implement the state transitions once the epoch is complete.

---

## 2. PumpSwap T22 TX Builder Fix (medium priority)

`pumpswap_local.py` builds TXs with the SPL token program ID. T22 accounts use `TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb` — mismatch produces `pumpswap_bad_pool_layout`.

**Effect today**: Jupiter rescue fallback handles graduated T22 sells. Fails gracefully but wastes a route attempt.

**Fix**: Detect T22 from `mint_classifier.get_token_program()` at TX-build time and substitute the T22 program ID in the instruction accounts.

---

## 3. Jupiter as Primary Route for Graduated T22 (medium priority)

With Z3 in place (`strategy_pure_rider` graduation = venue update only), post-graduation T22 sells flow through: pump-amm TX (fails: pool layout bug) → Jupiter rescue. Cut the failing step:

When `lifecycle_state == "graduated"` AND `token_program == "T22"`, set `venue_state_json = {"primary": "jupiter"}` at graduation detection time. `close_position()` Z4 venue routing picks it up directly.

---

## 4. Signal Age Gate for Stale TG Alerts (medium priority)

`pumpdotfunalert` posts batches 5-15 min after breakout. Entry optimisation can't fix stale signals — the token has already moved.

**Gate spec**: if `age_minutes > MAX_SIGNAL_AGE_MIN` (e.g. 8) at entry time, block live buy. Log reason as `stale_signal`. Use post-epoch live journal data to calibrate the threshold.

---

## 5. Entry Age vs Expected Return Analysis (post-epoch)

After the 15-20 canary trades, measure whether signal age > 5 min correlates with worse P&L. Use `signal_time` vs `entry_time` in the live journal. Make a data-driven gate decision.

---

## 6. T22 test_live_program_gate test updates

`memecoin/tests/test_live_program_gate.py` has 2 pre-existing failures (`test_t22_blocked`, `test_t22_with_complete_curve_still_blocked`) left over from commit 74d5675 which removed the T22 buy gate. Update these tests to assert `allowed=True` for clean T22 tokens, matching the current gate behaviour.

---

## 7. legacy_graduation_guard cohort removal

Once `strategy_pure_rider` is validated across the first epoch, retire `legacy_graduation_guard`. All positions become `strategy_pure_rider`. The split exists only for backward compatibility with positions created before Z7.
