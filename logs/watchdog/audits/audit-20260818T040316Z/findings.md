# Layer 2 Audit: audit-20260818T040316Z
Generated: 2026-08-18T04:03:15Z
Evidence SHA-256: `bb9c9dbd984f7d7fc28f7c96eba1982e99b491b00ad37b3bfcead0c55699863f`
Status: ok

## [WARN] F2: Working tree cleanliness
- **Claim**: Full suite on the VPS's real environment... 749 passed... zero regressions in V8/V7 runtime behavior.
- **Observed**: Ground truth confirms the working tree is not clean: 6 modified tracked files and numerous untracked files/directories including a new script and spool jsonl files (EV002). No evidence was gathered about actual test pass/fail counts.
- **Expected**: A claim of a clean, fully-tested, verified end-to-end deployment implies a working tree free of uncommitted changes, or at least an explanation of what remains dirty relative to the claimed final commit.
- **Evidence**: EV002
- **Impact**: Uncommitted local changes on the deployed host mean the actually-running code may diverge from the tested/committed state described in the claim, undermining the 'verified end to end' assertion.
- **Next step**: Run `git status --porcelain` and `git diff` on the VPS to enumerate exactly which files differ from `5f76491` and assess whether they affect V8 engine behavior.
- **Confidence**: medium

## [WARN] F8: research.path_collection incident vs. Item 5/backfill claims
- **Claim**: confirms the forward pipe keeps working continuously, not just as a one-time fix
- **Observed**: Ground truth shows `research.path_collection` is currently FIRING (since 1786917790, consecutive_failures: 6, having recovered once and re-fired), with path yield 29.1% (below 50% floor) and PumpPortal daily message budget exceeded (105,406/100,000, 88 tokens dropped) (EV005).
- **Expected**: A claim of a continuously-working forward data pipe is in tension with an active FIRING incident on the related research path-collection check at the same approximate evidence-capture window.
- **Evidence**: EV005
- **Impact**: There may be a live data-quality problem (low path yield, dropped tokens from budget overrun) affecting the same research/venue-state pipeline the claim describes as healthy, which could undermine confidence in the 121-row venue-state backfill claim's forward-looking continuity.
- **Next step**: Correlate timestamps of the `research.path_collection` FIRING incident with the venue_state_at_signal backfill runs to determine whether the low path yield affects the same data flow being claimed as fixed.
- **Confidence**: medium

## [INFO] F1: SHA / commit identity
- **Claim**: Final git SHA: `5f76491`, pushed to origin/main, pulled and verified on the VPS (`git rev-parse HEAD` matches on both sides).
- **Observed**: Ground truth shows deployed HEAD is `a82f90a0286197382938aa41fa4430e86e24addb` (EV002/EV005), and a separate commit `db32f53` appears in claims.batch.rc_closure (EV005). Neither of these matches `5f76491`, and no evidence in the ground-truth set references `5f76491`, `e50c42c`, or `8eebcff` at all.
- **Expected**: If the claim is accurate, the deployed HEAD SHA and/or the VPS `git rev-parse HEAD` should match `5f76491`, and this SHA should appear somewhere in watchdog/job receipts.
- **Evidence**: EV002, EV005
- **Impact**: Cannot verify from available evidence whether the Phase 2.1 engine code described in the claim is actually deployed/running on the audited host; the claim's SHA is unconfirmed against the ground-truth deployed SHA.
- **Next step**: Run `git rev-parse HEAD` and `git log --oneline -5` on the VPS and compare directly against `5f76491`, `a82f90a0`, and `db32f53` to establish their relationship.
- **Confidence**: high

## [INFO] F3: V8 readiness / claims.batch.v8_readiness
- **Claim**: `SELECTION_DATA_READY` remains false (unchanged in substance — the schema now works, the sample doesn't exist yet)... Holdout was not evaluated. No candidate was ranked.
- **Observed**: Ground truth shows `claims.batch.v8_readiness` is currently FIRING (consecutive_failures: 146) with 3/7 GREEN, 4/7 PARTIAL (N2, N4, N6, N7), 0 FAIL — explicitly noted as a case where CLI exit code 0 could mask incompleteness (EV005).
- **Expected**: If SELECTION_DATA_READY is genuinely false and this is expected/acknowledged, the claim's framing is consistent with an ongoing FIRING incident rather than a resolved state; however the claim does not mention the FIRING incident or its 146 consecutive failures.
- **Evidence**: EV005
- **Impact**: The claim's narrative of 'engine ready, just waiting for data' is plausible but omits that the corresponding automated readiness check has been continuously failing for a long duration, which should be cross-referenced before treating this status as fully benign.
- **Next step**: Inspect `claims.batch.v8_readiness` check detail history and confirm whether N2/N4/N6/N7 PARTIAL states are expected under 'waiting for SELECTION_DATA_READY' or indicate an unrelated problem.
- **Confidence**: medium

## [INFO] F4: V8 live-trading status
- **Claim**: V8 has never been live-traded... V7/V8 strategy code and live trading were not touched.
- **Observed**: Ground truth has no direct evidence confirming or refuting live-trading activity for V8; EV005's cron and service checks show quantbot.service and quantbot-research.service active with last exit success, but nothing in the ground-truth summary speaks to which strategy version is live-trading.
- **Expected**: No specific ground-truth evidence contradicts this claim, but none confirms it either — this is an assertion outside the audited evidence scope.
- **Evidence**: EV003, EV005
- **Impact**: Cannot independently verify the claim that V8 has never been live-traded; if false, this would represent a significant risk given the stated STATIC_ASSUMPTION pricing and unproven cohort matching.
- **Next step**: Query live trade execution logs/telemetry (e.g. `logs/trade_telemetry_summary.csv`) for strategy/version tags to confirm no V8-tagged live trades exist.
- **Confidence**: low

## [INFO] F5: Venue-state column backfill (Item 5)
- **Claim**: 121 rows now carry a non-null `venue_state_at_signal` (up from 97 at recovery time — confirms the forward pipe keeps working continuously).
- **Observed**: No evidence in the ground-truth summary references `venue_state_at_signal`, row counts, or the memecoin_live_journal.csv schema state at all.
- **Expected**: This is a specific, checkable numeric claim; ground truth has no corroborating or contradicting evidence.
- **Evidence**: EV002
- **Impact**: Cannot confirm this specific data-quality claim from current evidence; if inflated or stale, downstream V8 readiness metrics relying on this field would be affected.
- **Next step**: Run a row count query: `awk -F',' 'NR>1 && $VENUE_STATE_COL!=""' logs/memecoin_live_journal.csv | wc -l` to independently verify the 121-row figure.
- **Confidence**: low

## [INFO] F6: SOL/USD static conversion constant
- **Claim**: the price-capture pipeline itself computes `price_usd = price_sol * 175.0` as a fixed constant (verified against 202,385 integrity-qualified real ticks).
- **Observed**: No evidence in the ground-truth summary examines price-capture pipeline code, SOL/USD conversion logic, or tick counts.
- **Expected**: This is a verifiable code/data claim outside the scope of the original audit evidence (EV002-EV006); ground truth neither confirms nor denies it.
- **Evidence**: EV002
- **Impact**: The STATIC_ASSUMPTION pricing constant materially affects any USD-denominated P&L or cost metrics derived from V8 evidence; unverified provenance of the 175.0 constant is a latent risk to downstream financial reporting.
- **Next step**: Grep the price-capture module for the literal `175.0` and cross-check against a sample of raw tick data to confirm the claimed p10-p90 range.
- **Confidence**: low

## [INFO] F7: Test suite counts
- **Claim**: 749 passed (up from 703 before Phase 2.1)... Top-level tests/: 49 passed, same 7 pre-existing unrelated failures.
- **Observed**: Ground truth's test-collection checks (EV005) report collection counts only (memecoin 288, research 358, watchdog 103, layer2 39, quant-bot root 56 tests) and note these 'collect cleanly' — this is test *collection*, not test *execution/pass* counts, and does not corroborate the 749-passed or 49-passed/7-failed figures.
- **Expected**: If claim is accurate, a full test run log or CI artifact showing 749 passed / 49 passed with 7 known failures should exist and be inspectable.
- **Evidence**: EV005
- **Impact**: The collection-only checks in the ground truth cannot substitute for actual pass/fail verification; discrepancy between 'collects cleanly' and 'passed' terminology could mask real failures not caught by collection checks alone.
- **Next step**: Run `pytest research/tests/ watchdog/tests/ memecoin/tests/ -q` and `pytest tests/ -q` on the VPS and capture actual pass/fail summary to compare against the claimed 749/49 figures.
- **Confidence**: medium
