# Layer 2 Audit: audit-20260921T090921Z
Generated: 2026-09-21T09:09:20Z
Evidence SHA-256: `d7070fb8967c8b776db6ce0e99a5e95c93ecbe0c6f1c3d9b8d65415a38c10e94`
Status: ok

## [WARN] F3: v8_paper monitor interval tightening
- **Claim**: _MONITOR_INTERVAL_S tightened from 5.0s to 1.0s, committed 07d09f1, deployed, live-verified clean (startup log confirms interval=1s; zero errors, zero duplicates post-restart).
- **Observed**: Ground truth shows an active FIRING CRITICAL incident `funnel.v8` with 7676 consecutive failures and a stuck candidate in telegram_received stage, and a FIRING WARN incident `claims.batch.v8_readiness` with 4/7 items PARTIAL. No evidence confirms commit 07d09f1 deployment or startup log content for v8_paper interval.
- **Expected**: Claim implies v8-related pipeline is clean and functioning well post-deploy; ground truth shows ongoing, long-running v8 funnel failures inconsistent with a fully healthy 'live-verified clean' state.
- **Evidence**: EV005
- **Impact**: The claimed clean live-verification for v8_paper changes coexists with unresolved, long-standing v8 funnel and readiness incidents, suggesting the fix (if deployed) has not resolved or is unrelated to the ongoing funnel.v8 problem.
- **Next step**: Check watchdog `funnel.v8` incident detail and correlate stuck telegram_received candidate timestamps against the claimed 07d09f1 deploy time to see if the incident predates or postdates the fix.
- **Confidence**: medium

## [INFO] F1: positions.json batch-close performance fix
- **Claim**: Fixed: split _close() into _close_no_save()/_finish_close()/_close_batch(); post-fix 3000-position/~1000-close scenario takes ~150-300ms; committed 9589c03, deployed.
- **Observed**: Ground truth has no evidence of commit 9589c03, no evidence of positions.json rewrite behavior, batch-close performance, or test counts (52/52, 709 across suites). Deployed HEAD is b5e27fdd... per EV002; no relationship to 9589c03 is established.
- **Expected**: Claim implies commit 9589c03 is deployed and reflected in current HEAD/tests.
- **Evidence**: EV002, EV007
- **Impact**: Cannot verify whether the performance fix is actually present in the currently deployed code from audit evidence alone.
- **Next step**: Run `git log --oneline b5e27fdd913491c05d101be4ec40d2708841ec3 | grep 9589c03` to confirm the commit is an ancestor of deployed HEAD.
- **Confidence**: medium

## [INFO] F2: Live post-restart position book claim
- **Claim**: Live post-restart: zero errors, zero duplicates, book actively growing (236 total positions vs 194 at the last check, 3 genuinely open and tracking).
- **Observed**: No evidence in EV001-EV006 references positions.json contents, position counts, or error/duplicate logs for the position book.
- **Expected**: Claim asserts specific live operational metrics (236 vs 194 positions, 3 open) that would need to be independently observable in logs or state files.
- **Evidence**: EV002
- **Impact**: Unverified operational claims about position book health cannot be corroborated by the audit; risk of stale or inaccurate self-reporting.
- **Next step**: Inspect `memecoin/data/memecoin_signals.json` or equivalent positions.json for current position count and cross-check against journal logs for duplicate/error entries.
- **Confidence**: low

## [INFO] F4: PnL analysis claim (v8_paper trades)
- **Claim**: 26 trades closed since 2026-09-18 deploy; win rate 50%, mean -5.1%, net -$3.99; 4 hard_stop trades with -56.9%/-59.1% overshoot vs -35% target.
- **Observed**: No evidence in EV001-EV006 references trade counts, PnL figures, win rates, or hard_stop overshoot percentages.
- **Expected**: Claim asserts detailed trade-level financial outcomes that would require log/telemetry evidence to confirm.
- **Evidence**: EV002
- **Impact**: Financial performance claims are unverifiable from current audit evidence; decisions based on this PnL data carry unconfirmed risk.
- **Next step**: Query `logs/trade_telemetry_summary.csv` (noted as modified in EV002) for trade counts and PnL since 2026-09-18 to independently verify the 26-trade, -$3.99 net figures.
- **Confidence**: low

## [INFO] F5: Test suite counts
- **Claim**: 709 passing across all v8-related suites (referenced in both EV007 sections); 52/52 in the batch-close fix file.
- **Observed**: Ground truth EV005 reports test-collection counts of 335 (tests_memecoin), 635 (tests_research), 103 (tests_watchdog), 39 (tests_layer2), 56 (tests_quant-bot) for clean collection, not pass/fail counts, and does not mention a 709 or 52 figure or a v8-specific suite grouping.
- **Expected**: Claim's '709 across all v8-related suites' figure should be reconcilable with or derivable from watchdog test-collection numbers if v8-related suites are a subset of those collections.
- **Evidence**: EV005
- **Impact**: Discrepancy in test accounting terminology (collection count vs pass count, and suite scoping) makes it impossible to confirm the 709-passing claim against watchdog's own test-collection evidence.
- **Next step**: Run the actual v8-related test suites (`pytest -k v8`) and compare total collected/passed count against the claimed 709.
- **Confidence**: medium

## [INFO] F6: Not-yet-confirmed disclosure in claim
- **Claim**: 'Not yet confirmed: whether this actually reduces the overshoot magnitude on future hard_stop closes — needs more real trades to check.'
- **Observed**: Ground truth has no data on hard_stop closes, overshoot magnitude, or trade counts post-tightening; this is consistent with the claim's own admission of insufficient data, and no contradiction is found.
- **Expected**: Claim appropriately flags its own uncertainty; ground truth neither confirms nor denies this since no trade-level evidence exists in the audit bundle.
- **Evidence**: EV002
- **Impact**: No immediate operational risk since the claim self-identifies as unverified; but it underscores that broader PnL/trade claims in this document are not corroborated by current audit evidence.
- **Next step**: Track hard_stop trade closes going forward and compare overshoot percentages against the -35% target once more trades accumulate.
- **Confidence**: high
