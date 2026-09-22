# Layer 2 Audit: audit-20260922T084104Z
Generated: 2026-09-22T08:41:02Z
Evidence SHA-256: `f31ac801b93db3f13846412a7527dbd8adfe949b8ddc1e2f4cfa52d9338cf908`
Status: ok

## [WARN] F3: v8_paper monitor interval tightening (07d09f1)
- **Claim**: Tightened _MONITOR_INTERVAL_S from 5.0 to 1.0s; committed 07d09f1, deployed, live-verified clean (startup log confirms interval=1s; zero errors, zero duplicates post-restart).
- **Observed**: Ground truth has no evidence of a startup log confirming interval=1s, no commit 07d09f1 in the git evidence, and no independent confirmation of zero errors/duplicates for v8_paper specifically. The only related ground-truth signal is the ongoing funnel.v8 CRITICAL incident (7963 consecutive failures) describing unresolved candidates in the v8 funnel, which is a different but adjacent v8 subsystem issue.
- **Expected**: If deployed and live-verified as claimed, watchdog checks or logs should reflect a clean v8_paper monitor loop; the currently firing funnel.v8 CRITICAL incident is not mentioned or reconciled against this claim.
- **Evidence**: EV005, EV002
- **Impact**: The claim of a clean, error-free live-verified deploy is not corroborated by evidence, and coexists with an actively firing CRITICAL incident in a related v8 funnel path, raising doubt about whether 'zero errors' is accurate for the whole v8 pipeline.
- **Next step**: Grep the v8_paper service startup logs for 'interval=1s' and check timestamps against the funnel.v8 incident's first_seen (1786748706.66) to see if they correlate.
- **Confidence**: medium

## [WARN] F5: funnel.v8 CRITICAL incident vs. claims of clean deploys
- **Claim**: Both claims (positions batching fix and monitor interval tightening) assert clean, error-free, live-verified deploys with 'the system is doing real work under all six fixes... at once' and 'zero errors, zero duplicates post-restart.'
- **Observed**: EV005 shows funnel.v8 is CRITICAL and FIRING with 7963 consecutive failures as of the snapshot, and claims.batch.v8_readiness is WARN/FIRING with 1118 consecutive failures and recovered_at: null -- both directly contradicting a narrative of a fully clean, error-free v8 system.
- **Expected**: If all v8 fixes were deployed cleanly with zero errors as claimed, the funnel.v8 and v8_readiness checks would be expected to show OK/RECOVERED rather than actively firing CRITICAL/WARN incidents.
- **Evidence**: EV005
- **Impact**: Documentation claims of a fully healthy, error-free v8 deployment are inconsistent with two currently firing, long-running incidents in the same v8 subsystem, suggesting the claims overstate system health or refer to a narrower scope than the whole v8 pipeline.
- **Next step**: Correlate the funnel.v8 incident's first_seen/last_seen timestamps and the v8_readiness PARTIAL item IDs (N2,N4,N6,N7) against the deploy timestamps of commits 9589c03 and 07d09f1 to determine if these incidents predate, postdate, or overlap the claimed fixes.
- **Confidence**: high

## [INFO] F1: positions.json _close batching fix
- **Claim**: Fixed: split _close() into _close_no_save()/_finish_close()/_close_batch(); ~300-500x faster; 52/52 tests in file, 709 across all v8-related suites; committed 9589c03, deployed.
- **Observed**: Ground truth confirms deployed HEAD SHA f5a5097... and a nightly cron/watchdog pattern of successful runs, but has no commit metadata, no diff content, and no test-run evidence for commit 9589c03 or the claimed 709-test suite count.
- **Expected**: The claim implies a specific prior commit (9589c03) with a measured 89.66s -> 150-300ms performance fix and a verified test count, which would need to be visible in commit history or CI logs to confirm.
- **Evidence**: EV002, EV005
- **Impact**: Cannot verify the performance fix or test coverage claims from available evidence; if false, the batch-close O(n) risk described could still be present in production.
- **Next step**: Run `git log --oneline -- <positions module>` and `git show 9589c03` to confirm the commit exists and inspect the diff; re-run the test suite and time the batch-close benchmark.
- **Confidence**: medium

## [INFO] F2: book growth claim (positions count)
- **Claim**: Live post-restart: zero errors, zero duplicates, book actively growing (236 total positions vs 194 at the last check, 3 genuinely open and tracking).
- **Observed**: No evidence in the ground-truth summary (EV002/EV005/EV006) reports positions.json content or a position count; memecoin_signals.json is listed only as a modified tracked file with no content shown.
- **Expected**: A verifiable claim of this kind would require positions.json content or a corresponding check output showing counts of 236 vs 194.
- **Evidence**: EV002
- **Impact**: Unverified growth/error-free claim could mask duplicate-close or error conditions not caught by current watchdog checks.
- **Next step**: Inspect positions.json directly (`jq '. | length'` or equivalent) and cross-check against journal/log entries for duplicate close events.
- **Confidence**: low

## [INFO] F4: PnL / trade statistics claim
- **Claim**: 26 trades closed since 2026-09-18 deploy; win rate 50%, mean -5.1%, net -$3.99 at $3/trade; 4 hard_stop trades overshooting to -56.9%/-59.1% vs -35% target.
- **Observed**: No trade-level PnL, win-rate, or overshoot data appears anywhere in the ground-truth evidence (EV001-EV007); the closest related evidence is the funnel.v8 CRITICAL incident about unresolved candidates, which is not the same as closed-trade PnL reporting.
- **Expected**: Verifying this claim would require trade ledger/PnL evidence not present in the audited evidence set.
- **Evidence**: EV005
- **Impact**: Financial performance figures are unverifiable from current evidence; decisions based on this PnL data cannot be independently corroborated.
- **Next step**: Pull the v8_paper trade ledger/closed-positions log directly and recompute win rate and mean/net PnL for the stated date range.
- **Confidence**: low

## [INFO] F6: Commit SHA / version correlation
- **Claim**: Committed 9589c03, deployed. / Committed 07d09f1, deployed.
- **Observed**: Ground truth's only known deployed SHA is f5a50974087b3b8b0d934661992a52f94c9f990c (EV002/EV005); no commit history, ancestry, or message list is available to confirm 9589c03 or 07d09f1 are ancestors of this HEAD.
- **Expected**: If both commits are truly deployed, they should appear as ancestors of the current HEAD SHA in git log.
- **Evidence**: EV002
- **Impact**: Cannot confirm these specific fixes are actually part of the currently running code without commit ancestry verification.
- **Next step**: Run `git merge-base --is-ancestor 9589c03 f5a5097...` and same for 07d09f1 to confirm both are in the deployed history.
- **Confidence**: medium
