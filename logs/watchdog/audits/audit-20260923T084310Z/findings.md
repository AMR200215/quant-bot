# Layer 2 Audit: audit-20260923T084310Z
Generated: 2026-09-23T08:43:08Z
Evidence SHA-256: `87a477bc9f8081eb164d4dd2500d5cafe71320a0c94bc6aa2f49e00034abe7d3`
Status: ok

## [WARN] F3: funnel.v8 vs. claimed 'zero errors, zero duplicates' live health
- **Claim**: Live post-restart: zero errors, zero duplicates, book actively growing... the system is doing real work under all six fixes from 2026-09-12 through today at once.
- **Observed**: EV005 shows funnel.v8 incident is CRITICAL and FIRING with consecutive_failures: 8255, first_seen 1786748706, describing a candidate (event_id bd6003380054eb57) with no terminal disposition recorded within the 120s window — an open, unresolved 'silent disappearance' class failure ('V8-TWIN-FIX failure class').
- **Expected**: The claim's characterization of a clean, error-free, fully-working v8 pipeline is inconsistent with an actively firing CRITICAL funnel.v8 incident with a very high failure count.
- **Evidence**: EV005
- **Impact**: The documentation's upbeat 'zero errors' narrative masks a long-running, unresolved CRITICAL alert in the v8 funnel that directly concerns candidate tracking/disposition — a functional gap in the same v8 pipeline the claims describe as healthy.
- **Next step**: Investigate the funnel.v8 CRITICAL incident directly: pull recent telegram_received events lacking terminal disposition and check logs/memecoin_social_journal.csv and v8 disposition logic for the affected mint/event_id.
- **Confidence**: high

## [INFO] F1: docs/RECEIPTS.md timing correlation
- **Claim**: Fixed: split _close() into _close_no_save()/_finish_close()/_close_batch()... Committed 9589c03, deployed.
- **Observed**: EV002 lists docs/RECEIPTS.md as a modified (uncommitted) tracked file in the dirty working tree; EV007 mtime (1790121481.75) aligns with the v8_vs_v7_daily receipt write, not specifically with commit 9589c03.
- **Expected**: The claim implies a clean, committed state ('Committed 9589c03, deployed') for this documentation change and the underlying code fix.
- **Evidence**: EV002, EV007, EV005
- **Impact**: The narrative of a finished, committed fix cannot be independently confirmed as fully landed in a clean commit; the doc describing it is itself part of the dirty tree.
- **Next step**: Run `git show 9589c03 --stat` and `git diff HEAD -- docs/RECEIPTS.md` to confirm the commit exists and whether RECEIPTS.md changes are already committed or still pending.
- **Confidence**: medium

## [INFO] F2: positions.json batch-close performance fix
- **Claim**: Post-fix: the same 3000-position/~1000-close scenario takes ~150-300ms — roughly 300-500x faster... Live post-restart: zero errors, zero duplicates, book actively growing (236 total positions vs 194 at the last check).
- **Observed**: No evidence in the audit bundle (EV001-EV007) measures positions.json write performance, close-batch counts, or a 236-vs-194 position comparison. This is entirely outside the scope of what was audited.
- **Expected**: Claim asserts a specific measured performance improvement and live position-count growth that would require dedicated telemetry/logs not present in this bundle.
- **Evidence**: EV005, EV002
- **Impact**: Cannot corroborate or refute the performance fix or the position-count claim from current evidence; this is an unverified assertion.
- **Next step**: Pull the actual v8_paper positions.json snapshot and timing logs around the claimed restart window to verify position counts and close-batch latency.
- **Confidence**: low

## [INFO] F4: V8_PAPER_MONITOR_INTERVAL_TIGHTENED — commit 07d09f1
- **Claim**: Tightened to 1.0s... Committed 07d09f1, deployed, live-verified clean (startup log confirms interval=1s; zero errors, zero duplicates post-restart).
- **Observed**: No evidence in the bundle (EV001-EV007) references commit 07d09f1, a startup log with interval=1s, or _MONITOR_INTERVAL_S configuration. The deployed HEAD SHA per EV002/EV005 is 8337624c9d108d3ff402dd9e939820679cac6d5b, and the working tree is dirty.
- **Expected**: Claim implies this specific commit is deployed and verifiable via startup logs, distinct from the currently observed deployed SHA.
- **Evidence**: EV002, EV005
- **Impact**: Cannot confirm whether the claimed monitor-interval commit is actually part of the currently running deployed SHA or is a separate, later, or uncommitted change sitting in the dirty tree.
- **Next step**: Run `git log --oneline -5 8337624c9d108d3ff402dd9e939820679cac6d5b` and `git branch --contains 07d09f1` to confirm whether 07d09f1 is an ancestor of the currently deployed commit.
- **Confidence**: medium

## [INFO] F5: test suite counts (709 across v8-related suites)
- **Claim**: 709 passing across all v8-related suites (no test depended on the old value).
- **Observed**: EV005 latest_slow_check_results shows five test suites collecting cleanly: tests_memecoin 335, tests_research 635, tests_watchdog 103, tests_layer2 39, tests_quant-bot 56 — none of these totals or subsets sum cleanly to a distinctly-labeled '709 v8-related' figure in evidence; the bundle does not break out a 'v8-related suites' subset.
- **Expected**: Claim references a specific cross-suite v8 test count (709) that would need to be traceable to the collected suite data.
- **Evidence**: EV005
- **Impact**: The specific v8-suite pass count in the documentation cannot be cross-validated against the test-collection evidence available, though overall collection health (no drift) is independently confirmed.
- **Next step**: Run the v8-specific test filter (e.g. `pytest -k v8 --collect-only`) and compare the resulting count against the claimed 709.
- **Confidence**: low

## [INFO] F6: claims.batch.v8_readiness PARTIAL items vs. claims of 'live-verified clean'
- **Claim**: live-verified clean (startup log confirms interval=1s; zero errors, zero duplicates post-restart)
- **Observed**: EV005 shows claims.batch.v8_readiness in WARN/FIRING state with consecutive_failures: 1146 and 4/7 items (N2, N4, N6, N7) PARTIAL, explicitly noting the batch's CLI would exit 0 despite incompleteness.
- **Expected**: Claim's framing of 'live-verified clean' for v8-related changes sits alongside an actively firing WARN incident indicating the v8 readiness batch is not fully green.
- **Evidence**: EV005
- **Impact**: Declaring the v8 subsystem 'live-verified clean' overstates readiness when a related readiness-batch check has been WARN/FIRING for 1146 consecutive cycles with unresolved PARTIAL items.
- **Next step**: Review the v8_readiness batch definitions for items N2, N4, N6, N7 to determine what remains PARTIAL and whether it relates to the monitor-interval or batch-close fixes described in the claims.
- **Confidence**: medium

## [INFO] F7: General agreement on cron/scheduler health
- **Claim**: (Implicit in receipts narrative of stable, working deploys)
- **Observed**: EV003/EV004/EV005 confirm cron files are well-formed, all three scheduled jobs exit 0 on recent runs, watchdog fast/slow runs both report final_runner_status: ok, and no parser errors are present — consistent with the claims' general picture of a functioning, actively-scheduled system.
- **Expected**: Claims imply an operationally healthy, actively-running system, which aligns with these specific pieces of evidence.
- **Evidence**: EV003, EV004, EV005
- **Impact**: This portion of the operational narrative is corroborated by evidence and not disputed.
- **Next step**: No action needed; continue routine monitoring of cron_execution.* checks.
- **Confidence**: high
