# Layer 2 Audit: audit-20260828T153118Z
Generated: 2026-08-28T15:31:16Z
Evidence SHA-256: `bcd2297c7ebe5e32f9763f1d10cb4505d4b462d8b48840dd851bde1e63197d5f`
Status: ok

## [WARN] F2: claims.batch.v8_readiness vs. readiness-denominator fix narrative
- **Claim**: Claims describe a 'readiness denominator audit/correction' that fixed path_coverage/execution_proxy_coverage by separating population counts from collector-yield counts, implying readiness metrics should now be trustworthy/accurate.
- **Observed**: Ground truth shows `claims.batch.v8_readiness` is FIRING (WARN) with consecutive_failures: 435, first_seen 1786585581, and latest slow check shows only 3/7 GREEN, 4/7 PARTIAL (N2, N4, N6, N7), 0 FAIL — i.e. the readiness batch is not fully green despite the claimed fix.
- **Expected**: If the denominator bug fix fully resolved readiness measurement issues, one would expect the v8_readiness batch to show improvement toward GREEN rather than a persistent 435-failure FIRING WARN incident.
- **Evidence**: EV005
- **Impact**: The documented fix may address a different failure mode than what is driving the current v8_readiness PARTIAL results, so readiness is still not fully validated in production.
- **Next step**: Inspect the N2, N4, N6, N7 PARTIAL item definitions in the v8_readiness batch and cross-reference whether they depend on path_coverage/execution_proxy_coverage metrics claimed to be fixed.
- **Confidence**: medium

## [WARN] F3: funnel.v8 (V8-TWIN-FIX) vs. PumpPortal funding / pipeline claims
- **Claim**: Claims assert PumpPortal funding was verified live end-to-end with real production traffic, correct reserve price, valid path integrity, and 'no code changes needed' — implying the V8 collection pipeline is functioning correctly.
- **Observed**: Ground truth shows `funnel.v8` is FIRING CRITICAL with consecutive_failures: 747, describing a still-unresolved 'silent candidate disappearance' (V8-TWIN-FIX failure class) — a candidate entered telegram_received with no terminal disposition recorded, as of the latest fast run.
- **Expected**: If the pipeline is fully fixed and verified end-to-end as claimed, funnel.v8 would not still be in an actively FIRING critical state with a persistently growing failure streak.
- **Evidence**: EV005
- **Impact**: There is an active, unresolved critical funnel failure in production that the documentation's success narrative does not acknowledge or address.
- **Next step**: Trace event_id bd6003380054eb57 (mint AFf278av4oRQicFnpeGHFvcXqjmfNac3XaVyVZAhpump) through the v8 funnel logs to determine why no terminal disposition was recorded despite the claimed fix.
- **Confidence**: high

## [INFO] F1: SHA provenance / receipts narrative
- **Claim**: Claims reference specific git SHAs for work: `2c2ab8c` (readiness denominator audit), `7d2e319`/`333dbfb` (path_stats refactor), and 'SHA pending this commit' for the E3 exit candidate addition.
- **Observed**: Ground truth shows HEAD SHA a98c057e..., job receipts embedded in EV005 show git_sha 8fecc0e5..., and claims.batch.rc_closure is tied to commit db32f53. None of these match the SHAs cited in the claims text (2c2ab8c, 7d2e319, 333dbfb).
- **Expected**: If the claims narrative is current and accurate, at least one of the cited SHAs would be traceable to or consistent with the HEAD/job-receipt/rc_closure SHAs already flagged as mismatched in the audit.
- **Evidence**: EV002, EV005, EV007
- **Impact**: Cannot verify whether the described readiness-denominator fix, path_stats refactor, or E3 registry change are actually present in the deployed working tree or merely documented as done.
- **Next step**: git log --oneline --all | grep -E '2c2ab8c|7d2e319|333dbfb' and diff against HEAD a98c057e to confirm these commits are ancestors of the deployed SHA.
- **Confidence**: medium

## [INFO] F4: Test suite counts vs. claims narrative
- **Claim**: Claims state '18 new tests, 488/488 total green' (denominator fix) and later '2 new tests ... Full suite 498/498 green' (E3 exit candidate).
- **Observed**: Ground truth's latest test-collection check shows tests_research at 498 tests collecting cleanly, consistent with the final '498/498 green' figure cited in the claims; the intermediate 488/488 figure cannot be independently verified from current-state evidence since only the final collected count is visible.
- **Expected**: The final test count in claims (498) matches the currently observed collected test count for tests_research, which is a point of agreement.
- **Evidence**: EV005
- **Impact**: This specific figure is consistent, but reconciling with the SHA mismatches, this consistency alone does not confirm the surrounding functional narrative (readiness fixes, E3 evaluation status).
- **Next step**: Run pytest --collect-only on tests_research and diff against 498 to confirm count is not coincidental given SHA discrepancies.
- **Confidence**: medium

## [INFO] F5: E3 exit candidate — holdout/evaluation status
- **Claim**: Claims explicitly state E3 'has not been evaluated against E0/E1/E2' and 'Holdout untouched throughout,' with entry into the readiness/replay pipeline gated on SELECTION_DATA_READY.
- **Observed**: Ground truth has no evidence bundle item describing exit-registry contents, EXIT_REGISTRY_VERSION, SELECTION_DATA_READY, or holdout status — this is entirely outside what EV001-EV007 cover.
- **Expected**: Cannot confirm or deny; this is a self-contained claim about research methodology not covered by watchdog/systemd/cron evidence.
- **Evidence**: EV005
- **Impact**: No operational risk currently identifiable, but the claim is unverifiable from infrastructure-level evidence alone.
- **Next step**: Locate and inspect research/v8_exit_registry.py state (EXIT_REGISTRY_VERSION, ENGINE_READY, SELECTION_DATA_READY flags) directly in the deployed working tree.
- **Confidence**: low

## [INFO] F6: research.path_collection yield vs. claims of validated path integrity
- **Claim**: Claims assert 'path integrity VALID' and a working end-to-end path pipeline following PumpPortal funding on 2026-08-22.
- **Observed**: Ground truth shows research.path_collection reports OK overall for 2026-08-27 (86.0% yield) but raw hourly data shows usable_path_yield_pct: 0.0 for UTC hours 16-23 on that same day, and this check has a prior RECOVERED incident (first_seen 1786917790, recovered 1787875388).
- **Expected**: If the pipeline were fully and consistently validated as claimed, sustained zero-yield windows within a day rated overall OK would not be expected, or would be explained by the documentation.
- **Evidence**: EV005
- **Impact**: Path collection yield is inconsistent within observed windows despite an overall OK status and despite claims of a fully verified, funded end-to-end pipeline.
- **Next step**: Query research path collection logs for UTC hours 16-23 on 2026-08-27 to determine root cause of zero usable-path yield during that window.
- **Confidence**: medium
