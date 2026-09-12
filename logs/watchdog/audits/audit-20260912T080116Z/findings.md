# Layer 2 Audit: audit-20260912T080116Z
Generated: 2026-09-12T08:01:14Z
Evidence SHA-256: `8ff528a0e6c2c886444ffd3de6f5db9d7062d982b24808b62fd27366ca994f2b`
Status: ok

## [WARN] F2: test suite green claim vs dirty working tree
- **Claim**: 21 new tests, full suite green (635 local).
- **Observed**: EV002 shows the working tree is dirty with modifications to tracked files and numerous untracked files, including logs and data files, with no explanation of why. Test collection checks in EV005 show clean collection but do not confirm pass/fail status of the full suite, only that collection (import/discovery) succeeds.
- **Expected**: A claim of 'full suite green' implies all tests pass, which is a stronger and different assertion than 'collection succeeds with no errors' as reported by test_drift.collection checks.
- **Evidence**: EV002, EV005
- **Impact**: Ground truth cannot confirm actual pass/fail test results, only that test discovery/collection is clean; the claim of a fully green suite is unverified by available evidence.
- **Next step**: Run the full test suite directly on the deployed commit and compare pass/fail counts against the claimed 635 green.
- **Confidence**: medium

## [WARN] F3: V8_PAPER_ENTRY_GATE_SWITCH deploy vs commit SHA
- **Claim**: Deploy: committed 07dd982, pushed, pulled on VPS clean (no conflicts), systemctl restart quantbot. Live log line confirms gate=CURVE_ACTIVE [V8-P0], config_tag=v8_candidate_2026-09-10_v8p0.
- **Observed**: EV002 shows the deployed HEAD SHA is 128210d7cd93eeb9518e8e315a9422adcc90f50b, not 07dd982. No evidence item in the ground-truth bundle shows this commit hash, the referenced config_tag, or the claimed live log line.
- **Expected**: If the deploy claim is accurate, the deployed HEAD should either be 07dd982 or a descendant commit that includes it, and some evidence (log grep, config inspection) should surface the v8_candidate_2026-09-10_v8p0 tag or CURVE_ACTIVE gate log line.
- **Evidence**: EV002
- **Impact**: Cannot confirm from evidence that the claimed V8-P0 gate change is actually present in the running deployed code; the commit SHA discrepancy (also seen with claims.batch.rc_closure reporting db32f53) raises doubt about which commit is truly live.
- **Next step**: Run 'git log --oneline -1' and 'git log --all --grep=07dd982' on the VPS, and grep live logs for 'v8_candidate_2026-09-10_v8p0' or 'gate=CURVE_ACTIVE' to confirm the claimed deploy.
- **Confidence**: high

## [WARN] F5: funnel.v8 CRITICAL incident vs claims of system health
- **Claim**: (Implicit in both claims) System changes (TS-BATCH complete, V8_PAPER_ENTRY_GATE_SWITCH live-verified) presented as healthy, validated deployments with no mention of any active critical incident.
- **Observed**: EV005 shows funnel.v8 is CRITICAL and FIRING with consecutive_failures: 5035, and claims.batch.v8_readiness is WARN/FIRING with consecutive_failures: 841 and PARTIAL status on 4/7 readiness items -- both directly relevant to 'v8' functionality that the claims describe as recently changed/deployed.
- **Expected**: If V8_PAPER_ENTRY_GATE_SWITCH and TS-BATCH work were fully healthy and 'LIVE_VERIFIED'/'COMPLETE', one would expect no unresolved CRITICAL/WARN incidents specifically tied to v8 funnel/readiness at the time of this audit.
- **Evidence**: EV005
- **Impact**: There is an active, long-running CRITICAL incident in the v8 funnel pipeline and a WARN incident in v8_readiness batch checks that the claims do not acknowledge or reconcile, suggesting the claimed 'LIVE_VERIFIED' status may not reflect current operational health.
- **Next step**: Investigate funnel.v8 incident details (mint AFf278av4oRQicFnpeGHFvcXqjmfNac3XaVyVZAhpump, event bd6003380054eb57) and the 4 PARTIAL items (N2, N4, N6, N7) in claims.batch.v8_readiness to determine if they relate to the V8-P0 gate change.
- **Confidence**: medium

## [INFO] F1: receipts_tail artifact content
- **Claim**: TS-BATCH status: COMPLETE. Result: E0's current trail/TP configuration is not beaten by anything tested... 21 new tests, full suite green (635 local).
- **Observed**: EV007 shows only artifact metadata (mtime 1789171081.94, length 258315 bytes) with no content visible; EV005 test_drift.collection.tests_memecoin reports clean collection (304 tests) for memecoin, and research shows 635 tests collecting cleanly with no errors reported.
- **Expected**: The claim asserts a specific analytical result (TS-BATCH COMPLETE, no proposal drafted, 635 local tests green) that would need to be verified against actual test run output or receipts content.
- **Evidence**: EV007, EV005
- **Impact**: The narrative content of the TS-BATCH analysis cannot be independently confirmed from ground-truth evidence; the test collection counts are consistent but do not prove the specific claimed results (e.g., '0 qualifying cells', 'full suite green').
- **Next step**: Inspect research/thr_ts_grid_results.json and the actual receipts_tail content (not just metadata) to confirm the TS3/TS4 results match the claim.
- **Confidence**: medium

## [INFO] F4: V8_PAPER live-admission not yet confirmed
- **Claim**: Not yet done: no fresh V8-P0-gated paper trades have landed yet to confirm the new gate is actually admitting positions at the expected rate live (only the startup log line is verified so far).
- **Observed**: Ground truth has no direct evidence of v8_paper.py behavior, position admission rates, or the startup log line itself; EV002 lists memecoin/data/memecoin_positions.json and memecoin/data/memecoin_signals.json as modified tracked files, which is consistent with ongoing trading activity but does not confirm gate-specific behavior.
- **Expected**: The claim itself acknowledges this is unverified/live-observation-pending, which aligns with the absence of corroborating evidence in the ground-truth bundle.
- **Evidence**: EV002
- **Impact**: This is a self-acknowledged gap in the claim itself; no contradiction with ground truth, but also no independent confirmation exists.
- **Next step**: Query memecoin_positions.json / memecoin_signals.json for entries tagged with config_tag v8_candidate_2026-09-10_v8p0 to verify live admission behavior.
- **Confidence**: medium

## [INFO] F6: V8_CONFIG_TAG and gate parameters
- **Claim**: V8_CONFIG_TAG bumped to v8_candidate_2026-09-10_v8p0; V8_PROGRESS_MAX constant removed entirely.
- **Observed**: No evidence item inspects memecoin source code, config constants, or config_tag values currently in effect on the deployed HEAD.
- **Expected**: A code-level claim about constant removal and config tag values would require source inspection to confirm, which is outside the scope of the evidence gathered (systemd, cron, watchdog, git status only).
- **Evidence**: EV002
- **Impact**: Ground truth cannot confirm or deny this specific code-level claim; it is neither corroborated nor contradicted.
- **Next step**: grep for V8_PROGRESS_MAX and V8_CONFIG_TAG in the deployed memecoin/v8_paper.py to confirm the described code change is present.
- **Confidence**: low
