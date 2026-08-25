# Layer 2 Audit: audit-20260825T040557Z
Generated: 2026-08-25T04:05:55Z
Evidence SHA-256: `328e0f7b7234c9df593b39897b75b4cb807fdb99ac30a07c562a3e25af7b1ffe`
Status: ok

## [WARN] F2: SELECTION_DATA_READY / v8_readiness batch
- **Claim**: SELECTION_DATA_READY = FALSE (real blocker: candidate-specific path coverage is 0% for every candidate ... execution-proxy collector has 0 observations so far)
- **Observed**: The `claims.batch.v8_readiness` incident is FIRING with consecutive_failures: 340 and the corresponding check reports 4/7 items PARTIAL (['N2','N4','N6','N7']), explicitly noting batch-verify would exit 0 despite partial items (EV005).
- **Expected**: The claim's narrative of 0% path coverage across all candidates is broadly consistent with a persistently unhealthy/partial v8_readiness state, but the ground truth shows this as an active, long-running FIRING incident (340 consecutive failures since 1786585581) rather than a single fresh diagnostic result from 'this run'.
- **Evidence**: EV005
- **Impact**: The claimed readiness blocker appears to be a long-standing, ongoing condition rather than a newly-discovered one-off finding, which changes its urgency and suggests the underlying data pipeline issue has not been resolved despite documentation describing specific fixes.
- **Next step**: Query the watchdog incident history for `claims.batch.v8_readiness` to determine whether consecutive_failures count has changed since this claim's stated run, indicating whether the described fix actually altered the readiness state.
- **Confidence**: medium

## [WARN] F6: Working tree cleanliness vs claimed frozen registries
- **Claim**: frozen registries and holdout lock verified unchanged via `git status` (only the two report/state modules + two new supporting modules touched)
- **Observed**: EV002 shows the working tree is not clean: 5 modified tracked files and 19 untracked paths, including two Python scripts (`p15_1_2_audit.py`, `scratch_incident_check.py`) and several log/data directories -- more than the 'two report/state modules + two new supporting modules' the claim describes.
- **Expected**: The claim implies a narrowly-scoped, verified-clean diff limited to four specific modules.
- **Evidence**: EV002
- **Impact**: The actual working tree shows more modified/untracked files than the claim's narrow scope describes, which could mean either additional unrelated runtime artifacts are present (benign) or the claimed 'git status' check was run at a different time/state than this audit snapshot.
- **Next step**: Run `git status --porcelain` on the VPS now and diff the file list against both this audit's EV002 listing and the four modules named in the claim to reconcile scope.
- **Confidence**: medium

## [INFO] F1: Git SHA mismatch
- **Claim**: Final git SHA: `f2d7c75`, pushed to `origin/main`, pulled and verified on the VPS.
- **Observed**: Deployed HEAD SHA recorded in EV002 and matched across watchdog runs and job receipts is `1323fe1ecf008a6c123535a8026a3de1d457dbaa`, not `f2d7c75`.
- **Expected**: The claim implies the VPS is running/verified at commit `f2d7c75`.
- **Evidence**: EV002, EV005
- **Impact**: If the claimed SHA is not actually deployed, the described V8 readiness fixes and test results may not reflect the code currently running in production.
- **Next step**: Run `git log -1 --format=%H` on the VPS and compare to both `f2d7c75` and the watchdog-recorded SHA `1323fe1ec...` to determine which commit is actually live.
- **Confidence**: medium

## [INFO] F3: Pipeline health components (UNKNOWN state)
- **Claim**: FORWARD_DATA_PIPELINE_HEALTHY = UNKNOWN (3 of 5 components UNKNOWN from low natural yield; the 2 checkable components are both HEALTHY)
- **Observed**: Ground truth (EV005) does not directly enumerate these five named pipeline components (`progress_at_signal_flow`, `venue_state_at_signal_flow`, `live_pp_paths_flow`, `path_integrity_quality`, `execution_proxy_flow`); it does show `research.path_collection` FIRING with WARN for low path yield (46.2%, below 50% floor) and PumpPortal message budget exceeded, which is a related but not identical signal.
- **Expected**: The claim asserts a specific 5-component health breakdown with 2 HEALTHY and 3 UNKNOWN, attributing UNKNOWN to 'low natural yield, not corruption.'
- **Evidence**: EV005
- **Impact**: Cannot independently confirm the specific component-level health claims from watchdog evidence alone; the related `research.path_collection` FIRING incident suggests real yield problems exist, which is at least directionally consistent with the claim but not a direct verification.
- **Next step**: Inspect the raw output/logs of the pipeline health self-audit script referenced in the claim to cross-check its five component statuses against the `research.path_collection` check's underlying data.
- **Confidence**: low

## [INFO] F4: research.path_collection incident vs claim narrative
- **Claim**: all three UNKNOWNs are low-natural-yield, not corruption
- **Observed**: EV005 shows `research.path_collection` FIRING for 34 consecutive failures with WARN status: path yield 46.2% (below 50% floor) and PumpPortal daily message budget exceeded (104432/100000, 46 tokens dropped).
- **Expected**: The claim frames low yield as an expected/benign condition ('not corruption'), while the ground truth shows this same underlying yield issue is tracked as an actively FIRING incident with a WARN-level check breaching a defined floor threshold.
- **Evidence**: EV005
- **Impact**: Describing a breached, alerting threshold as merely 'low natural yield' downplays that this condition is actively flagged as a warning/incident by the monitoring system, not a passively accepted state.
- **Next step**: Review the `path_yield_floor` threshold configuration and incident history for `research.path_collection` to confirm whether 46.2% yield is within expected operating bounds or represents a regression.
- **Confidence**: medium

## [INFO] F5: Test suite counts
- **Claim**: Full suite on the VPS's real environment: 872 passed, zero regressions.
- **Observed**: Ground truth reports test collection counts by suite from the latest slow watchdog run: tests_memecoin 304, tests_research 484, tests_watchdog 103, tests_layer2 39, tests_quant-bot 56 (sum = 986), all collecting cleanly with no mock/patch drift (EV005). This is a collection count, not a pass/fail execution count, and does not match or refute the claimed 872 passed figure.
- **Expected**: The claim asserts a total of 872 tests passed with zero regressions from an actual test run.
- **Evidence**: EV005
- **Impact**: Cannot verify the claimed pass count or 'zero regressions' assertion from watchdog evidence, which only confirms clean test collection, not execution/pass results.
- **Next step**: Run the full test suite on the VPS directly (e.g. `pytest -q`) and compare actual pass/fail/skip counts to the claimed 872 passed figure.
- **Confidence**: medium

## [INFO] F7: rc_closure commit vs claimed SHA
- **Claim**: Final git SHA: `f2d7c75`, pushed to `origin/main`, pulled and verified on the VPS.
- **Observed**: EV005's `claims.batch.rc_closure` reports 3/3 GREEN at commit `db32f53`, which matches neither the deployed HEAD SHA (`1323fe1ec...`) nor the claimed final SHA (`f2d7c75`) in this document.
- **Expected**: If `f2d7c75` were truly the verified, deployed commit, related claims/batch verification evidence would be expected to reference it or the same lineage.
- **Evidence**: EV005, EV002
- **Impact**: Three different commit references (`1323fe1ec`, `db32f53`, `f2d7c75`) appear across evidence and claims with no reconciling explanation, making it hard to establish a single source of truth for what code is actually running and verified.
- **Next step**: Retrieve `git log --oneline -5` and correlate commit timestamps with the rc_closure batch verification timestamp and this document's stated push time to build a coherent commit timeline.
- **Confidence**: medium
