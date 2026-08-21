# Layer 2 Audit: audit-20260821T040716Z
Generated: 2026-08-21T04:07:15Z
Evidence SHA-256: `35388439dae18754b7b5bcae042313303fb88ad07778e425a052c17a20a6bce3`
Status: ok

## [WARN] F1: v8_readiness batch check vs claimed SELECTION_DATA_READY=FALSE
- **Claim**: SELECTION_DATA_READY = FALSE (real blocker: candidate-specific path coverage is 0% for every candidate -- path_file entries are recorded but the files don't exist on disk; execution-proxy collector has 0 observations so far)
- **Observed**: Ground truth shows a FIRING WARN incident claims.batch.v8_readiness, consecutive_failures: 229, with 4/7 items PARTIAL (N2, N4, N6, N7), unrecovered as of the latest slow run (EV005). The specific reasons for PARTIAL status on N2/N4/N6/N7 are not detailed in evidence.
- **Expected**: The claim's narrative (0% path coverage, 0 exec-proxy observations across all candidates) is directionally consistent with a persistently failing/partial readiness state, but the evidence bundle cannot confirm this specific root cause or map N2/N4/N6/N7 to the candidates named in the claim (BASELINE-0, V8-P0, V8-P1, V8-P3).
- **Evidence**: EV005
- **Impact**: The claim's explanation for why SELECTION_DATA_READY is FALSE cannot be independently verified from watchdog evidence alone, so root-cause attribution in the doc is unconfirmed.
- **Next step**: Inspect the v8_readiness batch check's raw item-level output (N2, N4, N6, N7 definitions) and cross-reference against the candidate table in RECEIPTS.md to confirm the mapping.
- **Confidence**: medium

## [INFO] F2: Consecutive failure duration vs claimed 'real live results, this run'
- **Claim**: Real live results (VPS, this run)... Final git SHA: f2d7c75, pushed to origin/main, pulled and verified on the VPS.
- **Observed**: Deployed HEAD SHA per EV002/EV005 is e6455f5..., not f2d7c75. The claims.batch.v8_readiness incident has been firing continuously for 229 consecutive checks (first_seen 1786585581), well before the described fix would have landed.
- **Expected**: If f2d7c75 were the currently deployed and verified commit as claimed, deployed HEAD SHA should match f2d7c75, and one might expect the long-firing incident to show recent state change reflecting the fix.
- **Evidence**: EV002, EV005
- **Impact**: The claimed commit does not match the deployed HEAD SHA, raising doubt about whether the described fix/verification is actually live in the running system.
- **Next step**: Run `git log --oneline -1` and `git rev-parse HEAD` on the VPS deployment path and diff against f2d7c75 to confirm deployment status.
- **Confidence**: high

## [INFO] F3: Pipeline health components (HEALTHY/UNKNOWN breakdown)
- **Claim**: Pipeline components (24h window): progress_at_signal_flow=HEALTHY (135/135, 100%), venue_state_at_signal_flow=HEALTHY (135/135, 100%), live_pp_paths_flow=UNKNOWN, path_integrity_quality=UNKNOWN, execution_proxy_flow=UNKNOWN — all three UNKNOWNs are low-natural-yield, not corruption.
- **Observed**: Ground truth has no direct evidence of these five named pipeline components; the closest related check is claims.batch.v8_readiness (WARN, PARTIAL) and pipeline.research_queue_lag / pipeline.research_spool (both OK) from EV005/EV006, which are different checks entirely.
- **Expected**: The claim describes a distinct component-level health model not present in the audited watchdog check set; ground truth cannot confirm or deny these specific component statuses.
- **Evidence**: EV005, EV006
- **Impact**: Cannot verify whether the fine-grained pipeline component health matrix described in the claim is actually being computed or is accurate, since it isn't part of the audited evidence.
- **Next step**: Locate the watchdog or report module that emits progress_at_signal_flow/venue_state_at_signal_flow/live_pp_paths_flow/path_integrity_quality/execution_proxy_flow and pull its most recent raw output for direct comparison.
- **Confidence**: low

## [INFO] F4: Test suite counts (872 passed) vs watchdog test collection counts
- **Claim**: Full suite on the VPS's real environment: 872 passed, zero regressions.
- **Observed**: Ground truth (EV005) reports test collection counts of 304 (memecoin), 466 (research), 103 (watchdog), 39 (layer2), 56 (top-level quant-bot) — summing to 968, not matching 872 — though collection counts and passed-test counts are not directly comparable metrics.
- **Expected**: If the claim's 872-passed figure represents the same overall suite as the watchdog's collected-test totals, some reconciliation would be expected; evidence does not confirm this relationship either way.
- **Evidence**: EV005
- **Impact**: Cannot cross-validate the claimed test pass count against independently observed test collection metrics, since they may be measuring different scopes (collected vs. passed, or different suite subsets).
- **Next step**: Run the full test suite (e.g., `pytest -q` at repo root) and compare the actual passed/failed/collected counts to both the claim (872 passed) and the watchdog's per-module collection counts.
- **Confidence**: low

## [INFO] F5: Working tree cleanliness vs claim of 'frozen registries and holdout lock verified unchanged via git status'
- **Claim**: frozen registries and holdout lock verified unchanged via git status (only the two report/state modules + two new supporting modules touched)
- **Observed**: Ground truth confirms the working tree is NOT clean: EV002 lists 5 modified tracked files and 18 untracked paths, including loose scripts (p15_1_2_audit.py, scratch_incident_check.py) not mentioned in the claim's list of touched modules.
- **Expected**: The claim implies a narrowly scoped, verified-clean git status showing only 4 specific modules touched; ground truth shows a broader and different set of changes (docs, logs, JSON data files, and untracked scripts) not accounted for by the claim.
- **Evidence**: EV002
- **Impact**: The claim's git-status verification appears to describe a different or narrower diff than what is actually present in the working tree, which could mean the audit's git status check was run at a different time/scope, or the claim is incomplete.
- **Next step**: Run `git status --porcelain` and `git diff --stat` on the deployment directory now and compare directly against the file list in EV002 and the claim's described 4-module diff.
- **Confidence**: medium

## [INFO] F6: rc_closure commit vs claimed final SHA
- **Claim**: Final git SHA: f2d7c75, pushed to origin/main, pulled and verified on the VPS.
- **Observed**: claims.batch.rc_closure reports commit db32f53 (EV005), which differs from both the deployed HEAD SHA (e6455f5..., EV002/EV005) and the claim's asserted final SHA f2d7c75. Evidence does not explain the relationship among these three distinct commit references.
- **Expected**: If f2d7c75 were truly the verified-on-VPS commit, one would expect it to align with either the deployed HEAD SHA or the rc_closure batch's referenced commit; instead there are three different SHAs in play.
- **Evidence**: EV002, EV005
- **Impact**: Multiple inconsistent commit references across evidence and claims make it difficult to confirm which code version is actually running in production.
- **Next step**: Reconcile db32f53, e6455f5, and f2d7c75 by checking `git log --all --oneline | grep` for each hash and their ancestry/branch relationships on the VPS.
- **Confidence**: medium
