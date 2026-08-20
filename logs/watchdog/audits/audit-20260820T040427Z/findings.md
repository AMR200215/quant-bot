# Layer 2 Audit: audit-20260820T040427Z
Generated: 2026-08-20T04:04:25Z
Evidence SHA-256: `aa605a40fae52f594277f2a31f3d8071a54ece62c7673ebea8ad015e8bf50c20`
Status: ok

## [CRITICAL] F1: git SHA / deployment state
- **Claim**: Final git SHA: `f2d7c75`, pushed to `origin/main`, pulled and verified on the VPS.
- **Observed**: HEAD is `08580ccc...` per EV002, and all watchdog runs and job receipts (EV005) report this same SHA. There is no evidence of `f2d7c75` anywhere in the audit bundle.
- **Expected**: The claim asserts the deployed/verified commit is `f2d7c75`, which should match or be reachable from the observed HEAD if truly pulled and verified.
- **Evidence**: EV002, EV005
- **Impact**: If the claimed commit was never actually deployed, the described 'live results' and 'final state' (SELECTION_DATA_READY, ENGINE_READY, etc.) may not reflect what is actually running on the VPS.
- **Next step**: Run `git log --oneline -1` and `git log --all --oneline | grep f2d7c75` on the VPS to confirm whether f2d7c75 exists in history and matches deployed HEAD.
- **Confidence**: high

## [WARN] F2: claims.batch.v8_readiness incident vs. SELECTION_DATA_READY claim
- **Claim**: SELECTION_DATA_READY = FALSE (real blocker: candidate-specific path coverage is 0% for every candidate ... execution-proxy collector has 0 observations so far)
- **Observed**: The watchdog reports an actively FIRING WARN incident `claims.batch.v8_readiness` with 201 consecutive failures over ~8 days, batch state 4/7 PARTIAL, 3 GREEN, 0 FAIL (EV005). This is broadly consistent with the claim's description of persistent non-readiness, but the ground truth gives no per-candidate path-coverage percentages or execution-proxy observation counts to corroborate the specific 0% and '0 obs' figures cited.
- **Expected**: The claim provides specific numeric detail (0.0% path coverage, 0 execution-proxy observations) that would explain the watchdog's PARTIAL/WARN state, but this detail is not present in or verifiable from the audit evidence.
- **Evidence**: EV005
- **Impact**: The qualitative direction (not yet ready) matches, but the claim's specific diagnostic numbers cannot be independently verified from watchdog evidence alone, so root-cause attribution should not be taken as confirmed.
- **Next step**: Pull the raw `claims.batch.v8_readiness` batch_verify output referenced by EV005 and cross-check the N2/N4/N6/N7 PARTIAL item details against the claimed 0% path coverage / 0 exec-proxy observations.
- **Confidence**: medium

## [INFO] F3: Pipeline health components (HEALTHY/UNKNOWN split)
- **Claim**: progress_at_signal_flow=HEALTHY (135/135), venue_state_at_signal_flow=HEALTHY (135/135), live_pp_paths_flow=UNKNOWN, path_integrity_quality=UNKNOWN, execution_proxy_flow=UNKNOWN — all three UNKNOWNs are low-natural-yield, not corruption.
- **Observed**: Ground truth has no direct evidence bundle covering these five named pipeline components; EV005's closest analog, `pipeline.research_queue_lag` and `pipeline.research_spool`, are OK/no-lag, which is a different check entirely.
- **Expected**: The claim describes a specific FORWARD_DATA_PIPELINE_HEALTHY breakdown that the audit evidence does not contain data to confirm or refute.
- **Evidence**: EV005
- **Impact**: This part of the claim is unverifiable from current evidence; it should be treated as an assertion pending its own audit pass, not as corroborated fact.
- **Next step**: Locate and inspect the raw forward_data_pipeline health report (if logged) to confirm component-level HEALTHY/UNKNOWN statuses match the claim.
- **Confidence**: low

## [INFO] F4: Test suite pass counts
- **Claim**: 63 new/updated tests ... Full suite on the VPS's real environment: 872 passed, zero regressions.
- **Observed**: Ground truth (EV005) confirms only that pytest collection succeeds cleanly for 5 suites (304+465+103+39+56 = 967 tests collected), explicitly noting 'no pass/fail counts are in evidence.'
- **Expected**: The claim asserts a specific pass count (872 passed) from an actual test run, which is a stronger and different claim than mere collection success.
- **Evidence**: EV005
- **Impact**: Cannot corroborate whether tests actually pass at the claimed rate; collection success alone does not validate this claim.
- **Next step**: Run `pytest -q` (or review CI run logs) on the VPS and compare actual pass/fail/skip counts to the claimed 872 passed figure.
- **Confidence**: medium

## [INFO] F5: rc_closure commit mismatch vs. claimed SHA
- **Claim**: Final git SHA: f2d7c75, pushed to origin/main, pulled and verified on the VPS.
- **Observed**: EV005 shows `claims.batch.rc_closure` is tied to commit `db32f53`, which itself does not match deployed HEAD `08580ccc...` (EV002) — a discrepancy already flagged as an open question in the ground truth. The claim now introduces a third distinct SHA (`f2d7c75`), none of which reconcile with each other in the available evidence.
- **Expected**: If the claim's SHA is accurate and current, it should align with or postdate the HEAD and rc_closure SHAs observed in evidence; instead three different commit identifiers appear across sources with no reconciling evidence.
- **Evidence**: EV002, EV005
- **Impact**: Multiple inconsistent commit references across deployed HEAD, batch-verify metadata, and documentation claims raise risk that the audited system state does not correspond to the code version described in the claims.
- **Next step**: Diff `git log` between `08580ccc`, `db32f53`, and `f2d7c75` to determine ancestry/ordering and confirm which commit is actually running.
- **Confidence**: medium

## [INFO] F6: Working tree cleanliness vs. 'pulled and verified' claim
- **Claim**: pushed to origin/main, pulled and verified on the VPS
- **Observed**: EV002 shows the working tree at HEAD is not clean: 6 modified tracked files and 15 untracked paths, including a script `p15_1_2_audit.py` and various research/logs/data artifacts.
- **Expected**: A claim of a clean 'pulled and verified' deployment would typically imply a clean working tree matching the pushed commit, not a dirty tree with modified/untracked files.
- **Evidence**: EV002
- **Impact**: Uncommitted local modifications alongside a claimed clean pull-and-verify suggest either post-pull local edits/artifacts or that the claim's described state predates or diverges from the current on-disk state.
- **Next step**: Run `git status --porcelain` and `git diff` on the VPS to characterize the 6 modified tracked files and determine whether they affect the claimed readiness results.
- **Confidence**: medium
