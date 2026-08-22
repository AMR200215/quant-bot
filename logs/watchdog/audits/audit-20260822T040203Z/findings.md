# Layer 2 Audit: audit-20260822T040203Z
Generated: 2026-08-22T04:02:01Z
Evidence SHA-256: `635200d694601b985123f1da2da87a5cdd1721030eb0afd54b642c74cf49cb15`
Status: ok

## [WARN] F1: v8_readiness claim vs watchdog incident
- **Claim**: SELECTION_DATA_READY = FALSE (real blocker: candidate-specific path coverage is 0% for every candidate -- path_file entries are recorded but the files don't exist on disk; execution-proxy collector has 0 observations so far)
- **Observed**: Ground truth confirms an active FIRING incident claims.batch.v8_readiness (3/7 GREEN, 4/7 PARTIAL items N2/N4/N6/N7, 0 FAIL, consecutive_failures=255) but has no evidence defining what N2/N4/N6/N7 substantively represent, so it cannot confirm or deny that the claim's stated root cause (0% path coverage, 0 exec-proxy observations) is the actual cause of the PARTIAL/WARN state.
- **Expected**: The claim asserts a specific, detailed root cause for readiness not being achieved (path files missing on disk, zero proxy observations) that would presumably correspond to the WARN/PARTIAL items in the watchdog check.
- **Evidence**: EV005, EV007
- **Impact**: The documented root cause cannot be independently verified against the watchdog's own check internals, so reliance on the claim alone for incident triage risks missing a discrepancy between what the doc says and what the monitoring actually measures.
- **Next step**: Inspect the v8_readiness check's underlying claim definitions/source (e.g., the batch-verify CLI output referenced in EV005) to confirm N2/N4/N6/N7 map to path-coverage and exec-proxy-observation failures as described in the claim.
- **Confidence**: medium

## [INFO] F2: Git SHA vs claim final SHA
- **Claim**: Final git SHA: `f2d7c75`, pushed to `origin/main`, pulled and verified on the VPS.
- **Observed**: Deployed HEAD SHA per ground truth is `c931c9596655ccc626066d4f1d8cef0f47a7f22c`, which does not match the short SHA `f2d7c75` cited in the claim.
- **Expected**: If the claim's stated deployment/verification step actually occurred as the most recent action, the deployed HEAD should reflect (or be a descendant matching the abbreviation of) `f2d7c75`.
- **Evidence**: EV002
- **Impact**: Cannot confirm whether the described work (SELECTION_DATA_READY fix, 63 new tests, etc.) is actually reflected in the currently running/deployed commit, or whether it is a prior/later commit than what's live.
- **Next step**: Run `git log --oneline -1 f2d7c75` and `git merge-base --is-ancestor f2d7c75 c931c9596655ccc` (or compare commit dates) to determine the relationship between the deployed HEAD and the claimed final SHA.
- **Confidence**: high

## [INFO] F3: Dirty working tree vs claim of frozen/verified state
- **Claim**: frozen registries and holdout lock verified unchanged via `git status` (only the two report/state modules + two new supporting modules touched)
- **Observed**: Ground truth shows the working tree is not clean: 5-6 modified tracked files plus at least 19 untracked files/directories, including two ad hoc scripts and several new log/data paths -- a broader footprint than 'two report/state modules + two new supporting modules'.
- **Expected**: If the claim's git status check at the time of that work showed only 4 touched modules, the current dirty-tree state (more files, untracked scripts) implies either additional changes happened afterward or the claim's git status snapshot is stale relative to the current audit.
- **Evidence**: EV002
- **Impact**: The current uncommitted changes are broader than what the documentation describes as having been touched, so the doc's git-status-based verification does not reflect the present state of the tree and shouldn't be relied upon as current.
- **Next step**: Run `git status --porcelain` now and diff the file list against the four modules named in the claim to identify what changed since that snapshot.
- **Confidence**: medium

## [INFO] F4: Pipeline health components (HEALTHY/UNKNOWN) vs watchdog check
- **Claim**: Pipeline components (24h window): progress_at_signal_flow=HEALTHY, venue_state_at_signal_flow=HEALTHY, live_pp_paths_flow=UNKNOWN, path_integrity_quality=UNKNOWN, execution_proxy_flow=UNKNOWN — all three UNKNOWNs are low-natural-yield, not corruption
- **Observed**: Ground truth's EV005 v8_readiness check reports counts (3/7 GREEN, 4/7 PARTIAL, 0 FAIL) but does not name or break down individual pipeline components like progress_at_signal_flow or execution_proxy_flow, so this specific component-level breakdown cannot be cross-checked against watchdog evidence.
- **Expected**: If this pipeline-health breakdown corresponds to the watchdog's v8_readiness PARTIAL items, the evidence should show a mapping between named components and N2/N4/N6/N7, which is absent.
- **Evidence**: EV005
- **Impact**: Cannot confirm the claimed component-level health breakdown is what the automated monitoring actually observed; it may be from an independent manual/self-audit run not captured in watchdog evidence.
- **Next step**: Locate and inspect the raw v8_readiness batch-verify CLI output referenced in EV005 to check whether it names these five pipeline components and their HEALTHY/UNKNOWN states.
- **Confidence**: low

## [INFO] F5: Test suite pass count claim vs ground-truth test evidence
- **Claim**: Full suite on the VPS's real environment: 872 passed, zero regressions.
- **Observed**: Ground truth only has collection-only test counts from EV005 (memecoin 304, research 466, watchdog 103, layer2 39, quant-bot 56) which confirm pytest could collect these suites, not that tests pass; no evidence of an 872-test full run or pass/fail results exists in the ground truth.
- **Expected**: The claim asserts an actual full test run with a pass count and zero regressions, which is a stronger and different assertion than collection-only success.
- **Evidence**: EV005
- **Impact**: The claim of full suite success cannot be corroborated by available monitoring evidence; test collection success does not imply the tests pass, so this claim remains unverified.
- **Next step**: Run `pytest` directly on the VPS (or check CI logs) and compare actual pass/fail counts to the claimed 872 passed, zero regressions.
- **Confidence**: medium

## [INFO] F6: receipts_tail file correspondence
- **Claim**: (implicit) RECEIPTS.md content reflects the v8 readiness work described, written as part of the v8_vs_v7_daily job
- **Observed**: Ground truth (EV007) confirms docs/RECEIPTS.md mtime matches the v8_vs_v7_daily job's finished_at timestamp, consistent with that cron job being the last writer -- this corroborates provenance of the file but not the accuracy of its content.
- **Expected**: The claim's content should be attributable to a specific job run, which the mtime match does support at the provenance level.
- **Evidence**: EV007, EV005
- **Impact**: Provenance (who/what wrote the file) is corroborated, but this says nothing about whether the substantive engineering claims within the file are accurate.
- **Next step**: Cross-reference the v8_vs_v7_daily job's own logs/output against the RECEIPTS.md content to confirm the numbers (e.g., 278 venue_qualified_n for V8-P0) were generated by that specific run.
- **Confidence**: high
