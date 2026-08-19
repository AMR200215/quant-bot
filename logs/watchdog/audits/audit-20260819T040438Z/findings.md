# Layer 2 Audit: audit-20260819T040438Z
Generated: 2026-08-19T04:04:36Z
Evidence SHA-256: `486131bdba1a6dc1280e735f12da782368b150fc0365dab2fda96c2b2cf51adc`
Status: ok

## [WARN] F2: Git SHA reconciliation
- **Claim**: Final git SHA: f2d7c75, pushed to origin/main, pulled and verified on the VPS.
- **Observed**: Ground truth shows deployed HEAD SHA as c106ee71028d57ef9c773374936b34e0038d953b, corroborated by both watchdog runs. Job receipts reference yet other SHAs (4e020b69f..., 8d0c3b8ea...), and claims.batch.rc_closure reports commit db32f53. None of these match f2d7c75, and the evidence provides no timestamp linkage among any of the four distinct SHA values.
- **Expected**: The claim implies f2d7c75 is the current, verified-on-VPS commit, which would be expected to match the deployed HEAD SHA.
- **Evidence**: EV002, EV005
- **Impact**: If f2d7c75 is truly the latest pushed/pulled commit but the deployed HEAD is c106ee7, the running service may not reflect the claimed engineering changes (including the 63 new tests and readiness logic fixes).
- **Next step**: Run `git log --oneline -1` and `git rev-parse HEAD` on the VPS deployment directory and compare against f2d7c75, c106ee7, db32f53, and the two job-receipt SHAs.
- **Confidence**: high

## [INFO] F1: receipts_tail content vs EV007 evidence
- **Claim**: Detailed narrative of pipeline health components, SELECTION_DATA_READY logic, live results table, test counts (872 passed), and final git SHA f2d7c75, all presented as a completed engineering report.
- **Observed**: EV007 as available to the ground-truth audit provided only an mtime (1787103520.26 ≈ 2026-08-19T02:38:40Z) and byte length (220811) for the receipts file — no content, job names, or pass/fail status. The detailed narrative now shown is asserted documentation text, not something the original evidence could substantiate.
- **Expected**: The claim implies the receipts file contains this exact detailed, verified content and that ground truth could confirm it.
- **Evidence**: EV007
- **Impact**: The claimed content cannot be independently verified from raw evidence; any downstream decision relying on this narrative's accuracy is unverified.
- **Next step**: Read the actual content of the receipts file (e.g. tail -c 5000 on the receipts file at the recorded mtime) and diff against the claimed narrative.
- **Confidence**: high

## [INFO] F3: Working tree cleanliness vs claimed push
- **Claim**: Final git SHA... pushed to origin/main, pulled and verified on the VPS.
- **Observed**: The working tree is not clean: 4 modified tracked files and 15 untracked paths (including a new script p15_1_2_audit.py and new research/ directories) were present at audit time (EV002).
- **Expected**: A 'pulled and verified' state after a clean push would typically be expected to leave a clean working tree, absent unrelated local scratch work.
- **Evidence**: EV002
- **Impact**: Untracked/modified files could indicate either unrelated in-progress work or an incomplete/impure deployment of the claimed commit; ambiguous without further inspection.
- **Next step**: Run `git status --porcelain` and `git diff` on the VPS checkout to enumerate exactly which files are modified/untracked and whether any relate to the claimed engineering work.
- **Confidence**: medium

## [INFO] F4: Test suite count claim
- **Claim**: Full suite on the VPS's real environment: 872 passed, zero regressions.
- **Observed**: Ground truth only confirms test-collection drift checks are clean with no drift, for specific suites: memecoin 304, research 465, watchdog 103, layer2 39, quant-bot root 56 tests (EV005). These are collection counts, not a full pass/fail execution report, and their sum (967) does not equal the claimed 872 passed — though these are different measurements (collection vs. execution) and not directly comparable.
- **Expected**: The claim implies an actual full-suite execution result of 872 passing tests, distinct from and not contradicted by, but also not corroborated by, the collection-drift check.
- **Evidence**: EV005
- **Impact**: Cannot independently confirm the claimed test execution outcome; test-collection health alone does not prove all tests pass.
- **Next step**: Run the project's full test command (e.g. pytest -q) on the VPS and compare the pass count/regressions against the claimed 872 passed figure.
- **Confidence**: medium

## [INFO] F5: SELECTION_DATA_READY / ENGINE_READY final state
- **Claim**: ENGINE_READY = TRUE; FORWARD_DATA_PIPELINE_HEALTHY = UNKNOWN (3 of 5 components UNKNOWN); SELECTION_DATA_READY = FALSE due to 0% path coverage across all candidates and zero execution-proxy observations.
- **Observed**: Ground truth's closest corroborating evidence is claims.batch.v8_readiness, WARN, FIRING for 173 consecutive failures, reporting '4/7 item(s) PARTIAL ([N2,N4,N6,N7])' — a different check with different item labels (N2/N4/N6/N7 vs BASELINE-0/V8-P0/V8-P1/V8-P3) and no direct correspondence to the claimed candidate table or component health breakdown.
- **Expected**: If the claimed readiness report were reflected in the watchdog's incident state, one would expect claims.batch.v8_readiness (or an equivalent check) to show consistent status/counts with the claimed FALSE readiness and UNKNOWN pipeline health.
- **Evidence**: EV005
- **Impact**: Cannot confirm from ground-truth evidence whether the detailed candidate-level readiness state described in the claim is the same system/state as the actively firing claims.batch.v8_readiness incident, or a separate/newer report not yet reflected in watchdog checks.
- **Next step**: Inspect the watchdog check definition for claims.batch.v8_readiness and any associated report file to see if it maps to the same BASELINE-0/V8-P0/V8-P1/V8-P3 candidate table referenced in the claim.
- **Confidence**: low

## [INFO] F6: p15_1_2_audit.py untracked file
- **Claim**: (none)
- **Observed**: EV002 lists p15_1_2_audit.py as an untracked new script with no content or purpose described anywhere in the evidence.
- **Expected**: The claims document describes extensive new engineering work (readiness monitor, report modules, tests) — it's plausible but unconfirmed that this untracked script relates to that work.
- **Evidence**: EV002
- **Impact**: An unreviewed, untracked script present on a production VPS is a minor operational hygiene concern regardless of its relation to the claims.
- **Next step**: Run `cat p15_1_2_audit.py` or `git diff --stat` to review its contents and determine if it should be committed, removed, or ignored.
- **Confidence**: medium
