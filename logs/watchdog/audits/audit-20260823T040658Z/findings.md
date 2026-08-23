# Layer 2 Audit: audit-20260823T040658Z
Generated: 2026-08-23T04:06:56Z
Evidence SHA-256: `3e639d6df3c4468e8bcbe4c6a6a509b99635a0958a563ebc55a55111d716a775`
Status: ok

## [WARN] F1: v8_readiness / SELECTION_DATA_READY
- **Claim**: SELECTION_DATA_READY = FALSE (real blocker: candidate-specific path coverage is 0% for every candidate -- path_file entries are recorded but the files don't exist on disk; execution-proxy collector has 0 observations so far)
- **Observed**: EV005 shows the claims.batch.v8_readiness incident FIRING (WARN, 284 consecutive failures) with 3/7 items GREEN and 4/7 (N2, N4, N6, N7) PARTIAL; the ground-truth summary explicitly notes no detail is given on what is missing for N2/N4/N6/N7.
- **Expected**: The claim asserts a specific, detailed root cause (0% path coverage due to missing files on disk, 0 execution-proxy observations) for SELECTION_DATA_READY=FALSE, which would presumably correspond to some/all of the PARTIAL items in the watchdog's v8_readiness batch check.
- **Evidence**: EV005
- **Impact**: The claim provides a plausible and consistent explanation for the ongoing WARN incident, but this cannot be independently verified from watchdog evidence alone, so the true root cause of the PARTIAL items remains unconfirmed from ground truth.
- **Next step**: Run the v8_readiness batch-verify CLI directly and inspect per-item (N2, N4, N6, N7) diagnostic output to confirm it matches the claimed path-coverage and execution-proxy-observation counts.
- **Confidence**: medium

## [WARN] F3: Git SHA / deployment state
- **Claim**: Final git SHA: f2d7c75, pushed to origin/main, pulled and verified on the VPS.
- **Observed**: EV002 shows deployed HEAD SHA is 2c2ab8c6bfc97c899e427c26248bd41abb687a9a, which does not match the short SHA f2d7c75 claimed in the receipt. No evidence bundle item shows f2d7c75 anywhere in the git log or ref history.
- **Expected**: If the claim is accurate, the deployed HEAD SHA should correspond to (or be a descendant/ancestor consistent with) f2d7c75.
- **Evidence**: EV002, EV005
- **Impact**: Cannot confirm from evidence that the described code (63 new/updated tests, corrected SELECTION_DATA_READY logic, etc.) is actually the code currently running in production; there is a discrepancy between the claimed final SHA and the observed deployed SHA.
- **Next step**: Run `git log --oneline | grep f2d7c75` and `git merge-base --is-ancestor f2d7c75 HEAD` on the deployed host to confirm whether f2d7c75 is present in the current HEAD's history.
- **Confidence**: medium

## [WARN] F4: Working tree cleanliness vs claimed 'frozen registries... verified unchanged via git status'
- **Claim**: frozen registries and holdout lock verified unchanged via `git status` (only the two report/state modules + two new supporting modules touched)
- **Observed**: EV002 shows the working tree is NOT clean: modifications to docs/RECEIPTS.md, docs/V8_INPUTS.md, several logs/data JSON/CSV files, plus numerous untracked files (new logs, research/data/, research/spool/*.jsonl, two ad-hoc python scripts, several memecoin/data/*.json files) -- a much broader footprint than 'two report/state modules + two new supporting modules'.
- **Expected**: The claim implies a narrow, well-scoped git status showing only 4 touched modules at the time of the described self-audit.
- **Evidence**: EV002
- **Impact**: The current dirty working tree is broader than what the claim describes, though this may reflect normal runtime artifact generation after the claimed audit rather than contradicting it; still, it means the present git status cannot be used to corroborate the claim's narrow-scope assertion.
- **Next step**: Run `git diff --stat` and `git status --porcelain` on the deployed host and compare the file list against the four modules named in the claim to see if the extra changes are unrelated runtime artifacts.
- **Confidence**: medium

## [INFO] F2: v8_readiness incident duration vs claim narrative
- **Claim**: Final state: ENGINE_READY = TRUE ... FORWARD_DATA_PIPELINE_HEALTHY = UNKNOWN ... SELECTION_DATA_READY = FALSE
- **Observed**: EV005 shows claims.batch.v8_readiness has been firing continuously since 1786585581 with 284 consecutive failures, last seen at 1787457791.97 (the most recent slow watchdog run) -- i.e., still actively firing at evidence-capture time.
- **Expected**: The claim describes this as a settled, understood, and monitored state ('collector and readiness monitor are left running'), consistent with an ongoing but expected/explained WARN condition rather than an unexpected failure.
- **Evidence**: EV005
- **Impact**: The long-running WARN incident is consistent with the claim's narrative of an known, accepted incomplete state, but ground truth cannot confirm the claim's specific interpretation (low natural yield vs. an actual defect) beyond what the watchdog reports as PARTIAL/GREEN.
- **Next step**: Check incident first-seen timestamp (1786585581) against the git SHA / deploy history for the claimed fix (f2d7c75) to confirm the incident predates or postdates the claimed engineering work.
- **Confidence**: medium

## [INFO] F5: Test suite counts
- **Claim**: Full suite on the VPS's real environment: 872 passed, zero regressions; 63 new/updated tests across four named test files.
- **Observed**: EV005 shows four/five distinct test suites collecting cleanly (tests_memecoin 304, tests_research 484, tests_watchdog 103, tests_layer2 39, tests_quant-bot 56) -- these are collection counts, not pass/fail run counts, and none is explicitly the 872-test full suite referenced in the claim, nor do the named test files (test_v8_candidate_path_coverage.py etc.) appear in the evidence.
- **Expected**: If accurate, a full-suite run of 872 passed tests should be reconcilable with, or at least not contradicted by, the collection counts reported by watchdog's test_drift checks.
- **Evidence**: EV005
- **Impact**: Ground truth cannot confirm or deny the specific 872-passed claim; the watchdog only reports collection health (no errors), not a full pass/fail run total, so this claim is unverifiable from current evidence.
- **Next step**: Run the full test suite (e.g. `pytest --collect-only -q` then `pytest -q`) on the VPS and compare the total pass count to the claimed 872.
- **Confidence**: low

## [INFO] F6: Pipeline health components (progress_at_signal_flow, venue_state_at_signal_flow, live_pp_paths_flow, path_integrity_quality, execution_proxy_flow)
- **Claim**: progress_at_signal_flow=HEALTHY (135/135, 100%), venue_state_at_signal_flow=HEALTHY (135/135, 100%), live_pp_paths_flow=UNKNOWN (0 recent rows), path_integrity_quality=UNKNOWN (0 recent rows), execution_proxy_flow=UNKNOWN (0 recent observations)
- **Observed**: No EV item in the ground-truth summary reports these five specific named pipeline-health components or their HEALTHY/UNKNOWN statuses; the closest related evidence is the v8_readiness batch check (3/7 GREEN, 4/7 PARTIAL) and pipeline.research_queue_lag / pipeline.research_spool checks, which are differently named and scoped.
- **Expected**: If accurate, these five component statuses should be discoverable in some watchdog check or receipt content within the evidence bundle.
- **Evidence**: EV005
- **Impact**: This detailed claim about component-level health cannot be corroborated or refuted from the audited evidence; it may correspond to a different/newer check not present in this audit run.
- **Next step**: Search watchdog check registry/config for the check name(s) 'progress_at_signal_flow', 'live_pp_paths_flow', 'execution_proxy_flow' to confirm they exist and inspect their latest run output.
- **Confidence**: low

## [INFO] F7: Receipts file timing consistency
- **Claim**: (implicit) RECEIPTS.md documents this v8 work as the latest entry.
- **Observed**: EV007 shows the receipts-adjacent artifact has mtime 1787444102.8 and length 221177 bytes, consistent in timing with the v8_vs_v7_daily job receipt (started_at 1787443081) [EV005, EV007]; no content of the receipts file itself is shown, so the claim text cannot be matched against a timestamped, attributable log entry.
- **Expected**: The claim's detailed narrative (63 tests, final SHA f2d7c75, etc.) should be traceable to a specific RECEIPTS.md entry with a timestamp consistent with the surrounding evidence.
- **Evidence**: EV007, EV005
- **Impact**: Cannot independently confirm the claim's timestamp or authorship placement within RECEIPTS.md, only that a receipts-like artifact of plausible size/mtime exists.
- **Next step**: Open docs/RECEIPTS.md directly and locate the entry containing 'f2d7c75' to confirm its position and timestamp align with the mtime evidence.
- **Confidence**: low

## [INFO] F8: Overall agreement on git dirtiness / untracked files
- **Claim**: (implicit) only the two report/state modules + two new supporting modules touched, i.e. a clean, minimal diff.
- **Observed**: EV002 confirms two ad-hoc Python scripts (p15_1_2_audit.py, scratch_incident_check.py) are present as untracked files, which is plausibly consistent with 'two new supporting modules,' though the naming doesn't clearly match 'report/state modules'.
- **Expected**: The claim's described diff footprint should roughly match the untracked/modified file list.
- **Evidence**: EV002
- **Impact**: Partial agreement exists (two ad-hoc scripts present) but full reconciliation of the claimed 'four modules touched' against the actual broader dirty tree (docs, logs, data JSON/CSV, memecoin data) is not possible from evidence alone.
- **Next step**: Run `git diff --stat HEAD` and map each changed/untracked file to the four modules named in the claim to identify any unexplained extras.
- **Confidence**: low
