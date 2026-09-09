# Layer 2 Audit: audit-20260909T081643Z
Generated: 2026-09-09T08:16:41Z
Evidence SHA-256: `7e56be21fc5ab57b4bc68a1a817d89e68c428846a07545b79da913db53d3c9b0`
Status: ok

## [WARN] F2: Test suite count consistency
- **Claim**: Full suite green (580 local, 32+37 module-specific).
- **Observed**: Ground truth (EV005) reports test collection counts of 304 (memecoin), 595 (research), 103 (watchdog), 39 (layer2), 56 (quant-bot root) — none of which is 580, and no breakdown matching '32+37 module-specific' appears in the evidence.
- **Expected**: If the claim's '580 local' figure refers to a specific suite (e.g. memecoin or research), it should be reconcilable with one of the watchdog's reported collection counts; as stated it does not match any of the five reported totals.
- **Evidence**: EV005
- **Impact**: Test count discrepancy raises doubt about whether the claimed test run was against the same codebase/suite the watchdog is currently validating, or whether counts are stale/from a different scope.
- **Next step**: Run the full test suite locally (pytest --collect-only) at current HEAD and compare exact counts per module against both the watchdog's collection numbers and the 580/32/37 figures in the claim.
- **Confidence**: medium

## [WARN] F3: Dirty working tree vs. claimed completed/committed work
- **Claim**: RESOLVED 2026-09-09 — T1 complete. Built compute_interpolated_xval ... research/thr_reconstruct_paths.py ...
- **Observed**: EV002 shows the working tree is not clean, with 25 modified/untracked entries including ad-hoc scripts and data files, and no commit history is provided to confirm the described module changes were committed. HEAD SHA e37d8ea3... also does not match SHAs on recent cron receipts or rc_closure commit db32f53.
- **Expected**: A 'COMPLETE' and 'RESOLVED' claim about new production code (a new module, new tests, a new constant) would typically correspond to a clean, committed working tree state, or at minimum the new files would be visible/tracked in the diff evidence.
- **Evidence**: EV002
- **Impact**: Uncommitted or partially-tracked changes claimed as 'complete' create risk that the described xval gate logic is not actually the code running in production, or could be lost/inconsistent across environments.
- **Next step**: Run git status and git diff --stat against thr_reconstruct_paths.py and related test files to confirm whether the described T1 work is committed, staged, or still untracked.
- **Confidence**: medium

## [INFO] F1: xval pilot / T1 claim vs evidence scope
- **Claim**: T1 status: COMPLETE. Exact reconstruction proven (byte-identical, twice), 2 pre-existing bugs found and fixed (new module only), xval gate redesigned and validated with a real, provenance-backed tolerance.
- **Observed**: No evidence item (EV001-EV007) in the ground-truth summary references compute_interpolated_xval, thr_reconstruct_paths.py, XVAL_TOLERANCE_PCT, the 20-token pilot, or any byte-identical reconstruction test. This entire claim originates solely from EV007 receipts_tail text, which is documentation/assertion, not independently-verified evidence in this audit.
- **Expected**: If T1 were truly complete and verified, corroborating evidence (e.g. passing test counts for the new module, a git commit for thr_reconstruct_paths.py, or a check result) would be expected somewhere in the watchdog/test-collection evidence.
- **Evidence**: EV007
- **Impact**: The claim of T1 completion cannot be independently corroborated from watchdog/test/state evidence; it rests entirely on self-reported documentation text.
- **Next step**: Run the module's test suite directly (e.g. pytest research/thr_reconstruct_paths.py and related xval tests) and diff HEAD against the commit referenced in RECEIPTS.md to confirm the described code exists at the deployed SHA.
- **Confidence**: high

## [INFO] F4: funnel.v8 vs xval claim scope
- **Claim**: (implicit) T1 work is presented as resolving/addressing path-reconstruction quality issues related to the broader V8 pipeline.
- **Observed**: Ground truth shows funnel.v8 is currently FIRING CRITICAL with 4162 consecutive failures, tied to a specific stuck-candidate failure mode (telegram_received with no terminal disposition) — a distinct issue from the xval/path-reconstruction tolerance work described in the claim.
- **Expected**: The claim does not assert it fixes funnel.v8, so no direct contradiction exists, but a reader could conflate 'xval pilot resolved' with 'V8 funnel issues resolved' — this is worth flagging as a scope boundary.
- **Evidence**: EV005
- **Impact**: Readers should not interpret the xval/T1 completion claim as evidence that the actively-firing funnel.v8 CRITICAL incident is resolved; it is a separate mechanism.
- **Next step**: Check whether thr_reconstruct_paths.py or compute_interpolated_xval is invoked anywhere in the funnel.v8 check code path to confirm they are unrelated systems.
- **Confidence**: medium

## [INFO] F5: Claim internal consistency (not independently verifiable)
- **Claim**: p50=2.106% p75=8.319% p90=72.584% mean=18.569%; XVAL_TOLERANCE_PCT = 15.0 set at ~2x p75; Pilot pass rate at this tolerance: 9/11 tokens (82%).
- **Observed**: No evidence item in the audit bundle contains these statistics, the tolerance constant, or pass-rate figures; this is purely narrative from EV007 documentation text with no corroborating check output, log, or test artifact in the ground-truth evidence set.
- **Expected**: Quantitative claims of this specificity would ideally be traceable to a logged pilot-run artifact or test fixture referenced elsewhere in evidence.
- **Evidence**: EV007
- **Impact**: These statistics cannot be verified or falsified from the audit evidence; they must be treated as unverified assertions pending direct inspection of pilot run outputs.
- **Next step**: Locate and inspect the raw pilot run output/log file (or re-run the pilot script) to confirm the p50/p75/p90/mean figures and 9/11 pass rate.
- **Confidence**: low
