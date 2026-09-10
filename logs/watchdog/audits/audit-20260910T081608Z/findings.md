# Layer 2 Audit: audit-20260910T081608Z
Generated: 2026-09-10T08:16:06Z
Evidence SHA-256: `0e0b6446ffac0c86b15a11249c21bd32a2f9d77fa66b18755a5fb61d4c15ffc4`
Status: ok

## [WARN] F2: Unverified analytical/statistical claims
- **Claim**: Table of trail_stop/profit_lock captured_fraction values (e.g. V8-P0 E0/E2/E3: 8 n, +53.0% mean realized, +207.7% mean eventual peak, 0.20 captured_fraction) and claims about hard_stop/time_stop calibration correctness.
- **Observed**: No evidence bundle (EV001-EV007) in the ground-truth summary contains any watchdog check, test result, or raw data file confirming these specific statistics (winner counts, percentages, captured fractions). These figures exist only in the claims/documentation layer.
- **Expected**: A verifiable claim of this specificity would typically be backed by a checked-in results file (thr_winner_exit_results.json) with matching figures, ideally referenced by a reproducible EV-numbered artifact or test run in the evidence chain.
- **Evidence**: EV007
- **Impact**: Trading/exit-strategy conclusions (e.g. 'trail_stop is the actual leak') may be acted upon by the team despite being unverified against raw ground truth, risking decisions based on unaudited analysis.
- **Next step**: Run/inspect research/thr_winner_exit_analysis.py directly against research/thr_winner_exit_results.json and cross-check the printed captured_fraction values against the table in the claim.
- **Confidence**: medium

## [INFO] F1: receipts_tail content vs prior evidence gap
- **Claim**: EV007 receipts_tail contains a detailed research analysis entry (T5 winner-exit analysis, trail_stop/profit_lock findings) and declares 'T5 status: COMPLETE. THR-BATCH (T1-T5) status: COMPLETE.'
- **Observed**: Ground truth previously noted EV007 only as an artifact with mtime 1789004747 and length 250223 bytes, with no content available to verify. This claim now supplies content, but that content is an assertion from documentation, not independently verified against raw evidence (e.g. no corresponding check in EV005 validates 'THR-BATCH' or T1-T5 completion status).
- **Expected**: If COMPLETE status were verifiable, we would expect a corresponding watchdog check, CI status, or file artifact (e.g. research/thr_winner_exit_results.json) confirmed present and non-empty in evidence, not just referenced in prose.
- **Evidence**: EV007, EV005
- **Impact**: The claimed completion of the THR-BATCH analysis (T1-T5) cannot be corroborated from ground-truth evidence; it rests solely on the documentation's own self-report.
- **Next step**: Check for existence and non-zero size of research/thr_winner_exit_results.json and research/thr_winner_exit_analysis.py on disk, and confirm via git log/diff whether these were committed or are part of the untracked/modified files noted in EV002.
- **Confidence**: medium

## [INFO] F3: Append-only correction convention
- **Claim**: This correction is appended, not silently substituted for the entry above, per this project's own standing convention (docs/OPEN_BRANCHES.md and every other correction in this file follows the same append-only pattern).
- **Observed**: Ground truth has no visibility into docs/OPEN_BRANCHES.md content or historical RECEIPTS.md structure/diffs; EV002 shows docs/RECEIPTS.md as a modified tracked file but its diff content is not in evidence.
- **Expected**: Verifying an 'append-only' convention claim would require diffing docs/RECEIPTS.md against its git history to confirm prior entries remain intact and new content was only appended.
- **Evidence**: EV002
- **Impact**: Cannot confirm documentation integrity/process-adherence claims; if the convention were violated, prior audit trail could be silently lost without detection.
- **Next step**: Run git diff/git log -p on docs/RECEIPTS.md to confirm only additions (no deletions/rewrites) occurred in the modified working-tree version relative to HEAD.
- **Confidence**: low

## [INFO] F4: Working tree state consistency
- **Claim**: Implicit in the claim: docs/RECEIPTS.md contains up-to-date, finalized (COMPLETE) analysis, implying the working tree changes are intentional, reviewed final-state edits.
- **Observed**: EV002 confirms docs/RECEIPTS.md is one of 5 modified tracked files in an unclean working tree, alongside 19+ untracked files including new scripts and logs; this working tree state was not committed at the time of the audit snapshot.
- **Expected**: A 'COMPLETE' status claim for a batch of work would typically coincide with a clean commit reflecting that completion, rather than uncommitted local modifications.
- **Evidence**: EV002
- **Impact**: The claimed COMPLETE status for THR-BATCH exists only in an uncommitted working-tree file, meaning it is not part of the deployed/audited SHA (d038929) and could be lost, altered, or diverge from what actually ran in production.
- **Next step**: Run git status --porcelain and git diff docs/RECEIPTS.md to confirm the exact uncommitted changes, then confirm whether this content is intended for commit.
- **Confidence**: high

## [INFO] F5: No contradiction with core audit findings
- **Claim**: Overall EV007 claim content pertains to a research/strategy analysis (exit-rule tuning) unrelated to systemd, cron, incidents, or SHA-mismatch topics.
- **Observed**: Ground truth's active-incident, cron, systemd, and SHA-mismatch findings (funnel.v8 FIRING, claims.batch.v8_readiness FIRING, SHA discrepancy) are not addressed, contradicted, or corroborated by this claim at all.
- **Expected**: No expectation of overlap; this is a scope note rather than a discrepancy.
- **Evidence**: EV005
- **Impact**: None directly; this claim does not affect assessment of the currently firing incidents or SHA mismatch, which remain the higher-severity open items from the ground truth audit.
- **Next step**: Continue tracking funnel.v8 and claims.batch.v8_readiness incidents independently of this documentation claim; no action needed based on this claim alone.
- **Confidence**: high
