# Layer 2 Audit: audit-20260911T081102Z
Generated: 2026-09-11T08:11:00Z
Evidence SHA-256: `6da4a740ed7ea0c239028a82b36646d89f9e693d40b300669e5012cf115283d9`
Status: ok

## [WARN] F2: V8_PAPER_ENTRY_GATE_SWITCH commit SHA vs deployed HEAD
- **Claim**: Deploy: committed `07dd982`, pushed, pulled on VPS clean (no conflicts), `systemctl restart quantbot`.
- **Observed**: Ground truth EV002/EV005 shows the deployed HEAD SHA is `188cc85b94d884b219bb08d26e889c0213f0d643`, echoed consistently across watchdog runs and job receipts. No evidence ties `07dd982` to the currently running commit.
- **Expected**: If the claim's deploy commit `07dd982` were the latest applied change, it would be expected to be an ancestor of or identical to the current HEAD; the evidence bundle does not show this relationship, and HEAD is a different, unexplained hash.
- **Evidence**: EV002, EV005
- **Impact**: Cannot confirm from evidence that the V8 paper entry gate change described in the claim is actually present in the currently running code — the working tree is also dirty, further obscuring what code is live.
- **Next step**: Run `git log --oneline -5` and `git show 188cc85:memecoin/v8_paper.py | grep -A3 passes_v8_gate` on the deployed host to confirm whether commit 07dd982 is an ancestor of current HEAD and whether the gate change is present in the running file.
- **Confidence**: medium

## [WARN] F4: funnel.v8 CRITICAL incident vs V8 gate change claims
- **Claim**: V8_PAPER_ENTRY_GATE_SWITCH — LIVE_VERIFIED (2026-09-10) ... Not yet done: no fresh V8-P0-gated paper trades have landed yet to confirm the new gate is actually admitting positions at the expected rate live.
- **Observed**: Ground truth shows an active, currently FIRING CRITICAL incident `funnel.v8` with consecutive_failures: 4745, describing a candidate stuck at telegram_received stage with no terminal disposition, unresolved as of the latest fast watchdog run.
- **Expected**: If the V8 gate switch were fully live and functioning end-to-end as implied by 'LIVE_VERIFIED,' one might expect the v8 funnel to show healthier throughput; the claim itself acknowledges this gap ('not yet done'), which is consistent with the observed firing incident, but the claim's 'LIVE_VERIFIED' framing may overstate operational readiness given this concurrent failure.
- **Evidence**: EV005
- **Impact**: Labeling the gate switch 'LIVE_VERIFIED' while a related v8 funnel incident is actively firing at high failure count could mislead stakeholders into believing the v8 pipeline is fully healthy when observed evidence shows unresolved stuck candidates.
- **Next step**: Cross-reference funnel.v8's stuck candidate mint/event IDs against v8_paper's recent signal log to determine if the CRITICAL funnel incident is related to or independent of the gate switch deployment.
- **Confidence**: low

## [INFO] F1: TS-BATCH / test suite size claim
- **Claim**: 21 new tests, full suite green (635 local).
- **Observed**: EV005 shows test_drift.collection.tests_memecoin (or the memecoin suite) collecting 304 tests and the 'watchdog' suite collecting 635 — the ground truth summary lists four suites (304, 635, 103, 56) as collection counts only, with no pass/fail execution results present in any evidence.
- **Expected**: The claim asserts the full 635-test suite is 'green' (i.e., passing), not merely collected.
- **Evidence**: EV005
- **Impact**: If the 635-count suite referenced in the claim is not actually passing, downstream confidence in TS-BATCH's 'unbeaten' conclusion is unfounded, though this cannot be confirmed or refuted from available evidence.
- **Next step**: Run the full test suite (`pytest`) and capture pass/fail counts to corroborate the 'full suite green (635 local)' claim.
- **Confidence**: medium

## [INFO] F3: V8 paper gate live log confirmation
- **Claim**: Live log line confirms: `v8_paper: monitor thread started (interval=5s, gate=CURVE_ACTIVE [V8-P0], config_tag=v8_candidate_2026-09-10_v8p0)`.
- **Observed**: No log content from v8_paper or any startup log line is present in any EV item in the ground-truth summary; this claim cannot be corroborated or contradicted by available evidence.
- **Expected**: The claim implies this log line exists in current, accessible logs.
- **Evidence**: EV002, EV005
- **Impact**: Without direct log verification, the actual runtime gate configuration for v8_paper remains an unverified assertion.
- **Next step**: Grep the current quantbot service journal or v8_paper log file for the string 'v8_paper: monitor thread started' to confirm the gate and config_tag in the live process.
- **Confidence**: low

## [INFO] F5: TS-BATCH proposal outcome
- **Claim**: TS3 (selection): select_proposals() — 0 qualifying cells on either candidate. No E4/E5 proposal drafted. ... TS-BATCH status: COMPLETE.
- **Observed**: No evidence item (EV001-EV007) in the ground-truth summary references thr_ts_grid_results.json, select_proposals(), TS3/TS4 gates, or any exit-spec grid testing artifacts.
- **Expected**: This claim describes an entirely separate research workstream (exit-spec threshold grid testing) not covered by any collected evidence; it can neither be confirmed nor refuted.
- **Evidence**: EV002, EV005
- **Impact**: No operational risk identified, but the claim's correctness is unverifiable from the audit evidence and should not be treated as confirmed.
- **Next step**: If verification is desired, inspect `research/thr_ts_grid_results.json` directly and check git history for the referenced 21 new tests.
- **Confidence**: high

## [INFO] F6: rc_closure commit mismatch vs new deploy commit
- **Claim**: Deploy: committed `07dd982` ... Exit config (E0) left unchanged — TS-BATCH already confirmed it's unbeaten by anything tested.
- **Observed**: Ground truth notes claims.batch rc_closure is tied to commit `db32f53`, which differs from both the current HEAD (`188cc85...`) and the claimed deploy commit `07dd982` — three distinct, unreconciled commit references now exist across evidence and claims.
- **Expected**: A coherent deployment history would show these commits as sequential/related (e.g., db32f53 -> 07dd982 -> 188cc85 or similar ancestry), but no such relationship is established by the evidence.
- **Evidence**: EV002, EV005
- **Impact**: Difficulty in reconstructing an accurate deployment/validation timeline increases risk that claims batch verifications (rc_closure) are stale relative to the currently running code.
- **Next step**: Run `git log --oneline --all | grep -E '188cc85|07dd982|db32f53'` to establish ancestry and chronological order of these three commits.
- **Confidence**: medium

## [INFO] F7: Working tree cleanliness vs claimed clean deploy
- **Claim**: pulled on VPS clean (no conflicts) ... Stash count verified back to baseline (50) after deploy.
- **Observed**: Ground truth EV002 shows the working tree is NOT clean — modified tracked files (docs/RECEIPTS.md, docs/V8_INPUTS.md, logs, position/signal JSON) and numerous untracked files/directories are present.
- **Expected**: The claim describes a clean pull and a stash restored to baseline, implying the working tree should be clean (or at most reflect only expected runtime-generated artifacts) post-deploy.
- **Evidence**: EV002
- **Impact**: A dirty working tree post-'clean deploy' claim raises the possibility of undocumented manual changes or incomplete stash restoration, though runtime-generated logs/positions are plausible innocent explanations.
- **Next step**: Run `git status --porcelain` on the VPS and diff modified tracked files (docs/RECEIPTS.md, docs/V8_INPUTS.md) against HEAD to determine if changes are runtime writes or manual edits.
- **Confidence**: medium
