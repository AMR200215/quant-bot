# Layer 2 Audit: audit-20260918T082304Z
Generated: 2026-09-18T08:23:02Z
Evidence SHA-256: `7b923d3db1d7ca8e8f113d1c36e48f1ddcb01fcac38a99644974afa870ea7434`
Status: ok

## [CRITICAL] F4: funnel.v8 incident vs 'live-verified clean' claims
- **Claim**: live-verified clean (zero errors, zero duplicate open positions on the actual live book post-restart)... Live post-restart: zero errors, zero duplicates, book actively growing...the system is doing real work under all six fixes from 2026-09-12 through today at once.
- **Observed**: EV005 shows the funnel.v8 check is FIRING at CRITICAL severity, first seen 1786748706.89, still firing at last_seen 1789719604.70, with consecutive_failures: 6791, and a specific stuck candidate (event_id bd6003380054eb57) stuck at telegram_received stage for over 120s with no terminal disposition.
- **Expected**: If the system were truly 'live-verified clean' and 'doing real work' without issue across all fixes, the funnel.v8 pipeline check would be expected to be OK/RECOVERED rather than actively firing as a critical, long-running incident.
- **Evidence**: EV005
- **Impact**: There is an active, long-running critical pipeline funnel issue at the time of audit that directly contradicts the documentation's blanket claim of a clean, fully-functioning live system; this could indicate the claimed fixes did not address (or are unrelated to) a separate ongoing funnel problem.
- **Next step**: Investigate the funnel.v8 stuck-candidate pipeline (trace event_id bd6003380054eb57 / mint AFf278av4oRQicFnpeGHFvcXqjmfNac3XaVyVZAhpump through telegram_received handling) to determine root cause and whether it relates to the claimed fixes or a separate defect.
- **Confidence**: high

## [WARN] F1: git commit / deployment SHA
- **Claim**: Committed b93b9de, deployed, live-verified clean... Committed 9589c03, deployed. Live post-restart: zero errors, zero duplicates, book actively growing...
- **Observed**: Deployed HEAD is a9f2b9aa053b122c9db50d2c85749c8acb8ddf9e (EV002, EV005). Neither b93b9de nor 9589c03 appear anywhere in the ground-truth evidence as the current HEAD or in any recorded git_sha field. The most recent cron job git_sha recorded is b9571e776ffea1424bce413be0dbd5ac92855ed4, also not matching either claimed commit.
- **Expected**: If these fixes were deployed as claimed, one of the claimed short SHAs (b93b9de, 9589c03) would be expected to be an ancestor of or match the currently deployed HEAD, or otherwise be traceable in deployment evidence.
- **Evidence**: EV002, EV005
- **Impact**: Cannot verify from evidence that the claimed concurrency and batch-save fixes are actually present in the currently running deployed code.
- **Next step**: Run `git log --oneline a9f2b9aa..HEAD` and `git merge-base --is-ancestor b93b9de HEAD` / same for 9589c03 to confirm these commits are ancestors of the deployed SHA.
- **Confidence**: high

## [WARN] F2: working tree cleanliness vs deploy claim
- **Claim**: Committed and deployed (implying a clean, reproducible deployment state) for both the TOCTOU fix and the batch-save performance fix.
- **Observed**: EV002 shows the working tree is NOT clean — multiple modified tracked files and numerous untracked files/scripts (including ad hoc scripts like p15_1_2_audit.py, scratch_incident_check.py) are present at the audited HEAD.
- **Expected**: A claim of 'committed, deployed' work would typically correspond to a clean or near-clean working tree matching the stated commit, without unrelated uncommitted changes.
- **Evidence**: EV002
- **Impact**: Uncommitted/untracked changes alongside claimed deploys make it hard to confirm exactly what code is running versus what was tested, risking drift between audited and live behavior.
- **Next step**: Run `git status --porcelain` and `git diff HEAD` on the deployed host to enumerate exactly what differs from the claimed committed state.
- **Confidence**: medium

## [INFO] F3: test suite counts
- **Claim**: 52/52 in this file, 709 across all v8-related suites.
- **Observed**: Ground truth test collection counts from EV005 are per broad suite (memecoin 335, research 635, watchdog 103, layer2 39, quant-bot 56) and do not break out a 'v8-related suites' subset or a specific file with 52 tests, so this specific claim cannot be independently confirmed or refuted.
- **Expected**: If accurate, the claimed 709 v8-related tests and 52-test file would be a subset reflected somewhere in the aggregate suite counts.
- **Evidence**: EV005
- **Impact**: No way to verify the specific v8 test count claim against current ground truth; it is an unverifiable assertion, not a contradiction.
- **Next step**: Run the v8-related test suite directly (e.g. `pytest -k v8 --collect-only -q`) and compare the reported count to 709.
- **Confidence**: low

## [INFO] F5: book growth claim (236 vs 194 positions)
- **Claim**: book actively growing (236 total positions vs 194 at the last check, 3 genuinely open and tracking)
- **Observed**: No position-count data (194, 236, or otherwise) appears anywhere in the ground-truth evidence bundle (EV001-EV007); this metric is not covered by any of the audited checks.
- **Expected**: N/A - this is an assertion outside the scope of available evidence.
- **Evidence**: EV005
- **Impact**: Cannot corroborate or refute the specific position-count growth claim; it is unverifiable from current evidence.
- **Next step**: Query the live positions.json or equivalent store directly for current total and open position counts to confirm the claimed figures.
- **Confidence**: low

## [INFO] F6: claims.batch.v8_readiness incident vs documentation completeness claims
- **Claim**: Implicit claim of thorough completion/readiness given detailed fix narratives (TOCTOU fix, batch-save performance fix) presented as fully resolved.
- **Observed**: EV005 shows claims.batch.v8_readiness is FIRING (WARN) with consecutive_failures: 1008, and 4 of 7 items (N2, N4, N6, N7) are PARTIAL rather than GREEN.
- **Expected**: If the described fixes fully resolved v8 readiness concerns, this batch check would be expected to show all items GREEN rather than a persistent PARTIAL/WARN state.
- **Evidence**: EV005
- **Impact**: Suggests broader v8 readiness gaps remain despite the specific concurrency and performance fixes described in the claims, which only address a narrow slice of the readiness criteria.
- **Next step**: Review the definitions of readiness items N2, N4, N6, N7 to determine what remains PARTIAL and whether it relates to the claimed fixes.
- **Confidence**: medium
