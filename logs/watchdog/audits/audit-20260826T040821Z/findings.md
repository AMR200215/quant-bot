# Layer 2 Audit: audit-20260826T040821Z
Generated: 2026-08-26T04:08:19Z
Evidence SHA-256: `780c2cf43d6fd6316393c7cd33efe35572334d223727dca2de94c6151133c1c5`
Status: ok

## [WARN] F1: V8 readiness / funnel.v8
- **Claim**: PumpPortal funding verified live end-to-end, real production traffic, not synthesized: subscribeTokenTrade accepted, real trade ticks received, corrected reserve price, v3 path CSV written and on disk, path integrity VALID, $2/$5 execution-proxy observation written.
- **Observed**: funnel.v8 is currently FIRING at CRITICAL severity with 24 consecutive failures, and the latest fast-check shows a candidate stuck in telegram_received stage for over 120s with no terminal disposition, described by the watchdog itself as the 'V8-TWIN-FIX failure class.' claims.batch.v8_readiness is also FIRING (WARN) with 367 consecutive failures, 4/7 items PARTIAL rather than GREEN.
- **Expected**: If the funding fix and pipeline corrections described in the claim fully resolved the readiness pipeline, funnel.v8 and claims.batch.v8_readiness would be expected to show RECOVERED or healthy state, not active firing incidents.
- **Evidence**: EV005
- **Impact**: The documentation describes a resolved, verified end-to-end pipeline, but the live watchdog shows an active critical funnel failure and a persistently degraded readiness batch, indicating the fix claimed in docs has not translated into current operational health.
- **Next step**: Inspect the funnel.v8 stuck candidate's telegram_received stage logs and cross-reference timestamps against the 2026-08-22 funding date to determine whether the claimed fix predates or postdates the current failure window.
- **Confidence**: high

## [WARN] F2: claims.batch.v8_readiness PARTIAL items
- **Claim**: New research/v8_collection_yield.py separates population counts from collector-yield counts; readiness denominator bug fixed; 18 new tests, 488/488 total green.
- **Observed**: claims.batch.v8_readiness slow-check shows 3/7 GREEN, 4/7 PARTIAL (items N2, N4, N6, N7), 0 FAIL. No evidence in the bundle explains what 'PARTIAL' means for these specific items, and test collection in EV005 shows 498 tests for tests_research (not 488 as claimed for the denominator-audit commit).
- **Expected**: If the denominator fix and full test suite pass described in the claim were the final state, the readiness batch would be expected to show fully GREEN items, and the test count referenced by the claim (488/488) should reconcile with the currently observed collection totals (498 for tests_research per EV005 tests_memecoin/tests_research counts).
- **Evidence**: EV005
- **Impact**: The claim's test count and 'fixed' framing cannot be reconciled with the currently PARTIAL batch state, making it unclear whether the described fix is fully deployed or only partially reflected in the current commit.
- **Next step**: Run the v8_readiness batch-verify tool with verbose output to see the specific reason codes for N2, N4, N6, N7, and diff against the 488→498 test count discrepancy by checking git log between the denominator-audit SHA (2c2ab8c) and current HEAD.
- **Confidence**: medium

## [WARN] F5: research.path_collection / PumpPortal budget
- **Claim**: PumpPortal funding verified live end-to-end, real production traffic; path integrity VALID; no code changes needed, pipeline was already correct, purely blocked on funding.
- **Observed**: research.path_collection is currently FIRING (WARN) with 62 consecutive failures; the latest slow-check for 2026-08-25 shows path yield at 36.2% (below a 50% floor) and PumpPortal daily message budget exceeded (105,667/100,000, 64 tokens dropped). The watchdog characterizes this as a 'known, cost-bounded constraint' but that is the tool's own assertion, not independently verified.
- **Expected**: If funding fully resolved the collection pipeline as claimed, one would expect path yield to be healthy (above the 50% floor) and message budgets to be respected, rather than an active WARN incident with sub-floor yield and budget overage.
- **Evidence**: EV005
- **Impact**: The funding fix may have resolved the specific 'Minimum balance not met' blocker but the pipeline still exhibits an active yield/budget problem, suggesting the claim of the pipeline being fully healthy post-funding is incomplete or outdated.
- **Next step**: Check research.path_collection historical trend since 2026-08-22 (funding date) to determine whether yield improved post-funding and whether the current WARN is a new, separate issue (budget/cost-bounded) rather than the original funding blocker.
- **Confidence**: medium

## [INFO] F3: Git SHA / commit lineage
- **Claim**: Readiness denominator audit/correction at git SHA 2c2ab8c; path_stats.py --valid-only at git SHA 7d2e319 (refactor 333dbfb); EXPERIMENT V2 E3 at 'git SHA pending this commit'.
- **Observed**: Deployed HEAD is 9fd7a7166f3972bc16936775f0ea60ac3bf4818b (EV002), and claims.batch.rc_closure reports commit db32f53 (EV005) -- neither of which matches any of the SHAs named in the claim (2c2ab8c, 7d2e319, 333dbfb). The working tree is also dirty with 6 modified tracked files and 19 untracked files.
- **Expected**: If the claim's narrative reflects the current deployed state, one would expect the deployed HEAD or the rc_closure commit to correspond to, or descend cleanly from, the SHAs referenced in the claim's changelog.
- **Evidence**: EV002, EV005
- **Impact**: Without reconciling these SHAs, it is impossible to confirm which of the claimed code changes (denominator fix, path_stats refactor, E3 exit candidate) are actually present in the currently running deployment.
- **Next step**: Run `git log --oneline 2c2ab8c..9fd7a7166f` and `git log --oneline db32f53..9fd7a7166f` to establish ancestry and confirm whether the claimed commits are ancestors of the current deployed HEAD.
- **Confidence**: medium

## [INFO] F4: Test suite count consistency
- **Claim**: Full suite 498/498 green (after EXPERIMENT V2 E3 addition, EXIT_REGISTRY_VERSION 1→2).
- **Observed**: EV005 slow-run test collection shows tests_research reporting 498 tests collecting cleanly with no errors -- this number matches the claim's final total.
- **Expected**: Claim's final stated test count (498) should match observed collection count, which it does.
- **Evidence**: EV005
- **Impact**: This specific claim about total test count is corroborated by ground-truth evidence, increasing confidence in this part of the changelog narrative even though other parts (readiness batch state) are not corroborated.
- **Next step**: No action needed; this data point is consistent, but continue to verify other numeric claims (e.g. 488/488, 18 new tests) independently since they belong to an earlier, unverified commit.
- **Confidence**: high

## [INFO] F6: Working tree dirtiness vs. claimed code changes
- **Claim**: Multiple code changes referenced (v8_collection_yield.py, path_stats.py --valid-only, v8_exit_registry.py EXPERIMENT V2 E3) with specific git SHAs for each.
- **Observed**: EV002 shows the working tree is dirty with 6 modified tracked files (none of which are the research/*.py files named in the claim) and 19 untracked files including two ad-hoc scripts, but no evidence indicates whether the claimed research files are committed, modified, or absent from the current tree.
- **Expected**: If the claimed research module changes are fully committed as described, they should not appear as uncommitted/dirty changes; but the bundle provides no direct evidence confirming their presence or absence in the working tree at all.
- **Evidence**: EV002
- **Impact**: Cannot confirm from ground truth whether the specific files referenced in the claims (v8_collection_yield.py, v8_exit_registry.py) are actually present, committed, and active in the deployed code.
- **Next step**: Run `git show HEAD:research/v8_collection_yield.py` and `git show HEAD:research/v8_exit_registry.py` to confirm these files exist at the claimed content/version in the deployed HEAD.
- **Confidence**: low
