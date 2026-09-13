# Layer 2 Audit: audit-20260913T082618Z
Generated: 2026-09-13T08:26:16Z
Evidence SHA-256: `fb5ab3b998af09aa0d0b03bf8dbb36997c1afe626ef6b042bb0641cae9f641c8`
Status: ok

## [CRITICAL] F1: v8_paper trading pipeline / funnel.v8 incident
- **Claim**: Fixed: widened _GATE_CAPTURE_WAIT_S to 2.3s and added curve-account price fallback for pp_unpriced; deployed via commit 900d47b, live restart confirmed clean.
- **Observed**: funnel.v8 is CRITICAL and FIRING at the time of the watchdog fast run (run_id 8c2806b9f000450c, same git SHA 57640f6 as the rest of the audit), with consecutive_failures=5332 and last_seen matching the run timestamp — i.e. still actively failing, not recovered. The flagged candidate event (mint AFf278av...) shows telegram_received with no terminal disposition, exceeding the 120s threshold.
- **Expected**: If the claimed fixes (widened gate-capture wait, curve-account price fallback) were deployed and working, funnel.v8 would be expected to show recovery or at least a much lower/reset consecutive_failures count, not a still-firing incident with a large failure streak at the same deployed SHA.
- **Evidence**: EV005, EV002
- **Impact**: The documentation claims a bug fix was deployed and live-restarted cleanly, but the watchdog's own incident state shows the underlying funnel/pipeline problem is still actively firing at the same SHA — the claimed fix's effectiveness cannot be confirmed and may not be resolving the issue.
- **Next step**: Check whether git SHA 900d47b (claimed fix commit) matches the deployed SHA 57640f698bafebdffe5e3c5aa3aea8c4064dc68e; if not, the fix in the receipts may not yet be live. Run `git log --oneline -5 57640f698bafebdffe5e3c5aa3aea8c4064dc68e` and compare against claimed commits 07dd982/900d47b.
- **Confidence**: high

## [WARN] F2: Deployed SHA vs claimed commits
- **Claim**: Committed `07dd982`, pushed, pulled on VPS clean... Committed `900d47b`, pushed, pulled/deployed...
- **Observed**: The ground truth confirms deployed SHA 57640f698bafebdffe5e3c5aa3aea8c4064dc68e consistently across watchdog runs and job receipts. No evidence in the ground-truth summary shows the relationship (ancestry) between 57640f6 and the claimed commits 07dd982 or 900d47b.
- **Expected**: If the claims are accurate, 57640f6 should be a descendant of (or equal to) 900d47b, since that was described as the most recent deploy.
- **Evidence**: EV002, EV005
- **Impact**: Cannot verify from current evidence whether the fixes described in the claims are actually present in the currently running deployed code.
- **Next step**: Run `git merge-base --is-ancestor 900d47b 57640f698bafebdffe5e3c5aa3aea8c4064dc68e && echo yes || echo no` on the VPS repo to confirm the claimed fix commit is included in the deployed SHA.
- **Confidence**: medium

## [WARN] F3: Working tree cleanliness vs deploy narrative
- **Claim**: pulled on VPS clean (no conflicts) ... Stash count verified back to baseline (50) after deploy.
- **Observed**: Working tree is not clean: 6 modified tracked files and 19 untracked paths, including two ad-hoc scripts at repo root (p15_1_2_audit.py, scratch_incident_check.py). No evidence distinguishes expected service-writes from manual edits.
- **Expected**: A claim of a clean pull and stash-count restored to baseline implies the working tree should be in a known, reconciled state post-deploy, with no unexplained modified/untracked files beyond what's expected from running services.
- **Evidence**: EV002
- **Impact**: Untracked ad-hoc scripts and modified tracked files not explained by the deploy narrative could indicate incomplete cleanup, uncommitted debugging work, or drift between what was tested and what is deployed.
- **Next step**: Run `git status --porcelain` and `git diff docs/RECEIPTS.md docs/V8_INPUTS.md` on the VPS to determine whether the dirty state is limited to expected log/data writes or includes undocumented manual edits.
- **Confidence**: medium

## [INFO] F4: Test suite pass/fail vs claimed test results
- **Claim**: 9 new tests (memecoin/tests/test_v8_paper.py, TestCurveFallbackPricing), 28/28 passing in that file, 685 passing across all v8-related suites.
- **Observed**: Ground truth evidence only shows test *collection* counts (e.g., memecoin: 311 tests collected) with no pass/fail run results present anywhere in the evidence bundle.
- **Expected**: The claim asserts specific pass counts (28/28, 685 total) from an actual test run, which would require pass/fail evidence not present in this audit's evidence set.
- **Evidence**: EV005
- **Impact**: The specific pass/fail numbers in the claim cannot be independently corroborated by the audit evidence; test collection succeeding does not confirm these pass counts.
- **Next step**: Run `pytest memecoin/tests/test_v8_paper.py -v` and the full v8-related suite on the deployed SHA to independently verify the claimed 28/28 and 685 passing counts.
- **Confidence**: medium

## [INFO] F5: claims.batch.v8_readiness incident vs claim narrative
- **Claim**: Implicit in the overall v8_paper fix narrative that upstream issues (progress_unknown, pp_unpriced) causing zero paper positions have been root-caused and addressed.
- **Observed**: claims.batch.v8_readiness is WARN/FIRING with consecutive_failures=870, batch detail 3/7 GREEN, 4/7 PARTIAL (N2, N4, N6, N7), 0 FAIL — a distinct, still-active incident about v8 readiness that predates and continues through this audit window.
- **Expected**: If the v8_paper fixes fully resolved the underlying readiness gaps, one might expect the PARTIAL items in claims.batch.v8_readiness to trend toward GREEN, though the claim text does not explicitly say this batch check should have cleared.
- **Evidence**: EV005
- **Impact**: There may be additional unresolved v8-readiness gaps beyond the two bugs described in the claim, warranting closer inspection of what N2/N4/N6/N7 represent.
- **Next step**: Inspect the batch_verify CLI output/detail for items N2, N4, N6, N7 to determine if they relate to the same progress_unknown/pp_unpriced issues or are independent readiness gaps.
- **Confidence**: low

## [INFO] F6: Live restart / startup log confirmation
- **Claim**: Startup log line from the VPS process confirms: `v8_paper: monitor thread started (interval=5s, gate=CURVE_ACTIVE [V8-P0], config_tag=v8_candidate_2026-09-10_v8p0)`.
- **Observed**: No systemd evidence beyond ActiveState/SubState/Result is available for quantbot.service (which would host v8_paper); no startup log lines are present in the ground-truth evidence bundle to corroborate or refute this specific claim.
- **Expected**: The claim's specific startup log line is an assertion not verifiable from the evidence gathered for this audit.
- **Evidence**: EV003
- **Impact**: Cannot confirm the process actually restarted with the described configuration; relying solely on the documentation's self-report for this detail.
- **Next step**: Run `journalctl -u quantbot.service | grep 'v8_paper: monitor thread started'` on the VPS to independently confirm the startup log line and its timestamp/config_tag.
- **Confidence**: medium

## [INFO] F7: Latent live pricing cache bug (executor.py _sol_price_usd)
- **Claim**: found separately while investigating, that cache bumps its own 'last updated' timestamp even when the Jupiter fetch fails ... This is a real, separate latent bug in live-money pricing code ... flagged here for visibility, not fixed.
- **Observed**: No evidence in the ground-truth summary evaluates memecoin/executor.py's _sol_price_usd() caching behavior, Jupiter fetch failure handling, or LIVE_TRADING flag state; this is entirely outside the scope of the audited evidence (EV001-EV007).
- **Expected**: N/A — this is a self-reported, unverified claim about a separate code path not covered by any audit evidence.
- **Evidence**: EV005
- **Impact**: An unverified but plausible live-money pricing risk is disclosed; if true and LIVE_TRADING were ever enabled without addressing it, stale price data could silently mask fetch failures.
- **Next step**: Review memecoin/executor.py's _sol_price_usd() exception handler directly and confirm current LIVE_TRADING flag value in deployed config before any go-live decision.
- **Confidence**: low
