# Layer 2 Audit: audit-20260915T084934Z
Generated: 2026-09-15T08:49:32Z
Evidence SHA-256: `9e0cb95c15d03ee07baf4164ab14b286491cbc0da86c55f73b3c6985fb8b7d91`
Status: ok

## [CRITICAL] F1: v8_paper / funnel.v8 discrepancy
- **Claim**: Fixed: widened to 2.3s (paper-only book, zero latency cost). ... 685 passing across all v8-related suites. Committed 900d47b, pushed, deployed ... Live restart confirmed clean via the same startup log line.
- **Observed**: Ground truth shows funnel.v8 is CRITICAL and actively FIRING with consecutive_failures: 5920, describing a candidate stuck at stage telegram_received with no terminal disposition >120s, explicitly flagged as the 'V8-TWIN-FIX failure class' pattern. Deployed SHA is 17a8c4e0..., not 900d47b or 07dd982 mentioned in the claim.
- **Expected**: If the claimed fixes (progress_unknown timeout widening, pp_unpriced fallback) were deployed and effective, funnel.v8 would be expected to show recovery or at least a different failure signature (stuck-at-stage vs. rejected-with-reason), not a long-running CRITICAL incident with thousands of consecutive failures.
- **Evidence**: EV005, EV007, EV002
- **Impact**: The claimed deploy/fix may not be reflected in the currently running system, or the fix did not resolve the underlying funnel blockage — real paper trading may still be non-functional despite documentation asserting a fix was shipped and verified.
- **Next step**: Diff deployed HEAD (17a8c4e0...) against commits 07dd982 and 900d47b to confirm whether the claimed fix commits are actually ancestors of the running SHA; inspect v8_paper.py's _GATE_CAPTURE_WAIT_S value on disk.
- **Confidence**: high

## [WARN] F2: Commit SHA mismatch
- **Claim**: Committed 07dd982, pushed, pulled on VPS clean ... Committed 900d47b, pushed, deployed (stash/pull/pop ...).
- **Observed**: Ground truth deployed SHA across all EV002/EV005 sources is 17a8c4e016e00515c0e61baafe55147e1f984882, which matches neither 07dd982 nor 900d47b referenced in the claim, nor db32f53 referenced elsewhere in evidence for claims.batch.rc_closure.
- **Expected**: If the claim's deploy narrative is current and accurate, the deployed HEAD should match one of the referenced commit hashes (ideally 900d47b as the latest).
- **Evidence**: EV002, EV005
- **Impact**: Cannot confirm the narrative in the claim reflects the currently running code; the bundle gives no commit graph to establish ordering or ancestry between 17a8c4e0, 07dd982, 900d47b, and db32f53.
- **Next step**: Run `git log --oneline 17a8c4e0 | grep -E '07dd982|900d47b'` on the deployed host to check ancestry.
- **Confidence**: medium

## [WARN] F4: Live-money pricing latent bug (executor.py _sol_price_usd)
- **Claim**: found separately while investigating, that cache bumps its own 'last updated' timestamp even when the Jupiter fetch fails ... This is a real, separate latent bug in live-money pricing code (used for real buy/sell fill pricing, not just this paper twin) — flagged here for visibility, not fixed.
- **Observed**: No evidence in the ground-truth bundle (EV001-EV007) directly confirms or refutes this claim about executor.py's staleness-check bug; it is an assertion about code behavior not covered by any watchdog check in evidence.
- **Expected**: N/A — this is a claim about a documented but unverified/unfixed bug; ground truth has no corroborating or contradicting check for it.
- **Evidence**: EV005
- **Impact**: If true and left unfixed, this could cause silent stale pricing in live trading once LIVE_TRADING is enabled, but current evidence cannot confirm the bug exists or its current fix status.
- **Next step**: Locate and review executor.py's _sol_price_usd() exception handler directly on the deployed SHA to verify whether the described timestamp-bump-on-failure behavior is present.
- **Confidence**: low

## [INFO] F3: Working tree cleanliness vs. claimed clean deploy
- **Claim**: pushed, pulled on VPS clean (no conflicts), systemctl restart quantbot ... stash count verified back to baseline (50) after deploy.
- **Observed**: EV002 shows the working tree is NOT clean — numerous modified tracked files and many untracked files/directories, with no evidence they are staged, committed, or explained as expected runtime artifacts.
- **Expected**: A claim of a clean pull and restored stash baseline implies the working tree should be clean (matching HEAD) aside from expected runtime-generated files.
- **Evidence**: EV002
- **Impact**: Uncommitted/untracked changes on the deployed host create risk of divergence between what was tested/reviewed and what is actually running, undermining the claim's assertion of a clean, verified deploy state.
- **Next step**: Run `git status --porcelain` and `git stash list` on the VPS to reconcile the claimed stash baseline of 50 against current state.
- **Confidence**: medium

## [INFO] F5: Test suite pass counts
- **Claim**: 9 new tests (memecoin/tests/test_v8_paper.py, TestCurveFallbackPricing), 28/28 passing in that file, 685 passing across all v8-related suites.
- **Observed**: Ground truth only confirms test collection counts (e.g., tests_memecoin: 313) via EV005's test_drift.collection checks, which indicate successful import/collection, not pass/fail results. No pass-count of 685 or 28/28 appears anywhere in the audited evidence.
- **Expected**: A verifiable pass-rate claim would need corresponding pass/fail run evidence, which is absent from the bundle.
- **Evidence**: EV005
- **Impact**: Cannot independently verify the claimed test pass rates; documentation claims of passing tests are unconfirmed by the audited evidence.
- **Next step**: Run the referenced test file directly (`pytest memecoin/tests/test_v8_paper.py -v`) on the deployed SHA and compare actual results to the claimed 28/28.
- **Confidence**: medium

## [INFO] F6: claims.batch.v8_readiness PARTIAL items vs. narrative
- **Claim**: (implicit) the v8 fix narrative implies v8 readiness should be improving/resolved.
- **Observed**: Ground truth shows claims.batch.v8_readiness still WARN/firing with consecutive_failures: 925, 4 of 7 items PARTIAL (N2, N4, N6, N7), explicitly noted as a case where underlying exit code would read as passing despite incomplete items.
- **Expected**: If the claimed fixes were fully deployed and effective, one would expect improvement in v8_readiness batch status rather than continued PARTIAL results across multiple items.
- **Evidence**: EV005
- **Impact**: The persistent PARTIAL/WARN state suggests the v8 rollout remains incomplete regardless of the specific bug fixes described in the claim, and the documentation's optimistic tone is not corroborated by the readiness batch data.
- **Next step**: Inspect the specific N2/N4/N6/N7 item definitions in the v8_readiness batch to determine what remains incomplete and whether it relates to the progress_unknown/pp_unpriced fixes.
- **Confidence**: medium

## [INFO] F7: Open/unverified claim of pending live confirmation
- **Claim**: Still genuinely open: no fresh paper trade has landed under the fixed code yet — the next real Telegram alert will confirm end-to-end.
- **Observed**: This portion of the claim is consistent with ground truth in spirit — the funnel.v8 incident (evidence-based) shows an unresolved, ongoing failure to reach a terminal disposition, which aligns with the claim's own admission that the fix is unconfirmed in production.
- **Expected**: The claim's self-acknowledged uncertainty matches the observed absence of resolution in funnel.v8; this is the one part of the claim that is not contradicted by ground truth.
- **Evidence**: EV005
- **Impact**: This is the most honest part of the claim and aligns with observed reality — no confirmed evidence exists either way that the underlying fix works end-to-end.
- **Next step**: Monitor funnel.v8 consecutive_failures count going forward; a decrease or reset to 0 would corroborate the claimed fix taking effect.
- **Confidence**: high
