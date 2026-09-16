# Layer 2 Audit: audit-20260916T084327Z
Generated: 2026-09-16T08:43:26Z
Evidence SHA-256: `a5edc090eaeb14cfca0b37f6622a95090dc289467aa2af2fb54e589535ad0385`
Status: ok

## [CRITICAL] F4: v8_paper zero paper positions bug (progress_unknown / pp_unpriced)
- **Claim**: v8_paper had opened zero paper positions despite processing 73 real Telegram signals: 33 rejected progress_unknown, 40 rejected pp_unpriced. Root-caused and fixed both.
- **Observed**: The ground-truth summary contains no direct evidence of paper-position open/reject counts, no telegram signal processing counts, and no verification of the described fixes. Separately, funnel.v8 is FIRING CRITICAL with consecutive_failures=6211 and a stuck telegram_received-stage candidate with no terminal disposition (EV005) -- a different but related-sounding funnel/pipeline problem than the one described in the claim.
- **Expected**: If the described bug was fully root-caused and fixed as of this claim's writing, funnel.v8 and related pipeline checks would be expected to show recovery or at least not still be in a long-running CRITICAL/FIRING state with thousands of consecutive failures.
- **Evidence**: EV005
- **Impact**: The claim of a fix for the zero-paper-positions issue cannot be corroborated, and an unrelated but adjacent funnel completion problem (funnel.v8) remains actively firing at very high failure count, suggesting the pipeline area this claim discusses is not fully healthy.
- **Next step**: Query memecoin/data/memecoin_positions.json and trade telemetry logs directly for post-fix paper trade entries, and correlate funnel.v8's stuck candidate (event_id bd6003380054eb57) against the described progress_unknown/pp_unpriced root causes to see if they are related.
- **Confidence**: medium

## [WARN] F1: v8_paper commit/deploy identity
- **Claim**: Committed `07dd982` ... deployed; later, committed `900d47b`, pushed, deployed. Live restart confirmed clean via the same startup log line.
- **Observed**: HEAD is at SHA 346c8c6fc872188f70439330abe50739c5d9fe92 (EV002), which matches all watchdog/job-receipt git_sha fields (EV005). Neither `07dd982` nor `900d47b` appears anywhere in the evidence bundle.
- **Expected**: If these commits were the most recent deploys referenced by the claim, the current HEAD SHA would either be one of them or a descendant reachable from them, and this would be visible/traceable in evidence.
- **Evidence**: EV002, EV005
- **Impact**: Cannot verify from evidence that the fixes described (widened gate wait, curve-fallback pricing) are actually present in the currently deployed code; the claimed commits are unverified against the observed HEAD.
- **Next step**: Run `git log --oneline -20` and `git show 07dd982 900d47b --stat` on the VPS to confirm these commits are ancestors of current HEAD 346c8c6f.
- **Confidence**: medium

## [WARN] F2: Dirty working tree vs. claimed clean deploy
- **Claim**: pushed, pulled on VPS clean (no conflicts)... Stash count verified back to baseline (50) after deploy.
- **Observed**: EV002 shows the working tree is NOT clean: 6 modified tracked files and ~19 untracked paths, including two ad hoc Python scripts (p15_1_2_audit.py, scratch_incident_check.py) at repo root.
- **Expected**: A claim of a 'clean' pull/deploy with stash restored to baseline implies a clean or near-clean working tree afterward, not persistent modified/untracked files.
- **Evidence**: EV002
- **Impact**: Uncommitted local changes on the VPS create risk of divergence between what was tested/reviewed and what is actually running, and could be silently lost or conflict with future deploys.
- **Next step**: Run `git status --porcelain` and `git stash list` on the VPS to confirm actual stash baseline count and diff the modified tracked files against HEAD.
- **Confidence**: high

## [WARN] F5: New test suite claim (test_v8_paper.py, TestCurveFallbackPricing)
- **Claim**: 9 new tests (memecoin/tests/test_v8_paper.py, TestCurveFallbackPricing), 28/28 passing in that file, 685 passing across all v8-related suites.
- **Observed**: EV005 confirms only that tests_memecoin (313 tests) and other suites collect cleanly at the time of the slow run; it does not report pass/fail counts, and does not mention test_v8_paper.py or a 685-count figure specifically.
- **Expected**: A claim of specific pass counts (28/28, 685 total) implies a test-execution report, which is a different signal than the collection-only check evidenced.
- **Evidence**: EV005
- **Impact**: No evidence confirms these tests actually pass as claimed; only that the broader suite collects without import errors, which is a weaker guarantee.
- **Next step**: Run `pytest memecoin/tests/test_v8_paper.py -k TestCurveFallbackPricing -v` and the full v8-related suite to confirm the claimed 28/28 and 685 pass counts.
- **Confidence**: medium

## [INFO] F3: v8_paper startup confirmation log line
- **Claim**: Startup log line from the VPS process confirms: `v8_paper: monitor thread started (interval=5s, gate=CURVE_ACTIVE [V8-P0], config_tag=v8_candidate_2026-09-10_v8p0)`.
- **Observed**: No log content matching this string appears anywhere in the evidence bundle (EV001-EV007); no evidence source contains raw process log lines for v8_paper startup.
- **Expected**: If this log line is being used as deploy confirmation, it should be retrievable/inspectable as evidence, not just asserted in a receipts document.
- **Evidence**: EV005, EV007
- **Impact**: Cannot independently corroborate that the described gate/config was actually active on the running process at the time of these claims.
- **Next step**: Run `journalctl -u quantbot.service | grep 'v8_paper: monitor thread started'` to retrieve and timestamp the actual startup line.
- **Confidence**: medium

## [INFO] F6: executor.py _sol_price_usd cache staleness bug (flagged, not fixed)
- **Claim**: executor.py's _sol_price_usd() cache bumps its 'last updated' timestamp even when the Jupiter fetch fails... flagged here for visibility, not fixed... LIVE_TRADING=false means it isn't actively mispricing anything right now.
- **Observed**: No evidence in the ground-truth bundle confirms or denies the LIVE_TRADING flag state, the existence of this caching bug, or executor.py's current behavior; this is entirely outside the evidence collected (EV001-EV007).
- **Expected**: N/A -- this is an assertion about live-money code with no corroborating or contradicting evidence available.
- **Evidence**: EV005
- **Impact**: A claimed live-pricing correctness bug is documented but unverified; if LIVE_TRADING were ever true, this could silently mask real price-feed outages.
- **Next step**: Grep executor.py for LIVE_TRADING and _sol_price_usd()'s exception handler to confirm the described caching behavior and current flag value.
- **Confidence**: low

## [INFO] F7: Open item: no fresh paper trade landed post-fix
- **Claim**: Still genuinely open: no fresh paper trade has landed under the fixed code yet -- the next real Telegram alert will confirm end-to-end.
- **Observed**: Consistent with an open/unverified state; ground truth shows telegram feed liveness is OK (last telegram_received 916s before check, EV005) but does not show any paper trade landing or its outcome, and funnel.v8 remains FIRING with a stuck telegram_received-stage candidate.
- **Expected**: This claim is self-described as open/unverified, which matches the absence of corroborating evidence in the ground truth -- no disagreement here, just an acknowledged gap.
- **Evidence**: EV005
- **Impact**: The core fix described in the claim remains functionally unverified end-to-end as of the available evidence.
- **Next step**: Monitor memecoin_positions.json and trade_telemetry_summary.csv for the next inbound Telegram signal to confirm a successful paper-position open under the new code path.
- **Confidence**: high
