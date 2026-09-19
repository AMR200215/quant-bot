# Layer 2 Audit: audit-20260919T081038Z
Generated: 2026-09-19T08:10:36Z
Evidence SHA-256: `96bcb2270af8ad424433f78ffbcef53ed72f06ce10f8e355830c71ac794c2ce9`
Status: ok

## [WARN] F2: funnel.v8 incident vs. claimed TOCTOU/performance fixes
- **Claim**: Fixed with an atomic reservation... Post-fix: 5/5 stress runs clean, zero duplicate simultaneous opens, zero exceptions... live-verified clean (zero errors, zero duplicate open positions on the actual live book post-restart).
- **Observed**: The `funnel.v8` incident is currently FIRING at CRITICAL severity with 7081 consecutive failures, and the latest fast check shows a candidate stuck at stage `telegram_received` for >120s with no terminal disposition -- described in the evidence itself as 'the exact V8-TWIN-FIX failure class.' This is an active, ongoing funnel-level failure, not a clean state.
- **Expected**: If the concurrency fix and performance fix are genuinely deployed and working as claimed ('live-verified clean'), the funnel.v8 incident should not still be firing at CRITICAL with a stuck candidate matching a known failure class.
- **Evidence**: EV005
- **Impact**: The claimed fixes address a different failure mode (duplicate-open race condition, batch-save performance) than the currently firing funnel.v8 incident (stuck-at-stage telegram_received), so the claims do not contradict the incident directly, but the overall narrative of 'system doing real work cleanly under all six fixes' is undermined by an active CRITICAL funnel incident concurrent with these claims.
- **Next step**: Inspect the funnel.v8 stuck event (event_id bd6003380054eb57, mint AFf278av4oRQicFnpeGHFvcXqjmfNac3XaVyVZAhpump) to determine if it is related to or independent of the alert-reservation fix path described in the claim.
- **Confidence**: medium

## [WARN] F5: working tree cleanliness vs. deploy claims
- **Claim**: Committed b93b9de, deployed... Committed 9589c03, deployed.
- **Observed**: EV002 shows the working tree is NOT clean -- numerous modified tracked files and many untracked files/scripts (including p15_1_2_audit.py, scratch_incident_check.py) exist on top of the deployed commit e0dd2c118a. The evidence does not explain why the tree is dirty.
- **Expected**: A claim of clean, discrete commits ('Committed X, deployed') is easier to trust when the working tree at HEAD is clean; a dirty tree with untracked ad hoc scripts raises the possibility of undocumented, uncommitted changes running alongside or instead of the claimed committed fixes.
- **Evidence**: EV002
- **Impact**: Cannot fully rule out that behavior in production is influenced by untracked/modified files not covered by the claimed commits, weakening confidence that 'deployed' precisely reflects the described fix.
- **Next step**: Run `git diff HEAD` and `git status --porcelain` and review each modified/untracked file to determine if any affect the alert-reservation or batch-close logic described in the claims.
- **Confidence**: medium

## [INFO] F1: git_sha / deployment consistency
- **Claim**: Committed b93b9de, deployed, live-verified clean... Committed 9589c03, deployed.
- **Observed**: EV002 shows deployed HEAD SHA e0dd2c118a9dda2acd874f0f89c900328cbadc17, matching watchdog runs and job receipts. Neither b93b9de nor 9589c03 appear anywhere in the ground-truth evidence bundle, so there is no way to confirm these specific commits are the ones currently deployed or that they are ancestors of the current HEAD.
- **Expected**: If the claims are accurate, the deployed HEAD SHA should be traceable to (or be) one of these commits, or git log should show them in history.
- **Evidence**: EV002, EV005
- **Impact**: Cannot verify from evidence alone that the described concurrency/performance fixes are actually present in the running code; the claim is unverified, not false.
- **Next step**: Run `git log --oneline | grep -E 'b93b9de|9589c03'` and `git show e0dd2c118a --stat` to confirm these commits are ancestors of deployed HEAD.
- **Confidence**: medium

## [INFO] F3: positions.json / book growth claim
- **Claim**: book actively growing (236 total positions vs 194 at the last check, 3 genuinely open and tracking)
- **Observed**: No evidence in the ground-truth bundle (EV001-EV006) contains position counts, positions.json contents, or book size figures. This cannot be confirmed or denied from the audited evidence.
- **Expected**: If true, some corroborating data (e.g. a positions.json snapshot or log entry) would ideally be visible in the evidence bundle.
- **Evidence**: EV002
- **Impact**: No operational risk identified, but the specific numeric claim about book size is unverifiable from the audit evidence and should be treated as an unconfirmed assertion.
- **Next step**: Query positions.json directly (e.g. `jq length positions.json` or equivalent) at audit time to cross-check the claimed counts.
- **Confidence**: low

## [INFO] F4: test suite counts
- **Claim**: 3 new tests... 52/52 in this file, 709 across all v8-related suites.
- **Observed**: EV005's test_drift.* checks confirm collection counts for memecoin (335), research (635), watchdog (103), layer2 (39), and quant-bot (56) tests, but these are collection-only checks (tests can be gathered, not that they pass) and none of these figures map directly to '709 across all v8-related suites' or '52/52 in this file' -- no v8-specific suite breakdown exists in the evidence.
- **Expected**: If accurate, some subset of the collected test counts should be attributable to v8-related suites summing to 709, and passing (not just collected).
- **Evidence**: EV005
- **Impact**: The specific pass/fail claim for 709 v8-related tests cannot be corroborated by the audit evidence, which only confirms collection succeeds for broader, non-v8-specific suite categories.
- **Next step**: Run the actual v8-related test suites (e.g. `pytest -k v8 -v`) and capture pass/fail counts to compare against the claimed 709.
- **Confidence**: low

## [INFO] F6: claims.batch.v8_readiness vs. narrative of system health
- **Claim**: the system is doing real work under all six fixes from 2026-09-12 through today at once
- **Observed**: EV005 shows `claims.batch.v8_readiness` incident is FIRING at WARN severity with 1036 consecutive failures; latest slow check shows only 3/7 GREEN, 4/7 PARTIAL items, and the evidence explicitly notes the underlying CLI would exit 0 despite incompleteness.
- **Expected**: A narrative of the system 'doing real work' under all fixes cleanly would be more consistent with a fully GREEN v8_readiness batch rather than a WARN-firing incident with 4/7 items still PARTIAL.
- **Evidence**: EV005
- **Impact**: Suggests the v8 pipeline has known incompleteness (PARTIAL items) that the narrative in the claims does not mention, potentially masking readiness gaps behind a success-sounding exit code.
- **Next step**: Run the v8_readiness batch CLI manually and inspect items N2, N4, N6, N7 to identify what PARTIAL status specifically indicates.
- **Confidence**: medium
