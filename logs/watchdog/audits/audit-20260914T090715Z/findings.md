# Layer 2 Audit: audit-20260914T090715Z
Generated: 2026-09-14T09:07:13Z
Evidence SHA-256: `339d80214d9c022381d6e8d753becb00ee3e82e23dda94a39d8df90f6ad779cc`
Status: ok

## [CRITICAL] F3: funnel.v8 incident vs claimed fix
- **Claim**: Fixed: widened to 2.3s (paper-only book, zero latency cost)... 28/28 passing... committed, pushed, deployed. Live restart confirmed clean.
- **Observed**: funnel.v8 incident is CRITICAL and FIRING with 5632 consecutive failures, first_seen 1786748706.89 through last_seen 1789376704.76 — an extremely long, still-ongoing failure streak, with the latest fast check flagging a candidate with no terminal disposition (EV005).
- **Expected**: If the progress_unknown/pp_unpriced fixes described in the claim were live and working, the funnel.v8 CRITICAL incident (which tracks exactly this kind of funnel-stage failure) would be expected to show recovery or at least a shrinking failure streak, not a continuing multi-thousand-count firing incident.
- **Evidence**: EV005
- **Impact**: The claimed root-cause fix for paper-trading zero-position bug may not be resolving the underlying funnel failures the watchdog is tracking, or the funnel.v8 check measures something unrelated to the claim — either way this needs reconciliation before trusting the claim's 'fixed' status.
- **Next step**: Inspect funnel.v8 check history/timestamps against the 900d47b deploy time to see if failure rate changed post-deploy; query telegram_received candidates for terminal disposition timing since the fix.
- **Confidence**: medium

## [WARN] F1: git_sha / deploy state
- **Claim**: Committed 900d47b, pushed, deployed (stash/pull/pop...). Live restart confirmed clean via the same startup log line.
- **Observed**: Deployed HEAD SHA is df6d274a7090719bb9693ae9c437fd13eb0d0106 (EV002), which matches watchdog fast/slow run git_sha (EV005). The last cron execution receipts recorded git_sha dd0d4b7b06c6b192531e922663dae1eb268ec10b (EV005), and there is no evidence of either 07dd982 or 900d47b appearing anywhere in the SHA evidence.
- **Expected**: The claim implies the deployed/running commit should correspond to 900d47b (or its ancestor 07dd982), i.e. these specific commit hashes should be reachable from or equal to the currently running HEAD.
- **Evidence**: EV002, EV005, EV007
- **Impact**: Cannot verify from evidence that the fixes described (progress_unknown/pp_unpriced timeout widening, curve-fallback pricing) are actually present in the code currently running on the VPS.
- **Next step**: git log --oneline | grep -E '07dd982|900d47b' on the deployed host and diff against df6d274a to confirm ancestry.
- **Confidence**: medium

## [WARN] F2: working tree cleanliness
- **Claim**: pushed, pulled on VPS clean (no conflicts) ... Stash count verified back to baseline (50) after deploy.
- **Observed**: Working tree is not clean at audit time: 5 tracked files modified and 19 untracked paths present, including two ad-hoc scripts not mentioned in the claim (EV002).
- **Expected**: The claim implies a clean deploy state with stash restored to baseline, i.e. no lingering uncommitted modifications beyond expected runtime-generated logs.
- **Evidence**: EV002
- **Impact**: Uncommitted changes to docs/config/logs make it hard to confirm the deployed code exactly matches what was described as tested and committed, risking silent drift between running code and the claimed fix.
- **Next step**: git status --porcelain and git diff on the 5 modified tracked files to determine if they are runtime-generated artifacts or unreviewed code changes.
- **Confidence**: medium

## [WARN] F6: executor.py live pricing cache bug (flagged, not fixed)
- **Claim**: This is a real, separate latent bug in live-money pricing code (used for real buy/sell fill pricing, not just this paper twin) — flagged here for visibility, not fixed... LIVE_TRADING=false means it isn't actively mispricing anything right now.
- **Observed**: No evidence in the ground-truth audit bundle (EV001-EV007) confirms or denies the LIVE_TRADING flag state, the executor.py caching behavior, or Jupiter fetch failure handling — this is entirely outside the scope of the collected evidence.
- **Expected**: A claim of this severity (live-money mispricing risk) would ideally be corroborated by a config/flag check (e.g., LIVE_TRADING=false confirmed in running environment) in the audit evidence.
- **Evidence**: EV002
- **Impact**: An unverified but self-reported live-money pricing risk exists per documentation; if LIVE_TRADING were ever toggled true without addressing this, real fills could be mispriced silently.
- **Next step**: Check current runtime config/env for LIVE_TRADING value and review executor.py's _sol_price_usd() exception handling directly.
- **Confidence**: low

## [INFO] F4: claims.batch.v8_readiness vs claimed test pass
- **Claim**: 9 new tests ... 28/28 passing in that file, 685 passing across all v8-related suites.
- **Observed**: claims.batch.v8_readiness incident is WARN/FIRING with 898 consecutive failures; the corresponding batch check shows 3/7 GREEN, 4/7 PARTIAL (N2, N4, N6, N7), 0 FAIL (EV005). Ground truth cannot confirm or deny the specific unit test claim (28/28, 685 total) since no test-run evidence for v8_paper specifically was captured.
- **Expected**: The claim's assertion of comprehensive test passing does not by itself explain or resolve the still-firing v8_readiness batch WARN, which evidence shows is unrelated to unit tests and concerns readiness batch items N2/N4/N6/N7.
- **Evidence**: EV005
- **Impact**: Passing unit tests do not guarantee the live readiness batch check is green; the PARTIAL items may reflect a separate real operational gap not addressed by the described code fix.
- **Next step**: Retrieve the detailed reason text for N2/N4/N6/N7 PARTIAL items to determine if they relate to progress_unknown/pp_unpriced or a distinct readiness criterion.
- **Confidence**: medium

## [INFO] F5: v8_paper live verification status
- **Claim**: Still genuinely open: no fresh paper trade has landed under the fixed code yet — the next real Telegram alert will confirm end-to-end.
- **Observed**: Ground truth has no evidence bundle showing v8_paper trade counts, position opens, or Telegram alert processing rates post-deploy; feed.telegram is OK with last event 1703s before check (EV005), consistent with alerts still arriving, but no data on paper positions opened.
- **Expected**: This is an open item per the claim itself and is consistent with (not contradicted by) the ground truth — no ground-truth evidence exists to confirm or deny end-to-end success yet.
- **Evidence**: EV005
- **Impact**: This claim is self-flagged as unverified by its author and remains unverifiable from current evidence; it should not be treated as resolved.
- **Next step**: Query v8_paper position-open logs/telemetry for entries timestamped after the 900d47b deploy to check for nonzero paper positions.
- **Confidence**: high

## [INFO] F7: receipts file identity (EV007)
- **Claim**: N/A - EV007 tail content itself
- **Observed**: Ground truth previously noted EV007 shows only mtime (1789350067.23) and length (262065 bytes) with no filename/path/content given, limiting ability to verify file identity (per original summary's explicit gap).
- **Expected**: The claims text now shown as 'from EV007 (receipts_tail.tail)' supplies narrative content, but ground truth cannot independently confirm this tail content corresponds to the same file whose mtime/length were recorded, since no path was given.
- **Evidence**: EV007
- **Impact**: Cannot fully corroborate that the receipts narrative claims originate from the audited file rather than a different or newer version.
- **Next step**: Retrieve full path and diff of docs/RECEIPTS.md (noted as modified/untracked-adjacent in EV002) against the EV007 tail content to confirm consistency.
- **Confidence**: low
