# Layer 2 Audit: audit-20260829T101539Z
Generated: 2026-08-29T10:15:37Z
Evidence SHA-256: `529092ff16c5766e2e6d254d1ca4d8f9b8456af4993f27017a43bb2ffec6335a`
Status: ok

## [CRITICAL] F3: funnel.v8 incident vs YD-BATCH root-cause narrative
- **Claim**: YD3 root-causes execution-proxy coverage collapse (72.76%→47.58%→45.50%, 08-27 to 08-29) to PumpPortal account running out of balance again, 'same external root cause as original pre-funding incident,' confirmed via systemctl status, path_collection_daily.json ticks=0 for 08-28, and per-mint trace of 0/127 admitted mints getting ticks.
- **Observed**: Ground truth's active incidents show funnel.v8 CRITICAL FIRING since 1786748706.89 with 975 consecutive failures, reason 'candidate entered telegram_received stage >120s ago with no terminal disposition,' example mint AFf278...pump at ts 1787710096.66 (2026-08-25 timeframe). Ground truth also flagged feed.pumpportal as OK with 'most recent event was a successful WS connect at 2026-08-29 10:14:37' and noted reconnect frequency not evaluated. Ground truth's research.path_collection check showed OK status despite usable_paths:0 for many hours on 2026-08-28, which is consistent with the claimed balance-exhaustion narrative but was NOT flagged as a problem by the check itself in ground truth.
- **Expected**: If the claimed PumpPortal balance exhaustion (08-27 15:19-15:30 UTC onward) is the true root cause of degraded path/tick data, this should manifest as feed.pumpportal or funnel.v8 becoming unhealthy around that window; however feed.pumpportal is reported OK with a successful connect at audit time, and funnel.v8's cited example event is from 08-25, predating the claimed 08-27 balance exhaustion, so the specific funnel.v8 incident evidence does not directly corroborate the claimed root cause.
- **Evidence**: EV005, EV006
- **Impact**: The claim of a fully diagnosed and 'not a code defect' external root cause for degraded data collection cannot be confirmed or refuted by the ground-truth evidence collected; the funnel.v8 CRITICAL incident (975 consecutive failures since 08-25) predates and is not clearly the same issue as the claimed 08-27 PumpPortal balance exhaustion, raising the possibility of two distinct unresolved problems being conflated.
- **Next step**: Query path_collection_daily.json directly for 2026-08-27 through 2026-08-29 tick/pp_messages counts and cross-reference against feed.pumpportal reconnect logs and funnel.v8 candidate timestamps to determine if these are the same incident or separate issues.
- **Confidence**: medium

## [WARN] F2: SHA reconciliation vs claimed 'no frozen registry touched' work
- **Claim**: YD-BATCH claims read-only analysis only, 'no frozen registry touched, no threshold changed, holdout untouched' as of 2026-08-29, alongside prior E3 hard_stop work implying recent commits.
- **Observed**: Ground truth found three distinct, unreconciled SHAs: deployed HEAD 416f3115... (EV002/EV005 runs), job-receipt SHA 3b7d985f... (EV005 recent_job_receipts), and claims.batch.rc_closure commit db32f53 (EV005). No evidence explains which SHA is actually executing in quantbot.service/quantbot-research.service.
- **Expected**: If the claimed work (E3 hard_stop addition, YD1/YD3/YD2 analyses) was truly committed and deployed as described, the deployed HEAD SHA should plausibly correspond to one of the SHAs seen in the rc_closure or job receipts, but no evidence ties the claim's described work to any specific SHA.
- **Evidence**: EV002, EV005
- **Impact**: Cannot confirm that the code currently running matches the state described in the claims (e.g., E3 hard_stop, frozen registry changes) versus an older or newer commit.
- **Next step**: Run `git log -1 --format=%H` on the deployed host and diff against 416f3115..., 3b7d985f..., and db32f53 to determine actual relationship (ancestor/descendant/divergent).
- **Confidence**: medium

## [INFO] F1: receipts_tail content vs EV007 metadata
- **Claim**: EV007 receipts_tail.tail contains detailed narrative about FROZEN_EXIT_COUNT, E3 hard_stop testing, YD-BATCH analyses, and full suite 498/498 green.
- **Observed**: Ground truth EV007 only provided mtime (1787971052.56) and length (229345 bytes) for the receipts file; no content was available to verify at audit time.
- **Expected**: The claim now supplies the actual tail content, which was previously unavailable in evidence.
- **Evidence**: EV007
- **Impact**: The receipts content itself is now visible via claim, but cannot be independently corroborated against other raw evidence (e.g. actual frozen registry SHA256, test suite counts) since no such evidence was collected.
- **Next step**: Pull the full contents of RECEIPTS.md and cross-check the stated 498/498 test count against a live test run output.
- **Confidence**: medium

## [INFO] F4: claims.batch.v8_readiness incident vs YD2 readiness rescope proposal
- **Claim**: YD2 proposes splitting the 'overloaded path_data_ready gate' into SELECTION (poll-keyed) and EXIT-derivation (path-keyed) because path coverage is only ~16% vs poll-based outcome coverage of 97-99.8%, and this is 'awaiting user sign-off, not adopted yet.'
- **Observed**: Ground truth shows claims.batch.v8_readiness incident WARN FIRING since 1786585581.22, 457 consecutive failures, latest slow check status WARN: 'batch v8_readiness: 4/7 item(s) PARTIAL ([N2,N4,N6,N7]).' Ground truth's research.path_collection check also independently showed usable_paths:0 for many hours despite 100% path admission rate, consistent with the low path coverage figures cited in YD2.
- **Expected**: The claim's characterization of low path coverage (~16%) as a known, analyzed, and not-yet-fixed issue is broadly consistent with the ongoing WARN-severity v8_readiness incident and the path-yield gap ground truth independently observed; however ground truth cannot confirm the specific 16% or 97-99.8% figures, nor that N2/N4/N6/N7 PARTIAL items map to the same root cause described in YD2.
- **Evidence**: EV005
- **Impact**: The persistent WARN incident is plausibly explained by the same structural gate-design issue described in the claim, but this linkage is not verified by raw evidence, so the incident should not be assumed resolved or fully understood pending the proposal's adoption.
- **Next step**: Inspect the v8_readiness batch check's N2/N4/N6/N7 item definitions to confirm whether they reference path_data_ready gate logic as described in YD2.
- **Confidence**: low

## [INFO] F5: Full suite test count claim
- **Claim**: Full suite 498/498 green (stated in EV007 tail content) and '26 new tests, full suite green' for YD1.
- **Observed**: Ground truth's test collection evidence (EV005) shows tests_memecoin 304, tests_research 531, tests_watchdog 103, tests_quant-bot 56, tests_layer2 39 collected cleanly with no drift errors -- these are collection counts per suite, not a single combined '498/498' pass/fail run result, and none of these numbers sum to 498.
- **Expected**: If the claim's '498/498 green' refers to a specific test suite or subset run, ground truth evidence does not contain a matching total; the closest evidence is per-suite collection counts from a different check (test_drift, not a pass/fail run).
- **Evidence**: EV005
- **Impact**: Cannot verify the claimed test pass count against any collected evidence; the numbers do not obviously reconcile, though they may refer to different scopes (collection vs. execution, or a specific module).
- **Next step**: Run the actual test suite referenced (likely a memecoin-specific EXIT-strategy suite) and compare the total count to 498 to confirm which subset this refers to.
- **Confidence**: low

## [INFO] F6: quantbot-research service restart claim
- **Claim**: YD3 states 'service auto-restarted 2026-08-28 06:21:51, unrelated to any deploy this session' per systemctl status quantbot-research.
- **Observed**: Ground truth explicitly notes: 'No evidence is provided about quantbot.service or quantbot-research.service uptime/start time, restart counts, or resource usage -- only current ActiveState/SubState/last exec result (EV003).'
- **Expected**: The claim asserts a specific restart timestamp and cause that ground truth's EV003 evidence does not contain any restart-history data to confirm or deny.
- **Evidence**: EV003
- **Impact**: The specific restart timestamp and its attributed cause cannot be independently verified from collected evidence; if the restart is inaccurately dated or attributed, this could mask a deploy-related disruption.
- **Next step**: Run `systemctl status quantbot-research` and `journalctl -u quantbot-research --since '2026-08-28 06:00:00' --until '2026-08-28 06:30:00'` to confirm the restart timestamp and its trigger.
- **Confidence**: medium
