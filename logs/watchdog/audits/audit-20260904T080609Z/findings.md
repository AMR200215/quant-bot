# Layer 2 Audit: audit-20260904T080609Z
Generated: 2026-09-04T08:06:06Z
Evidence SHA-256: `06b9dcb32021abd9a84d3697756bc41f2a99697f2c9da6f16aa2786abb9c0247`
Status: ok

## [WARN] F2: funnel.v8 incident vs SELECTION_DATA_READY claim
- **Claim**: SELECTION_DATA_READY = True for V8-P0 and V8-P3 for the first time in the project; full local + VPS suites green.
- **Observed**: The audit found `funnel.v8` CRITICAL, firing continuously since 1786748706 through 1788509105 (2700 consecutive failures), with a candidate stuck at stage telegram_received with no terminal disposition, described as 'the exact V8-TWIN-FIX failure class' (EV005).
- **Expected**: If SELECTION_DATA_READY was newly achieved and gates cleared as claimed, one might expect some improvement or resolution in the V8 funnel's failure state, or at least no contradiction with an ongoing critical funnel failure.
- **Evidence**: EV005
- **Impact**: The readiness-gate claim and the live funnel incident are not necessarily contradictory (readiness concerns data sufficiency for selection, not funnel/ingestion health), but the juxtaposition should be verified so a 'green' claim doesn't mask an unrelated but still critical pipeline failure.
- **Next step**: Check whether `funnel.v8` incident and `selection_data_ready`/`exit_derivation_data_ready` gates are causally linked or fully independent subsystems; review research/v8_readiness_engine.py logic against funnel.v8's telegram_received stall.
- **Confidence**: medium

## [WARN] F3: claims.batch.v8_readiness vs YD2 IMPLEMENTED claim
- **Claim**: YD2 IMPLEMENTED: gate split live-verified, SELECTION_DATA_READY = True for V8-P0 and V8-P3.
- **Observed**: `claims.batch.v8_readiness` incident is WARN, firing continuously since 1786585581 (621 consecutive failures), with 4 of 7 items (N2, N4, N6, N7) PARTIAL, 3 GREEN, 0 FAIL, and evidence explicitly notes the batch-verify CLI would exit 0 despite this partial state (EV005).
- **Expected**: If the YD2 gate-split and SELECTION_DATA_READY claims are fully implemented and verified as stated, one might expect the v8_readiness claims batch to show fewer PARTIAL items or a corresponding recovery, rather than a persistent WARN incident spanning before and after the claimed implementation window.
- **Evidence**: EV005
- **Impact**: There is an unresolved tension between the documentation's assertion of a completed, live-verified readiness milestone and the watchdog's own batch-verification check showing partial (not fully green) status; risk of overstating completeness in reporting.
- **Next step**: Inspect the v8_readiness claims-batch detail for items N2, N4, N6, N7 to determine whether they map to SELECTION vs EXIT-derivation gates, and whether PARTIAL is expected post-split or indicates incomplete rollout.
- **Confidence**: medium

## [WARN] F5: feed.telegram ambiguity vs claim silence
- **Claim**: No claim text directly addresses feed.telegram staleness/dead-connection ambiguity.
- **Observed**: Ground truth flags `feed.telegram` CRITICAL, firing since 1787300101 with a brief recovery then re-firing by last_seen 1788509105, and explicitly notes the evidence does not resolve whether this is a quiet channel or dead connection (EV005).
- **Expected**: Given the receipts tail discusses funding-drain and readiness work extensively, one might expect some mention of the concurrently firing feed.telegram incident if it were considered significant, but the claims are silent on it.
- **Evidence**: EV005
- **Impact**: A currently firing CRITICAL incident (feed.telegram) is not addressed in the latest documentation narrative, which may mean it is either considered non-critical/already understood, or simply not yet written up — this gap should not be read as resolution.
- **Next step**: Check for any Telegram-side channel activity logs or connection-health metrics independent of the watchdog check to determine if the silence is intentional or an oversight.
- **Confidence**: medium

## [INFO] F1: receipts_tail content vs receipts file metadata
- **Claim**: Receipts tail describes YD2 implementation, SELECTION_DATA_READY live-verification, funding-drain re-confirmation, and V8 entry-EV report content.
- **Observed**: Ground truth (EV007) only established mtime (1788479882.16) and length (232412 bytes) of the receipts file, matching a v8_vs_v7_daily receipt's started_at timestamp; the actual textual content was explicitly noted as not available in the evidence bundle at audit time.
- **Expected**: The claim now supplies the actual tail content of RECEIPTS.md, which was an unresolved gap in the ground-truth audit.
- **Evidence**: EV007
- **Impact**: The claim fills a previously flagged evidence gap but cannot be independently corroborated by other ground-truth evidence (e.g. no code diff, test run logs, or VPS session evidence was in the original bundle).
- **Next step**: Cross-check research/v8_readiness_engine.py, research/v8_forward_readiness_report.py, and research/v8_entry_ev_report.py existence/diffs against git log and run the referenced test suites to confirm the 543/12-new-test claims.
- **Confidence**: low

## [INFO] F4: Funding-drain re-confirmation claim
- **Claim**: Direct WebSocket test against PumpPortal succeeded (5 real trade messages in 20s); 3,739 real ticks written for one mint in-session; 8,811 real tick rows written in prior 24h.
- **Observed**: Ground truth shows `feed.pumpportal` incident as RECOVERED with a non-null recovered_at timestamp (EV005); no tick-count or WebSocket-test evidence was present in the original evidence bundle.
- **Expected**: The claim's specific tick counts and live WebSocket test are consistent with a recovered pumpportal feed state, but are not independently verifiable from the ground-truth evidence collected.
- **Evidence**: EV005
- **Impact**: The claim is plausible given the RECOVERED status but represents new, unverified operational detail beyond what the audit evidence directly supports.
- **Next step**: Query the tick-log storage (e.g., logs/memecoin or research/data tick tables) directly for row counts matching the claimed 3,739 and 8,811 figures and timestamps.
- **Confidence**: low

## [INFO] F6: V8 entry-EV report table vs ground truth
- **Claim**: V8-P0 n=1145, win_rate 67.1%, mean +178.0%; V8-P3 n=458, win_rate 62.2%, mean +158.7%; both clear SELECTION_DATA_READY at floor-clearing sample sizes; read-only, no registry/threshold/code changes to entry/exit logic.
- **Observed**: No evidence in the ground-truth bundle (EV001-EV007 as summarized) directly reports pct_change_peak values, win rates, or entry-EV statistics; the audit's `claims.batch.v8_readiness` only reports GREEN/PARTIAL/FAIL counts, not the underlying statistical detail.
- **Expected**: The claim provides granular statistical detail that the ground-truth evidence bundle does not contain and therefore cannot confirm or refute.
- **Evidence**: EV005
- **Impact**: These specific performance figures are unverified assertions; treating them as fact for decision-making (e.g., readiness sign-off) without independent recomputation risks acting on unaudited numbers.
- **Next step**: Re-run research/v8_entry_ev_report.py against the same train+validation dataset and diff the output against the claimed table.
- **Confidence**: low

## [INFO] F7: rc_closure commit mismatch — unaffected by claims
- **Claim**: No claim addresses the db32f53 vs deployed HEAD SHA discrepancy noted in ground truth.
- **Observed**: Ground truth flagged that `claims.batch.rc_closure` is tied to commit `db32f53`, which differs from the deployed HEAD SHA (456460e908f5a0b1fd338754747b77af15c19a71) in EV002/EV005, and this discrepancy remains unexplained (EV005, EV002).
- **Expected**: If the receipts narrative reflects the most current state of the repo including all described implementations (YD2, entry-EV report, etc.), one might expect commit references to be reconciled or at least mentioned, but the claims text does not address this mismatch.
- **Evidence**: EV002, EV005
- **Impact**: Unclear commit provenance for rc_closure verification could mean readiness/closure claims are validated against a stale commit rather than the currently deployed code.
- **Next step**: Run `git log --oneline -1 db32f53` and compare to deployed HEAD to determine how many commits separate the rc_closure verification point from current deployment.
- **Confidence**: medium
