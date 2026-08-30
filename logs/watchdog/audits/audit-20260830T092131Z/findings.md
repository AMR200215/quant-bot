# Layer 2 Audit: audit-20260830T092131Z
Generated: 2026-08-30T09:21:29Z
Evidence SHA-256: `be8490f5ec91ebfbe8b391ef635700471d7f60cbc0c52ad85cb589028c3095d3`
Status: ok

## [WARN] F2: funded PumpPortal account balance / execution-proxy coverage
- **Claim**: YD3 root-causes execution-proxy coverage collapse to PumpPortal balance depletion ('Minimum balance not met for PumpSwap websocket data'), confirmed via path_collection_daily.json ticks=0 for 2026-08-28, unrelated to any deploy.
- **Observed**: Ground truth independently confirms feed.pumpportal is OK with a successful WS connect at 09:19:46 (EV005), and pipeline.research_queue_lag shows gap=0 / consumer caught up (EV005, EV006). Ground truth also separately flagged that research.path_collection shows 100% scheduled-vs-path-files yield for 2026-08-29 but usable_path_yield_pct/ticks_ge1/ticks_ge2/usable_paths are uniformly zero all day, unexplained by the check itself — this is directly consistent with (and corroborates) the claimed balance-depletion root cause, though the evidence bundle alone does not name the WebSocket rejection reason or reference path_collection_daily.json ticks=0 for 08-28 specifically.
- **Expected**: If the claim is accurate, the zero-usable-path anomaly independently observed in ground truth should trace to this external PumpPortal funding issue rather than a code defect.
- **Evidence**: EV005, EV006
- **Impact**: If confirmed, this explains a previously-unexplained ground-truth anomaly (100% path yield but 0 usable paths/ticks) as an external funding issue rather than a pipeline defect, but the current WS connect success shown in feed.pumpportal (09:19:46) suggests the account may have since been reconnected — unclear if balance issue is still active as of the evidence snapshot.
- **Next step**: Check current PumpPortal account balance and query path_collection_daily.json for the most recent date to see if ticks/pp_messages remain at 0, and cross-reference with feed.pumpportal's successful connect timestamp to determine if the funding gap has been resolved.
- **Confidence**: medium

## [WARN] F5: claims.batch.v8_readiness / YD2 readiness rescope proposal
- **Claim**: YD2 proposes splitting path_data_ready gate; claims poll-based outcome coverage is 97-99.8% vs path coverage ~16%, implying current readiness gate (path-keyed) is overly strict and driving the WARN state.
- **Observed**: Ground truth confirms claims.batch.v8_readiness is in an active WARN/FIRING incident state (consecutive_failures: 484, first seen far in the past, still unresolved as of evidence timestamp), with batch v8_readiness at 3/7 GREEN, 4/7 PARTIAL (EV005). This is consistent with the claim's premise that path-based readiness gating is causing persistent PARTIAL/WARN status, but the proposal itself is explicitly 'not adopted yet' and no code change has occurred.
- **Expected**: If the proposal is not yet adopted, the active WARN incident should be expected to persist until either the proposal is implemented or root cause (PumpPortal funding) is resolved — consistent with what evidence shows.
- **Evidence**: EV005
- **Impact**: The active WARN incident on v8_readiness is likely to remain unresolved (as evidence shows it currently is) until either the funding issue is fixed or the proposed gate rescope is implemented and adopted.
- **Next step**: Track claims.batch.v8_readiness state after PumpPortal re-funding and/or after user sign-off on READINESS_RESCOPE_PROPOSAL.md to see if consecutive_failures resets.
- **Confidence**: medium

## [INFO] F1: receipts_tail / test suite claim
- **Claim**: Full suite 498/498 green.
- **Observed**: EV005 test_drift.collection checks show clean collection with nonzero test counts (304, 531, 103, 56) across four suites — these counts do not obviously sum or map to '498/498', and the evidence only confirms collection (no import errors), not pass/fail results.
- **Expected**: Claim implies a full run of 498 tests all passing, which would require pass/fail execution data not present in the evidence bundle.
- **Evidence**: EV005
- **Impact**: Cannot verify the claimed green test suite status from available evidence; the claim may be true but is unconfirmed and unreconcilable with the collection counts shown.
- **Next step**: Run the actual test suite (e.g. pytest -q) and diff reported pass count against the claimed 498/498, and reconcile with the four per-suite collection counts in EV005.
- **Confidence**: medium

## [INFO] F3: top10_holder_pct data availability discrepancy
- **Claim**: top10_holder_pct present=394/411 (95.9%, unexpectedly high vs the registry's stale 7.5% figure — flagged, not investigated further, out of scope).
- **Observed**: No evidence in the ground-truth bundle (EV001-EV007) covers this registry field or its historical 7.5% figure; this is entirely outside the scope of the audited evidence.
- **Expected**: Claim is self-flagged as an anomaly by the authors and explicitly deferred, not asserted as resolved.
- **Evidence**: EV005
- **Impact**: No operational impact from the audit's perspective; noted as an open item the claim authors themselves flagged, unconfirmable either way from evidence.
- **Next step**: Locate the registry documentation referenced in the claim and inspect the historical 7.5% figure's provenance vs the new 95.9% observation.
- **Confidence**: low

## [INFO] F4: SHA/deploy state vs claims of recent research work
- **Claim**: Claims describe active development on 2026-08-29 (YD-BATCH) and reference recent commits/tests, implying a maintained, actively-deployed codebase.
- **Observed**: Ground truth found two different git SHAs reported by different subsystems (fast-run/HEAD: fe6f642..., slow-run/cron receipts: 7addaa1...) and a dirty working tree with untracked ad hoc scripts — the evidence does not confirm which SHA (if either) corresponds to the code described in the claims.
- **Expected**: If claims describe work already merged/deployed, one would expect a clean, single, identifiable SHA consistent across watchdog fast/slow runs and cron jobs.
- **Evidence**: EV002, EV005
- **Impact**: Cannot verify from evidence that the code implementing the claimed changes (new tests, E3 evaluation, YD1/YD3 scripts) is actually the code currently running in production across all subsystems.
- **Next step**: Run 'git log -1 --format=%H' against the checkout paths used by the slow watchdog run and cron jobs, and diff against the fast-run/HEAD SHA to resolve the discrepancy.
- **Confidence**: medium

## [INFO] F6: funnel.v8 CRITICAL incident vs claims
- **Claim**: Claims (YD-BATCH) discuss path predictability, execution-proxy trend, and readiness rescope, but do not mention or address the funnel.v8 CRITICAL incident (candidates stuck at telegram_received stage) that ground truth shows is actively firing.
- **Observed**: Ground truth shows funnel.v8 CRITICAL/FIRING with consecutive_failures: 1256, an ongoing unresolved incident as of the evidence timestamp (EV005), entirely separate from the topics covered in the claims text provided.
- **Expected**: No claim asserts this incident is resolved or being worked; this is simply an omission in the provided claims excerpt relative to what ground truth shows as an active problem.
- **Evidence**: EV005
- **Impact**: The most severe active incident (funnel.v8 CRITICAL) is not addressed by the claims under review, so no independent corroboration or explanation of that incident is available from documentation.
- **Next step**: Check RECEIPTS.md or related docs for any entries specifically addressing funnel.v8 / telegram_received stalls, separate from the YD-BATCH entries reviewed here.
- **Confidence**: high
