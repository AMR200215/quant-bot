# Layer 2 Audit: audit-20260917T084651Z
Generated: 2026-09-17T08:46:50Z
Evidence SHA-256: `23254646ae52baa79b522bba49319c584a2b37c949c870a57f9a77695736a741`
Status: ok

## [CRITICAL] F1: funnel.v8 incident vs. claimed lifecycle completeness
- **Claim**: Status: the full lifecycle (entry gate → price resolution → durable tracking → exit evaluation → stale-data safety net) has now been systematically reviewed end-to-end... No further code gaps are known at this time.
- **Observed**: The funnel.v8 check is CRITICAL in the most recent watchdog fast run, firing continuously with 6504 consecutive failures since first_seen 1786748706, still active at the fast-run timestamp 1789634704 with no recovered_at. A concrete stuck candidate (event_id bd6003380054eb57) is cited as having entered telegram_received with no terminal disposition.
- **Expected**: If the claimed end-to-end lifecycle review and stale-data safety net (stale_deadman closure, deadman logic, batched fallback) were fully deployed and effective, funnel.v8 would be expected to recover or at least show reduced failures, not remain FIRING with thousands of consecutive failures through the latest run.
- **Evidence**: EV005
- **Impact**: The claimed fix does not appear to have resolved the underlying funnel/exit-tracking issue that watchdog is actively alerting on, meaning live positions may still be getting stuck without terminal disposition despite deploy claims.
- **Next step**: Query the funnel.v8 check detail for event_id bd6003380054eb57 and cross-reference its entry/exit timestamps against commit b94ced5's deploy time to determine whether this candidate predates or postdates the claimed fix.
- **Confidence**: high

## [WARN] F2: git SHA vs. claimed commits
- **Claim**: Committed b94ced5, deployed, live-verified via the close-reason breakdown above. ... Committed b93b9de, deployed, live-verified clean.
- **Observed**: The deployed/reported SHAs in evidence are 4fed9546c98512d89f86e1b620a50ba1c5433de7 (watchdog runs, k5_nightly receipt) and 9c7712daf6f46c5edcf8975ee7aa55727e50dbf0 (epoch_daily and v8_vs_v7_daily receipts). Neither b94ced5 nor b93b9de appears anywhere in the evidence bundle.
- **Expected**: If b94ced5 and b93b9de were the deployed commits referenced by the claims, one would expect the deployed SHA reported in EV002/EV005 to match, or for the working tree to be clean at one of those commits.
- **Evidence**: EV002, EV005
- **Impact**: Cannot verify from evidence alone that the claimed commits are actually what is running in production; the SHA discrepancy noted in the ground truth summary is unexplained and unresolved by these claims.
- **Next step**: Run `git log --oneline -1 b94ced5` and `git log --oneline -1 b93b9de` against the deployed SHAs 4fed9546c9... and 9c7712daf6... to determine ancestry/relationship, and check `git status` on the live host against EV002's modified/untracked file list.
- **Confidence**: medium

## [WARN] F3: Uncommitted working tree vs. claimed clean deploy
- **Claim**: Committed b94ced5, deployed... Committed b93b9de, deployed, live-verified clean (zero errors, zero duplicate open positions on the actual live book post-restart).
- **Observed**: EV002 shows the working tree is NOT clean: 5 modified tracked files and ~19 untracked paths, including two ad-hoc scripts (p15_1_2_audit.py, scratch_incident_check.py) at repo root, at the time of the audit snapshot.
- **Expected**: A claim of clean commit-and-deploy for both b94ced5 and b93b9de would typically imply a clean or near-clean working tree post-deploy, absent unrelated ad-hoc scratch files.
- **Evidence**: EV002
- **Impact**: Uncommitted changes alongside claimed deploys raise the risk that what's running differs from what was reviewed/tested, or that scratch/debug artifacts are mixed into the production checkout.
- **Next step**: Run `git diff` and `git status --porcelain` on the deployed host to enumerate exactly which of the 5 modified tracked files differ from HEAD and whether they relate to the claimed exit-lifecycle or TOCTOU fix.
- **Confidence**: medium

## [INFO] F4: claims.batch.v8_readiness incident vs. claimed lifecycle review
- **Claim**: the full lifecycle (entry gate → price resolution → durable tracking → exit evaluation → stale-data safety net) has now been systematically reviewed end-to-end
- **Observed**: claims.batch.v8_readiness is WARN with 4 of 7 items (N2, N4, N6, N7) PARTIAL, firing continuously (980 consecutive failures, recovered_at: null) as of the slow-run timestamp.
- **Expected**: A completed, systematically reviewed end-to-end lifecycle would be expected to correlate with v8_readiness batch items moving toward GREEN/PASS rather than remaining PARTIAL across multiple items.
- **Evidence**: EV005
- **Impact**: The readiness batch check does not corroborate the claimed completeness of the v8 lifecycle work; items N2/N4/N6/N7 remain unresolved per the watchdog's own readiness scoring.
- **Next step**: Pull the definitions of readiness items N2, N4, N6, N7 from the claims.batch.v8_readiness check source/config to determine whether they map to the entry-gate/price-resolution/exit-evaluation components referenced in the claim.
- **Confidence**: medium

## [INFO] F5: Test counts vs. claimed test additions
- **Claim**: 17 new tests covering the batched fallback... and the deadman logic... 48/48 passing in this file, 705 across all v8-related suites... Added one permanent regression test using a real threading.Event... 5/5 passing standalone.
- **Observed**: EV005's slow-run test-collection drift checks report OK for tests_memecoin (332), tests_research (635), tests_watchdog (103), tests_layer2 (39), tests_quant-bot (56) collected, but these are aggregate collection counts with no breakdown by suite content, file, or the specific '705 v8-related' or '48/48' or '5/5' figures claimed.
- **Expected**: If the claim's specific test counts (17 new, 48/48, 705 total, 5/5 regression) are accurate, they should be reconcilable against or at least not contradicted by the aggregate collection numbers in EV005, but no such reconciliation is possible from evidence alone.
- **Evidence**: EV005
- **Impact**: Cannot independently verify the specific test-count claims from available evidence; the aggregate drift-check numbers are necessary but not sufficient to confirm or refute them.
- **Next step**: Run the actual test suite file(s) referenced (batched fallback / deadman logic tests) with `pytest -k deadman -v --collect-only` and compare counts against the claimed 17/48/705/5 figures.
- **Confidence**: low

## [INFO] F6: Prior Layer 2 audit critical finding vs. claimed fixes
- **Claim**: V8_PAPER_TOCTOU_RACE — fixed 2026-09-17 ... Fixed with an atomic reservation... Post-fix: 5/5 stress runs clean, zero duplicate simultaneous opens, zero exceptions.
- **Observed**: Ground truth notes the prior Layer 2 audit (audit-20260916T084327Z) reported 7 findings including 1 critical, but its content is not in evidence, so it cannot be determined whether that critical finding corresponds to the TOCTOU race or funnel.v8 issue.
- **Expected**: If the TOCTOU race was the critical Layer 2 finding being remediated, one would expect layer2.staleness or a related check to show improvement/recovery correlated with the fix date.
- **Evidence**: EV005
- **Impact**: Without the prior audit's content, it's impossible to confirm the claimed TOCTOU fix addressed the specific critical finding flagged by Layer 2, leaving a traceability gap between claim and evidence.
- **Next step**: Retrieve and diff the findings list from audit-20260916T084327Z against this audit to confirm whether the critical finding referenced there matches the TOCTOU race described in the claim.
- **Confidence**: low

## [INFO] F7: Dry-run vs. live close-reason breakdown discrepancy
- **Claim**: Predicted 167 stale_deadman / 11 hard_stop+time_stop / 2 stay open. Live result on the actual first monitor cycle after deploy: 116 stale_deadman / 52 hard_stop / 10 time_stop / 2 stay open — same total resolved (178/180, matching within the dry-run's rough bucketing)
- **Observed**: No evidence in the audited bundle (EV001-EV007) contains position-level close-reason breakdowns, dry-run predictions, or a 180-position book; this is asserted only in the documentation claim itself.
- **Expected**: A claim of this specificity (167 vs 116 stale_deadman, 11 vs 52+10 hard_stop/time_stop) would ideally be corroborable via a position-book snapshot, logs/trade_telemetry_summary.csv, or a dedicated receipt, none of which is present or referenced in the evidence bundle.
- **Evidence**: EV002
- **Impact**: The large gap between predicted (11) and actual (62 combined hard_stop+time_stop) non-deadman closures is self-described as 'within rough bucketing' but represents a >5x difference, which is unverified against any raw evidence in this audit.
- **Next step**: Inspect logs/trade_telemetry_summary.csv (flagged as modified in EV002) directly for the close-reason distribution on the date of this deploy to independently verify the 116/52/10/2 breakdown.
- **Confidence**: low
