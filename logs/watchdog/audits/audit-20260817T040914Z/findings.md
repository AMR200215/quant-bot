# Layer 2 Audit: audit-20260817T040914Z
Generated: 2026-08-17T04:09:12Z
Evidence SHA-256: `8e1e0ef0ef14c0792673f56cd120f72d5eb51dcee1b48b2f236da76a693537b5`
Status: ok

## [WARN] F4: P16-8 Phase 2 foundations — new modules deployment status
- **Claim**: Four new modules (research/v8_feature_enforcement.py, research/v8_candidate_registry.py, research/v8_split.py, research/v8_experiment_manifest.py) exist, are tested, and are deployed. 25 new tests, all passing. Full combined suite: 708 passed.
- **Observed**: Ground truth's EV002 evidence shows the working tree at HEAD de724c7f1... is NOT clean, with numerous untracked files/directories (research/spool/*.jsonl, research/data/, p15_1_2_audit.py, etc.) but does not explicitly enumerate or confirm the presence of the four named modules (v8_feature_enforcement.py, v8_candidate_registry.py, v8_split.py, v8_experiment_manifest.py) as committed/tracked files. EV005's test-collection checks report clean collection for research (229 tests) and others, but the raw evidence summary does not confirm the specific test count of 708 passed / 25 new tests for this batch.
- **Expected**: Claim asserts these four modules exist, are tested and deployed with a specific test count increase (683 -> 708).
- **Evidence**: EV002, EV005
- **Impact**: Cannot confirm from raw evidence that these specific modules are committed/deployed at the audited HEAD SHA, or that the stated test-count increase actually occurred; claim is unverified, not necessarily false.
- **Next step**: git show HEAD:research/v8_feature_enforcement.py (and the other three module paths) to confirm they are tracked and committed at de724c7f1, then re-run the full test suite and diff the count against the claimed 708.
- **Confidence**: medium

## [WARN] F5: SELECTION_DATA_READY status vs v8_readiness batch check
- **Claim**: SELECTION_DATA_READY: still not true for anything beyond crude entry-only comparisons... Unchanged from Phase 1.5's own assessment.
- **Observed**: Ground truth shows claims.batch.v8_readiness is in an active FIRING incident state (first_seen 1786585581, still firing as of latest slow run, consecutive_failures=118), with 4 of 7 batch items (N2, N4, N6, N7) in PARTIAL state — this is the only currently active/unresolved incident in the whole watchdog evidence set.
- **Expected**: The claim's self-assessed 'not ready' status for SELECTION_DATA_READY is directionally consistent with the ground truth's FIRING/PARTIAL v8_readiness signal, suggesting the documentation's honesty about incompleteness aligns with the operational evidence.
- **Evidence**: EV005
- **Impact**: The claim's own caveat is corroborated by hard evidence of an unresolved WARN/FIRING batch-verification incident, reinforcing that this is a genuine, not just a disclosed, gap.
- **Next step**: Inspect the N2/N4/N6/N7 batch item PARTIAL states directly to confirm they correspond to the same gaps (exit registry, replay interface, execution cost model) named in the claim as 'not attempted.'
- **Confidence**: medium

## [INFO] F1: docs/RECEIPTS.md content vs metadata
- **Claim**: Real admission log entry observed within seconds of restart: {"token_address": "AM84wet...", "utc_hour": 23, "path_eligible": true, "path_admitted": true, ...}
- **Observed**: EV007 only provides mtime (1786924681.59) and length (175,125 bytes) for docs/RECEIPTS.md; no content or diff was available to the ground-truth audit, and this mtime aligns with a v8_vs_v7_daily scheduler receipt write, not specifically with an admission-controller log event.
- **Expected**: The claim asserts specific structured log content (a JSON admission record) was written into/observed via RECEIPTS.md around the same time.
- **Evidence**: EV007, EV005
- **Impact**: Cannot independently verify the admission-controller behavior described; the claim may be accurate but is unconfirmed by raw evidence available to this audit.
- **Next step**: grep the actual docs/RECEIPTS.md content for the AM84wet... token_address entry and cross-check its timestamp against the claimed 'seconds after restart' window.
- **Confidence**: medium

## [INFO] F2: research.path_collection / hourly concentration
- **Claim**: utc_hour: 23 admission event example, consistent with ongoing path collection work
- **Observed**: EV005's research.path_collection check independently reports for 2026-08-16 that all activity is concentrated in UTC hour 23, with 6 tokens scheduled and 4 path files (66.7% yield).
- **Expected**: Claim's example admission log entry at utc_hour 23 is consistent with this pattern.
- **Evidence**: EV005
- **Impact**: This specific detail in the claim aligns with ground-truth evidence, increasing confidence in that part of the narrative.
- **Next step**: Confirm the AM84wet... token appears in the day's path_collection_daily.json output alongside the other 5 scheduled tokens.
- **Confidence**: medium

## [INFO] F3: sampled_admit/sampled_reject branch (probabilistic decay)
- **Claim**: The probabilistic-decay branch (sampled_admit/sampled_reject) can only be observed once real hourly budget pressure actually builds — not forceable, not yet observed as of this commit.
- **Observed**: No evidence in the ground-truth summary (EV001-EV007) confirms or denies budget-pressure events, sampled_admit/sampled_reject occurrences, or hourly budget saturation; research.path_collection shows budget usage of 10,122 of 100,000 messages, well under capacity, consistent with the claim's premise that pressure hasn't built.
- **Expected**: Claim is a forward-looking honest limitation, not a completion assertion; ground truth's low budget usage figure is consistent with (does not contradict) this being unobserved so far.
- **Evidence**: EV005
- **Impact**: No immediate risk; this is a disclosed known-gap in test coverage of the admission controller, consistent with low observed budget pressure.
- **Next step**: Monitor path_collection_daily.json's hourly breakdown for future sampled_admit/sampled_reject entries once budget usage approaches capacity.
- **Confidence**: medium

## [INFO] F6: ENGINE_DESIGN_READY status vs test/deployment evidence
- **Claim**: ENGINE_DESIGN_READY: substantially advanced. Feature enforcement, candidate registry, grouped chronological split, and experiment manifest all exist, are tested, and are deployed.
- **Observed**: No raw evidence item directly confirms or denies the existence/deployment of these four specific files at the audited commit; EV002 shows an unclean working tree with many untracked files, which is at minimum consistent with recent, not-yet-fully-committed work matching this description.
- **Expected**: Claim implies these modules are already committed and integrated (deployed), not merely present as untracked/local files.
- **Evidence**: EV002
- **Impact**: If these modules are among the untracked files noted in EV002, calling them 'deployed' may overstate their integration/production status pending a commit.
- **Next step**: git status --porcelain and git log -- research/v8_feature_enforcement.py research/v8_candidate_registry.py research/v8_split.py research/v8_experiment_manifest.py to determine tracked/committed status.
- **Confidence**: low

## [INFO] F7: PostgREST schema-cache reload / venue_state_at_signal
- **Claim**: Two items still need the user directly: the PostgREST schema-cache reload (NOTIFY pgrst, 'reload schema';) for venue_state_at_signal, and — no new ask this batch — everything else proceeds under existing authorization.
- **Observed**: No evidence in the ground-truth summary (EV001-EV007) references PostgREST, schema caches, or venue_state_at_signal at all.
- **Expected**: Claim asserts an outstanding user action item related to database schema cache; ground truth is silent on this entirely.
- **Evidence**: EV005
- **Impact**: Cannot verify whether this pending action has been completed or is still outstanding; represents an unconfirmed open item.
- **Next step**: Query the PostgREST/PostgreSQL instance directly for schema cache state on venue_state_at_signal, or check for a corresponding NOTIFY log entry.
- **Confidence**: low

## [INFO] F8: Overall claim-to-evidence reconciliation
- **Claim**: General narrative of P16-8 progress, honest limitations, and status split as required
- **Observed**: None of the ground-truth evidence (EV001-EV007) directly contradicts the claims; the claims largely concern application-level feature work (admission controller, Phase 2 foundations) that lies outside the scope of what EV001-EV007 (git/systemd/cron/watchdog/file-metadata evidence) directly measure, except for the v8_readiness FIRING incident which aligns with the claim's own stated incompleteness.
- **Expected**: Claims should be checkable against operational evidence where overlap exists (e.g., test counts, deployment state, commit status).
- **Evidence**: EV002, EV005, EV007
- **Impact**: Most narrative claims are neither confirmed nor refuted by the ground-truth audit; only the FIRING v8_readiness incident provides direct corroboration of the claim's own admitted limitations.
- **Next step**: Re-run audit evidence collection (EV002 git diff, EV005 full watchdog check list, test suite execution) after these claims are made to directly verify test counts and commit state.
- **Confidence**: medium
