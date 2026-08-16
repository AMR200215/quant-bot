# Layer 2 Audit: audit-20260816T040523Z
Generated: 2026-08-16T04:05:22Z
Evidence SHA-256: `797d642067a5e9c199be12105f30b7567966c4985d5a4a2a2abcf81d47248a1b`
Status: ok

## [WARN] F3: Path provenance / research data volume (PATH_VOLUME_GAP_UNEXPLAINED)
- **Claim**: 27 real (non-header-only) files total across 14 date-directories for forward/naturalistic paths; 0 files currently present for case-control backfill paths under logs/research_paths/backfill/. Status: PATH_VOLUME_GAP_UNEXPLAINED, a known gap, not resolved.
- **Observed**: Ground truth (EV002) lists logs/price_paths/ and research/data/ as untracked directories but does not evaluate file counts, header-only status, or the backfill subdirectory's contents. This is an unconfirmed but plausible self-reported gap from the claim, not something the evidence bundle checks or contradicts.
- **Expected**: The claim itself acknowledges this as an unresolved known gap, consistent with treating research data volume as under-populated — this is not a discrepancy with ground truth, but an area outside evidence coverage.
- **Evidence**: EV002
- **Impact**: Low data volume for forward-path research constrains the statistical power of any downstream V8 experiment design; the claim's own acknowledgment suggests this should block strategy decisions until resolved.
- **Next step**: Run `find logs/research_paths -type f | wc -l` and `find logs/research_paths/backfill -type f | wc -l` to independently verify the claimed 27-file and 0-file counts.
- **Confidence**: low

## [INFO] F1: Test suite counts
- **Claim**: Full combined suite (memecoin/tests + research/tests + watchdog/tests + watchdog/layer2/tests + tests): 652 passed (up from 641), same 7 pre-existing, unrelated failures as every prior run this session.
- **Observed**: Ground truth only confirms test collection counts (tests_memecoin 288, tests_research 181, tests_watchdog 95, tests_layer2 39, tests_quant-bot 56) with no collection errors (EV005) — collection is not the same as pass/fail execution results, and no pass/fail counts or '7 pre-existing failures' figure appear anywhere in the evidence bundle.
- **Expected**: The claim implies a specific execution result (652 passed, 7 known failures) that should be independently verifiable via a test run receipt or CI log.
- **Evidence**: EV005
- **Impact**: Cannot verify whether the claimed 652-passing / 7-failing state is accurate; if the 7 'pre-existing' failures are misclassified or have grown, this claim could mask a regression.
- **Next step**: Run the full combined test suite locally (pytest memecoin/tests research/tests watchdog/tests watchdog/layer2/tests tests) and diff pass/fail counts against the claimed 652/7 figures.
- **Confidence**: medium

## [INFO] F2: New artifacts (v8_feature_registry.yaml, v8_clean_cohort.py, test_v8_fd_phase1_artifacts.py)
- **Claim**: research/v8_feature_registry.yaml, research/v8_clean_cohort.py, and research/tests/test_v8_fd_phase1_artifacts.py (11 tests) are new artifacts, all passing.
- **Observed**: EV002 shows the working tree has untracked files including research/data/ and other paths, and modified/staged files under docs/ and logs/, but the ground-truth summary does not enumerate research/v8_feature_registry.yaml, research/v8_clean_cohort.py, or research/tests/test_v8_fd_phase1_artifacts.py specifically as present in the untracked/modified file list.
- **Expected**: If these are genuinely new, uncommitted artifacts, they should appear as untracked ('??') entries in EV002's git status.
- **Evidence**: EV002
- **Impact**: Without confirming these specific files exist in the working tree or are committed, the claimed Phase 1 deliverables cannot be verified as actually present on disk.
- **Next step**: Run `git status --porcelain` and `git log --oneline -- research/v8_feature_registry.yaml research/v8_clean_cohort.py research/tests/test_v8_fd_phase1_artifacts.py` to confirm existence and commit state.
- **Confidence**: medium

## [INFO] F4: v8_readiness WARN incident vs FD7 deployability matrix
- **Claim**: FD7 declares several feature classes DEPLOYABLE_NOW or BLOCKED with confident reasoning (e.g., DexScreener/rugcheck fields BLOCKED due to latency; smart_money fields BLOCKED due to Helius constraint).
- **Observed**: Ground truth shows an active, still-firing WARN incident on claims.batch.v8_readiness with 4 of 7 items (N2, N4, N6, N7) PARTIAL, ongoing for ~90 consecutive failures over roughly a day (EV005). The specific content of those PARTIAL items is not detailed in evidence, so it cannot be confirmed whether they correspond to, contradict, or are unrelated to the FD7 deployability classifications in this claim.
- **Expected**: If FD7's classifications are meant to represent the current state of v8_readiness, a fully resolved/complete classification might be expected to correlate with a non-firing v8_readiness check, or the claim should explain the relationship to the ongoing WARN.
- **Evidence**: EV005
- **Impact**: There is a live monitoring signal indicating v8_readiness incompleteness that is not addressed or reconciled anywhere in this claim; readers might assume Phase 1 research work resolves or is disconnected from that operational WARN when the relationship is unconfirmed either way.
- **Next step**: Inspect the watchdog claims.batch.v8_readiness check definition and its N2/N4/N6/N7 item definitions to determine if they map to the FD7 feature classes or cohort work described in this claim.
- **Confidence**: low

## [INFO] F5: Phase-gate status assertion (V8_FD_PHASE1_READY)
- **Claim**: Phase 1 status: V8_FD_PHASE1_READY — no filter was ranked, no threshold was picked, no holdout was touched, and memecoin/v8_paper.py was not modified.
- **Observed**: Ground truth (EV002) shows memecoin/data/memecoin_positions.json and memecoin/data/memecoin_signals.json have uncommitted changes (modified / MM), but does not confirm the state of memecoin/v8_paper.py specifically — no evidence entry names that file's diff status.
- **Expected**: If memecoin/v8_paper.py was genuinely untouched as claimed, this should be verifiable by its absence from any modified/staged file list in git status.
- **Evidence**: EV002
- **Impact**: Cannot independently confirm the claim's core phase-gate guarantee (no trading logic changes) purely from the ground-truth evidence available; this is a self-asserted boundary condition for the research/production separation.
- **Next step**: Run `git diff --stat memecoin/v8_paper.py` and `git log -1 memecoin/v8_paper.py` to confirm no recent modification to this file.
- **Confidence**: medium
