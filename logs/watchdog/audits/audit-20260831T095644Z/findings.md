# Layer 2 Audit: audit-20260831T095644Z
Generated: 2026-08-31T09:56:42Z
Evidence SHA-256: `0c27637056b13714ae81427bc1443ebaabae75c11ef5e825dcd567d37949ad6f`
Status: ok

## [CRITICAL] F3: funnel.v8 active incident vs claim silence
- **Claim**: No claim in the provided documentation addresses the funnel.v8 FIRING incident (CRITICAL, 1555 consecutive failures) or the stuck mint bd6003380054eb57.
- **Observed**: Ground truth shows funnel.v8 is currently in state FIRING with severity CRITICAL, last_seen 1788170105.16, consecutive_failures 1555, and a specific stuck event/mint with no terminal disposition (EV005).
- **Expected**: Given the severity and duration of this incident, one would expect project documentation/receipts to reference or explain it, especially given other, less severe items (e.g., PumpPortal funding) are documented in detail in the claims.
- **Evidence**: EV005
- **Impact**: A CRITICAL, long-running, unresolved funnel failure appears to be undocumented in the reviewed claims, suggesting either an unmonitored blind spot in reporting or that this issue is not being tracked/actioned by the team.
- **Next step**: Query incident tracker or receipts log directly for 'funnel.v8' and event_id bd6003380054eb57 to determine if there is separate documentation not included in this claims excerpt.
- **Confidence**: medium

## [WARN] F2: test suite pass claims vs collection-only checks
- **Claim**: Full suite 498/498 green (receipts) and 26 new tests, full suite green (YD1 claim)
- **Observed**: Ground truth's test_drift.collection.* checks confirm only that test suites collect without import/syntax errors (counts: 304, 531, 103, 39, 56) — these are collection-only checks and do not indicate pass/fail status.
- **Expected**: The claim implies actual test execution results (498/498 passing), which is a stronger and different assertion than collection succeeding.
- **Evidence**: EV005
- **Impact**: The claimed 'green' test suite result cannot be confirmed or refuted by watchdog evidence, since watchdog only checks collection, not execution/pass status; if the claim is inaccurate, defects could be silently present.
- **Next step**: Run the full test suite directly (e.g., pytest -q) and capture the actual pass/fail summary to compare against the claimed 498/498.
- **Confidence**: medium

## [INFO] F1: receipts_tail content vs metadata
- **Claim**: receipts_tail.tail describes E3 test additions, full suite 498/498 green, FROZEN_EXIT_COUNT 3→4, ENGINE_READY reconfirmed True
- **Observed**: Ground truth only confirms receipts file mtime (1788134282.03) and length (229528 bytes) via EV007; no content of the file was independently verified in the evidence bundle.
- **Expected**: The claim asserts specific substantive content (test counts, registry state, engine readiness) that would require reading the file contents to verify.
- **Evidence**: EV007
- **Impact**: Cannot corroborate whether the described E3/registry work actually occurred or matches the described test results; relying on this claim without content verification risks accepting unverified assertions as fact.
- **Next step**: cat or diff the actual receipts.md content against the claimed tail text and cross-check FROZEN_EXIT_COUNT and test suite pass count in the test runner output.
- **Confidence**: high

## [INFO] F4: PumpPortal funding root-cause claim vs feed.pumpportal OK status
- **Claim**: YD3 claims the funded PumpPortal account 'ran out of balance again' around 2026-08-27 15:19-15:30 UTC, causing execution-proxy coverage collapse (down to 45.50% by 08-29), and that this is root-caused as an external balance issue, not a code defect.
- **Observed**: Ground truth shows feed.pumpportal check as OK at snapshot time, with most recent event a successful WS connect at 2026-08-31 09:54:16 (EV005), and a prior WARN incident for feed.pumpportal recovered at 1788032405.09.
- **Expected**: If the funding issue is real and unresolved as of 08-29 per the claim, one might expect either an active incident or a footprint of degraded execution-proxy/path coverage to still show in current evidence as of 08-31, unless it was refunded/resolved between the claim date and snapshot.
- **Evidence**: EV005
- **Impact**: The claimed funding collapse and its described resolution timeline are not independently verifiable from the current evidence snapshot alone; if unresolved, current OK statuses could be masking degraded functionality outside what these specific checks measure.
- **Next step**: Check research/data/path_collection_daily.json for 2026-08-29 through 2026-08-31 ticks/pp_messages counts to confirm whether coverage actually recovered after the claimed re-funding.
- **Confidence**: low

## [INFO] F5: claims.batch.v8_readiness PARTIAL items vs YD-BATCH proposal
- **Claim**: YD2 proposes splitting path_data_ready gate into SELECTION (poll-keyed) and EXIT-derivation (path-keyed) due to low path coverage (~16%) vs high poll-outcome coverage (97-99.8%); not yet adopted.
- **Observed**: Ground truth shows claims.batch.v8_readiness currently has 4 PARTIAL items (N2, N4, N6, N7) out of 7, with substantive content of those items not defined in evidence (EV005).
- **Expected**: If YD2's diagnosis (low path coverage causing readiness gate issues) is correct and unaddressed, it would plausibly explain some or all of the PARTIAL items in v8_readiness, but evidence does not confirm this linkage.
- **Evidence**: EV005
- **Impact**: Without confirming whether the PARTIAL items correspond to the path-coverage issue described in YD2, it's unclear whether the proposed rescope would actually resolve the current WARN incident on claims.batch.v8_readiness.
- **Next step**: Retrieve the detailed content/definitions of N2, N4, N6, N7 items in the v8_readiness batch check to see if they reference path_data_ready or execution-proxy coverage.
- **Confidence**: low

## [INFO] F6: git SHA discrepancy vs claim timeline
- **Claim**: Claims describe dated work entries (2026-08-29 for YD-BATCH) implying a commit history sequence, but do not mention or explain the SHA mismatch between fast watchdog run, slow watchdog run, and rc_closure commit.
- **Observed**: Ground truth documents an unresolved discrepancy between fast-run HEAD (9f63a13b...), slow-run SHA (2036fe8c...), and rc_closure commit (db32f53), with no evidence explaining the relationship (EV002, EV005).
- **Expected**: Given the claims describe ongoing dated work (through 08-29) that would presumably correspond to commits, one might expect the claims to reference specific SHAs or a deploy log entry explaining the sequence, but none is provided.
- **Evidence**: EV002, EV005
- **Impact**: Without SHA-to-work mapping, it is not possible to confirm which of the claimed changes (E3 tests, YD1/YD2/YD3 analyses) are actually present in the currently deployed HEAD versus still pending merge/deploy.
- **Next step**: Run 'git log --oneline -20' and 'git show <SHA>:docs/RECEIPTS.md' for each of the three SHAs to reconcile which commit corresponds to which claimed work item.
- **Confidence**: medium

## [INFO] F7: Working tree dirty state vs claim of 'no frozen registry touched, no threshold changed'
- **Claim**: YD-BATCH states 'no frozen registry touched, no threshold changed, holdout untouched' for the 2026-08-29 analysis work.
- **Observed**: Ground truth shows the working tree is not clean, with 6 modified tracked files and 19 untracked paths including ad-hoc scripts and new log/data directories (EV002), but the specific nature of those changes (whether they touch frozen registry/thresholds) is not detailed in evidence.
- **Expected**: The claim asserts a constrained, read-only nature to the YD-BATCH work; ground truth cannot confirm or deny this because the diff content of the modified/untracked files is not in the evidence bundle.
- **Evidence**: EV002
- **Impact**: Cannot independently verify the claim of a read-only/non-invasive change set; if the modified files include registry or threshold-related data, this claim could be inaccurate.
- **Next step**: Run 'git diff' on the six modified tracked files and inspect untracked files for any changes to registry/threshold-related code or config.
- **Confidence**: low
