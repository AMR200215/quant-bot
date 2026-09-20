# Layer 2 Audit: audit-20260920T084305Z
Generated: 2026-09-20T08:43:03Z
Evidence SHA-256: `3a51c502961e5662a0def16249dff88eabe867724e23ac2436294bf862126325`
Status: ok

## [WARN] F3: Live post-restart claim vs. funnel.v8 CRITICAL incident
- **Claim**: Live post-restart: zero errors, zero duplicates, book actively growing (236 total positions vs 194 at the last check, 3 genuinely open and tracking) — the system is doing real work under all six fixes ... at once.
- **Observed**: Ground truth shows funnel.v8 is CRITICAL/FIRING with consecutive_failures=7379, first seen 1786748706, last seen 1789893604, describing a candidate stuck at telegram_received with no terminal disposition, explicitly called 'the exact V8-TWIN-FIX failure class.' This is an active, currently-firing incident on the same system claimed to be running cleanly.
- **Expected**: Claim's characterization of 'zero errors, zero duplicates' and general system health implies no currently active critical funnel/pipeline issues in the v8 path.
- **Evidence**: EV005
- **Impact**: A currently firing CRITICAL funnel.v8 incident directly contradicts or at least is not accounted for by the claimed clean live-verification, suggesting the 'zero errors' claim may only cover the specific batch-close code path and not the broader v8 pipeline.
- **Next step**: Correlate the funnel.v8 stuck event_id (bd6003380054eb57) timeline against the deploy timestamps of commits 9589c03 and 07d09f1 to determine if the stuck candidate predates or postdates these fixes.
- **Confidence**: medium

## [WARN] F5: V8_PAPER_MONITOR_INTERVAL_TIGHTENED claim vs. claims.batch.v8_readiness WARN
- **Claim**: Tightened to 1.0s ... 709 passing ... deployed, live-verified clean (startup log confirms interval=1s; zero errors, zero duplicates post-restart).
- **Observed**: Ground truth shows claims.batch.v8_readiness is WARN/FIRING with consecutive_failures=1063, 4 of 7 items (N2, N4, N6, N7) PARTIAL, and the check notes batch_verify's exit code would be 0/success despite this partial state — i.e., a currently active partial-readiness condition in the v8 claim-verification pipeline that the audit cannot explain the root cause of.
- **Expected**: Claim's 'live-verified clean' framing implies no outstanding partial/warn conditions in v8-related readiness checks.
- **Evidence**: EV005
- **Impact**: An active WARN state in v8_readiness alongside a claim of clean live verification means the two data sources are not fully reconciled; masking exit-code success (noted explicitly by the check) could hide this from casual monitoring.
- **Next step**: Inspect the batch_verify item-level output for N2, N4, N6, N7 to determine whether these PARTIAL items are related to the monitor-interval or batch-close changes.
- **Confidence**: medium

## [INFO] F1: positions.json batch-close performance fix
- **Claim**: Fixed: split `_close()` into `_close_no_save()`/`_finish_close()`/`_close_batch()`; post-fix ~150-300ms vs 89.66s; committed 9589c03, deployed.
- **Observed**: Ground truth has no evidence of positions.json, _close_batch, commit 9589c03, or any performance benchmark for this code path. EV002 shows an uncommitted/dirty working tree but no content diffs, and no git log data confirming this commit is in history.
- **Expected**: Claim implies a specific committed and deployed fix with measured before/after timings exists in the codebase and git history.
- **Evidence**: EV002
- **Impact**: Cannot verify from audit evidence whether this performance fix is actually present in the deployed code, so its claimed 300-500x speedup is unconfirmed by ground truth.
- **Next step**: Run `git log --oneline --grep=9589c03` and `git show 9589c03 --stat` against the deployed SHA, or inspect trade_manager/_close_batch source directly on the running host.
- **Confidence**: medium

## [INFO] F2: Test suite counts
- **Claim**: 52/52 in this file, 709 across all v8-related suites.
- **Observed**: Ground truth (EV005) confirms four test suites collect cleanly (tests_memecoin 335, tests_research 635, tests_watchdog 103, tests_layer2 39, tests_quant-bot 56) but does not break out a 'v8-related suites' subset or a 709 total, nor a 52/52 count for any single file.
- **Expected**: Claim implies a specific, verifiable 709-test aggregate across v8-related suites and a 52/52 pass for the file containing the batch-close tests.
- **Evidence**: EV005
- **Impact**: The specific test-count claims cannot be cross-checked against audit evidence, so pass/fail status of the new batch-close tests is unconfirmed.
- **Next step**: Run `pytest -k v8 --collect-only -q` and `pytest path/to/trade_manager_test_file.py -q` to independently reproduce the 709 and 52/52 counts.
- **Confidence**: medium

## [INFO] F4: Commit SHAs referenced in claims vs. deployed SHA
- **Claim**: Committed `9589c03`, deployed. ... Committed `07d09f1`, deployed.
- **Observed**: Ground truth confirms deployed commit is 1e891c44e9e61521f2f58a83d1080e752d99461d (EV002, EV005), a different (full) SHA. There is no evidence linking short SHAs 9589c03 or 07d09f1 to this deployed commit's ancestry.
- **Expected**: Claim implies these short SHAs are ancestors of or equal to the currently deployed commit.
- **Evidence**: EV002
- **Impact**: Without confirming ancestry, it is unverified whether the deployed build actually contains these two claimed fixes.
- **Next step**: Run `git merge-base --is-ancestor 9589c03 1e891c44e9e61521f2f58a83d1080e752d99461d && git merge-base --is-ancestor 07d09f1 1e891c44e9e61521f2f58a83d1080e752d99461d` to confirm inclusion.
- **Confidence**: medium

## [INFO] F6: PnL analysis (win rate, hard_stop overshoot)
- **Claim**: 26 trades closed since 2026-09-18 deploy: win rate 50%, mean -5.1%, net -$3.99; hard_stop overshoot -56.9%/-59.1% vs -35% target.
- **Observed**: Ground truth contains no trade-level PnL data, win-rate figures, or hard_stop overshoot measurements — no evidence (EV001-EV007) covers trade telemetry content, only that logs/trade_telemetry_summary.csv is a modified tracked file per EV002 with no diff shown.
- **Expected**: Claim implies specific, auditable trade-level statistics are available and correct.
- **Evidence**: EV002
- **Impact**: PnL and overshoot figures cannot be independently corroborated from the audit evidence, so their accuracy is unverified.
- **Next step**: Pull logs/trade_telemetry_summary.csv content and recompute win rate/mean return/net PnL for the 26-trade window directly.
- **Confidence**: medium

## [INFO] F7: Dirty working tree correlation with claimed changes
- **Claim**: Both fixes (batch-close split, monitor interval tightening) are committed and deployed.
- **Observed**: EV002 confirms the working tree is not clean, with modifications to docs/RECEIPTS.md, docs/V8_INPUTS.md, logs/memecoin_social_journal.csv, logs/trade_telemetry_summary.csv, memecoin/data/memecoin_signals.json, plus numerous untracked files — consistent with recent documentation/log updates around these claims, but ground truth cannot confirm the actual code changes (_close_batch, _MONITOR_INTERVAL_S) are committed as claimed since no diff content was available.
- **Expected**: If both fixes are truly committed and deployed as claimed, the working tree would be expected to be clean apart from unrelated artifacts, or the modified docs files would directly correspond to this documentation being added.
- **Evidence**: EV002
- **Impact**: Presence of uncommitted changes to RECEIPTS.md/V8_INPUTS.md alongside these claims is plausible (docs being updated to describe the fix) but cannot be confirmed as the sole explanation, leaving some ambiguity about repo state integrity.
- **Next step**: Run `git diff -- docs/RECEIPTS.md docs/V8_INPUTS.md` to see if uncommitted changes match the claimed fix descriptions verbatim.
- **Confidence**: low
