# Layer 2 Audit: audit-20260926T084445Z
Generated: 2026-09-26T08:44:43Z
Evidence SHA-256: `50a6ce338b5f6ee836c16bc53de7d18631e4b89b7ba88d4e60826883083e66fa`
Status: ok

## [WARN] F2: funnel.v8 CRITICAL incident vs. claimed V8 fix reliability
- **Claim**: Fix for the observability gap (same commit as the follow-up)... Next time a hard_stop overshoot happens, the real sample cadence will be directly visible.
- **Observed**: EV005 shows funnel.v8 is currently FIRING (CRITICAL), consecutive_failures=9131, with an active candidate (event_id bd6003380054eb57) stuck in telegram_received with no terminal disposition -- a 'silent-disappearance' failure class distinct from the hard_stop overshoot narrative in the claim.
- **Expected**: Claim implies the V8 pipeline's problem areas are being actively fixed and monitored (fallback pricing overshoot), but does not mention or reconcile with the separately-tracked funnel.v8 CRITICAL incident, which appears to be an unrelated, currently-unresolved issue in the same funnel.
- **Evidence**: EV005
- **Impact**: A CRITICAL funnel.v8 incident is actively firing at the time of this audit; documentation claims about V8 fixes do not address this ongoing issue, raising risk that it is either unrecognized or unaddressed by the team.
- **Next step**: Query watchdog incident detail for funnel.v8 fingerprint bd6003380054eb57 and trace mint AFf278av4oRQicFnpeGHFvcXqjmfNac3XaVyVZAhpump through the pipeline logs to determine why it never reached a terminal state.
- **Confidence**: high

## [INFO] F1: V8 fallback price throttle / hard_stop overshoot
- **Claim**: 52/52 passing in memecoin/tests/test_v8_paper.py (no test asserted the old throttle value); follow-up investigation of V8a57ffd overshoot documented in detail.
- **Observed**: Ground truth confirms test collection succeeds (335 tests collected in memecoin package per EV005) but does not verify pass/fail status of any specific test file or test count; no evidence bundle item covers test_v8_paper.py results, hard_stop closes, curve_fallback events, or the V8a57ffd incident.
- **Expected**: Claim asserts a specific passing test count (52/52) and a detailed narrative of a specific trading incident (V8a57ffd overshoot to -91.07%) that ground-truth evidence cannot corroborate.
- **Evidence**: EV005
- **Impact**: The narrative about the fallback-pricing fix and its follow-up validation cannot be independently confirmed from watchdog/system evidence; it rests solely on documentation.
- **Next step**: Run `pytest memecoin/tests/test_v8_paper.py -v` directly and grep journalctl for 'fallback price sample' log lines to confirm the new logging is live and the 52/52 count is current.
- **Confidence**: medium

## [INFO] F3: size_usd bump to $3.0 (V8_PAPER_SIZE_USD_BUMPED)
- **Claim**: Changed size_usd to 3.0 (memecoin/v8_paper.py:587)... 52/52 passing in memecoin/tests/test_v8_paper.py.
- **Observed**: Ground truth shows the working tree is dirty with modifications to memecoin/data/memecoin_signals.json and logs/trade_telemetry_summary.csv (EV002), consistent with recent code/data changes, but no diff content is available to confirm the specific size_usd=3.0 edit or its line number.
- **Expected**: Claim asserts a precise code change (size_usd=1.0 -> 3.0) and test validation that ground truth cannot directly verify given no diff evidence.
- **Evidence**: EV002
- **Impact**: Cannot confirm whether the documented sizing change has actually been deployed/committed or is still part of the uncommitted working-tree changes.
- **Next step**: Run `git diff HEAD -- memecoin/v8_paper.py` and `git log -p -1 -- memecoin/v8_paper.py` to confirm the size_usd change is present and check its commit status.
- **Confidence**: medium

## [INFO] F4: Working tree dirtiness vs. documentation claims
- **Claim**: Both claims describe specific, already-deployed code/doc changes (fallback price logging fix, size_usd bump) as if finalized ("same commit as the follow-up", "Changed to 3.0").
- **Observed**: EV002 confirms the working tree is not clean, including modifications to docs/RECEIPTS.md itself and memecoin data/log files, with no diff content available to distinguish committed vs. uncommitted changes.
- **Expected**: If the claims describe finalized, deployed changes, the working tree should be clean with respect to these files, or the changes should be identifiable in git history at HEAD 032a7e5.
- **Evidence**: EV002
- **Impact**: Documentation may be describing work that is only partially committed or still in a dirty/uncommitted state, creating risk that HEAD does not actually reflect the described fixes.
- **Next step**: Run `git status --porcelain` and `git diff --stat` to enumerate exactly which files are modified and cross-reference against the two claimed changes (v8_paper.py fallback logging, size_usd bump).
- **Confidence**: medium

## [INFO] F5: claims.batch.v8_readiness PARTIAL state vs. claim narrative
- **Claim**: Deliberately not done as part of this same change... needs its own investigation into free-tier accountSubscribe limits/feasibility before committing to it.
- **Observed**: EV005 shows claims.batch.v8_readiness is currently FIRING (WARN), with 4 of 7 items (N2, N4, N6, N7) in PARTIAL state, consecutive_failures=1229, and the watchdog explicitly notes the batch CLI's exit code would report success (0) despite this partial status.
- **Expected**: The claim's tone suggests deliberate, well-scoped incremental progress on V8, but ground truth shows an active, long-running (1229 consecutive failures) WARN-level incident on a V8-readiness batch check that the claim does not address.
- **Evidence**: EV005
- **Impact**: V8 readiness may be less mature than the documentation narrative implies; the partial-state items (N2, N4, N6, N7) are unexplained by any claim text reviewed.
- **Next step**: Inspect the v8_readiness batch check definitions for items N2, N4, N6, N7 and run the batch CLI manually with verbose output to see why they report PARTIAL rather than GREEN.
- **Confidence**: medium

## [INFO] F6: Receipts file freshness vs. claim content
- **Claim**: Both claims appear as tail content of docs/RECEIPTS.md, implying this is the most recent, up-to-date record of changes.
- **Observed**: EV007 shows RECEIPTS.md was last modified at 1790380681.42 (length 282,572 bytes), correlating with the v8_vs_v7_daily cron job's timestamps (EV005), but EV007 provides no content/validity check -- so freshness of the mtime is confirmed but not that the tail content shown is the true final state of the file.
- **Expected**: Claim implicitly asserts this is current, authoritative documentation of the latest V8 changes.
- **Evidence**: EV007, EV005
- **Impact**: Low risk, but the mtime correlation with a cron job (not a manual doc edit) is unexplained and worth clarifying to confirm the doc wasn't touched by an automated process rather than the developer who wrote the claim text.
- **Next step**: Run `git log -1 --format=%cd -- docs/RECEIPTS.md` and compare against the v8_vs_v7_daily job's write behavior to confirm whether the cron job or a manual commit produced the current tail content.
- **Confidence**: low
