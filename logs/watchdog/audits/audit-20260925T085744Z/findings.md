# Layer 2 Audit: audit-20260925T085744Z
Generated: 2026-09-25T08:57:43Z
Evidence SHA-256: `a4ba7b4806a7105486c24ea5b24c12a95f3287f65f1836e9d7a3309e0dec7bc4`
Status: ok

## [WARN] F2: funnel.v8 hard_stop overshoot investigation
- **Claim**: Follow-up check, 2026-09-24: 4 new hard_stop closes landed, 3/4 within ~4pp of target; investigated and ruled out batch contention for the V8a57ffd overshoot, attributing it most likely to a fast flash-crash inside a sample gap.
- **Observed**: Ground truth shows funnel.v8 is a CRITICAL, currently FIRING incident with 8842 consecutive failures and at least one candidate (event_id bd6003380054eb57) stuck at telegram_received stage for >120s as of the latest fast run (EV005). The claim's narrative concerns hard_stop/curve_fallback exit pricing, a different mechanism than the stuck-candidate funnel issue currently firing.
- **Expected**: The claim implies V8 pricing/fallback issues are being actively investigated and improving, but does not address or explain the currently firing funnel.v8 incident about candidates stuck at telegram_received.
- **Evidence**: EV005
- **Impact**: The documentation narrative about V8 fixes may be addressing a different failure mode than the active CRITICAL incident, leaving the funnel-stall issue unaddressed and unexplained by the claims.
- **Next step**: Cross-reference event_id bd6003380054eb57 and mint AFf278av4oRQicFnpeGHFvcXqjmfNac3XaVyVZAhpump against the telegram_received pipeline stage to determine if this is related to or distinct from the hard_stop/curve_fallback issue described in RECEIPTS.md.
- **Confidence**: medium

## [INFO] F1: memecoin/v8_paper.py test suite
- **Claim**: 52/52 passing in `memecoin/tests/test_v8_paper.py` (no test asserted the old throttle value).
- **Observed**: Ground truth (EV005) reports memecoin test collection as 335 tests collected cleanly, with no per-file breakdown of test_v8_paper.py or a 52/52 pass count available.
- **Expected**: Claim asserts a specific 52/52 pass count for a single test file, which is a finer-grained assertion than what the audit evidence captured.
- **Evidence**: EV005
- **Impact**: Cannot independently verify the specific test file pass rate claimed in documentation from the audit evidence alone.
- **Next step**: Run `pytest memecoin/tests/test_v8_paper.py -v` and confirm 52/52 passing.
- **Confidence**: medium

## [INFO] F3: docs/RECEIPTS.md dirty working tree
- **Claim**: Documentation describes recent changes (fallback price sample logging fix, size_usd bump to 3.0) as deployed/committed work.
- **Observed**: EV002 shows docs/RECEIPTS.md itself is among the 5 modified tracked files in the dirty working tree at HEAD d222f48a90fca9665f13d733e3c8289a0e9ceeaf, meaning the claims text may not yet be committed or may reflect uncommitted edits.
- **Expected**: Claims presented as settled documentation entries would normally imply the underlying code and doc changes are committed and reflected in the deployed SHA.
- **Evidence**: EV002
- **Impact**: The RECEIPTS.md claims about the size_usd bump and fallback logging fix cannot be confirmed as matching the actual deployed commit state, since the file itself is locally modified and not clean at HEAD.
- **Next step**: Run `git diff HEAD -- docs/RECEIPTS.md` and `git diff HEAD -- memecoin/v8_paper.py` to see if the claimed changes are committed or still pending.
- **Confidence**: medium

## [INFO] F4: size_usd change to memecoin/v8_paper.py:587
- **Claim**: Changed size_usd from 1.0 to 3.0 at memecoin/v8_paper.py:587; only affects positions opened after this deploy; 306 existing journal rows unaffected.
- **Observed**: Ground truth (EV002) shows memecoin/data/memecoin_signals.json and logs/memecoin_social_journal.csv are among the modified tracked files, and memecoin/v8_paper.py is not explicitly listed as modified in EV002's 5 tracked-file diff list.
- **Expected**: If size_usd was changed in memecoin/v8_paper.py as claimed, one would expect that file to appear in the modified-files list or be part of HEAD's committed history explaining the change.
- **Evidence**: EV002
- **Impact**: Cannot confirm from audit evidence whether the size_usd code change described in the claim is actually present in the deployed or working-tree version of v8_paper.py.
- **Next step**: Run `git log -p -- memecoin/v8_paper.py | grep -A3 'size_usd'` and `sed -n '580,595p' memecoin/v8_paper.py` to verify the current value at line 587.
- **Confidence**: low

## [INFO] F5: claims.batch.rc_closure commit mismatch
- **Claim**: Documentation entries imply ongoing, current work reflected at deployed HEAD.
- **Observed**: Ground truth already flags that claims.batch.rc_closure (EV005) references commit db32f53, which does not match deployed HEAD d222f48a90fca9665f13d733e3c8289a0e9ceeaf (EV002), and this is unexplained by any evidence.
- **Expected**: Claims narrative assumes a single coherent deployed state matching the documentation timeline, but the audit found two different commit references without reconciliation.
- **Evidence**: EV002, EV005
- **Impact**: Uncertainty about which commit is authoritative could mean the RECEIPTS.md narrative describes changes not present at the currently running HEAD, or vice versa.
- **Next step**: Run `git log --oneline -1 db32f53` and compare against `git log --oneline -1 d222f48` to determine the relationship between these two SHAs.
- **Confidence**: medium

## [INFO] F6: Observability fix for fallback price sampling
- **Claim**: `_monitor_loop` now logs one INFO line per resolved fallback price sample recording elapsed time since previous fallback attempt, bounded by `_ONGOING_FALLBACK_THROTTLE_S`.
- **Observed**: No evidence in the ground-truth summary (systemd status, watchdog checks, or log-based checks) confirms the presence or absence of this specific log line format or its emission rate.
- **Expected**: If this logging fix were live, it would presumably be visible in recent journalctl output for quantbot.service or quantbot-research.service.
- **Evidence**: EV003
- **Impact**: Cannot verify from audit evidence whether this observability improvement is actually active in the running service.
- **Next step**: Run `journalctl -u quantbot.service --since '-1h' | grep 'fallback price sample'` to confirm the new log line is being emitted.
- **Confidence**: low
