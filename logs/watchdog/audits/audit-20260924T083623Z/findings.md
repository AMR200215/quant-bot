# Layer 2 Audit: audit-20260924T083623Z
Generated: 2026-09-24T08:36:20Z
Evidence SHA-256: `078e936d82c4c16384e0c0e05a048f7f23e1824da15de2f1194454da3a90cd06`
Status: ok

## [WARN] F2: funnel.v8 incident vs. throttle fix narrative
- **Claim**: Implicit: the 2026-09-23 throttle tightening (60.0s -> 1.0s) was intended to fix missed/late hard_stop detection on rug pulls, addressing a real production gap
- **Observed**: Ground truth shows funnel.v8 incident is CRITICAL and currently FIRING with consecutive_failures: 8546, first_seen 1786748706.89 (well before the 2026-09-23 claimed fix date) and last_seen matching the latest watchdog run — i.e. still actively failing after the claimed fix
- **Expected**: If the throttle fix from the claim were effective, one might expect funnel.v8 to show improvement or recovery, not continuous firing with a rising failure count through the most recent run
- **Evidence**: EV005
- **Impact**: The claimed fix does not appear to have resolved the underlying funnel/detection problem it targeted, or the incident is measuring something orthogonal to the fix -- either way this is worth reconciling before trusting the fix claim.
- **Next step**: Pull the funnel.v8 check's specific failing event_id/mint history since 2026-09-23 and compare hard_stop close timing against the new 1.0s throttle to see if detection latency actually improved.
- **Confidence**: medium

## [WARN] F4: working tree cleanliness vs. claimed deployed change
- **Claim**: Implicit: the throttle change (_ONGOING_FALLBACK_THROTTLE_S 60.0 -> 1.0) in memecoin/v8_paper.py has been deployed as part of the current running code
- **Observed**: Ground truth shows the working tree is not clean, with 5 modified tracked files and 19 untracked paths (EV002), and memecoin/v8_paper.py is not explicitly listed among the modified files shown in evidence
- **Expected**: If this change was deployed and committed as claimed, the current deployed HEAD SHA 9f8635c1... should reflect it, or the file should appear as part of tracked modifications if not yet committed
- **Evidence**: EV002
- **Impact**: Cannot confirm from evidence whether the described throttle code change is actually present in the currently running deployed commit versus still pending/uncommitted.
- **Next step**: Run git show 9f8635c1086e399a4a805bb189616b9b1554868c:memecoin/v8_paper.py | grep -A2 _ONGOING_FALLBACK_THROTTLE_S to confirm the deployed value.
- **Confidence**: medium

## [INFO] F1: receipts_tail content correlation
- **Claim**: EV007 receipts_tail.tail contains the V8_PAPER_FALLBACK_THROTTLE_TIGHTENED entry dated 2026-09-23
- **Observed**: Ground truth summary noted EV007 shows a file of length 278803 bytes with mtime 1790207881.47, matching the v8_vs_v7_daily job receipt timing, but did not have visibility into the actual tail text content now shown in this claim
- **Expected**: The claim text is a narrative changelog entry, not independently verifiable against EV003/EV004/EV005 style structured evidence
- **Evidence**: EV007
- **Impact**: The changelog content itself is unverifiable against independent evidence; only its file metadata (size/mtime) was previously corroborated.
- **Next step**: Diff current memecoin/v8_paper.py:477 _ONGOING_FALLBACK_THROTTLE_S value against the claimed 1.0 to confirm the code change actually matches the narrative.
- **Confidence**: medium

## [INFO] F3: test suite claim
- **Claim**: 52/52 passing in memecoin/tests/test_v8_paper.py (no test asserted the old throttle value)
- **Observed**: Ground truth confirms tests_memecoin suite (335 tests total) collects cleanly with no drift errors (EV005), but has no evidence of a specific 52/52 pass count for test_v8_paper.py, nor confirmation that no test asserts the old throttle value
- **Expected**: The claim provides a specific sub-suite pass count that ground-truth evidence cannot corroborate at that granularity
- **Evidence**: EV005
- **Impact**: Cannot verify the specific test claim; only the aggregate suite collection status is confirmed, which is a weaker guarantee than a targeted pass/fail count.
- **Next step**: Run pytest memecoin/tests/test_v8_paper.py -v and capture the actual pass count to corroborate the 52/52 claim.
- **Confidence**: low

## [INFO] F5: Helius cost/rate-limit claims
- **Claim**: Cost check claims: ~28,500 credits/day, ~2.85%/day of 1M/month free budget, CURVE_BATCH_SIZE=100 covers peak concurrency of 180 in 2 calls, rate ceiling nowhere near 10 req/s
- **Observed**: No evidence in the ground-truth summary (EV001-EV007) covers Helius API usage, credit consumption, or rate-limit metrics; this is entirely outside the audited evidence set
- **Expected**: These are unverifiable operational/cost claims presented with specific numbers but no corroborating monitoring evidence in the audit
- **Evidence**: EV005
- **Impact**: Cost and rate-limit sustainability claims are unaudited; if incorrect, could result in unexpected quota exhaustion or throttling not caught by current watchdog checks.
- **Next step**: Check for a Helius-specific watchdog check or dashboard (none currently identified in EV005 checks) and add one if credit usage is a real operational risk.
- **Confidence**: low
