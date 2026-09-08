# Layer 2 Audit: audit-20260908T081316Z
Generated: 2026-09-08T08:13:14Z
Evidence SHA-256: `8950b3ce85bf468d69ce0fe247d3b7337e5bb350686ac960c9b87772958a26ff`
Status: ok

## [WARN] F2: funnel.v8 CRITICAL incident vs THR-BATCH staleness findings
- **Claim**: THR-BATCH T1 pilot found that most admitted tokens have no real trades anywhere near T1m/T3m/T10m comparison points, i.e., real on-chain trading stopped 43-59s before T1m, attributed to thin-liquidity characteristics.
- **Observed**: Ground truth shows an active FIRING CRITICAL incident (funnel.v8, consecutive_failures=3869) where a candidate entered telegram_received and was never assigned a terminal disposition -- the 'V8-TWIN-FIX failure class' -- with no evidence linking this to a thin-liquidity/staleness explanation.
- **Expected**: If the claimed thin-liquidity/staleness confound is the dominant explanation for pipeline anomalies, it should also plausibly relate to why candidates in funnel.v8 never resolve to a terminal state, but no evidence bundle item establishes or rules out this connection.
- **Evidence**: EV005
- **Impact**: Without confirming whether the funnel.v8 stuck-candidate issue is caused by the same thin-liquidity/staleness dynamics described in the claim, the CRITICAL incident's root cause remains unaddressed and unquantified.
- **Next step**: Cross-reference the funnel.v8 stuck event_id (bd6003380054eb57, mint AFf278av...) against on-chain trade activity around its telegram_received timestamp to check for a staleness/thin-liquidity signature.
- **Confidence**: low

## [INFO] F1: receipts_tail content vs EV007 metadata
- **Claim**: V8-P0 and V8-P3 clear SELECTION_DATA_READY (floor-clearing sample); performance table with win rates and pct_change figures shown in receipts_tail.tail
- **Observed**: Ground truth (EV007) only recorded a receipts-tail file's mtime (1788840172.75) and length (235881 bytes), with no content or context available at audit time.
- **Expected**: The claim now supplies the actual tail content, which cannot be independently verified against raw performance data, trade logs, or the SELECTION_DATA_READY threshold definition from the original evidence bundle.
- **Evidence**: EV007
- **Impact**: Performance claims (win rates, pct_change_peak distributions, floor-clearing status) are unverifiable from audit evidence alone and rely entirely on self-reported documentation.
- **Next step**: Pull the raw underlying trade/sample data referenced by V8-P0/V8-P3 and recompute win-rate and pct_change_peak statistics independently of the receipts file.
- **Confidence**: high

## [INFO] F3: Working tree dirtiness vs new research scripts
- **Claim**: research/thr_reconstruct_paths.py is a new module; two pre-existing bugs were found and fixed only in the new module, original files (e.g. backfill_paths.py) unchanged.
- **Observed**: Ground truth (EV002) shows the working tree is not clean, with numerous untracked files/directories including research/data/, research/spool/*.jsonl, p15_1_2_audit.py, and scratch_incident_check.py -- but thr_reconstruct_paths.py specifically is not named in the audited evidence.
- **Expected**: If thr_reconstruct_paths.py is a genuinely new, uncommitted module as claimed, it should appear among the untracked files in EV002's git status, but the ground-truth summary does not confirm its presence or absence explicitly.
- **Evidence**: EV002
- **Impact**: Cannot confirm from audit evidence alone whether the new reconstruction module and its claimed non-invasive bug fixes are actually reflected in the current dirty working tree, or already committed/untracked elsewhere.
- **Next step**: Run `git status --porcelain` and `git diff --stat` and check specifically for research/thr_reconstruct_paths.py and backfill_paths.py to confirm the claimed file-scope of changes.
- **Confidence**: medium

## [INFO] F4: Read-only claim vs SHA/dirty tree discrepancy
- **Claim**: Read-only; no frozen registry touched, no threshold changed, no code change to entry/exit logic.
- **Observed**: Ground truth shows the working tree is dirty with many modified and untracked files (EV002), and a SHA mismatch exists between HEAD and the SHA recorded in the three most recent cron job receipts (EV002 vs EV005), with no evidence explaining either discrepancy.
- **Expected**: A 'read-only, no code change' claim is difficult to reconcile with a working tree containing numerous modified tracked files and untracked scripts, absent evidence distinguishing log/data churn from actual code changes.
- **Evidence**: EV002, EV005
- **Impact**: If the dirty tree includes actual logic changes (not just data/log churn), the 'read-only' assurance in the claim may not hold, creating risk that untracked/modified code could affect production behavior despite the stated safety guarantee.
- **Next step**: Run `git diff` on all modified tracked files and inspect untracked .py files to confirm none alter entry/exit logic or frozen registries, and reconcile HEAD SHA against the cron-recorded SHA.
- **Confidence**: medium

## [INFO] F5: xval pilot / T2 gating decision
- **Claim**: Not yet resolved, held before T2: the xval gate as specified can't yet produce a trustworthy numeric tolerance from available pilot data; flagged to user before proceeding to T2's real Helius-credit-spend run.
- **Observed**: No evidence in the ground-truth audit bundle (EV001-EV007) references THR-BATCH, T1/T2 phases, xval gates, or Helius credit spend at all -- this entire workstream is outside the scope of the audited evidence.
- **Expected**: The claim describes a self-contained research decision/status that the audit evidence neither confirms nor contradicts, since it was never in scope of the watchdog/systemd/cron evidence collected.
- **Evidence**: EV005, EV006
- **Impact**: This claim cannot be validated or invalidated by current audit evidence; treating it as verified would be an overextension of the audit's actual coverage.
- **Next step**: If this claim needs verification, request direct evidence (e.g., xval pilot run logs, reconstructed trade tables) rather than relying on the summary text in RECEIPTS.md.
- **Confidence**: high
