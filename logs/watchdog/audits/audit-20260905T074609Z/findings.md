# Layer 2 Audit: audit-20260905T074609Z
Generated: 2026-09-05T07:46:07Z
Evidence SHA-256: `79c7ebcffd2b0f63ed1f0e1402a5b973a44965d4af9567816f0a9124514eeb00`
Status: ok

## [WARN] F1: funnel.v8 incident vs claimed readiness progress
- **Claim**: YD2 IMPLEMENTED: SELECTION_DATA_READY = True for V8-P0 and V8-P3 for the first time in the project; full local + VPS suites green.
- **Observed**: EV005 shows the `funnel.v8` incident is still FIRING (CRITICAL, consecutive_failures: 2988, last_seen matching the latest fast run) with a candidate stuck in telegram_received with no terminal disposition, described by the watchdog itself as the exact V8-TWIN-FIX failure class. The related `claims.batch.v8_readiness` incident is also FIRING (WARN, 648 consecutive failures) with 4/7 batch members only PARTIAL.
- **Expected**: If SELECTION_DATA_READY were now True and the readiness gate split fully resolved the underlying data-completeness problem, the claims.batch.v8_readiness and/or funnel.v8 incidents would be expected to show recovery or improvement, not continued FIRING with a multi-thousand consecutive-failure count.
- **Evidence**: EV005, EV007
- **Impact**: The documentation claims forward progress on V8 readiness gating, but the live watchdog state shows the core funnel and readiness batch checks are still actively failing, so the claimed fix has not yet resolved (or is unrelated to) the monitored incident conditions.
- **Next step**: Re-run the fast/slow watchdog checks for funnel.v8 and claims.batch.v8_readiness after the claimed YD2 implementation and confirm consecutive_failures resets or state transitions to RECOVERED.
- **Confidence**: medium

## [WARN] F5: Unverifiable claim: 'first module in the project to read pct_change_peak VALUES'
- **Claim**: research/v8_entry_ev_report.py — first module in the project to read pct_change_peak VALUES (not just counts); live run on the VPS, 2026-09-02, with table of win_rate/mean/median/percentiles per candidate.
- **Observed**: No evidence in the ground-truth bundle (EV001-EV007) references v8_entry_ev_report.py, its execution, or any pct_change_peak value distributions. The working tree dirty-file list (EV002) does not enumerate this specific script among modified/untracked files noted in the summary.
- **Expected**: A claim of a live VPS run producing a specific statistical table would ideally be corroborated by a corresponding cron/watchdog run record, log artifact, or file listing in the raw evidence.
- **Evidence**: EV002, EV005
- **Impact**: This is a substantive analytical claim (win rates, percentiles) that cannot be independently corroborated from the audited evidence, so it should be treated as an assertion pending verification, not a confirmed fact.
- **Next step**: Locate and inspect research/v8_entry_ev_report.py output artifacts or logs on the VPS filesystem and cross-check reported win_rate/percentile figures against raw pct_change_peak data.
- **Confidence**: low

## [INFO] F2: Funding-drain re-confirmation claim
- **Claim**: Funding-drain re-confirmed live before proceeding: direct WebSocket test against PumpPortal succeeded (5 real trade messages in 20s), 3,739 real ticks written for one mint in-session, 8,811 real tick rows in prior 24h.
- **Observed**: EV005 shows feed.pumpportal is OK with last connect line timestamped 2026-09-05 07:44:36 UTC, consistent with a live, functioning feed. No per-mint tick counts or 24h aggregate tick counts appear anywhere in the ground-truth evidence bundle (EV001-EV007).
- **Expected**: The claim's specific tick-count figures (3,739; 8,811) would need to be independently verifiable in raw evidence to be confirmed as true rather than merely asserted.
- **Evidence**: EV005
- **Impact**: The general feed-health claim is consistent with observed feed.pumpportal status, but the specific quantitative tick counts cannot be verified from the evidence available, so they remain an unconfirmed assertion.
- **Next step**: Query the raw tick log / research/data store directly for per-mint and 24h tick counts to corroborate the 3,739 and 8,811 figures.
- **Confidence**: medium

## [INFO] F3: Test suite green claims
- **Claim**: Full local + VPS suites green for the V8 entry-EV report module (12 new tests, full local 543 + VPS suite green) and for YD2 IMPLEMENTED changes.
- **Observed**: EV005 confirms all four test suites collect cleanly with counts 304 (memecoin), 543 (research), 103 (watchdog), 56 (layer2/quant-bot root), matching the '543' figure cited in the claim for the research suite.
- **Expected**: Claim's cited research suite count of 543 matches ground truth exactly.
- **Evidence**: EV005
- **Impact**: This specific claimed test count is corroborated by independent evidence, increasing confidence in that portion of the documentation.
- **Next step**: No action needed; consider spot-checking that the 12 new tests referenced are included within the 543 count via pytest --collect-only diff against a prior commit.
- **Confidence**: high

## [INFO] F4: Receipts file freshness vs claim content
- **Claim**: Narrative in EV007 tail describes ongoing readiness engineering work (YD2, V8 entry-EV report) as recent/live-verified activity.
- **Observed**: EV007 shows a tail read of 232503 bytes with mtime 1788566281.61, matching the finished_at timestamp of the v8_vs_v7_daily cron job receipt (EV005), confirming the receipts file was recently written to and is being actively maintained by the pipeline, consistent with active development narrative.
- **Expected**: Claim implies active, ongoing documentation of real work; ground truth confirms the file itself is fresh and tied to a real, recent job execution.
- **Evidence**: EV007, EV005
- **Impact**: The receipts file's recency supports that the narrative entries are being actively appended by the automated pipeline rather than being stale or manually backdated, lending some procedural credibility to the claims.
- **Next step**: Diff the full RECEIPTS.md history against cron job receipt timestamps to confirm each narrative entry aligns with a real job run.
- **Confidence**: medium

## [INFO] F6: No code change / read-only claims
- **Claim**: Read-only; no frozen registry touched, no threshold changed, no code change to entry/exit logic (for v8_entry_ev_report.py); holdout structurally never read.
- **Observed**: Ground truth cannot confirm or deny code-level claims about which files were read vs. modified beyond the dirty/untracked file list in EV002, which does not itemize read-only guarantees or holdout-access patterns.
- **Expected**: Verifying 'holdout never read' or 'no threshold changed' requires source-level static analysis not present in the audited evidence.
- **Evidence**: EV002
- **Impact**: Claims about code safety properties (read-only, holdout isolation) are unverifiable from this evidence set and rely entirely on the documentation's own assertion.
- **Next step**: Run a grep-level static check (as the claim itself describes doing for _compute_diagnostics_feasibility) against v8_entry_ev_report.py to confirm no holdout file paths are referenced.
- **Confidence**: low

## [INFO] F7: Overall consistency of narrative with dirty working tree
- **Claim**: Multiple new/modified files implied by claims (research/v8_readiness_engine.py, research/v8_forward_readiness_report.py, research/v8_entry_ev_report.py, docs/READINESS_RESCOPE_PROPOSAL.md).
- **Observed**: EV002's ground truth confirms the working tree is dirty with modifications to docs/RECEIPTS.md, docs/V8_INPUTS.md, and various research/data files, and untracked files including research/data/ and research/spool/*.jsonl -- broadly consistent with active development matching the claims' subject area, though the exact filenames cited in the claims are not individually enumerated in the summary.
- **Expected**: A dirty tree touching docs and research files is consistent with, but does not prove, the specific implementation claims made.
- **Evidence**: EV002
- **Impact**: The general shape of the dirty tree is plausible corroboration for ongoing README/research work, but does not confirm the specific engine/report modules exist or function as claimed.
- **Next step**: Run git status --porcelain and git diff --stat on the deployed working tree to confirm presence of research/v8_readiness_engine.py and research/v8_entry_ev_report.py.
- **Confidence**: low
