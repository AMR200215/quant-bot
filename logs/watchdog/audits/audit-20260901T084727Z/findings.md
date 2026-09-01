# Layer 2 Audit: audit-20260901T084727Z
Generated: 2026-09-01T08:47:25Z
Evidence SHA-256: `7810867ece1b52220492d17a944be836fb7cad80855454589b1fe3369a6d1cd4`
Status: ok

## [WARN] F2: funnel.v8 CRITICAL incident vs claimed 'root-caused' PumpPortal balance issue
- **Claim**: YD3: execution-proxy coverage collapse root-caused to PumpPortal account running out of balance again ('Minimum balance not met for PumpSwap websocket data'), described as 'Not a code defect' and 'User acknowledged, will re-fund' — implying the issue is understood/handled, not an open unresolved incident.
- **Observed**: Ground truth shows funnel.v8 is CRITICAL and FIRING with consecutive_failures=1832, still firing as of the latest fast check, matching 'V8-TWIN-FIX failure class' with no terminal disposition for at least one candidate. claims.batch.v8_readiness is also WARN/FIRING with 538 consecutive failures. Evidence bundle contains no operator acknowledgment, ticket, or remediation status for either incident.
- **Expected**: If the claim's narrative is accurate and the balance issue is the root cause with user acknowledgment/re-funding pending, one might expect the funnel.v8 and v8_readiness incidents to reflect a known, being-addressed state rather than appearing as raw ongoing CRITICAL/WARN firings with no annotation.
- **Evidence**: EV005
- **Impact**: The documentation claims the issue is understood and being remediated by the user, but the watchdog evidence shows it as an unacknowledged, still-firing, escalating-failure-count incident; operators relying on the doc claim alone might under-prioritize an actually active critical incident.
- **Next step**: Query PumpPortal account balance/status directly and cross-check watchdog incident annotations or ticket system for any linkage to the YD3 root-cause narrative.
- **Confidence**: medium

## [INFO] F1: receipts_tail content vs receipts file evidence
- **Claim**: EV007 receipts_tail.tail describes E3 hard_stop candidate work, YD-BATCH analysis (YD1/YD2/YD3), full suite 498/498 green, 26 new tests, etc.
- **Observed**: Ground truth EV007 only established mtime (1788220681.66) and length (229,620 bytes) of the receipts file, matching the v8_vs_v7_daily job finished_at. No content of the receipts file was in the original evidence bundle.
- **Expected**: The claim supplies detailed narrative content allegedly from that same receipts file/tail.
- **Evidence**: EV007, EV005
- **Impact**: The narrative content of the receipts tail cannot be independently verified against ground-truth evidence; it is asserted, not proven, so downstream conclusions drawn solely from this text should be treated as unverified.
- **Next step**: Fetch and diff the actual docs/RECEIPTS.md content (e.g. tail -c 5000 docs/RECEIPTS.md) and compare against this claimed text for consistency.
- **Confidence**: high

## [INFO] F3: Full test suite count discrepancy
- **Claim**: Receipts tail claims 'Full suite 498/498 green' after E3 addition, and YD1 claims '26 new tests, full suite green.'
- **Observed**: Ground truth EV005 test collection shows 5 separate suites: tests_memecoin (304), tests_research (531), tests_watchdog (103), tests_layer2 (39), tests_quant-bot (56) — none matching 498, and no single 'full suite' total of 498 is present in evidence.
- **Expected**: If claim is accurate, some suite or aggregate should total 498 tests matching evidence-based collection counts.
- **Evidence**: EV005
- **Impact**: Without reconciling test suite naming/counts, the claimed '498/498 green' figure cannot be corroborated against the actual test collection evidence, leaving test-health claims unverifiable.
- **Next step**: Run the full test suite locally (e.g. pytest --collect-only -q) and sum reported counts to check against both the 498 figure and EV005's per-suite breakdown.
- **Confidence**: medium

## [INFO] F4: Commit/SHA reference ambiguity
- **Claim**: Receipts narrative describes dated work (2026-08-29 YD-BATCH, frozen registry changes, E3 addition) but does not state a commit SHA.
- **Observed**: Ground truth notes deployed HEAD SHA is e3f1b0872dba5545956620e101cbf2d452d23d44, while claims.batch.rc_closure in EV005 references a different commit db32f53, with no evidenced relationship between the two, and working tree is dirty with untracked/modified files including docs/RECEIPTS.md itself.
- **Expected**: For claimed work (frozen registry changes, new tests) to be considered 'landed' and reflected in the current deployed state, one would expect it to correspond to a specific commit consistent with or ancestor to HEAD.
- **Evidence**: EV002, EV005
- **Impact**: Since docs/RECEIPTS.md is itself listed as modified in the dirty working tree, the claimed narrative may not correspond to any committed, deployed state, making it impossible to confirm the claimed changes are actually live.
- **Next step**: Run git log --oneline -- docs/RECEIPTS.md and git diff HEAD -- docs/RECEIPTS.md to determine whether the claimed content is committed or only staged/uncommitted.
- **Confidence**: medium

## [INFO] F5: path_collection yield vs YD-BATCH population figures
- **Claim**: YD1/YD2 reference a 'funded-era admitted population' of n=411 and n=427/994/405, tied to path/outcome coverage analysis dated 2026-08-29.
- **Observed**: Ground truth EV005 research.path_collection reports a different, specific figure: 2026-08-31, 120 tokens scheduled, 111 path files produced (92.5% yield), PumpPortal budget 47,524/100,000 — explicitly a single-day metric, not a cumulative funded-era population count.
- **Expected**: The claim's larger cumulative population figures (411, 427, 994, 405) are a different metric scope than the single-day figure in ground truth; no evidence confirms or contradicts the cumulative figures directly.
- **Evidence**: EV005
- **Impact**: The cumulative funded-era statistics in the claims cannot be cross-checked against the only population-level evidence available (a single day's yield), so their accuracy is unverified.
- **Next step**: Query the research population dataset directly (e.g. the source used by v8_path_predictability.py) to confirm n=411/427 figures independently of the single-day path_collection metric.
- **Confidence**: low

## [INFO] F6: No contradiction found on core deployed SHA and dirty tree
- **Claim**: N/A - general consistency check
- **Observed**: Ground truth deployed SHA e3f1b087... and dirty working tree (including docs/RECEIPTS.md as modified) are consistent with EV007's receipts file having a recent mtime matching a cron job's finished_at, and with the claims text plausibly being the freshly-written/edited content of that dirty file.
- **Expected**: No contradiction; this is simply confirming plausible consistency between file mtime timing and claim content being recently authored.
- **Evidence**: EV002, EV007, EV005
- **Impact**: No corrective action needed for this specific point; it supports (without proving) that the claims text is recently written material, consistent with the dirty tree state.
- **Next step**: No action required; retain as supporting context only.
- **Confidence**: medium
