# Layer 2 Audit: audit-20260814T180049Z
Generated: 2026-08-14T18:00:47Z
Evidence SHA-256: `20086314cdc0e9b1c400e4ba7285ed2002896f21a4f03906ce2f8d4fd4bcf12f`
Status: ok

## [WARN] F5: memecoin test suite breakage vs Layer 2 claims scope
- **Claim**: Claim text focuses entirely on Layer 2/watchdog code readiness and does not mention the memecoin test collection failure.
- **Observed**: test_drift.collection.tests_memecoin is actively FIRING (WARN, 49 consecutive failures) due to ImportError for SIGNALS_FILE and GRAD_SOL_UI from memecoin.config (EV005) — a real, currently broken part of the codebase unrelated to Layer 2 but co-existing in the same repo state.
- **Expected**: The claim makes no assertion about memecoin test health, so there's no direct disagreement, but the omission is notable given both are part of the same repo/deployment being described as generally sound ('no trading logic touched').
- **Evidence**: EV005
- **Impact**: A reader relying solely on the Layer 2 claim doc could be unaware of a currently-active, unrelated test suite breakage in production code.
- **Next step**: Run `pytest --collect-only memecoin/tests` to confirm the ImportError persists and identify which commit introduced the missing config symbols.
- **Confidence**: high

## [INFO] F1: layer2.staleness check origin
- **Claim**: Layer 1 now checks whether Layer 2 has gone stale (>30h since last successful audit — default WARN); heartbeat committed via GitHub Actions, Layer 1 sees it after VPS's twice-daily git pull.
- **Observed**: layer2.staleness is in SUSPECT state, consecutive_failures: 0, status UNKNOWN, reason: 'no Layer 2 heartbeat file found — either Layer 2 has never run successfully yet, or this VPS hasn't pulled a commit containing one' (EV005).
- **Expected**: The claim's described mechanism (heartbeat file committed by GitHub Actions, picked up via git pull, checked for >30h staleness) is consistent with the check's existence and reason text, but the claim does not resolve which of the two disambiguating explanations applies.
- **Evidence**: EV005
- **Impact**: The check is functioning as designed (it exists and evaluates), but its live utility is unproven since no heartbeat has ever been observed by Layer 1, meaning Layer 2 audits may never have successfully run in production or the deployment/credential step is incomplete.
- **Next step**: Check whether .github/workflows/layer2-audit.yml has ever executed successfully (gh api or Actions UI) and whether the VPS repo has a heartbeat file path present after latest git pull.
- **Confidence**: high

## [INFO] F2: Layer 2 credential/deployment checkpoint
- **Claim**: Anthropic API key and read-only SSH credential are deliberately not provisioned in this batch; deploy/layer2/install.sh is written but not run; GitHub Actions workflow will run once secrets are added.
- **Observed**: No evidence in the ground-truth summary directly confirms or denies whether these credentials/secrets have been added, or whether install.sh has been run — this is outside the scope of any EV item examined.
- **Expected**: Given the claim, Layer 2 audit workflow should currently be non-functional / never fired due to missing secrets, which is consistent with the observed absence of any Layer 2 heartbeat.
- **Evidence**: EV005
- **Impact**: The consistency between the claimed missing-credential state and the observed missing heartbeat lends plausibility to the claim, but neither can be independently verified from evidence collected.
- **Next step**: Inspect GitHub repo secrets configuration (masked) and VPS /etc/sudoers.d or system users for a 'layer2audit' account to confirm install.sh has/has not been run.
- **Confidence**: medium

## [INFO] F3: Layer 2 status label WATCHDOG_CODE_READY vs WATCHDOG_LIVE_VERIFIED
- **Claim**: Status: WATCHDOG_CODE_READY — implementation/tests complete, not live-proven; not yet WATCHDOG_LIVE_VERIFIED.
- **Observed**: Ground truth shows layer2.staleness as UNKNOWN/SUSPECT with no heartbeat ever observed, and no evidence of any Layer 2 audit run, findings, or Telegram alert having occurred. This is consistent with a code-complete-but-not-live-verified state.
- **Expected**: The claim's self-assessment (not yet live-verified) matches the observed absence of any live Layer 2 activity in the evidence.
- **Evidence**: EV005
- **Impact**: No discrepancy — the claim's own caveat matches observed reality, so no false confidence is being introduced by this specific claim.
- **Next step**: No action needed beyond periodic monitoring of layer2.staleness state for eventual transition out of UNKNOWN once/if a heartbeat appears.
- **Confidence**: high

## [INFO] F4: Test suite counts (watchdog)
- **Claim**: 24 new watchdog/layer2/ tests + 5 layer2_staleness tests (126/126 total watchdog tests passing).
- **Observed**: Ground truth (EV005) confirms the 'watchdog' test package collects cleanly with 90 tests reported at collection time; it does not confirm a 126/126 pass count or break down layer2-specific test counts.
- **Expected**: The claim asserts a specific pass count (126/126) that is more granular/different from the 90 collected tests noted in the ground-truth evidence.
- **Evidence**: EV005
- **Impact**: Cannot verify the specific test pass count claimed; discrepancy between 90 (collected, EV005) and 126 (claimed) is unexplained by available evidence and could reflect different test scopes (e.g., collection-only vs full run, or a different point in time).
- **Next step**: Run `pytest watchdog/ --collect-only -q` and `pytest watchdog/ -q` on the current HEAD and compare actual counts against the claimed 126/126.
- **Confidence**: medium

## [INFO] F6: No trading logic touched / LIVE_TRADING=false
- **Claim**: No trading logic touched; LIVE_TRADING=false unchanged throughout.
- **Observed**: Ground truth has no direct evidence confirming or denying the LIVE_TRADING flag value or whether trading logic files were modified — this was not part of the audited EV items.
- **Expected**: The claim asserts an unchanged safety-relevant configuration value that ground truth cannot independently confirm.
- **Evidence**: EV002
- **Impact**: Cannot verify a safety-critical claim about live trading status from available evidence; if false, this would be a significant risk.
- **Next step**: Grep the current working tree and env/config for LIVE_TRADING and diff against last known-good commit to confirm the flag's value and touch history.
- **Confidence**: low

## [INFO] F7: Modified/added files list vs dirty working tree
- **Claim**: Lists specific new/modified files: watchdog/layer2/*, watchdog/checks/layer2_staleness.py, deploy/layer2/*, .github/workflows/layer2-audit.yml, watchdog/checks.yaml, watchdog/runner.py.
- **Observed**: Ground truth (EV002) shows the dirty working tree consists of 5 modified tracked files (docs/RECEIPTS.md, docs/V8_INPUTS.md, logs/memecoin_social_journal.csv, logs/trade_telemetry_summary.csv, memecoin/data/memecoin_signals.json) and 22 untracked files, mostly logs/research/spool artifacts — none of the file paths in ground truth evidence obviously match the claimed layer2 file set by name.
- **Expected**: If the claimed layer2 files were added in this batch, they'd be expected to appear as new/untracked or already-committed files at HEAD; the claim implies these are already committed (given SHA references), which is plausible but not directly confirmed by the specific file list captured in EV002.
- **Evidence**: EV002
- **Impact**: Cannot confirm from evidence alone that the described layer2 files exist at the current working tree state or HEAD commit as claimed.
- **Next step**: Run `git show --stat HEAD -- watchdog/layer2/ deploy/layer2/ .github/workflows/layer2-audit.yml` to confirm these paths exist and were committed at the reported HEAD SHA.
- **Confidence**: medium

## [INFO] F8: SHA discrepancy relevance to Layer 2 claim
- **Claim**: N/A — claim text does not address git SHA discrepancies at all.
- **Observed**: Ground truth flags an unresolved discrepancy between HEAD SHA (3c34b5c8...) and multiple different SHAs recorded in watchdog run metadata/job receipts (EV002, EV005).
- **Expected**: If Layer 2 code was truly committed and wired into watchdog/runner.py as claimed, the SHA under which watchdog last ran (per EV005) should ideally match or postdate the commit introducing layer2_staleness.py; this cannot be confirmed given the SHA inconsistency.
- **Evidence**: EV002, EV005
- **Impact**: The unresolved SHA mismatch adds uncertainty about whether the currently-running watchdog process actually includes the newly claimed Layer 2 code, independent of whether the code exists in the repo.
- **Next step**: Compare `git log --oneline -1` at each of the divergent SHAs to determine if watchdog/checks/layer2_staleness.py exists in the commit the running watchdog process was actually built/started from.
- **Confidence**: medium
