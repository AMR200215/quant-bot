"""watchdog/layer2 — the externally-scheduled LLM audit agent (W12-W17).

Runs from GitHub Actions, not the VPS -- a VPS-wide outage (or a Layer 1
outage) must not also take down the thing supervising it. Evidence
collection is read-only by construction (deploy/layer2/evidence_dump.py,
installed as a forced-command for a dedicated restricted SSH key: the
credential can run that ONE script and nothing else, ever). Detection
only -- this package never modifies code, config, or production state.
"""
