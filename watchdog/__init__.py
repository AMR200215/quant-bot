"""quant-bot watchdog — Layer 1 (deterministic, no-LLM) production truth checks.

See docs/RECEIPTS.md's "WATCHDOG-BATCH" section for the design rationale.
Core rule: detect only. Never restarts services, never repairs cron, never
touches trading. Primary evidence (live process/scheduler/journal state)
over claims (RECEIPTS.md, commit messages, prior findings).
"""

WATCHDOG_VERSION = "1.0.0-phase1"
