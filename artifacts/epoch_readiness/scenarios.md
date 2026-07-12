# Epoch Readiness — 30 Scenario Results

> Run: 2026-07-12T00:43:15.557542+00:00Z  |  **30/30 passed**

## Results by Category

### C1_graduation_state (7/7)

| # | Name | Result | ms |
|---|---|---|---|
| 1 | GradState: PP fresh observation selected | ✅ | 5.5 |
| 2 | GradState: stale PP falls back to curve | ✅ | 0.0 |
| 3 | GradState: PP vsol=0 treated as absent | ✅ | 0.0 |
| 4 | GradState: decide NONE below threshold | ✅ | 0.0 |
| 5 | GradState: decide PRE_GRAD_EXIT at threshold | ✅ | 0.0 |
| 6 | GradState: decide GRAD_EXIT when complete=True | ✅ | 0.0 |
| 7 | GradState: decide ALREADY_GONE when account_missing | ✅ | 0.0 |

### C3_exit_orchestrator (9/9)

| # | Name | Result | ms |
|---|---|---|---|
| 8 | Orchestrator: CONFIRMED_SUCCESS | ✅ | 1.7 |
| 9 | Orchestrator: CONFIRMED_REVERT | ✅ | 0.0 |
| 10 | Orchestrator: SENT_PENDING sets global pending (R5) | ✅ | 0.0 |
| 11 | Orchestrator: R5 — second dispatch blocked when SENT_PENDING | ✅ | 0.0 |
| 12 | Orchestrator: R3 — no_route does NOT set pending sig | ✅ | 0.0 |
| 13 | Orchestrator: clear_pending unblocks next dispatch | ✅ | 0.0 |
| 14 | Orchestrator: FATAL_PRE_SEND for build_failed | ✅ | 0.0 |
| 15 | Orchestrator: ZERO_BALANCE outcome | ✅ | 0.0 |
| 16 | Orchestrator: venue_attempts track independently | ✅ | 0.0 |

### C2_entry_gate (5/5)

| # | Name | Result | ms |
|---|---|---|---|
| 17 | EntryGate: SPL token allowed | ✅ | 247.4 |
| 18 | EntryGate: T22 token blocked | ✅ | 0.0 |
| 19 | EntryGate: UNKNOWN program blocked | ✅ | 0.0 |
| 20 | EntryGate: None classification blocked | ✅ | 0.0 |
| 21 | EntryGate: classification with error blocked | ✅ | 0.0 |

### C5_execution_receipts (4/4)

| # | Name | Result | ms |
|---|---|---|---|
| 22 | Receipts: write and read round-trip | ✅ | 2.8 |
| 23 | Receipts: secret fields stripped | ✅ | 0.7 |
| 24 | Receipts: filter by mint | ✅ | 0.7 |
| 25 | Receipts: never raises on None input | ✅ | 0.6 |

### C6_telemetry (5/5)

| # | Name | Result | ms |
|---|---|---|---|
| 26 | Telemetry: UUID-based trace_id (no collisions) | ✅ | 4.5 |
| 27 | Telemetry: timestamp_wall no +00:00Z double suffix | ✅ | 0.9 |
| 28 | Telemetry: secret field redaction | ✅ | 0.9 |
| 29 | Telemetry: bind_position + link_pair API | ✅ | 0.7 |
| 30 | Telemetry: 44+ canonical E_* constants defined | ✅ | 0.0 |
