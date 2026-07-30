"""
memecoin/tools/run_epoch_readiness_sim.py — Epoch readiness simulation.

Runs 30 end-to-end scenario tests against production modules:
  - graduation_state.py (C1)
  - exit_orchestrator.py (C3)
  - evaluate_live_entry_program_gate() in portfolio.py (C2)
  - execution_receipts.py (C5)
  - telemetry.py (C6)

All 30 must pass. Outputs:
  artifacts/epoch_readiness/scenarios.json — machine-readable results
  artifacts/epoch_readiness/scenarios.md   — human-readable report

Usage:
    cd /path/to/quant-bot && python memecoin/tools/run_epoch_readiness_sim.py
    python memecoin/tools/run_epoch_readiness_sim.py --fail-fast
"""

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

REPO = Path(__file__).parent.parent.parent
ARTIFACTS = REPO / "artifacts" / "epoch_readiness"

# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    id: int
    name: str
    category: str
    passed: bool
    error: str = ""
    elapsed_ms: float = 0.0
    details: dict = field(default_factory=dict)


RESULTS: list[ScenarioResult] = []


def scenario(id_: int, name: str, category: str):
    """Decorator: register a scenario function."""
    def decorator(fn):
        fn._scenario_id = id_
        fn._scenario_name = name
        fn._scenario_category = category
        SCENARIOS.append(fn)
        return fn
    return decorator


SCENARIOS = []


def run_all(fail_fast: bool = False) -> list[ScenarioResult]:
    results = []
    for fn in sorted(SCENARIOS, key=lambda f: f._scenario_id):
        t0 = time.monotonic()
        try:
            details = fn() or {}
            result = ScenarioResult(
                id=fn._scenario_id,
                name=fn._scenario_name,
                category=fn._scenario_category,
                passed=True,
                elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
                details=details if isinstance(details, dict) else {},
            )
        except Exception as exc:
            result = ScenarioResult(
                id=fn._scenario_id,
                name=fn._scenario_name,
                category=fn._scenario_category,
                passed=False,
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                elapsed_ms=round((time.monotonic() - t0) * 1000, 1),
            )
            if fail_fast:
                results.append(result)
                return results
        results.append(result)
    return results


# ===========================================================================
# CATEGORY A: graduation_state.py (C1) — 7 scenarios
# ===========================================================================

@scenario(1, "GradState: PP fresh observation selected", "C1_graduation_state")
def s01():
    from memecoin.graduation_state import ProgressObservation, select_progress_observation
    now = time.monotonic()
    pp = ProgressObservation("mintA", "pp", now - 1.0, 50.0, False, "normal", 0.0)
    cv = ProgressObservation("mintA", "curve", now - 2.0, 60.0, False, "normal", 0.0)
    result = select_progress_observation(pp, cv, now, freshness_sec=5.0)
    assert result is pp, f"Expected PP, got {result}"
    return {"selected_source": result.source}


@scenario(2, "GradState: stale PP falls back to curve", "C1_graduation_state")
def s02():
    from memecoin.graduation_state import ProgressObservation, select_progress_observation
    now = time.monotonic()
    pp = ProgressObservation("mintB", "pp", now - 10.0, 50.0, False, "normal", 0.0)  # stale
    cv = ProgressObservation("mintB", "curve", now - 1.0, 60.0, False, "normal", 0.0)
    result = select_progress_observation(pp, cv, now, freshness_sec=5.0)
    assert result is cv, f"Expected curve, got {result}"
    return {"selected_source": result.source}


@scenario(3, "GradState: PP vsol=0 treated as absent", "C1_graduation_state")
def s03():
    from memecoin.graduation_state import ProgressObservation, select_progress_observation
    now = time.monotonic()
    pp = ProgressObservation("mintC", "pp", now - 1.0, 0.0, False, "normal", 0.0)  # vsol=0
    cv = ProgressObservation("mintC", "curve", now - 1.0, 70.0, False, "normal", 0.0)
    result = select_progress_observation(pp, cv, now, freshness_sec=5.0)
    assert result is cv, f"Expected curve (PP vsol=0), got {result}"
    return {"selected_source": result.source}


@scenario(4, "GradState: decide NONE below threshold", "C1_graduation_state")
def s04():
    from memecoin.graduation_state import ProgressObservation, GraduationAction, decide_graduation_action
    obs = ProgressObservation("mintD", "pp", time.monotonic(), 50.0, False, "normal", 0.0)
    action = decide_graduation_action(obs, grad_sol_ui=115.0, pregrad_trigger_pct=0.85)
    assert action == GraduationAction.NONE, f"Expected NONE, got {action}"
    return {"action": action.value}


@scenario(5, "GradState: decide PRE_GRAD_EXIT at threshold", "C1_graduation_state")
def s05():
    from memecoin.graduation_state import ProgressObservation, GraduationAction, decide_graduation_action
    trigger = 115.0 * 0.85  # = 97.75
    obs = ProgressObservation("mintE", "pp", time.monotonic(), trigger, False, "normal", 0.0)
    action = decide_graduation_action(obs, grad_sol_ui=115.0, pregrad_trigger_pct=0.85)
    assert action == GraduationAction.PRE_GRAD_EXIT, f"Expected PRE_GRAD_EXIT, got {action}"
    return {"action": action.value, "vsol": trigger}


@scenario(6, "GradState: decide GRAD_EXIT when complete=True", "C1_graduation_state")
def s06():
    from memecoin.graduation_state import ProgressObservation, GraduationAction, decide_graduation_action
    obs = ProgressObservation("mintF", "pp", time.monotonic(), 200.0, True, "curve_complete", 0.0)
    action = decide_graduation_action(obs, grad_sol_ui=115.0, pregrad_trigger_pct=0.85)
    assert action == GraduationAction.GRAD_EXIT, f"Expected GRAD_EXIT, got {action}"
    return {"action": action.value}


@scenario(7, "GradState: decide ALREADY_GONE when account_missing", "C1_graduation_state")
def s07():
    from memecoin.graduation_state import ProgressObservation, GraduationAction, decide_graduation_action
    obs = ProgressObservation("mintG", "pp", time.monotonic(), 0.0, False, "account_missing", 0.0)
    action = decide_graduation_action(obs, grad_sol_ui=115.0, pregrad_trigger_pct=0.85)
    assert action == GraduationAction.ALREADY_GONE, f"Expected ALREADY_GONE, got {action}"
    return {"action": action.value}


# ===========================================================================
# CATEGORY B: exit_orchestrator.py (C3) — 9 scenarios
# ===========================================================================

@scenario(8, "Orchestrator: CONFIRMED_SUCCESS", "C3_exit_orchestrator")
def s08():
    from memecoin.exit_orchestrator import ExitOrchestrator, RouteOutcome
    orch = ExitOrchestrator("pos_s08")
    result = orch.dispatch("pump_amm", lambda: {"success": True, "sig": "sig1",
                           "fill_price": 0.0001, "sol_received": 0.01,
                           "error_class": None})
    assert result.outcome == RouteOutcome.CONFIRMED_SUCCESS, f"Got {result.outcome}"
    return {"outcome": result.outcome.value, "sig": result.sig}


@scenario(9, "Orchestrator: CONFIRMED_REVERT", "C3_exit_orchestrator")
def s09():
    from memecoin.exit_orchestrator import ExitOrchestrator, RouteOutcome
    orch = ExitOrchestrator("pos_s09")
    result = orch.dispatch("pump_amm", lambda: {"success": False, "sig": None,
                           "error_class": "revert", "fill_price": None,
                           "sol_received": None})
    assert result.outcome == RouteOutcome.CONFIRMED_REVERT, f"Got {result.outcome}"
    return {"outcome": result.outcome.value}


@scenario(10, "Orchestrator: SENT_PENDING sets global pending (R5)", "C3_exit_orchestrator")
def s10():
    from memecoin.exit_orchestrator import ExitOrchestrator, RouteOutcome
    orch = ExitOrchestrator("pos_s10")
    result = orch.dispatch("pump_amm", lambda: {"success": False, "sig": "pend_sig",
                           "error_class": "pending", "fill_price": None,
                           "sol_received": None})
    assert result.outcome == RouteOutcome.SENT_PENDING, f"Got {result.outcome}"
    assert orch.pending_sig() == "pend_sig", "pending_sig not set"
    return {"outcome": result.outcome.value, "pending_sig": orch.pending_sig()}


@scenario(11, "Orchestrator: R5 — second dispatch blocked when SENT_PENDING", "C3_exit_orchestrator")
def s11():
    from memecoin.exit_orchestrator import ExitOrchestrator, RouteOutcome
    orch = ExitOrchestrator("pos_s11")
    orch.dispatch("pump_amm", lambda: {"success": False, "sig": "pend_sig2",
                  "error_class": "pending", "fill_price": None, "sol_received": None})
    result2 = orch.dispatch("jupiter", lambda: {"success": True, "sig": "sig_jup",
                            "error_class": None, "fill_price": 0.0001, "sol_received": 0.01})
    assert result2.outcome == RouteOutcome.NO_SEND, f"Expected NO_SEND (R5 block), got {result2.outcome}"
    return {"outcome": result2.outcome.value}


@scenario(12, "Orchestrator: R3 — no_route does NOT set pending sig", "C3_exit_orchestrator")
def s12():
    from memecoin.exit_orchestrator import ExitOrchestrator, RouteOutcome
    orch = ExitOrchestrator("pos_s12")
    result = orch.dispatch("jupiter", lambda: {"success": False, "sig": None,
                           "error_class": "no_route", "fill_price": None, "sol_received": None})
    assert result.outcome == RouteOutcome.NO_SEND, f"Expected NO_SEND for no_route, got {result.outcome}"
    assert orch.pending_sig() is None, "no_route must NOT set pending sig (R3)"
    return {"outcome": result.outcome.value, "pending_after": orch.pending_sig()}


@scenario(13, "Orchestrator: clear_pending unblocks next dispatch", "C3_exit_orchestrator")
def s13():
    from memecoin.exit_orchestrator import ExitOrchestrator, RouteOutcome
    orch = ExitOrchestrator("pos_s13")
    orch.dispatch("pump_amm", lambda: {"success": False, "sig": "sig_pend",
                  "error_class": "pending", "fill_price": None, "sol_received": None})
    orch.clear_pending()
    result = orch.dispatch("jupiter", lambda: {"success": True, "sig": "sig_jup2",
                           "error_class": None, "fill_price": 0.00009, "sol_received": 0.009})
    assert result.outcome == RouteOutcome.CONFIRMED_SUCCESS, f"Expected SUCCESS after clear, got {result.outcome}"
    return {"outcome": result.outcome.value}


@scenario(14, "Orchestrator: FATAL_PRE_SEND for build_failed", "C3_exit_orchestrator")
def s14():
    from memecoin.exit_orchestrator import ExitOrchestrator, RouteOutcome
    orch = ExitOrchestrator("pos_s14")
    result = orch.dispatch("pump_amm", lambda: {"success": False, "sig": None,
                           "error_class": "build_failed", "fill_price": None, "sol_received": None})
    assert result.outcome == RouteOutcome.FATAL_PRE_SEND, f"Got {result.outcome}"
    return {"outcome": result.outcome.value}


@scenario(15, "Orchestrator: ZERO_BALANCE outcome", "C3_exit_orchestrator")
def s15():
    from memecoin.exit_orchestrator import ExitOrchestrator, RouteOutcome
    orch = ExitOrchestrator("pos_s15")
    result = orch.dispatch("pump_amm", lambda: {"success": False, "sig": None,
                           "error_class": "zero_balance", "fill_price": None, "sol_received": None})
    assert result.outcome == RouteOutcome.ZERO_BALANCE, f"Got {result.outcome}"
    return {"outcome": result.outcome.value}


@scenario(16, "Orchestrator: venue_attempts track independently", "C3_exit_orchestrator")
def s16():
    from memecoin.exit_orchestrator import ExitOrchestrator
    orch = ExitOrchestrator("pos_s16")
    for _ in range(3):
        orch.dispatch("pump_amm", lambda: {"success": False, "sig": None,
                      "error_class": "revert", "fill_price": None, "sol_received": None})
    orch.dispatch("jupiter", lambda: {"success": False, "sig": None,
                  "error_class": "revert", "fill_price": None, "sol_received": None})
    assert orch.venue_attempts("pump_amm") == 3, f"pump_amm attempts={orch.venue_attempts('pump_amm')}"
    assert orch.venue_attempts("jupiter") == 1, f"jupiter attempts={orch.venue_attempts('jupiter')}"
    return {"pump_amm_attempts": orch.venue_attempts("pump_amm"),
            "jupiter_attempts": orch.venue_attempts("jupiter")}


# ===========================================================================
# CATEGORY C: entry program gate (C2) — 5 scenarios
# ===========================================================================

@scenario(17, "EntryGate: SPL token allowed", "C2_entry_gate")
def s17():
    from memecoin.portfolio import evaluate_live_entry_program_gate
    from memecoin.mint_classifier import MintClassification
    cls = MintClassification(
        mint="mintSPL",
        mint_owner_program="TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
        token_program="SPL",
        token_extensions=[],
        unsupported_extensions=[],
        transfer_hook_present=False,
        transfer_fee_present=False,
        policy_category="standard",
        detection_source="rpc",
        detection_timestamp_wall="2026-07-12T00:00:00Z",
        detection_timestamp_monotonic=0.0,
        rpc_commitment="confirmed",
        classification_version="1.0",
        error=None,
    )
    result = evaluate_live_entry_program_gate(cls, curve_observation=None)
    assert result["allowed"] is True, f"SPL must be allowed: {result}"
    return result


@scenario(18, "EntryGate: T22 token blocked", "C2_entry_gate")
def s18():
    from memecoin.portfolio import evaluate_live_entry_program_gate
    from memecoin.mint_classifier import MintClassification
    cls = MintClassification(
        mint="mintT22",
        mint_owner_program="TokenzQdBNbLqP5VEhdkAS6EPFsNP3Qf9EKaPg4HLkx",
        token_program="T22",
        token_extensions=["transfer_fee_config"],
        unsupported_extensions=[],
        transfer_hook_present=False,
        transfer_fee_present=True,
        policy_category="transfer_fee",
        detection_source="rpc",
        detection_timestamp_wall="2026-07-12T00:00:00Z",
        detection_timestamp_monotonic=0.0,
        rpc_commitment="confirmed",
        classification_version="1.0",
        error=None,
    )
    result = evaluate_live_entry_program_gate(cls, curve_observation=None)
    assert result["allowed"] is False, f"T22 must be blocked: {result}"
    assert result.get("token_program") == "T22", f"Expected token_program=T22: {result}"
    return result


@scenario(19, "EntryGate: UNKNOWN program blocked", "C2_entry_gate")
def s19():
    from memecoin.portfolio import evaluate_live_entry_program_gate
    from memecoin.mint_classifier import MintClassification
    cls = MintClassification(
        mint="mintUNK",
        mint_owner_program="",
        token_program="UNKNOWN",
        token_extensions=[],
        unsupported_extensions=[],
        transfer_hook_present=False,
        transfer_fee_present=False,
        policy_category="unknown",
        detection_source="rpc",
        detection_timestamp_wall="2026-07-12T00:00:00Z",
        detection_timestamp_monotonic=0.0,
        rpc_commitment="confirmed",
        classification_version="1.0",
        error=None,
    )
    result = evaluate_live_entry_program_gate(cls, curve_observation=None)
    assert result["allowed"] is False, f"UNKNOWN must be blocked: {result}"
    return result


@scenario(20, "EntryGate: None classification blocked", "C2_entry_gate")
def s20():
    from memecoin.portfolio import evaluate_live_entry_program_gate
    result = evaluate_live_entry_program_gate(None, curve_observation=None)
    assert result["allowed"] is False, f"None classification must be blocked: {result}"
    return result


@scenario(21, "EntryGate: classification with error blocked", "C2_entry_gate")
def s21():
    from memecoin.portfolio import evaluate_live_entry_program_gate
    from memecoin.mint_classifier import MintClassification
    cls = MintClassification(
        mint="mintERR",
        mint_owner_program="",
        token_program="UNKNOWN",
        token_extensions=[],
        unsupported_extensions=[],
        transfer_hook_present=False,
        transfer_fee_present=False,
        policy_category="unknown",
        detection_source="rpc",
        detection_timestamp_wall="2026-07-12T00:00:00Z",
        detection_timestamp_monotonic=0.0,
        rpc_commitment="confirmed",
        classification_version="1.0",
        error="RPC timeout",
    )
    result = evaluate_live_entry_program_gate(cls, curve_observation=None)
    assert result["allowed"] is False, f"Error classification must be blocked: {result}"
    return result


# ===========================================================================
# CATEGORY D: execution_receipts.py (C5) — 4 scenarios
# ===========================================================================

@scenario(22, "Receipts: write and read round-trip", "C5_execution_receipts")
def s22():
    from memecoin.execution_receipts import write_receipt, read_receipts
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        write_receipt({"sig": "TESTSIG", "action": "buy", "mint": "mintR",
                       "sol_amount": 0.01}, _path_override=path)
        results = read_receipts(_path_override=path)
        assert len(results) == 1, f"Expected 1 receipt, got {len(results)}"
        assert results[0]["sig"] == "TESTSIG"
        return {"receipts_count": len(results)}
    finally:
        os.unlink(path)


@scenario(23, "Receipts: secret fields stripped", "C5_execution_receipts")
def s23():
    from memecoin.execution_receipts import write_receipt, read_receipts
    import json
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        write_receipt({"sig": "S2", "action": "sell", "private_key": "SECRET",
                       "sol_amount": 0.05}, _path_override=path)
        with open(path) as f:
            content = f.read()
        assert "SECRET" not in content, "private_key must be stripped"
        assert "sol_amount" in content, "non-secret field must remain"
        return {"secret_stripped": True}
    finally:
        os.unlink(path)


@scenario(24, "Receipts: filter by mint", "C5_execution_receipts")
def s24():
    from memecoin.execution_receipts import write_receipt, read_receipts
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        write_receipt({"sig": "S3", "action": "buy", "mint": "mintA"}, _path_override=path)
        write_receipt({"sig": "S4", "action": "buy", "mint": "mintB"}, _path_override=path)
        results = read_receipts(mint="mintA", _path_override=path)
        assert len(results) == 1 and results[0]["mint"] == "mintA"
        return {"filtered_count": len(results)}
    finally:
        os.unlink(path)


@scenario(25, "Receipts: never raises on None input", "C5_execution_receipts")
def s25():
    from memecoin.execution_receipts import write_receipt
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        write_receipt(None, _path_override=path)  # type: ignore
        write_receipt({}, _path_override=path)
        return {"raised": False}
    finally:
        os.unlink(path)


# ===========================================================================
# CATEGORY E: telemetry.py (C6) — 5 scenarios
# ===========================================================================

def _tmp_telemetry_path():
    import tempfile
    f = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    f.close()
    return f.name


@scenario(26, "Telemetry: UUID-based trace_id (no collisions)", "C6_telemetry")
def s26():
    import memecoin.telemetry as T
    path = _tmp_telemetry_path()
    orig = T.TELEMETRY_FILE
    T.TELEMETRY_FILE = path
    try:
        ids = {T.start_trace(f"pos_{i}", "mint", "TKN", "live") for i in range(5)}
        for tid in ids:
            T.finish_trace(tid)
        assert len(ids) == 5, "All trace_ids must be unique"
        for tid in ids:
            assert tid.startswith("tr_") and len(tid) == 35, f"Bad format: {tid}"
        return {"trace_ids_unique": True, "count": 5}
    finally:
        T.TELEMETRY_FILE = orig
        os.unlink(path)


@scenario(27, "Telemetry: timestamp_wall no +00:00Z double suffix", "C6_telemetry")
def s27():
    import json
    import memecoin.telemetry as T
    path = _tmp_telemetry_path()
    orig = T.TELEMETRY_FILE
    T.TELEMETRY_FILE = path
    try:
        tid = T.start_trace("pos27", "mint27", "TKN", "live")
        T.finish_trace(tid)
        with open(path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        for rec in lines:
            wall = rec.get("timestamp_wall", "")
            assert not wall.endswith("+00:00Z"), f"Double suffix: {wall!r}"
            assert wall.endswith("Z"), f"Missing Z suffix: {wall!r}"
        return {"timestamps_checked": len(lines)}
    finally:
        T.TELEMETRY_FILE = orig
        os.unlink(path)


@scenario(28, "Telemetry: secret field redaction", "C6_telemetry")
def s28():
    import memecoin.telemetry as T
    path = _tmp_telemetry_path()
    orig = T.TELEMETRY_FILE
    T.TELEMETRY_FILE = path
    try:
        tid = T.start_trace("pos28", "mint28", "TKN", "live")
        T.event(tid, "buy_sent", private_key="TOP_SECRET", route="pp", sol=0.01)
        T.finish_trace(tid)
        with open(path) as f:
            content = f.read()
        assert "TOP_SECRET" not in content
        assert "sol" in content
        return {"redacted": True}
    finally:
        T.TELEMETRY_FILE = orig
        os.unlink(path)


@scenario(29, "Telemetry: bind_position + link_pair API", "C6_telemetry")
def s29():
    import memecoin.telemetry as T
    path = _tmp_telemetry_path()
    orig = T.TELEMETRY_FILE
    T.TELEMETRY_FILE = path
    try:
        tid = T.start_trace("", "mint29", "TKN", "live")
        T.bind_position(tid, "pos29_BOUND")
        T.link_pair(tid, "PAIR-29")
        meta = T.summarize_trace(tid)
        T.finish_trace(tid)
        assert meta["meta"]["pos_id"] == "pos29_BOUND"
        assert meta["meta"]["pair_id"] == "PAIR-29"
        return {"pos_id": meta["meta"]["pos_id"], "pair_id": meta["meta"]["pair_id"]}
    finally:
        T.TELEMETRY_FILE = orig
        os.unlink(path)


@scenario(30, "Telemetry: 44+ canonical E_* constants defined", "C6_telemetry")
def s30():
    import memecoin.telemetry as T
    constants = [k for k in dir(T) if k.startswith("E_")]
    assert len(constants) >= 44, f"Expected 44+ E_* constants, got {len(constants)}: {constants}"
    return {"constant_count": len(constants)}


# ===========================================================================
# Output
# ===========================================================================

def write_outputs(results: list[ScenarioResult]) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # JSON
    data = {
        "run_at": datetime.now(timezone.utc).isoformat() + "Z",
        "total": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "scenarios": [
            {
                "id": r.id, "name": r.name, "category": r.category,
                "passed": r.passed, "error": r.error,
                "elapsed_ms": r.elapsed_ms, "details": r.details,
            }
            for r in results
        ],
    }
    json_path = ARTIFACTS / "scenarios.json"
    json_path.write_text(json.dumps(data, indent=2, default=str))

    # Markdown
    lines = [
        "# Epoch Readiness — 30 Scenario Results",
        f"\n> Run: {data['run_at']}  |  **{data['passed']}/{data['total']} passed**\n",
        "## Results by Category\n",
    ]
    categories: dict[str, list] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)
    for cat, rlist in categories.items():
        passed = sum(1 for r in rlist if r.passed)
        lines.append(f"### {cat} ({passed}/{len(rlist)})\n")
        lines.append("| # | Name | Result | ms |")
        lines.append("|---|---|---|---|")
        for r in rlist:
            icon = "✅" if r.passed else "❌"
            lines.append(f"| {r.id} | {r.name} | {icon} | {r.elapsed_ms} |")
        lines.append("")

    if data["failed"] > 0:
        lines.append("## Failures\n")
        for r in results:
            if not r.passed:
                lines.append(f"### Scenario {r.id}: {r.name}\n```\n{r.error}\n```\n")

    md_path = ARTIFACTS / "scenarios.md"
    md_path.write_text("\n".join(lines))

    print(f"\nOutputs written:")
    print(f"  {json_path}")
    print(f"  {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    print(f"Running {len(SCENARIOS)} epoch readiness scenarios...\n")
    results = run_all(fail_fast=args.fail_fast)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    for r in results:
        icon = "✅" if r.passed else "❌"
        print(f"  {icon} [{r.id:02d}] {r.name} ({r.elapsed_ms}ms)")
        if not r.passed:
            first_line = r.error.splitlines()[0] if r.error else ""
            print(f"        ERROR: {first_line}")

    print(f"\n{'='*60}")
    print(f"  {passed}/{len(results)} passed  |  {failed} failed")
    print(f"{'='*60}")

    write_outputs(results)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
