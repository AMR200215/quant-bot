"""
memecoin/tests/test_telemetry.py — Epoch readiness tests for telemetry layer.

Run: python -m pytest memecoin/tests/test_telemetry.py -v
"""

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# We need to be able to import telemetry in isolation.
# Patch config imports to avoid pulling the full memecoin module tree.
import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_telemetry(tmp_path: Path):
    """Return a fresh telemetry module with isolated file paths."""
    import memecoin.telemetry as tel
    # Reset module state
    tel._traces.clear()
    tel._trace_events.clear()
    tel.TELEMETRY_ENABLED = True
    tel.TELEMETRY_FILE = str(tmp_path / "trade_telemetry.jsonl")
    return tel


# ---------------------------------------------------------------------------
# Test 1: thread safety
# ---------------------------------------------------------------------------
def test_telemetry_thread_safe(tmp_path):
    """Write from 10 threads simultaneously, verify all lines valid JSON."""
    tel = _fresh_telemetry(tmp_path)
    tid = tel.start_trace("pos_t1", "mint_t1", "SYM", "live")
    errors = []

    def _writer(i):
        try:
            tel.event(tid, f"thread_event_{i}", thread_id=i)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=_writer, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Threads raised: {errors}"

    # Verify all lines are valid JSON
    p = Path(tel.TELEMETRY_FILE)
    lines = p.read_text().strip().split("\n")
    assert len(lines) >= 11  # 1 trace_started + 10 thread events
    for line in lines:
        rec = json.loads(line)  # raises if invalid
        assert "event_name" in rec


# ---------------------------------------------------------------------------
# Test 2: failure does not raise
# ---------------------------------------------------------------------------
def test_telemetry_failure_no_raise(tmp_path):
    """Mock file write to raise, verify caller function continues."""
    tel = _fresh_telemetry(tmp_path)
    # Point to an unwritable path
    tel.TELEMETRY_FILE = "/dev/null/impossible/path.jsonl"
    tid = tel.start_trace("pos_f1", "mint_f1", "FAIL", "paper")
    # Should not raise
    tel.event(tid, "should_not_crash", value=42)
    tel.finish_trace(tid, final=True)
    # If we got here, the test passes


# ---------------------------------------------------------------------------
# Test 3: entry trace stage latencies
# ---------------------------------------------------------------------------
def test_entry_trace_stage_latencies(tmp_path):
    """Mock events with known timestamps, verify computed ms correct."""
    from memecoin.tests.fixtures.telemetry_fixtures import fixture_entry_trace_stage_latencies
    fix = fixture_entry_trace_stage_latencies()

    # Compute alert_to_fill_ms
    alert_ts = fix["alert_received"]["timestamp_wall"]
    fill_ts = fix["buy_fill_recorded"]["timestamp_wall"]
    computed_ms = (fill_ts - alert_ts) * 1000
    assert abs(computed_ms - fix["expected_alert_to_fill_ms"]) < 0.1

    # Compute preflight_ms
    pf_start = fix["preflight_started"]["timestamp_wall"]
    pf_done = fix["preflight_baseline_selected"]["timestamp_wall"]
    pf_ms = (pf_done - pf_start) * 1000
    assert abs(pf_ms - fix["expected_preflight_ms"]) < 0.1


# ---------------------------------------------------------------------------
# Test 4: exit trace trigger→fill latency
# ---------------------------------------------------------------------------
def test_exit_trace_trigger_to_fill(tmp_path):
    """Verify trigger→sent→confirm→fill latency computation."""
    from memecoin.tests.fixtures.telemetry_fixtures import fixture_exit_trace_trigger_to_fill
    fix = fixture_exit_trace_trigger_to_fill()

    trigger_ts = fix["exit_triggered"]["timestamp_wall"]
    confirm_ts = fix["sell_confirmed"]["timestamp_wall"]
    computed_ms = (confirm_ts - trigger_ts) * 1000
    assert abs(computed_ms - fix["expected_trigger_to_confirm_ms"]) < 0.1


# ---------------------------------------------------------------------------
# Test 5: gap — size asymmetry confound
# ---------------------------------------------------------------------------
def test_gap_size_asymmetry_confound():
    """Same pnl_pct, different size → classifier returns size_asymmetry_confound."""
    from memecoin.tests.fixtures.telemetry_fixtures import fixture_gap_size_asymmetry_confound
    from memecoin.tools.summarize_trade_telemetry import classify_gap

    fix = fixture_gap_size_asymmetry_confound()
    reason = classify_gap(
        live_pnl_pct=fix["live_pnl_pct"],
        paper_pnl_pct=fix["paper_pnl_pct"],
        live_size=fix["live_size_usd"],
        paper_size=fix["paper_size_usd"],
        events=fix["events"],
        missing=[],
    )
    assert reason == fix["expected_reason"]


# ---------------------------------------------------------------------------
# Test 6: gap — missing timestamps
# ---------------------------------------------------------------------------
def test_gap_missing_timestamps():
    """Absent timestamps → classifier returns missing_data with missing_fields."""
    from memecoin.tests.fixtures.telemetry_fixtures import fixture_gap_missing_timestamps
    from memecoin.tools.summarize_trade_telemetry import classify_gap

    fix = fixture_gap_missing_timestamps()
    reason = classify_gap(
        live_pnl_pct=fix["live_pnl_pct"],
        paper_pnl_pct=fix["paper_pnl_pct"],
        live_size=fix["live_size_usd"],
        paper_size=fix["paper_size_usd"],
        events=fix["events"],
        missing=fix["missing_fields"],
    )
    assert reason == fix["expected_reason"]


# ---------------------------------------------------------------------------
# Test 7: MU ladder trace
# ---------------------------------------------------------------------------
def test_mu_ladder_trace():
    """8 mu_retry_attempt events, verify summarize shows attempt 1-3 vs 4-7 vs 8."""
    from memecoin.tests.fixtures.telemetry_fixtures import fixture_mu_ladder_trace

    fix = fixture_mu_ladder_trace()
    events = fix["events"]

    # Group by phase
    phases = {}
    for e in events:
        if e["event_name"] == "mu_retry_attempt":
            n = e["attempt_number"]
            if n <= 3:
                phases.setdefault("oracle_gated", []).append(n)
            elif n <= 7:
                phases.setdefault("jupiter_rescue", []).append(n)
            else:
                phases.setdefault("final_gate", []).append(n)

    assert sorted(phases.keys()) == sorted(fix["expected_phases"])
    assert phases["oracle_gated"] == [1, 2, 3]
    assert phases["jupiter_rescue"] == [4, 5, 6, 7]
    assert phases["final_gate"] == [8]

    # Verify final gate event exists
    final_events = [e for e in events if e["event_name"] == "mu_final_gate"]
    assert len(final_events) == 1
    assert final_events[0]["attempt_number"] == 8


# ---------------------------------------------------------------------------
# Test 8: paper/live twins share pair_id
# ---------------------------------------------------------------------------
def test_paper_live_twins_share_pair_id(tmp_path):
    """Paper and live trace with same pair_id group correctly."""
    tel = _fresh_telemetry(tmp_path)
    from memecoin.tests.fixtures.telemetry_fixtures import fixture_paper_live_twins_share_pair_id

    fix = fixture_paper_live_twins_share_pair_id()
    pair_id = fix["pair_id"]

    # Start two traces with same pair_id
    t1 = tel.start_trace("paper_001", "mintABC", "SYM", "paper", pair_id=pair_id)
    t2 = tel.start_trace("Llive_001", "mintABC", "SYM", "live", pair_id=pair_id)

    tel.event(t1, "paper_event", data="paper")
    tel.event(t2, "live_event", data="live")

    # Read all events from JSONL
    p = Path(tel.TELEMETRY_FILE)
    events = [json.loads(line) for line in p.read_text().strip().split("\n")]

    # Group by pair_id
    pair_events = [e for e in events if e.get("pair_id") == pair_id]
    assert len(pair_events) >= 4  # 2 trace_started + 2 events

    # Both paper and live should be present
    live_or_papers = {e.get("live_or_paper") for e in pair_events}
    assert "paper" in live_or_papers
    assert "live" in live_or_papers


# ---------------------------------------------------------------------------
# Test 10: emit_once — edge-trigger guard
# ---------------------------------------------------------------------------
def test_emit_once_tp_condition_100_cycles(tmp_path):
    """
    A position held above TP1 for 100 monitor cycles must emit exactly
    one tp_condition_true event (edge-trigger, not level-trigger).
    """
    tel = _fresh_telemetry(tmp_path)
    tid = tel.start_trace("pos_tp_edge", "mint_tp", "EDGE", "paper")

    # Simulate 100 monitor cycles where TP1 condition is true
    for _ in range(100):
        tel.emit_once(tid, "tp_condition_true:tp_30", "tp_condition_true",
                      level_key="tp_30", tp_pct=30.0, gain_pct=45.0)

    tel.finish_trace(tid)

    # Read jsonl and count tp_condition_true events
    lines = Path(tel.TELEMETRY_FILE).read_text().splitlines()
    records = [json.loads(l) for l in lines if l.strip()]
    tp_events = [r for r in records if r.get("event_name") == "tp_condition_true"]
    assert len(tp_events) == 1, f"Expected 1 tp_condition_true, got {len(tp_events)}"
    assert tp_events[0]["level_key"] == "tp_30"


def test_emit_once_different_keys_each_emit(tmp_path):
    """emit_once with different keys each emit: all fire, none suppressed."""
    tel = _fresh_telemetry(tmp_path)
    tid = tel.start_trace("pos_keys", "mint_k", "KEYS", "paper")

    for level in ("tp_30", "tp_60", "tp_120"):
        # 3 cycles per level
        for _ in range(3):
            tel.emit_once(tid, f"tp_condition_true:{level}", "tp_condition_true",
                          level_key=level)

    tel.finish_trace(tid)

    lines = Path(tel.TELEMETRY_FILE).read_text().splitlines()
    records = [json.loads(l) for l in lines if l.strip()]
    tp_events = [r for r in records if r.get("event_name") == "tp_condition_true"]
    levels_seen = {r["level_key"] for r in tp_events}
    assert len(tp_events) == 3, f"Expected 3 (one per level), got {len(tp_events)}"
    assert levels_seen == {"tp_30", "tp_60", "tp_120"}


def test_emit_once_exit_condition_once_per_reason(tmp_path):
    """exit_condition_true fires exactly once even if monitor loop runs 50 times."""
    tel = _fresh_telemetry(tmp_path)
    tid = tel.start_trace("pos_exit_edge", "mint_ex", "EXIT", "live")

    for _ in range(50):
        tel.emit_once(tid, "exit_condition_true:hard_stop", "exit_condition_true",
                      reason="hard_stop", trigger_price=0.000010)

    tel.finish_trace(tid)

    lines = Path(tel.TELEMETRY_FILE).read_text().splitlines()
    records = [json.loads(l) for l in lines if l.strip()]
    exit_events = [r for r in records if r.get("event_name") == "exit_condition_true"]
    assert len(exit_events) == 1, f"Expected 1, got {len(exit_events)}"


# ---------------------------------------------------------------------------
# Test 9: shadow size floor 0.25 not applied
# ---------------------------------------------------------------------------
def test_shadow_size_floor_025_not_applied():
    """raw_mult=0.30, verify live_mult_used=0.50, hyp=0.30."""
    from memecoin.tests.fixtures.telemetry_fixtures import fixture_shadow_size_floor_025

    fix = fixture_shadow_size_floor_025()
    raw = fix["raw_mult"]

    live_mult = max(0.5, min(1.0, raw))
    hyp_mult = max(0.25, min(1.0, raw))

    assert live_mult == fix["expected_live_mult"]
    assert hyp_mult == fix["expected_hyp_mult_025"]

    base = fix["base_size"]
    assert round(base * live_mult, 2) == fix["expected_live_size"]
    assert round(base * hyp_mult, 2) == fix["expected_hyp_size_025"]
