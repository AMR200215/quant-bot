"""
memecoin/tests/fixtures/telemetry_fixtures.py — Fixtures for telemetry epoch readiness tests.

Each function returns a dict or list of dicts suitable for the corresponding test.
"""

import time


def _base_meta(pos_id="pos_001", mint="So111...abc", symbol="TEST", live_or_paper="live"):
    return {
        "pos_id": pos_id,
        "mint": mint,
        "symbol": symbol,
        "live_or_paper": live_or_paper,
        "pair_id": f"tg_{mint[:16]}_{int(time.time())}",
    }


def fixture_thread_safe():
    """10 concurrent event dicts to write."""
    return [
        {"event_name": f"thread_event_{i}", "thread_id": i, "value": i * 10}
        for i in range(10)
    ]


def fixture_failure_no_raise():
    """Event that should be written to a broken path."""
    return {"event_name": "broken_write", "should_not_raise": True}


def fixture_entry_trace_stage_latencies():
    """Known timestamps for entry stage latency computation."""
    t0 = 1000000.0
    return {
        "alert_received": {"timestamp_wall": t0, "event_name": "alert_received"},
        "preflight_started": {"timestamp_wall": t0 + 0.5, "event_name": "preflight_started"},
        "preflight_baseline_selected": {"timestamp_wall": t0 + 1.2, "event_name": "preflight_baseline_selected"},
        "buy_build_started": {"timestamp_wall": t0 + 1.3, "event_name": "buy_build_started"},
        "buy_confirmed": {"timestamp_wall": t0 + 3.5, "event_name": "buy_confirmed"},
        "buy_fill_recorded": {"timestamp_wall": t0 + 3.6, "event_name": "buy_fill_recorded"},
        "expected_alert_to_fill_ms": 3600.0,
        "expected_preflight_ms": 700.0,
    }


def fixture_exit_trace_trigger_to_fill():
    """Known timestamps for exit latency computation."""
    t0 = 2000000.0
    return {
        "exit_triggered": {"timestamp_wall": t0, "event_name": "exit_triggered"},
        "sell_confirmed": {"timestamp_wall": t0 + 2.5, "event_name": "sell_confirmed"},
        "expected_trigger_to_confirm_ms": 2500.0,
    }


def fixture_gap_size_asymmetry_confound():
    """Same pnl_pct, different size => size_asymmetry_confound."""
    return {
        "live_pnl_pct": -10.0,
        "paper_pnl_pct": -12.0,  # gap < 5pp
        "live_size_usd": 3.0,
        "paper_size_usd": 5.0,
        "events": [
            {"event_name": "buy_fill_recorded", "live_size_usd": 3.0},
        ],
        "expected_reason": "size_asymmetry_confound",
    }


def fixture_gap_missing_timestamps():
    """Absent timestamps => missing_data."""
    return {
        "live_pnl_pct": None,
        "paper_pnl_pct": None,
        "live_size_usd": 3.0,
        "paper_size_usd": 3.0,
        "events": [],
        "missing_fields": [
            "alert_received_ts", "preflight_start_ts", "preflight_done_ts",
            "buy_build_start_ts", "buy_sent_ts", "buy_confirmed_ts",
            "fill_recorded_ts", "exit_triggered_ts",
        ],
        "expected_reason": "missing_data",
    }


def fixture_mu_ladder_trace():
    """8 mu_retry_attempt events (attempts 1-8)."""
    events = []
    for i in range(1, 9):
        phase = "oracle_gated" if i <= 3 else ("jupiter_rescue" if i <= 7 else "final_gate")
        events.append({
            "event_name": "mu_retry_attempt",
            "attempt_number": i,
            "oracle_complete": i >= 4,
            "route_chosen": "bc_retry" if i <= 3 else "jupiter",
            "phase": phase,
        })
    if True:
        events.append({"event_name": "mu_final_gate", "attempt_number": 8})
    return {"events": events, "expected_phases": ["oracle_gated", "jupiter_rescue", "final_gate"]}


def fixture_paper_live_twins_share_pair_id():
    """Paper and live traces sharing the same pair_id."""
    pair_id = "tg_abc123_1000000"
    return {
        "pair_id": pair_id,
        "paper_trace": {
            "trace_id": "tr_paper_001_1000000",
            "pair_id": pair_id,
            "pos_id": "paper_001",
            "live_or_paper": "paper",
        },
        "live_trace": {
            "trace_id": "tr_live_001_1000000",
            "pair_id": pair_id,
            "pos_id": "Llive_001",
            "live_or_paper": "live",
        },
    }


def fixture_shadow_size_floor_025():
    """raw_mult=0.30 => live_mult=0.50, hyp_mult=0.30."""
    return {
        "raw_mult": 0.30,
        "expected_live_mult": 0.50,
        "expected_hyp_mult_025": 0.30,
        "base_size": 3.0,
        "expected_live_size": 1.50,
        "expected_hyp_size_025": 0.90,
    }
