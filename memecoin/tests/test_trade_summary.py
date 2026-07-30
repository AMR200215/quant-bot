"""
memecoin/tests/test_trade_summary.py — Fixture-based tests for summarize_trade_telemetry.py

10 tests covering: paper PnL, buy_sent_ts, token_program, pair_traces, classify_gap, summarize.
"""

import csv
import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from memecoin.tools.summarize_trade_telemetry import (
    classify_gap,
    load_paper_journal,
    pair_traces,
    summarize,
    _get_buy_sent_ts,
    _get_token_program,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(event_name: str, **kwargs) -> dict:
    return {"event_name": event_name, "timestamp_wall": "2026-07-11T10:00:00Z", **kwargs}


def _write_paper_journal(path: Path, rows: list[dict]) -> None:
    """Write a minimal paper journal CSV."""
    fieldnames = ["id", "pair_id", "symbol", "mint", "pnl_usd", "pnl_pct"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in fieldnames}
            w.writerow(row)


def _write_telemetry(path: Path, events: list[dict]) -> None:
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


# ---------------------------------------------------------------------------
# Test 1: paper PnL is populated from social journal when pair_id matches
# ---------------------------------------------------------------------------

def test_paper_pnl_populated_from_social_journal(tmp_path):
    """paper_pnl_usd and paper_pnl_pct are set from memecoin_social_journal.csv."""
    paper_journal = tmp_path / "memecoin_social_journal.csv"
    _write_paper_journal(paper_journal, [
        {"id": "pos1", "pair_id": "pair-abc", "symbol": "TOKEN", "mint": "mint1",
         "pnl_usd": "3.50", "pnl_pct": "12.5"},
    ])

    telemetry = tmp_path / "trade_telemetry.jsonl"
    _write_telemetry(telemetry, [
        {"trace_id": "t1", "pair_id": "pair-abc", "pos_id": "pos1",
         "event_name": "alert_received", "timestamp_wall": "2026-07-11T10:00:00Z",
         "symbol": "TOKEN", "mint": "mint1"},
    ])

    live_journal = tmp_path / "memecoin_live_journal.csv"
    live_journal.write_text("id,pnl_usd,pnl_pct\npos1,2.00,8.0\n")

    with (
        mock.patch("memecoin.tools.summarize_trade_telemetry.TELEMETRY_FILE", telemetry),
        mock.patch("memecoin.tools.summarize_trade_telemetry.LIVE_JOURNAL", live_journal),
        mock.patch("memecoin.tools.summarize_trade_telemetry.PAPER_JOURNAL", paper_journal),
    ):
        rows = summarize()

    assert len(rows) == 1
    row = rows[0]
    assert row["paper_pnl_usd"] == 3.50
    assert row["paper_pnl_pct"] == 12.5


# ---------------------------------------------------------------------------
# Test 2: buy_sent_ts maps to "buy_sent" event, not "buy_build_done"
# ---------------------------------------------------------------------------

def test_buy_sent_ts_uses_buy_sent_event():
    """buy_sent_ts should come from 'buy_sent' event when present."""
    events = [
        {"event_name": "buy_sent",      "timestamp_wall": "2026-07-11T10:00:05Z"},
        {"event_name": "buy_build_done","timestamp_wall": "2026-07-11T10:00:03Z"},
    ]
    ts = _get_buy_sent_ts(events)
    # buy_sent is at 10:00:05, buy_build_done is at 10:00:03
    # Must use buy_sent (later)
    from datetime import datetime, timezone
    expected = datetime.fromisoformat("2026-07-11T10:00:05").replace(tzinfo=timezone.utc).timestamp()
    assert ts == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Test 3: buy_sent_ts falls back to "buy_build_done" when "buy_sent" is missing
# ---------------------------------------------------------------------------

def test_buy_sent_ts_fallback_to_buy_build_done():
    """buy_sent_ts falls back to buy_build_done when buy_sent event is absent."""
    events = [
        {"event_name": "buy_build_done", "timestamp_wall": "2026-07-11T10:00:03Z"},
    ]
    ts = _get_buy_sent_ts(events)
    from datetime import datetime, timezone
    expected = datetime.fromisoformat("2026-07-11T10:00:03").replace(tzinfo=timezone.utc).timestamp()
    assert ts == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Test 4: token_program extracted from "entry_gate_checked" event
# ---------------------------------------------------------------------------

def test_token_program_extracted_from_entry_gate_checked():
    """token_program field is pulled from entry_gate_checked event."""
    events = [
        _make_event("entry_gate_checked", token_program="TokenzQdBNbLqP5VgrSeGCQZbFDVgJgZRQRCaT9wZVRp"),
    ]
    result = _get_token_program(events)
    assert result == "TokenzQdBNbLqP5VgrSeGCQZbFDVgJgZRQRCaT9wZVRp"


# ---------------------------------------------------------------------------
# Test 5: token_program blank when "entry_gate_checked" absent
# ---------------------------------------------------------------------------

def test_token_program_blank_when_event_absent():
    """token_program returns empty string when entry_gate_checked is not present."""
    events = [
        _make_event("buy_confirmed", route_used="pumpswap"),
    ]
    result = _get_token_program(events)
    assert result == ""


# ---------------------------------------------------------------------------
# Test 6: pair_traces matches by pair_id
# ---------------------------------------------------------------------------

def test_pair_traces_matches_by_pair_id():
    """pair_traces links live and paper traces sharing the same pair_id."""
    live_traces = {
        "live-t1": [{"event_name": "alert_received", "pair_id": "pair-XYZ",
                     "timestamp_wall": "2026-07-11T10:00:00Z"}],
    }
    paper_traces = {
        "paper-t1": [{"event_name": "alert_received", "pair_id": "pair-XYZ",
                      "timestamp_wall": "2026-07-11T10:00:01Z"}],
    }
    results = pair_traces(live_traces, paper_traces)
    assert len(results) == 1
    r = results[0]
    assert r["live_trace_id"] == "live-t1"
    assert r["paper_trace_id"] == "paper-t1"
    assert r["match_method"] == "pair_id"


# ---------------------------------------------------------------------------
# Test 7: pair_traces produces MISSING_PAIR when no match found
# ---------------------------------------------------------------------------

def test_pair_traces_missing_pair():
    """pair_traces returns MISSING_PAIR when live trace has no matching paper trace."""
    live_traces = {
        "live-t99": [{"event_name": "alert_received", "pair_id": "pair-NOPE",
                      "mint": "mintXXX", "timestamp_wall": "2026-07-11T10:00:00Z"}],
    }
    paper_traces = {
        "paper-t55": [{"event_name": "alert_received", "pair_id": "pair-OTHER",
                       "mint": "mintYYY", "timestamp_wall": "2026-07-11T11:00:00Z"}],
    }
    results = pair_traces(live_traces, paper_traces)
    assert len(results) == 1
    r = results[0]
    assert r["paper_trace_id"] == "MISSING_PAIR"
    assert r["match_method"] == "MISSING_PAIR"


# ---------------------------------------------------------------------------
# Test 8: classify_gap returns "strategy_loss" for negative live+paper PnL
# ---------------------------------------------------------------------------

def test_classify_gap_strategy_loss():
    """classify_gap returns 'strategy_loss' when both live and paper PnL are negative."""
    result = classify_gap(
        live_pnl_pct=-15.0,
        paper_pnl_pct=-12.0,
        live_size=5.0,
        paper_size=5.0,
        events=[],
        missing=[],
    )
    assert result == "strategy_loss"


# ---------------------------------------------------------------------------
# Test 9: summarize() returns empty list when no telemetry file
# ---------------------------------------------------------------------------

def test_summarize_returns_empty_when_no_telemetry(tmp_path):
    """summarize() returns [] when trade_telemetry.jsonl does not exist."""
    missing_file = tmp_path / "trade_telemetry.jsonl"  # does not exist
    live_journal = tmp_path / "memecoin_live_journal.csv"
    live_journal.write_text("id,pnl_usd,pnl_pct\n")
    paper_journal = tmp_path / "memecoin_social_journal.csv"
    paper_journal.write_text("id,pair_id,symbol,mint,pnl_usd,pnl_pct\n")

    with (
        mock.patch("memecoin.tools.summarize_trade_telemetry.TELEMETRY_FILE", missing_file),
        mock.patch("memecoin.tools.summarize_trade_telemetry.LIVE_JOURNAL", live_journal),
        mock.patch("memecoin.tools.summarize_trade_telemetry.PAPER_JOURNAL", paper_journal),
    ):
        rows = summarize()

    assert rows == []


# ---------------------------------------------------------------------------
# Test 10: gap_reason "missing_data" when key timing events absent
# ---------------------------------------------------------------------------

def test_gap_reason_missing_data_when_timing_events_absent(tmp_path):
    """gap_reason is 'missing_data' when all key timing events are missing."""
    telemetry = tmp_path / "trade_telemetry.jsonl"
    # Only supply one low-info event — all timing fields will be None → missing
    _write_telemetry(telemetry, [
        {"trace_id": "t-missing", "pair_id": "p1", "pos_id": "pos9",
         "event_name": "some_other_event", "timestamp_wall": "2026-07-11T10:00:00Z",
         "symbol": "XYZ", "mint": "mintABC"},
    ])

    live_journal = tmp_path / "memecoin_live_journal.csv"
    live_journal.write_text("id,pnl_usd,pnl_pct\n")
    paper_journal = tmp_path / "memecoin_social_journal.csv"
    paper_journal.write_text("id,pair_id,symbol,mint,pnl_usd,pnl_pct\n")

    with (
        mock.patch("memecoin.tools.summarize_trade_telemetry.TELEMETRY_FILE", telemetry),
        mock.patch("memecoin.tools.summarize_trade_telemetry.LIVE_JOURNAL", live_journal),
        mock.patch("memecoin.tools.summarize_trade_telemetry.PAPER_JOURNAL", paper_journal),
    ):
        rows = summarize()

    assert len(rows) == 1
    assert rows[0]["gap_reason"] == "missing_data"
