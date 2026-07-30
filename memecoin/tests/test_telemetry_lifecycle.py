"""
C6 — Telemetry lifecycle: 11 tests verifying repaired telemetry.py API.
R6: calls production functions from memecoin.telemetry directly.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import memecoin.telemetry as T


def _with_tmp_telemetry(fn):
    """
    Run fn(tmp_path) with T.TELEMETRY_FILE redirected to a temp file.
    Returns whatever fn returns. Cleans up file after fn completes.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
    tmp.close()
    orig = T.TELEMETRY_FILE
    T.TELEMETRY_FILE = tmp.name
    try:
        return fn(tmp.name)
    finally:
        T.TELEMETRY_FILE = orig
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _read_jsonl(path: str) -> list[dict]:
    recs = []
    try:
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    recs.append(json.loads(ln))
    except OSError:
        pass
    return recs


class TestTelemetryLifecycle(unittest.TestCase):

    def test_trace_id_is_uuid_based(self):
        """start_trace returns a UUID-based trace_id (tr_<32hex>), not timestamp."""
        def run(path):
            tid = T.start_trace("pos1", "mintA", "TKN", "live")
            T.finish_trace(tid)
            return tid
        tid = _with_tmp_telemetry(run)
        self.assertRegex(tid, r'^tr_[0-9a-f]{32}$',
                         "trace_id must be tr_<uuid4_hex>, not timestamp-based")

    def test_trace_id_unique_across_same_pos(self):
        """Two start_trace calls for the same pos produce different trace_ids."""
        def run(path):
            t1 = T.start_trace("pos2", "mintA", "TKN", "live")
            t2 = T.start_trace("pos2", "mintA", "TKN", "live")
            T.finish_trace(t1)
            T.finish_trace(t2)
            return t1, t2
        t1, t2 = _with_tmp_telemetry(run)
        self.assertNotEqual(t1, t2)

    def test_timestamp_wall_no_double_suffix(self):
        """timestamp_wall must be ISO8601 with Z suffix, never +00:00Z."""
        def run(path):
            tid = T.start_trace("pos3", "mintB", "TKN", "paper")
            T.finish_trace(tid)
            return _read_jsonl(path)
        lines = _with_tmp_telemetry(run)
        self.assertGreater(len(lines), 0)
        for rec in lines:
            wall = rec.get("timestamp_wall", "")
            self.assertFalse(
                wall.endswith("+00:00Z"),
                f"timestamp_wall has double-suffix: {wall!r}",
            )
            self.assertTrue(
                wall.endswith("Z"),
                f"timestamp_wall must end with Z: {wall!r}",
            )

    def test_secret_fields_redacted(self):
        """Fields with secret names are not written to the JSONL file."""
        def run(path):
            tid = T.start_trace("pos4", "mintC", "TKN", "live")
            T.event(tid, "buy_sent", tx_sig="abc123", private_key="SEKRIT",
                    api_key="ALSO_SEKRIT", route="pumpportal")
            T.finish_trace(tid)
            with open(path) as f:
                return f.read()
        content = _with_tmp_telemetry(run)
        self.assertNotIn("SEKRIT", content)
        self.assertNotIn("ALSO_SEKRIT", content)
        self.assertIn("abc123", content)

    def test_bind_position_updates_pos_id(self):
        """bind_position(trace_id, pos_id) updates pos_id in trace meta."""
        def run(path):
            tid = T.start_trace("", "mintD", "TKN", "live")
            T.bind_position(tid, "pos_NEW")
            result = T.summarize_trace(tid)
            T.finish_trace(tid)
            return result
        result = _with_tmp_telemetry(run)
        self.assertEqual(result["meta"]["pos_id"], "pos_NEW")

    def test_get_trace_id_for_pos_uses_index(self):
        """get_trace_id_for_pos returns the trace_id for an active position."""
        def run(path):
            tid = T.start_trace("pos5", "mintE", "TKN", "live")
            found = T.get_trace_id_for_pos("pos5")
            T.finish_trace(tid)
            return tid, found
        tid, found = _with_tmp_telemetry(run)
        self.assertEqual(found, tid)

    def test_link_pair_sets_pair_id(self):
        """link_pair(trace_id, pair_id) sets pair_id in trace meta."""
        def run(path):
            tid = T.start_trace("pos6", "mintF", "TKN", "live")
            T.link_pair(tid, "PAIR-XYZ")
            result = T.summarize_trace(tid)
            T.finish_trace(tid)
            return result
        result = _with_tmp_telemetry(run)
        self.assertEqual(result["meta"]["pair_id"], "PAIR-XYZ")

    def test_restore_trace_returns_existing(self):
        """restore_trace_for_position returns existing trace_id if alive."""
        def run(path):
            tid = T.start_trace("pos7", "mintG", "TKN", "live")
            restored = T.restore_trace_for_position("pos7", "mintG", "TKN", "live")
            T.finish_trace(tid)
            return tid, restored
        tid, restored = _with_tmp_telemetry(run)
        self.assertEqual(restored, tid)

    def test_restore_trace_creates_new_when_missing(self):
        """restore_trace_for_position creates a new trace when none active."""
        def run(path):
            new_tid = T.restore_trace_for_position("pos8", "mintH", "TKN", "live")
            T.finish_trace(new_tid)
            return new_tid
        new_tid = _with_tmp_telemetry(run)
        self.assertRegex(new_tid, r'^tr_[0-9a-f]{32}$')

    def test_emit_once_suppresses_duplicate_events(self):
        """emit_once with the same event_key only emits the event once."""
        def run(path):
            tid = T.start_trace("pos9", "mintI", "TKN", "live")
            r1 = T.emit_once(tid, "hard_stop:key", T.E_EXIT_CONDITION_TRUE, reason="hard_stop")
            r2 = T.emit_once(tid, "hard_stop:key", T.E_EXIT_CONDITION_TRUE, reason="hard_stop")
            T.finish_trace(tid)
            events = [r["event_name"] for r in _read_jsonl(path)]
            return r1, r2, events
        r1, r2, events = _with_tmp_telemetry(run)
        self.assertTrue(r1)
        self.assertFalse(r2, "Second emit_once call must be suppressed")
        count = events.count(T.E_EXIT_CONDITION_TRUE)
        self.assertEqual(count, 1, f"Expected 1 {T.E_EXIT_CONDITION_TRUE} event, got {count}")

    def test_canonical_event_constants_exist(self):
        """All 30+ canonical E_* constants are defined in telemetry module."""
        required = [
            "E_ALERT_RECEIVED", "E_PREFLIGHT_STARTED", "E_PREFLIGHT_BASELINE",
            "E_PREFLIGHT_BLOCKED", "E_ENTRY_GATE_CHECKED", "E_ENTRY_GATE_BLOCKED",
            "E_BUY_BUILD_STARTED", "E_BUY_BUILD_DONE", "E_BUY_SENT",
            "E_BUY_CONFIRMED", "E_BUY_FILL_RECORDED", "E_BUY_FAILED",
            "E_ABORT_TRIPWIRE", "E_ABORT_SELL_SENT", "E_ABORT_SELL_CONFIRMED",
            "E_PRICE_TICK", "E_TP_CONDITION_TRUE", "E_TP_INFLIGHT",
            "E_EXIT_CONDITION_TRUE", "E_EXIT_QUEUED", "E_EXIT_TRIGGERED",
            "E_PREGRAD_THRESHOLD", "E_GRAD_DETECTED", "E_GRAD_FAST_WINDOW",
            "E_SELL_BUILD_STARTED", "E_SELL_SENT", "E_SELL_CONFIRMED",
            "E_SELL_FAILED", "E_SELL_STUCK", "E_SELL_ROUTE_CHANGED",
            "E_MU_ATTEMPT", "E_MU_ESCALATE", "E_MU_FINAL_GATE",
            "E_MU_MANUAL_REQUIRED", "E_RESCUE_STARTED", "E_RESCUE_SUCCEEDED",
            "E_RESCUE_FAILED", "E_RECONCILED_GONE",
            "E_JOURNAL_WRITTEN", "E_RECEIPT_WRITTEN", "E_RECEIPT_PROMOTED",
            "E_TRACE_STARTED", "E_TRACE_FINISHED", "E_SIZE_SHADOW",
        ]
        missing = [name for name in required if not hasattr(T, name)]
        self.assertEqual(missing, [], f"Missing canonical event constants: {missing}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
