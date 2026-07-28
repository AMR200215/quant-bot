"""
RF4 — Realert tracking tests.

Tests _check_dedup() and _record_realert() in isolation via mocked Supabase.
No network, no actual DB.
"""

import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call


def _make_tracker(sb=None):
    """
    Build a Tracker instance without starting threads or connecting to Supabase.
    """
    import queue
    # Patch the config so we don't need a .env file
    with patch.dict("os.environ", {
        "SUPABASE_URL": "https://fake.supabase.co",
        "SUPABASE_KEY": "fake-key",
    }):
        from research.tracker import Tracker
        t = Tracker(
            in_queue=queue.Queue(),
            poll_schedule_cb=lambda *a: None,
            peak_schedule_cb=None,
        )
    t._sb = sb or MagicMock()
    return t


def _make_alert(
    token_address: str = "TokenAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    alert_time: datetime = None,
    tg_message_id: int = None,
):
    """Build a minimal TGAlert-like object."""
    from research.tg_listener import TGAlert
    a = TGAlert.__new__(TGAlert)
    a.token_address = token_address
    a.alert_time    = alert_time or datetime.now(timezone.utc)
    a.chain         = "solana"
    a.raw_text      = "test"
    a.tg_message_id = tg_message_id
    # Fields added by backfill path — set defaults
    a.backfilled        = False
    a.source            = "telegram_live"
    a.event_id          = ""
    a.backfill_batch_id = None
    return a


class TestCheckDedup(unittest.TestCase):
    """Tests for Tracker._check_dedup()."""

    def test_first_alert_returns_true_none(self):
        """No existing row → (True, None)."""
        sb = MagicMock()
        sb.table.return_value.select.return_value \
            .eq.return_value.gte.return_value \
            .limit.return_value.execute.return_value.data = []

        t = _make_tracker(sb)
        is_new, existing_id = t._check_dedup("TokenXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        self.assertTrue(is_new)
        self.assertIsNone(existing_id)

    def test_existing_row_within_window_returns_false_id(self):
        """Row found within 24h → (False, existing_id)."""
        sb = MagicMock()
        existing_uuid = "abc123"
        sb.table.return_value.select.return_value \
            .eq.return_value.gte.return_value \
            .limit.return_value.execute.return_value.data = [{"id": existing_uuid}]

        t = _make_tracker(sb)
        is_new, existing_id = t._check_dedup("TokenAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        self.assertFalse(is_new)
        self.assertEqual(existing_id, existing_uuid)

    def test_db_error_returns_true_none_safe_default(self):
        """On DB error, returns (True, None) so insert path runs."""
        sb = MagicMock()
        sb.table.return_value.select.return_value \
            .eq.return_value.gte.return_value \
            .limit.return_value.execute.side_effect = Exception("DB timeout")

        t = _make_tracker(sb)
        is_new, existing_id = t._check_dedup("TokenAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        self.assertTrue(is_new)
        self.assertIsNone(existing_id)

    def test_no_sb_returns_true_none(self):
        """If Supabase client is None (init failed), return (True, None)."""
        t = _make_tracker(sb=None)
        t._sb = None
        is_new, existing_id = t._check_dedup("TokenAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        self.assertTrue(is_new)
        self.assertIsNone(existing_id)


class TestRecordRealert(unittest.TestCase):
    """Tests for Tracker._record_realert()."""

    def _make_sb_with_row(self, existing_id, realert_count=0, realert_times=None,
                          realert_message_ids=None):
        """
        Return a mock Supabase client that returns a specific row on SELECT
        and accepts UPDATE calls.
        """
        sb = MagicMock()
        row = {
            "realert_count":       realert_count,
            "realert_times":       realert_times or [],
            "realert_message_ids": realert_message_ids or [],
        }
        # SELECT chain
        sb.table.return_value.select.return_value \
            .eq.return_value.limit.return_value \
            .execute.return_value.data = [row]
        # UPDATE chain
        sb.table.return_value.update.return_value \
            .eq.return_value.execute.return_value = MagicMock()
        return sb

    def test_genuine_realert_increments_count(self):
        """_record_realert increments realert_count by 1."""
        existing_id = "row-uuid-001"
        sb = self._make_sb_with_row(existing_id, realert_count=0)
        t = _make_tracker(sb)

        alert = _make_alert(tg_message_id=42)
        t._record_realert(existing_id, alert)

        # Check UPDATE was called with count=1
        update_call = sb.table.return_value.update.call_args
        update_dict = update_call[0][0]
        self.assertEqual(update_dict["realert_count"], 1)

    def test_realert_appends_time(self):
        """realert_times gets the new alert_time appended."""
        existing_id = "row-uuid-002"
        existing_time = "2026-07-01T10:00:00+00:00"
        sb = self._make_sb_with_row(existing_id, realert_count=1,
                                     realert_times=[existing_time])
        t = _make_tracker(sb)

        new_alert_time = datetime(2026, 7, 2, 12, 0, 0, tzinfo=timezone.utc)
        alert = _make_alert(alert_time=new_alert_time)
        t._record_realert(existing_id, alert)

        update_dict = sb.table.return_value.update.call_args[0][0]
        self.assertEqual(update_dict["realert_count"], 2)
        self.assertIn(existing_time, update_dict["realert_times"])
        self.assertIn(new_alert_time.isoformat(), update_dict["realert_times"])

    def test_realert_appends_message_id(self):
        """realert_message_ids gets the new tg_message_id appended."""
        existing_id = "row-uuid-003"
        sb = self._make_sb_with_row(existing_id, realert_count=0, realert_message_ids=[111])
        t = _make_tracker(sb)

        alert = _make_alert(tg_message_id=222)
        t._record_realert(existing_id, alert)

        update_dict = sb.table.return_value.update.call_args[0][0]
        self.assertIn(111, update_dict["realert_message_ids"])
        self.assertIn(222, update_dict["realert_message_ids"])

    def test_none_message_id_filtered_out(self):
        """tg_message_id=None is stripped from realert_message_ids list."""
        existing_id = "row-uuid-004"
        sb = self._make_sb_with_row(existing_id, realert_count=0)
        t = _make_tracker(sb)

        alert = _make_alert(tg_message_id=None)
        t._record_realert(existing_id, alert)

        update_dict = sb.table.return_value.update.call_args[0][0]
        self.assertNotIn(None, update_dict["realert_message_ids"])

    def test_no_row_found_returns_silently(self):
        """If SELECT returns no data, _record_realert returns without error."""
        sb = MagicMock()
        sb.table.return_value.select.return_value \
            .eq.return_value.limit.return_value \
            .execute.return_value.data = []
        t = _make_tracker(sb)

        alert = _make_alert()
        t._record_realert("nonexistent-id", alert)  # should not raise

        # UPDATE should NOT have been called
        sb.table.return_value.update.assert_not_called()

    def test_db_error_is_swallowed(self):
        """Exception in _record_realert is caught and logged, not raised."""
        sb = MagicMock()
        sb.table.return_value.select.return_value \
            .eq.return_value.limit.return_value \
            .execute.side_effect = Exception("network error")
        t = _make_tracker(sb)

        alert = _make_alert()
        try:
            t._record_realert("some-id", alert)
        except Exception:
            self.fail("_record_realert raised an exception instead of swallowing it")


class TestProcessIntegration(unittest.TestCase):
    """
    Integration-level tests for Tracker._process() dedup/realert branching.
    _insert and external calls are mocked.
    """

    def _make_tracker_with_sb(self, existing_id=None):
        """
        Return a Tracker where:
          - _check_dedup is mocked to return (True, None) or (False, existing_id)
          - _insert returns a fake row id
          - _record_realert is mocked
          - fetch_snapshot_with_retry is mocked
        """
        import queue, sys
        # We need the module imported; patch external deps
        with patch.dict("os.environ", {
            "SUPABASE_URL": "https://fake.supabase.co",
            "SUPABASE_KEY": "fake-key",
        }):
            from research.tracker import Tracker
            t = Tracker(
                in_queue=queue.Queue(),
                poll_schedule_cb=lambda *a: None,
                peak_schedule_cb=None,
            )
        t._sb = MagicMock()
        return t

    def test_first_alert_calls_insert(self):
        """(True, None) from _check_dedup → _insert is called."""
        t = self._make_tracker_with_sb()
        t._check_dedup  = MagicMock(return_value=(True, None))
        t._insert       = MagicMock(return_value="new-row-id")
        t._record_realert = MagicMock()

        alert = _make_alert()

        with patch("research.tracker.fetch_snapshot_with_retry",
                   return_value=({"snapshot_ok": False}, 1)), \
             patch("research.tracker._read_pp_snapshot", return_value={}), \
             patch("research.tracker._assign_category", return_value="social_alert_bc"):
            t._process(alert)

        t._insert.assert_called_once()
        t._record_realert.assert_not_called()

    def test_genuine_realert_calls_record_realert(self):
        """(False, id) from _check_dedup → _record_realert is called, _insert is not."""
        t = self._make_tracker_with_sb()
        existing = "existing-row-uuid"
        t._check_dedup    = MagicMock(return_value=(False, existing))
        t._insert         = MagicMock()
        t._record_realert = MagicMock()

        alert = _make_alert()
        t._process(alert)

        t._record_realert.assert_called_once_with(existing, alert)
        t._insert.assert_not_called()

    def test_later_distinct_event_inserts_new_row(self):
        """
        Alert comes in after 25h (beyond DEDUP_WINDOW_HOURS=24).
        _check_dedup returns (True, None) → _insert is called.
        """
        t = self._make_tracker_with_sb()
        t._check_dedup    = MagicMock(return_value=(True, None))   # new event
        t._insert         = MagicMock(return_value="brand-new-row")
        t._record_realert = MagicMock()

        # Alert time is irrelevant here — the dedup window check is already mocked
        old_alert = _make_alert(
            alert_time=datetime.now(timezone.utc) - timedelta(hours=25)
        )

        with patch("research.tracker.fetch_snapshot_with_retry",
                   return_value=({"snapshot_ok": False}, 1)), \
             patch("research.tracker._read_pp_snapshot", return_value={}), \
             patch("research.tracker._assign_category", return_value="social_alert_bc"):
            t._process(old_alert)

        t._insert.assert_called_once()
        t._record_realert.assert_not_called()

    def test_concurrent_realerts_accumulate_count(self):
        """Two sequential realerts accumulate count correctly."""
        existing_id = "row-uuid-concurrent"
        # Simulate DB starting at count=0, then count=1 after first update
        call_count = {"n": 0}

        def _select_side_effect():
            mock_resp = MagicMock()
            mock_resp.data = [{"realert_count": call_count["n"],
                                "realert_times": [],
                                "realert_message_ids": []}]
            return mock_resp

        sb = MagicMock()
        sb.table.return_value.select.return_value \
            .eq.return_value.limit.return_value \
            .execute.side_effect = lambda: _select_side_effect()
        sb.table.return_value.update.return_value \
            .eq.return_value.execute.return_value = MagicMock()

        t = _make_tracker(sb)

        alert1 = _make_alert(tg_message_id=100)
        alert2 = _make_alert(tg_message_id=101)

        t._record_realert(existing_id, alert1)
        call_count["n"] = 1   # DB is updated after first call
        t._record_realert(existing_id, alert2)

        # Check second update passed count=2
        all_update_calls = sb.table.return_value.update.call_args_list
        self.assertEqual(len(all_update_calls), 2)
        last_count = all_update_calls[-1][0][0]["realert_count"]
        self.assertEqual(last_count, 2)

    def test_realert_count_starts_at_zero_on_first_insert(self):
        """
        Verify that _insert includes realert_count=0 in the row for first alerts.
        We inspect the _pp_extras dict that gets merged into the insert payload.
        """
        t = self._make_tracker_with_sb()
        inserted_rows = []

        def _capture_insert(base, extra):
            row = {**base, **extra}
            inserted_rows.append(row)
            return "new-id"

        t._check_dedup = MagicMock(return_value=(True, None))
        t._record_realert = MagicMock()

        # We need to intercept the actual _do_insert call
        # Replace the Supabase INSERT to capture the merged dict
        sb = MagicMock()
        mock_resp = MagicMock()
        mock_resp.data = [{"id": "new-row-id"}]
        sb.table.return_value.insert.return_value.execute.return_value = mock_resp
        t._sb = sb

        alert = _make_alert()
        snap = {"snapshot_ok": False, "price_usd": None}

        with patch("research.tracker.fetch_snapshot_with_retry",
                   return_value=(snap, 1)), \
             patch("research.tracker._read_pp_snapshot", return_value={}), \
             patch("research.tracker._assign_category", return_value="social_alert_bc"):
            t._process(alert)

        # Capture the dict passed to sb.table().insert()
        self.assertTrue(sb.table.return_value.insert.called)
        inserted_dict = sb.table.return_value.insert.call_args[0][0]
        self.assertEqual(inserted_dict.get("realert_count"), 0)
        self.assertEqual(inserted_dict.get("realert_times"), [])


if __name__ == "__main__":
    unittest.main()
