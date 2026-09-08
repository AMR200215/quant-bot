"""research/tests/test_thr_reconstruct_paths.py — THR-BATCH T1.

Fixture blobs below are REAL "Program data:" log lines captured live from
mainnet-beta, 2026-09-08 (see research/thr_reconstruct_paths.py's module
docstring for the full verification: both decoded to virtual_sol_reserves/
virtual_token_reserves values bit-identical to a live getAccountInfo read
of the same curve account at the same moment). Not synthetic.

Run: python -m pytest research/tests/test_thr_reconstruct_paths.py -v
"""

import base64
import unittest

from research.thr_reconstruct_paths import (
    _decode_trade_events, curve_event_price_usd, extract_rows_exact,
    compute_xval_diffs, classify_xval, XvalOffsetDiff,
)

# Real event, mint HnAVbEMfF1iLdBsSF2YGf9LHWvurwX63cTGHyKaWpump, a SellV2.
_REAL_EVENT_1_MINT = "HnAVbEMfF1iLdBsSF2YGf9LHWvurwX63cTGHyKaWpump"
_REAL_EVENT_1_B64 = (
    "vdt/007mYe75TNxlkX/y4DNd7rh9Do9vZnNu+iiqsfk/uK0O4tQ5v7diQAMAAAAAuTh1EsIBAAAA"
    "OQMBrV7X0XyRxiF7MdJ5+4mQfDDjkGir/FRIYipc+OzgbZ9qAAAAAPVS2gIHAAAAXeBZ0zzMAwD1"
    "prYGAAAAAF1IR4erzQIAY4NzAA6iLLJk00r/ZKBLXvq/u3TdzQSJl7GYFUfX0RBfAAAAAAAAAF3o"
    "BwAAAAAAu+ySfNwWzXDRM33O5FlZksrSG9USMWObXh6Cy2LiS2QeAAAAAAAAAEZ/AgAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAHNlbGwAAAAAAAAAAAAAAAAAAAAA"
    "AIgTAAAAAAAALvQDAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAC3YkAD"
    "AAAAAPVS2gIHAAAA9aa2BgAAAAA="
)
# Verified live ground truth for the event above (getAccountInfo, same moment):
_REAL_EVENT_1_EXPECTED_VSOL = 30.112633589
_REAL_EVENT_1_EXPECTED_VTOKEN_RAW = 1068986546118749


def _mint_bytes(mint: str) -> bytes:
    import base58
    return base58.b58decode(mint)


class TestDecodeTradeEvents(unittest.TestCase):

    def test_real_event_decodes_to_verified_ground_truth(self):
        log_messages = [f"Program data: {_REAL_EVENT_1_B64}"]
        events = _decode_trade_events(log_messages, _mint_bytes(_REAL_EVENT_1_MINT))
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertAlmostEqual(ev["virtual_sol_reserves"] / 1e9, _REAL_EVENT_1_EXPECTED_VSOL, places=9)
        self.assertEqual(ev["virtual_token_reserves"], _REAL_EVENT_1_EXPECTED_VTOKEN_RAW)
        self.assertFalse(ev["is_buy"])  # this real trade was a sell

    def test_wrong_mint_bytes_excludes_real_event(self):
        log_messages = [f"Program data: {_REAL_EVENT_1_B64}"]
        wrong_mint_bytes = _mint_bytes("Fg8f3Q8GGhdzCw4QRjgCioQ3scq2SXcZt3mzZytcpump")
        events = _decode_trade_events(log_messages, wrong_mint_bytes)
        self.assertEqual(events, [])

    def test_non_program_data_lines_ignored(self):
        log_messages = ["Program log: Instruction: SellV2", "Program 111 success"]
        events = _decode_trade_events(log_messages, _mint_bytes(_REAL_EVENT_1_MINT))
        self.assertEqual(events, [])

    def test_garbage_base64_does_not_crash(self):
        log_messages = ["Program data: !!!not-valid-base64!!!"]
        events = _decode_trade_events(log_messages, _mint_bytes(_REAL_EVENT_1_MINT))
        self.assertEqual(events, [])

    def test_short_blob_excluded(self):
        short_b64 = base64.b64encode(b"\x00" * 40).decode()
        events = _decode_trade_events([f"Program data: {short_b64}"], _mint_bytes(_REAL_EVENT_1_MINT))
        self.assertEqual(events, [])

    def test_empty_log_messages(self):
        self.assertEqual(_decode_trade_events([], _mint_bytes(_REAL_EVENT_1_MINT)), [])
        self.assertEqual(_decode_trade_events(None, _mint_bytes(_REAL_EVENT_1_MINT)), [])


class TestCurveEventPriceUsd(unittest.TestCase):

    def test_matches_curve_oracle_formula_at_known_reserves(self):
        # Reproduces research/curve_oracle.py's PF0 formula by hand for a
        # known reserve pair, confirming this module's formula agrees.
        event = {"virtual_sol_reserves": 30_112_633_589, "virtual_token_reserves": 1_068_986_546_118_749}
        sol_price_usd = 200.0
        price = curve_event_price_usd(event, sol_price_usd)
        expected_price_sol = (30_112_633_589 / 1e9) / (1_068_986_546_118_749 / 1e6)
        self.assertAlmostEqual(price, expected_price_sol * sol_price_usd, places=9)

    def test_price_scales_linearly_with_sol_price(self):
        event = {"virtual_sol_reserves": 30_000_000_000, "virtual_token_reserves": 1_000_000_000_000_000}
        p1 = curve_event_price_usd(event, 100.0)
        p2 = curve_event_price_usd(event, 200.0)
        self.assertAlmostEqual(p2, p1 * 2, places=9)


class TestExtractRowsExact(unittest.TestCase):

    def _tx_with_event(self, mint, block_time=1788833248):
        return {
            "blockTime": block_time,
            "meta": {"err": None, "logMessages": [f"Program data: {_REAL_EVENT_1_B64}"]},
        }

    def test_matching_event_produces_exact_row(self):
        tx = self._tx_with_event(_REAL_EVENT_1_MINT)
        exact_rows, heuristic = extract_rows_exact([tx], _REAL_EVENT_1_MINT, sol_price=200.0)
        self.assertEqual(len(exact_rows), 1)
        self.assertEqual(heuristic, [])
        row = exact_rows[0]
        self.assertEqual(row["source"], "reconstructed_curve_exact")
        self.assertEqual(row["venue_state"], "CURVE_ACTIVE")
        self.assertEqual(row["side"], "sell")
        self.assertAlmostEqual(row["vsol"], _REAL_EVENT_1_EXPECTED_VSOL, places=6)
        self.assertEqual(row["vtok"], _REAL_EVENT_1_EXPECTED_VTOKEN_RAW)

    def test_no_matching_event_falls_through_to_heuristic_candidates(self):
        tx = {"blockTime": 123, "meta": {"err": None, "logMessages": ["Program log: nothing relevant"]}}
        exact_rows, heuristic = extract_rows_exact([tx], _REAL_EVENT_1_MINT, sol_price=200.0)
        self.assertEqual(exact_rows, [])
        self.assertEqual(len(heuristic), 1)

    def test_errored_tx_excluded_entirely(self):
        tx = {"blockTime": 123, "meta": {"err": {"InstructionError": []}, "logMessages": []}}
        exact_rows, heuristic = extract_rows_exact([tx], _REAL_EVENT_1_MINT, sol_price=200.0)
        self.assertEqual(exact_rows, [])
        self.assertEqual(heuristic, [])

    def test_rows_sorted_by_ts_ms(self):
        tx_late = self._tx_with_event(_REAL_EVENT_1_MINT, block_time=200)
        tx_early = self._tx_with_event(_REAL_EVENT_1_MINT, block_time=100)
        exact_rows, _ = extract_rows_exact([tx_late, tx_early], _REAL_EVENT_1_MINT, sol_price=200.0)
        self.assertEqual([r["ts_ms"] for r in exact_rows], sorted(r["ts_ms"] for r in exact_rows))


class TestXvalGate(unittest.TestCase):

    def test_diffs_computed_when_poll_and_reconstructed_both_present(self):
        alert_ts = 1_700_000_000.0
        rows = [
            {"ts_ms": int((alert_ts + 55) * 1000), "price_usd": 1.05},
            {"ts_ms": int((alert_ts + 175) * 1000), "price_usd": 1.10},
        ]
        token_row = {"price_t1m": 1.0, "price_t3m": 1.10, "price_t10m": None,
                      "pct_change_peak": None}
        diffs = compute_xval_diffs(rows, token_row, alert_ts)
        by_label = {d.label: d for d in diffs}
        self.assertAlmostEqual(by_label["T1m"].pct_diff, 5.0, places=6)
        self.assertAlmostEqual(by_label["T3m"].pct_diff, 0.0, places=6)
        self.assertIsNone(by_label["T10m"].pct_diff)

    def test_no_reconstructed_rows_before_offset_gives_none(self):
        alert_ts = 1_700_000_000.0
        rows = [{"ts_ms": int((alert_ts + 700) * 1000), "price_usd": 2.0}]  # only after T10m
        token_row = {"price_t1m": 1.0, "price_t3m": 1.0, "price_t10m": 1.0, "pct_change_peak": None}
        diffs = compute_xval_diffs(rows, token_row, alert_ts)
        by_label = {d.label: d for d in diffs}
        self.assertIsNone(by_label["T1m"].pct_diff)

    def test_peak_diff_uses_pct_change_peak_and_t1m_reference(self):
        alert_ts = 1_700_000_000.0
        rows = [{"ts_ms": int((alert_ts + 60) * 1000), "price_usd": 1.5}]
        token_row = {"price_t1m": 1.0, "price_t3m": None, "price_t10m": None, "pct_change_peak": 50.0}
        diffs = compute_xval_diffs(rows, token_row, alert_ts)
        peak = next(d for d in diffs if d.label == "peak")
        self.assertAlmostEqual(peak.poll_price, 1.5, places=6)
        self.assertAlmostEqual(peak.pct_diff, 0.0, places=6)

    def test_classify_pass_within_tolerance(self):
        diffs = [XvalOffsetDiff(label="T1m", reconstructed_price=1.0, poll_price=1.0, pct_diff=2.0)]
        result = classify_xval(diffs, tolerance_pct=5.0)
        self.assertEqual(result.status, "PASS")

    def test_classify_fail_beyond_tolerance(self):
        diffs = [XvalOffsetDiff(label="T1m", reconstructed_price=1.0, poll_price=1.0, pct_diff=12.0)]
        result = classify_xval(diffs, tolerance_pct=5.0)
        self.assertEqual(result.status, "FAIL")
        self.assertAlmostEqual(result.max_abs_pct_diff, 12.0, places=6)

    def test_classify_insufficient_data_when_all_diffs_none(self):
        diffs = [XvalOffsetDiff(label="T1m", reconstructed_price=None, poll_price=None, pct_diff=None)]
        result = classify_xval(diffs, tolerance_pct=5.0)
        self.assertEqual(result.status, "INSUFFICIENT_DATA")

    def test_classify_uses_max_abs_across_multiple_offsets(self):
        diffs = [
            XvalOffsetDiff(label="T1m", reconstructed_price=1.0, poll_price=1.0, pct_diff=-1.0),
            XvalOffsetDiff(label="T3m", reconstructed_price=1.0, poll_price=1.0, pct_diff=9.0),
        ]
        result = classify_xval(diffs, tolerance_pct=5.0)
        self.assertEqual(result.status, "FAIL")
        self.assertAlmostEqual(result.max_abs_pct_diff, 9.0, places=6)


if __name__ == "__main__":
    unittest.main()
