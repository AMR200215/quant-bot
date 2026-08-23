"""research/tests/test_v8_collection_yield.py — READINESS DENOMINATOR
AUDIT/CORRECTION batch: era/admission-restricted collector-yield
accounting, as opposed to whole-historical-population denominators.

Run: python -m pytest research/tests/test_v8_collection_yield.py -v
"""

import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from research.v8_collection_yield import (
    compute_collection_yield, trustworthy_collection_era_start, pp_funded_era_start,
    PRICE_CORRECTION_DEPLOY_UTC, load_admission_log_by_mint,
)
from research.v8_candidate_registry import CANDIDATES
from research.path_schema import PATH_HEADER

_BASELINE = next(c for c in CANDIDATES if c["candidate_id"] == "BASELINE-0")


def _write_path_file(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PATH_HEADER)
        w.writeheader()
        for r in rows:
            full = {k: "" for k in PATH_HEADER}
            full.update(r)
            w.writerow(full)


def _real_row(ts_ms, price_usd="0.00005"):
    return {
        "schema_version": "3", "ts_ms": str(ts_ms), "price_usd": price_usd, "price_sol": "0.0000003",
        "vsol": "50.0", "vtok": "1000000000", "venue_state": "CURVE_ACTIVE", "source": "live_pp",
        "backfilled": "false", "data_status": "ok",
    }


def _backfilled_row(ts_ms):
    row = _real_row(ts_ms)
    row["backfilled"] = "true"
    return row


def _write_admission_log(root: Path, rows: list):
    out = root / "logs" / "research_admission"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "admission_log.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _admission_row(mint, ts, admitted=True, probability=1.0):
    return {
        "ts": ts, "token_address": mint, "utc_hour": 0, "path_eligible": True,
        "path_admitted": admitted, "path_sampling_probability": probability,
        "admission_reason": "under_hourly_pace" if probability >= 1.0 else "sampled_admit",
        "budget_used": 0, "budget_remaining": 100000,
    }


def _write_proxy_log(root: Path, rows: list):
    out = root / "logs" / "research_execution_proxy"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "execution_proxy_log.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _event(mint, alert_iso, path_file=None):
    return {
        "event_id": f"e_{mint}", "token_address": mint, "alert_time": alert_iso,
        "progress_at_signal": 0.3, "venue_state_at_signal": "CURVE_ACTIVE",
        "path_file": path_file, "progress_capture_lag_ms": 500, "pct_change_peak": 10.0,
    }


_IN_ERA_ISO = "2026-08-20T12:00:00+00:00"
_IN_ERA_TS_MS = int(datetime.fromisoformat(_IN_ERA_ISO).timestamp() * 1000)
_PRE_ERA_ISO = "2026-08-01T12:00:00+00:00"


class TestEraBoundary(unittest.TestCase):

    def test_undetermined_when_no_evidence_at_all(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertIsNone(pp_funded_era_start(root, execution_proxy_rows=[]))
            self.assertIsNone(trustworthy_collection_era_start(root, execution_proxy_rows=[]))

    def test_derived_from_earliest_proxy_observation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = [
                {"token_address": "A", "status": "OK", "observed_at": "2026-08-22T23:48:17+00:00"},
                {"token_address": "B", "status": "OK", "observed_at": "2026-08-22T20:00:00+00:00"},
            ]
            start = pp_funded_era_start(root, execution_proxy_rows=rows)
            self.assertEqual(start.isoformat(), "2026-08-22T20:00:00+00:00")

    def test_trustworthy_era_is_later_of_price_fix_and_funding(self):
        """Funding observed AFTER the price-correction deploy -> funding
        is the binding constraint."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = [{"token_address": "A", "status": "OK", "observed_at": "2026-08-22T00:00:00+00:00"}]
            era = trustworthy_collection_era_start(root, execution_proxy_rows=rows)
            self.assertEqual(era.isoformat(), "2026-08-22T00:00:00+00:00")

    def test_price_fix_deploy_wins_if_funding_predates_it(self):
        """Synthetic/adversarial case: even if some proxy row somehow
        predates the price-fix deploy, the era start can never be
        earlier than PRICE_CORRECTION_DEPLOY_UTC."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = [{"token_address": "A", "status": "OK", "observed_at": "2026-01-01T00:00:00+00:00"}]
            era = trustworthy_collection_era_start(root, execution_proxy_rows=rows)
            self.assertEqual(era, PRICE_CORRECTION_DEPLOY_UTC)


class TestComputeCollectionYield(unittest.TestCase):

    def test_era_undetermined_gives_zero_eligible_not_a_crash(self):
        events = [_event("A", _IN_ERA_ISO)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = compute_collection_yield(events, events, _BASELINE, root, execution_proxy_rows=[])
        self.assertTrue(result.era_undetermined)
        self.assertEqual(result.path_collection_eligible_n, 0)
        self.assertEqual(result.candidate_venue_qualified_n, 1)  # population preserved

    def test_pre_era_events_excluded_from_eligible_but_kept_in_population(self):
        events = [_event("A", _PRE_ERA_ISO), _event("B", _IN_ERA_ISO)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proxy_rows = [{"token_address": "ANCHOR", "status": "OK", "observed_at": _IN_ERA_ISO}]
            result = compute_collection_yield(events, events, _BASELINE, root, execution_proxy_rows=proxy_rows)
        self.assertEqual(result.candidate_venue_qualified_n, 2)   # population: both preserved
        self.assertEqual(result.path_collection_eligible_n, 1)    # only the in-era one

    def test_no_admission_record_surfaced_separately(self):
        events = [_event("A", _IN_ERA_ISO)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proxy_rows = [{"token_address": "ANCHOR", "status": "OK", "observed_at": _IN_ERA_ISO}]
            # No admission_log.jsonl written at all.
            result = compute_collection_yield(events, events, _BASELINE, root, execution_proxy_rows=proxy_rows)
        self.assertEqual(result.path_collection_eligible_n, 1)
        self.assertEqual(result.no_admission_record_n, 1)
        self.assertEqual(result.path_admitted_n, 0)

    def test_sampled_reject_is_eligible_but_not_admitted(self):
        events = [_event("A", _IN_ERA_ISO)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proxy_rows = [{"token_address": "ANCHOR", "status": "OK", "observed_at": _IN_ERA_ISO}]
            admission = {"A": [_admission_row("A", 1, admitted=False, probability=0.3)]}
            result = compute_collection_yield(events, events, _BASELINE, root,
                                               admission_log_by_mint=admission, execution_proxy_rows=proxy_rows)
        self.assertEqual(result.path_collection_eligible_n, 1)
        self.assertEqual(result.no_admission_record_n, 0)
        self.assertEqual(result.path_admitted_n, 0)

    def test_admitted_with_tick_and_valid_usable_path(self):
        mint = "AdmittedRealMint"
        rel = f"logs/research_paths/2026-08-20/{mint}.csv"
        events = [_event(mint, _IN_ERA_ISO, path_file=rel)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_path_file(root / rel, [_real_row(_IN_ERA_TS_MS), _real_row(_IN_ERA_TS_MS + 1000)])
            admission = {mint: [_admission_row(mint, 1, admitted=True, probability=1.0)]}
            proxy_rows = [{"token_address": mint, "status": "OK", "observed_at": _IN_ERA_ISO}]
            result = compute_collection_yield(events, events, _BASELINE, root,
                                               admission_log_by_mint=admission, execution_proxy_rows=proxy_rows)
        self.assertEqual(result.path_admitted_n, 1)
        self.assertEqual(result.admitted_with_tick_n, 1)
        self.assertEqual(result.admitted_with_valid_usable_path_n, 1)
        self.assertEqual(result.admitted_path_yield_pct, 100.0)
        self.assertEqual(result.execution_proxy_collection_eligible_n, 1)
        self.assertEqual(result.execution_proxy_observed_n, 1)
        self.assertEqual(result.execution_proxy_coverage_pct, 100.0)
        self.assertEqual(result.unique_forward_days, 1)

    def test_admitted_but_only_backfilled_data_does_not_count_as_a_tick(self):
        """A path file that only has backfilled rows (no live_pp data)
        must not count as natural forward yield -- backfilled data
        proves nothing about whether the live collector is working."""
        mint = "BackfilledOnlyMint"
        rel = f"logs/research_paths/2026-08-20/{mint}.csv"
        events = [_event(mint, _IN_ERA_ISO, path_file=rel)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_path_file(root / rel, [_backfilled_row(_IN_ERA_TS_MS)])
            admission = {mint: [_admission_row(mint, 1)]}
            result = compute_collection_yield(events, events, _BASELINE, root,
                                               admission_log_by_mint=admission,
                                               execution_proxy_rows=[{"token_address": "ANCHOR", "status": "OK",
                                                                       "observed_at": _IN_ERA_ISO}])
        self.assertEqual(result.path_admitted_n, 1)
        self.assertEqual(result.admitted_with_tick_n, 0)
        self.assertEqual(result.admitted_with_valid_usable_path_n, 0)

    def test_admitted_with_zero_real_trades_is_natural_yield_not_missing(self):
        """Admitted, in-era, but the mint simply had no real trades in
        its tracking window (header-only path file, or no path_file at
        all) -- admitted_with_tick_n correctly stays 0, path_admitted_n
        still counts it (it WAS a fair collection attempt)."""
        mint = "ThinYieldMint"
        events = [_event(mint, _IN_ERA_ISO, path_file=None)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            admission = {mint: [_admission_row(mint, 1)]}
            result = compute_collection_yield(events, events, _BASELINE, root,
                                               admission_log_by_mint=admission,
                                               execution_proxy_rows=[{"token_address": "ANCHOR", "status": "OK",
                                                                       "observed_at": _IN_ERA_ISO}])
        self.assertEqual(result.path_admitted_n, 1)
        self.assertEqual(result.admitted_with_tick_n, 0)
        self.assertEqual(result.admitted_path_yield_pct, 0.0)

    def test_gz_rotated_admitted_path_still_found(self):
        """Consistency with the earlier gz-rotation fix in
        v8_candidate_path_coverage.py -- this module must also resolve
        a *.csv.gz sibling when the recorded path_file has been rotated."""
        import gzip
        mint = "GzMint"
        rel = f"logs/research_paths/2026-08-20/{mint}.csv"
        events = [_event(mint, _IN_ERA_ISO, path_file=rel)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            plain = root / rel
            _write_path_file(plain, [_real_row(_IN_ERA_TS_MS), _real_row(_IN_ERA_TS_MS + 1000)])
            with open(plain, "rb") as f_in, gzip.open(str(plain) + ".gz", "wb") as f_out:
                f_out.write(f_in.read())
            plain.unlink()
            admission = {mint: [_admission_row(mint, 1)]}
            result = compute_collection_yield(events, events, _BASELINE, root,
                                               admission_log_by_mint=admission,
                                               execution_proxy_rows=[{"token_address": "ANCHOR", "status": "OK",
                                                                       "observed_at": _IN_ERA_ISO}])
        self.assertEqual(result.admitted_with_valid_usable_path_n, 1)

    def test_ambiguous_mints_excluded_from_every_count(self):
        mint = "AmbiguousMint"
        rel = f"logs/research_paths/2026-08-20/{mint}.csv"
        events = [_event(mint, _IN_ERA_ISO, path_file=rel), _event(mint, _IN_ERA_ISO, path_file=rel)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_path_file(root / rel, [_real_row(_IN_ERA_TS_MS)])
            admission = {mint: [_admission_row(mint, 1)]}
            result = compute_collection_yield(events, events, _BASELINE, root,
                                               admission_log_by_mint=admission,
                                               execution_proxy_rows=[{"token_address": "ANCHOR", "status": "OK",
                                                                       "observed_at": _IN_ERA_ISO}])
        self.assertEqual(result.candidate_venue_qualified_n, 2)  # population preserved (event-level)
        self.assertEqual(result.ambiguous_excluded_mints_n, 1)
        self.assertEqual(result.path_collection_eligible_n, 0)
        self.assertEqual(result.path_admitted_n, 0)

    def test_ipw_effective_n_shrinks_under_unequal_probabilities(self):
        events, admission, proxy_rows = [], {}, [{"token_address": "ANCHOR", "status": "OK", "observed_at": _IN_ERA_ISO}]
        for i in range(10):
            mint = f"M{i}"
            events.append(_event(mint, _IN_ERA_ISO))
            prob = 1.0 if i < 5 else 0.1   # 5 unpressured, 5 heavily downweighted-then-admitted
            admission[mint] = [_admission_row(mint, 1, admitted=True, probability=prob)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = compute_collection_yield(events, events, _BASELINE, root,
                                               admission_log_by_mint=admission, execution_proxy_rows=proxy_rows)
        self.assertEqual(result.path_admitted_n, 10)
        self.assertLess(result.ipw_effective_n, 10.0)   # unequal weights shrink Kish ESS below raw n

    def test_ipw_effective_n_equals_raw_n_when_all_probabilities_are_one(self):
        events, admission = [], {}
        proxy_rows = [{"token_address": "ANCHOR", "status": "OK", "observed_at": _IN_ERA_ISO}]
        for i in range(10):
            mint = f"M{i}"
            events.append(_event(mint, _IN_ERA_ISO))
            admission[mint] = [_admission_row(mint, 1, admitted=True, probability=1.0)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = compute_collection_yield(events, events, _BASELINE, root,
                                               admission_log_by_mint=admission, execution_proxy_rows=proxy_rows)
        self.assertEqual(result.ipw_effective_n, 10.0)

    def test_execution_proxy_numerator_and_denominator_are_both_mint_level(self):
        """The exact units-mismatch bug this audit was asked to check:
        with two events for one already-excluded ambiguous mint plus one
        clean mint, eligible_n and observed_n must both reflect the SAME
        (mint-level) units, never an event count vs a mint count."""
        clean_mint = "CleanMint"
        rel = f"logs/research_paths/2026-08-20/{clean_mint}.csv"
        amb_mint = "AmbiguousMint2"
        events = [
            _event(clean_mint, _IN_ERA_ISO, path_file=rel),
            _event(amb_mint, _IN_ERA_ISO), _event(amb_mint, "2026-08-20T13:00:00+00:00"),
        ]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_path_file(root / rel, [_real_row(_IN_ERA_TS_MS)])
            admission = {clean_mint: [_admission_row(clean_mint, 1)], amb_mint: [_admission_row(amb_mint, 1)]}
            proxy_rows = [
                {"token_address": clean_mint, "status": "OK", "observed_at": _IN_ERA_ISO},
                {"token_address": amb_mint, "status": "OK", "observed_at": _IN_ERA_ISO},
            ]
            result = compute_collection_yield(events, events, _BASELINE, root,
                                               admission_log_by_mint=admission, execution_proxy_rows=proxy_rows)
        # amb_mint is excluded entirely (ambiguous); only clean_mint is eligible/admitted/observed.
        self.assertEqual(result.execution_proxy_collection_eligible_n, 1)
        self.assertEqual(result.execution_proxy_observed_n, 1)
        self.assertEqual(result.execution_proxy_coverage_pct, 100.0)


class TestLoadAdmissionLogByMint(unittest.TestCase):

    def test_groups_rows_by_mint(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_admission_log(root, [_admission_row("A", 1), _admission_row("A", 2), _admission_row("B", 1)])
            by_mint = load_admission_log_by_mint(root)
        self.assertEqual(len(by_mint["A"]), 2)
        self.assertEqual(len(by_mint["B"]), 1)

    def test_missing_file_gives_empty_dict_not_error(self):
        with tempfile.TemporaryDirectory() as d:
            by_mint = load_admission_log_by_mint(Path(d))
        self.assertEqual(by_mint, {})


if __name__ == "__main__":
    unittest.main()
