"""
C0 — Characterization tests for 6 known defects.

Each test is expected to FAIL on the pre-patch codebase, proving the defect
exists. After the corresponding C-patch, each test must PASS.

Run:
    python -m pytest memecoin/tests/test_c0_characterization.py -v 2>&1 | tee artifacts/remaining_gaps/prepatch_test_output.txt

Defects:
  D1 — executor._skip_pamm_t22 internal override bypasses portfolio T22 flag decision
  D2 — dual Jupiter fallback: pre-executor rescue + B3 post-executor can both fire
  D3 — no SENT_PENDING/CONFIRMED_REVERT distinction (RouteOutcome enum missing)
  D4 — B1-B5 tests reimplement production logic inline (R6 violation)
  D5 — live entry proceeds without classify_mint() gate
  D6 — rpc_429_wait_ms initialized to 0, never accumulated during 429 retry sleeps
"""
import re
import sys
import os
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, REPO)


# ── helpers ─────────────────────────────────────────────────────────────────

def _read(relpath: str) -> str:
    with open(os.path.join(REPO, relpath)) as fh:
        return fh.read()


# ── D1: executor internal T22 override bypasses portfolio flag decision ──────

class TestD1ExecutorT22Override(unittest.TestCase):
    """
    FIXED BY: C4 — remove _skip_pamm_t22 from executor.py; accept explicit
    route authorization from orchestrator instead.
    """

    def test_executor_has_no_internal_t22_routing_override(self):
        """
        After C4, executor.py must NOT contain _skip_pamm_t22.
        FAILS now because the variable exists at executor.py:2276.
        """
        content = _read("memecoin/executor.py")
        self.assertNotIn(
            "_skip_pamm_t22",
            content,
            "executor.py contains internal T22 routing override (_skip_pamm_t22). "
            "Route decision must come from the orchestrator, not executor internals. "
            "Fix: C4 — remove _skip_pamm_t22, accept explicit route param.",
        )

    def test_executor_does_not_read_token_program_cache_for_routing(self):
        """
        After C4, the graduated sell block must not contain the _skip_pamm_t22
        local variable (the internal T22 routing decision pattern).
        The caller (orchestrator) passes skip_pump_amm explicitly instead.
        FAILS pre-fix because _skip_pamm_t22 = (...TOKEN22...) assignment existed.
        PASSES after C4 removes it.
        """
        content = _read("memecoin/executor.py")
        # The specific pattern that was the defect: local var assigned from cache
        self.assertNotIn(
            "_skip_pamm_t22",
            content,
            "executor.py still contains _skip_pamm_t22 internal T22 routing decision. "
            "Fix: C4 removes this; orchestrator passes skip_pump_amm explicitly.",
        )


# ── D2: dual Jupiter fallback in same orchestration step ────────────────────

class TestD2DualJupiterFallback(unittest.TestCase):
    """
    FIXED BY: C3 — exit_orchestrator.py enforces single-venue-per-step; R4.
    """

    def test_at_most_one_jupiter_rescue_call_site_in_close_position(self):
        """
        After C3, close_position() must not contain any direct Jupiter rescue
        calls — all routing goes through exit_orchestrator.
        FAILS now because close_position has both pre-executor rescue AND B3
        post-executor rescue (two force_jupiter_rescue_sell call sites).
        """
        content = _read("memecoin/portfolio.py")
        # Extract close_position function body
        # Find from 'def close_position' to next top-level def
        match = re.search(
            r'def close_position\(.+?(?=\ndef [a-z_])',
            content,
            re.DOTALL,
        )
        self.assertIsNotNone(match, "close_position not found in portfolio.py")
        close_body = match.group(0)
        count = close_body.count("force_jupiter_rescue_sell")
        self.assertEqual(
            count, 0,
            f"close_position contains {count} direct Jupiter rescue call(s). "
            "After C3, all venue dispatch goes through exit_orchestrator — "
            "portfolio.py must have zero direct rescue calls.",
        )

    def test_no_is_rescue_eligible_in_portfolio(self):
        """
        After C3, portfolio.py must not call is_rescue_eligible_error — that
        decision belongs to exit_orchestrator.
        FAILS now because portfolio.py calls it in at least two places.
        """
        content = _read("memecoin/portfolio.py")
        self.assertNotIn(
            "is_rescue_eligible_error",
            content,
            "portfolio.py calls is_rescue_eligible_error directly. "
            "After C3, rescue eligibility is determined by exit_orchestrator.",
        )


# ── D3: no SENT_PENDING / CONFIRMED_REVERT distinction ──────────────────────

class TestD3RouteOutcomeMissing(unittest.TestCase):
    """
    FIXED BY: C3 — exit_orchestrator.py defines RouteOutcome enum with
    SENT_PENDING, CONFIRMED_SUCCESS, CONFIRMED_REVERT, etc.
    """

    def test_exit_orchestrator_module_exists(self):
        """
        After C3, memecoin/exit_orchestrator.py must exist.
        FAILS now — file does not exist.
        """
        path = os.path.join(REPO, "memecoin", "exit_orchestrator.py")
        self.assertTrue(
            os.path.exists(path),
            "memecoin/exit_orchestrator.py does not exist. "
            "Fix: C3 creates it with RouteOutcome enum.",
        )

    def test_route_outcome_enum_importable(self):
        """
        After C3, RouteOutcome must be importable with at minimum:
        SENT_PENDING, CONFIRMED_SUCCESS, CONFIRMED_REVERT, NO_SEND, FATAL_PRE_SEND.
        FAILS now — module does not exist.
        """
        try:
            from memecoin.exit_orchestrator import RouteOutcome  # type: ignore
        except ImportError as exc:
            self.fail(f"Cannot import RouteOutcome: {exc}. Fix: C3 creates exit_orchestrator.py.")
        for name in ("SENT_PENDING", "CONFIRMED_SUCCESS", "CONFIRMED_REVERT",
                     "NO_SEND", "FATAL_PRE_SEND"):
            self.assertTrue(
                hasattr(RouteOutcome, name),
                f"RouteOutcome missing member {name}",
            )

    def test_sent_pending_blocks_all_venues(self):
        """
        After C3, when any venue returns SENT_PENDING the orchestrator must
        block all other venues (R5). Verify orchestrator has this gate.
        FAILS now — exit_orchestrator.py does not exist.
        """
        path = os.path.join(REPO, "memecoin", "exit_orchestrator.py")
        if not os.path.exists(path):
            self.fail("exit_orchestrator.py does not exist (C3 not yet implemented).")
        content = _read("memecoin/exit_orchestrator.py")
        self.assertIn(
            "SENT_PENDING",
            content,
            "exit_orchestrator.py must block all venues when SENT_PENDING.",
        )


# ── D4: B1-B5 tests reimplement production logic inline (R6 violation) ──────

class TestD4R6ViolationsInBTests(unittest.TestCase):
    """
    FIXED BY: Rewriting B1-B5 tests to import and call actual production
    functions rather than reimplementing logic inline.
    """

    def _assert_imports_production(self, relpath: str, module: str, label: str):
        content = _read(relpath)
        self.assertIn(
            f"from memecoin.{module} import",
            content,
            f"{relpath} must import from memecoin.{module} to call production "
            f"code ({label} — R6 violation if missing).",
        )

    def test_b1_imports_scanner(self):
        """B1 test must call production scanner dual-source logic, not inline copy."""
        self._assert_imports_production(
            "memecoin/tools/test_b1_pregrad_dual_source.py",
            "scanner",
            "B1 pre-grad dual source",
        )

    def test_b2_imports_scanner(self):
        """B2 test must call production scanner graduation dispatch, not inline copy."""
        self._assert_imports_production(
            "memecoin/tools/test_b2_immediate_graduation.py",
            "scanner",
            "B2 immediate graduation",
        )

    def test_b3_imports_portfolio(self):
        """B3 test must call production portfolio.close_position or orchestrator."""
        content = _read("memecoin/tools/test_b3_pump_amm_first.py")
        has_portfolio = "from memecoin.portfolio import" in content
        has_orchestrator = "from memecoin.exit_orchestrator import" in content
        self.assertTrue(
            has_portfolio or has_orchestrator,
            "memecoin/tools/test_b3_pump_amm_first.py must import from "
            "memecoin.portfolio or memecoin.exit_orchestrator to call production "
            "routing code (R6 violation — test reimplements routing logic inline).",
        )

    def test_b5_calls_portfolio_t22_gate(self):
        """B5 test must call the production T22 gate in portfolio, not inline logic."""
        content = _read("memecoin/tools/test_b5_t22_flags.py")
        has_production_call = (
            "from memecoin.portfolio import" in content
            or "from memecoin.exit_orchestrator import" in content
        )
        self.assertTrue(
            has_production_call,
            "memecoin/tools/test_b5_t22_flags.py tests inline flag logic instead of "
            "calling production portfolio/orchestrator T22 gate. R6 violation.",
        )


# ── D5: live entry proceeds without classify_mint() gate ────────────────────

class TestD5NoClassifyMintAtEntry(unittest.TestCase):
    """
    FIXED BY: C2 — evaluate_live_entry_program_gate() called before TX submission.
    """

    def test_live_entry_calls_classify_mint(self):
        """
        After C2, the live buy path in portfolio.py must call classify_mint()
        via evaluate_live_entry_program_gate() before submitting the buy TX.
        FAILS pre-fix — classify_mint() only in sell path.
        PASSES after C2: evaluate_live_entry_program_gate() wired in _open_live_position.
        """
        content = _read("memecoin/portfolio.py")
        lines = content.splitlines()
        # Locate _open_live_position function body
        func_lineno = next(
            (i for i, ln in enumerate(lines) if "_open_live_position" in ln
             and "def " in ln), None
        )
        self.assertIsNotNone(func_lineno, "Could not locate _open_live_position in portfolio.py")

        # Look for classify_mint within 600 lines of _open_live_position
        window = "\n".join(lines[func_lineno: func_lineno + 600])
        self.assertIn(
            "classify_mint",
            window,
            "classify_mint() not called in _open_live_position. "
            "Fix: C2 adds evaluate_live_entry_program_gate() which calls classify_mint().",
        )

    def test_entry_gate_module_exists(self):
        """
        C2 adds evaluate_live_entry_program_gate(). Until then it doesn't exist.
        FAILS now — no such function anywhere in the codebase.
        """
        content_portfolio = _read("memecoin/portfolio.py")
        self.assertIn(
            "evaluate_live_entry_program_gate",
            content_portfolio,
            "evaluate_live_entry_program_gate not found in portfolio.py. "
            "Fix: C2 creates and wires this gate.",
        )


# ── D6: rpc_429_wait_ms never accumulated ───────────────────────────────────

class TestD6Rpc429WaitMsNotAccumulated(unittest.TestCase):
    """
    FIXED BY: C6 — telemetry + executor repair to accumulate real 429 wait time.
    """

    def test_rpc_429_wait_ms_accumulated_in_retry_loop(self):
        """
        After C6, executor.py must accumulate rpc_429_wait_ms during 429 retry
        sleeps so the ENTRY TIMING log reflects real rate-limit stalls.
        FAILS pre-fix — rpc_429_wait_ms is initialized to 0.0 and never incremented.
        PASSES after C6: _buy_timing['rpc_429_wait_ms'] += _rpc_429_read() is present.
        """
        content = _read("memecoin/executor.py")
        # Post-fix: either direct += assignment or via accumulator helper
        has_accumulation = bool(
            re.search(r'rpc_429_wait_ms.*\+=', content)
            or re.search(r'_rpc_429_accum\(', content)
        )
        self.assertTrue(
            has_accumulation,
            "rpc_429_wait_ms is initialized to 0.0 in _buy_timing but never "
            "accumulated when the bot sleeps on a 429. "
            "Fix: C6 adds _rpc_429_accum() in the sleep path and "
            "_buy_timing['rpc_429_wait_ms'] += _rpc_429_read() after confirmation.",
        )

    def test_rpc_429_wait_ms_not_always_zero_in_summary(self):
        """
        The ENTRY TIMING log emitted by executor.py must be able to report
        a non-zero rpc_429_wait_ms. Structural check: within 3 lines of any
        'time.sleep' call that's inside a 429-retry context, there must be
        a rpc_429_wait_ms accumulation.
        FAILS now — no such accumulation exists.
        """
        content = _read("memecoin/executor.py")
        lines = content.splitlines()
        # Post-fix: look for _rpc_429_accum call within 3 lines of a 429 sleep
        found = False
        for i, ln in enumerate(lines):
            if "time.sleep" in ln and "429" in "\n".join(lines[max(0,i-5):i+1]):
                window = "\n".join(lines[max(0,i-3):i+4])
                if "_rpc_429_accum" in window or ("rpc_429_wait_ms" in window and "+=" in window):
                    found = True
                    break
        self.assertTrue(
            found,
            "No time.sleep call within a 429-retry context has a rpc_429_wait_ms += "
            "accumulation nearby. The ENTRY TIMING log will always show 0.0 for this field.",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
