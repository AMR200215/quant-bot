"""E2: Jupiter quote future is off the critical path — build/send never await it."""
import sys, os, time, re, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def _read(relpath):
    with open(os.path.join(REPO, relpath)) as f:
        return f.read()

class TestE2QuoteOffCriticalPath(unittest.TestCase):

    def test_quote_result_not_joined_before_land_start(self):
        """Structural: _quote_fut.result() must not appear inside the PP backend if/else block.

        The PP backend block spans from 'if EXECUTOR_BACKEND == "pumpportal":' to the
        closing 'else: # Jupiter' branch. _t_land_start is set *after* this if/else,
        so the invariant is that _quote_fut.result( is absent from the entire PP arm —
        it is only collected post-confirm in a separate block.
        """
        content = _read("memecoin/executor.py")
        # Find the PP backend block (everything inside the if arm, before 'else: # Jupiter')
        match = re.search(
            r'if EXECUTOR_BACKEND == .pumpportal.(.+?)else:\s+# Jupiter',
            content, re.DOTALL
        )
        self.assertIsNotNone(match, "Could not find PP backend block")
        pp_block = match.group(1)
        self.assertNotIn(
            "_quote_fut.result(",
            pp_block,
            "Quote future is joined inside the PP backend build/send block — it IS on the critical path"
        )

    def test_quote_ms_in_timing_dict(self):
        """E1 instrument: quote_ms must appear in the buy() timing return dict."""
        content = _read("memecoin/executor.py")
        self.assertIn("quote_ms", content,
                      "quote_ms not found in executor.py — E1 instrumentation missing")

if __name__ == "__main__":
    unittest.main(verbosity=2)
