"""
journal_io.py — Shared journal I/O primitives.

Provides JOURNAL_LOCK, a module-level threading.Lock that must be acquired
by all code that appends or rewrites a journal CSV file:

  - portfolio._append_journal()      — append one row
  - journal_reconciler._process_file() — rewrite corrected rows (Phase 2 only)

Rules
-----
- NEVER hold JOURNAL_LOCK during RPC or network calls.
- Journal writes (append / atomic rewrite) must happen under JOURNAL_LOCK.
- Reading (snapshot for reconciler Phase 1) does NOT require the lock.
"""

import threading

JOURNAL_LOCK = threading.Lock()
