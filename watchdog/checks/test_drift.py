"""watchdog/checks/test_drift.py — W8: test-suite drift detection.

Two independent checks, both static/subprocess-bounded, no arbitrary
production-module imports inside the watchdog process itself (per the
design spec's explicit safety requirement):

1. check_stale_mocks() — the exact class of bug that made
   test_v8_paper.py silently fail for 3 days during PROGRESS-FIX (it
   mocked a symbol a prior commit had already removed, and nobody
   noticed until a full-suite run was done by hand). Pure AST parsing of
   both the test file (to extract patch() targets) and the target
   module's source (to check the symbol still exists there) -- never
   imports either file. Reports WARN, not CRITICAL: static analysis of
   dynamic attributes, __getattr__, or runtime-constructed names can
   false-positive, so this needs a human glance, not an auto-page.

2. check_test_collection() — runs `pytest --collect-only` as a bounded
   subprocess per known test directory, separately (not combined -- the
   combined tree has a pre-existing, unrelated sys.modules stubbing
   collision across some files, confirmed during V8-TWIN-FIX and out of
   scope here). Collection failure (import error, syntax error) is a
   real, distinct problem from a test merely failing its assertions, and
   is worth its own signal.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from watchdog.checks import CheckResult, STATUS_CRITICAL, STATUS_OK, STATUS_UNKNOWN, STATUS_WARN, cap_severity

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_KNOWN_PACKAGES = {"memecoin", "research", "watchdog", "app", "tools", "sniper", "wallet_db"}
_COLLECT_TIMEOUT_S = 60


# ---------------------------------------------------------------------------
# check_stale_mocks
# ---------------------------------------------------------------------------

def _find_patch_targets(test_source: str) -> list[str]:
    """Extract string-literal patch()/patch.object() targets via AST --
    only handles the literal-string form (the common case in this repo);
    dynamically-constructed target strings are invisible to this and that
    is intentionally fine (we'd rather miss a target than misresolve one
    built at runtime)."""
    targets: list[str] = []
    try:
        tree = ast.parse(test_source)
    except SyntaxError:
        return targets

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name not in ("patch", "object"):
            continue
        # patch("a.b.c")
        if isinstance(func, ast.Name) or (isinstance(func, ast.Attribute) and func.attr == "patch"):
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                targets.append(node.args[0].value)
        # patch.object(module_or_obj, "symbol") -- only handle the case
        # where the first arg is itself a dotted Name (module reference by
        # import alias); skip local variables/fixtures, which we can't
        # resolve statically anyway.
        if isinstance(func, ast.Attribute) and func.attr == "object" and len(node.args) >= 2:
            first = node.args[0]
            second = node.args[1]
            dotted = _dotted_name(first)
            if dotted and isinstance(second, ast.Constant) and isinstance(second.value, str):
                targets.append(f"{dotted}.{second.value}")
    return targets


def _dotted_name(node: ast.AST) -> Optional[str]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _resolve_module_file(module_path: str, repo_root: Path) -> Optional[Path]:
    top = module_path.split(".", 1)[0]
    if top not in _KNOWN_PACKAGES:
        return None  # third-party/stdlib -- do not attempt to resolve
    parts = module_path.split(".")
    candidate = repo_root.joinpath(*parts).with_suffix(".py")
    if candidate.exists():
        return candidate
    pkg_init = repo_root.joinpath(*parts, "__init__.py")
    if pkg_init.exists():
        return pkg_init
    return None


def _iter_module_level_statements(body: list[ast.stmt]):
    """Yields statements that live in module scope, descending into
    try/except and if/else bodies (which do NOT introduce a new scope --
    a name assigned/imported inside one is still a module attribute).
    Does not descend into function/class bodies, which do introduce a new
    scope. This specifically fixes a real false positive found live: a
    defensive `try: from x import y \\n except ImportError: y = None`
    pattern (memecoin/journal_reconciler.py's read_sol_delta, which its
    own code comment explicitly says "Tests patch this name directly")
    was invisible to a naive top-level-only scan."""
    for node in body:
        yield node
        if isinstance(node, ast.Try):
            yield from _iter_module_level_statements(node.body)
            for h in node.handlers:
                yield from _iter_module_level_statements(h.body)
            yield from _iter_module_level_statements(node.orelse)
            yield from _iter_module_level_statements(node.finalbody)
        elif isinstance(node, ast.If):
            yield from _iter_module_level_statements(node.body)
            yield from _iter_module_level_statements(node.orelse)


def _module_defines_symbol(module_file: Path, symbol: str) -> Optional[bool]:
    """Returns True/False if statically determinable, None if the file
    can't be parsed (never guess in that case)."""
    try:
        tree = ast.parse(module_file.read_text())
    except Exception:
        return None
    for node in _iter_module_level_statements(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == symbol:
            return True
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == symbol:
                    return True
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == symbol:
            return True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if (alias.asname or alias.name.split(".")[-1]) == symbol:
                    return True
    return False


def check_stale_mocks(test_dirs: Optional[list[Path]] = None,
                       severity_ceiling: str = "WARN",
                       check_id: str = "test_drift.stale_mocks") -> list[CheckResult]:
    dirs = test_dirs if test_dirs is not None else [
        REPO_ROOT / "memecoin" / "tests", REPO_ROOT / "research" / "tests",
        REPO_ROOT / "watchdog" / "tests", REPO_ROOT / "tests",
    ]

    findings = []
    checked_targets = 0
    for d in dirs:
        if not d.exists():
            continue
        for f in sorted(d.glob("test_*.py")):
            try:
                source = f.read_text()
            except Exception:
                continue
            for target in _find_patch_targets(source):
                if "." not in target:
                    continue
                module_path, symbol = target.rsplit(".", 1)
                module_file = _resolve_module_file(module_path, REPO_ROOT)
                if module_file is None:
                    continue  # not one of our own packages, or unresolvable -- skip, don't guess
                checked_targets += 1
                exists = _module_defines_symbol(module_file, symbol)
                if exists is False:
                    findings.append({
                        "test_file": str(f.relative_to(REPO_ROOT)),
                        "target": target,
                        "resolved_module_file": str(module_file.relative_to(REPO_ROOT)),
                    })

    if not findings:
        return [CheckResult(
            check_id=check_id, status=STATUS_OK,
            reason=f"no stale patch() targets found ({checked_targets} resolvable targets checked "
                   f"across own-package modules; targets in third-party/stdlib modules are skipped, "
                   f"not verified)",
        )]

    return [CheckResult(
        check_id=check_id, status=cap_severity(STATUS_WARN, severity_ceiling),
        reason=(f"{len(findings)} patch() target(s) reference a symbol that no longer appears "
                f"(as a top-level def/class/assignment/import) in its resolved module -- possible "
                f"stale mock, needs a human look (static AST analysis can false-positive on "
                f"dynamic attributes; this is TEST_TARGET_STALE, not an auto-fix)"),
        severity=severity_ceiling,
        evidence={"findings": findings[:10], "total_findings": len(findings)},
    )]


# ---------------------------------------------------------------------------
# check_test_collection
# ---------------------------------------------------------------------------

def check_test_collection(test_dirs: Optional[list[Path]] = None,
                           python_bin: Optional[str] = None,
                           severity_ceiling: str = "WARN",
                           check_id_prefix: str = "test_drift.collection") -> list[CheckResult]:
    # Default to the CURRENTLY RUNNING interpreter (sys.executable), not a
    # bare "python3" -- found live: "python3" resolves to system Python on
    # a dev machine (no croniter/PyYAML installed there), producing a
    # false ModuleNotFoundError against watchdog's OWN test suite, which
    # actually passes 71/71 when run through the real venv. On the VPS,
    # sys.executable is already correct since systemd's ExecStart invokes
    # .venv/bin/python directly -- this fix matters most for local/dev
    # verification, but is the right default in both places regardless.
    python_bin = python_bin or sys.executable
    dirs = test_dirs if test_dirs is not None else [
        REPO_ROOT / "memecoin" / "tests", REPO_ROOT / "research" / "tests",
        REPO_ROOT / "watchdog" / "tests", REPO_ROOT / "tests",
    ]

    results = []
    for d in dirs:
        check_id = f"{check_id_prefix}.{d.name}_{d.parent.name}"
        if not d.exists():
            results.append(CheckResult(check_id=check_id, status=STATUS_UNKNOWN,
                                        reason=f"test directory does not exist: {d}",
                                        severity=severity_ceiling))
            continue
        try:
            proc = subprocess.run(
                [python_bin, "-m", "pytest", "--collect-only", "-q", str(d)],
                cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=_COLLECT_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            results.append(CheckResult(
                check_id=check_id, status=cap_severity(STATUS_WARN, severity_ceiling),
                reason=f"pytest --collect-only timed out after {_COLLECT_TIMEOUT_S}s for {d}",
                severity=severity_ceiling,
            ))
            continue
        except Exception as e:
            results.append(CheckResult(check_id=check_id, status=STATUS_UNKNOWN,
                                        reason=f"failed to run pytest --collect-only: {e!r}",
                                        severity=severity_ceiling))
            continue

        if proc.returncode != 0:
            results.append(CheckResult(
                check_id=check_id, status=cap_severity(STATUS_WARN, severity_ceiling),
                reason=f"pytest --collect-only failed (exit {proc.returncode}) for {d} — "
                       f"import/syntax error, not just a failing assertion",
                severity=severity_ceiling,
                evidence={"stdout_tail": proc.stdout[-1500:], "stderr_tail": proc.stderr[-1500:]},
            ))
        else:
            collected = proc.stdout.count(" tests collected") or proc.stdout.count(" test collected")
            results.append(CheckResult(
                check_id=check_id, status=STATUS_OK,
                reason=f"collects cleanly ({d})",
                evidence={"stdout_tail": proc.stdout[-300:]},
            ))
    return results
