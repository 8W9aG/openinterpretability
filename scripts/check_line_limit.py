"""Fail if any source file or function exceeds its line limit.

Files: at most 500 lines. Tests, configuration, lockfiles, vendored code, the
changelog, papers and generated data (audit records and wiki pages) are exempt.

Functions: at most 40 lines: the signature as one line, plus the body.
Decorators, the parameter list and the docstring do not count, so long Click
option lists are free. Tests are exempt, as for files. For the rare function
that reads better whole, put `# long-function: <reason>` on its `def` line.
(Ruff warns on unknown `# noqa` codes, so the marker is not a noqa.)

Run with `make lint`; CI runs it in the lint step.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path

MAX_LINES = 500
MAX_FUNCTION_LINES = 40
EXEMPT_MARKER = re.compile(r"#\s*long-function:\s*\S")

EXEMPT = (
    "openinterp/data/public_suffix_list.dat",  # a copy of publicsuffix.org's list, not code
    "tests/*",
    "wiki/*",  # generated experiment pages
    "docs/analyses/*.jsonl",  # generated analysis rows
    "data/pole-effects/*.json",  # generated tables (openinterp pole-effects)
    "papers/*/results*.jsonl",  # generated result rows of a paper
    "papers/*/README.md",  # a paper is one page, however long its argument
    "benchmark/*.jsonl",  # frozen benchmark data (openinterp benchmark freeze)
    "hf_models/*.jsonl",  # dictionary training logs
    ".claude/skills/asd-ste100/*",  # vendored as is from danyuchn/asd-ste100-skill
    "CHANGELOG.md",  # one file that grows with each release, not code
    "uv.lock",
    "*.toml",
    "*.yml",
    "*.yaml",
    ".gitignore",
    ".python-version",
    "LICENSE",
)


def tracked_files() -> list[str]:
    """Every tracked or new file that is not exempt."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [f for f in out.splitlines() if f and not any(fnmatch(f, p) for p in EXEMPT)]


def function_lengths(source: str) -> dict[str, tuple[int, bool]]:
    """Map each function's qualified name to (length, exempt by marker).

    The signature counts as one line, however it is wrapped. So a Click command's
    parameter list (one per option) doesn't count. The decorators and the docstring
    don't count either. The body counts from the first line after the
    docstring (or the first statement) to the last line.
    """
    lines = source.splitlines()
    found: dict[str, tuple[int, bool]] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                name = f"{prefix}{child.name}"
                first = child.body[0]
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                ):
                    start = first.end_lineno + 1
                else:
                    start = first.lineno
                length = 1 + max(0, child.end_lineno - start + 1)
                exempt = bool(EXEMPT_MARKER.search(lines[child.lineno - 1]))
                found[name] = (length, exempt)
                visit(child, f"{name}.")
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            else:
                visit(child, prefix)

    visit(ast.parse(source), "")
    return found


def check_functions(files: list[str]) -> list[str]:
    """One error for each function over :data:`MAX_FUNCTION_LINES`."""
    errors = []
    for name in files:
        if not name.endswith(".py"):
            continue
        try:
            source = Path(name).read_text(encoding="utf-8")
        except FileNotFoundError:
            continue  # deleted but still tracked
        for func, (length, exempt) in function_lengths(source).items():
            if length > MAX_FUNCTION_LINES and not exempt:
                errors.append(
                    f"{name}::{func}: {length} lines (max {MAX_FUNCTION_LINES}); split it, "
                    "or mark the def with `# long-function: <reason>`"
                )
    return errors


def main() -> int:
    """Print every file and function over its limit; return 1 if any is."""
    files = tracked_files()
    too_long = []
    for name in files:
        try:
            with open(name, encoding="utf-8") as fh:
                count = sum(1 for _ in fh)
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue  # binary or deleted files
        if count > MAX_LINES:
            too_long.append((count, name))
    for count, name in sorted(too_long, reverse=True):
        print(f"{name}: {count} lines (max {MAX_LINES}); split it into smaller modules")
    function_errors = check_functions(files)
    for error in function_errors:
        print(error)
    return 1 if too_long or function_errors else 0


if __name__ == "__main__":
    sys.exit(main())
