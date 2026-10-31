"""Fail if any source file or function exceeds its line limit.

Files: at most 500 lines. Tests, configuration, lockfiles and generated data
(audit records and wiki pages) are exempt.

Functions: at most 40 lines: the signature as one line, plus the body.
Decorators, the parameter list and the docstring do not count, so long Click
option lists are free. The limit is ratcheted: `scripts/long_functions.txt` lists the functions
that were already longer, with their lengths. A listed function may not grow,
and a function that is not listed may not pass the limit. When a listed
function shrinks, lower its entry; when it is split, remove the entry (the
check fails on stale entries, so the list only ratchets down). For the
rare function that reads better whole, put `# long-function: <reason>` on its
`def` line. (Ruff warns on unknown `# noqa` codes, so the marker is not a noqa.)

Run via `make lint`; CI runs it in the lint step.
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
ALLOWLIST = Path(__file__).with_name("long_functions.txt")
EXEMPT_MARKER = re.compile(r"#\s*long-function:\s*\S")

EXEMPT = (
    "tests/*",
    "wiki/*",  # generated experiment pages
    "docs/analyses/*.jsonl",  # generated analysis rows
    "hf_models/*.jsonl",  # dictionary training logs
    "uv.lock",
    "*.toml",
    "*.yml",
    "*.yaml",
    ".gitignore",
    ".python-version",
    "LICENSE",
)


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [f for f in out.splitlines() if f and not any(fnmatch(f, p) for p in EXEMPT)]


def function_lengths(source: str) -> dict[str, tuple[int, bool]]:
    """Map each function's qualified name to (length, exempt by marker).

    The signature counts as one line, however it is wrapped, so a Click
    command's parameter list (one per option) doesn't count; nor do the
    decorators or the docstring. The body counts from the first line after the
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


def read_allowlist() -> dict[str, int]:
    allowed = {}
    for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            key, length = line.rsplit(maxsplit=1)
            allowed[key] = int(length)
    return allowed


def check_functions(files: list[str]) -> list[str]:
    allowed = read_allowlist()
    errors = []
    seen: dict[str, int] = {}
    for name in files:
        if not name.endswith(".py"):
            continue
        try:
            source = Path(name).read_text(encoding="utf-8")
        except FileNotFoundError:
            continue  # deleted but still tracked
        for func, (length, exempt) in function_lengths(source).items():
            key = f"{name}::{func}"
            seen[key] = length
            if key in allowed or exempt or length <= MAX_FUNCTION_LINES:
                continue
            errors.append(
                f"{key}: {length} lines (max {MAX_FUNCTION_LINES}); split it, "
                "or mark the def with `# long-function: <reason>`"
            )
    for key, limit in sorted(allowed.items()):
        length = seen.get(key)
        if length is None:
            errors.append(f"{ALLOWLIST.name}: {key} no longer exists; remove its entry")
        elif length > limit:
            errors.append(
                f"{key}: {length} lines, grew from {limit}; "
                f"split it to at most {MAX_FUNCTION_LINES} lines rather than adding to it"
            )
        elif length <= MAX_FUNCTION_LINES:
            errors.append(f"{ALLOWLIST.name}: {key} is {length} lines now; remove its entry")
        elif length < limit:
            errors.append(f"{ALLOWLIST.name}: {key} shrank to {length}; lower its entry")
    return errors


def main() -> int:
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
