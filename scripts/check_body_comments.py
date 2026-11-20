"""Fail on a long function whose body holds no comment (issue #3159).

A docstring says what a function does. Inside a long body, a reader also needs to know why
each step is there, and a ``#`` comment is where that goes. So a function or method whose
body, after its docstring, runs to :data:`MIN_LINES` lines or more must hold at least one
comment. Functions are capped at 40 lines (``check_line_limit.py``), so this is the long
half of them.

The rule is ratcheted in. The functions that broke it when it was added are listed in
:data:`BASELINE`, one ``path::Qualified.name`` a line. A function that breaks the rule and
isn't listed fails. So does a listed function that now passes or no longer exists, so the
list only shrinks. When you add a comment to a listed function, delete its line.

Run with ``make lint``; CI runs it in the lint step. ``--write-baseline`` rewrites the list.
"""

from __future__ import annotations

import ast
import io
import subprocess
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "scripts" / "body_comments_baseline.txt"
MIN_LINES = 20
FOLDERS = ("openinterpretability/", "scripts/")


def _comments(source: str) -> set[int]:
    """The line numbers that hold a ``#`` comment."""
    lines = set()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            lines.add(token.start[0])
    return lines


def _body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    """The function's statements after its docstring."""
    body = node.body
    first = body[0] if body else None
    value = first.value if isinstance(first, ast.Expr) else None
    docstring = isinstance(value, ast.Constant) and isinstance(value.value, str)
    return body[1:] if docstring else body


def offenders(path: str, source: str) -> list[str]:
    """``path::Qualified.name`` of each long function in ``source`` with no comment in its body."""
    comments = _comments(source)
    found: list[str] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{child.name}"
                body = _body(child)
                if body:
                    start, end = body[0].lineno, child.end_lineno or body[-1].lineno
                    long = end - start + 1 >= MIN_LINES
                    if long and not any(start <= line <= end for line in comments):
                        found.append(f"{path}::{name}")
                visit(child, f"{name}.")
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")

    visit(ast.parse(source), "")
    return found


def every_offender() -> list[str]:
    """The offenders in every tracked Python file of :data:`FOLDERS`."""
    files = subprocess.run(["git", "ls-files", "*.py"], capture_output=True, text=True,
                           check=True, cwd=ROOT).stdout.split()  # fmt: skip
    out = []
    for name in sorted(f for f in files if f.startswith(FOLDERS)):
        out += offenders(name, (ROOT / name).read_text(encoding="utf-8"))
    return out


def main() -> int:
    """Check against the baseline; exit 1 on a new offender or a stale baseline line."""
    found = every_offender()
    if sys.argv[1:] == ["--write-baseline"]:
        BASELINE.write_text("".join(f"{o}\n" for o in found), encoding="utf-8")
        print(f"{len(found)} functions → {BASELINE.name}")
        return 0
    listed = set(BASELINE.read_text(encoding="utf-8").split()) if BASELINE.exists() else set()
    new = [o for o in found if o not in listed]
    stale = sorted(listed - set(found))
    for o in new:
        print(f"{o}: a body of {MIN_LINES}+ lines with no comment. Say why its steps are there.")
    for o in stale:
        print(f"{o}: now passes or is gone. Delete its line from {BASELINE.name}.")
    return 1 if new or stale else 0


if __name__ == "__main__":
    sys.exit(main())
