"""Fail on a function whose docstring doesn't describe every argument (issue #3938).

Every function and method, private and nested ones too, describes each of its arguments in
the Google ``Args:`` section of its docstring. A function with no docstring describes none
of them. ``__init__`` may describe them in its class's docstring, as Google style does.
``self`` and ``cls`` are skipped. ``*args`` and ``**kwargs`` count, under their
names with or without the stars. Click commands get no exception: a Click option's
``help=`` is for the command line, and the docstring is for the reader of the code. The
section must also name no argument that the signature lacks.

The rule is ratcheted in, as ``check_body_comments.py`` is. The functions that broke it
when it was added are listed in :data:`BASELINE`, one ``path::Qualified.name`` a line. A
function that breaks the rule and isn't listed fails. So does a listed function that now
passes or no longer exists, so the list only shrinks. When you describe a listed function's
arguments, delete its line.

Run with ``make lint``; CI runs it in the lint step. ``--write-baseline`` rewrites the list.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "scripts" / "docstring_args_baseline.txt"
FOLDERS = ("openinterpretability/", "scripts/")
SKIPPED = frozenset({"self", "cls"})
SECTION = re.compile(r"^(\s*)(Args|Arguments|Parameters):\s*$")
ENTRY = re.compile(r"^\s*\**([A-Za-z_][\w.]*)\s*(?:\([^)]*\))?\s*:")


def arguments(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """The names of the function's arguments, without ``self``, ``cls`` or stars.

    Args:
        node: The function's definition.
    """
    a = node.args
    names = [x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)]
    names += [x.arg for x in (a.vararg, a.kwarg) if x is not None]
    return [n for n in names if n not in SKIPPED]


def described(docstring: str | None) -> set[str]:
    """The argument names that the docstring's ``Args:`` section describes.

    Args:
        docstring: The function's docstring, or None.
    """
    return entries(docstring, SECTION)


def entries(docstring: str | None, section: re.Pattern) -> set[str]:
    """The names that start the entries of the docstring's ``section``.

    The first line after the heading sets the entries' indent. A line at that indent starts
    an entry: a name (stars and dots allowed), an optional type in brackets, then a colon. A
    deeper line continues an entry, and a shallower one ends the section.

    Args:
        docstring: The function's docstring, or None.
        section: A pattern that matches the section's heading line, with its indent as
            group 1.
    """
    names: set[str] = set()
    heading = entry = None  # the indents of the section's heading and of its entries
    for line in (docstring or "").splitlines():
        found = section.match(line)
        if found:
            heading, entry = len(found.group(1)), None
            continue
        if heading is None or not line.strip():
            continue
        depth = len(line) - len(line.lstrip())
        if depth <= heading:  # the section ended
            heading = None
            continue
        entry = depth if entry is None else entry
        name = ENTRY.match(line)
        if depth == entry and name:
            names.add(name.group(1))
    return names


def offenders(path: str, source: str) -> list[str]:
    """``path::Qualified.name`` of each function in ``source`` whose arguments aren't described.

    Args:
        path: The file's path, which starts each name.
        source: The file's text.
    """
    found: list[str] = []

    def visit(node: ast.AST, prefix: str) -> None:
        """Check each function under ``node``, and recurse into functions and classes.

        Args:
            node: The node whose children to check.
            prefix: The qualified name of ``node``, with a trailing dot, or "".
        """
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{child.name}"
                said = described(ast.get_docstring(child))
                if child.name == "__init__" and isinstance(node, ast.ClassDef):
                    said |= described(ast.get_docstring(node))  # Google style: the class's
                if set(arguments(child)) != said:
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
        print(f"{o}: describe every argument in the docstring's Args: section.")
    for o in stale:
        print(f"{o}: now passes or is gone. Delete its line from {BASELINE.name}.")
    return 1 if new or stale else 0


if __name__ == "__main__":
    sys.exit(main())
