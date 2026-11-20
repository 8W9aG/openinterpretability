"""Fail on a public function whose docstring misses a Returns, Yields or Raises section (#5361).

``check_docstring_args.py`` checks the ``Args:`` section of every function. This check adds
the other Google sections, for public functions and methods: names with no leading
underscore, in modules outside a private (``_``) path.

- **Returns.** A function that isn't a generator, and whose own body returns a value, needs
  a ``Returns:`` section. ``return`` and ``return None`` don't count. A ``@property`` getter
  is skipped, because its docstring describes the attribute.
- **Yields.** A generator needs a ``Yields:`` section.
- **Raises.** Each exception that the function's own body raises by name (``raise X`` or
  ``raise X(...)``) is named in ``Raises:``. ``Raises:`` names no exception that the body
  doesn't raise. A bare ``raise`` (a re-raise) and a raise inside a nested function don't
  count.

Ruff's DOC rules are left off. They need ``preview = true`` for the whole linter, and they
treat properties and re-raises differently, so the two checks would block each other.

The rule is ratcheted in, as ``check_docstring_args.py`` is. :data:`BASELINE` lists the
functions that broke it when it was added, one ``path::Qualified.name`` a line. A function
that breaks it and isn't listed fails. So does a listed function that now passes or no
longer exists, so the list only shrinks.

Run with ``make lint``; CI runs it in the lint step. ``--write-baseline`` rewrites the list.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

from check_docstring_args import entries

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "scripts" / "docstring_sections_baseline.txt"
PACKAGE = "openinterpretability/"
RETURNS = re.compile(r"^(\s*)Returns:\s*$", re.M)
YIELDS = re.compile(r"^(\s*)Yields:\s*$", re.M)
RAISES = re.compile(r"^(\s*)Raises:\s*$")
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def own_nodes(func: ast.AST):
    """The nodes in ``func``'s own body, not inside a nested function, class or lambda.

    Args:
        func: A function's definition.

    Yields:
        Each node of the body, outermost first.
    """
    stack = list(ast.iter_child_nodes(func))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, (*FUNCTIONS, ast.ClassDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(node))


def _raised(node: ast.Raise) -> str | None:
    """The name of the exception that ``node`` raises, or None for a bare re-raise.

    Args:
        node: A ``raise`` statement.

    Returns:
        The exception's name as written, such as ``ValueError`` or ``click.ClickException``,
        or None when the statement re-raises or raises an expression with no plain name.
    """
    exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
    return ast.unparse(exc) if isinstance(exc, (ast.Name, ast.Attribute)) else None


def problems(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """What ``func``'s docstring leaves out: ``Returns``, ``Yields`` or a ``Raises`` mismatch.

    Args:
        func: A public function's definition.

    Returns:
        One short reason for each missing or wrong section, or an empty list.
    """
    doc = ast.get_docstring(func) or ""
    nodes = list(own_nodes(func))
    generator = any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in nodes)
    returns = any(
        isinstance(n, ast.Return) and n.value is not None
        and not (isinstance(n.value, ast.Constant) and n.value.value is None)
        for n in nodes
    )  # fmt: skip
    prop = any(ast.unparse(d).endswith("property") for d in func.decorator_list)
    out = []
    if returns and not generator and not prop and not RETURNS.search(doc):
        out.append("Returns")
    if generator and not YIELDS.search(doc):
        out.append("Yields")
    raised = {name for n in nodes if isinstance(n, ast.Raise) and n.exc for name in [_raised(n)]}
    raised.discard(None)
    if raised != entries(doc, RAISES):
        out.append("Raises")
    return out


def offenders(path: str, source: str) -> list[str]:
    """``path::Qualified.name: sections`` of each public function in ``source`` that fails.

    Args:
        path: The file's path, which starts each name.
        source: The file's text.

    Returns:
        One line per failing function, in source order.
    """
    found: list[str] = []

    def visit(node: ast.AST, prefix: str) -> None:
        """Check each public function under ``node``, and recurse into public classes.

        Args:
            node: The node whose children to check.
            prefix: The qualified name of ``node``, with a trailing dot, or "".
        """
        for child in ast.iter_child_nodes(node):
            if isinstance(child, FUNCTIONS) and not child.name.startswith("_"):
                missing = problems(child)
                if missing:
                    found.append(f"{path}::{prefix}{child.name}: {', '.join(missing)}")
            elif isinstance(child, ast.ClassDef) and not child.name.startswith("_"):
                visit(child, f"{prefix}{child.name}.")

    visit(ast.parse(source), "")
    return found


def every_offender() -> list[str]:
    """The offenders in every tracked public module of the package.

    Returns:
        One line per failing function, by file.
    """
    files = subprocess.run(["git", "ls-files", "*.py"], capture_output=True, text=True,
                           check=True, cwd=ROOT).stdout.split()  # fmt: skip
    out = []
    for name in sorted(f for f in files if f.startswith(PACKAGE)):
        parts = Path(name).parts
        if any(p.startswith("_") and p != "__init__.py" for p in parts):
            continue
        out += offenders(name, (ROOT / name).read_text(encoding="utf-8"))
    return out


def main() -> int:
    """Check against the baseline; exit 1 on a new offender or a stale baseline line.

    Returns:
        The exit code: 0 when every offender is listed and every line still fails.
    """
    found = {line.split(": ")[0]: line for line in every_offender()}
    if sys.argv[1:] == ["--write-baseline"]:
        BASELINE.write_text("".join(f"{name}\n" for name in found), encoding="utf-8")
        print(f"{len(found)} functions → {BASELINE.name}")
        return 0
    listed = set(BASELINE.read_text(encoding="utf-8").split()) if BASELINE.exists() else set()
    new = [line for name, line in found.items() if name not in listed]
    stale = sorted(listed - set(found))
    for line in new:
        print(f"{line}: document it in the docstring (the commenting skill).")
    for name in stale:
        print(f"{name}: now passes or is gone. Delete its line from {BASELINE.name}.")
    return 1 if new or stale else 0


if __name__ == "__main__":
    sys.exit(main())
