"""Every public function and method has a doctest, or is listed (#5356).

A public function is one whose name has no leading underscore, in a module outside a
private (``_``) path. Its docstring must hold a ``>>>`` example, which pytest runs
(``--doctest-modules``). Two lists in ``scripts/`` hold the rest:

- ``doctests_baseline.txt``: module files that still have public functions with no doctest.
  It only shrinks. A listed file that now passes fails, so delete its line.
- ``doctests_exempt.txt``: single functions that can't run offline, each with a reason.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "openinterpretability"
BASELINE = ROOT / "scripts" / "doctests_baseline.txt"
EXEMPT = ROOT / "scripts" / "doctests_exempt.txt"


def _lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return [line.split()[0] for line in text.splitlines() if line.strip() and line[0] != "#"]


def missing(path: Path, source: str) -> list[str]:
    """``path::Qualified.name`` of each public function in ``source`` with no doctest."""
    found = []

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(child) or ""
                if not child.name.startswith("_") and ">>>" not in doc:
                    found.append(f"{path}::{prefix}{child.name}")
            elif isinstance(child, ast.ClassDef) and not child.name.startswith("_"):
                visit(child, f"{prefix}{child.name}.")

    visit(ast.parse(source), "")
    return found


def every_missing() -> dict[str, list[str]]:
    """Each public module file's functions with no doctest, by file."""
    out = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if any(part.startswith("_") and part != "__init__.py" for part in rel.parts):
            continue
        found = missing(rel, path.read_text(encoding="utf-8"))
        if found:
            out[str(rel)] = found
    return out


def test_every_public_function_has_a_doctest_or_is_listed():
    exempt, listed = set(_lines(EXEMPT)), set(_lines(BASELINE))
    gaps = {f: [n for n in names if n not in exempt] for f, names in every_missing().items()}
    gaps = {f: names for f, names in gaps.items() if names}
    new = sorted(n for f, names in gaps.items() if f not in listed for n in names)
    stale = sorted(listed - set(gaps))
    assert not new, f"write a doctest (see the commenting skill): {new[:10]}"
    assert not stale, f"these files have every doctest now; delete their lines: {stale}"


def test_a_new_undocumented_function_is_found():
    shown = 'def shown():\n    """Shown.\n\n    >>> 1\n    1\n    """\n'
    plain = 'def plain():\n    """No."""\n'
    box = "class Box:\n    def open(self):\n        pass\n\n    def _hidden(self):\n        pass\n"
    source = "\n".join((shown, plain, box))
    assert missing(Path("m.py"), source) == ["m.py::plain", "m.py::Box.open"]


def test_every_exemption_names_a_real_function():
    names = {n for found in every_missing().values() for n in found}
    for line in _lines(EXEMPT):
        assert line in names, f"{line} has a doctest now or no longer exists"
