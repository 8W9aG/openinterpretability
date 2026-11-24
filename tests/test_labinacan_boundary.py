"""``labinacan/`` stands alone: it imports nothing from the project around it (#6628).

It is the part of this repository that the template repository ``8W9aG/labinacan`` holds,
and that this repository merges from (#6207). An import of ``openinterpretability`` would
not exist there.
"""

import ast
from pathlib import Path

LABINACAN = Path(__file__).parents[1] / "labinacan"
PROJECT = "openinterpretability"


def project_imports(source: str) -> list[str]:
    """The modules of the project that ``source`` imports, at any depth.

    Args:
        source: A module's text.

    Returns:
        The imported module names that start with the project's package.
    """
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.append(node.module)
    return [m for m in found if m == PROJECT or m.startswith(f"{PROJECT}.")]


def test_project_imports_are_found_at_any_depth():
    source = (
        "import os\nfrom openinterpretability.core import x\n"
        "def f():\n    import openinterpretability\n"
    )
    assert project_imports(source) == ["openinterpretability.core", "openinterpretability"]
    assert project_imports("from . import sibling\nimport openinterpretability_other\n") == []


def test_labinacan_imports_nothing_from_the_project():
    offenders = {
        str(path.relative_to(LABINACAN.parent)): found
        for path in sorted(LABINACAN.rglob("*.py"))
        if (found := project_imports(path.read_text(encoding="utf-8")))
    }
    assert offenders == {}
    assert len(list(LABINACAN.rglob("*.py"))) > 2  # the check reads the moved modules
