"""``labinacan/`` stands alone: it imports nothing from the project around it (#6628).

It is the part of a project's repository that the labinacan template holds, and that the
project merges from (#6207). An import of the project's own package would not exist there.
"""

import ast
from pathlib import Path

import pytest

from labinacan.config import config

LABINACAN = Path(__file__).parents[1] / "labinacan"


def project_imports(source: str, project: str) -> list[str]:
    """The modules of the project that ``source`` imports, at any depth.

    Args:
        source: A module's text.
        project: The project's package (``[tool.labinacan] package``).

    Returns:
        The imported module names that start with the project's package.
    """
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.append(node.module)
    return [m for m in found if m == project or m.startswith(f"{project}.")]


def test_project_imports_are_found_at_any_depth():
    source = "import os\nfrom aproject.core import x\ndef f():\n    import aproject\n"
    assert project_imports(source, "aproject") == ["aproject.core", "aproject"]
    assert project_imports("from . import sibling\nimport aproject_other\n", "aproject") == []


def test_labinacan_imports_nothing_from_the_project():
    project = config().package
    if project == "labinacan":  # the template's own example project: nothing around it
        pytest.skip("the project is labinacan itself")
    offenders = {
        str(path.relative_to(LABINACAN.parent)): found
        for path in sorted(LABINACAN.rglob("*.py"))
        if (found := project_imports(path.read_text(encoding="utf-8"), project))
    }
    assert offenders == {}
    assert len(list(LABINACAN.rglob("*.py"))) > 2  # the check reads the moved modules
