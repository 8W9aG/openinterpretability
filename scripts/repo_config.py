"""The project's packages and generated files, from its config, for the scripts (#6631).

The packages are ``[tool.mutmut] source_paths``, the list that mutmut itself copies. The
large folders that no unit test reads are ``[tool.labinacan] mutation_skip``. The files
that a tool writes, which the checks of hand-written files skip, are ``[tool.labinacan]
generated``. So no script names a package, a folder or a file of a project.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def _tables(root: Path) -> tuple[dict, dict]:
    """The ``[tool.mutmut]`` and ``[tool.labinacan]`` tables of ``root``'s pyproject.toml.

    Args:
        root: The repository's root.

    Returns:
        The two tables, each empty when the file lacks it.
    """
    tool = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8")).get("tool", {})
    return tool.get("mutmut", {}), tool.get("labinacan", {})


def packages(root: Path = Path(".")) -> tuple[str, ...]:
    """The packages that mutmut copies and mutates: ``[tool.mutmut] source_paths`` (#6631).

    Args:
        root: The repository's root.

    Returns:
        The packages' folders, relative to ``root``.

    Examples:
        >>> "labinacan" in packages(Path(__file__).resolve().parents[1])
        True
    """
    return tuple(_tables(root)[0].get("source_paths", ()))


def not_copied(root: Path = Path(".")) -> set[str]:
    """What mutmut's copy of the repository leaves out, beside the packages and the tests.

    mutmut copies the packages and the tests itself. ``[tool.labinacan] mutation_skip`` names
    the large folders that no unit test reads, such as a project's records (#6631).

    Args:
        root: The repository's root.

    Returns:
        The paths, relative to ``root``.

    Examples:
        >>> "tests" in not_copied(Path(__file__).resolve().parents[1])
        True
    """
    return {*packages(root), "tests", *_tables(root)[1].get("mutation_skip", ())}


def generated(root: Path = Path(".")) -> tuple[str, ...]:
    """The project's generated files, as ``fnmatch`` patterns: ``[tool.labinacan] generated``.

    The checks of hand-written files skip them: the line limit, Markdown style and step
    comments.

    Args:
        root: The repository's root.

    Returns:
        The patterns, relative to ``root``.

    Examples:
        >>> isinstance(generated(Path(__file__).resolve().parents[1]), tuple)
        True
    """
    return tuple(_tables(root)[1].get("generated", ()))


def automatic_workflows(root: Path = Path(".")) -> tuple[str, ...]:
    """The workflows whose failure issues an agent may work.

    They are ``[tool.labinacan] automatic_issue_workflows``. ``scripts/issue_state.py``
    trusts an issue from the workflows' bot only under the fixed titles of these workflows. A
    workflow left out, such as a link check, opens issues that an agent never takes (#6631).

    Args:
        root: The repository's root.

    Returns:
        The workflows' file names, in ``.github/workflows``.

    Examples:
        >>> "deploy.yml" in automatic_workflows(Path(__file__).resolve().parents[1])
        True
    """
    return tuple(_tables(root)[1].get("automatic_issue_workflows", ()))
