"""The project's own values, from its ``pyproject.toml`` (#6629).

A generic file holds no value that differs between projects. It reads the value here.
``[project]`` gives the distribution's name, which is also its command, its one-line
description, its URLs and its maintainer. ``[tool.labinacan]`` gives the rest. That is the
project's name as people read it, its Python package and the prefix of its environment
variables. It is also its Linear team, the maintainer's GitHub login and its Hugging Face
dataset.

The project tells this module where its ``pyproject.toml`` is, with :func:`use`. The first
path that holds a ``[tool.labinacan]`` table counts. Until then, the working directory's
``pyproject.toml`` counts.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from email.utils import parseaddr
from functools import cache
from pathlib import Path

_PATHS: list[Path] = [Path("pyproject.toml")]


@dataclass(frozen=True)
class Brand:
    """The project's brand: what its mark and wordmark are drawn from."""

    folder: str  # the folder of the brand's files, inside the project's package
    tile: str  # the colour of the mark's tile
    ink: dict[str, str]  # the wordmark's colour, by "light" and "dark"
    colours: dict[str, str]  # the mark's colours, by name
    # The mark's cells, each (column, row, share filled, colour name). An empty cell has
    # share 0 and colour "".
    cells: tuple[tuple[int, int, float, str], ...]


def brand_of(table: dict | None) -> Brand | None:
    """The brand in a ``[tool.labinacan.brand]`` table, or None when there is no table.

    Args:
        table: The table. Each cell gives the share filled in thirds.

    Returns:
        The brand.

    Examples:
        >>> brand_of(None) is None
        True
        >>> b = brand_of({"folder": "b", "tile": "#000", "ink": {}, "colours": {},
        ...               "cells": [[1, 2, 1, "red"], [0, 0, 0, ""]]})
        >>> b.cells == ((1, 2, 1 / 3, "red"), (0, 0, 0.0, ""))
        True
    """
    if table is None:
        return None
    return Brand(
        folder=table["folder"], tile=table["tile"], ink=dict(table["ink"]),
        colours=dict(table["colours"]),
        cells=tuple((col, row, thirds / 3, colour) for col, row, thirds, colour in table["cells"]),
    )  # fmt: skip


@dataclass(frozen=True)
class ProjectConfig:
    """The values of one project that the generic code reads."""

    name: str  # as people read it, such as "A Project"
    package: str  # the Python package, which the changelog page names
    command: str  # the distribution's name and its command, such as "a-project"
    env_prefix: str  # environment variables are "<env_prefix>_<NAME>"
    description: str  # the one-line description, for social cards
    site_url: str  # the site's root, with a final "/"
    repo_url: str
    x_url: str
    contact_email: str
    linear_team: str  # the key of the Linear team that mirrors the issues, such as "ABC"
    maintainer: str  # the maintainer's GitHub login
    hf_dataset: str  # the Hugging Face dataset of the public inputs, as "owner/name"
    brand: Brand | None = None  # [tool.labinacan.brand], or None for a project with no brand

    @property
    def repo(self) -> str:
        """The GitHub repository as ``owner/name``, from ``repo_url``.

        Returns:
            The slug.

        Examples:
            >>> config().repo == config().repo_url.removeprefix("https://github.com/")
            True
        """
        return self.repo_url.removeprefix("https://github.com/").strip("/")


def use(*paths: Path) -> None:
    """Read the config from the first of ``paths`` that holds a ``[tool.labinacan]`` table.

    Args:
        *paths: The candidate ``pyproject.toml`` files, in order.

    Examples:
        >>> saved = list(_PATHS)
        >>> use(Path("no-such-file.toml"))
        >>> config()
        Traceback (most recent call last):
        ...
        LookupError: no [tool.labinacan] table in no-such-file.toml
        >>> use(*saved)
    """
    _PATHS[:] = [Path(p) for p in paths]
    config.cache_clear()


def _tables(path: Path) -> tuple[dict, dict] | None:
    """``([project], [tool.labinacan])`` of ``path``, or None when either is missing.

    Args:
        path: A ``pyproject.toml``.

    Returns:
        The two tables, or None.
    """
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    project, own = data.get("project"), data.get("tool", {}).get("labinacan")
    return (project, own) if project and own else None


@cache
def config() -> ProjectConfig:
    """The project's config, from the first path given to :func:`use`.

    Returns:
        The config.

    Raises:
        LookupError: No path holds both tables.

    Examples:
        >>> config() is config()  # read once, until use() is called again
        True
    """
    for path in _PATHS:
        # The first path that holds both tables counts, as use() says.
        if (found := _tables(path)) is not None:
            project, own = found
            urls = project["urls"]
            return ProjectConfig(
                name=own["name"],
                package=own["package"],
                command=project["name"],
                env_prefix=own["env_prefix"],
                description=project["description"].split(": ", 1)[-1],
                site_url=urls["Homepage"],
                repo_url=urls["Repository"],
                x_url=urls["X"],
                contact_email=contact_email(project["maintainers"][0]),
                linear_team=own["linear_team"],
                maintainer=own["maintainer"],
                hf_dataset=own["hf_dataset"],
                brand=brand_of(own.get("brand")),
            )
    raise LookupError(f"no [tool.labinacan] table in {', '.join(map(str, _PATHS))}")


def env(name: str) -> str:
    """The project's name for the environment variable ``name``.

    Args:
        name: The variable's own part, such as ``CHROME``.

    Returns:
        ``<env_prefix>_<name>``.

    Examples:
        >>> env("CHROME") == config().env_prefix + "_CHROME"
        True
    """
    return f"{config().env_prefix}_{name}"


def contact_email(maintainer: dict) -> str:
    """The address of one ``[project] maintainers`` entry.

    Args:
        maintainer: The entry, with an ``email`` key, which may hold a ``Name <address>``.

    Returns:
        The address.

    >>> contact_email({"name": "A Project", "email": "contact@example.com"})
    'contact@example.com'
    """
    return parseaddr(maintainer["email"])[1]


def x_handle(url: str) -> str:
    """The X handle in an account's URL, so that the handle is not written a second time.

    Args:
        url: The account's URL.

    Returns:
        ``@`` and the URL's last path part.

    >>> x_handle("https://x.com/example")
    '@example'
    >>> x_handle("https://x.com/example/")
    '@example'
    """
    return "@" + url.rstrip("/").rsplit("/", 1)[-1]


def project_links() -> dict[str, str]:
    """The project's values that every page's template gets.

    Returns:
        The site and repo URLs, the contact email, the one-line description, the X account
        and its handle, and the project's name and package.

    Examples:
        >>> sorted(project_links())  # doctest: +NORMALIZE_WHITESPACE
        ['contact_email', 'description', 'package', 'project_name', 'repo_url',
         'site_url', 'x_handle', 'x_url']
    """
    c = config()
    return {
        "site_url": c.site_url,
        "repo_url": c.repo_url,
        "contact_email": c.contact_email,
        "description": c.description,  # for social cards (#3289)
        "x_url": c.x_url,
        "x_handle": x_handle(c.x_url),  # twitter:site (#5389)
        "project_name": c.name,
        "package": c.package,
    }


def site_url() -> str:
    """The public URL of the site's root, with a final ``/``.

    Returns:
        The URL.

    Examples:
        >>> site_url().endswith("/")
        True
    """
    return config().site_url
