"""Where the registry keeps its files: the wiki directory and its subdirectories.

Every command and function that takes a ``wiki_dir`` defaults to :data:`DEFAULT_WIKI_DIR`
and finds records through these helpers, so the layout is written down once.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_WIKI_DIR = Path("wiki")


def experiments_dir(wiki_dir: Path | str) -> Path:
    """Experiment records and pages, one directory per unit: ``experiments/<slug>/``."""
    return Path(wiki_dir) / "experiments"


def draws_dir(wiki_dir: Path | str) -> Path:
    """Random draws the scan stopped on, one directory per unit: ``draws/<slug>/``."""
    return Path(wiki_dir) / "draws"
