"""Where the registry keeps its files: the wiki directory and its subdirectories.

Every command and function that takes a ``wiki_dir`` defaults to :data:`DEFAULT_WIKI_DIR`
and finds records through these helpers, so the layout is written down once.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_WIKI_DIR = Path("wiki")
DEFAULT_SUBMISSIONS_DIR = Path("submissions")
DEFAULT_EXCHANGE_DIR = Path("exchange")
"""Where the CLI writes agent prompts and answers. ``$OPENINTERP_EXCHANGE_DIR`` changes it."""

PROMPT_SUFFIX = ".prompt.md"
"""The end of an agent's prompt file name, e.g. ``review.prompt.md``."""
PROBES_SUFFIX = ".probes.json"
"""The end of a reviewer's answer file name: ``review.prompt.md`` is answered in
``review.probes.json``."""


def experiments_dir(wiki_dir: Path | str) -> Path:
    """Experiment records and pages, one directory per unit: ``experiments/<slug>/``."""
    return Path(wiki_dir) / "experiments"


def draws_dir(wiki_dir: Path | str) -> Path:
    """Random draws the scan stopped on, one directory per unit: ``draws/<slug>/``."""
    return Path(wiki_dir) / "draws"


def ledger_dir(wiki_dir: Path | str) -> Path:
    """Each draw that ``open-interpretability next --steps`` made, by day: ``ledger/<day>/``."""
    return Path(wiki_dir) / "ledger"


def stops_dir(wiki_dir: Path | str) -> Path:
    """Chosen units the scan stopped on, one directory per unit: ``stops/<slug>/``."""
    return Path(wiki_dir) / "stops"
