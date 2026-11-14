"""Read a built unit page together with its experiments' pages."""

from pathlib import Path


def experiment_pages(unit: Path) -> list[Path]:
    """The pages of the experiments on the unit whose page is ``unit``.

    They are in directories next to the unit's page. A neuron's are under
    ``N<idx>-<pole>/``. A head's are under ``H<idx>/``, and a feature's under ``F<idx>/``.
    """
    stem = unit.stem
    dirs = [d for d in unit.parent.iterdir() if d.is_dir() and d.name.split("-")[0] == stem]
    return sorted(p for d in dirs for p in d.glob("*.html"))


def unit_pages(unit: Path) -> str:
    """The text of the unit's page, then of each of its experiments' pages."""
    return "\n".join(p.read_text() for p in [unit, *experiment_pages(unit)])
