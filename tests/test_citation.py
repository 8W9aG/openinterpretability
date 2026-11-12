"""The citation file (`CITATION.cff`) that GitHub and Zenodo read."""

from pathlib import Path

import yaml

CITATION = Path(__file__).resolve().parent.parent / "CITATION.cff"


def test_citation_names_one_licence():
    # Zenodo fails a release's archive with "Citation metadata load failed" when
    # `license` is a list, although CFF allows one.
    citation = yaml.safe_load(CITATION.read_text())
    assert isinstance(citation["license"], str)
