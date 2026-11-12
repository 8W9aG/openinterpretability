"""The citation file (`CITATION.cff`) that GitHub and Zenodo read."""

from pathlib import Path

import yaml

CITATION = Path(__file__).resolve().parent.parent / "CITATION.cff"


def test_citation_names_one_licence():
    # Zenodo fails a release's archive with "Citation metadata load failed" when
    # `license` is a list, although CFF allows one.
    citation = yaml.safe_load(CITATION.read_text())
    assert isinstance(citation["license"], str)


def test_citation_names_the_zenodo_concept_doi():
    # The concept DOI names every archived release. The README badge and the
    # *Citing* section must name the same DOI.
    citation = yaml.safe_load(CITATION.read_text())
    doi = citation["doi"]
    assert doi.startswith("10.5281/zenodo.")
    readme = (CITATION.parent / "README.md").read_text()
    assert f"https://doi.org/{doi}" in readme
