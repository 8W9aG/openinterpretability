"""Superpositions on the site (openinterpretability/builder/superpositions.py)."""

from types import SimpleNamespace

from openinterpretability.builder.superpositions import members_of


def test_a_superposition_whose_newest_record_has_no_profile_has_no_members(record):
    sub = record.submission.model_copy(update={"superposition_profile": None})
    newest = SimpleNamespace(id="b", record=record.model_copy(update={"submission": sub}))
    older = SimpleNamespace(id="a", record=record)
    unit = SimpleNamespace(experiments=[older, newest], layer=0, profile="unset")
    assert members_of(SimpleNamespace(slug="m"), unit) == []
    assert unit.profile is None
