"""Overlapping matches in ``labinacan.site.pii``, and attribution that names no top example."""

import pytest
from labinacan.site import pii, takedowns
from labinacan.site.pii import REMOVED_TEXT

from openinterpretability.builder import registry_pii
from openinterpretability.core.attribution import AttributedToken, ContextExample, InputAttribution

# The phone and the payment card recognisers both match here, and their spans overlap.
CARD = "Phone: 4111 1111 1111 1111 now."


@pytest.fixture
def no_cache(monkeypatch):
    """Searches run each time, with no spans read from or kept on disk."""

    def fresh():
        for f in (pii._cache_file, pii._stored, pii.find_spans):
            f.cache_clear()

    monkeypatch.setenv("OPENINTERPRETABILITY_PII_CACHE", "off")
    fresh()
    yield
    fresh()


def test_overlapping_personal_matches_are_joined_into_one_span(no_cache):
    assert pii.find_spans(CARD) == ((7, 26),)
    assert pii.redact_text(CARD) == f"Phone: {REMOVED_TEXT} now."


def test_a_takedown_that_overlaps_personal_data_is_joined_with_it(no_cache, monkeypatch):
    passage = "the passage ends with Phone 4111 1111 1111"
    monkeypatch.setattr(takedowns, "_active", frozenset(takedowns.shingle_hashes(passage)))
    text = "We say the passage ends with Phone: 4111 1111 1111 1111 now."
    assert pii.redact_text(text) == f"We say {REMOVED_TEXT} now."


def test_attribution_of_an_example_past_the_top_examples_is_kept(record):
    context = ContextExample.model_construct(
        example=len(record.submission.top_examples),
        peak_token="@",
        top=[AttributedToken.model_construct(offset=-1, token=" jane", attribution=0.5)],
    )
    attribution = InputAttribution.model_construct(examples=[context])
    sub = record.submission.model_copy(update={"input_attribution": attribution})
    (found,) = registry_pii._scrub_attribution(sub).input_attribution.examples
    assert found.peak_token == "@" and [t.token for t in found.top] == [" jane"]


def test_with_no_settings_the_core_keeps_nothing_on_disk():
    from labinacan.site.pii import _Settings

    assert _Settings().cache_dir() is None and _Settings().name == "labinacan.sensitive"


def test_the_registry_gives_the_core_its_cache_folder_and_the_frozen_name(monkeypatch, tmp_path):
    """Records' label_detector and the cache key name the search as before (#6628)."""
    from labinacan.site import pii as core

    from openinterpretability.builder import registry_pii
    from openinterpretability.core.confounds import PII_DETECTOR

    assert core._settings.name == PII_DETECTOR
    monkeypatch.setenv(registry_pii.CACHE_ENV, str(tmp_path))
    assert core._settings.cache_dir() == tmp_path
    monkeypatch.setenv(registry_pii.CACHE_ENV, "off")
    assert core._settings.cache_dir() is None
