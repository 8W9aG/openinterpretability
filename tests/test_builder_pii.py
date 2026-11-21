"""Overlapping matches in ``builder.pii``, and attribution that names no top example."""

import pytest

from openinterpretability.builder import pii
from openinterpretability.builder.pii import REMOVED_TEXT
from openinterpretability.core import takedowns
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
    (found,) = pii._scrub_attribution(sub).input_attribution.examples
    assert found.peak_token == "@" and [t.token for t in found.top] == [" jane"]
