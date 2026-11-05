"""How the toxicity labels read a document, with the models replaced (offline)."""

import json

import pytest

from openinterp import toxicity
from openinterp.toxicity import _HATE, _TOXICITY, _label, _matches, sentences


def _scores(**found):
    return {label: found.get(label, 0.0) for label in (*_TOXICITY.labels, "toxicity")}


def test_a_sentence_ends_at_a_stop_followed_by_a_space_or_at_a_line_end():
    text = "One here. Two is v1.2 now!\n\n  Three"
    assert [text[a:b] for a, b in sentences(text)] == ["One here.", "Two is v1.2 now!", "Three"]
    assert sentences("") == [] and sentences(" \n ") == []


def test_a_long_sentence_is_read_in_pieces_that_cover_it():
    spans = sentences("x" * 2500)
    assert spans == [(0, 1000), (1000, 2000), (2000, 2500)]


def test_a_sentence_takes_its_strongest_label():
    assert _label(_TOXICITY, _scores(toxicity=0.9, insult=0.6, threat=0.8)) == "threat"


def test_the_general_label_is_only_for_a_sentence_with_no_specific_one():
    assert _label(_TOXICITY, _scores(toxicity=0.9, insult=0.6)) == "insult"
    assert _label(_TOXICITY, _scores(toxicity=0.9)) == "toxicity"
    assert _label(_TOXICITY, _scores(toxicity=0.4, insult=0.4)) is None


def test_hate_speech_has_no_general_label_and_ignores_other():
    scores = dict.fromkeys(_HATE.labels, 0.0)
    assert _label(_HATE, {**scores, "other": 0.9, "not_hate": 0.1}) is None
    assert _label(_HATE, {**scores, "sexism": 0.7, "not_hate": 0.3}) == "sexism"


def test_identity_outputs_label_nothing():
    assert not {"male", "female", "black", "white", "muslim"} & set(_TOXICITY.labels)


@pytest.fixture
def scored(monkeypatch, tmp_path):
    """Score a sentence as an insult when it says "bad". Return the texts that were read."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    read = []

    def score(classifier, texts):
        read.extend(texts)
        return [_scores(toxicity=0.9, insult=0.9 if "bad" in t else 0.0) for t in texts]

    monkeypatch.setattr(toxicity, "_score", score)
    return read


def test_matches_label_whole_sentences_and_stay_on_disk(scored, tmp_path):
    text = "You are bad. Fine then."
    expected = ((0, 12, "insult"), (13, 23, "toxicity"))
    assert _matches(_TOXICITY, text) == expected
    assert scored == ["You are bad.", "Fine then."]
    assert _matches(_TOXICITY, text) == expected
    assert len(scored) == 2, "the second call reads the disk"
    (path,) = (tmp_path / "openinterp" / "toxicity").rglob("*.json")
    assert json.loads(path.read_text()) == [[0, 12, "insult"], [13, 23, "toxicity"]]


def test_a_document_with_no_sentences_needs_no_model(scored):
    assert _matches(_TOXICITY, "  \n") == ()
    assert _matches(_TOXICITY, "Fluker.") == ()
    assert scored == []


def test_the_detector_pins_both_models():
    assert _TOXICITY.revision[:12] in toxicity.DETECTOR
    assert _HATE.revision[:12] in toxicity.DETECTOR
