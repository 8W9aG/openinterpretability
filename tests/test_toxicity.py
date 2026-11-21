"""How the toxicity labels read a document, with the models replaced (offline)."""

import json
from types import SimpleNamespace

import pytest

from openinterpretability import toxicity
from openinterpretability.toxicity import _HATE, _TOXICITY, _label, _matches, sentences
from openinterpretability.toxicity import hate_matches as real_hate_matches
from openinterpretability.toxicity import toxicity_matches as real_toxicity_matches


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
    (path,) = (tmp_path / "openinterpretability" / "toxicity").rglob("*.json")
    assert json.loads(path.read_text()) == [[0, 12, "insult"], [13, 23, "toxicity"]]


def test_a_document_with_no_sentences_needs_no_model(scored):
    assert _matches(_TOXICITY, "  \n") == ()
    assert _matches(_TOXICITY, "Fluker.") == ()
    assert scored == []


def test_the_detector_pins_both_models():
    assert _TOXICITY.revision[:12] in toxicity.DETECTOR
    assert _HATE.revision[:12] in toxicity.DETECTOR


def test_each_label_set_reads_with_its_own_model(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    read = []

    def score(classifier, texts):
        read.append(classifier.name)
        return [{label: 0.0 for label in (*classifier.labels, "toxicity")} | {"insult": 0.9,
                 "racism": 0.8} for _ in texts]  # fmt: skip

    monkeypatch.setattr(toxicity, "_score", score)
    assert real_toxicity_matches("You are bad.") == ((0, 12, "insult"),)
    assert real_hate_matches("You are bad.") == ((0, 12, "racism"),)
    assert read == ["toxicity", "hate"]


class _Tokenizer:
    """Turns each text into one token per word, padded, as a real tokenizer's batch."""

    def __init__(self):
        self.batches = []

    def __call__(self, texts, truncation, max_length, padding, return_tensors):
        import torch

        self.batches.append((list(texts), truncation, max_length))
        width = max(len(t.split()) for t in texts)
        ids = [[1] * len(t.split()) + [0] * (width - len(t.split())) for t in texts]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(ids)}


class _Model:
    """Two outputs: the first scores the text's length in words, the second is always 0."""

    config = SimpleNamespace(id2label={0: "insult", 1: "threat"}, num_labels=2)

    def __call__(self, input_ids, attention_mask):
        import torch

        words = attention_mask.sum(-1).float()
        return SimpleNamespace(logits=torch.stack([words, torch.zeros_like(words)], -1))


def test_scores_come_from_the_model_in_batches(monkeypatch):
    tokenizer = _Tokenizer()
    monkeypatch.setattr(toxicity, "_model", lambda classifier: (tokenizer, _Model()))
    monkeypatch.setattr(toxicity, "_BATCH", 2)
    found = toxicity._score(_TOXICITY, ["a b", "a", "a b c"])
    assert [round(s["insult"], 4) for s in found] == [0.8808, 0.7311, 0.9526]  # sigmoid
    assert [s["threat"] for s in found] == [0.5, 0.5, 0.5]
    assert [b[0] for b in tokenizer.batches] == [["a b", "a"], ["a b c"]]
    assert tokenizer.batches[0][1:] == (True, toxicity._MAX_TOKENS)
    softmax = toxicity._score(_HATE, ["a"])
    assert softmax[0]["insult"] + softmax[0]["threat"] == pytest.approx(1.0)


def test_the_models_load_at_their_pinned_commits(monkeypatch):
    import transformers

    loads = []

    class Loaded:
        def eval(self):
            loads.append("eval")
            return self

    def tokenizer(repo, revision):
        loads.append(("tokenizer", repo, revision))
        return "tokenizer"

    def model(repo, revision):
        loads.append(("model", repo, revision))
        return Loaded()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification, "from_pretrained", model)
    toxicity._model.cache_clear()
    try:
        tok, loaded = toxicity._model(_HATE)
        assert toxicity._model(_HATE) == (tok, loaded)  # loaded once
    finally:
        toxicity._model.cache_clear()
    assert tok == "tokenizer" and isinstance(loaded, Loaded)
    assert loads == [
        ("tokenizer", _HATE.repo, _HATE.revision),
        ("model", _HATE.repo, _HATE.revision),
        "eval",
    ]
