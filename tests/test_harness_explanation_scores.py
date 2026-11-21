"""Windows, control scores and embedding similarity of explanation scores (offline)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from openinterpretability.core.explanation import ControlScore, ExplanationScores, ScoredWindow
from openinterpretability.harness.explanation_scores import (
    cut_windows,
    embedding_similarities,
    finalize_scores,
)


class _CharTokenizer:
    """One token per character, so a window can be all spaces."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def test_blank_windows_are_dropped():
    runner = SimpleNamespace(tokenizer=_CharTokenizer())
    assert cut_windows(runner, ["ab  cd"], 2) == ["ab", "cd"]


def _window(label, judged):
    return ScoredWindow(text="t", tokens=["t"], activations=[0.0], label=label, judged=judged)


def test_a_control_with_the_wrong_number_of_answers_is_not_scored():
    windows = [_window(True, True), _window(True, True), _window(False, False)]
    control = ControlScore(kind="empty", explanation="", detection_judged=[False, False])
    scores = ExplanationScores(
        slug="toy/L1/N2-positive", explanation="x", threshold=1.0, window_tokens=8,
        quantiles=1, quantile_edges=[], seed=0, corpus="c", candidate_windows=3,
        active_windows=2, detection_windows=windows, controls=[control],
    )  # fmt: skip
    scores = finalize_scores(scores)
    assert scores.detection.balanced_accuracy == 1.0
    assert scores.controls[0].detection is None


@pytest.fixture
def tiny_encoder(tmp_path, monkeypatch):
    """A random one-layer BERT and a word vocabulary, in place of a Hub download."""
    from transformers import BertConfig, BertModel, BertTokenizer

    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "fires", "on", "quotes", "cats", "dogs"]
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(words) + "\n")
    tokenizer = BertTokenizer(str(vocab))
    torch.manual_seed(0)
    config = BertConfig(vocab_size=len(words), hidden_size=8, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=16)  # fmt: skip
    model = BertModel(config)
    loaded = []

    def tokenizer_from(name):
        loaded.append(name)
        return tokenizer

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", tokenizer_from)
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda name: model)
    return loaded


def test_embedding_similarity_is_one_for_the_query_itself(tiny_encoder):
    texts = ["fires on quotes"] + ["cats dogs"] * 32  # more than one batch of 32
    sims = embedding_similarities("enc", "fires on quotes", texts)
    assert tiny_encoder == ["enc"]
    assert len(sims) == 33
    assert sims[0] == pytest.approx(1.0, abs=1e-5)
    assert all(-1.0 <= s <= 1.0 for s in sims) and sims[1] < sims[0]
    assert len(set(sims[1:])) == 1  # the same text has the same similarity in each batch
