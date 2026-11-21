import pytest

from openinterpretability.harness.hooks import NeuronHookRunner
from openinterpretability.harness.scan import Piece, TopExamples, batches, cut_pieces, scan_neuron
from tests.conftest import FakeTokenizer


def test_an_empty_document_gives_no_windows():
    pieces = list(cut_pieces(FakeTokenizer(), ["", "abc"], window_tokens=2, stride=1))
    assert [(p.doc, p.start, p.end) for p in pieces] == [(1, 0, 2), (1, 1, 3)]
    assert [p.fresh for p in pieces] == [None, 1]


def test_no_pieces_give_no_batches(toy_runner):
    assert list(batches(toy_runner, [])) == []


def test_a_piece_with_no_strengths_is_not_a_candidate():
    top = TopExamples()
    top.add(Piece(0, 0, 1, ""), [], [])
    top.add(Piece(1, 0, 1, "a"), [3], [0.5])
    assert [piece.doc for piece, _, _ in top.candidates] == [1]


def test_scan_neuron_needs_a_tokenizer(toy_model):
    with pytest.raises(ValueError, match="requires a tokenizer"):
        scan_neuron(NeuronHookRunner(toy_model), ["abc"], 1, 0, "positive")
