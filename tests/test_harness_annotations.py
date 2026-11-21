"""Token labels and the label confound (harness/annotations.py)."""

import torch

from openinterpretability.harness.annotations import (
    NO_MATCH,
    LabelAccumulator,
    MatchTagger,
    Span,
    TokenLabels,
    activation_bins,
)
from openinterpretability.harness.scan import Piece
from tests.test_annotations import CharTokenizer, WordTagger


def test_a_match_that_ends_the_text_leaves_no_gap_after_it():
    tagger = MatchTagger("digits", lambda text: [(3, 5, "digit")])
    assert tagger.spans(0, "ab 12") == [Span(3, 5, "digit"), Span(0, 3, NO_MATCH)]


def test_a_piece_that_starts_past_the_documents_tokens_is_unlabelled():
    labels = TokenLabels(CharTokenizer(), ["ab cd"], [WordTagger()])
    piece = Piece(0, 9, 11, "ab")  # the document has only 5 tokens
    assert labels.of(piece, torch.tensor([0, 1]))["word"].tolist() == [-1, -1]


def test_no_labelled_tokens_give_an_empty_confound():
    confound = LabelAccumulator().finalize(["en", "de"], line=1.0)
    assert confound.r2 == 0.0 and confound.mutual_information_bits == 0.0
    assert (confound.labelled_tokens, confound.unlabelled_tokens, confound.levels) == (0, 0, [])


def test_with_no_positive_strength_every_value_is_in_bin_zero():
    assert activation_bins(torch.tensor([0.0, -1.0, 0.0])).tolist() == [0, 0, 0]
