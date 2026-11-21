import pytest

from openinterpretability.harness import scan_backend
from openinterpretability.harness.scan import Piece, TopExamples


def _top(peaks):
    top = TopExamples()
    for doc, peak in enumerate(peaks):
        top.add(Piece(doc, 0, 3, f"doc{doc}"), [1, 2, 3], [0.0, peak, 0.0])
    return top


def test_rescoring_stops_after_its_last_round_even_with_examples_left():
    top = _top([float(1000 - i) for i in range(600)])
    # Each measured piece drops out, so every round the top k holds new pieces.
    measured = scan_backend.rescore(top, lambda texts: [[0.0, -1e3, 0.0] for _ in texts], k=60)
    dropped = sum(v[1] == -1e3 for *_, v in top.candidates)
    assert measured == dropped
    assert measured > (scan_backend.RESCORE_ROUNDS - 1) * 60
    assert measured < len(top.candidates)  # pieces were left when the rounds ran out


def test_a_measure_with_the_wrong_token_count_is_an_error():
    with pytest.raises(ValueError, match="measured 2 tokens of a 3-token piece"):
        scan_backend.rescore(_top([1.0]), lambda texts: [[0.0, 1.0] for _ in texts], k=1)


def test_nothing_to_choose_from_measures_nothing():
    assert scan_backend.chosen([]) == set()
    assert scan_backend.rescore(TopExamples(), lambda texts: [], k=5) == 0


def test_kept_skips_an_overlapping_piece_and_keeps_fewer_than_k_when_short():
    top = TopExamples()
    top.add(Piece(0, 0, 3, "a"), [1, 2, 3], [0.0, 5.0, 0.0])
    top.add(Piece(0, 2, 5, "b"), [3, 4, 5], [0.0, 4.0, 0.0])  # shares token 2 with "a"
    top.add(Piece(1, 0, 3, "c"), [1, 2, 3], [0.0, 3.0, 0.0])
    assert scan_backend.kept(top.candidates, k=10) == {0, 2}
