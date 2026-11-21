"""``ActivatingExample`` refuses tokens, peaks and attention that do not line up."""

import pytest
from pydantic import ValidationError

from openinterpretability.core.examples import ActivatingExample

GOOD = dict(text="a b c", tokens=["a", " b", " c"], activations=[0.1, 0.5, 0.2],
            max_activation=0.5, max_token_index=1)  # fmt: skip


def test_a_well_formed_example_with_attention_is_accepted():
    example = ActivatingExample(**GOOD, attends_to=[1, 0], attention_weights=[0.6, 0.3])
    assert example.attends_to == [1, 0]


@pytest.mark.parametrize(
    "change, message",
    [
        ({"max_token_index": 3}, "max_token_index out of range"),
        ({"attends_to": [0]}, "attends_to and attention_weights must be the same length"),
        (
            {"attends_to": [2], "attention_weights": [0.5]},
            "attends_to must index tokens up to the peak",
        ),
    ],
)
def test_a_misaligned_example_is_refused(change, message):
    with pytest.raises(ValidationError, match=message):
        ActivatingExample(**GOOD | change)
