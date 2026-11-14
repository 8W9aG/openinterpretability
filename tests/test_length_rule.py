"""The proposed probe-length confound baseline (issue #3525)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from openinterp.agents.reviewer import AdversarialProbe, best_baseline  # noqa: E402
from openinterp.core.schemas import ReviewPolicy  # noqa: E402
from openinterp.harness.confounds import length_predictions  # noqa: E402


class Words:
    """A tokenizer with one token per word."""

    def __call__(self, texts, return_tensors=None):
        import torch

        return {"input_ids": torch.tensor([[0] * len(texts[0].split())])}


SHORT = ["a b", "c d e", "f g", "h i j"]
LONG = ["a b c d e f g h i j", "k l m n o p q r s t u", "a " * 12, "b " * 15]


def test_one_candidate_per_cut_and_the_first_token_left_out():
    got = length_predictions(Words(), ["a b", "a b c d"])
    assert got == {
        "probe length ≥ 2 tokens": [True, True],
        "probe length ≥ 4 tokens": [False, True],
    }
    floored = length_predictions(Words(), ["a b", "a b c d"], skip_first=True)
    assert set(floored) == {"probe length ≥ 1 tokens", "probe length ≥ 3 tokens"}


def test_it_explains_short_negatives_and_long_holdouts_perfectly():
    # The pattern of #3202: short hard negatives, long holdouts, a unit that fires late.
    probes = [AdversarialProbe(text=t, expected_active=False, rationale="") for t in SHORT]
    probes += [AdversarialProbe(text=t, expected_active=True, rationale="") for t in LONG]
    actual = [False] * 4 + [True] * 4
    name, score = best_baseline(length_predictions(Words(), [p.text for p in probes]), probes,
                                actual, ReviewPolicy())  # fmt: skip
    assert score == 1.0 and name.startswith("probe length ≥")
