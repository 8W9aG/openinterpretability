"""The probe-length confound baseline of protocol 0.9, and length-matched probes (#3525)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from types import SimpleNamespace  # noqa: E402

from openinterpretability.agents.reviewer import (  # noqa: E402
    LENGTH_MATCHED,
    LENGTH_RATIO_WARN,
    AdversarialProbe,
    ReviewerAgent,
    best_baseline,
    confound_baseline,
    length_ratio,
    length_warning,
)
from openinterpretability.core.schemas import ReviewPolicy  # noqa: E402
from openinterpretability.harness.confounds import length_predictions  # noqa: E402


class Words:
    """A tokenizer with one token per word."""

    def __call__(self, texts, return_tensors=None):
        import torch

        return {"input_ids": torch.tensor([[0] * len(texts[0].split())])}

    def encode(self, text):
        return [0] * len(text.split())


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


def _probes(negatives, holdouts):
    return [AdversarialProbe(text=t, expected_active=False, rationale="") for t in negatives] + [
        AdversarialProbe(text=t, expected_active=True, rationale="") for t in holdouts
    ]


def _submission(floor=False):
    confounds = SimpleNamespace(trigger_tokens=[], position_means=[])
    return SimpleNamespace(confounds=confounds, first_token_floor=floor)


def test_the_confound_baseline_includes_probe_length():
    probes = _probes(SHORT, LONG)
    actual = [False] * 4 + [True] * 4
    runner = SimpleNamespace(tokenizer=Words())
    name, score = confound_baseline(_submission(), runner, probes, actual, 1.0, ReviewPolicy())
    assert score == 1.0 and name.startswith("probe length ≥")  # a length-only claim can't beat it


def test_length_matched_probes_leave_the_length_rule_no_edge():
    same = ["a b c d", "e f g h", "i j k l", "m n o p"]
    probes = _probes(same, ["q r s t", "u v w x", "y z a b", "c d e f"])
    actual = [False, True, False, True, True, False, True, False]  # activity unrelated to length
    runner = SimpleNamespace(tokenizer=Words())
    _, score = confound_baseline(_submission(), runner, probes, actual, 1.0, ReviewPolicy())
    assert score == 0.5


def test_the_length_warning_fires_above_the_ratio_only():
    unequal = _probes(SHORT, LONG)
    assert length_ratio(unequal, Words(), False) > LENGTH_RATIO_WARN
    assert "differ" in length_warning(unequal, Words(), False)
    matched = _probes(["a b c d"] * 2, ["a b c d e"] * 2)  # 1.25×
    assert length_warning(matched, Words(), False) is None
    assert length_ratio(_probes(SHORT, []), Words(), False) is None  # one group is empty


def test_the_reviewer_is_asked_for_length_matched_probes(submission):
    assert LENGTH_MATCHED in ReviewerAgent.probe_prompt(submission, 10)
