"""The reviewer's confound baseline (agents/reviewer.py)."""

from types import SimpleNamespace

from openinterpretability.agents.reviewer import (
    AdversarialProbe,
    ReviewerAgent,
    confound_baseline,
    repeated_runs,
)
from openinterpretability.core.fence import fence
from openinterpretability.core.identifiers import HeadIdentifier
from openinterpretability.core.schemas import ReviewPolicy


class Words:
    """A tokenizer with one token per word."""

    def encode(self, text):
        return [0] * len(text.split())


PROBES = [AdversarialProbe(text="a b c", expected_active=True, rationale="")]
PROFILE = SimpleNamespace(trigger_tokens=[], position_means=[])


def _baseline(confounds, tokenizer, probes):
    submission = SimpleNamespace(confounds=confounds, first_token_floor=False)
    runner = SimpleNamespace(tokenizer=tokenizer)
    actual = [True] * len(probes)
    return confound_baseline(submission, runner, probes, actual, 1.0, ReviewPolicy())


def test_no_confound_baseline_without_a_profile_a_tokenizer_or_probes():
    assert _baseline(None, Words(), PROBES) is None  # a scan from before protocol 0.3
    assert _baseline(PROFILE, None, PROBES) is None
    assert _baseline(PROFILE, Words(), []) is None


class Vocab:
    """A tokenizer with one token per word, which decodes each token with a leading space."""

    def __init__(self):
        self.ids = {}

    def encode(self, text):
        return [self.ids.setdefault(w, len(self.ids)) for w in text.split()]

    def decode(self, ids):
        return " " + {i: w for w, i in self.ids.items()}[ids[0]]


CHECK = (
    "Check each label against what the hypothesis predicts for these tokens, not for the "
    "words. Where a label is wrong, rewrite the probe so it tests what you meant (preferred, "
    "to keep the share of hard negatives), or relabel it. Leave the other probes unchanged "
    "and write the whole probe set again.\n\n"
)
INTRO = "\n\nYou wrote the probes below. Here they are as the model's tokens. "


def _length_warning(ratio):
    return (
        f"The median token lengths of the active and inactive probes differ {ratio}×, above "
        "1.5×. A rule that reads only probe length may then explain the probes as well as the "
        "hypothesis does, and the review fails a hypothesis that doesn't beat it. Rewrite some "
        "probes to match the lengths. "
    )


UNEVEN = [
    AdversarialProbe(text="he said a b a b", expected_active=True, rationale=""),
    AdversarialProbe(text="no quote", expected_active=False, rationale=""),
]


def test_the_token_check_shows_each_probe_as_the_models_tokens(submission):
    prompt = ReviewerAgent.token_check_prompt(submission, UNEVEN, Vocab())
    blocks = (
        "0. expected active: 'he said a b a b'\n"
        "   tokens: ' he' | ' said' | ' a' | ' b' | ' a' | ' b'\n\n"
        "1. expected inactive: 'no quote'\n   tokens: ' no' | ' quote'"
    )
    assert prompt == (
        ReviewerAgent.probe_prompt(submission, 2) + INTRO + _length_warning("3.0") + CHECK
        + fence(blocks)
    )  # fmt: skip


def test_the_token_check_of_a_head_lists_repeated_runs_and_reads_the_floor(submission):
    head = HeadIdentifier(model_name="Qwen/Qwen2.5-0.5B", layer=3, head_idx=2)
    sub = submission.model_copy(update={"identifier": head, "first_token_floor": True})
    probes = [AdversarialProbe(text="a b x a b y x", expected_active=True, rationale=""),
              UNEVEN[1]]  # fmt: skip
    prompt = ReviewerAgent.token_check_prompt(sub, probes, Vocab())
    blocks = (
        "0. expected active: 'a b x a b y x'\n"
        "   tokens: ' a' | ' b' | ' x' | ' a' | ' b' | ' y' | ' x'\n"
        "   repeated token runs: ' a b' (2), ' x' (1)\n\n"
        "1. expected inactive: 'no quote'\n   tokens: ' no' | ' quote'\n"
        "   repeated token runs: none"
    )
    repeats = (
        "Each 'repeated token runs' entry is a longest run of tokens that also occurs earlier "
        "in the probe, with its length in tokens.\n"
    )
    assert prompt == (
        ReviewerAgent.probe_prompt(sub, 2) + INTRO + repeats + _length_warning("6.0") + CHECK
        + fence(blocks)
    )  # fmt: skip


def test_probes_of_even_length_get_no_length_warning(submission):
    probes = [
        AdversarialProbe(text="he said", expected_active=True, rationale=""),
        AdversarialProbe(text="no quote", expected_active=False, rationale=""),
    ]
    prompt = ReviewerAgent.token_check_prompt(submission, probes, Vocab())
    assert INTRO + CHECK in prompt


def test_repeated_runs_are_the_longest_runs_that_occur_earlier():
    assert repeated_runs([]) == []
    assert repeated_runs([1, 2, 3]) == []
    assert repeated_runs([5, 1, 2, 5, 1, 2]) == [(3, 6)]
    assert repeated_runs([1, 2, 9, 1, 2, 8]) == [(3, 5)]
    assert repeated_runs([7, 7, 7]) == [(1, 3)]
    assert repeated_runs([1, 2, 1, 3, 2, 3]) == [(2, 3), (4, 5), (5, 6)]
