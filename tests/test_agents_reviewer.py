"""The reviewer's confound baseline (agents/reviewer.py)."""

from types import SimpleNamespace

from openinterpretability.agents.reviewer import AdversarialProbe, confound_baseline
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
