"""Part of speech in the review and in triage (issue #3631)."""

from openinterp.agents import probe_hints
from openinterp.agents.reviewer import ReviewerAgent
from openinterp.core.confounds import PART_OF_SPEECH, ConfoundProfile, LabelConfound, LabelStat
from tests.conftest import FakeTokenizer


def confounds(pos_r2: float | None, token_r2: float = 0.4) -> ConfoundProfile:
    labels = None
    if pos_r2 is not None:
        level = LabelStat(label="DET", count=500, mean_strength=1.0, active=480)
        labels = {PART_OF_SPEECH: LabelConfound(
            r2=pos_r2, mutual_information_bits=0.3, labelled_tokens=900, unlabelled_tokens=0,
            levels=[level],
        )}  # fmt: skip
    return ConfoundProfile(
        token_r2=token_r2, position_r2=0.0, frequency_correlation=0.0, min_count=5,
        trigger_tokens=[], position_means=[], labels=labels,
    )  # fmt: skip


def tagged(submission, pos_r2=0.05):
    return submission.model_copy(update={"confounds": confounds(pos_r2)})


def test_the_probe_prompt_asks_for_another_role_only_when_tagged(submission):
    assert probe_hints.POS_PROBES in ReviewerAgent.probe_prompt(tagged(submission))
    assert probe_hints.POS_PROBES not in ReviewerAgent.probe_prompt(submission)


def test_the_token_check_tags_each_probes_words(submission, monkeypatch):
    from openinterp.agents.reviewer import AdversarialProbe

    def stub(text):
        return ((0, 3, "VERB"), (4, 7, "DET")) if text.startswith("run") else ()

    monkeypatch.setattr("openinterp.names.pos_matches", stub)
    probes = [AdversarialProbe(text="run the race", expected_active=False, rationale="")]
    prompt = ReviewerAgent.token_check_prompt(tagged(submission), probes, FakeTokenizer())
    assert "parts of speech: run/VERB the/DET" in prompt and probe_hints.POS_CHECK in prompt
    plain = ReviewerAgent.token_check_prompt(submission, probes, FakeTokenizer())
    assert "parts of speech" not in plain


def test_the_proposer_is_flagged_only_when_part_of_speech_explains_most():
    assert "Part of speech alone explains 30%" in probe_hints.pos_flag(confounds(0.3, 0.4))
    assert probe_hints.pos_flag(confounds(0.15, 0.2)) == ""  # below 0.2
    assert probe_hints.pos_flag(confounds(0.25, 0.9)) == ""  # under half of token identity
    assert probe_hints.pos_flag(confounds(None)) == ""
