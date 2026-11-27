"""The agents pass each argument on, and write their prompts and notes exactly (#6491)."""

from __future__ import annotations

import json

import pytest

from openinterpretability.agents import base
from openinterpretability.agents.judge import DetectionAnswer, ExplanationJudge, FuzzingAnswer
from openinterpretability.agents.narrator import Narrative, Narrator, ReportedNarrative
from openinterpretability.agents.probe_hints import pos_line
from openinterpretability.agents.proposer import ProposerAgent
from openinterpretability.agents.proposer_profiles import (
    SHOWN_MEMBERS,
    format_ov_lens,
    format_superposition_profile,
)
from openinterpretability.core.profiles import (
    HeadProfile,
    OVLens,
    OVMapping,
    SuperpositionMember,
    SuperpositionProfile,
)


def _asked(agent, monkeypatch, answer=None):
    """Record each (response model, prompt) that ``agent`` asks, and answer ``answer``."""
    calls = []

    def structured(model, prompt):
        calls.append((model, prompt))
        return answer

    monkeypatch.setattr(agent, "structured", structured)
    return calls


def test_a_text_with_no_words_has_no_parts_of_speech():
    assert pos_line("") == "   parts of speech: none"


PROPOSAL_ARGS = ("lens", "confounds", "correlated", "head_profile", "attribution",
                 "dictionary_profile", "first_token_floor", "full_snippets", "related",
                 "superposition_profile", "manifold")  # fmt: skip


def test_propose_passes_every_argument_to_the_prompt_in_order(monkeypatch):
    seen = []
    monkeypatch.setattr(ProposerAgent, "proposal_prompt",
                        staticmethod(lambda *args: seen.append(args) or "task"))  # fmt: skip
    agent = ProposerAgent("x")
    calls = _asked(agent, monkeypatch)
    agent.propose("id", "examples")
    assert seen[-1] == ("id", "examples", None, None, (), None, None, None, False, False, "",
                        None, None)  # fmt: skip
    marks = {name: object() for name in PROPOSAL_ARGS}
    agent.propose("id", "examples", **marks)
    assert seen[-1] == ("id", "examples", *marks.values())
    assert calls[-1][1] == "task"


@pytest.mark.parametrize("noun", [None, "head"])
def test_the_judge_asks_each_question_with_its_answer_type_and_noun(monkeypatch, noun):
    judge = ExplanationJudge("x")
    calls = _asked(judge, monkeypatch)
    extra = {"noun": noun} if noun else {}
    judge.detect("fires on cats", [], **extra)
    judge.fuzz("fires on cats", [], **extra)
    noun = noun or "neuron"
    assert calls == [
        (DetectionAnswer, ExplanationJudge.detection_prompt("fires on cats", [], noun)),
        (FuzzingAnswer, ExplanationJudge.fuzzing_prompt("fires on cats", [], noun)),
    ]
    assert f"Explanation of the {noun}:" in calls[0][1]


def _narrative(**extra):
    return {"sections": [{"title": "T", "text": "x"}], "paragraph": "p"} | extra


def test_the_narrator_asks_with_the_evidence_as_indented_json(monkeypatch):
    narrator = Narrator("some-model")
    answer = Narrative.model_validate(_narrative())
    calls = _asked(narrator, monkeypatch, answer)
    evidence = {"a": [1, 2], "b": {"c": "d"}}
    assert narrator.narrate(evidence) is answer
    task = ("Explain why the model gives this answer, from this evidence only.\n\n"
            f"```json\n{json.dumps(evidence, indent=1)}\n```")  # fmt: skip
    assert calls == [(Narrative, task)]


def test_an_external_narrator_returns_the_narrative_without_its_model(monkeypatch):
    narrator = Narrator("external/claude-code")
    told = ReportedNarrative.model_validate(_narrative(model="claude-haiku-5-5"))
    calls = _asked(narrator, monkeypatch, told)
    told_back = narrator.narrate({})
    assert type(told_back) is Narrative and told_back == Narrative.model_validate(_narrative())
    assert calls[0][0] is ReportedNarrative


def test_an_external_answer_waits_one_poll_for_a_response_caught_mid_write(
    tmp_path, monkeypatch, capsys
):
    exchange = tmp_path / "a" / "b"  # it makes the missing parents
    agent = Narrator("external/x", exchange_dir=exchange, timeout=5, poll_interval=0)
    response = exchange / "Narrator.response.json"
    whole = json.dumps(_narrative())
    sleeps = []

    def sleep(_seconds):
        sleeps.append(_seconds)
        if len(sleeps) == 1:  # the first read was half a file; the answerer finishes it
            response.write_text(whole)

    monkeypatch.setattr(base.time, "sleep", sleep)
    real_write = base._write_whole

    def write(path, text):
        real_write(path, text)
        if path.name.endswith(".prompt.md"):
            response.write_text(whole[:10])  # half-written

    monkeypatch.setattr(base, "_write_whole", write)
    assert agent._external(Narrative, "the task") == Narrative.model_validate(_narrative())
    prompt = exchange / "Narrator.prompt.md"
    assert prompt.read_text() == Narrator.render_prompt(Narrative, "the task", response)
    err = capsys.readouterr().err
    assert f"Waiting for {response} (prompt: {prompt})…" in err
    assert "Invalid response" not in err


def test_an_external_answer_that_stays_invalid_is_reported(tmp_path, monkeypatch, capsys):
    agent = Narrator("external/x", exchange_dir=tmp_path, timeout=5, poll_interval=0)
    response, error = tmp_path / "Narrator.response.json", tmp_path / "Narrator.error.txt"
    polls = []

    def sleep(_seconds):
        polls.append(_seconds)
        if len(polls) == 1:
            response.write_text("{bad")
        elif error.exists() and not response.exists():
            response.write_text(json.dumps(_narrative()))

    monkeypatch.setattr(base.time, "sleep", sleep)
    agent._external(Narrative, "the task")
    assert f"Invalid response, see {error}" in capsys.readouterr().err


def _head(maps):
    lens = OVLens(copying_score=0.5, copy_share=0.1, suppression_share=0.0, median_rank=3,
                  tokens=10, seed=0, k=10, mappings=maps)  # fmt: skip
    return HeadProfile(mean_output=[0.0], mean_tokens=1, previous_token_score=0.0,
                       duplicate_token_score=0.0, induction_score=0.0, pattern_sequences=1,
                       pattern_length=2, pattern_seed=0, ov_lens=lens)  # fmt: skip


def test_the_ov_note_is_one_line_and_names_mappings_only_when_there_are_some():
    assert format_ov_lens(_head([]).model_copy(update={"ov_lens": None})) == ""
    plain = format_ov_lens(_head([]))
    assert plain.endswith("median own rank 3.\n") and plain.count("\n") == 1
    mapped = format_ov_lens(_head([OVMapping(token="a", top=["b"])]))
    assert mapped.endswith("median own rank 3. The top examples' attended tokens map to: "
                           "'a' -> 'b'.\n")  # fmt: skip


def _superposition(weights):
    members = [SuperpositionMember(neuron_idx=n, weight=w) for n, w in enumerate(weights)]
    return SuperpositionProfile(members=members, method="circuit", source="test")


def test_the_superposition_note_lists_members_by_relative_weight():
    note = format_superposition_profile(_superposition([0.5, -2.0, 1.0]))
    assert note == (
        "This unit is a superposition: one direction over 3 of the layer's neurons, active "
        "where the weighted sum of their activations is positive. Members, by relative weight: "
        "N1 (-1.00), N2 (+0.50), N0 (+0.25). It was found by circuit (test). Explain what the "
        "direction as a whole responds to, not one neuron."
    )


@pytest.mark.parametrize(
    ("count", "rest"), [(SHOWN_MEMBERS, ""), (SHOWN_MEMBERS + 1, ", and 1 more")]
)
def test_the_superposition_note_counts_the_members_it_leaves_out(count, rest):
    note = format_superposition_profile(_superposition([float(n + 1) for n in range(count)]))
    shown = note.split("by relative weight: ")[1].split(". It was found")[0]
    assert shown.endswith(
        f"N{count - SHOWN_MEMBERS} ({(count - SHOWN_MEMBERS + 1) / count:+.2f}){rest}"
    )
