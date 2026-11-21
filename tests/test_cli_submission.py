"""The readouts that ``audit`` takes after the proposer answers, and their skips."""

from __future__ import annotations

from types import SimpleNamespace

import click
import pytest

from openinterpretability.agents.proposer import HypothesisDraft, ProposerAgent
from openinterpretability.cli import cli  # noqa: F401  (loads the commands first)
from openinterpretability.cli_submission import (
    build_submission,
    read_patchscope,
    read_prior_art,
    read_specificity,
    read_steering,
    read_wiring,
)
from openinterpretability.core.schemas import ActivatingExample, NeuronIdentifier
from tests.test_heads import PROFILE

TOY = NeuronIdentifier(model_name="toy", layer=1, neuron_idx=2, pole="positive")


def _draft(base="a b", counterfactual="a c"):
    pair = {"base_prompt": base, "counterfactual_prompt": counterfactual,
            "expected_delta": 0.5, "rationale": "r"}  # fmt: skip
    return HypothesisDraft(explanation="x", activating_tokens=[], mechanism="retrospective",
                           counterfactuals=[pair] * 3)  # fmt: skip


@pytest.fixture
def head_submission(submission):
    data = submission.model_dump(mode="json")
    data["identifier"] = {"kind": "attn_head", "model_name": "toy", "layer": 1, "head_idx": 0}
    data["head_profile"] = PROFILE.model_dump()
    return type(submission).model_validate(data)


def test_a_head_has_no_patchscope_and_no_steering(toy_runner, head_submission):
    assert read_patchscope(toy_runner, head_submission) is None
    assert read_steering(toy_runner, head_submission) is None


def test_a_pole_without_a_tokenizer_has_no_patchscope(toy_model, submission):
    from openinterpretability.harness.hooks import NeuronHookRunner

    runner = NeuronHookRunner(toy_model, None)
    assert read_patchscope(runner, submission.model_copy(update={"identifier": TOY})) is None


def test_a_pole_without_a_maximum_has_no_steering(toy_runner, submission):
    sub = submission.model_copy(update={"identifier": TOY, "max_activation": 0.0})
    assert read_steering(toy_runner, sub) is None


def test_prior_art_says_it_looks_when_turned_on(toy_runner, submission, monkeypatch, capsys):
    monkeypatch.setenv("OPENINTERPRETABILITY_PRIOR_ART", "1")
    sub = submission.model_copy(update={"identifier": TOY})
    assert read_prior_art(toy_runner, sub) == []  # the toy model has no hosted dictionaries
    assert "Looking up prior art on Neuronpedia" in capsys.readouterr().err


def test_specificity_without_a_category_of_the_peak_words_is_skipped(
    toy_runner, submission, monkeypatch
):
    from openinterpretability import wordnet
    from tests.test_wordnet import FakeLexicon

    monkeypatch.setattr(wordnet, "lexicon", FakeLexicon)
    sub = submission.model_copy(update={"identifier": TOY})  # its peak word is a quote mark
    assert read_specificity(toy_runner, sub, _draft()) is None


def test_wiring_on_an_architecture_without_heads_is_skipped(toy_runner, head_submission, capsys):
    assert read_wiring(toy_runner, head_submission) is None
    assert "Skipping wiring:" in capsys.readouterr().err


def test_a_one_token_counterfactual_of_a_floor_unit_is_a_click_error(toy_runner):
    example = ActivatingExample(text="ab", tokens=["a", "b"], activations=[0.0, 1.0],
                                max_activation=1.0, max_token_index=1)  # fmt: skip
    evidence = SimpleNamespace(
        identifier=TOY, top_examples=[example], activation_histogram=None, provenance=None,
        confounds=None, correlated_neurons=[], vocabulary_lens=None, input_attribution=None,
        head_profile=None, dictionary_profile=None, first_token_floor=True,
        superposition_profile=None, manifold=None,
    )  # fmt: skip
    agent = ProposerAgent("openai/test")
    with pytest.raises(click.ClickException, match="Counterfactual: 'a' is one token"):
        build_submission(agent, _draft(base="a"), toy_runner, evidence)
