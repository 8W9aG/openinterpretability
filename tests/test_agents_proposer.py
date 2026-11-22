"""The proposer's notes and attribution text when parts are missing, and an unmeasured draft."""

from openinterpretability.agents.proposer import (
    CounterfactualDraft,
    HypothesisDraft,
    ProposerAgent,
    dropped_note,
    format_input_attribution,
)
from openinterpretability.core.attribution import AttributedToken, ContextExample, InputAttribution
from openinterpretability.core.schemas import NeuronIdentifier


def attribution(examples, **fields):
    return InputAttribution(baseline_token="<|endoftext|>", tolerance=0.1, examples=examples,
                            distances=[], **fields)  # fmt: skip


def test_no_dropped_examples_give_no_note():
    assert dropped_note([]) == ""
    assert dropped_note([2, 5]) == "(Omitted as near-duplicates of an earlier example: 2, 5.)\n"


def test_attribution_with_no_examples_gives_no_text():
    assert format_input_attribution(attribution([])) == ""


def test_attribution_without_distances_or_ablation_lists_only_the_tokens():
    example = ContextExample(
        example=0, peak_token='"', tokens=4, steps=16, converged=False,
        top=[AttributedToken(offset=-1, token="Hi", attribution=0.3)], completeness_error=0.5,
    )  # fmt: skip
    text = format_input_attribution(attribution([example]))
    assert "converged on 0): Tokens driving each peak most" in text
    assert "on the peak token itself" not in text and "rho =" not in text
    assert text.endswith("  example 1, peak '\"' (not converged): 'Hi' (-1) 0.30/?\n")


def test_a_draft_built_without_a_runner_keeps_its_claims_unmeasured():
    ident = NeuronIdentifier(model_name="toy", layer=1, neuron_idx=2, pole="positive")
    pair = CounterfactualDraft(base_prompt="aaaa", counterfactual_prompt="abcd",
                               expected_delta=0.5, rationale="r")  # fmt: skip
    draft = HypothesisDraft(explanation="Fires on repeated letters.", activating_tokens=["a"],
                            mechanism="retrospective", counterfactuals=[pair] * 3)  # fmt: skip
    sub = ProposerAgent("openai/test").build_submission(ident, draft)
    assert sub.natural_language_explanation == "Fires on repeated letters."
    assert [cf.observed_delta for cf in sub.counterfactuals] == [None] * 3
