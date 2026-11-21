"""The multi-dimensional feature check (issue #3231): its shapes, null and wiring."""

import math

import torch

from openinterpretability.core.manifold import (
    FamilyManifold,
    ManifoldCheck,
    ManifoldPole,
    format_for_proposer,
)
from openinterpretability.core.schemas import ActivatingExample
from openinterpretability.harness import manifold


def circle_rows(n: int, phase: float = 0.0) -> torch.Tensor:
    """``n`` members in order round a circle in the first two poles, with a quiet third."""
    rows = [
        [1 + math.cos(2 * math.pi * k / n + phase), 1 + math.sin(2 * math.pi * k / n + phase)]
        for k in range(n)
    ]
    noise = torch.randn(n, 1, generator=torch.Generator().manual_seed(0)) * 0.01
    return torch.cat([torch.tensor(rows), noise], dim=1)


def test_members_in_order_round_a_circle_score_near_one_with_a_small_p():
    found = manifold.family_manifold("weekdays", circle_rows(7, 0.4), threshold=0.5)
    assert found.shape == "circle" and found.score > 0.95
    assert found.p_value < 0.01 and found.components == [1, 2]
    assert found.pole_share > 0.9  # the audited pole is one axis of the circle
    assert found.roundness > 0.9


def test_a_reversed_circle_scores_as_high_as_a_forward_one():
    reversed_rows = circle_rows(12).flip(0)
    assert manifold.family_manifold("months", reversed_rows, 0.5).score > 0.95


def test_unordered_members_show_no_circle():
    rows = torch.randn(12, 6, generator=torch.Generator().manual_seed(3))
    assert manifold.family_manifold("months", rows, 0.5).p_value > 0.05


def test_numbers_along_a_line_score_near_one():
    values = torch.arange(12, dtype=torch.float32)[:, None]
    rows = torch.cat([values, torch.ones(12, 1), -values], dim=1)
    found = manifold.family_manifold("numbers", rows, 0.5)
    assert found.shape == "line" and found.score > 0.99 and found.components == [1]
    assert found.p_value < 0.01


def test_constant_rows_have_no_shape_and_no_components():
    found = manifold.family_manifold("weekdays", torch.ones(7, 3), 0.5)
    assert (found.score, found.p_value, found.components) == (0.0, 1.0, [])
    assert found.explained_variance == []


def test_active_members_count_the_audited_pole_only():
    rows = torch.tensor([[0.0, 9.0], [1.0, 9.0], [2.0, 0.0]] * 4)
    assert manifold.family_manifold("months", rows, 1.0).active_members == 8


def test_the_p_value_replays():
    rows = torch.randn(7, 4, generator=torch.Generator().manual_seed(5))
    first = manifold.family_manifold("weekdays", rows, 0.5)
    assert manifold.family_manifold("weekdays", rows, 0.5) == first


def test_participation_ratio_counts_equal_dimensions():
    assert manifold.participation_ratio(torch.tensor([1.0, 1.0, 1.0])) == 3.0
    assert manifold.participation_ratio(torch.tensor([1.0, 0.0])) == 1.0
    assert manifold.participation_ratio(torch.zeros(2)) == 0.0


def family(**update) -> FamilyManifold:
    base = dict(
        family="weekdays", shape="circle", members=7, active_members=5,
        explained_variance=[0.6, 0.3], components=[2, 3], score=0.9, p_value=0.01,
        pole_share=0.8, roundness=0.8,
    )  # fmt: skip
    return FamilyManifold(**{**base, **update})


def check(*families) -> ManifoldCheck:
    poles = [
        ManifoldPole(neuron_idx=3, pole="positive"),
        ManifoldPole(neuron_idx=8, pole="negative"),
    ]
    return ManifoldCheck(
        poles=poles, activating_examples=20, activating_explained_variance=[0.5, 0.5],
        participation_ratio=2.0, families=list(families),
    )  # fmt: skip


def test_the_proposer_sees_a_shape_and_the_allowed_form():
    note = format_for_proposer(check(family()))
    assert "N3+, N8-" in note and "circle in PC2–PC3" in note
    assert "one coordinate of a multi-dimensional feature" in note


def test_the_proposer_sees_no_shape_without_significance_or_activity():
    for quiet in (family(p_value=0.2), family(active_members=1), family(roundness=0.1)):
        note = format_for_proposer(check(quiet))
        assert "trace" not in note and "coordinate" not in note


def test_the_check_runs_on_a_model_and_round_trips(toy_runner):
    text = "the cat sat on the mat"
    ex = ActivatingExample(
        text=text, tokens=list(text), activations=[0.1] * len(text), max_activation=1.0,
        max_token_index=3,
    )  # fmt: skip
    poles = [
        ManifoldPole(neuron_idx=0, pole="positive"),
        ManifoldPole(neuron_idx=1, pole="negative"),
    ]
    found = manifold.manifold_check(
        toy_runner, 0, poles, [ex, ex.model_copy(update={"text": "a b"})]
    )
    assert found.poles == poles and found.activating_examples == 2
    assert [f.family for f in found.families] == list(manifold.FAMILIES)
    assert ManifoldCheck.model_validate_json(found.model_dump_json()) == found


def test_the_proposer_prompt_carries_the_check():
    from openinterpretability.agents.proposer import ProposerAgent
    from openinterpretability.core.schemas import NeuronIdentifier

    ident = NeuronIdentifier(model_name="m", layer=0, neuron_idx=3, pole="positive")
    prompt = ProposerAgent.proposal_prompt(ident, [], manifold=check(family()))
    assert "one coordinate of a multi-dimensional feature" in prompt


def test_a_partner_silent_on_the_family_does_not_fill_a_component():
    rows = circle_rows(7)  # the third pole jitters by 0.01 around 0
    found = manifold.family_manifold("weekdays", rows, 0.5, scales=torch.tensor([2.0, 2.0, 2.0]))
    assert found.components == [1, 2] and found.score > 0.95 and found.explained_variance[2] < 0.01


def test_a_pole_with_no_scale_is_zero():
    rows = torch.tensor([[1.0, 4.0], [3.0, 8.0]])
    assert manifold.scaled(rows, torch.tensor([2.0, 0.0])).tolist() == [[0.5, 0.0], [1.5, 0.0]]


def test_a_line_has_no_roundness():
    values = torch.arange(12, dtype=torch.float32)[:, None]
    assert manifold.family_manifold("numbers", values, 0.5).roundness is None


def test_a_tokenizer_whose_offsets_do_not_match_the_tokens_gives_no_family(toy_model):
    from openinterpretability.harness.hooks import NeuronHookRunner
    from tests.conftest import FakeTokenizer

    class ExtraOffset(FakeTokenizer):
        """Gives one more offset than the model sees tokens, as a BOS-only offset would."""

        def __call__(self, texts, **kwargs):
            out = super().__call__(texts, **kwargs)
            if isinstance(texts, str):
                out["offset_mapping"] = [(0, 0), *out["offset_mapping"]]
            return out

    runner = NeuronHookRunner(toy_model, ExtraOffset())
    poles = [ManifoldPole(neuron_idx=0, pole="positive")]
    assert manifold.word_vector(runner, 0, poles, "Monday") is None
    text = "the cat sat"
    ex = ActivatingExample(
        text=text, tokens=list(text), activations=[0.1] * len(text), max_activation=1.0,
        max_token_index=3,
    )  # fmt: skip
    found = manifold.manifold_check(runner, 0, poles, [ex])
    assert found.families == [] and found.activating_examples == 1
