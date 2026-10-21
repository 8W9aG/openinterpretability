"""Attention-head identifiers and hooks, on a randomly initialised GPT-2 (no download)."""

import pytest
import torch
from pydantic import TypeAdapter, ValidationError

from openinterp.core.schemas import (
    ComponentIdentifier,
    HeadIdentifier,
    HeadProfile,
    NeuronAuditSubmission,
    NeuronIdentifier,
)
from openinterp.harness.hooks import NeuronHookRunner, ablate_head

IDS = torch.tensor([[1, 5, 7, 2, 9, 3]])
PROFILE = HeadProfile(
    mean_output=[0.0] * 16,
    mean_tokens=1,
    previous_token_score=0.1,
    duplicate_token_score=0.2,
    induction_score=0.3,
    pattern_sequences=1,
    pattern_length=4,
    pattern_seed=0,
)


@pytest.fixture
def runner(tiny_gpt2):
    return NeuronHookRunner(tiny_gpt2)


def test_component_identifiers_round_trip_and_old_records_still_validate():
    adapter = TypeAdapter(ComponentIdentifier)
    old = '{"model_name": "gpt2", "layer": 3, "neuron_idx": 7, "pole": "negative"}'
    neuron = adapter.validate_json(old)
    assert isinstance(neuron, NeuronIdentifier) and neuron.kind == "mlp_neuron_pole"
    assert NeuronIdentifier.model_validate_json(old) == neuron
    assert neuron.slug == "gpt2/L3/N7-negative"

    head = adapter.validate_json(
        '{"kind": "attn_head", "model_name": "gpt2", "layer": 5, "head_idx": 1}'
    )
    assert isinstance(head, HeadIdentifier)
    assert (head.slug, str(head), head.index) == ("gpt2/L5/H1", "gpt2 L5/H1", 1)
    assert (neuron.label, neuron.index) == ("L3/N7-", 7)
    assert adapter.validate_json(head.model_dump_json()) == head
    with pytest.raises(ValidationError):
        adapter.validate_json(
            '{"kind": "attn_head", "model_name": "gpt2", "layer": 5, "pole": "positive"}'
        )


def test_a_head_submission_needs_its_profile_and_a_neuron_has_none(submission):
    data = submission.model_dump(mode="json")
    data["identifier"] = {"kind": "attn_head", "model_name": "gpt2", "layer": 0, "head_idx": 1}
    with pytest.raises(ValidationError, match="needs a head_profile"):
        NeuronAuditSubmission.model_validate(data)
    data["head_profile"] = PROFILE.model_dump()
    head = NeuronAuditSubmission.model_validate(data)
    assert head.identifier.label == "L0/H1" and head.identifier.pole is None

    data["identifier"] = submission.identifier.model_dump()
    with pytest.raises(ValidationError, match="only for attention heads"):
        NeuronAuditSubmission.model_validate(data)


def test_head_geometry(runner, tiny_gpt2):
    assert runner.attn_out(0) is tiny_gpt2.transformer.h[0].attn.c_proj
    assert (runner.n_heads, runner.head_dim(0)) == (2, 16)
    assert runner.head_output_weights(1, 1).shape == (32, 16)
    with pytest.raises(IndexError):
        runner.head_output_weights(0, 2)


def test_heads_sum_to_the_attention_output(runner):
    """The decomposition the hooks rely on: attn output = Σ_h W_O,h z_h + bias."""
    captured = []
    proj = runner.attn_out(1)
    handle = proj.register_forward_hook(lambda _m, _a, out: captured.append(out))
    runner.run_with_interventions(IDS)
    handle.remove()
    z = runner.head_outputs(IDS, 1)
    assert z.shape == (1, IDS.shape[1], 2, 16)
    rebuilt = sum(z[..., h, :] @ runner.head_output_weights(1, h).T for h in range(2)) + proj.bias
    torch.testing.assert_close(rebuilt, captured[-1], atol=1e-5, rtol=1e-5)

    norms = runner.head_write_norm(IDS, 1, 0)
    torch.testing.assert_close(
        norms, (z[..., 0, :] @ runner.head_output_weights(1, 0).T).norm(dim=-1)
    )


def test_ablating_one_head_leaves_the_other(runner):
    seen = []
    clean = runner.run_with_interventions(IDS)
    ablated = runner.run_with_interventions(
        IDS, head_interventions={0: lambda z: seen.append(ablate_head(0, 16)(z)) or seen[-1]}
    )
    z_clean = runner.head_outputs(IDS, 0).flatten(-2)
    assert torch.all(seen[-1][..., :16] == 0)
    torch.testing.assert_close(seen[-1][..., 16:], z_clean[..., 16:])
    assert not torch.allclose(clean, ablated)


def test_attention_patterns_are_causal_distributions(runner, tiny_gpt2):
    tiny_gpt2.set_attn_implementation("eager")
    patterns = runner.attention_patterns(IDS, 1)
    seq = IDS.shape[1]
    assert patterns.shape == (1, 2, seq, seq)
    torch.testing.assert_close(patterns.sum(-1), torch.ones(1, 2, seq))
    assert torch.all(patterns.triu(diagonal=1) == 0)  # no attention to later tokens
