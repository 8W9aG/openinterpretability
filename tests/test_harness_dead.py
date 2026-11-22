"""The dead-pole proof on an RMSNorm, an unknown norm, a biased gated MLP and a missing MLP."""

import torch
from torch import nn

from openinterpretability.harness.dead import negative_pole_ceiling, prove_dead
from tests.test_dead import gpt2_runner, neuron, qwen_runner  # noqa: F401  (fixtures)


def test_an_rmsnorm_before_the_mlp_bounds_the_pre_activation(
    gpt2_runner,  # noqa: F811
    tiny_gpt2,
):
    block = tiny_gpt2.transformer.h[1]
    block.ln_2 = nn.RMSNorm(32)
    with torch.no_grad():
        block.ln_2.weight.uniform_(0.5, 1.5)
    x = torch.randn(4000, 32) * torch.logspace(-3, 3, 4000).unsqueeze(1)
    with torch.no_grad():
        h = block.mlp.c_fc(block.ln_2(x))[:, 5]
    pos = prove_dead(gpt2_runner, neuron(1, 5))
    neg = prove_dead(gpt2_runner, neuron(1, 5, "negative"))
    assert not pos.proven and not neg.proven
    assert h.max() <= pos.bound and -h.min() <= neg.bound
    with torch.no_grad():
        block.mlp.c_fc.bias[5] = -1000.0
    assert prove_dead(gpt2_runner, neuron(1, 5)).proven


def test_a_gemma_style_rmsnorm_gives_no_bound(gpt2_runner, tiny_gpt2):  # noqa: F811
    class GemmaRMSNorm(nn.RMSNorm):
        pass  # Gemma scales by 1 + weight, so the proof must not use its weight

    tiny_gpt2.transformer.h[1].ln_2 = GemmaRMSNorm(32)
    with torch.no_grad():
        tiny_gpt2.transformer.h[1].mlp.c_fc.bias[5] = -1000.0
    proof = prove_dead(gpt2_runner, neuron(1, 5))
    assert not proof.proven and proof.bound is None
    assert proof.reason == "no bound for the norm GemmaRMSNorm"


def test_an_unknown_norm_gives_no_bound(gpt2_runner, tiny_gpt2):  # noqa: F811
    tiny_gpt2.transformer.h[1].ln_2 = nn.Identity()
    proof = prove_dead(gpt2_runner, neuron(1, 5))
    assert not proof.proven and proof.bound is None
    assert proof.reason == "no bound for the norm Identity"


def test_a_gated_mlp_with_biases_is_not_proven(qwen_runner):  # noqa: F811
    mlp = qwen_runner.model.model.layers[1].mlp
    mlp.gate_proj = nn.Linear(32, 48, bias=True)
    with torch.no_grad():
        mlp.gate_proj.weight[7] = 0.0  # a zero read proves nothing when a bias is added
    proof = prove_dead(qwen_runner, neuron(1, 7, model="q"))
    assert not proof.proven
    assert proof.reason == "no bound for a gated MLP with biases"


def test_an_mlp_with_no_known_input_projection_is_not_proven(
    gpt2_runner,  # noqa: F811
    tiny_gpt2,
):
    tiny_gpt2.transformer.h[1].mlp.c_fc = None
    proof = prove_dead(gpt2_runner, neuron(1, 5))
    assert not proof.proven
    assert proof.reason == "MLP input projection not found"
    assert negative_pole_ceiling(gpt2_runner, 1) is None
