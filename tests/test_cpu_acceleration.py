"""The CPU options script's agreement check and fused GELU (scripts/cpu_acceleration.py, #3268)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers.activations import NewGELUActivation

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import cpu_acceleration as ca  # noqa: E402


def test_agreement_uses_reproduces_atol_of_the_largest_activation():
    reference = torch.tensor([0.0, 2.0, -4.0])
    assert ca.agreement(reference.clone(), reference)["within_reproduce_atol"]
    close = reference + torch.tensor([0.0, 0.003, 0.0])  # 0.003 <= 0.001 * 4
    assert ca.agreement(close, reference)["within_reproduce_atol"]
    far = reference + torch.tensor([0.0, 0.005, 0.0])
    got = ca.agreement(far, reference)
    assert not got["within_reproduce_atol"]
    assert abs(got["max_abs_diff_over_max"] - 0.005 / 4) < 1e-6


def test_fuse_gelu_swaps_each_block_and_stays_within_tolerance():
    blocks = [SimpleNamespace(mlp=SimpleNamespace(act=NewGELUActivation())) for _ in range(2)]
    model = SimpleNamespace(transformer=SimpleNamespace(h=blocks))
    x = torch.linspace(-8, 8, 10_001)
    before = blocks[0].mlp.act(x)
    ca.fuse_gelu(model)
    assert all(isinstance(b.mlp.act, torch.nn.GELU) for b in blocks)
    assert ca.agreement(blocks[0].mlp.act(x), before)["within_reproduce_atol"]
