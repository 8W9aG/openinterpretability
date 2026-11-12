"""The MLP swap behind scripts/pretrained_push.py."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from pretrained_push import ConvMLP, move_stats  # noqa: E402


def test_conv_mlp_gives_the_gpt_neox_mlps_output():
    from transformers import GPTNeoXConfig
    from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXMLP

    config = GPTNeoXConfig(
        hidden_size=16, intermediate_size=64, num_attention_heads=2, hidden_act="gelu"
    )
    torch.manual_seed(0)
    mlp = GPTNeoXMLP(config)
    x = torch.randn(2, 5, 16)
    assert ConvMLP(mlp)(x).allclose(mlp(x), atol=1e-5)


def test_move_stats_compare_the_push_with_the_move_of_each_rest():
    push = torch.tensor([1.0, -1.0, 2.0, -2.0, 0.5])
    before = torch.zeros(5)
    stats = move_stats(push, before, before + 0.5 * push)
    assert stats["push_move_correlation"] == pytest.approx(1.0)
    assert stats["share_push_sign_moved"] == 1.0
    assert stats["median_moved_to_next"] == 0.25
