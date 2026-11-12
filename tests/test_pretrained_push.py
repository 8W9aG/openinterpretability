"""The helpers behind scripts/pretrained_push.py."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from pretrained_push import ConvMLP, move_stats, write_checkpoint_rows  # noqa: E402


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


def test_each_checkpoint_row_but_the_last_gets_the_move_to_the_next(tmp_path):
    push = torch.tensor([1.0, -1.0, 2.0])
    rest = torch.zeros(3)
    checkpoints = [
        ({"step": step, "layers": [{}]}, [(push, rest + step * push)]) for step in (0, 1, 2)
    ]
    out = tmp_path / "rows.jsonl"
    write_checkpoint_rows(out, iter(checkpoints))
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 1, 2]
    assert rows[0]["layers"][0]["share_push_sign_moved"] == 1.0
    assert "share_push_sign_moved" not in rows[2]["layers"][0]
