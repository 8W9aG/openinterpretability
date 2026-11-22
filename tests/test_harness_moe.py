"""``experts_forward`` on a hand-built experts module: placeholders, norms and nested hooks."""

import torch
from torch import nn

from openinterpretability.harness.moe import expert_hook, experts_forward

D_MODEL, D_FF, N_EXPERTS = 3, 4, 2


class Experts(nn.Module):
    """The attributes ``use_experts_implementation`` gives an experts module, no gate or bias."""

    has_gate = False
    has_bias = False
    is_transposed = False
    has_post_expert_norm = True
    num_experts = N_EXPERTS

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.up_proj = nn.Parameter(torch.randn(N_EXPERTS, D_FF, D_MODEL, generator=g))
        self.down_proj = nn.Parameter(torch.randn(N_EXPERTS, D_MODEL, D_FF, generator=g))
        self.act_fn = torch.relu

    def post_expert_norm(self, y):
        return 2 * y


def _expert(experts, e, x):
    with torch.no_grad():
        return experts.post_expert_norm(
            torch.relu(x @ experts.up_proj[e].T) @ experts.down_proj[e].T
        )


def test_placeholder_experts_are_skipped_and_outputs_pass_the_post_expert_norm():
    experts = Experts()
    hidden = torch.randn(2, D_MODEL, generator=torch.Generator().manual_seed(1))
    index = torch.tensor([[0, N_EXPERTS], [1, N_EXPERTS]])  # N_EXPERTS is a placeholder
    weights = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
    with torch.no_grad():
        out = experts_forward(experts, hidden, index, weights)
    expected = torch.stack(
        [0.7 * _expert(experts, 0, hidden[0]), 0.4 * _expert(experts, 1, hidden[1])]
    )
    torch.testing.assert_close(out, expected)


def test_an_inner_hook_gives_way_to_the_outer_one_when_it_ends():
    experts = Experts()
    hidden = torch.ones(1, D_MODEL)
    index, weights = torch.tensor([[0]]), torch.tensor([[1.0]])
    calls = []

    def hook(name):
        def fn(act, e, tokens, w):
            calls.append(name)
            return act

        return fn

    with torch.no_grad(), expert_hook(experts, hook("outer")):
        with expert_hook(experts, hook("inner")):
            experts_forward(experts, hidden, index, weights)
        experts_forward(experts, hidden, index, weights)
    with torch.no_grad():
        experts_forward(experts, hidden, index, weights)  # no hook is left
    assert calls == ["inner", "outer"]
