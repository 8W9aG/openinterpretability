"""The canonical model graph's names, refusals and hooks (openinterpretability/patches, #5229)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openinterpretability.patches.graph import Graph, hook_point, on_input, on_output, parse
from openinterpretability.patches.schema import PatchError


def test_parse_splits_a_name_and_refuses_one_that_is_not_canonical():
    name = parse("blocks.3.mlp.hook_post[index=5]")
    assert (name.layer, name.part, name.select, name.index) == (3, "mlp.hook_post", "index", 5)
    with pytest.raises(PatchError, match="not a canonical name: 'blocks.1.mlp.hook_post"):
        parse("blocks.1.mlp.hook_post[unit=2]")


def test_a_model_type_without_a_graph_is_refused():
    model = nn.Linear(2, 2)
    model.config = SimpleNamespace(model_type="llama")
    with pytest.raises(PatchError, match="no canonical graph for model type 'llama'"):
        Graph(model)


def test_block_refuses_a_missing_layer_and_a_part_the_architecture_lacks(tiny_gpt2):
    graph = Graph(tiny_gpt2)
    with pytest.raises(PatchError, match="no block 2 in a 2-layer model"):
        graph.block(2, "mlp")
    with pytest.raises(PatchError, match="no block None"):
        graph.block(None, "mlp")
    with pytest.raises(PatchError, match="gpt2 has no W_gate"):
        graph.parameter("blocks.0.mlp.W_gate")


def test_hook_point_refuses_an_unknown_point_and_a_missing_layer(tiny_gpt2):
    graph = Graph(tiny_gpt2)
    with pytest.raises(PatchError, match="unknown hook point 'blocks.0.mlp.hook_mid'"):
        hook_point(graph, "blocks.0.mlp.hook_mid", lambda x: None)
    with pytest.raises(PatchError, match="no block 5 in a 2-layer model"):
        hook_point(graph, "blocks.5.hook_resid_post", lambda x: None)
    with pytest.raises(PatchError, match="no block None"):
        hook_point(graph, "hook_resid_post", lambda x: None)


class _Pair(nn.Module):
    """Returns its input and its input doubled, as attention returns output and pattern."""

    def forward(self, hidden_states):
        return hidden_states, hidden_states * 2


def test_an_output_hook_replaces_one_field_of_a_tuple():
    module = _Pair()
    handle = on_output(module, lambda pattern: pattern + 1, field=1)
    out, pattern = module(torch.ones(2))
    handle.remove()
    assert torch.equal(out, torch.ones(2)) and torch.equal(pattern, torch.full((2,), 3.0))


def test_an_input_hook_reads_and_replaces_hidden_states_passed_by_keyword():
    module = _Pair()
    seen = []
    watch = on_input(module, lambda x: seen.append(x.clone()))
    module(hidden_states=torch.ones(2))
    watch.remove()
    assert len(seen) == 1 and torch.equal(seen[0], torch.ones(2))
    handle = on_input(module, lambda x: x * 5)
    out, _ = module(hidden_states=torch.ones(2))
    handle.remove()
    assert torch.equal(out, torch.full((2,), 5.0))
