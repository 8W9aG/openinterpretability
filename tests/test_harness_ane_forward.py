import copy
import math

import torch

from openinterpretability.harness.ane import forward
from tests.conftest import VOCAB
from tests.test_ane import tiny_qwen2

__all__ = ["tiny_qwen2"]  # the fixture, used by name below

IDS = torch.randint(0, VOCAB, (2, 16), generator=torch.Generator().manual_seed(3))


def hf_layer1_activations(model):
    seen = []
    write = model.transformer.h[1].mlp.c_proj
    handle = write.register_forward_pre_hook(lambda _m, args: seen.append(args[0]))
    with torch.no_grad():
        model(IDS)
    handle.remove()
    return seen[0]


def test_a_later_gpt2_chunk_reads_the_residual_stream_of_the_first(tiny_gpt2):
    first = forward.ToLayer(copy.deepcopy(tiny_gpt2), 0, 2, 16, resid_out=True).eval()
    with torch.no_grad():
        resid = first(IDS)
        second = forward.ToLayer(copy.deepcopy(tiny_gpt2), 1, 2, 16, start=1, sample=resid)
        acts = second.eval()(resid)
    assert resid.shape == (2, 16, 32)
    assert "positions" not in forward.shape_inputs(tiny_gpt2.config, 16, start=1)
    assert torch.allclose(acts, hf_layer1_activations(tiny_gpt2), atol=1e-5, rtol=1e-4)


def test_a_projection_with_a_large_output_gets_a_smaller_scale(tiny_gpt2):
    with torch.no_grad():
        tiny_gpt2.transformer.h[0].ln_1.weight.mul_(1e4)  # c_attn's inputs become huge
    rms = float(tiny_gpt2.transformer.h[0].attn.c_attn.weight.detach().pow(2).mean().sqrt())
    net = forward.ToLayer(copy.deepcopy(tiny_gpt2), 0, 2, 16)
    c_attn = net.body.h[0].attn.c_attn
    assert isinstance(c_attn, forward.Rescaled)
    assert c_attn.scale < 2.0 ** round(math.log2(forward.WEIGHT_RMS / rms))


def test_a_qwen2_chain_with_its_inputs_by_position_matches_hf(tiny_qwen2):
    # The traced model gets its other inputs by position, in the order of names().
    first = forward.ToLayer(copy.deepcopy(tiny_qwen2), 0, 2, 16, resid_out=True).eval()
    with torch.no_grad():
        resid = first(IDS, *first.extra.values())
        second = forward.ToLayer(copy.deepcopy(tiny_qwen2), 1, 2, 16, start=1, sample=resid)
        acts = second.eval()(resid, *second.extra.values())
    assert first.names() == ["mask", "cos", "sin"]
    assert resid.shape == (2, 16, 32)
    seen = []
    write = tiny_qwen2.model.layers[1].mlp.down_proj
    handle = write.register_forward_pre_hook(lambda _m, args: seen.append(args[0]))
    with torch.no_grad():
        tiny_qwen2(IDS)
    handle.remove()
    assert torch.allclose(acts, seen[0], atol=1e-5, rtol=1e-4)
