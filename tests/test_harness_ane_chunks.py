"""Chunks of a model for the ANE: their sizes, and weights read from sharded checkpoints."""

from pathlib import Path

import pytest
import torch

from openinterpretability.harness import memory
from openinterpretability.harness.ane import chunks
from tests.conftest import VOCAB


@pytest.fixture
def qwen2():
    """A random two-layer Qwen2, as ``tests/test_ane.py`` builds it."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB,
                         max_position_embeddings=64)  # fmt: skip
    return Qwen2ForCausalLM(config).eval()


def _numel(module):
    return sum(p.numel() for p in module.parameters())


def test_only_chunk_0_counts_the_embeddings(qwen2):
    layer = _numel(qwen2.model.layers[1])
    embeds = _numel(qwen2.model.embed_tokens)
    assert chunks.chunk_bytes(qwen2, 1, 1) == 4 * layer
    assert chunks.chunk_bytes(qwen2, 0, 1) == 4 * (layer + _numel(qwen2.model.layers[0]) + embeds)


def test_a_streamed_model_reads_its_chunk_from_a_sharded_checkpoint(qwen2, tmp_path):
    qwen2.save_pretrained(tmp_path, safe_serialization=True, max_shard_size="20KB")
    assert (tmp_path / "model.safetensors.index.json").exists()
    streamed = memory._empty_model(str(tmp_path), None)  # every weight on the meta device
    part = chunks.part(streamed, str(tmp_path), None, 1, 1)
    for name, want in qwen2.model.layers[1].state_dict().items():
        assert torch.equal(part.model.layers[1].state_dict()[name], want.float()), name
    assert part.model.layers[0].self_attn.q_proj.weight.is_meta


def test_a_checkpoint_name_without_the_base_prefix_is_found():
    index = {"layers.1.mlp.weight": Path("a.safetensors")}
    assert chunks._find(index, "layers.1", "mlp.weight", "model") == "layers.1.mlp.weight"


def test_a_tensor_missing_from_the_checkpoint_is_an_error():
    with pytest.raises(KeyError, match="layers.1.mlp.weight is not in the checkpoint"):
        chunks._find({}, "layers.1", "mlp.weight", "model")


def test_chunk_0_keeps_the_embeddings_and_drops_the_head(qwen2):
    part = chunks.part(qwen2, "tiny", None, 0, 0)
    assert isinstance(part.model.embed_tokens, torch.nn.Embedding)
    assert torch.equal(part.model.embed_tokens.weight, qwen2.model.embed_tokens.weight)
    assert len(part.model.layers) == 1 and isinstance(part.lm_head, torch.nn.Identity)
    assert len(qwen2.model.layers) == 2 and isinstance(qwen2.lm_head, torch.nn.Linear)
