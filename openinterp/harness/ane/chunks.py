"""Splitting a model into chunks of layers, each its own Core ML model.

The ANE loses precision on a whole model when the model is large. On Qwen2.5-1.5B a
pass to layer 16 (1.9 GB of fp16 weights) was as precise as fp16 on the GPU. A pass to
layer 20 (2.3 GB) had 3 times the error. The same pass as two models, layers 0-9 and
10-20, was as precise as the GPU again (docs/analyses/ane-qualification.md). So a pass
is a chain of chunks, each with at most :data:`CHUNK_BYTES` of fp16 weights. A chunk
reads token ids (the first) or the residual stream, and writes the residual stream or,
the last, the target layer's MLP activations.

The chunk boundaries depend only on the model, not on the target layer. So every chunk
before the last is the same for each target layer after it, and the cache keeps it once.

A chunk gets its weights in one of two ways. A model loaded whole gives a copy of the
chunk's modules only. A model that streams (``--memory stream``) gives nothing: the
chunk's weights are read from the checkpoint files. So converting a streamed model never
holds more than one chunk in memory.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch import nn

CHUNK_BYTES = int(1.6 * 2**30)  # fp16 weights in one Core ML model; 1.9 GB was still precise


def parts(model: nn.Module) -> tuple[nn.Module, nn.ModuleList, list[str]]:
    """The model's body, its decoder layers, and the names of its embedding modules."""
    gpt2 = model.config.model_type == "gpt2"
    body = model.transformer if gpt2 else model.model
    return body, (body.h if gpt2 else body.layers), (["wte", "wpe"] if gpt2 else ["embed_tokens"])


def _numel(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def boundaries(model: nn.Module) -> list[int]:
    """The first layer of each chunk, and the number of layers at the end.

    Chunk 0 holds the embeddings. Each chunk holds as many whole layers as fit in
    :data:`CHUNK_BYTES` of fp16 weights, and at least one.
    """
    body, blocks, embeds = parts(model)
    size = 2 * sum(_numel(getattr(body, name)) for name in embeds)
    starts = [0]
    for i, block in enumerate(blocks):
        layer = 2 * _numel(block)
        if size + layer > CHUNK_BYTES and i > starts[-1]:
            starts.append(i)
            size = 0
        size += layer
    return [*starts, len(blocks)]


def chain(model: nn.Module, layer: int) -> list[tuple[int, int, bool]]:
    """The chunks of a pass to ``layer``: ``(start, last layer, is the final chunk)``.

    Each chunk but the final one ends at a boundary and writes the residual stream. The
    final chunk runs from its boundary to ``layer`` and writes ``layer``'s activations.
    """
    starts = [b for b in boundaries(model)[:-1] if b <= layer]
    links = [(a, b - 1, False) for a, b in zip(starts, starts[1:], strict=False)]
    return [*links, (starts[-1], layer, True)]


def chunk_bytes(model: nn.Module, start: int, last: int) -> int:
    """fp32 bytes of a chunk's weights: its layers, and the embeddings for chunk 0."""
    body, blocks, embeds = parts(model)
    n = sum(_numel(blocks[i]) for i in range(start, last + 1))
    if start == 0:
        n += sum(_numel(getattr(body, name)) for name in embeds)
    return 4 * n


def part(model: nn.Module, name: str, revision: str | None, start: int, last: int) -> nn.Module:
    """A copy of ``model`` with real fp32 weights only in layers ``start..last``.

    It also has the embeddings when ``start`` is 0. Other modules are left empty. The
    weights come from memory, or from the checkpoint if ``model`` streams.
    """
    from openinterp.harness.memory import is_streamed

    if is_streamed(model):
        return _from_checkpoint(model, name, revision, start, last)
    return _from_memory(model, start, last).to("cpu", torch.float32)


def _from_memory(model: nn.Module, start: int, last: int) -> nn.Module:
    """A deep copy of ``model`` that skips every module outside the chunk."""
    body, blocks, embeds = parts(model)
    name = "h" if model.config.model_type == "gpt2" else "layers"
    keep = [blocks[i] if start <= i <= last else nn.Identity() for i in range(last + 1)]
    saved_embeds = {e: getattr(body, e) for e in embeds}
    head = model.get_output_embeddings()
    setattr(body, name, nn.ModuleList(keep))
    if start > 0:
        for e in embeds:
            setattr(body, e, nn.Identity())
    model.set_output_embeddings(nn.Identity())
    try:
        return copy.deepcopy(model)
    finally:
        setattr(body, name, blocks)
        for e, module in saved_embeds.items():
            setattr(body, e, module)
        model.set_output_embeddings(head)


def _from_checkpoint(model, name: str, revision: str | None, start: int, last: int):
    """An empty copy of ``model``'s architecture, with the chunk's weights from the checkpoint."""
    from safetensors import safe_open

    from openinterp.harness.memory import _empty_model, _snapshot

    snapshot = _snapshot(name, revision)
    skeleton = _empty_model(name, revision)
    body, blocks, embeds = parts(skeleton)
    wanted = [(f"{n}", getattr(body, n)) for n in embeds] if start == 0 else []
    wanted += [(f"{_layers_name(skeleton)}.{i}", blocks[i]) for i in range(start, last + 1)]
    index = _tensor_files(snapshot)
    opened: dict[Path, object] = {}
    for prefix, module in wanted:
        module.to_empty(device="cpu")
        for key, tensor in module.state_dict().items():
            stored = _find(index, prefix, key, skeleton.base_model_prefix)
            path = index[stored]
            if path not in opened:
                opened[path] = safe_open(str(path), framework="pt", device="cpu")
            tensor.copy_(opened[path].get_tensor(stored).to(torch.float32))
    return skeleton.float()


def _layers_name(model: nn.Module) -> str:
    return "h" if model.config.model_type == "gpt2" else "layers"


def _tensor_files(snapshot: Path) -> dict[str, Path]:
    """Each tensor name in the checkpoint, and the safetensors file that holds it."""
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        return {k: snapshot / v for k, v in weight_map.items()}
    from safetensors import safe_open

    found = {}
    for path in snapshot.glob("*.safetensors"):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            found.update({k: path for k in handle.keys()})  # noqa: SIM118 (safe_open has no iter)
    return found


def _find(index: dict[str, Path], prefix: str, key: str, base: str) -> str:
    """The checkpoint's name for a body module's tensor. A checkpoint may leave out ``base``."""
    for name in (f"{base}.{prefix}.{key}", f"{prefix}.{key}"):
        if name in index:
            return name
    raise KeyError(f"{prefix}.{key} is not in the checkpoint")
