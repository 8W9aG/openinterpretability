"""Streaming mode: run a model larger than this machine's memory, slowly.

In streaming mode the decoder layers stay in the checkpoint's safetensors files on disk.
``accelerate`` reads a layer's weights when the forward pass reaches it, and frees them
after. The embeddings, the final norm and the output head stay in memory. Peak memory is
then about one layer, those modules and one batch's activations. The cost is time: every
forward pass reads its layers from disk again.

It is never the default for a model that fits. ``--memory auto`` (the default) streams
only when the weights at the chosen dtype, plus :data:`WORKING_MARGIN_BYTES`, exceed free
RAM plus free swap. ``--memory stream`` forces it, and ``--memory normal`` never streams.
The maths is the same, so a streamed record reproduces in normal mode.

A streamed scan reads its layers once per batch, so it uses larger batches
(:func:`stream_batch_tokens`) to read each layer fewer times.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import click
import torch
from torch import nn

MemoryMode = Literal["auto", "normal", "stream"]
MEMORY_MODES: tuple[MemoryMode, ...] = ("auto", "normal", "stream")

# What a scan or review holds beyond the weights, measured on CPU in float32. It is about
# 2.5–3 GB for GPT-2 and Qwen2.5-0.5B and 1.5B (docs/experiments.md, Memory).
WORKING_MARGIN_BYTES = 3 * 2**30

# A streamed scan's batch: as many tokens as its activations fit in, within these bounds.
# The floor is the normal scan's batch (harness/scan.py, BATCH_TOKENS).
STREAM_BATCH_TOKENS_MAX = 65536

# The most rows a streamed pass of single prompts runs at once (:func:`stream_rows`).
STREAM_ROWS_MAX = 64

MEMORY_ENV = "OPENINTERP_MEMORY_GB"  # the memory to plan for, in place of what is free

_requested: MemoryMode = "auto"


def request(mode: MemoryMode) -> None:
    """Set the mode that the next model load uses (the ``--memory`` option)."""
    global _requested
    _requested = mode


def requested() -> MemoryMode:
    """The mode that ``--memory`` asked for."""
    return _requested


@dataclass(frozen=True)
class MemoryPlan:
    """Whether to stream a model, and the numbers that decided it."""

    stream: bool
    need_bytes: int | None  # weights at the chosen dtype, plus the working margin
    available_bytes: int | None  # free RAM plus free swap
    reason: str


def available_bytes(device: str | torch.device | None = None) -> int | None:
    """Free memory to plan for on ``device``, or ``None`` when it can't be read.

    On a CUDA device that is the GPU's free memory. Elsewhere it is free RAM plus free
    swap. ``device`` defaults to the one the last model loaded on. ``$OPENINTERP_MEMORY_GB``
    overrides it, to run as if only that much were free: on a shared machine, or to
    test streaming on a large one.
    """
    budget = os.environ.get(MEMORY_ENV)
    if budget:
        return int(float(budget) * 2**30)
    device = torch.device(device if device is not None else _budget_device)
    try:
        if device.type == "cuda":
            return int(torch.cuda.mem_get_info(device)[0])
        import psutil

        return int(psutil.virtual_memory().available + psutil.swap_memory().free)
    except Exception:  # psutil missing, an unsupported platform, or no CUDA
        return None


_budget_device = "cpu"  # where the last model was loaded: what available_bytes plans for


def checkpoint_bytes(snapshot: Path) -> int:
    """Total size of the safetensors (else PyTorch) weight files in a model snapshot."""
    files = list(snapshot.glob("*.safetensors")) or list(snapshot.glob("*.bin"))
    return sum(f.stat().st_size for f in files)


def weights_bytes(snapshot: Path, config, dtype: torch.dtype | str) -> int:
    """The weights' size in memory at ``dtype``: the checkpoint scaled from its own dtype."""
    stored = getattr(config, "torch_dtype", None) or getattr(config, "dtype", None)
    size = checkpoint_bytes(snapshot)
    if dtype == "auto" or stored is None:
        return size
    stored = getattr(torch, stored) if isinstance(stored, str) else stored
    return size * _itemsize(dtype) // _itemsize(stored)


def plan(mode: MemoryMode, need: int | None, have: int | None) -> MemoryPlan:
    """Decide whether to stream: always for ``stream``, never for ``normal``.

    For ``auto``, stream only when ``need`` is known to exceed ``have``.
    """
    if mode == "stream":
        return MemoryPlan(True, need, have, "--memory stream")
    if mode == "normal" or need is None or have is None:
        return MemoryPlan(False, need, have, f"--memory {mode}")
    if need <= have:
        return MemoryPlan(False, need, have, "the model fits")
    reason = f"it needs about {_gb(need)} and {_gb(have)} is free (RAM and swap)"
    return MemoryPlan(True, need, have, reason)


def warn(memory: MemoryPlan, model_name: str) -> None:
    """Say on stderr that ``model_name`` streams from disk, and why."""
    click.echo(
        f"Streaming {model_name} from disk, because {memory.reason}. This is much "
        "slower: every forward pass reads its layers from disk. Pass --memory normal "
        "to load it whole.",
        err=True,
    )


def load_causal_lm(model_name: str, device: str, dtype, revision: str | None, **extra):
    """Load ``model_name`` whole on ``device``, or streamed from disk (see the module doc).

    A streamed model runs on the CPU. Its offloaded layers read the checkpoint files in
    place, so streaming writes no copy of them. A mixture-of-experts checkpoint that
    stores each expert apart is the exception. transformers stacks its experts as it
    loads, and streaming writes the stacks to :func:`offload_dir`.
    """
    from transformers import AutoModelForCausalLM

    from openinterp.harness.moe import use_openinterp_experts

    global _budget_device
    memory = _plan_load(model_name, dtype, revision, device)
    if not memory.stream:
        kwargs = {"dtype": dtype, "revision": revision, **extra}
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(device)
        use_openinterp_experts(model)
        _budget_device = device
        return model
    warn(memory, model_name)
    resident = stream_device(device)
    if resident != device:
        click.echo(f"Streaming runs on the CPU, not {device}.", err=True)
    empty = _empty_model(model_name, revision)
    device_map = stream_device_map(empty, _layers_path(empty, model_name), resident)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        revision=revision,
        device_map=device_map,
        offload_folder=str(offload_dir()),
        **extra,
    )
    keep_checkpoint_open(model)
    use_openinterp_experts(model)  # reads only the routed experts (harness/moe.py)
    _budget_device = resident
    return model


def stream_device(device: str) -> str:
    """Where a streamed model runs: ``device`` if it is CUDA, else the CPU.

    On CUDA the embeddings, the final norm and the head stay on the GPU. Each layer is
    read from disk and copied to the GPU when a pass reaches it. ``from_pretrained``
    with a device map on MPS ends in a segmentation fault (transformers 5.20), so MPS
    streams on the CPU.
    """
    return device if torch.device(device).type == "cuda" else "cpu"


class OpenCheckpoint(Mapping):
    """``accelerate``'s offloaded weights, read from safetensors files that stay open.

    ``accelerate`` opens the checkpoint file again for each tensor it reads. On
    Qwen2.5-0.5B that took about 19 ms a tensor, and a pass reads 288 tensors. So it was
    5.6 s of a 5.9 s pass over 16 tokens. With each file opened once, the pass took
    0.18 s, and 0.14 s whole. The tensors read are the same. An open file is
    memory-mapped. The pages it reads are in the file cache, which the system can free.
    They are not the process's own memory.
    """

    def __init__(self, loader) -> None:
        self.loader = loader  # accelerate's OffloadedWeightsLoader
        self.files: dict[str, object] = {}

    def __getitem__(self, key: str) -> torch.Tensor:
        info = self.loader.index.get(key)
        if key in self.loader.state_dict or not info or not info.get("safetensors_file"):
            return self.loader[key]
        path = info["safetensors_file"]
        if path not in self.files:
            from safetensors import safe_open

            self.files[path] = safe_open(path, framework="pt", device="cpu")
        tensor = self.files[path].get_tensor(info.get("weight_name", key))
        return tensor.to(getattr(torch, info["dtype"])) if "dtype" in info else tensor

    def slice(self, key: str, index: int) -> torch.Tensor:
        """Row ``index`` of the stacked tensor ``key``, read without reading the rest.

        A mixture-of-experts layer stacks its experts in one tensor. A pass reads only
        the experts its tokens are routed to (:mod:`openinterp.harness.moe`).
        """
        info = self.loader.index.get(key)
        if key in self.loader.state_dict or not info or not info.get("safetensors_file"):
            return self.loader[key][index]
        path = info["safetensors_file"]
        if path not in self.files:
            from safetensors import safe_open

            self.files[path] = safe_open(path, framework="pt", device="cpu")
        row = self.files[path].get_slice(info.get("weight_name", key))[index]
        return row.to(getattr(torch, info["dtype"])) if "dtype" in info else row

    def __iter__(self) -> Iterator[str]:
        return iter(self.loader)

    def __len__(self) -> int:
        return len(self.loader)


def keep_checkpoint_open(model: nn.Module) -> None:
    """Make every streamed module of ``model`` read through one :class:`OpenCheckpoint`."""
    from accelerate.utils import PrefixedDataset

    opened: dict[int, OpenCheckpoint] = {}
    for module in model.modules():
        weights = getattr(getattr(module, "_hf_hook", None), "weights_map", None)
        if isinstance(weights, PrefixedDataset) and hasattr(weights.dataset, "index"):
            key = id(weights.dataset)
            weights.dataset = opened.setdefault(key, OpenCheckpoint(weights.dataset))


def offload_dir() -> Path:
    """Where ``accelerate`` may write weights it can't read in place (a ``.bin`` checkpoint).

    ``$XDG_CACHE_HOME/openinterp/offload``, else ``~/.cache/openinterp/offload``. It stays
    empty for a safetensors checkpoint.
    """
    root = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(root) / "openinterp" / "offload"


def _plan_load(model_name: str, dtype, revision: str | None, device: str = "cpu") -> MemoryPlan:
    """The streaming decision for loading ``model_name`` at ``dtype`` on ``device``."""
    mode = requested()
    if mode == "normal":
        return plan(mode, None, None)
    need = None
    try:
        snapshot = _snapshot(model_name, revision)
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, revision=revision)
        need = weights_bytes(snapshot, config, dtype) + WORKING_MARGIN_BYTES
    except Exception:  # offline, or an unknown layout: don't stream unless asked
        pass
    return plan(mode, need, available_bytes(device))


def _snapshot(model_name: str, revision: str | None) -> Path:
    """The local folder that holds ``model_name``'s weights, downloading them if needed."""
    if Path(model_name).is_dir():
        return Path(model_name)
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "*.safetensors"]
    return Path(snapshot_download(model_name, revision=revision, allow_patterns=patterns))


def _empty_model(model_name: str, revision: str | None) -> nn.Module:
    """``model_name``'s architecture with every weight on the meta device (no memory)."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name, revision=revision)
    with init_empty_weights():
        return AutoModelForCausalLM.from_config(config)


def _layers_path(model: nn.Module, model_name: str) -> str:
    """The dotted path of ``model``'s decoder layers."""
    from openinterp.harness.hooks import LAYER_PATHS, _try_path

    for path in LAYER_PATHS:
        layers = _try_path(model, path)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return path
    from openinterp.core.exceptions import ArchitectureNotSupportedError

    raise ArchitectureNotSupportedError(f"no decoder layers found on {model_name}")


def stream_device_map(model: nn.Module, layers_path: str, resident: str = "cpu") -> dict[str, str]:
    """An ``accelerate`` device map with each decoder layer on disk, the rest on ``resident``.

    ``model`` may be on the meta device. ``layers_path`` names its decoder layers.
    ``accelerate`` runs the layers on the first device in the map that isn't the CPU or
    the disk, so on a GPU when ``resident`` is one.
    """
    device_map: dict[str, str] = {}

    def assign(module: nn.Module, prefix: str) -> None:
        for name, child in module.named_children():
            path = f"{prefix}{name}"
            if path == layers_path:
                device_map.update({f"{path}.{i}": "disk" for i in range(len(child))})
            elif layers_path.startswith(path + "."):
                assign(child, path + ".")
            else:
                device_map[path] = resident

    assign(model, "")
    return device_map


def stream_batch_tokens(config, dtype: torch.dtype, have: int | None, floor: int) -> int:
    """A streamed scan's batch: the tokens whose activations fit in half of ``have``.

    One token of one layer holds about 4 residual-width and 3 MLP-width vectors at once
    (the residual, attention and MLP inputs and outputs). The result is between
    ``floor`` and :data:`STREAM_BATCH_TOKENS_MAX`.
    """
    width = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)
    inner = getattr(config, "intermediate_size", None) or getattr(config, "n_inner", None)
    if not width or have is None:
        return floor
    per_token = _itemsize(dtype) * (4 * width + 3 * (inner or 4 * width))
    tokens = (have // 2) // per_token
    return int(max(floor, min(STREAM_BATCH_TOKENS_MAX, tokens)))


def stream_rows(runner, tokens_per_row: int, default: int, *, logits: bool) -> int:
    """Rows per forward pass: ``default``, or more when ``runner``'s model streams.

    Every streamed pass reads its layers from disk again. So a streamed model runs as
    many rows at once as a quarter of the planned memory holds (:func:`available_bytes`).
    A pass that returns ``logits`` holds them and one log-softmax copy: 1.2 MB a token at
    Qwen2.5's 151,936 tokens. Rows are independent, so only float rounding changes. The
    result is at least ``default`` and at most :data:`STREAM_ROWS_MAX`.
    """
    have = available_bytes()
    if not is_streamed(runner.model) or have is None:
        return default
    have //= _sharing  # passes that run at once share the memory (layer_major)
    config = runner.model.config
    dtype = next(runner.model.parameters()).dtype
    width = getattr(config, "hidden_size", None) or getattr(config, "n_embd", 1)
    inner = getattr(config, "intermediate_size", None) or getattr(config, "n_inner", None)
    per_token = _itemsize(dtype) * (4 * width + 3 * (inner or 4 * width))
    if logits:
        per_token += 2 * 4 * getattr(config, "vocab_size", 0)  # float32 logits and log-probs
    rows = (have // 4) // max(1, per_token * max(1, tokens_per_row))
    return int(max(default, min(STREAM_ROWS_MAX, rows)))


_sharing = 1  # how many passes run at once (openinterp.harness.layer_major)


@contextmanager
def shared_by(passes: int) -> Iterator[None]:
    """Plan each pass's memory as a share of the budget while ``passes`` run at once."""
    global _sharing
    previous, _sharing = _sharing, max(1, passes)
    try:
        yield
    finally:
        _sharing = previous


def loaded(module: nn.Module) -> AbstractContextManager:
    """A context in which ``module``'s own weights are real tensors, also when streamed.

    A streamed module keeps its weights on the meta device between passes. Read them
    inside this context, and copy what must outlive it. Child modules are not loaded.
    """
    if getattr(module, "_hf_hook", None) is None:
        return nullcontext()
    from accelerate.utils import align_module_device

    return align_module_device(module)


@contextmanager
def recompute_below(runner, layer: int) -> Iterator[None]:
    """In a streamed model, recompute the layers below ``layer`` in a backward pass.

    Autograd keeps the weights a layer used until the backward pass. So a gradient
    through streamed layers holds all of them at once. Inside this block each layer
    below ``layer`` keeps only its input. The backward pass runs it again, which reads
    its weights from disk again. The values are the same. Take Qwen2.5-1.5B at layer
    11, with frozen weights (``cli._load``). Integrated gradients over 1,024 tokens
    peaked at 1.8 GB with this, and at 4.5 GB without. It took 5.2 s, not 4.2 s.

    ``layer`` itself isn't recomputed, since a hook there may end the forward pass. The
    key-value cache is off, since a recomputed layer would add to it twice. A model that
    isn't streamed is left as it is.
    """
    if not is_streamed(runner.model):
        yield
        return
    from torch.utils.checkpoint import checkpoint

    def recomputed(forward):
        def run(*args, **kwargs):
            if not torch.is_grad_enabled():
                return forward(*args, **kwargs)
            return checkpoint(forward, *args, use_reentrant=False, **kwargs)

        return run

    config = runner.model.config
    use_cache, config.use_cache = getattr(config, "use_cache", None), False
    blocks = list(runner.layers)[:layer]
    forwards = [block.forward for block in blocks]
    for block, forward in zip(blocks, forwards, strict=True):
        block.forward = recomputed(forward)
    try:
        yield
    finally:
        for block, forward in zip(blocks, forwards, strict=True):
            block.forward = forward
        config.use_cache = use_cache


def is_streamed(model: nn.Module) -> bool:
    """Whether some of ``model``'s weights stay on disk between passes."""
    return any(p.device.type == "meta" for p in model.parameters())


def _itemsize(dtype: torch.dtype | str) -> int:
    dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    return torch.empty((), dtype=dtype).element_size()


def _gb(n: int) -> str:
    return f"{n / 2**30:.1f} GB"
