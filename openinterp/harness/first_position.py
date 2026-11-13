"""A neuron's activation at a text's first position, on every token of the vocabulary.

The first-token floor check (protocol 0.8, ``scan.has_first_token_floor``) needs it. At
the first position, attention sees only the token itself. So these values depend only on
the weights, not on the corpus. They are a pass of the whole vocabulary: 151,646 inputs
of one token for Qwen2.5. On a 50,000-token Qwen2.5-0.5B audit, the two poles took 25 s.
With the model streamed they took 123 s, because each batch reads every layer from disk.

So the raw activations are computed once for both poles, and kept on disk in
``$OPENINTERP_FIRST_POSITION_CACHE`` (default ``~/.cache/openinterp/first-position``, ``off``
to disable it). An entry is keyed by the model, its revision, the dtype, the device type,
the layer and the neuron, and the torch and transformers versions. One entry is about
600 KB. A streamed model runs as many inputs at once as its planned memory holds.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from openinterp.core.cache_dirs import cache_root
from openinterp.harness.hooks import pole_strength

FORMAT = 1
ENV = "OPENINTERP_FIRST_POSITION_CACHE"
BATCH = 8192  # one-token inputs in one pass of a model loaded whole


def vocabulary(runner) -> list[int]:
    """Every token id but the special ones."""
    tokenizer = runner.tokenizer
    try:
        size = len(tokenizer)
    except TypeError:  # a tokenizer without a length: one id per embedding row
        size = runner.model.get_input_embeddings().num_embeddings
    special = set(getattr(tokenizer, "all_special_ids", None) or [])
    return [i for i in range(size) if i not in special]


def strengths_of(runner, strengths, batch_size: int = BATCH) -> torch.Tensor:
    """``strengths`` at the first position of each token of :func:`vocabulary`."""
    vocab, out = vocabulary(runner), []
    for start in range(0, len(vocab), batch_size):
        ids = torch.tensor(vocab[start : start + batch_size], device=runner.device).unsqueeze(1)
        out.append(strengths({"input_ids": ids, "attention_mask": torch.ones_like(ids)})[:, 0])
    return torch.cat(out).float().cpu() if out else torch.zeros(0)


class NeuronReader:
    """One neuron pole's strengths, and its values at the first position, kept for both poles.

    The scan calls it as ``Strengths``. ``first_position()`` gives the vocabulary's values,
    from memory, from the disk cache, or from one pass of the vocabulary.
    """

    def __init__(self, runner, layer: int, neuron: int, pole) -> None:
        self.runner, self.layer, self.neuron, self.pole = runner, layer, neuron, pole

    def __call__(self, inputs) -> torch.Tensor:
        """The pole's strength at each token of ``inputs``."""
        return self.runner.neuron_activations(inputs, self.layer, self.neuron, self.pole)

    def first_position(self) -> torch.Tensor:
        """The pole's strength at the first position, on every token of the vocabulary."""
        return pole_strength(self._raw(), self.pole)

    def _raw(self) -> torch.Tensor:
        fields = _fields(self.runner, self.layer, self.neuron)
        key = json.dumps(fields, sort_keys=True)
        memo = self.runner.__dict__.setdefault("_first_position", {})  # both poles of one run
        if key in memo:
            return memo[key]
        path = _path(fields)
        raw = _read(path)
        if raw is None:
            raw = strengths_of(
                self.runner,
                lambda inputs: self.runner.neuron_activations(inputs, self.layer, self.neuron),
                _batch_size(self.runner),
            )
            _write(path, raw)
        memo[key] = raw
        return raw


def _batch_size(runner) -> int:
    from openinterp.harness.memory import stream_rows

    return stream_rows(runner, 1, BATCH, logits=False)


def _fields(runner, layer: int, neuron: int) -> dict:
    import transformers

    config = getattr(runner.model, "config", None)
    return {
        "format": FORMAT,
        "model": getattr(runner, "model_name", None) or getattr(config, "_name_or_path", None),
        "revision": getattr(runner, "model_revision", None),
        "dtype": str(next(runner.model.parameters()).dtype).removeprefix("torch."),
        "device": runner.device.type,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "vocabulary": len(vocabulary(runner)),
        "layer": layer,
        "neuron": neuron,
    }


def _path(fields: dict) -> Path | None:
    root = cache_root(ENV, "first-position")
    if root is None or not fields["model"]:
        return None
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:32]
    return root / f"{digest}.pt"


def _read(path: Path | None) -> torch.Tensor | None:
    if path is None or not path.exists():
        return None
    try:
        return torch.load(path, weights_only=True)
    except Exception:  # a damaged entry: compute it again
        return None


def _write(path: Path | None, raw: torch.Tensor) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(raw.clone(), tmp)
    tmp.replace(path)  # atomic: a reader sees no entry or a whole one
