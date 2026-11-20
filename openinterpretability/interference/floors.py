"""The two sides of a comparison, and the matched random changes of the noise floor (#5230).

A *side* is a runner in one state: the base model, the changed model, or a floor. The
report reads both sides of one comparison batch by batch, so a patch can be applied and
removed on the same runner between them.

- :class:`Plain` is a runner as it is: the base model, or another model of a pair.
- :class:`Patched` is a runner with a patch chain applied (#5229).
- :func:`random_patch` replaces every tensor in a patch's ops with Gaussian noise of the
  same shape and norm. It is the matched random change of a patch.
- :class:`Perturbed` adds Gaussian noise to each weight of the base. The noise is as large
  as that weight's change in the other model of the pair. The weights are restored
  afterwards. It is the matched random change of a model pair.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import torch

SEED = 5230  # the random changes' seed, so a floor can be run again


class Plain:
    """A runner as it is."""

    def __init__(self, runner, name: str):
        self.runner, self.name = runner, name

    @contextmanager
    def entered(self):
        """The runner, unchanged."""
        yield self.runner


class Patched:
    """A runner with ``ops`` applied (``patches.history.replay``)."""

    def __init__(self, runner, ops: Sequence, name: str, root: Path | None = None):
        self.runner, self.ops, self.name, self.root = runner, list(ops), name, root

    def forced_open(self) -> Patched:
        """This patch with every gate forced open: the worst case, read without a forward."""
        ops = [(ref, op.model_copy(update={"params": {**op.params, "force": "open"}})
                if op.type == "gate" else op) for ref, op in self.ops]  # fmt: skip
        return Patched(self.runner, ops, self.name, self.root)

    @contextmanager
    def entered(self):
        """The runner with the patch applied, which comes off again afterwards."""
        from openinterpretability.patches.apply import applied

        with applied(self.runner.model, self.ops, root=self.root):
            yield self.runner


def _noise_like(tensor: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Gaussian noise with the shape and the Frobenius norm of ``tensor``."""
    noise = torch.randn(tensor.shape, generator=generator, dtype=torch.float32)
    norm = float(noise.norm())
    return noise * (float(tensor.float().norm()) / norm) if norm > 0 else noise


def _randomised(value, generator: torch.Generator, root: Path):
    """A parameter tree with each tensor replaced by noise of its shape and norm."""
    from openinterpretability.patches.tensors import is_tensor_ref, load, put_tensor

    if is_tensor_ref(value):
        return put_tensor(_noise_like(load(value), generator), root)
    if isinstance(value, dict):
        return {k: _randomised(v, generator, root) for k, v in value.items()}
    return value


def random_patch(ops: Sequence, root: Path | None = None) -> tuple[list, Path]:
    """``ops`` with every stored tensor replaced by noise of the same shape and norm.

    Gates keep their parameters, so the noise acts where the patch would. Return the ops
    and the store that holds the noise.
    """
    root = Path(root or tempfile.mkdtemp(prefix="interference-floor-"))
    generator = torch.Generator().manual_seed(SEED)
    out = []
    for ref, op in ops:
        params = op.params if op.type == "gate" else _randomised(op.params, generator, root)
        out.append((ref, op.model_copy(update={"params": params})))
    return out, root


class Perturbed:
    """The base runner with each weight moved by noise as large as its change in ``other``."""

    def __init__(self, runner, other, name: str):
        self.runner, self.other, self.name = runner, other, name

    @contextmanager
    def entered(self):
        """The perturbed runner; each weight is restored from a copy afterwards."""
        generator = torch.Generator().manual_seed(SEED)
        theirs = dict(self.other.model.named_parameters())
        saved = {}
        with torch.no_grad():
            for name, weight in self.runner.model.named_parameters():
                change = theirs[name].detach().float().cpu() - weight.detach().float().cpu()
                saved[name] = weight.detach().cpu().clone()
                weight.add_(_noise_like(change, generator).to(weight.device, weight.dtype))
        try:
            yield self.runner
        finally:
            with torch.no_grad():
                for name, weight in self.runner.model.named_parameters():
                    weight.copy_(saved[name].to(weight.device))
