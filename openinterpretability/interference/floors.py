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

    @property
    def gated(self) -> bool:
        """Whether any op of the patch is gated."""
        return any(op.gate for _, op in self.ops)

    @contextmanager
    def entered(self, observe: list | None = None):
        """The runner with the patch applied, which comes off again afterwards."""
        from openinterpretability.patches.apply import applied

        with applied(self.runner.model, self.ops, root=self.root, observe=observe):
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
        self._noise: dict[str, torch.Tensor] | None = None
        self._saved: dict[str, torch.Tensor] = {}

    def _prepare(self) -> None:
        """Draw the noise and copy the weights once, on first use.

        The report enters this side once per batch and once per layer. Drawing the noise
        for every weight each time took most of a run (#5230).
        """
        generator = torch.Generator().manual_seed(SEED)
        theirs = dict(self.other.model.named_parameters())
        self._noise = {}
        for name, weight in self.runner.model.named_parameters():
            change = theirs[name].detach().float().cpu() - weight.detach().float().cpu()
            self._saved[name] = weight.detach().cpu().clone()
            self._noise[name] = _noise_like(change, generator).to(weight.dtype)

    @contextmanager
    def entered(self):
        """The perturbed runner; each weight is restored from its copy afterwards."""
        if self._noise is None:
            self._prepare()
        with torch.no_grad():
            for name, weight in self.runner.model.named_parameters():
                weight.add_(self._noise[name].to(weight.device))
        try:
            yield self.runner
        finally:
            with torch.no_grad():
                for name, weight in self.runner.model.named_parameters():
                    weight.copy_(self._saved[name].to(weight.device))
