"""A scan's forward pass on Apple's Neural Engine (ANE), through Core ML.

:class:`AneScan` gives the same :class:`~openinterp.harness.scan.Chunk` objects as the
torch pass. It reads the same batches (``batches`` in ``harness/scan.py``), so the
tokens, padding and counted tokens are the same. The ANE needs input shapes that do not
change. So each batch goes in the smallest of :data:`SEQS` tokens that holds its longest
piece, with ``2,048 / seq`` rows. With causal attention, padding on the right does not
change the real tokens' activations. If a piece is longer than the largest shape, the
scan runs on torch.

The ANE computes in fp16, so its activations are about 1% from fp32. The scan therefore
measures its top candidates again on the torch device (``TopExamples.rescore`` in
``harness/scan.py``). Then the numbers that ``openinterp reproduce`` checks are exact.
"""

from __future__ import annotations

import copy
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from openinterp.harness.ane import convert
from openinterp.harness.ane.cache import MAX_ERROR, ConversionCache, ConversionRejected, fields
from openinterp.harness.ane.forward import ToLayer, supported

if TYPE_CHECKING:
    from openinterp.harness.hooks import NeuronHookRunner
    from openinterp.harness.scan import Chunk, Piece

NAME = "coreml-ane"  # recorded as the scan's backend
SEQS = (128, 256, 512, 1024, 2048)
SHAPE_TOKENS = 2048  # rows × seq of each shape, as BATCH_TOKENS


def shape_for(length: int, longest: int) -> tuple[int, int] | None:
    """``(rows, seq)`` of the smallest shape that holds ``length`` tokens, or None."""
    for seq in SEQS:
        if length <= seq <= longest:
            return SHAPE_TOKENS // seq, seq
    return None


class AneScan:
    """Runs a scan's forward pass on the ANE, for one model."""

    name = NAME

    def __init__(self, model, model_name: str, revision: str | None, cache) -> None:
        self.model = model
        self.model_name = model_name
        self.revision = revision
        self.cache: ConversionCache | None = cache
        self._loaded: dict[tuple[int, int, int], object] = {}
        self._scratch: list[Path] = []
        self.used = False  # True once a scan ran on it

    @classmethod
    def for_model(cls, model, model_name: str, revision: str | None) -> AneScan | None:
        """An :class:`AneScan` for ``model``, or None if Core ML or the architecture is missing."""
        if not convert.available() or not supported(model):
            return None
        return cls(model, model_name, revision, ConversionCache.default())

    def longest(self) -> int:
        """The longest piece it can run: the largest shape, or the model's context."""
        cfg = self.model.config
        context = getattr(cfg, "n_positions", None) or getattr(cfg, "max_position_embeddings", None)
        return min(SEQS[-1], context or SEQS[-1])

    def why_not(self, lengths: Sequence[int]) -> str | None:
        """Why it can't run pieces of these token lengths, or None if it can."""
        too_long = max(lengths, default=0)
        if too_long > self.longest():
            return f"a piece has {too_long} tokens, more than the ANE's {self.longest()}"
        return None

    def chunks(
        self, runner: NeuronHookRunner, pieces: Sequence[Piece], layer: int, batch_tokens: int
    ) -> Iterator[Chunk]:
        """``layer``'s activations over ``pieces``, batch by batch, from the ANE."""
        from openinterp.harness.scan import Chunk, batches

        for batch in batches(runner, pieces, batch_tokens):
            width = batch.ids.shape[1]
            rows, seq = shape_for(width, self.longest()) or (0, 0)
            if not rows:
                raise ValueError(f"no ANE shape holds {width} tokens")
            model = self.compiled(layer, rows, seq)
            for start in range(0, len(batch.pieces), rows):
                part = slice(start, start + rows)
                ids = torch.zeros((rows, seq), dtype=torch.long)
                n = len(batch.pieces[part])
                ids[:n, :width] = batch.ids[part]
                acts = convert.predict(model, ids)[:n, :width]
                mask = batch.mask[part]
                yield Chunk(
                    pieces=batch.pieces[part],
                    lengths=mask.sum(1).tolist(),
                    ids=batch.ids[part][mask],
                    positions=batch.positions[part][mask],
                    counted=batch.counted[part][mask],
                    acts=acts[mask],
                )

    def compiled(self, layer: int, rows: int, seq: int):
        """The loaded Core ML model for ``layer`` and shape ``(rows, seq)``."""
        key = (layer, rows, seq)
        if key not in self._loaded:
            build, check = self._builder(layer, rows, seq)
            if self.cache is None:
                path = self._uncached(build, check)
            else:
                key_fields = fields(self.model_name, self.revision, layer, rows, seq)
                path = self.cache.get(key_fields, build, check)
            self._loaded[key] = convert.load(path)
        return self._loaded[key]

    def _builder(self, layer: int, rows: int, seq: int):
        """``build`` and ``check`` for :meth:`ConversionCache.get`."""
        state: dict[str, torch.Tensor] = {}

        def build(dest: Path) -> None:
            source = copy.deepcopy(self.model).to("cpu", torch.float32)
            net = ToLayer(source, layer, rows, seq).eval()
            generator = torch.Generator().manual_seed(2)
            ids = torch.randint(0, net_vocab(source), (rows, seq), generator=generator)
            with torch.no_grad():
                state["ref"], state["ids"] = net(ids), ids
            convert.compile_to(net, rows, seq, dest)

        def check(path: Path) -> float:
            got = convert.predict(convert.load(path), state["ids"])
            ref = state["ref"]
            return float((ref - got).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())

        return build, check

    def _uncached(self, build, check) -> Path:
        """Build into a temporary directory, for a run with the cache turned off."""
        work = Path(tempfile.mkdtemp(prefix="openinterp-ane-"))
        self._scratch.append(work)
        path = work / "model.mlmodelc"
        build(path)
        error = check(path)
        if not error <= MAX_ERROR:
            raise ConversionRejected(f"relative error {error:.4f} is more than {MAX_ERROR}")
        return path

    def close(self) -> None:
        """Delete the temporary models of a run with the cache turned off."""
        for work in self._scratch:
            shutil.rmtree(work, ignore_errors=True)
        self._scratch.clear()


def net_vocab(model) -> int:
    """The model's vocabulary size."""
    return int(model.config.vocab_size)
