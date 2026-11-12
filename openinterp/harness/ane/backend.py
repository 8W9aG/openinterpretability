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

import os
import shutil
import tempfile
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from openinterp.harness.ane import chunks, convert
from openinterp.harness.ane.cache import (
    MAX_ERROR,
    META,
    MODEL,
    ConversionCache,
    ConversionRejected,
    fields,
)
from openinterp.harness.ane.forward import ToLayer, supported
from openinterp.harness.ane.qualified import qualified

if TYPE_CHECKING:
    from openinterp.harness.hooks import NeuronHookRunner
    from openinterp.harness.scan import Chunk, Piece

NAME = "coreml-ane"  # recorded as the scan's backend
SEQS = (128, 256, 512, 1024, 2048)
SHAPE_TOKENS = 2048  # rows × seq of each shape, as BATCH_TOKENS
PREPARE_ENV = "OPENINTERP_ANE_PREPARE_MINUTES"
# The longest that a scan waits for its models to convert and compile, before it runs on
# torch. A first compile can take minutes for each shape (Qwen2.5-0.5B: about 1-3 min).
DEFAULT_PREPARE_MINUTES = 20.0
# Peak memory of a conversion, as a multiple of the fp32 weights it converts. Measured:
# 4.2 (GPT-2 L11), 3.1 (Qwen2.5-0.5B L23) and 2.5 (Qwen2.5-1.5B L14), with a margin.
CONVERSION_OVERHEAD = 4.5


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
        self._loaded: dict[tuple[int, int, int], list] = {}
        self.paths: dict[tuple[int, int, int], list[Path]] = {}  # each pass's chunk models
        self._scratch: list[Path] = []
        self.used = False  # True once a scan ran on it
        self.noted: set[str] = set()  # reasons already shown, so each shows once
        self.broken: str | None = None  # why it may not run again in this process
        self._decided: dict[tuple, str | None] = {}

    @classmethod
    def for_model(cls, model, model_name: str, revision: str | None) -> AneScan | None:
        """An :class:`AneScan` for ``model``, or None if it can't or may not use the ANE.

        It needs coremltools, a supported architecture, and a model that passed
        qualification (``harness/ane/qualified.py``).
        """
        if not convert.available() or not supported(model) or not qualified(model_name):
            return None
        return cls(model, model_name, revision, ConversionCache.default())

    def longest(self) -> int:
        """The longest piece it can run: the largest shape, or the model's context."""
        cfg = self.model.config
        context = getattr(cfg, "n_positions", None) or getattr(cfg, "max_position_embeddings", None)
        return min(SEQS[-1], context or SEQS[-1])

    def prepare(self, lengths: Sequence[int], layer: int, batch_tokens: int) -> str | None:
        """Make ready every model a scan of pieces of these token lengths needs.

        Return why the ANE can't run that scan, or None if it can. The answer is kept,
        so every part of one scan (its cache key, its pass, its rescoring) agrees.
        """
        if self.broken:
            return self.broken
        key = (layer, batch_tokens, tuple(lengths))
        if key not in self._decided:
            self._decided[key] = self._prepare(lengths, layer, batch_tokens)
        return self._decided[key]

    def _prepare(self, lengths: Sequence[int], layer: int, batch_tokens: int) -> str | None:
        shapes = needed_shapes(lengths, batch_tokens, self.longest())
        if shapes is None:
            too_long = max(lengths, default=0)
            return f"a piece has {too_long} tokens, more than the ANE's {self.longest()}"
        missing = any(not self._cached(layer, *shape) for shape in shapes)
        if missing and (short := self._memory_short(layer)):
            return short
        return self._ready_in_time(layer, sorted(shapes))

    def _ready_in_time(self, layer: int, shapes: list) -> str | None:
        """Load (and if needed convert) ``shapes`` within the time limit; else say why not.

        The work runs on a background thread, because a call into Core ML can't be
        stopped. If it takes longer than the limit, the ANE is not used again in this
        process, and the scan runs on torch.
        """
        done, failed = threading.Event(), []

        def work() -> None:
            try:
                for shape in shapes:
                    self.compiled(layer, *shape)
            except Exception as exc:  # any failure to convert: the scan runs on torch
                failed.append(exc)
            finally:
                done.set()

        threading.Thread(target=work, daemon=True, name="ane-prepare").start()
        minutes = prepare_minutes()
        if not done.wait(timeout=minutes * 60):
            self.broken = (
                f"converting and compiling for the ANE took more than {minutes:g} minutes "
                f"(${PREPARE_ENV}); a later run tries again"
            )
            return self.broken
        if failed:
            return f"its conversion failed ({type(failed[0]).__name__}: {failed[0]})"
        return None

    def _cached(self, layer: int, rows: int, seq: int) -> bool:
        if (layer, rows, seq) in self._loaded:
            return True
        return all(self._entry(link, rows, seq) for link in chunks.chain(self.model, layer))

    def _entry(self, link: tuple[int, int, bool], rows: int, seq: int) -> Path | None:
        """The cached compiled model of one chunk, or None if the cache has none."""
        if self.cache is None:
            return None
        start, last, final = link
        key_fields = fields(self.model_name, self.revision, last, rows, seq, start, not final)
        entry = self.cache.path(key_fields)
        return entry / MODEL if (entry / META).exists() else None

    def _memory_short(self, layer: int) -> str | None:
        """Why converting the chunks of a pass to ``layer`` would not fit in memory, or None.

        Chunks are converted one at a time, so the largest chunk sets the need.
        """
        from openinterp.harness.memory import available_bytes

        links = chunks.chain(self.model, layer)
        largest = max(chunks.chunk_bytes(self.model, start, last) for start, last, _ in links)
        need = int(largest * CONVERSION_OVERHEAD)
        have = available_bytes("cpu")
        if have is None or need <= have:
            return None
        return f"converting it needs about {need / 2**30:.1f} GB and {have / 2**30:.1f} GB is free"

    def chunks(
        self, runner: NeuronHookRunner, pieces: Sequence[Piece], layer: int, batch_tokens: int
    ) -> Iterator[Chunk]:
        """``layer``'s activations over ``pieces``, batch by batch, from the ANE."""
        from openinterp.harness.scan import Chunk, batches

        for batch in batches(runner, pieces, batch_tokens):
            acts = self.batch_acts(batch, layer)
            mask = batch.mask
            yield Chunk(
                pieces=batch.pieces,
                lengths=mask.sum(1).tolist(),
                ids=batch.ids[mask],
                positions=batch.positions[mask],
                counted=batch.counted[mask],
                acts=acts[mask],
            )

    def batch_acts(self, batch, layer: int) -> torch.Tensor:
        """``layer``'s activations on one scan batch, ``[pieces, width, d_mlp]`` in fp32.

        The batch goes in the smallest shape that holds it, in parts of that shape's rows.
        Each part goes through the chain of chunks (``chunks.py``).
        """
        width = batch.ids.shape[1]
        rows, seq = shape_for(width, self.longest()) or (0, 0)
        if not rows:
            raise ValueError(f"no ANE shape holds {width} tokens")
        from openinterp.harness.scan_backend import AneError

        try:
            models = self.compiled(layer, rows, seq)
            parts = []
            for start in range(0, len(batch.pieces), rows):
                x = torch.zeros((rows, seq), dtype=torch.long)
                part = batch.ids[start : start + rows]
                x[: len(part), :width] = part
                for model in models:
                    x = convert.predict(model, x)
                parts.append(x[: len(part), :width])
        except Exception as exc:  # Core ML's errors have no common type
            raise AneError(f"{type(exc).__name__}: {exc}") from exc
        return torch.cat(parts)

    def compiled(self, layer: int, rows: int, seq: int) -> list:
        """The loaded Core ML models, chunk by chunk, of a pass to ``layer`` at ``(rows, seq)``."""
        key = (layer, rows, seq)
        if key not in self._loaded:
            links = chunks.chain(self.model, layer)
            paths = [self._entry(link, rows, seq) for link in links]
            if not all(paths):
                paths = self._build(links, rows, seq)
            self.paths[key] = paths
            self._loaded[key] = [convert.load(path) for path in paths]
        return self._loaded[key]

    def _build(self, links: list, rows: int, seq: int) -> list[Path]:
        """Convert and check each chunk that the cache doesn't hold, in order.

        Each chunk's fp32 output is the next chunk's input, both to calibrate its scales
        and to check it. Only one chunk's weights are in memory at a time.
        """
        vocab = int(self.model.config.vocab_size)
        x = torch.randint(0, vocab, (rows, seq), generator=torch.Generator().manual_seed(2))
        paths = []
        for start, last, final in links:
            source = chunks.part(self.model, self.model_name, self.revision, start, last)
            net = ToLayer(source, last, rows, seq, start, not final, sample=x).eval()
            with torch.no_grad():
                ref = net(x)
            key_fields = fields(self.model_name, self.revision, last, rows, seq, start, not final)
            paths.append(self._get(key_fields, net, x, ref))
            x = ref
            del source, net
        return paths

    def _get(self, key_fields: dict, net: ToLayer, x: torch.Tensor, ref: torch.Tensor) -> Path:
        """One chunk's compiled model, from the cache or built now, and checked if built."""

        def build(dest: Path) -> None:
            convert.compile_to(net, x, dest)

        def check(path: Path) -> float:
            got = convert.predict(convert.load(path), x)
            return float((ref - got).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt())

        if self.cache is None:
            return self._uncached(build, check)
        return self.cache.get(key_fields, build, check)

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


def needed_shapes(lengths: Sequence[int], batch_tokens: int, longest: int) -> set | None:
    """The ``(rows, seq)`` shapes that a scan's batches need, or None if one has none.

    The batches are those of ``batches`` in ``harness/scan.py``: each is as wide as its
    longest piece.
    """
    from openinterp.harness.scan import plan_batches

    shapes, start = set(), 0
    for size in plan_batches(list(lengths), batch_tokens):
        shape = shape_for(max(lengths[start : start + size]), longest)
        if shape is None:
            return None
        shapes.add(shape)
        start += size
    return shapes


def prepare_minutes() -> float:
    """``$OPENINTERP_ANE_PREPARE_MINUTES``, else :data:`DEFAULT_PREPARE_MINUTES`."""
    return float(os.environ.get(PREPARE_ENV) or DEFAULT_PREPARE_MINUTES)
