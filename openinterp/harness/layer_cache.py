"""On-disk cache of a scan's whole-layer activations.

One forward pass over the corpus captures every neuron of a layer, so a scan stores
all of them. Later audits of any neuron in that layer, and per-token analysis
(``openinterp activations``), replay the stored tokens instead of re-running the model.

An entry is keyed by everything that sets its numbers: the model, its revision, the
weights' dtype, the device type, the torch and transformers versions, the layer, the
token budget per batch (which sets padding) and the exact pieces scanned (texts and
windows). Replaying an entry gives the same scan result as running the model, bit for
bit.

Each entry is a directory ``<root>/<model>/L<layer>/<key>/`` holding ``meta.json``,
``pieces.json``, ``acts.bin`` (``[rows, width]`` in the activation dtype) and per-row
``ids.npy``, ``positions.npy``, ``piece.npy`` and ``counted.npy``. The rows are every
real (non-padding) token of every piece, in scan order, and ``meta.json`` lists how many
pieces each forward pass took. Entries are written to a temporary directory and renamed
into place, so concurrent scans never see a partial one. Once the cache outgrows its
size cap, the least recently used entries go.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
from collections.abc import Iterator, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.scan import Chunk, Piece

FORMAT = 2  # 2: batches by token budget, recorded in meta.json
ENV = "OPENINTERP_SCAN_CACHE"  # cache directory, or "off"
SIZE_ENV = "OPENINTERP_SCAN_CACHE_GB"
DEFAULT_GB = 40.0

# numpy has no bfloat16: store its bits as int16 and view them back.
_BITS = {torch.bfloat16: torch.int16}


def default_root() -> Path | None:
    """``$OPENINTERP_SCAN_CACHE``, else ``$XDG_CACHE_HOME/openinterp/scans``; None if off."""
    value = os.environ.get(ENV)
    if value is not None:
        return None if value.lower() in ("", "0", "off", "false") else Path(value)
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "openinterp" / "scans"


@dataclass(frozen=True)
class Entry:
    """One cached layer scan."""

    path: Path
    meta: dict

    @property
    def size(self) -> int:
        return sum(f.stat().st_size for f in self.path.iterdir())

    @property
    def last_used(self) -> float:
        return (self.path / "meta.json").stat().st_mtime

    def pieces(self) -> list[Piece]:
        return [Piece(**p) for p in json.loads((self.path / "pieces.json").read_text())]

    def rows(self, name: str) -> np.ndarray:
        """A per-row array: ``ids``, ``positions``, ``piece`` or ``counted``."""
        return np.load(self.path / f"{name}.npy", mmap_mode="r")

    def acts(self) -> np.memmap:
        shape = (self.meta["rows"], self.meta["width"])
        return np.memmap(self.path / "acts.bin", dtype=self.meta["storage"], mode="r", shape=shape)

    def column(self, neuron_idx: int) -> torch.Tensor:
        """One neuron's activation on every row, as float32 (as the scan reads it)."""
        return self._tensor(np.ascontiguousarray(self.acts()[:, neuron_idx])).float()

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.array(array))
        dtype = getattr(torch, self.meta["dtype"])
        return tensor.view(dtype) if dtype in _BITS else tensor

    def chunks(self) -> Iterator[Chunk]:
        """The scan's batches again, as the model produced them."""
        pieces = self.pieces()
        acts = self.acts()
        ids, positions = self.rows("ids"), self.rows("positions")
        counted, piece = self.rows("counted"), self.rows("piece")
        lengths = np.bincount(piece, minlength=len(pieces)).tolist()
        start = row = 0
        for size in self.meta["batches"]:
            batch_lengths = lengths[start : start + size]
            end = row + sum(batch_lengths)
            yield Chunk(
                pieces=pieces[start : start + size],
                lengths=batch_lengths,
                ids=torch.from_numpy(np.array(ids[row:end])).long(),
                positions=torch.from_numpy(np.array(positions[row:end])).long(),
                counted=torch.from_numpy(np.array(counted[row:end])),
                acts=self._tensor(acts[row:end]),
            )
            start, row = start + size, end


class LayerCache:
    def __init__(self, root: Path, max_bytes: int | None = None) -> None:
        self.root = Path(root)
        if max_bytes is None:
            max_bytes = int(float(os.environ.get(SIZE_ENV, DEFAULT_GB)) * 1e9)
        self.max_bytes = max_bytes
        self.last: Path | None = None  # the entry the last scan replayed or stored
        self.replayed = False

    @classmethod
    def default(cls) -> LayerCache | None:
        root = default_root()
        return None if root is None else cls(root)

    # -- keys ----------------------------------------------------------------

    @staticmethod
    def fields(
        runner: NeuronHookRunner, layer: int, pieces: Sequence[Piece], batch_tokens: int
    ) -> dict | None:
        """What an entry is keyed by; None for a model with no name to key it by."""
        config = getattr(runner.model, "config", None)
        model = getattr(runner, "model_name", None) or getattr(config, "_name_or_path", None)
        if not model:
            return None
        try:
            import transformers

            transformers_version = transformers.__version__
        except ImportError:
            transformers_version = None
        digest = hashlib.sha256()
        for piece in pieces:
            digest.update(json.dumps(asdict(piece), ensure_ascii=False).encode() + b"\n")
        return {
            "format": FORMAT,
            "model": model,
            "revision": getattr(runner, "model_revision", None),
            "dtype": str(next(runner.model.parameters()).dtype).removeprefix("torch."),
            "device": runner.device.type,
            "torch": torch.__version__,
            "transformers": transformers_version,
            "layer": layer,
            "batch_tokens": batch_tokens,
            "pieces_sha256": digest.hexdigest(),
        }

    def path(self, fields: dict) -> Path:
        key = hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:20]
        return self.root / fields["model"].replace("/", "--") / f"L{fields['layer']}" / key

    # -- reading -------------------------------------------------------------

    def lookup(
        self, runner: NeuronHookRunner, layer: int, pieces: Sequence[Piece], batch_tokens: int
    ) -> Entry | None:
        fields = self.fields(runner, layer, pieces, batch_tokens)
        if fields is None:
            return None
        path = self.path(fields)
        try:
            meta = json.loads((path / "meta.json").read_text())
        except (OSError, ValueError):
            return None
        os.utime(path / "meta.json")  # least recently used goes first
        return Entry(path, meta)

    def entries(self) -> list[Entry]:
        found = []
        for meta in self.root.glob("*/L*/*/meta.json"):
            if meta.parent.name.startswith("."):
                continue  # still being written
            try:
                found.append(Entry(meta.parent, json.loads(meta.read_text())))
            except (OSError, ValueError):
                continue
        return sorted(found, key=lambda e: e.last_used)

    def cached_layers(self, model: str) -> set[int]:
        entries = self.entries()
        return {e.meta["fields"]["layer"] for e in entries if e.meta["fields"]["model"] == model}

    # -- writing -------------------------------------------------------------

    def through(
        self,
        runner: NeuronHookRunner,
        layer: int,
        pieces: Sequence[Piece],
        batch_tokens: int,
        chunks: Iterator[Chunk],
    ) -> Iterator[Chunk]:
        """The cached chunks if this scan is cached, else ``chunks``, stored as they pass."""
        entry = self.lookup(runner, layer, pieces, batch_tokens)
        self.replayed, self.last = entry is not None, entry and entry.path
        if entry is not None:
            return entry.chunks()
        fields = self.fields(runner, layer, pieces, batch_tokens)
        if fields is None:
            return chunks
        self.last = self.path(fields)
        return self._record(fields, pieces, chunks)

    def _record(self, fields: dict, pieces: Sequence[Piece], chunks: Iterator[Chunk]):
        final = self.path(fields)
        tmp = final.with_name(f".{final.name}.{os.getpid()}.{secrets.token_hex(4)}")
        tmp.mkdir(parents=True)
        rows: dict[str, list[np.ndarray]] = {"ids": [], "positions": [], "piece": [], "counted": []}
        sizes: list[int] = []
        index = 0
        width = storage = None
        try:
            with open(tmp / "acts.bin", "wb") as fh:
                for chunk in chunks:
                    acts = chunk.acts.contiguous()
                    bits = acts.view(_BITS[acts.dtype]) if acts.dtype in _BITS else acts
                    fh.write(bits.numpy().tobytes())
                    width, storage = acts.shape[1], str(bits.numpy().dtype)
                    rows["ids"].append(chunk.ids.numpy().astype(np.int32))
                    rows["positions"].append(chunk.positions.numpy().astype(np.int32))
                    rows["counted"].append(chunk.counted.numpy())
                    sizes.append(len(chunk.pieces))
                    for n in chunk.lengths:
                        rows["piece"].append(np.full(n, index, dtype=np.int32))
                        index += 1
                    yield chunk
            for name, parts in rows.items():
                np.save(tmp / f"{name}.npy", np.concatenate(parts) if parts else np.zeros(0))
            (tmp / "pieces.json").write_text(json.dumps([asdict(p) for p in pieces]))
            total = int(sum(len(p) for p in rows["ids"]))
            meta = {
                "fields": fields,
                "rows": total,
                "width": width or 0,
                "dtype": fields["dtype"],
                "storage": storage or "float32",
                "batches": sizes,
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
            with suppress(OSError):  # another scan stored it first
                tmp.rename(final)
            self.evict(keep=final)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def evict(self, keep: Path | None = None) -> list[Entry]:
        """Delete least recently used entries until the cache fits its cap."""
        entries = self.entries()
        total = sum(e.size for e in entries)
        removed = []
        for entry in entries:
            if total <= self.max_bytes:
                break
            if entry.path == keep:
                continue
            total -= entry.size
            shutil.rmtree(entry.path, ignore_errors=True)
            removed.append(entry)
        return removed

    def clear(self) -> int:
        entries = self.entries()
        for entry in entries:
            shutil.rmtree(entry.path, ignore_errors=True)
        return len(entries)
