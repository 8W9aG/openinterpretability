"""The tensors that patch operations hold, in a content-addressed store (issue #5229).

A parameter that holds a tensor is written as ``{"tensor_sha256": "<hash>"}``. The blob is
the tensor's safetensors bytes, kept beside the record evidence (``core/evidence.py``) under
``tensors/<sha[:2]>/<sha>.safetensors``. Reading checks the hash, so a signed record vouches
for its tensors as it does for its evidence. A small list of numbers may stay inline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

KEY = "tensor_sha256"


def store_root(root: Path | None = None) -> Path:
    """The tensor store: ``root``, else ``tensors/`` in the evidence store."""
    from openinterpretability.core.evidence import store_dir

    return Path(root) if root is not None else store_dir() / "tensors"


def _path(sha: str, root: Path | None) -> Path:
    return store_root(root) / sha[:2] / f"{sha}.safetensors"


def put_tensor(tensor: torch.Tensor, root: Path | None = None) -> dict[str, str]:
    """Store ``tensor`` and return its reference."""
    from safetensors.torch import save

    data = save({"t": tensor.detach().contiguous().cpu()})
    sha = hashlib.sha256(data).hexdigest()
    path = _path(sha, root)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return {KEY: sha}


def is_tensor_ref(value: Any) -> bool:
    """Whether ``value`` is a reference that :func:`put_tensor` wrote."""
    return isinstance(value, dict) and set(value) == {KEY}


def load(value: Any, root: Path | None = None) -> torch.Tensor:
    """The tensor that ``value`` names: a reference, a nested list of numbers or a number."""
    from safetensors.torch import load as from_bytes

    if is_tensor_ref(value):
        sha = value[KEY]
        data = _path(sha, root).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha:
            raise ValueError(f"tensor blob {sha[:12]} does not match its hash")
        return from_bytes(data)["t"].float()
    return torch.as_tensor(value, dtype=torch.float32)


def refs_in(value: Any) -> list[str]:
    """Every tensor hash in a parameter tree, for publishing the blobs with the record."""
    if is_tensor_ref(value):
        return [value[KEY]]
    if isinstance(value, dict):
        return [sha for v in value.values() for sha in refs_in(v)]
    if isinstance(value, list):
        return [sha for v in value for sha in refs_in(v)]
    return []
