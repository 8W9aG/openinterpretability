"""Read the next streamed layer from disk while the current one runs.

A streamed model (``--memory stream``, :mod:`openinterp.harness.memory`) reads each
layer's weights from the checkpoint when a pass reaches it. The file cache may not hold
the checkpoint, and then that read waits on the disk. With :func:`install_readahead`, a
pass that enters layer ``i`` asks for layer ``i + 1``, or for layer 0 after the last. So
the disk works while the layer computes.

The byte ranges come from each safetensors file's header. On Linux the request is
``posix_fadvise(WILLNEED)``: the kernel reads the pages into the file cache in the
background. Elsewhere a thread reads them. Either way only the file cache keeps them. So
a layer that is evicted again before it runs is read again, as it was before.
"""

from __future__ import annotations

import json
import os
import struct
import threading
from collections.abc import Callable
from functools import cache

from torch import nn

Ranges = list[tuple[str, int, int]]  # (file, offset, length)


@cache
def _header(path: str) -> tuple[int, dict]:
    """A safetensors file's data offset and header: name -> {"data_offsets": [start, end]}."""
    with open(path, "rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        return 8 + size, json.loads(f.read(size))


def layer_ranges(layer: nn.Module) -> Ranges:
    """The byte ranges in the checkpoint that hold ``layer``'s streamed weights."""
    ranges: Ranges = []
    for module in layer.modules():
        hook = getattr(module, "_hf_hook", None)
        weights = getattr(hook, "weights_map", None)
        index = getattr(getattr(getattr(weights, "dataset", None), "loader", None), "index", None)
        if index is None:
            continue
        for name, _ in module.named_parameters(recurse=False):
            info = index.get(weights.prefix + name) or {}
            path = info.get("safetensors_file")
            if not path:
                continue
            start, header = _header(path)
            begin, end = header[info.get("weight_name", weights.prefix + name)]["data_offsets"]
            ranges.append((path, start + begin, end - begin))
    return _merged(ranges)


def _merged(ranges: Ranges) -> Ranges:
    """Join ranges of one file that touch, so the disk reads few long runs."""
    merged: Ranges = []
    for path, offset, length in sorted(ranges):
        if merged and merged[-1][0] == path and merged[-1][1] + merged[-1][2] >= offset:
            last = merged[-1]
            merged[-1] = (path, last[1], max(last[2], offset + length - last[1]))
        else:
            merged.append((path, offset, length))
    return merged


def request(ranges: Ranges) -> None:
    """Start reading ``ranges`` into the file cache without waiting for them."""
    if hasattr(os, "posix_fadvise"):
        for path, offset, length in ranges:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, offset, length, os.POSIX_FADV_WILLNEED)
            finally:
                os.close(fd)
        return
    threading.Thread(target=_read, args=(ranges,), daemon=True).start()


def _read(ranges: Ranges, chunk: int = 8 << 20) -> None:
    for path, offset, length in ranges:
        with open(path, "rb", buffering=0) as f:
            end = offset + length
            while offset < end:
                offset += len(os.pread(f.fileno(), min(chunk, end - offset), offset) or b"x")


def install_readahead(layers: list[nn.Module]) -> bool:
    """Make a pass that enters layer ``i`` request layer ``i + 1``; False if none stream."""
    ranges = [layer_ranges(layer) for layer in layers]
    if not any(ranges):
        return False

    def ahead(index: int, forward: Callable) -> Callable:
        following = ranges[(index + 1) % len(layers)]

        def run(*args, **kwargs):
            if following:
                request(following)
            return forward(*args, **kwargs)

        return run

    for index, layer in enumerate(layers):
        layer.forward = ahead(index, layer.forward)
    return True
