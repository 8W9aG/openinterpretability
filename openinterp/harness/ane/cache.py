"""On-disk cache of compiled Core ML models, so a machine converts each model only once.

Converting a model to one layer takes 5–50 s. The ANE also compiles a model when it first
loads it, which takes 10–35 s. macOS keeps that compile, keyed by the model's path. A
later load from the same path takes 0.2 s, but a copy at a new path compiles again. So
an entry is built in a temporary directory and renamed to its final path. Then it is
loaded there once, which warms the ANE's compile, and checked (:data:`MAX_ERROR`).

An entry is keyed by everything that changes the compiled model:

- the model, its revision, the layer and the input shape;
- a hash of the source of the conversion code (``forward.py`` and ``convert.py``),
  so any change to that code makes new entries;
- the coremltools and torch versions;
- the macOS version and the chip, because the ANE's compiler comes with macOS.

An entry is ``<root>/<model>/L<layer>/<key>/`` with ``model.mlmodelc`` and
``meta.json``. It is complete only when ``meta.json`` exists, which is written last.
A lock per key stops two processes from building one entry at the same time. Once the
cache is larger than its cap, the least recently used entries go.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import secrets
import shutil
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path

import torch

from openinterp.core.cache_dirs import cache_root
from openinterp.harness.ane import convert

FORMAT = 1
ENV = "OPENINTERP_COREML_CACHE"  # cache directory, or "off"
SIZE_ENV = "OPENINTERP_COREML_CACHE_GB"
DEFAULT_GB = 20.0
MAX_ERROR = 0.03  # largest relative RMS error against fp32 (measured: 0.003 to 0.011)
SOURCES = ("forward.py", "convert.py")  # the code that sets what an entry holds
MODEL = "model.mlmodelc"
META = "meta.json"


class ConversionRejected(RuntimeError):
    """A converted model did not agree with fp32 closely enough to use."""


def code_digest() -> str:
    """A hash of the source of the conversion code."""
    digest = hashlib.sha256()
    for name in SOURCES:
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()[:16]


def machine() -> dict:
    """The macOS version and the chip, which set what the ANE's compiler makes."""
    chip = None
    with suppress(OSError, subprocess.SubprocessError):
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
        ).stdout.strip()
    return {"macos": platform.mac_ver()[0] or None, "chip": chip or platform.machine()}


def fields(model: str, revision: str | None, layer: int, rows: int, seq: int) -> dict:
    """What an entry is keyed by."""
    return {
        "format": FORMAT,
        "model": model,
        "revision": revision,
        "layer": layer,
        "shape": [rows, seq],
        "code": code_digest(),
        "coremltools": convert.version(),
        "torch": torch.__version__,
        **machine(),
    }


class ConversionCache:
    """Compiled Core ML models on disk, built once per machine and key."""

    def __init__(self, root: Path, max_bytes: int | None = None) -> None:
        self.root = root
        self.max_bytes = max_bytes

    @classmethod
    def default(cls) -> ConversionCache | None:
        """The cache at ``$OPENINTERP_COREML_CACHE`` (default ``~/.cache/openinterp/coreml``)."""
        root = cache_root(ENV, "coreml")
        if root is None:
            return None
        gb = float(os.environ.get(SIZE_ENV) or DEFAULT_GB)
        return cls(root, int(gb * 2**30))

    def path(self, key_fields: dict) -> Path:
        """The entry directory for ``key_fields``."""
        key = hashlib.sha256(json.dumps(key_fields, sort_keys=True).encode()).hexdigest()[:20]
        model = key_fields["model"].replace("/", "--")
        return self.root / model / f"L{key_fields['layer']}" / key

    def get(
        self,
        key_fields: dict,
        build: Callable[[Path], None],
        check: Callable[[Path], float],
    ) -> Path:
        """The compiled model for ``key_fields``; build and check it if the cache has none.

        ``build`` writes a ``.mlmodelc`` to the path it gets. ``check`` loads the model
        at its final path and returns its relative error against fp32. An entry with an
        error above :data:`MAX_ERROR` is deleted, and this raises :class:`ConversionRejected`.
        """
        entry = self.path(key_fields)
        if (entry / META).exists():
            self._touch(entry)
            return entry / MODEL
        with self._lock(entry):
            if not (entry / META).exists():
                self._build(entry, key_fields, build, check)
        self.evict(keep=entry)
        return entry / MODEL

    def _build(self, entry: Path, key_fields: dict, build, check) -> None:
        shutil.rmtree(entry, ignore_errors=True)  # left by a build that stopped
        tmp = entry.parent / f".tmp-{entry.name}-{secrets.token_hex(4)}"
        try:
            tmp.mkdir(parents=True)
            build(tmp / MODEL)
            os.rename(tmp, entry)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        error = check(entry / MODEL)  # at the final path, so macOS keeps the ANE's compile
        if not error <= MAX_ERROR:
            shutil.rmtree(entry, ignore_errors=True)
            raise ConversionRejected(f"relative error {error:.4f} is more than {MAX_ERROR}")
        meta = {"fields": key_fields, "error": error, "built": _now(), "last_used": _now()}
        _write_json(entry / META, meta)

    @contextmanager
    def _lock(self, entry: Path) -> Iterator[None]:
        locks = self.root / ".locks"
        locks.mkdir(parents=True, exist_ok=True)
        with open(locks / f"{entry.name}.lock", "w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _touch(self, entry: Path) -> None:
        with suppress(OSError, ValueError):
            meta = json.loads((entry / META).read_text())
            meta["last_used"] = _now()
            _write_json(entry / META, meta)

    def entries(self) -> list[tuple[Path, int, str]]:
        """Each complete entry, its size in bytes, and when it was last used."""
        found = []
        for meta in self.root.glob(f"*/L*/*/{META}"):
            with suppress(OSError, ValueError):
                used = json.loads(meta.read_text()).get("last_used", "")
                size = sum(f.stat().st_size for f in meta.parent.rglob("*") if f.is_file())
                found.append((meta.parent, size, used))
        return found

    def evict(self, keep: Path | None = None) -> list[Path]:
        """Delete the least recently used entries until the cache is inside its cap."""
        if self.max_bytes is None:
            return []
        entries = sorted(self.entries(), key=lambda e: e[2])
        total = sum(size for _, size, _ in entries)
        gone = []
        for path, size, _ in entries:
            if total <= self.max_bytes:
                break
            if path == keep:
                continue
            shutil.rmtree(path, ignore_errors=True)
            total -= size
            gone.append(path)
        return gone


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
    os.replace(tmp, path)
