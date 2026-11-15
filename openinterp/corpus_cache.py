"""On-disk cache of the Parquet data that ``open-interpretability corpus`` reads from the Hub.

A file at a dataset commit never changes, so an entry can't go stale. The cache keeps only
paths that name a full commit hash, as ``_read_globs`` in :mod:`openinterp.corpus` writes
them. It stores each file's row-group count, and each row group's columns.

A repeat of a build reads almost nothing from row groups on the Hub, for example
``open-interpretability corpus --same-as``. A build with a new seed reads other row groups, so it
saves little. A cache failure never fails a build: the code reads the Hub instead. Once
the cache outgrows its size cap, the least recently used entries go.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from openinterp.core.cache_dirs import cache_root

ENV = "OPENINTERP_CORPUS_CACHE"  # cache directory, or "off"
SIZE_ENV = "OPENINTERP_CORPUS_CACHE_GB"
DEFAULT_GB = 5.0  # a 500-document build stores about 100 MB at most
_PINNED = re.compile(r"@[0-9a-f]{40}/")


class ParquetCache:
    """The cache under ``root``, capped at ``max_bytes``."""

    def __init__(self, root: Path, max_bytes: int | None = None) -> None:
        self.root = Path(root)
        if max_bytes is None:
            max_bytes = int(float(os.environ.get(SIZE_ENV, DEFAULT_GB)) * 1e9)
        self.max_bytes = max_bytes

    @classmethod
    def default(cls) -> ParquetCache | None:
        """The cache at the default location, or ``None`` when caching is off."""
        root = cache_root(ENV, "corpora")
        return None if root is None else cls(root)

    def row_groups(self, fs, path: str) -> int:
        """Return how many row groups the file at ``path`` has."""
        entry = self.root / f"{_key(path)}.groups"
        with suppress(OSError, ValueError):
            count = int(entry.read_text(encoding="utf-8"))
            os.utime(entry)
            return count
        count = _remote_row_groups(fs, path)
        self._store(entry, lambda tmp: tmp.write_text(str(count), encoding="utf-8"))
        return count

    def read(self, fs, path: str, group: int, columns: Sequence[str]) -> pa.Table:
        """Return ``columns`` of one row group of the file at ``path``."""
        entry = self.root / f"{_key(path, str(group), *columns)}.parquet"
        with suppress(OSError, pa.ArrowInvalid):
            table = pq.read_table(entry)
            os.utime(entry)
            return table
        table = _remote_read(fs, path, group, columns)
        self._store(entry, lambda tmp: pq.write_table(table, tmp, compression="zstd"))
        return table

    def entries(self) -> list[Path]:
        """Every entry, least recently used first."""
        found = []
        for path in self.root.glob("*"):
            with suppress(OSError):  # another build can evict it during this loop
                found.append((path.stat().st_mtime, path))
        return [path for _, path in sorted(found)]

    def size(self, entries: Sequence[Path] | None = None) -> int:
        """Return the bytes that ``entries`` use, by default the whole cache."""
        total = 0
        for path in self.entries() if entries is None else entries:
            with suppress(OSError):
                total += path.stat().st_size
        return total

    def evict(self, keep: Path | None = None) -> None:
        """Delete least recently used entries until the cache fits its cap."""
        entries = self.entries()
        total = self.size(entries)
        for path in entries:
            if total <= self.max_bytes:
                return
            if path != keep:
                total -= self.size([path])
                path.unlink(missing_ok=True)

    def clear(self) -> int:
        """Delete every entry and return how many there were."""
        entries = self.entries()
        for path in entries:
            path.unlink(missing_ok=True)
        return len(entries)

    def _store(self, entry: Path, write) -> None:
        """Write ``entry`` through a temporary file, so a concurrent read never sees part of it."""
        tmp = entry.with_name(f"{entry.name}.{secrets.token_hex(4)}.tmp")
        with suppress(OSError):
            self.root.mkdir(parents=True, exist_ok=True)
            try:
                write(tmp)
                tmp.replace(entry)
            finally:
                tmp.unlink(missing_ok=True)
            self.evict(keep=entry)


def row_groups(fs, path: str) -> int:
    """Return how many row groups the file has, through the cache if ``path`` is pinned."""
    cache = _cache_for(path)
    return cache.row_groups(fs, path) if cache else _remote_row_groups(fs, path)


def read_row_group(fs, path: str, group: int, columns: Sequence[str]) -> pa.Table:
    """Return ``columns`` of one row group, through the cache if ``path`` is pinned."""
    cache = _cache_for(path)
    return cache.read(fs, path, group, columns) if cache else _remote_read(fs, path, group, columns)


def _cache_for(path: str) -> ParquetCache | None:
    return ParquetCache.default() if _PINNED.search(path) else None


def _key(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def _remote_row_groups(fs, path: str) -> int:
    with fs.open(path, "rb") as fh:
        return pq.ParquetFile(fh).num_row_groups


def _remote_read(fs, path: str, group: int, columns: Sequence[str]) -> pa.Table:
    with fs.open(path, "rb") as fh:
        return pq.ParquetFile(fh).read_row_group(group, columns=list(columns))
