"""A small on-disk memo for slow, pure results that a site build computes again each time.

Records never change once written, so most of what a build computes from them is the
same as last time. A memo keeps such results in an SQLite file under
``$XDG_CACHE_HOME/openinterp/build``. ``$OPENINTERP_BUILD_CACHE`` moves it, and ``off``
turns it off. The file's name holds a hash of ``version``. Pass the source of the code
that computes the results, so that a change to it starts a new file.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from openinterp.core.cache_dirs import cache_root


class DiskMemo:
    """Results by key, read from disk at the first lookup and added to by :meth:`save`."""

    def __init__(self, name: str, version: bytes) -> None:
        self.name, self.version = name, version
        self._stored: dict[str, Any] | None = None
        self._new: dict[str, Any] = {}

    def path(self) -> Path | None:
        """Return the memo's file, or ``None`` when the cache is turned off."""
        root = cache_root("OPENINTERP_BUILD_CACHE", "build")
        if root is None:
            return None
        return root / f"{self.name}-{hashlib.sha256(self.version).hexdigest()[:16]}.sqlite"

    def get(self, key: str) -> Any | None:
        """Return the result stored for ``key``, or ``None``."""
        if self._stored is None:
            self._stored = self._load()
        return self._new.get(key, self._stored.get(key))

    def put(self, key: str, value: Any) -> None:
        """Keep ``value`` for ``key``, to write at the next :meth:`save`."""
        self._new[key] = value

    def save(self) -> None:
        """Add the results kept since the last save to the file."""
        path = self.path()
        if path is None or not self._new:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path, timeout=30) as db:
            db.execute("CREATE TABLE IF NOT EXISTS memo (key TEXT PRIMARY KEY, value TEXT)")
            rows = [(k, json.dumps(v)) for k, v in self._new.items()]
            db.executemany("INSERT OR IGNORE INTO memo VALUES (?, ?)", rows)
        if self._stored is not None:
            self._stored.update(self._new)
        self._new.clear()

    def _load(self) -> dict[str, Any]:
        path = self.path()
        if path is None or not path.exists():
            return {}
        with sqlite3.connect(path, timeout=30) as db:
            return {k: json.loads(v) for k, v in db.execute("SELECT key, value FROM memo")}
