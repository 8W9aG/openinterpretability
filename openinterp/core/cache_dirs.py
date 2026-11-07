"""Where the on-disk caches live."""

from __future__ import annotations

import os
from pathlib import Path


def cache_root(env: str, name: str) -> Path | None:
    """``$<env>``, else ``$XDG_CACHE_HOME/openinterp/<name>``; None if ``$<env>`` turns it off."""
    value = os.environ.get(env)
    if value is not None:
        return None if value.lower() in ("", "0", "off", "false") else Path(value)
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "openinterp" / name
