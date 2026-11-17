"""Where the on-disk caches live."""

from __future__ import annotations

import os
from pathlib import Path


def cache_root(env: str, name: str) -> Path | None:
    """``$<env>``, else ``<cache folder>/<name>`` (``legacy_names.cache_base``); None if off."""
    value = os.environ.get(env)
    if value is not None:
        return None if value.lower() in ("", "0", "off", "false") else Path(value)
    from openinterpretability.core.legacy_names import cache_base

    return cache_base() / name
