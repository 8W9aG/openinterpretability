"""Where the on-disk caches live, in this project's cache folder (``labinacan/cache.py``)."""

from __future__ import annotations

from pathlib import Path

from labinacan import cache


def cache_root(env: str, name: str) -> Path | None:
    """``$<env>``, else ``<cache folder>/<name>`` (``legacy_names.cache_base``); None if off."""
    from openinterpretability.core.legacy_names import cache_base

    return cache.cache_root(env, name, cache_base())
