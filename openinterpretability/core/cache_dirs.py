"""Where the on-disk caches live, in this project's cache folder (``labinacan.cache``)."""

from __future__ import annotations

from pathlib import Path

from labinacan import cache


def build_cache() -> Path | None:
    """The build's memo folder: ``$OPENINTERPRETABILITY_BUILD_CACHE``, else ``<cache>/build``.

    ``off`` turns it off (``labinacan.site.disk_memo``, #6628).

    Returns:
        The folder, or None when the cache is off.
    """
    return cache_root("OPENINTERPRETABILITY_BUILD_CACHE", "build")


def cache_root(env: str, name: str) -> Path | None:
    """``$<env>``, else ``<cache folder>/<name>`` (``legacy_names.cache_base``); None if off."""
    from openinterpretability.core.legacy_names import cache_base

    return cache.cache_root(env, name, cache_base())
