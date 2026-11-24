"""Where a project's on-disk caches live (#6628)."""

from __future__ import annotations

import os
from pathlib import Path


def cache_root(env: str, name: str, base: Path) -> Path | None:
    """``$<env>``, else ``<base>/<name>``. ``None`` when ``$<env>`` turns the cache off.

    Args:
        env: The environment variable that moves the cache, or turns it off.
        name: The cache's folder in ``base``.
        base: The project's cache folder.

    Returns:
        The cache's folder, or ``None`` when it is off.

    Examples:
        >>> cache_root("LAB_NO_SUCH_CACHE", "pii", Path("/c")).as_posix()
        '/c/pii'
    """
    value = os.environ.get(env)
    if value is not None:
        return None if value.lower() in ("", "0", "off", "false") else Path(value)
    return base / name
