"""Keep loaded models across commands that run in one process (``open-interpretability loop``).

Each ``open-interpretability`` command loads its model afresh, which is right for one command
per process. ``open-interpretability loop`` runs many commands in one process, so it turns this
cache on. Then each experiment on a model after the first skips the load. The cache is off by
default.
"""

from __future__ import annotations

import functools

_runners: dict[tuple, object] | None = None  # None: the cache is off


def keep() -> None:
    """Turn the cache on for the rest of this process."""
    global _runners
    if _runners is None:
        _runners = {}


def cached(load):
    """Wrap ``load`` so that, while the cache is on, equal arguments return one runner."""

    @functools.wraps(load)
    def wrapper(*args, **kwargs):
        if _runners is None:
            return load(*args, **kwargs)
        key = (args, tuple(sorted(kwargs.items())))
        if key not in _runners:
            _runners[key] = load(*args, **kwargs)
        return _runners[key]

    return wrapper
