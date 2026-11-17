"""The project's old name, read only where a user's machine or a signed record still has it.

The package was called ``openinterp`` until issue #4089. That name belongs to an unrelated
project, so the code doesn't use it, except in this module. ``scripts/check_old_name.py``
fails ``make lint`` if the old name appears anywhere else, outside the frozen places it
lists. This module reads the old names as a fallback:

- *Environment variables.* :func:`adopt_legacy_env` copies each ``OPENINTERP_<X>`` that is
  set to ``OPENINTERPRETABILITY_<X>`` when that is unset, and warns once. The package calls
  it at import. It goes away one release after 0.0.22.
- *The cache folder.* A machine that has ``~/.cache/openinterp`` and no
  ``~/.cache/openinterpretability`` keeps the old folder (:func:`cache_base`). That folder
  holds the locks, leases and pause files that running loops share (``core/heavy_lock.py``,
  ``harness/devices.py``). New code and old code must use the same one. Nothing moves.
- *The signing key and keychain items.* The new name first, then the old one.

The record field ``openinterp_version`` and the signing namespace ``openinterp-record``
keep the old name in ``core/run_environment.py`` and ``signing.py``. About 6,100 signed
records carry them, and their signatures cover them.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import MutableMapping
from pathlib import Path

NAME = "openinterpretability"
LEGACY = "openinterp"
ENV_PREFIX = "OPENINTERPRETABILITY_"
LEGACY_ENV_PREFIX = LEGACY.upper() + "_"


def adopt_legacy_env(environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Copy each set legacy variable to its new name when that is unset; return the new names."""
    environ = os.environ if environ is None else environ
    adopted = []
    for key, value in list(environ.items()):
        if key.startswith(LEGACY_ENV_PREFIX) and not key.startswith(ENV_PREFIX):
            new = ENV_PREFIX + key[len(LEGACY_ENV_PREFIX) :]
            if new not in environ:
                environ[new] = value
                adopted.append(new)
    if adopted:
        warnings.warn(
            f"{LEGACY_ENV_PREFIX}* variables are renamed {ENV_PREFIX}* (#4089). "
            f"Read for now: {', '.join(sorted(adopted))}.",
            DeprecationWarning,
            stacklevel=2,
        )
    return adopted


def cache_base() -> Path:
    """The cache folder: ``$XDG_CACHE_HOME/openinterpretability``, else ``~/.cache/…``.

    A machine with only the old folder keeps using it (module docstring).
    """
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    new, old = base / NAME, base / LEGACY
    return old if old.is_dir() and not new.exists() else new


def signing_key() -> Path:
    """The default signing key: ``~/.ssh/openinterpretability_signing``, else the old file."""
    new = Path(f"~/.ssh/{NAME}_signing")
    old = Path(f"~/.ssh/{LEGACY}_signing")
    return old if old.expanduser().exists() and not new.expanduser().exists() else new


def keychain_services(service: str) -> list[str]:
    """The keychain services to try for ``service``: itself, then its old name."""
    prefix = NAME + "-"
    if service.startswith(prefix):
        return [service, LEGACY + "-" + service[len(prefix) :]]
    return [service]
