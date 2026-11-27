"""Print the labinacan requirement that ``pyproject.toml`` pins, without its extras (#7097).

A workflow that runs a script with no project installed gives it this one package::

    uv run --no-project --with "$(python3 scripts/labinacan_pin.py)" python scripts/x.py

The extras (``r2``, ``pii``) are for the site build, and the issue scripts need neither.
It reads the file with the standard library only, so it runs before anything is installed.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def pin(pyproject: Path = PYPROJECT) -> str:
    """The ``labinacan`` requirement of ``pyproject``, with its extras left out.

    Args:
        pyproject: The project's ``pyproject.toml``.

    Returns:
        A requirement such as ``labinacan==0.1.3``.

    Raises:
        SystemExit: ``pyproject`` names no ``labinacan`` dependency.

    Examples:
        >>> pin().startswith("labinacan==")
        True
    """
    deps = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["dependencies"]
    for dep in deps:
        name, _, rest = re.match(r"([A-Za-z0-9_.-]+)(\[[^\]]*\])?(.*)", dep.strip()).groups()
        if name.lower() == "labinacan":
            return f"labinacan{rest.strip()}"
    raise SystemExit(f"{pyproject} names no labinacan dependency")


if __name__ == "__main__":
    print(pin(Path(sys.argv[1]) if len(sys.argv) > 1 else PYPROJECT))
