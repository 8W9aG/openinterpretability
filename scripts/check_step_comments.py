"""Fail on a comment that starts with a step number, such as ``# 1. Scan evidence.`` (#3163).

The number goes stale when a step is added or moved, and the code's order is already the
order. Write ``# Scan evidence.`` instead. Only a comment on a line of its own counts:
``#`` in Python, shell, YAML and the Makefile, ``//`` in JavaScript, ``{#`` in templates.
Docstrings, Markdown and references such as "§3 step 9" are not comments, so they pass.

Run with ``make lint``; CI runs it in the lint step.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

MARKERS = {
    "#": (".py", ".sh", ".yml", ".yaml", ".toml", "Makefile"),
    "//": (".js", ".mjs"),
    "{#": (".html", ".jinja", ".j2"),
}
STEP = r"\s*\d+[.)]\s"


def pattern(marker: str) -> re.Pattern[str]:
    """A whole-line comment with ``marker`` that starts with a step number."""
    return re.compile(rf"^\s*{re.escape(marker)}{STEP}")


def offences(path: str, text: str) -> list[str]:
    """``path:line: text`` for each numbered step comment in ``text``."""
    found = []
    for marker, endings in MARKERS.items():
        if not path.endswith(endings):
            continue
        rule = pattern(marker)
        found += [
            f"{path}:{n}: {line.strip()}"
            for n, line in enumerate(text.splitlines(), 1)
            if rule.match(line)
        ]
    return found


def main() -> int:
    """Check every tracked file outside ``wiki/``. Exit 1 on any offence."""
    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True)
    found = []
    for name in files.stdout.splitlines():
        path = Path(name)
        if name.startswith("wiki/") or not path.is_file():
            continue
        try:
            found += offences(name, path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
    for line in found:
        print(f"{line}\n  Remove the step number: the code's order is the order.")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
