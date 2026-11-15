"""Run the STE linter on prose only: a paper's reference list is not prose (issue #3137).

APA 7 lists every author, so a reference can run past the linter's 25 words. This copies
the given Markdown files into a temporary folder, at the same relative paths, with each
line of a ``## References`` section blanked. Line numbers and paths stay the same. Then it
runs ``ste-lint.py`` there, with the arguments given, and returns its exit code.

    python3 scripts/ste_prose.py [ste-lint options] FILE...
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LINTER = Path(".claude/skills/asd-ste100/scripts/ste-lint.py").resolve()


def without_references(text: str) -> str:
    """``text`` with each line of its ``## References`` section blank."""
    out, inside = [], False
    for line in text.splitlines(keepends=True):
        if line.startswith("## "):
            inside = line.strip() == "## References"
        out.append("\n" if inside and not line.startswith("## ") else line)
    return "".join(out)


def main(args: list[str]) -> int:
    """Lint ``args``' files, as copies with their references blanked."""
    with tempfile.TemporaryDirectory() as tmp:
        for arg in args:
            path = Path(arg)
            if arg.startswith("-") or not path.is_file():
                continue
            copy = Path(tmp) / path
            copy.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".md":
                copy.write_text(without_references(path.read_text(encoding="utf-8")))
            else:
                shutil.copy(path, copy)
        return subprocess.run([sys.executable, str(LINTER), *args], cwd=tmp).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
