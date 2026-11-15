"""IFTTT lint: a block that changes must change with the files it names (issue #3161).

Mark a block that has a copy elsewhere, in any comment syntax:

    # LINT.IfChange
    ALLOWED_MODELS: …
    # LINT.ThenChange(pyproject.toml)

The check fails when a diff changes or removes a line inside such a block but doesn't
change every file that its ``ThenChange`` names. A named file that doesn't exist fails
too. Paths are from the repository root. Several are separated by commas.

The diff is against the merge base with ``--base`` (default ``origin/main``), and it
includes uncommitted edits. With no base to compare, the check passes.

    python3 scripts/ifttt_lint.py [--base origin/main]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

IF_CHANGE = "LINT.IfChange"
THEN_CHANGE = re.compile(r"LINT\.ThenChange\(([^)]*)\)")
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.M)


def _git(*args: str) -> str | None:
    done = subprocess.run(["git", *args], capture_output=True, text=True)
    return done.stdout if done.returncode == 0 else None


def changed_lines(diff: str) -> dict[str, set[int]]:
    """Each file's changed lines in its new version, from ``git diff -U0`` output.

    A hunk that only removes lines marks the line after the removal, so a removal
    inside a block counts as a change to it.
    """
    out: dict[str, set[int]] = {}
    for part in re.split(r"^diff --git ", diff, flags=re.M)[1:]:
        name = re.search(r"^\+\+\+ (?:b/)?(.+)$", part, re.M)
        old = re.search(r"^--- (?:a/)?(.+)$", part, re.M)
        path = name.group(1) if name and name.group(1) != "/dev/null" else old and old.group(1)
        lines = out.setdefault(path, set())
        for start, count in HUNK.findall(part):
            first, n = int(start), int(count or 1)
            lines.update(range(first, first + n) if n else (first, first + 1))
    return out


def blocks(text: str) -> list[tuple[int, int, list[str]]]:
    """``(first line, last line, targets)`` of each IfChange block, 1-based and inclusive."""
    found, start = [], None
    for number, line in enumerate(text.splitlines(), 1):
        if IF_CHANGE in line:
            start = number
        elif start is not None and (m := THEN_CHANGE.search(line)):
            targets = [t.strip() for t in m.group(1).split(",") if t.strip()]
            found.append((start, number, targets))
            start = None
    return found


def problems(changed: dict[str, set[int]], root: Path = Path(".")) -> list[str]:
    """What fails: a block changed without its targets, or a target that doesn't exist."""
    out = []
    for path, lines in sorted(changed.items()):
        file = root / path
        if not file.is_file():
            continue
        for first, last, targets in blocks(file.read_text(encoding="utf-8", errors="replace")):
            for target in targets:
                if not (root / target).exists():
                    out.append(f"{path}:{last}: ThenChange names {target}, which doesn't exist")
            touched = any(first <= n <= last for n in lines)
            missing = [t for t in targets if t not in changed]
            if touched and missing:
                out.append(
                    f"{path}:{first}-{last}: this block changed, so change {', '.join(missing)} too"
                )
    return out


def main() -> int:
    """Check the diff against ``--base``; print each problem and return 1 if any."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main")
    base = _git("merge-base", parser.parse_args().base, "HEAD")
    if base is None:
        print("ifttt: no base to compare with, skipped")
        return 0
    found = problems(changed_lines(_git("diff", "-U0", "--no-renames", base.strip()) or ""))
    for problem in found:
        print(problem)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
