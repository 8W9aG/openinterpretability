"""The repository's size against its budget (issue #3254, docs/analyses/repo-size.md).

Every experiment adds a record of about 175 KB to ``wiki/``, so the repository grows with
the registry. That slows every clone, CI checkout and agent worktree. This script reports
the tracked size and file count, in total and per top-level directory, and the packed size
of ``.git``. It compares them with ``[tool.openinterp.size_budget]`` in ``pyproject.toml``.

Over budget, it prints a GitHub Actions warning. It never fails, because where records
live in the long term is still open (#3254). In CI it also writes a line to the job
summary (``$GITHUB_STEP_SUMMARY``).

    uv run python scripts/repo_size.py [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tomllib
from collections import defaultdict
from pathlib import Path

DEFAULT_BUDGET = {"tracked_mb": 1000, "files": 20000}  # [tool.openinterp.size_budget] overrides
MB = 1024 * 1024


def tracked(ls_tree: str) -> dict[str, dict[str, int]]:
    """``{top-level dir: {"bytes": n, "files": n}}`` from ``git ls-tree -r -l`` output."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"bytes": 0, "files": 0})
    for line in ls_tree.splitlines():
        meta, _, path = line.partition("\t")
        size = meta.split()[-1]
        if not path or not size.isdigit():  # submodules and links have no size
            continue
        top = path.split("/", 1)[0] if "/" in path else "."
        out[top]["bytes"] += int(size)
        out[top]["files"] += 1
    return dict(out)


def packed_bytes(count_objects: str) -> int:
    """The packed size of ``.git``, from ``git count-objects -v`` output (in KiB there)."""
    fields = dict(line.split(": ", 1) for line in count_objects.splitlines() if ": " in line)
    return (int(fields.get("size-pack", 0)) + int(fields.get("size", 0))) * 1024


def budget(pyproject: Path = Path("pyproject.toml")) -> dict[str, int]:
    """``[tool.openinterp.size_budget]``, over the defaults."""
    try:
        with pyproject.open("rb") as f:
            config = tomllib.load(f).get("tool", {}).get("openinterp", {})
    except (OSError, tomllib.TOMLDecodeError):
        config = {}
    return DEFAULT_BUDGET | config.get("size_budget", {})


def report(dirs: dict[str, dict[str, int]], packed: int, limits: dict[str, int]) -> dict:
    """The totals, the per-directory sizes and which limits are passed."""
    total = sum(d["bytes"] for d in dirs.values())
    files = sum(d["files"] for d in dirs.values())
    over = []
    if total / MB > limits["tracked_mb"]:
        over.append(f"tracked {total / MB:.0f} MB > {limits['tracked_mb']} MB")
    if files > limits["files"]:
        over.append(f"{files} files > {limits['files']}")
    largest = sorted(dirs.items(), key=lambda kv: kv[1]["bytes"], reverse=True)
    return {
        "tracked_mb": round(total / MB, 1),
        "files": files,
        "packed_mb": round(packed / MB, 1),
        "by_directory": {k: {"mb": round(v["bytes"] / MB, 1), "files": v["files"]}
                         for k, v in largest},
        "budget": limits,
        "over_budget": over,
    }  # fmt: skip


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def main() -> None:
    """Print the report, warn when over budget, and add a line to the CI job summary."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--json", action="store_true", help="Print the report as JSON.")
    args = ap.parse_args()
    result = report(
        tracked(_git("ls-tree", "-r", "-l", "HEAD")),
        packed_bytes(_git("count-objects", "-v")),
        budget(),
    )
    line = (
        f"Repository size: {result['tracked_mb']} MB tracked in {result['files']} files "
        f"({result['by_directory'].get('wiki', {}).get('mb', 0)} MB in wiki/), "
        f".git packed {result['packed_mb']} MB."
    )
    print(json.dumps(result, indent=1) if args.json else line)
    for problem in result["over_budget"]:
        print(f"::warning title=Repository size budget::{problem} (see issue #3254)")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
