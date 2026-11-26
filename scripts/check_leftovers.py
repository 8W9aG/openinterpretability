"""Find this project's words in the files that the labinacan template takes (#6631).

The template must hold no word of this project's field and no name, URL or mark colour of
this project. ``scripts/template_paths.txt`` lists its files, and
``scripts/leftover_terms.txt`` lists the words, one regular expression on each line.

The check is a ratchet. ``scripts/leftovers_baseline.txt`` gives the number of hits that
each file may still have. The check fails when a file has more hits than its line allows,
and asks for a smaller line when a file has fewer. The template is cut when the baseline is
empty. In the template, the baseline is empty from the start.

    python3 scripts/check_leftovers.py            # check
    python3 scripts/check_leftovers.py --update   # write the baseline from the hits now
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATHS = ROOT / "scripts" / "template_paths.txt"
TERMS = ROOT / "scripts" / "leftover_terms.txt"
BASELINE = ROOT / "scripts" / "leftovers_baseline.txt"
EXEMPT = {"scripts/leftover_terms.txt", "scripts/leftovers_baseline.txt"}  # they name the words
BINARY = {".png", ".jpg", ".ico", ".ttf", ".otf", ".woff", ".woff2", ".pyc", ".dat", ".gz"}


def lines(path: Path) -> list[str]:
    """The lines of ``path`` that are not empty and not comments.

    Args:
        path: A list file.

    Returns:
        The lines, stripped, or [] when the file is missing.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     f = Path(tmp) / "list.txt"
        ...     _ = f.write_text("# a comment\\n\\nlabinacan\\n")
        ...     lines(f), lines(Path(tmp) / "missing.txt")
        (['labinacan'], [])
    """
    if not path.exists():
        return []
    stripped = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in stripped if line and not line.startswith("#")]


def files(paths: list[str], root: Path = ROOT) -> list[Path]:
    """Each text file under ``paths``: a file as it is, a folder in full.

    Args:
        paths: Paths relative to ``root``.
        root: The repository's root.

    Returns:
        The files, sorted, without caches, binary files and :data:`EXEMPT`.

    Examples:
        >>> [f.name for f in files(["scripts/check_leftovers.py"])]
        ['check_leftovers.py']
    """
    found = []
    for rel in paths:
        path = root / rel
        found += sorted(path.rglob("*")) if path.is_dir() else [path]
    return sorted(
        f
        for f in found
        if f.is_file()
        and "__pycache__" not in f.parts
        and f.suffix not in BINARY
        and f.relative_to(root).as_posix() not in EXEMPT
    )


def hits(path: Path, terms: list[re.Pattern]) -> int:
    """The number of matches of ``terms`` in ``path``.

    Args:
        path: A text file.
        terms: The words, compiled.

    Returns:
        The count, over every term.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     f = Path(tmp) / "a.md"
        ...     _ = f.write_text("A neuron, and Neurons.")
        ...     hits(f, [re.compile(r"(?i)\\bneurons?\\b")])
        2
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    return sum(len(term.findall(text)) for term in terms)


def counts(root: Path = ROOT) -> dict[str, int]:
    """Each template file that holds a leftover word, with its number of hits.

    Args:
        root: The repository's root, with the two list files under ``scripts/``.

    Returns:
        ``{path relative to root: hits}``, for the files with at least one hit.
    """
    terms = [re.compile(t) for t in lines(root / TERMS.relative_to(ROOT))]
    found = {}
    for f in files(lines(root / PATHS.relative_to(ROOT)), root):
        n = hits(f, terms)
        if n:
            found[f.relative_to(root).as_posix()] = n
    return found


def compare(now: dict[str, int], allowed: dict[str, int]) -> tuple[list[str], list[str]]:
    """The files over their allowance, and the files whose allowance can come down.

    Args:
        now: The hits now, by file.
        allowed: The baseline, by file.

    Returns:
        The problems, and the files whose line in the baseline is too high or not needed.

    Examples:
        >>> compare({"a": 2, "b": 1}, {"a": 1, "c": 3})
        (['a: 2 hits, the baseline allows 1', 'b: 1 hits, the baseline allows 0'], ['c'])
    """
    over = [
        f"{path}: {n} hits, the baseline allows {allowed.get(path, 0)}"
        for path, n in sorted(now.items())
        if n > allowed.get(path, 0)
    ]
    lower = [path for path, n in sorted(allowed.items()) if now.get(path, 0) < n]
    return over, lower


def read_baseline(path: Path = BASELINE) -> dict[str, int]:
    """The baseline: ``<count> <path>`` on each line.

    Args:
        path: The baseline file.

    Returns:
        ``{path: count}``.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     f = Path(tmp) / "b.txt"
        ...     _ = f.write_text("# comment\\n3 labinacan/a.py\\n")
        ...     read_baseline(f)
        {'labinacan/a.py': 3}
    """
    return {rel: int(n) for n, rel in (line.split(maxsplit=1) for line in lines(path))}


def write_baseline(found: dict[str, int], path: Path = BASELINE) -> None:
    """Write ``found`` as the baseline, the file with the most hits first.

    Args:
        found: The hits, by file.
        path: The baseline file.
    """
    head = (
        "# The leftover words that each template file may still hold "
        "(scripts/check_leftovers.py, #6631).\n"
        "# <hits> <path>. Lower a line when a file loses words. The template is cut when "
        "this list is empty.\n"
    )
    rows = sorted(found.items(), key=lambda item: (-item[1], item[0]))
    path.write_text(head + "".join(f"{n} {rel}\n" for rel, n in rows), encoding="utf-8")


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    """Check the template's files against the baseline, or write it.

    Args:
        argv: The arguments, or None for the command line's.
        root: The repository's root, with the three list files under ``scripts/``.

    Returns:
        0 when no file is over its allowance and none can come down, else 1.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update", action="store_true", help="Write the baseline.")
    args = parser.parse_args(argv)
    now, baseline = counts(root), root / BASELINE.relative_to(ROOT)
    if args.update:
        write_baseline(now, baseline)
        print(f"Wrote {len(now)} files, {sum(now.values())} hits, to {baseline.name}")
        return 0
    over, lower = compare(now, read_baseline(baseline))
    for problem in over:
        print(problem)
    if lower:
        print("Fewer hits now in:", ", ".join(lower))
        print("Run python3 scripts/check_leftovers.py --update to lower the baseline.")
    return 1 if over or lower else 0


if __name__ == "__main__":
    sys.exit(main())
