"""Mutation testing of the package code that a branch changes (#5856).

``make mutation BASE=origin/main`` runs this script. It finds the ``.py`` files under
``openinterpretability/`` that the branch changes against the base. That covers committed,
uncommitted and untracked changes, but not deleted files. It mutates only those files, and
it runs only the tests that import them. Of their mutants, it runs only those in the
functions that the branch changes, and at most :data:`CAP` of them, a seeded sample
(#5857). A change to a large file then still takes minutes. mutmut reads its config from
``pyproject.toml`` alone, so the script overrides two keys in this process before mutmut starts:
``only_mutate`` and ``pytest_add_cli_args_test_selection``. It does not change the file.
Without ``--base``, mutmut runs the fixed verdict-code config of ``[tool.mutmut]`` (#5763).

The script prints each mutant that survives, or that no test reaches. It exits with 1 when
there is one, so the run can gate a pull request. A mutant that changes nothing a caller can
see is recorded on its line with mutmut's pragma and the reason:
``# pragma: no mutate, equivalent: <why>``.

    uv run python scripts/mutation.py --base origin/main
    uv run python scripts/mutation.py --base origin/main show <mutant>
"""

from __future__ import annotations

import ast
import random
import re
import subprocess
import sys
from pathlib import Path

PACKAGE = "openinterpretability"
FAILING = ("survived", "no tests", "timeout", "suspicious")
# Left out of mutmut's copy of the repository. mutmut copies the package and the tests itself.
# No unit test reads the 840 MB of experiment records.
NOT_COPIED = {PACKAGE, "tests", "wiki/experiments"}
# With --base, at most this many mutants run, a seeded sample of those in the changed
# functions. So a large change still takes minutes, not most of an hour (#5857).
CAP = 100
SEED = 0
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.M)


def _git(root: Path, *args: str) -> list[str]:
    """The non-empty lines that a git command prints.

    Args:
        root: The repository's root.
        *args: The git command's arguments.

    Returns:
        The lines.
    """
    out = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
    return [line for line in out.stdout.splitlines() if line]


def changed_files(base: str, root: Path = Path(".")) -> list[str]:
    """The package's ``.py`` files that differ from ``base``, and the new ones.

    Args:
        base: The git ref to compare with, such as ``origin/main``.
        root: The repository's root.

    Returns:
        The paths relative to ``root``, sorted. A deleted file is left out.
    """
    names = set(_git(root, "diff", "--name-only", "--diff-filter=d", f"{base}...HEAD"))
    names |= set(_git(root, "diff", "--name-only", "--diff-filter=d", "HEAD"))
    names |= set(_git(root, "ls-files", "--others", "--exclude-standard"))
    ours = (n for n in names if n.startswith(f"{PACKAGE}/") and n.endswith(".py"))
    return sorted(n for n in ours if (root / n).exists())


def changed_lines(base: str, path: str, root: Path = Path(".")) -> set[int] | None:
    """The lines of ``path`` that the branch adds or changes, against its merge base.

    A deletion counts as a change of the line after it. Committed and uncommitted
    changes both count.

    Args:
        base: The git ref to compare with.
        path: A path relative to ``root``.
        root: The repository's root.

    Returns:
        The 1-based line numbers, or None for a file that git doesn't track yet.
    """
    if not _git(root, "ls-files", path):
        return None
    merge_base = _git(root, "merge-base", base, "HEAD")[0]
    diff = "\n".join(_git(root, "diff", "-U0", merge_base, "--", path))
    lines: set[int] = set()
    for start, count in HUNK.findall(diff):
        first, n = int(start), int(count or 1)
        lines |= set(range(first, first + n)) if n else {first + 1}  # +c,0 follows line c
    return lines


def function_spans(source: str) -> dict[str, tuple[int, int]]:
    r"""Each function's name as mutmut writes it, and its first and last line.

    mutmut names a module's function ``x_<name>`` and a class's method
    ``xǁ<class>ǁ<name>``. A nested function belongs to the one around it.

    Args:
        source: A module's code.

    Returns:
        ``{mangled name: (first line, last line)}``.

    Examples:
        >>> function_spans("def f():\n    pass\n\nclass C:\n    def m(self):\n        pass\n")
        {'x_f': (1, 2), 'xǁCǁm': (5, 6)}
    """
    spans = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            spans[f"x_{node.name}"] = (node.lineno, node.end_lineno)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    spans[f"xǁ{node.name}ǁ{item.name}"] = (item.lineno, item.end_lineno)
    return spans


def touched(names: list[str], spans: dict[str, tuple[int, int]], lines: set[int] | None):
    """The mutant names whose function holds a changed line; all of them when ``lines`` is None.

    Args:
        names: mutmut's mutant names for one file, such as ``x_f__mutmut_3``.
        spans: :func:`function_spans` of the file.
        lines: :func:`changed_lines` of the file.

    Returns:
        The names, in their order.

    Examples:
        >>> touched(["x_f__mutmut_1", "x_g__mutmut_1"], {"x_f": (1, 2), "x_g": (4, 5)}, {2})
        ['x_f__mutmut_1']
    """
    if lines is None:
        return list(names)
    span = [spans.get(n.partition("__mutmut_")[0], (0, -1)) for n in names]
    return [n for n, (first, last) in zip(names, span, strict=True)
            if any(first <= line <= last for line in lines)]  # fmt: skip


def sample(keys: list[str], cap: int = CAP, seed: int = SEED) -> list[str]:
    """``keys``, or a seeded sample of ``cap`` of them when there are more.

    Args:
        keys: The mutants' full names.
        cap: The most to keep.
        seed: The sample's seed, so a rerun tests the same mutants.

    Returns:
        The names, sorted.

    Examples:
        >>> sample(["b", "a"]), len(sample([str(i) for i in range(5)], cap=2))
        (['a', 'b'], 2)
    """
    keys = sorted(keys)
    return keys if len(keys) <= cap else sorted(random.Random(seed).sample(keys, cap))


def selected(paths: list[str], base: str, root: Path = Path(".")) -> tuple[list[str], int]:
    """The mutants to run: those in the changed functions of ``paths``, at most :data:`CAP`.

    Args:
        paths: The changed package files.
        base: The git ref to compare with.
        root: The repository's root.

    Returns:
        The mutants' full names, and how many the changed functions hold before the cap.
    """
    from mutmut.mutation.file_mutation import mutate_file_contents
    from mutmut.utils.format_utils import get_mutant_name

    keys = []
    for path in paths:
        source = (root / path).read_text(encoding="utf-8")
        names = mutate_file_contents(path, source).mutant_names
        lines = changed_lines(base, path, root)
        keys += [get_mutant_name(Path(path), n)
                 for n in touched(names, function_spans(source), lines)]  # fmt: skip
    return sample(keys), len(keys)


def module_name(path: str) -> str:
    """The dotted module of a package path.

    Args:
        path: A path such as ``openinterpretability/core/stats.py``.

    Returns:
        The module's dotted name.

    Examples:
        >>> module_name("openinterpretability/core/stats.py")
        'openinterpretability.core.stats'
        >>> module_name("openinterpretability/interference/__init__.py")
        'openinterpretability.interference'
    """
    parts = Path(path).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imports(text: str, module: str) -> bool:
    """Whether a test file's ``text`` imports ``module`` or names it.

    A file counts when it names the module's dotted path. It also counts when it imports the
    module's last part from its package, as in ``from openinterpretability.core import stats``.

    Args:
        text: The test file's source.
        module: The dotted module name.

    Returns:
        True when the file uses the module.

    Examples:
        >>> imports("from pkg.core import stats, schemas", "pkg.core.stats")
        True
        >>> imports("from pkg.core import statsmodels", "pkg.core.stats")
        False
    """
    if re.search(rf"\b{re.escape(module)}\b", text):
        return True
    package, _, name = module.rpartition(".")
    for match in re.finditer(
        rf"from {re.escape(package)} import \(?([^)\n]*(?:\n[^)]*)?)\)?", text
    ):
        if re.search(rf"\b{re.escape(name)}\b", match.group(1)):
            return True
    return False


def tests_for(paths: list[str], tests: Path = Path("tests")) -> list[str]:
    """The test files that import any of ``paths``.

    Args:
        paths: The changed package files.
        tests: The test folder.

    Returns:
        The test files, sorted.
    """
    modules = [module_name(p) for p in paths]
    return sorted(str(t) for t in tests.rglob("test_*.py")
                  if any(imports(t.read_text(encoding="utf-8"), m) for m in modules))  # fmt: skip


def copied(root: Path = Path(".")) -> list[str]:
    """What mutmut copies beside the package, so that any selected test finds its files.

    The fixed config copies only what the verdict tests read. A changed file can bring in
    any test, so this copies every tracked top-level entry but those in :data:`NOT_COPIED`.
    It copies ``wiki/`` entry by entry for the same reason.

    Args:
        root: The repository's root.

    Returns:
        The paths, relative to ``root``.
    """
    top = [
        n
        for n in _git(root, "ls-tree", "--name-only", "HEAD")
        if n not in NOT_COPIED and n != "wiki"
    ]
    wiki = [n for n in _git(root, "ls-tree", "--name-only", "HEAD", "wiki/") if n not in NOT_COPIED]
    # mutmut copies a file into an existing folder only, so the wiki's folders go first.
    return top + sorted(wiki, key=lambda n: not (root / n).is_dir())


def override(only_mutate: list[str], selection: list[str], also_copy: list[str]) -> None:
    """Make mutmut read the files to mutate, to run and to copy from here, not the config.

    Args:
        only_mutate: The files to mutate.
        selection: The test files to run.
        also_copy: The paths that mutmut copies beside the package and the tests.
    """
    from mutmut import configuration

    reader = configuration._config_reader
    values = {"only_mutate": only_mutate, "pytest_add_cli_args_test_selection": selection,
              "also_copy": also_copy}  # fmt: skip

    def patched():
        read = reader()
        return lambda key, default: values[key] if key in values else read(key, default)

    configuration._config_reader = patched
    configuration.reset_config()


def failing(results: str) -> list[str]:
    """The lines of ``mutmut results`` for mutants that a run must not leave.

    Args:
        results: The output of ``mutmut results``.

    Returns:
        The lines, stripped.

    Examples:
        >>> failing("    a__mutmut_1: survived\\n    b__mutmut_2: no tests\\n")
        ['a__mutmut_1: survived', 'b__mutmut_2: no tests']
    """
    return [line.strip() for line in results.splitlines()
            if line.strip().rpartition(": ")[2] in FAILING]  # fmt: skip


def main(argv: list[str]) -> int:
    """Run mutmut on the changed files, or on the verdict code, and report.

    Args:
        argv: ``[--base <ref>] [<mutmut command>...]``. With a mutmut command, such as
            ``show <mutant>``, the script runs that command under the same selection.

    Returns:
        The exit status: 1 when a mutant survives or no test reaches it, else 0.
    """
    import contextlib
    import io

    from mutmut.__main__ import cli

    command = argv
    if argv[:1] == ["--base"]:
        base, command = argv[1], argv[2:]
        paths = changed_files(base)
        if not paths:
            print(f"No package file changes against {base}: nothing to mutate.")
            return 0
        selection = tests_for(paths)
        if not command:
            print(f"Mutating {len(paths)} files with {len(selection)} test files:", *paths,
                  sep="\n  ")  # fmt: skip
        override(paths, selection, copied())
    if command:
        cli(command, standalone_mode=False)
        return 0
    run = ["run"]
    if argv[:1] == ["--base"]:
        names, total = selected(paths, base)
        if not names:
            print("No mutant in the changed functions: nothing to run.")
            return 0
        print(f"Running {len(names)} of the {total} mutants in the changed functions"
              f" (at most {CAP}, sampled with seed {SEED}).")  # fmt: skip
        run += names
    cli(run, standalone_mode=False)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cli(["results"], standalone_mode=False)
    bad = failing(out.getvalue())
    print(*bad, sep="\n") if bad else print("No mutant survived.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
