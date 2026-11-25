"""Repeated mutation runs, each in its own test order (#6359).

``mutation.py --runs N`` calls :func:`repeat`. It runs mutmut N times, each from a clean
``mutants/``. Run k gives pytest ``-p randomly --randomly-seed=k``, so each run has its own
test order. It reports each run's counts. It also reports each mutant that fails in a run, and
each mutant whose result differs between runs. A result that differs means that a test
depends on order, timing or state: a flaky test.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path


def statuses(results: str) -> dict[str, str]:
    """Each mutant's status in the output of ``mutmut results --all true``.

    Args:
        results: The output.

    Returns:
        ``{mutant: status}``.

    Examples:
        >>> statuses("    a__mutmut_1: killed\\n    b__mutmut_2: no tests\\n")
        {'a__mutmut_1': 'killed', 'b__mutmut_2': 'no tests'}
    """
    pairs = (line.strip().rpartition(": ") for line in results.splitlines())
    return {name: status for name, sep, status in pairs if sep}


def changed(runs: list[dict[str, str]]) -> list[str]:
    """The mutants whose status differs between runs, each with its status in each run.

    Args:
        runs: :func:`statuses` of each run.

    Returns:
        Lines such as ``a__mutmut_1: killed, survived``, sorted.

    Examples:
        >>> changed([{"a": "killed", "b": "killed"}, {"a": "survived", "b": "killed"}])
        ['a: killed, survived']
        >>> changed([{"a": "killed"}, {}])
        ['a: killed, missing']
    """
    names = sorted(set().union(*runs))
    return [f"{n}: {', '.join(r.get(n, 'missing') for r in runs)}" for n in names
            if len({r.get(n, 'missing') for r in runs}) > 1]  # fmt: skip


def repeat(runs: int, base: str | None, script: Path, failing: Sequence[str],
           root: Path = Path(".")) -> int:  # fmt: skip
    """Run mutmut ``runs`` times, each in its own test order, and compare the results.

    Args:
        runs: How many runs. Run k has the pytest-randomly seed k.
        base: The git ref for ``--base``, or None for the verdict code.
        script: ``mutation.py``, which runs each run and reads its results.
        failing: The statuses that a run must not leave, such as ``survived``.
        root: The repository's root, which holds ``mutants/``.

    Returns:
        The exit status: 1 when a mutant fails in a run or changes between runs, else 0.
    """
    tables = []
    for seed in range(1, runs + 1):
        shutil.rmtree(root / "mutants", ignore_errors=True)
        options = [*(["--base", base] if base else []), "--order-seed", str(seed)]
        command = [sys.executable, str(script), *options]
        subprocess.run(command, cwd=root, check=False)
        out = subprocess.run([*command, "results", "--all", "true"], cwd=root, check=False,
                             capture_output=True, text=True).stdout  # fmt: skip
        tables.append(statuses(out))
        counts = Counter(tables[-1].values())
        print(f"Run {seed} (--randomly-seed={seed}): {len(tables[-1])} mutants,",
              ", ".join(f"{n} {s}" for s, n in sorted(counts.items())) or "no results")  # fmt: skip
    bad = sorted({f"run {k}: {n}: {s}" for k, t in enumerate(tables, 1) for n, s in t.items()
                  if s in failing})  # fmt: skip
    moved = changed(tables)
    print(*bad, sep="\n") if bad else print(f"No mutant failed in {runs} runs.")
    print(*moved, sep="\n") if moved else print("No mutant changed its result between runs.")
    return 1 if bad or moved or not all(tables) else 0
