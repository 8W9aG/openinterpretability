"""The ledger of benchmark runs: one row per run of the Benchmarks workflow (issue #4011).

The workflow (``.github/workflows/benchmarks.yml``) measures memory
(``scripts/memory_benchmark.py``) and time (``scripts/perf_bench.py``). Then:

* ``check`` compares the run with the median of the last :data:`WINDOW` rows from the same
  runner. It exits 1 if a measure grew past its tolerance. With fewer than
  :data:`MIN_HISTORY` earlier rows it only reports.
* ``append`` adds the run to :data:`LEDGER`. The workflow commits the ledger through a
  pull request that merges by itself, so the library's performance is on record through
  time.

Rows count as the same runner when the OS, the architecture, the CPU count and the
device are the same. The runner's kernel and image change often, so they don't count.

    python scripts/benchmark_ledger.py check --memory memory.json --perf perf.json
    python scripts/benchmark_ledger.py append --memory memory.json --perf perf.json \\
        --commit "$GITHUB_SHA" --run "$GITHUB_RUN_ID"
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

LEDGER = Path("docs/analyses/benchmark-ledger.jsonl")
WINDOW = 7
MIN_HISTORY = 3
# Times vary by about a third between runs, peaks by a few percent (docs/analyses/perf.md).
TOLERANCE = {"_s": 1.5, "_mb": 1.25}


def runner(perf: dict) -> dict:
    """The fields that decide whether two runs are comparable, from perf's machine."""
    m = perf["machine"]
    system = m.get("platform", "").split("-")[0]
    return {"system": system, "arch": m.get("processor"), "cpus": m.get("cpus"),
            "device": m.get("device")}  # fmt: skip


def measures(memory: dict, perf: dict) -> dict[str, float]:
    """Every measure of a run, flat: ``memory/<model>/<field>`` and ``perf/<workload>/<field>``."""
    out = {}
    for model, fields in memory["models"].items():
        for field, value in fields.items():
            out[f"memory/{model}/{field}"] = value
    for workload, fields in perf["workloads"].items():
        for field, value in fields.items():  # wall_s, cpu_s, peak_mb
            out[f"perf/{workload}/{field}"] = value
    return out


def row(memory: dict, perf: dict, commit: str, run: str, when: datetime) -> dict:
    """The ledger row of one run."""
    return {
        "date": when.strftime("%Y-%m-%dT%H:%M:%SZ"), "commit": commit, "run": run,
        "runner": runner(perf), "machine": perf["machine"], "metric": memory.get("metric"),
        "measures": measures(memory, perf),
    }  # fmt: skip


def read(ledger: Path) -> list[dict]:
    """The ledger's rows, oldest first; none if there is no ledger yet."""
    if not ledger.exists():
        return []
    return [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line]


def tolerance(name: str) -> float:
    """The tolerance of a measure, by its unit (``_s`` or ``_mb``)."""
    return TOLERANCE["_s"] if name.endswith("_s") else TOLERANCE["_mb"]


def check(now: dict, history: list[dict]) -> tuple[list[str], list[str]]:
    """(regressions, notes): each measure past its tolerance times the recent median."""
    same = [r for r in history if r["runner"] == now["runner"]][-WINDOW:]
    if len(same) < MIN_HISTORY:
        return [], [f"{len(same)} earlier runs on this runner: not compared yet"]
    regressions = []
    for name, value in now["measures"].items():
        before = [r["measures"][name] for r in same if name in r["measures"]]
        if len(before) < MIN_HISTORY:
            continue
        median = statistics.median(before)
        if median > 0 and value > tolerance(name) * median:
            regressions.append(
                f"{name}: {value:g} against a median of {median:g} over {len(before)} runs "
                f"({value / median:.2f}x, past {tolerance(name):g}x)"
            )
    return regressions, [f"compared with the median of {len(same)} runs on this runner"]


def main(argv: list[str] | None = None) -> int:
    """``check`` or ``append`` one run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["check", "append"])
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--perf", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--commit", default="")
    parser.add_argument("--run", default="")
    args = parser.parse_args(argv)
    memory = json.loads(args.memory.read_text())
    perf = json.loads(args.perf.read_text())
    now = row(memory, perf, args.commit, args.run, datetime.now(UTC))
    if args.action == "append":
        with args.ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps(now, sort_keys=True) + "\n")
        print(f"Appended run {args.run or '?'} to {args.ledger}")
        return 0
    regressions, notes = check(now, read(args.ledger))
    for note in notes:
        print(note)
    for line in regressions:
        print(f"regression: {line}")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
