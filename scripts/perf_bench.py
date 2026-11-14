"""Wall time, CPU time and peak memory of fixed workloads (issue #3265, docs/analyses/perf.md).

Each workload is an ``openinterp`` command on GPT-2 small, with no LLM call. It runs in a
child process under ``/usr/bin/time``, so the measurement doesn't change the code that it
measures. The peak memory is the *peak memory footprint* on macOS, and the maximum
resident set size elsewhere. On macOS the resident set also counts the clean pages of
memory-mapped weight files. So there the footprint is the better measure of memory
pressure (#3262).

``--write`` saves the results and the machine to the baseline file. Without it, the script
compares the run with the baseline, and warns where a measure grows past its ``TOLERANCE`` times
its baseline. It fails only with ``--strict``. Timings from another machine or device only
warn, because they aren't comparable.

    make perf                                   # run, and check against the baseline
    uv run python scripts/perf_bench.py --write # run, and replace the baseline
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BASELINE = Path("docs/analyses/perf-baseline.json")
# A measure past this many times its baseline is a regression. Times vary by about a
# third from run to run on one machine, and peak memory by about 2% (docs/analyses/perf.md).
TOLERANCE = {"wall_s": 1.5, "cpu_s": 1.5, "peak_mb": 1.2}
MEASURES = ("wall_s", "cpu_s", "peak_mb")
# One fixed pole and one fixed record of GPT-2 small, so that every run does the same work.
SCAN_POLE = ("openai-community/gpt2", 5, 1019, "positive")
RECORD = "wiki/experiments/openai-community--gpt2/L5/N1019-positive/20261106T190528Z.json"
SCAN_TOKENS = 20_000
CLI = "from openinterp.cli import main; main()"
MB = 1024 * 1024


def workloads(out: Path, device: str, build: bool) -> dict[str, list[str]]:
    """``{name: openinterp arguments}`` for each workload, writing only under ``out``."""
    model, layer, neuron, pole = SCAN_POLE
    jobs = {
        "scan": [
            *("audit", "--model", model, "--layer", str(layer), "--neuron", str(neuron)),
            *("--pole", pole, "--corpus", "corpus.jsonl", "--max-tokens", str(SCAN_TOKENS)),
            *("--window-tokens", "512"),  # GPT-2 reads at most 1,024 tokens
            *("--device", device, "--evidence-out", str(out / "ev.json")),
        ],
        "reproduce": ["reproduce", "--include-signed", "--device", device, RECORD],
    }
    if build:
        jobs["build"] = ["build", "--output-dir", str(out / "site")]
    return jobs


def parse_time(stderr: str) -> dict[str, float]:
    """The measures from the output of BSD ``time -l`` or GNU ``time -v``."""
    bsd = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", stderr)
    if bsd:
        real, user, system = map(float, bsd.groups())
        footprint = re.search(r"(\d+)\s+peak memory footprint", stderr)
        rss = re.search(r"(\d+)\s+maximum resident set size", stderr)
        peak = int((footprint or rss).group(1))  # bytes
        return {"wall_s": real, "cpu_s": user + system, "peak_mb": peak / MB}
    wall = re.search(
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (?:(\d+):)?(\d+):([\d.]+)", stderr
    )
    user = re.search(r"User time \(seconds\): ([\d.]+)", stderr)
    system = re.search(r"System time \(seconds\): ([\d.]+)", stderr)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", stderr)
    if not (wall and user and system and rss):
        raise ValueError("no time measurements in the output")
    hours, minutes, seconds = wall.groups()
    real = int(hours or 0) * 3600 + int(minutes) * 60 + float(seconds)
    cpu = float(user.group(1)) + float(system.group(1))
    return {"wall_s": real, "cpu_s": cpu, "peak_mb": int(rss.group(1)) * 1024 / MB}


def measure(args: list[str]) -> dict[str, float]:
    """Run ``openinterp <args>`` under ``/usr/bin/time`` and return its measures."""
    flag = "-l" if sys.platform == "darwin" else "-v"
    command = ["/usr/bin/time", flag, sys.executable, "-c", CLI, *args]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"openinterp {args[0]} failed:\n{done.stderr[-2000:]}")
    return {k: round(v, 2) for k, v in parse_time(done.stderr).items()}


def machine(device: str) -> dict[str, str | int]:
    """What the timings depend on, besides the code."""
    import torch

    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpus": os.cpu_count() or 0,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
    }


def compare(results: dict, baseline: dict, tolerance: dict = TOLERANCE) -> list[str]:
    """One line per measure past its ``tolerance`` times its baseline, or per changed machine."""
    problems = []
    if results["machine"] != baseline["machine"]:
        changed = sorted(
            k for k in results["machine"] if results["machine"][k] != baseline["machine"].get(k)
        )
        problems.append(f"machine differs from the baseline's ({', '.join(changed)})")
    for name, now in results["workloads"].items():
        before = baseline["workloads"].get(name)
        if before is None:
            continue
        for key in MEASURES:
            if before[key] > 0 and now[key] > tolerance[key] * before[key]:
                problems.append(
                    f"{name} {key}: {now[key]:g} against {before[key]:g} "
                    f"({now[key] / before[key]:.2f}x, past {tolerance[key]:g}x)"
                )
    return problems


def main() -> int:
    """Run the workloads, then write or check the baseline."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu", help="cpu (default), mps, cuda or auto.")
    parser.add_argument("--build", action="store_true", help="Also time `openinterp build`.")
    parser.add_argument("--write", action="store_true", help=f"Replace {BASELINE}.")
    parser.add_argument("--strict", action="store_true", help="Exit 1 on a regression.")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    opts = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        jobs = workloads(Path(tmp), opts.device, opts.build)
        results = {"machine": machine(opts.device), "workloads": {}}
        for name, args in jobs.items():
            results["workloads"][name] = measure(args)
            print(f"{name}: {results['workloads'][name]}")
    if opts.write:
        opts.baseline.write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {opts.baseline}")
        return 0
    baseline = json.loads(opts.baseline.read_text())
    problems = compare(results, baseline)
    for line in problems:
        print(f"warning: {line}")
    if not problems:
        print(f"Within tolerance of {opts.baseline}.")
    regressions = [p for p in problems if not p.startswith("machine")]
    return 1 if opts.strict and regressions else 0


if __name__ == "__main__":
    sys.exit(main())
