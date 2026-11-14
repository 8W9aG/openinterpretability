"""Measure peak memory of a scan and a review per model, and compare with the baseline.

The release skill runs it, so a change that makes a scan or a review use more memory is
caught before a release. For each model it runs a 20,000-token ``open-interpretability audit``
scan of one fixed unit. Then it runs ``open-interpretability reproduce`` of one record of that
unit. Times are printed, not compared: they vary too much with other load.

On macOS it compares each command's peak memory footprint (``/usr/bin/time -l``). Peak
resident memory (RSS) moved by more than 1 GB between runs of the same review, and the
footprint by 1%. RSS also counts shared libraries and the mapped checkpoint. Elsewhere it
compares peak RSS (``os.wait4``), the one figure there is on every system.

    uv run python scripts/memory_benchmark.py            # compare with the baseline
    uv run python scripts/memory_benchmark.py --update   # write a new baseline

It exits 1 if a peak is more than ``--tolerance`` above the baseline. A peak more than
that below the baseline is reported, so the baseline can be lowered with ``--update``.
Peaks depend on the platform, so the baseline records it, and a run on another platform
only reports. See docs/experiments.md, Memory.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "scripts" / "memory_baseline.json"
MAX_TOKENS = 20_000


@dataclass(frozen=True)
class Benchmark:
    """One model's benchmark: a unit to scan, and a record of that unit to reproduce."""

    model: str
    layer: int
    neuron: int
    pole: str
    record: str  # under wiki/experiments


BENCHMARKS = (
    Benchmark(
        "openai-community/gpt2",
        9,
        340,
        "positive",
        "openai-community--gpt2/L9/N340-positive/20261109T162236Z.json",
    ),
    Benchmark(
        "Qwen/Qwen2.5-0.5B",
        6,
        794,
        "negative",
        "Qwen--Qwen2.5-0.5B/L6/N794-negative/20261109T132651Z.json",
    ),
)


def metric() -> str:
    """The peak this platform compares: ``footprint`` on macOS, else ``rss``."""
    return "footprint" if sys.platform == "darwin" else "rss"


def run_peak(command: list[str]) -> tuple[int, float]:
    """Run ``command``; return its peak memory in MB (:func:`metric`) and its time in s."""
    if metric() == "footprint":
        command = ["/usr/bin/time", "-l", *command]
    start = time.perf_counter()
    with tempfile.TemporaryFile() as err:  # a file, not a pipe, so a long log can't block
        proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=err)
        _, status, usage = os.wait4(proc.pid, 0)
        err.seek(0)
        log = err.read().decode(errors="replace")
    if os.waitstatus_to_exitcode(status) != 0:
        sys.exit(f"failed: {' '.join(command)}\n{log[-2000:]}")
    if metric() == "footprint":
        line = next(x for x in log.splitlines() if x.strip().endswith("peak memory footprint"))
        peak = int(line.split()[0]) / 2**20
    else:
        peak = usage.ru_maxrss / 2**10  # KB on Linux
    return round(peak), round(time.perf_counter() - start, 1)


def window_args(model: str) -> list[str]:
    """``--window-tokens`` for a model that ``[tool.openinterp.window_tokens]`` sets."""
    sys.path.insert(0, str(ROOT))
    from openinterp.planner import configured_windows

    window = configured_windows(ROOT / "pyproject.toml").get(model)
    return ["--window-tokens", str(window)] if window else []


def measure(bench: Benchmark, device: str) -> dict:
    """Peak memory (MB) and time (s) of ``bench``'s scan and review."""
    cli = [str(Path(sys.executable).parent / "openinterp")]
    with tempfile.TemporaryDirectory() as tmp:
        scan = [
            *cli, "audit", "--model", bench.model, "--layer", str(bench.layer),
            "--neuron", str(bench.neuron), "--pole", bench.pole, "--device", device,
            "--max-tokens", str(MAX_TOKENS), "--no-scan-cache",
            "--evidence-out", str(Path(tmp) / "ev.json"), *window_args(bench.model),
        ]  # fmt: skip
        scan_mb, scan_s = run_peak(scan)
    record = ROOT / "wiki" / "experiments" / bench.record
    review = [*cli, "reproduce", "--include-signed", "--device", device, str(record)]
    review_mb, review_s = run_peak(review)
    return {"scan_mb": scan_mb, "scan_s": scan_s, "review_mb": review_mb, "review_s": review_s}


def compare(baseline: dict, now: dict, tolerance: float) -> tuple[list[str], list[str]]:
    """Return (peaks above ``baseline`` by more than ``tolerance``, peaks that far below)."""
    above, below = [], []
    for model, peaks in now.items():
        for key in ("scan_mb", "review_mb"):
            old = baseline.get(model, {}).get(key)
            if not old:
                continue
            line = f"{model} {key}: {peaks[key]} MB, baseline {old} MB"
            if peaks[key] > old * (1 + tolerance):
                above.append(line)
            elif peaks[key] < old * (1 - tolerance):
                below.append(line)
    return above, below


def platform_name() -> str:
    """The platform a baseline is valid on, e.g. ``Darwin-arm64``."""
    return f"{platform.system()}-{platform.machine()}"


def main(argv: list[str] | None = None) -> int:
    """Measure every benchmark, then write the baseline or compare with it."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--update", action="store_true", help="write a new baseline")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tolerance", type=float, default=0.25)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args(argv)

    now = {}
    for bench in BENCHMARKS:
        now[bench.model] = measure(bench, args.device)
        m = now[bench.model]
        print(f"{bench.model}: scan {m['scan_mb']} MB, {m['scan_s']} s · "
              f"review {m['review_mb']} MB, {m['review_s']} s")  # fmt: skip
    if args.update:
        data = {
            "platform": platform_name(),
            "metric": metric(),
            "device": args.device,
            "max_tokens": MAX_TOKENS,
        }
        args.baseline.write_text(json.dumps({**data, "models": now}, indent=2) + "\n")
        print(f"Wrote {args.baseline}")
        return 0
    base = json.loads(args.baseline.read_text())
    same = (base.get("platform"), base.get("metric"), base.get("device"))
    if same != (platform_name(), metric(), args.device):
        print(f"The baseline is for {base.get('platform')} on {base.get('device')}: not compared.")
        return 0
    worse, better = compare(base["models"], now, args.tolerance)
    for line in better:
        print(f"Lower than the baseline, which --update would record: {line}")
    for line in worse:
        print(f"Above the baseline by more than {args.tolerance:.0%}: {line}")
    return 1 if worse else 0


if __name__ == "__main__":
    raise SystemExit(main())
