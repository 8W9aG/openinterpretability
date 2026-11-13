"""Measure how jobs that share Apple's GPU, the ANE and the CPU slow each other down.

The device planner (``openinterp/harness/devices.py``) sends a job to the ANE or the
CPU when the GPU is busy. This script measures whether that helps, and sets the GPU
utilization at which the GPU counts as busy. For each *mix* of jobs it starts one
process per job at the same moment. Each job runs scan-sized forward passes (2,048
tokens, up to one layer's MLP activations) for ``--seconds``:

- ``mps`` and ``cpu``: the torch pass of a scan (``NeuronHookRunner.layer_activations``);
- ``ane``: the scan's pass on the ANE (``AneScan``), from the Core ML cache.

Each job reports its passes per second. Each job also measures one fixed batch at the
start and at the end. A job whose two results differ, or differ from a job alone on
that device, shows that sharing a device changed the numbers. While a mix runs, the
script reads the GPU's utilization as the planner does. Results go to
docs/analyses/device-contention.md.

    uv run python scripts/device_contention.py openai-community/gpt2 5 --out results.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

MIXES = [
    [],  # idle: the GPU's utilization with no job
    ["mps"],
    ["mps", "mps"],
    ["mps", "mps", "mps", "mps"],
    ["ane"],
    ["mps", "ane"],
    ["cpu"],
    ["mps", "cpu"],
]
ROWS, SEQ = 8, 256  # 2,048 tokens, as BATCH_TOKENS
STAGGER_SECONDS = 8  # between the starts of a mix's jobs
GRACE_SECONDS = 120  # after the end of a mix, before a job counts as hung


def worker(args) -> None:
    """Run one job until ``args.stop_at``; print its result as one JSON line."""
    import torch

    time.sleep(args.delay)  # start jobs apart: fresh processes on MPS can hang in Metal's cache
    run, batch = _job(args)
    check = torch.randint(0, 50000, (ROWS, SEQ), generator=torch.Generator().manual_seed(9))
    first = run(check)
    while time.time() < args.start_at:
        time.sleep(0.01)
    passes, start = 0, time.perf_counter()
    while time.time() < args.stop_at:
        run(batch)
        passes += 1
    seconds = time.perf_counter() - start
    last = run(check)
    result = json.dumps(
        {
            "device": args.device,
            "passes": passes,
            "seconds": seconds,
            "per_second": passes / seconds,
            "check": first.double().sum().item(),
            "drift": float((first - last).abs().max()),
            "check_values": first.flatten()[:: max(1, first.numel() // 256)].tolist(),
        }
    )
    Path(args.result).write_text(result)


def _job(args):
    """``run(ids) -> acts`` on the job's device, and one batch of token ids."""
    import torch

    from openinterp.cli import _load_runner

    device = "cpu" if args.device == "ane" else args.device
    runner = _load_runner(args.model, device, "float32")
    vocab = int(runner.model.config.vocab_size)
    batch = torch.randint(0, vocab, (ROWS, SEQ), generator=torch.Generator().manual_seed(8))
    if args.device == "ane":
        from openinterp.harness.ane import convert
        from openinterp.harness.ane.backend import AneScan

        ane = AneScan.for_model(runner.model, args.model, runner.model_revision)
        model = ane.compiled(args.layer, ROWS, SEQ)
        return (lambda ids: convert.predict(model, ids)), batch

    def run(ids):
        with torch.no_grad():
            acts = runner.layer_activations({"input_ids": ids.to(runner.device)}, [args.layer])
        out = acts[args.layer].float()
        if out.device.type == "mps":
            torch.mps.synchronize()
        return out.cpu()

    return run, batch


def run_mix(model: str, layer: int, mix: list[str], seconds: float, warmup: float) -> dict:
    """Start one process per job in ``mix``, all timed together, and gather their results."""
    import tempfile

    from openinterp.harness.devices import gpu_utilization

    start_at = time.time() + warmup
    stop_at = start_at + seconds
    work = Path(tempfile.mkdtemp(prefix="contention-"))
    jobs = [_start(model, layer, d, i, start_at, stop_at, work) for i, d in enumerate(mix)]
    while time.time() < start_at + 2:
        time.sleep(0.2)
    readings, others = [], set()
    while time.time() < stop_at - 2:
        readings.append(gpu_utilization())
        others |= other_jobs()
    return {
        "model": model,
        "layer": layer,
        "mix": mix,
        "gpu_percent": readings,
        "jobs": _collect(jobs, mix, work, stop_at),
        "other_jobs": sorted(others),  # openinterp commands outside this script; skews a mix
    }


def _start(model, layer, device, i, start_at, stop_at, work) -> subprocess.Popen:
    """Start job ``i`` of a mix. Each job starts :data:`STAGGER_SECONDS` after the one before."""
    times = ["--start-at", str(start_at), "--stop-at", str(stop_at)]
    rest = ["--delay", str(STAGGER_SECONDS * i), "--result", str(work / f"{i}.json")]
    command = [sys.executable, __file__, "worker", model, str(layer), "--device", device]
    return subprocess.Popen(
        [*command, *times, *rest], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def _collect(jobs, mix, work: Path, stop_at: float) -> list[dict]:
    """Each job's result. A job that has not ended :data:`GRACE_SECONDS` after the stop hung."""
    results = []
    for i, (job, device) in enumerate(zip(jobs, mix, strict=True)):
        try:
            job.wait(timeout=max(1.0, stop_at + GRACE_SECONDS - time.time()))
            results.append(json.loads((work / f"{i}.json").read_text()))
        except (subprocess.TimeoutExpired, OSError, ValueError):
            job.kill()
            results.append({"device": device, "hung": True, "returncode": job.poll()})
    return results


def other_jobs() -> set[str]:
    """``openinterp`` commands that run now, outside this script."""
    found = subprocess.run(["pgrep", "-fl", "openinterp "], capture_output=True, text=True)
    return {
        " ".join(line.split()[1:4])
        for line in found.stdout.splitlines()
        if "device_contention" not in line and "pgrep" not in line
    }


def main() -> None:
    """Run every mix for one model and layer, and write one JSON line per mix."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", nargs="?", default="run", choices=["run", "worker"])
    parser.add_argument("model")
    parser.add_argument("layer", type=int)
    parser.add_argument("--device")
    parser.add_argument("--start-at", type=float)
    parser.add_argument("--stop-at", type=float)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--result")
    parser.add_argument("--seconds", type=float, default=40.0)
    parser.add_argument("--warmup", type=float, default=60.0, help="time to load each job")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--mixes", help="comma-separated indexes of MIXES to run (default: all)")
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args)
        return
    chosen = [MIXES[int(i)] for i in args.mixes.split(",")] if args.mixes else MIXES
    for mix in chosen:
        result = run_mix(args.model, args.layer, mix, args.seconds, args.warmup)
        line = json.dumps(result)
        print(line, flush=True)
        if args.out:
            with args.out.open("a") as handle:
                handle.write(line + "\n")


if __name__ == "__main__":
    main()
