# Performance baseline

Issue #3265 asks for performance tests on memory, compute and wall time. `scripts/perf_bench.py` measures fixed workloads and compares them with `perf-baseline.json`. Run it with `make perf`, which takes about 90 seconds. Reported, not gated.

## What it measures

Each workload is one `open-interpretability` command on GPT-2 small, with no LLM call:

| Workload | Command | What it stands for |
| --- | --- | --- |
| scan | `audit --evidence-out` on L5/N1019 positive, over a 20,000-token sample of `corpus.jsonl`, in 512-token windows | The scan of an explore draw, at a tenth of its default 200,000 tokens |
| reproduce | `reproduce --include-signed` of one fixed L5/N1019 record | A review's measurements: scan evidence, counterfactuals and probes, as CI's reproduction runs them |
| build (`--build` only) | `build` of the whole site | The site build that CI and a release run |

Each command runs in a child process under `/usr/bin/time`, so the measurement doesn't change the code that it measures. Three measures come back:

- **Wall time** in seconds.
- **CPU time** in seconds: user plus system. Above wall time, it shows how many cores a workload uses.
- **Peak memory** in MB. On macOS this is the *peak memory footprint*, and on Linux the maximum resident set size. On macOS the resident set also counts the clean pages of memory-mapped weight files. In #3262 it moved the other way from the footprint, so the footprint is the measure of memory pressure.

The workloads run on the CPU by default, because the GPU's speed depends on what else is using it. `--device mps` or `--device cuda` measures a GPU instead, against a baseline written on that device.

## Baseline

Written on 2026-11-14 on an Apple-silicon Mac with 10 cores, macOS 15.6, Python 3.12 and torch 2.14.1, on the CPU:

| Workload | Wall time | CPU time | Peak memory |
| --- | --- | --- | --- |
| scan | 57 s | 130 s | 1,390 MB |
| reproduce | 30 s | 52 s | 1,325 MB |
| build | 58 s | 157 s | 3,979 MB |

The build's time is with its on-disk memo warm (`openinterp/builder/disk_memo.py`), as on a machine that has built the site before. A first build takes longer.

**Noise.** Three runs of scan and reproduce gave wall times of 56–58 s and 23–31 s, and peak memory within 2% of each other. So a run warns at 1.5 times the baseline for wall and CPU time, and at 1.2 times for peak memory (`TOLERANCE`). A smaller regression in time needs several runs to see.

## How to use it

- **Before and after a change that could cost time or memory:** run `make perf` on both commits on one machine. It prints each measure and a warning for each one past its tolerance.
- **A new baseline:** `uv run python scripts/perf_bench.py --write --build`, on a machine with nothing else running. A baseline from another machine or device gets a warning only, because its timings aren't comparable.
- **In a script:** `--strict` exits 1 on a regression.

`make perf` takes the heavy lock (`openinterp/core/heavy_lock.py`) when it is turned on, so another heavy job can't overlap the timings. It doesn't run in CI. Runner timings vary between runs, and CI has no `corpus.jsonl`.

## Limits

- **One pole and one record.** A change that affects only another kind of unit, such as an attention head or a dictionary latent, isn't measured.
- **GPT-2 small only.** Costs that grow with the model's size, such as Qwen2.5-1.5B's memory, are not measured. The sister search's own measurements are in `sister-neurons.md`.
- **The LLM agents aren't timed.** On 2026-10-20 an explore draw took 25–40 minutes, of which the CPU scan was 13–15 (#3265). Most of the rest is the proposer and the reviewer.
- **CPU options.** Threads, half precision, a fused GELU and compiling to C++ are compared in `cpu-acceleration.md` (#3268).
