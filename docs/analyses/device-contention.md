# Jobs that share Apple's GPU, the ANE and the CPU

Run on 2026-11-12 on an M1 Max (64 GB) with macOS 15. It sets the device planner's GPU threshold (`openinterp/harness/devices.py`). It also tests the planner's order: MPS, then Apple's Neural Engine (ANE), then the CPU. Raw results are in `device-contention.jsonl`, one line per mix of jobs. The script is `scripts/device_contention.py`.

## Method

- **Jobs.** Each job is its own process. It runs scan-sized forward passes (2,048 tokens, to one layer's MLP activations) for 40 s, and reports its passes per second. An `mps` or `cpu` job runs the torch pass of a scan (`NeuronHookRunner.layer_activations`). An `ane` job runs the scan's pass on the ANE (`AneScan`, from the Core ML cache). The models are GPT-2 L5 and Qwen2.5-0.5B L12, in fp32 for torch.
- **Mixes.** The jobs of a mix start loading 8 s apart and run their passes at the same time.
- **Utilization.** During each mix, the script reads the GPU's utilization as the planner does: the mean of 5 `ioreg` readings over about 2 s, again and again.
- **Same numbers.** Each job measures one fixed batch before and after its passes. A job's two results must be equal, and equal to a job alone on that device.
- **A quiet machine.** Experiment loops ran on this machine at the same time. So the script recorded every other `openinterp` command that ran during a mix, and ran that mix again. Each result here is from a mix during which no other `openinterp` command ran. Some mixes took three attempts.

## Results

Passes per second of each job. *GPU* is the mean of the planner's readings during the mix.

| Jobs | GPT-2: each job | GPT-2: total | GPU | Qwen 0.5B: each job | Qwen 0.5B: total | GPU |
| --- | --- | --- | --- | --- | --- | --- |
| None | – | – | 1% | – | – | 1% |
| 1 MPS | 29.4 | 29.4 | 94% | 7.63 | 7.63 | 97% |
| 2 MPS | 15.6, 15.7 | 31.3 | 99% | 3.54, 3.47 | 7.01 | 100% |
| 4 MPS | 7.6 each | 30.3 | 100% | 1.9 each | 7.52 | 100% |
| 1 ANE | 15.0 | 15.0 | 1% | 4.60 | 4.60 | 1% |
| 1 MPS + 1 ANE | 28.2 (MPS), 13.8 (ANE) | 42.0 | 93% | 7.70 (MPS), 4.55 (ANE) | 12.3 | 96% |
| 1 CPU | 5.4 | 5.4 | 0% | 1.47 | 1.47 | 1% |
| 1 MPS + 1 CPU | 27.7 (MPS), 4.7 (CPU) | 32.4 | 93% | 7.26 (MPS), 1.44 (CPU) | 8.70 | 96% |

- **The GPU divides evenly between jobs.** Two jobs on MPS each get half of one job's speed, and four get a quarter. The total does not change. So a second job on the GPU halves the first job's speed and adds nothing.
- **The ANE adds a second lane.** A job on the ANE beside a job on MPS barely slows either one. The MPS job keeps 96% (GPT-2) and 101% (Qwen) of its speed alone. The total goes up by 43% (GPT-2) and 61% (Qwen) over one job alone. Compared with two jobs on MPS, it goes up by 34% and 75%.
- **The CPU is the slowest lane.** A job on the CPU runs at 18–19% of a free GPU's speed. But it slows the MPS job by only 5–6%.
- **The planner's order matches this.** A second job is best on the ANE. That gives the most work per second. The first job keeps nearly all its speed. The CPU comes after the ANE, as a lane that doesn't slow the GPU.
- **The numbers did not change.** In every mix, each job's fixed batch gave the same result as a job alone on that device. It also gave the same result before and after the job's passes.

### The GPU threshold

With no job, the planner's mean reading was 0.6% to 1.2%. Single readings got to 21%. With one job on MPS, the mean was 93% to 97%, and no reading was less than 91%. An ANE or CPU job alone gave 0% to 1.2%, so a job on the ANE does not look like GPU load. The planner's threshold of 50% is far from both. A job that the leases do not show, from another program, makes the GPU busy at 91% or more.

## A hang when processes start on MPS together

In the first attempt, four MPS jobs started at the same moment. Two of them stopped for more than 35 minutes in their first copy to MPS. Their stacks were in Metal's compiler (`MTLCompilerFSCache::openSync`). That function opens Metal's shared cache of compiled GPU programs on disk. Jobs that start 8 s apart did not stop. Any `openinterp` command that starts on MPS at the same time as another can meet this. A guard against it is a separate change.

## Also found

The first `ane` jobs stopped with a segmentation fault. When a Core ML model is idle for about 0.5 s, Core ML resets its execution stream on a background thread. Then coremltools 9.0 frees the last input without the interpreter's lock. `openinterp/harness/ane/convert.py` now keeps each input array until the process stops. `tests/test_ane.py` has a test that fails with exit code −11 without that change.
