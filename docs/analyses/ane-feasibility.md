# Can scans run on Apple's Neural Engine?

Run on 2026-11-12 on an M1 Max (64 GB) with macOS 15. It answers the Performance TODO item "Can we run any of this on Apple's Neural Engine (ANE)?". Raw results are in `ane-feasibility.jsonl`, one line per model. The script is `scripts/ane_benchmark.py`.

**Answer: it can, but it is slower than MPS.** As first converted, the ANE gave Qwen2.5 activations with no relation to fp32. Three changes to the model's code make it as accurate as any other fp16 backend (see "Making the ANE accurate"). Even so, a forward pass takes 1.5–1.6× as long as on MPS. And on Qwen, fp16 on any device is often outside the tolerance of `openinterp reproduce`. A scan can still record exact values. The ANE ranks every token, and fp32 measures each pole's top 40 again (see "Screen on the ANE, then measure the top exactly").

## Method

- **Workload.** One scan-sized forward pass: a batch of 8 × 256 random token ids (2,048 tokens, as `BATCH_TOKENS` in `openinterp/harness/scan.py`), up to one layer's MLP activations. The layers are GPT-2 small L5 (width 3,072), Qwen2.5-0.5B L12 (width 4,864) and Qwen2.5-1.5B L8 (width 8,960).
- **Access to the ANE.** PyTorch cannot use the ANE. The only way is Core ML, so the script traces each model and converts it with coremltools 9.0 to fp16, the ANE's only float type. Core ML then picks a device for each operation when given a compute-unit setting. The run tries all four settings.
- **A second copy of the forward pass.** coremltools can't convert the HF modelling code. transformers 5's mask code uses `torch.diff`, which has no converter, and 4.46's code converts shapes to Python ints, which fails. So the script runs its own copy of GPT-2 and Qwen2 on the HF weights. In fp32 it agrees with HF to a relative error of 2–8 × 10⁻⁶.
- **Two changes keep fp16 finite on Qwen2.5,** on every device:
  - The residual stream gets to about 6,600 (1.5B) and 1,700 (0.5B). RMSNorm squares it (6,600² ≈ 4 × 10⁷, but fp16's largest value is 65,504). HF's RMSNorm moves to fp32 for this step, which the ANE can't do. The copy divides each row by its largest value first. That is exact.
  - The k projections have large biases (|k| up to 220), so q·k gets to about 10⁵ before the 1/√d scale. The copy scales q first.
- **Agreement.** The script compares each result with the fp32 CPU activations. It reports three figures. They are the relative RMS error over all values, the worst Pearson correlation over 64 neurons, and the overlap of neuron 0's top 20 tokens. The 64 neurons are spread across the layer. A separate check uses the tolerance of `openinterp reproduce`: 1e-3 × the neuron's peak + 1e-2 × the value.
- **Timing.** The median of 5 runs after a warm-up, with no other jobs on the machine. Timings changed by up to 2× between runs. Read them as rough.

## Making the ANE accurate

We found each error by bisection. First, a model that returns every intermediate tensor shows where the ANE's error starts. Then each operation runs alone on the exact fp32 inputs of a real forward pass, on the ANE and on the GPU.

1. **`pow` is wrong in a chain.** The first error was in layer 0's RMSNorm, whose output was about 5× too small. `pow(x, 2)`, `reduce_mean` and `rsqrt` were each correct alone, but the chain `x * rsqrt(mean(pow(x, 2)))` was wrong. Its error depended on the scale of `x`: 0.98 at the embedding's scale, 0.06 at 64×. The converted program is only `pow → reduce_mean → rsqrt → mul`, so the ANE's compiler causes the error, not coremltools. With `x * x` the error is 0.0005. GPT-2's GELU uses `pow(x, 3)`, which becomes `x * x * x`.
2. **`silu`, `sigmoid` and `tanh` are coarse approximations.** On the inputs of Qwen2.5-0.5B's layer 12, `silu(g) * u` had an error of 0.021, 58× the GPU's. `sigmoid` alone had 0.0034, and the `tanh` form of sigmoid 0.0014. `g * u / (1 + exp(-g))` has 0.0006. GPT-2's GELU uses the same identity, `0.5 · (1 + tanh(z)) = 1 / (1 + exp(−2z))`. If `exp` overflows for a very negative input, the result is 0, the correct limit.
3. **A matmul loses precision when its products are small.** On random data, weights of RMS 0.001 gave 14× the GPU's error, and 0.1 or more gave the same error. The length of the dot product had no effect. Qwen's `o_proj` (weight RMS 0.015, input RMS 0.18) had 6× the GPU's error. Multiplying the weights by 2⁴ and dividing the output by 2⁴ made them equal. A power of two changes only the exponent, so this is exact. The copy scales every projection so that its weights' RMS is about 0.25. One fp32 pass caps the scale so that no scaled output is above 16,384.

Relative RMS error of the MLP activations on the ANE:

| Changes | GPT-2 L5 | Qwen 0.5B L12 | Qwen 1.5B L8 |
| --- | --- | --- | --- |
| None | 0.012 | 1.00 | 1.00 |
| 1 | 0.005 | 0.040 | 0.031 |
| 1, 2 and 3 | 0.003 | 0.008 | 0.011 |
| Core ML on the GPU, for comparison | 0.002 | 0.008 | 0.010 |

The tolerance check below also shows the effect of 1 and 2 without 3.

## Results

Milliseconds per 2,048-token batch, with all three changes. *Rel. err* is the relative RMS error against fp32.

| Backend | GPT-2 L5 | rel. err | Qwen 0.5B L12 | rel. err | Qwen 1.5B L8 | rel. err |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch CPU, fp32 | 1,171 | – | 953 | – | 1,412 | – |
| PyTorch MPS, fp32 | 41 | – | 147 | – | 314 | – |
| PyTorch MPS, fp16 | 34 | 0.002 | 131 | 0.008 | 230 | 0.011 |
| Core ML, CPU only | 198 | 0.017 | 541 | 0.037 | 1,254 | 0.044 |
| Core ML, CPU + GPU | 39 | 0.002 | 115 | 0.008 | 222 | 0.010 |
| Core ML, CPU + ANE | 67 | 0.003 | 218 | 0.008 | 504 | 0.011 |
| Core ML, all units | 98 | 0.002 | 353 | 0.008 | 2,426 | 0.010 |

- **The ANE is slower than MPS.** It takes 1.6× as long as MPS fp32 on GPT-2, 1.5× on Qwen2.5-0.5B and 1.6× on Qwen2.5-1.5B. This is not because Core ML runs part of the work elsewhere. Its plan puts all but 4 of 251 (GPT-2), 1,106 (0.5B) and 766 (1.5B) operations on the ANE. The other 4 are the embedding lookup (index checks and a gather). The three changes cost little time: before them the ANE took 51, 190 and 485 ms.
- **The CPU fp32 times vary a lot.** GPT-2 took 306–1,171 ms in different runs, so don't compare them closely.
- **"All units" is not better than "CPU + ANE".** Core ML splits the work between devices. On Qwen2.5-1.5B the copies between them made it 5× slower.
- **The start-up costs are large.** The ANE compiles the model when it loads it: 11 s (GPT-2) and 25–29 s (Qwen). The conversion takes another 5–28 s. Core ML for the ANE also needs input shapes that do not change. A scan's batches vary in shape (`plan_batches`), so it would need a model for each shape, or padding to one shape.

### Against the reproduce tolerance

`openinterp reproduce` runs a record again at the dtype it pins (`RunEnvironment.dtype`), on CI's CPU runners. So the fair reference for an fp16 record is PyTorch fp16 on the CPU. `reproduce` compares each top example's per-token strengths. As a stand-in, this check takes each neuron's top 20 tokens in the batch. A neuron *passes* if all 20 are inside the tolerance. Share of neurons that pass:

| Backend, against | GPT-2 | Qwen 0.5B | Qwen 1.5B |
| --- | --- | --- | --- |
| CPU fp16, against fp32 | 0.966 | 0.515 | 0.301 |
| MPS fp16, against CPU fp16 | 0.911 | 0.636 | 0.500 |
| Core ML GPU, against CPU fp16 | 0.939 | 0.512 | 0.165 |
| ANE with change 1, against CPU fp16 | 0.576 | 0.000 | 0.000 |
| ANE with changes 1 and 2, against CPU fp16 | 0.801 | 0.075 | 0.138 |
| ANE with all changes, against CPU fp16 | 0.792 | 0.487 | 0.144 |
| ANE with all changes, against fp32 | 0.851 | 0.499 | 0.297 |

- **On Qwen, the ANE is now as accurate as fp16 on the CPU.** Against fp32, 50% and 30% of its neurons pass, and 52% and 30% of CPU fp16's.
- **On GPT-2 a gap remains:** 85% against fp32, and 97% for CPU fp16. The next section gives what we found.
- **fp16 is the limit on Qwen, not the device.** PyTorch fp16 on MPS and on the CPU disagree with each other for 36–50% of Qwen's neurons. The tolerance suits fp32. A Qwen record made in fp16 on a Mac would often fail `reproduce` on CI's CPU runners, on any device.

These figures are from random token ids, not real text, so take them as an indication.

### The gap on GPT-2

We looked for a cause of GPT-2's gap in the same way, and found no single operation:

- Through all 6 layers, the ANE's error is 1.5–2× that of CPU fp16 at every intermediate tensor. It starts at the first LayerNorm.
- LayerNorm alone is as accurate on the ANE as on the GPU. Followed by a linear layer, its error goes up by about 1.5×, because the ANE compiler fuses the two operations. We tried four alternatives, and none was better. They were LayerNorm from basic operations, γ and β moved into the linear layer, rows divided by their largest value, and larger weights.
- No other form of GELU did better than the `exp` form. Core ML's own `gelu` operation had 8× the error. Applying GELU on the host in fp32, to the ANE's pre-activations, changed nothing. The error is already in the pre-activations.

A probable cause is how each device rounds. PyTorch's fp16 on the CPU computes each operation in fp32 and rounds only its result. The ANE computes in fp16 from start to end, so it rounds more often. We did not find a change that removes this.

### Screen on the ANE, then measure the top exactly

`reproduce` checks the top examples' strengths and the scale (the largest strength). A scan can get those exactly while the ANE does most of the work:

1. Run the whole corpus on the ANE, and rank every token by the ANE's strengths.
2. Take each pole's top K candidates. Measure them again in fp32 on the CPU or MPS, which is a few windows of work.
3. Keep the best 20 of those exact values.

Share of neurons whose 20 kept values match fp32's top 20 inside the tolerance:

| K | GPT-2 + | GPT-2 − | Qwen 0.5B + | Qwen 0.5B − | Qwen 1.5B + | Qwen 1.5B − |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 0.981 | 1.000 | 0.962 | 0.956 | 0.965 | 0.963 |
| 30 | 0.995 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 40 | 0.996 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The check compares values, not which tokens are chosen. GPT-2's top negative values are all at GELU's lowest value, −0.170, with a median gap of 2 × 10⁻⁵ between the 20th and 21st. So any fp16 device, CPU fp16 included, orders them differently, but their values agree.

Then the numbers that `reproduce` checks are fp32 numbers, and a record can pin fp32. The other scan statistics (histogram, partner correlations, confound profile) would have the ANE's fp16 error of about 1%. `reproduce` does not check them.

## Other findings

- **MPS fp16 is accurate here, given the changes for fp16 above.** Its error is 0.2–1.1%. Every sampled neuron has r ≥ 0.9997 with fp32. It is at most 1.4× faster than MPS fp32 in these runs. This bears on the "Try bf16/fp16 on MPS" item of the sister-search TODO, and so does the tolerance check above. We also ran HF's own code (transformers 4.46, SDPA attention) in fp16 on MPS on the same inputs. Its activations were finite, with errors of 0.008 (0.5B) and 0.010 (1.5B). So the overflows above are not a problem for the HF models we run.
- **The forward pass may be a small part of a scan.** Take one pass over a corpus of about 212k tokens, which is about 104 batches. At these speeds, the model's share of it on MPS is about 4 s on GPT-2 and 33 s on Qwen2.5-1.5B. The baseline in TODO.md gives 13–15 minutes for a whole scan. If both figures hold for the current scan, most of the time is outside the forward pass. Copying the whole layer to the CPU and the statistics on it are candidates. That is an estimate, not a profile. The TODO item "Can we do performance tests on memory, compute and wall time?" is where to measure it.

## What would change the answer

- **Speed.** Apple's [ml-ane-transformers](https://github.com/apple/ml-ane-transformers) rewrites a transformer for the ANE: channels-first layout, attention split per head, and LayerNorm built from ANE-friendly operations. It reports large speed-ups for BERT-sized models. That is a second implementation of every architecture, with its own correctness check per model. Unless a profile shows the forward pass is most of a scan's time, it isn't worth doing.
- **Use as a second device.** With the three changes, the ANE could run scans next to MPS. With the screen and exact measurement above, the record's checked numbers are fp32, so the record can pin fp32. Without it, a record must pin fp16, and the reproduce tolerance must suit fp16 first. Either way, the scan needs a Core ML path. That is a second copy of each model architecture, and a dependency on coremltools, which can't yet convert transformers 5's code.
