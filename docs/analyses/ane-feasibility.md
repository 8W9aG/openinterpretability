# Can scans run on Apple's Neural Engine?

Run on 2026-11-12 on an M1 Max (64 GB) with macOS 15. It answers the Performance TODO item "Can we run any of this on Apple's Neural Engine (ANE)?". Raw results are in `ane-feasibility.jsonl`, one line per model. The script is `scripts/ane_benchmark.py`.

**Answer: not usefully.** The ANE can run a scan's forward pass, but it is slower than MPS on every model we tried. It is also less accurate than any other fp16 backend. Its first results on Qwen2.5 were wrong because the ANE computes `pow` incorrectly. With `x * x` in place of `pow(x, 2)`, they are close to fp32, but not inside the tolerance of `openinterp reproduce`.

## Method

- **Workload.** One scan-sized forward pass: a batch of 8 × 256 random token ids (2,048 tokens, as `BATCH_TOKENS` in `openinterp/harness/scan.py`), up to one layer's MLP activations. The layers are GPT-2 small L5 (width 3,072), Qwen2.5-0.5B L12 (width 4,864) and Qwen2.5-1.5B L8 (width 8,960).
- **Access to the ANE.** PyTorch cannot use the ANE. The only way is Core ML, so the script traces each model and converts it with coremltools 9.0 to fp16, the ANE's only float type. Core ML then picks a device for each operation when given a compute-unit setting. The run tries all four settings.
- **A second copy of the forward pass.** coremltools can't convert the HF modelling code. transformers 5's mask code uses `torch.diff`, which has no converter, and 4.46's code converts shapes to Python ints, which fails. So the script runs its own copy of GPT-2 and Qwen2 on the HF weights. In fp32 it agrees with HF to a relative error of 2–8 × 10⁻⁶.
- **Three changes for fp16 and the ANE.**
  - The residual stream gets to about 6,600 (1.5B) and 1,700 (0.5B). RMSNorm squares it (6,600² ≈ 4 × 10⁷, but fp16's largest value is 65,504). HF's RMSNorm moves to fp32 for this step, which the ANE can't do. The copy divides each row by its largest value first. That is exact.
  - The k projections have large biases (|k| up to 220), so q·k gets to about 10⁵ before the 1/√d scale. The copy scales q first.
  - The copy squares with `x * x` and cubes with `x * x * x`, never with `pow` (see "The `pow` bug" below).
- **Agreement.** The script compares each result with the fp32 CPU activations. It reports three figures. The first is the relative RMS error over all values. The second is the worst Pearson correlation over 64 neurons spread across the layer. The third is the overlap of neuron 0's top 20 tokens. A separate check counts the values inside the tolerance of `openinterp reproduce`: 1e-3 × the neuron's peak + 1e-2 × the value.
- **Timing.** The median of 5 runs after a warm-up, with no other jobs on the machine. Timings changed by up to 2× between runs (MPS fp16 on Qwen2.5-1.5B: 217 ms, 392 ms, then 263 ms). Read them as rough.

## The `pow` bug

The first runs gave Qwen2.5 activations on the ANE that had no relation to fp32 (relative error 1.0, worst neuron r = −0.42). The same Core ML model was accurate on the GPU, so the conversion was not the cause. We bisected it:

1. A model that returns every intermediate tensor showed the first error in layer 0's RMSNorm, the first operation after the embedding. Its output was about 5× too small. Every later error followed from it.
2. Each part of RMSNorm was correct when it was the model's only output: `pow(x, 2)`, `reduce_mean` and `rsqrt`. The chain `x * rsqrt(mean(pow(x, 2)))` was wrong. Its error depended on the scale of `x`: 0.98 at the embedding's scale, 0.06 at 64×, 0.60 after division by the row's largest value.
3. The converted program is only `pow → reduce_mean → rsqrt → mul`. So the ANE's compiler, not coremltools, causes the error.
4. `x * x` in place of `pow(x, 2)` makes the error 0.0005 at every scale. `F.rms_norm` and LayerNorm on `[x, −x]` (whose mean is 0) are also correct.

GPT-2's GELU (HF `gelu_new`) uses `pow(x, 3)`. Writing it as `x * x * x` reduced GPT-2's error on the ANE from 0.012 to 0.005. We did not find out why `pow` fails only in a chain.

## Results

Milliseconds per 2,048-token batch, with `x * x` in place of `pow`. *Rel. err* is the relative RMS error against fp32.

| Backend | GPT-2 L5 | rel. err | Qwen 0.5B L12 | rel. err | Qwen 1.5B L8 | rel. err |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch CPU, fp32 | 306 | – | 848 | – | 1,791 | – |
| PyTorch MPS, fp32 | 38 | – | 129 | – | 335 | – |
| PyTorch MPS, fp16 | 32 | 0.003 | 122 | 0.008 | 263 | 0.011 |
| Core ML, CPU only | 125 | 0.017 | 551 | 0.037 | 1,341 | 0.044 |
| Core ML, CPU + GPU | 34 | 0.002 | 117 | 0.008 | 226 | 0.010 |
| Core ML, CPU + ANE | 61 | 0.005 | 188 | 0.040 | 485 | 0.031 |
| Core ML, all units | 69 | 0.003 | 340 | 0.009 | 2,368 | 0.031 |

- **The ANE is slower than MPS.** It takes 1.6× as long as MPS fp32 on GPT-2, 1.5× on Qwen2.5-0.5B and 1.4× on Qwen2.5-1.5B. This is not because Core ML runs part of the work elsewhere. Its plan puts all but 4 of 229 (GPT-2), 977 (0.5B) and 677 (1.5B) operations on the ANE. The other 4 are the embedding lookup (index checks and a gather).
- **The ANE is the least accurate fp16 backend on Qwen.** Its error is 3–4%, about the same as Core ML's fp16 on the CPU, and 3–5× that of the GPU. The worst sampled neuron has r = 0.997 (0.5B) and 0.999 (1.5B) with fp32.
- **"All units" is not better than "CPU + ANE".** Core ML splits the work between devices. On Qwen2.5-1.5B the copies between them made it 5× slower.
- **The start-up costs are large.** The ANE compiles the model when it loads it: 8 s (GPT-2) and 29–34 s (Qwen). The conversion takes another 4–31 s. Core ML for the ANE also needs input shapes that do not change. A scan's batches vary in shape (`plan_batches`), so it would need a model for each shape, or padding to one shape.

### Against the reproduce tolerance

Share of activation values inside the tolerance of `openinterp reproduce`, and share of neurons whose top token is the same as in fp32:

| Backend | GPT-2: values | top token | Qwen 0.5B: values | top token | Qwen 1.5B: values | top token |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch MPS, fp16 | 0.988 | 0.981 | 0.846 | 0.981 | 0.881 | 0.983 |
| Core ML, CPU + GPU | 0.998 | 0.994 | 0.851 | 0.980 | 0.880 | 0.983 |
| Core ML, CPU + ANE | 0.955 | 0.978 | 0.246 | 0.912 | 0.430 | 0.948 |

No fp16 backend is inside the tolerance for every value. A record made in fp16 can't be reproduced in fp32, but records pin their dtype (`RunEnvironment.dtype`), so reproduce runs them in fp16 too. The ANE is much further away than the other fp16 backends. On Qwen fewer than half its values are inside the tolerance, and 5–9% of neurons have a different top token. A planner that moved work to the ANE when MPS is busy would make results depend on the machine's load. That breaks the rule the memory planner keeps (`openinterp/harness/memory.py`): the maths is the same in every mode.

## Other findings

- **MPS fp16 is accurate here, given the changes above.** Its error is 0.3–1.1%. Every sampled neuron has r ≥ 0.9997 with fp32, and 98% of neurons keep their top token. It is at most 1.3× faster than MPS fp32 in these runs. This bears on the "Try bf16/fp16 on MPS" item of the sister-search TODO. We also ran HF's own code (transformers 4.46, SDPA attention) in fp16 on MPS on the same inputs. Its activations were finite, with errors of 0.008 (0.5B) and 0.010 (1.5B). So the overflows above are not a problem for the HF models we run.
- **The forward pass may be a small part of a scan.** Take one pass over a corpus of about 212k tokens, which is about 104 batches. At these speeds, the model's share of it on MPS is about 4 s on GPT-2 and 35 s on Qwen2.5-1.5B. The baseline in TODO.md gives 13–15 minutes for a whole scan. If both figures hold for the current scan, most of the time is outside the forward pass. Copying the whole layer to the CPU and the statistics on it are candidates. That is an estimate, not a profile. The TODO item "Can we do performance tests on memory, compute and wall time?" is where to measure it.

## What would change the answer

Apple's [ml-ane-transformers](https://github.com/apple/ml-ane-transformers) rewrites a transformer for the ANE: channels-first layout, attention split per head, and LayerNorm built from ANE-friendly operations. It reports large speed-ups for BERT-sized models. That is a second implementation of every architecture. It would need its own correctness check per model, and fp16 would still limit its accuracy. Unless a profile shows the forward pass is most of a scan's time, it isn't worth doing.
