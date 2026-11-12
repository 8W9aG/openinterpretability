# Can scans run on Apple's Neural Engine?

Run on 2026-11-12 on an M1 Max (64 GB) with macOS 15. It answers the Performance TODO item "Can we run any of this on Apple's Neural Engine (ANE)?". Raw results are in `ane-feasibility.jsonl`, one line per model. The script is `scripts/ane_benchmark.py`.

**Answer: no.** On the scan's own workload, the ANE is slower than MPS on every model we tried. On Qwen2.5 it also gives wrong activations. MPS is what `openinterp` uses now, so there is nothing to gain from it.

## Method

- **Workload.** One scan-sized forward pass: a batch of 8 × 256 random token ids (2,048 tokens, as `BATCH_TOKENS` in `openinterp/harness/scan.py`), up to one layer's MLP activations. The layers are GPT-2 small L5 (width 3,072), Qwen2.5-0.5B L12 (width 4,864) and Qwen2.5-1.5B L8 (width 8,960).
- **Access to the ANE.** PyTorch cannot use the ANE. The only way is Core ML, so the script traces each model and converts it with coremltools 9.0 to fp16, the ANE's only float type. Core ML then picks a device for each operation when given a compute-unit setting. The run tries all four settings.
- **A second copy of the forward pass.** coremltools can't convert the HF modelling code. transformers 5's mask code uses `torch.diff`, which has no converter, and 4.46's code converts shapes to Python ints, which fails. So the script runs its own copy of GPT-2 and Qwen2 on the HF weights. In fp32 it agrees with HF to a relative error of 2–8 × 10⁻⁶.
- **Two fp16 fixes.** Without these, Qwen2.5 overflows in fp16 on every device, MPS included:
  - The residual stream gets to about 6,600 (1.5B) and 1,700 (0.5B). RMSNorm squares it (6,600² ≈ 4 × 10⁷, but fp16's largest value is 65,504). HF's RMSNorm moves to fp32 for this step, which the ANE can't do. The copy divides each row by its largest value first. That is exact.
  - The k projections have large biases (|k| up to 220), so q·k gets to about 10⁵ before the 1/√d scale. The copy scales q first.
- **Agreement.** The script compares each result with the fp32 CPU activations. It reports three figures. The first is the relative RMS error over all values. The second is the worst Pearson correlation over 64 neurons spread across the layer. The third is the overlap of neuron 0's top 20 tokens.
- **Timing.** The median of 5 runs after a warm-up, with no other jobs on the machine. Timings changed by up to 2× between runs (MPS fp16 on Qwen2.5-1.5B: 217 ms, then 392 ms). Read them as rough.

## Results

Milliseconds per 2,048-token batch. *Rel. err* is the relative RMS error against fp32.

| Backend | GPT-2 L5 | rel. err | Qwen 0.5B L12 | rel. err | Qwen 1.5B L8 | rel. err |
| --- | --- | --- | --- | --- | --- | --- |
| PyTorch CPU, fp32 | 408 | – | 1,632 | – | 1,731 | – |
| PyTorch MPS, fp32 | 39 | – | 159 | – | 306 | – |
| PyTorch MPS, fp16 | 34 | 0.003 | 127 | 0.008 | 392 | 0.011 |
| Core ML, CPU only | 129 | 0.017 | 588 | 0.037 | 1,794 | 0.044 |
| Core ML, CPU + GPU | 36 | 0.002 | 118 | 0.008 | 231 | 0.010 |
| Core ML, CPU + ANE | 51 | 0.012 | 190 | **1.00** | 485 | **1.00** |
| Core ML, all units | 85 | 0.012 | 340 | **1.03** | 1,096 | **1.00** |

- **The ANE is slower than MPS.** It takes 1.3× as long on GPT-2, 1.2× on Qwen2.5-0.5B and 1.6× on Qwen2.5-1.5B. This is not because Core ML runs part of the work elsewhere. Its plan puts all but 4 of 181 (GPT-2), 977 (0.5B) and 677 (1.5B) operations on the ANE. The other 4 are the embedding lookup (index checks and a gather).
- **The ANE is wrong on Qwen2.5.** A relative error of 1.0 is the size of the signal. The worst neuron's correlation with fp32 is −0.15 (0.5B) and −0.42 (1.5B), and at most 3 of neuron 0's top 20 tokens match. The same fp16 Core ML model is accurate on the GPU (error 0.008–0.010). So the ANE's own arithmetic fails here, not the conversion. We did not find which operation fails. On GPT-2 the ANE's error is 0.012 (worst neuron r = 0.9993, top 20 all match), which is 7× the GPU's.
- **"All units" is worse than "CPU + ANE".** Core ML splits the work between devices, and the copies between them cost more than they save.
- **The fixed costs are large.** The ANE compiles the model when it loads it: 11 s (GPT-2) and 28–35 s (Qwen). The conversion takes another 6–42 s. Core ML for the ANE also needs fixed input shapes. A scan's batches vary in shape (`plan_batches`), so it would need a model for each shape, or padding to one shape.

## Other findings

- **MPS fp16 is accurate here, given the two fixes above.** Its error is 0.3–1.1%. Every sampled neuron has r ≥ 0.9997 with fp32, and all top-20 tokens match. It is at most 1.25× faster than MPS fp32 in these runs. This bears on the "Try bf16/fp16 on MPS" item of the sister-search TODO. Qwen's RMSNorm and attention logits are where fp16 can overflow. We also ran HF's own code (transformers 4.46, SDPA attention) in fp16 on MPS on the same inputs. Its activations were finite, with errors of 0.008 (0.5B) and 0.010 (1.5B). So these overflows are not a problem for the HF models we run.
- **The forward pass may be a small part of a scan.** Take one pass over a corpus of about 212k tokens, which is about 104 batches. At these speeds, the model's share of it on MPS is about 4 s on GPT-2 and 30 s on Qwen2.5-1.5B. The baseline in TODO.md gives 13–15 minutes for a whole scan. If both figures hold for the current scan, most of the time is outside the forward pass. Copying the whole layer to the CPU and the statistics on it are candidates. That is an estimate, not a profile. The TODO item "Can we do performance tests on memory, compute and wall time?" is where to measure it.

## What would change the answer

Apple's [ml-ane-transformers](https://github.com/apple/ml-ane-transformers) rewrites a transformer for the ANE: channels-first layout, attention split per head, and LayerNorm built from ANE-friendly operations. It reports large speed-ups for BERT-sized models. That is a second implementation of every architecture. It would need its own correctness check per model, and fp16 would still limit it. Unless a profile shows the forward pass is most of a scan's time, it isn't worth doing.
