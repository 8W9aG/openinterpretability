# CPU acceleration

Issue #3268 asks whether SIMD or other CPU options speed up CPU runs. It asks for a profile of a CPU scan and a CPU review first, then for each option against float32. Run on 2026-11-14 with `scripts/cpu_acceleration.py`, on two machines. The rows are in `cpu-acceleration.json`. Reported, not gated.

**Answer.** Three options make a forward pass faster and stay within the tolerance of `reproduce`:

- **Fewer threads on a shared machine:** 2.0× faster on the loaded Linux machine. This is the largest gain, and it depends on the load.
- **The fused GELU:** 1.12–1.27× faster.
- **Compiling to C++:** 1.1–1.2× faster with `torch.compile` or AOTInductor.

Half precision is about 30 to 290 times slower on these CPUs, and it fails the tolerance. Nothing changes by default here. The follow-ups are in #3735.

## The machines

| | Mac | Linux |
| --- | --- | --- |
| CPU | Apple M1 Max: 8 performance, 2 efficiency cores | Intel i7-13700KF: 8 performance, 8 efficiency cores, 24 threads |
| SIMD | NEON with fp16. No BF16, I8MM or SME. | AVX2. No AVX-512, AVX-512 BF16 or AMX. |
| torch 2.14.1 | CPU capability `DEFAULT`, BLAS from Accelerate, no oneDNN, 8 threads | CPU capability `AVX2`, BLAS from MKL, oneDNN, 16 threads |
| Load average | 5 to 13 on 10 cores | 42 to 50 on 24 threads, from three experiment loops |

Neither CPU has AMX or AVX-512, so the issue's bfloat16 and int8 options on such CPUs can't be measured here. The Linux machine's timings are about 10 times the Mac's, because other loops shared it. So compare options within one machine, not across them.

## Where the time goes

`perf_bench.py`'s two workloads ran under `cProfile` on the Mac CPU. They use GPT-2 small, as in `perf.md`.

- **The scan (20,000 tokens, 44 s).** The corpus pass took 6 s. The rest is evidence that comes after it:
  - the vocabulary lenses: 20 s, of which the Jacobian lens's 14 extra passes took 12 s
  - the first-position check: 11 s. It includes a 4.7 s `torch.save`, which writes a cache entry once per layer.
  - input attribution: 8 s
  - the WordNet lexicon load: 4.5 s, once per process

  At the default 200,000 tokens, the corpus pass grows about tenfold. `layer_activations` already stops the pass after the target layer, and so skips the logits.
- **The review (`reproduce` of one record, 20 s).** The record checks took 12.9 s, of which forward passes took 12.2 s. The rest is start-up and loading.
- **Inside a forward pass** (Mac, 512 tokens):

  | Operator | Share |
  | --- | --- |
  | Matrix products of the blocks | 40% |
  | The unembedding (only when the logits are needed) | 15% |
  | Attention | 13% |
  | Elementwise `tanh`, `mul`, `add` and `pow`, mostly the GELU's | about 23% |

  GPT-2's `NewGELU` is written in Python, so it runs as five separate kernels. It took 7.3 s of the scan and 2.9 s of the review.

So most of the CPU time is forward passes, in the corpus pass and in the passes that the evidence and the review make. The tokenizer and the top-k heaps don't show in the profile.

## Options against float32

The workload is `runner.layer_activations` on one 2,048-token batch from `corpus.jsonl`, the call that the scan makes per batch. It stops at layer 5, the layer of `perf_bench.py`'s pole, or at layer 11. Each option ran in 5 interleaved rounds of 3 passes, and the table gives the median. *Agrees* means that every MLP activation of the layer is within `reproduce`'s `atol`: 0.001 times the largest float32 activation.

| Option | Mac, L5 | Mac, L11 | Linux, L5 | Linux, L11 | Largest difference, of the max | Agrees |
| --- | --- | --- | --- | --- | --- | --- |
| float32, default threads (8 / 16) | 1.00 | 1.00 | 1.00 | 1.00 | 0 | yes |
| 1 thread | 0.40 | 0.41 | 1.29 | 1.25 | 8.4e-6 | yes |
| 2 threads | 0.60 | 0.61 | 1.08 | 1.05 | 6.2e-6 | yes |
| 4 threads | 0.79 | 0.79 | 2.02 | 1.98 | 0 | yes |
| 8 threads (Linux) | | | 1.82 | 1.71 | 1.5e-5 | yes |
| 10 threads | 0.93 | 0.95 | 1.58 | 1.51 | 1.0e-5 | yes |
| Fused GELU | 1.12 | 1.13 | 1.27 | 1.24 | 0 (Mac), 6.7e-6 (Linux) | yes |
| float16 | 0.003 | 0.003 | 0.04 | 0.03 | 0.4% to 7.7% | no |
| bfloat16 | 0.003 | 0.003 | 0.03 | 0.03 | 3.5% to 26% | no |

The numbers are speedups, the float32 time over the option's time.

- **Threads.** On the Mac with a light load, the default 8 threads, one per performance core, is fastest. The 2 efficiency cores make 10 threads slower. On the Linux machine, the default 16 threads compete with the other loops' threads, and 4 threads run twice as fast. MKL's sums depend on the thread count, so the results move slightly, by up to 1.5e-5 of the max. That is far inside the tolerance.
- **The fused GELU.** It replaces each block's `NewGELU` with `torch.nn.GELU(approximate="tanh")`, the same formula in one vectorised kernel. On the Mac CPU its output is identical bit for bit (`torch.equal` over 4 million values). On AVX2 and on MPS the vectorised `tanh` differs by up to 5e-7 and 1e-6.
- **Half precision.** Neither BLAS has a fast fp16 or bf16 product on these CPUs. Torch falls back to its generic single-threaded loop (`cpublas_gemm_impl`, sampled on the Mac). Half precision also moves activations by more than the tolerance allows. So `--dtype float16` and `bfloat16` aren't options for a CPU run.
- **Not tested: int8 dynamic quantisation.** `quantize_dynamic` converts `nn.Linear`, and GPT-2's MLP uses transformers' `Conv1D`, so it would quantise nothing. Its error would also be larger than float16's.

## Compiling to C++

The scan's hooks stop a pass by raising an exception, and a compiled graph can't do that without a break. So the compilers were timed on the full model without hooks, logits included. That bounds what compiling could save.

| Compiler | Mac | Linux | Build and first call (Mac / Linux) | Largest difference, of the max |
| --- | --- | --- | --- | --- |
| Eager | 1.00 | 1.00 | | |
| `torch.compile`: Inductor emits C++ with OpenMP | 1.21 | 1.14 | 8 s / 20 s | 3.2e-5 |
| AOTInductor: a C++ shared library, compiled ahead of time | 1.23 | 1.11 | 15 s / 26 s | 3.2e-5 |
| TorchScript: traced, frozen and optimised for inference | 0.96 | 0.91 | 6 s / 22 s | 7.8e-5 |

Inductor's gain is mostly the fused elementwise kernels, the same work as the fused GELU. TorchScript keeps the separate kernels, and was slower on both machines. A compiled model takes 8 to 26 seconds to build. At 20% of about 19 s of passes, a 200,000-token scan saves about 4 s on the Mac. So compiling pays only with a model cached across runs.

## For the harness

- **Threads on shared machines.** Loops that share a machine could each cap torch's threads at about the cores divided by the loops. On the Linux machine that would have halved forward-pass time. It needs a rule for how a loop knows the load, so it is a follow-up in #3735, beside the device planner (`openinterp/harness/devices.py`).
- **The fused GELU.** It's a two-line change at load time, for GPT-2 only, because Qwen's MLP already uses a fused SiLU. It would cut a GPT-2 scan's forward passes by about a tenth to a fifth. It changes results only on AVX2 and MPS, by about 1e-8 of a value. Someone must decide whether records may differ by that much. #3735 asks for it.
- **Compiling.** It saves no more than the fused GELU for the scan, and needs a version of the hooks without the exception. Not worth it yet.

## Limits

- **GPT-2 small only.** For larger models, more of the time is matrix products, which Accelerate and MKL already vectorise. The GELU's share falls with it.
- **A loaded Linux machine.** The thread result there is about contention, not about the CPU. On an idle machine, the default thread count may be best, as it is on the Mac.
- **One batch per option.** The end-to-end scan wasn't re-timed with each option. The profile gives the share of forward passes that each gain applies to.
