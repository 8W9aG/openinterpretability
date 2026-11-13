# Can the ANE run a backward pass?

Run on 2026-11-12 on an M1 Max (64 GB) with macOS 15, coremltools 9.0 and torch 2.14. The script is `scripts/ane_backward.py`. Raw results are in `ane-backward.jsonl`, one line per layer and neuron.

Core ML has no training mode for the ANE, but a backward pass is a graph of ordinary operations too. Input attribution (`harness/input_attribution.py`) needs the gradient of one neuron with respect to the input embeddings. That is about 5.5% of an experiment's compute. So this prototype writes that gradient out as a forward graph and converts it.

## Method

- **The gradient.** It is of one GPT-2 neuron's activation, summed over 128 random tokens, with respect to the input embeddings, through layers 0 to L. `torch.func.grad` gives it, scaled by 256 so that fp16 doesn't lose it.
- **The graph.** `make_fx` traces the gradient to plain aten operations, with PyTorch's core decompositions. `torch.export` gives the graph, and coremltools converts it to fp16.
- **Three changes** make it convert and give finite values:
  - LayerNorm is written out in plain operations. `native_layer_norm` also returns its mean and inverse standard deviation, and the gradient reads them. coremltools binds only the first output of an operation.
  - The core decompositions rewrite gradient operations, such as `select_backward`, as forward ones that coremltools knows.
  - GELU's `exp` gets an argument of at most 11. In the form that `harness/ane/forward.py` uses, `exp` can overflow. The forward value is then a correct 0, but its gradient is inf/inf.
- **Comparisons.** The gradient on the ANE and on Core ML's GPU, both fp16, against torch fp32 on the CPU. torch fp16 on the CPU is given too. Placement is the share of operations that `MLComputePlan` puts on the ANE.

`torch.export` of `torch.func.grad` directly failed ("Cannot access data pointer of Tensor that doesn't have storage"), and `torch.jit.trace` of the `make_fx` graph failed on an argument type. coremltools 9.0 is tested only up to torch 2.7.

## Results

| Layer | On the ANE | ANE error | ANE ms | Core ML GPU error | CPU fp32 ms | torch fp16 error |
| --- | --- | --- | --- | --- | --- | --- |
| L0 | 98.8% | 0.16% | 3–4 | 0.09% | 30–44 | 0.1% |
| L3 | 99.7% | 0.47–0.49% | 11–13 | 0.26–0.28% | 150–152 | 24–32% |
| L6 | 99.8% | 0.58–0.72% | 20–25 | 0.40–0.43% | 287–313 | 63–80% |
| L7 | no plan | – | – | 0.34–0.35% | 438–526 | 68% |
| L11 | no plan | – | – | 0.35–0.51% | 429–752 | 77–241% |

Each row is two neurons, 100 and 2000. The error is the relative RMS error against fp32. The cosine with fp32 was at least 0.99998 for every gradient that ran.

- **To layer 6, the ANE runs the gradient well.** It is 10–15 times faster than fp32 on the CPU, and at most 0.72% off. torch's own fp16 is 24–80% off from layer 3, so the rewritten graph matters as much as the precision.
- **From layer 7, Core ML can't build the ANE's plan.** Loading fails with error −7, for both neurons. The graph grows by about 160 operations a layer, and this looks like a limit on its size. The same graph runs on Core ML's GPU at 0.35–0.51%.
- **The CPU's times vary.** Other jobs ran on the machine, so the speed-ups are rough.

## What it would take

A backward pass that goes deeper needs chunks, as the forward pass has (`harness/ane/chunks.py`). Each chunk would take the residual stream and the gradient from the chunk after it. It would return the gradient at its own input. That is the chain rule across Core ML models. The forward pass has to run first, chunk by chunk, to give each chunk its input. That is not built. The gradient is also only for GPT-2, on random tokens, at one shape. Qwen's RMSNorm and SwiGLU, and the real attribution's interpolated inputs, are not tried.
