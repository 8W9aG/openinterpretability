# Which models may scan on the ANE

Run on 2026-11-12 on an M1 Max (64 GB) with macOS 15. Scans run on Apple's Neural Engine (ANE) only for a model in `openinterp/harness/ane/qualified.py`. A model is listed after `scripts/ane_qualify.py` passes it on a real corpus. Raw results are in `ane-qualification.jsonl`, one line per model and layer, failed runs included.

## Method

The script runs a scan of `corpus.jsonl` (500 documents) for each layer, in two ways on the same batches. One is on the ANE, and the other in fp32 on Apple's GPU. A layer passes if it passes four checks:

- **Top 20.** For every neuron and both poles, the scan's rescoring rule (`harness/scan_backend.py`) gives the exact values of the 20 strongest pieces. That is the top 60, and any within 6% of the 20th, at most 120.
- **Correlation.** Each neuron's ANE activations correlate with fp32 at r ≥ 0.9998 over all tokens. The histogram and partners keep the ANE's values.
- **Placement.** Core ML puts at least 95% of the operations on the ANE (`MLComputePlan`).
- **Speed.** The ANE pass is faster than the fp32 pass on the CPU, the planner's next choice. The CPU time is the mean of 10 batches.

Every new model must also pass the check of each conversion: a relative error of at most 3% against fp32 on a test batch.

## Results

| Model | Layer | Pieces | Top 20 | Worst r | On the ANE | Speed vs the CPU | Passed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-2 (512-token windows) | 5 | 824 | 1.000 | 0.99993 | 98.4% | 3.16× | yes |
| GPT-2 (512-token windows) | 11 | 824 | 1.000 | 0.99993 | 99.2% | 2.65× | yes |
| Qwen2.5-0.5B | 12 | 500 | 1.000 | 0.99992 | 99.6% | 1.42× | yes |
| Qwen2.5-0.5B | 20 | 500 | 1.000 | 0.99989 | 99.8% | 1.35× | yes |
| Qwen2.5-1.5B | 8 | 500 | 1.000 | 0.99986 | 99.5% | 1.19× | yes |
| Qwen2.5-1.5B | 20 | – | – | – | – | – | no: conversion error 4.7% |

GPT-2 and Qwen2.5-0.5B are qualified. Qwen2.5-1.5B is not.

- **Whole documents are slow on the ANE.** Qwen's scans read whole documents. A long document needs the 1,024- or 2,048-token shape, which the ANE runs much less well than short ones. So Qwen2.5-0.5B is only 1.35–1.42× faster than the CPU, and its layer 20 is about as fast as a free GPU. GPT-2 was scanned in 512-token windows and is 2.6–3.2× faster than the CPU. On fixed 256-token batches, the ANE was about 3× faster than the CPU for both models (`device-contention.md`).
- **GPT-2 needs windows.** `corpus.jsonl` has a document of 1,689 GPT-2 tokens, longer than GPT-2's context of 1,024. A whole-document GPT-2 scan of that corpus can't run on torch either.
- **Qwen2.5-1.5B's deep layers are worse on the ANE.** Its layer 20 failed the conversion check. On the test batch, the ANE's error against fp32 was 4.25%. On the same batch, Core ML on the GPU (fp16) had 1.37% and PyTorch fp16 on the CPU 1.28%. So the ANE itself adds error that grows with depth in this model. Its last layer, 27, was at 3.09%. Finding that error is a TODO item. Until then, Qwen2.5-1.5B scans on torch.

## Also found

A first compile for the ANE can take minutes for each input shape. So a scan now waits at most 20 minutes (`$OPENINTERP_ANE_PREPARE_MINUTES`) for its models. After that it runs on torch, and the ANE is not used again in that process.
