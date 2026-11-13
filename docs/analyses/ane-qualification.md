# Which models may scan on the ANE

Run on 2026-11-12 on an M1 Max (64 GB) with macOS 15. Scans run on Apple's Neural Engine (ANE) only for a model in `openinterp/harness/ane/qualified.py`. A model is listed after `scripts/ane_qualify.py` passes it on a real corpus. Raw results are in `ane-qualification.jsonl`, one line per model and layer, failed runs included.

## Method

The script runs a scan of `corpus.jsonl` (500 documents) for each layer, in two ways on the same batches. One is on the ANE, and the other in fp32 on Apple's GPU. A layer passes if it passes five checks:

- **Top 20.** For every neuron and both poles, the scan's rescoring rule (`harness/scan_backend.py`) gives the exact values of the 20 strongest pieces. That is the top 60, and any within 6% of the 20th, at most 120.
- **Correlation.** Each neuron's ANE activations correlate with fp32 at r ≥ 0.9998 over all tokens. The histogram and partners keep the ANE's values.
- **Worst batch.** No batch has a relative error of more than 5% against fp32. One bad batch can hide in the averages.
- **Placement.** Core ML puts at least 95% of the operations on the ANE (`MLComputePlan`), for the shortest and the longest input shape that the scan used.
- **Speed.** The ANE pass is faster than the fp32 pass on the CPU, the planner's next choice. The CPU time is the mean of 10 batches.

Every new model must also pass the check of each conversion: a relative error of at most 3% against fp32 on a test batch.

## Results

These are from the second run, with models split into chunks (see the next section) and the scale fix below. The first run's results are in the `.jsonl` too.

| Model | Layer | Pieces | Top 20 | Worst r | Worst batch | On the ANE | Speed vs the CPU | Passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-2 (512-token windows) | 5 | 824 | 1.000 | 0.99993 | 0.4% | 98.4% | 2.52× | yes |
| GPT-2 (512-token windows) | 11 | 824 | 1.000 | 0.99993 | 0.8% | 99.2% | 2.84× | yes |
| Qwen2.5-0.5B | 12 | 500 | 1.000 | 0.99992 | 1.6% | 99.6% | 1.71× | yes |
| Qwen2.5-0.5B | 20 | 500 | 1.000 | 0.99989 | 1.3% | 99.8% | 1.95× | yes |
| Qwen2.5-1.5B | 8 | 500 | 1.000 | 0.99986 | 1.4% | 99.5% | 1.85× | yes |
| Qwen2.5-1.5B | 20 | 500 | 1.000 | 0.99984 | 1.7% | 99.6% | 1.19× | yes |
| Qwen2.5-1.5B | 27 | 500 | 0.9997 | 0.97510 | 1.8% | 99.6% | 1.17× | no |

GPT-2 and Qwen2.5-0.5B are qualified. Qwen2.5-1.5B is qualified to layer 20, and its deeper layers scan on torch (`DEEPEST` in `qualified.py`). Layers 21 to 26 were not checked.

- **Layer 27 is past fp16's precision.** Its worst neurons fail on torch's fp16 (on Apple's GPU) by the same amount as on the ANE. Neuron 3987 has r = 0.9751 on the ANE and 0.9746 in torch fp16. 378 neurons are below r = 0.9998 on the ANE, and 342 in torch fp16. They are neurons with large peaks (30 to 100) and errors of about 0.15 at those peaks. So no change to the ANE pass can fix the layer.
- **Whole documents are slow on the ANE.** Qwen's scans read whole documents. A long document needs the 1,024- or 2,048-token shape, which the ANE runs much less well than short ones. So Qwen is only 1.2–2.0× faster than the CPU. GPT-2 was scanned in 512-token windows and is 2.5–2.8× faster. On fixed 256-token batches, the ANE was about 3× faster than the CPU for both models (`device-contention.md`).
- **GPT-2 needs windows.** `corpus.jsonl` has a document of 1,689 GPT-2 tokens, longer than GPT-2's context of 1,024. A whole-document GPT-2 scan of that corpus can't run on torch either.
- **Qwen2.5-1.5B's deep layers failed at first.** In the first run, its layer 20 failed the conversion check. On the test batch, the ANE's error against fp32 was 4.25%. On the same batch, Core ML on the GPU (fp16) had 1.37% and PyTorch fp16 on the CPU 1.28%. The next section finds the cause.
- **A scale on random tokens was too large for real text.** The conversion scales each small projection up by a power of 2 for precision, measured on random tokens. In the second run, one batch of Qwen2.5-1.5B (a 1,604-token document) was 23% off. Its layer-1 `down_proj` output was 4 times larger on that text than on random tokens, and the scale sent the ANE's fp16 sums past fp16's range. Now no projection is scaled past an output of 2,048 on random tokens (`OUTPUT_PEAK` in `forward.py`), and that batch is 0.5% off.

## The cause: the size of one Core ML model

The error did not come from depth. Its source was the size of the whole Core ML model.

- **Layer by layer**, in the pass to layer 20, the ANE's error on the residual stream was already 4.2% after layer 0. For fp16 on the CPU it was 1.5%. It stayed at about three times fp16's error through every later layer.
- **Layer 0 alone**, as its own Core ML model, had 1.3% on the ANE and 1.6% on the GPU. So the same operations lost precision only inside the large model.
- **By the end layer** of a pass, the error of the MLP activations was:

| Pass to | fp16 weights | ANE | Core ML GPU | CPU fp16 |
| --- | --- | --- | --- | --- |
| L8 | 1.22 GB | 1.78% | 2.17% | 1.91% |
| L12 | 1.57 GB | 1.47% | 1.68% | 1.52% |
| L16 | 1.92 GB | 1.39% | 1.64% | 1.50% |
| L20 | 2.27 GB | 4.25% | 1.37% | 1.28% |

- **The same pass to layer 20 as two models**, layers 0–9 and then 10–20, had 1.18% on the ANE. The residual stream went from one to the next in fp16.

So the ANE handles a Core ML model of more than about 2 GB of fp16 weights less precisely, all through. Now a pass is a chain of chunks of at most 1.6 GB each (`openinterp/harness/ane/chunks.py`). The same chunks let a model that streams scan on the ANE: a conversion reads one chunk's weights from the checkpoint at a time.

## Also found

A first compile for the ANE can take minutes for each input shape. So a scan now waits at most 20 minutes (`$OPENINTERP_ANE_PREPARE_MINUTES`) for its models. After that it runs on torch, and the ANE is not used again in that process.
