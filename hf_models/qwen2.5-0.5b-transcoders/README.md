---
license: mit
base_model: Qwen/Qwen2.5-0.5B
datasets:
- NeelNanda/pile-10k
tags:
- sparse-autoencoder
- transcoder
- interpretability
---

# Skip transcoders for Qwen2.5-0.5B

One skip transcoder per layer (all 24) of
[Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), trained for
[openinterp](https://github.com/8W9aG/openinterpretability). Each one maps the MLP's input
to the MLP's output through 16,384 sparse latents plus a dense linear skip path.

## Sites

- **Input:** the MLP's input, after `post_attention_layernorm` with its gain (a forward
  pre-hook on `model.layers[L].mlp`).
- **Output:** the MLP's output (`model.layers[L].mlp` forward output), before it is added to
  the residual stream.

## Files

`layer_<L>.safetensors`, float32, with plain weights. No input or output scaling is left to
apply.

| Tensor | Shape |
|---|---|
| `W_enc` | [896, 16384] |
| `b_enc` | [16384] |
| `threshold` | [16384] |
| `W_dec` | [16384, 896] |
| `b_dec` | [896] |
| `W_skip` | [896, 896], stored [d_out, d_in] |

```python
pre = x @ W_enc + b_enc
acts = pre * (pre > threshold)          # JumpReLU
out = acts @ W_dec + b_dec + x @ W_skip.T
```

`layer_<L>.json` holds each layer's hyperparameters and held-out statistics. `train.jsonl` is
the training log.

## Training

- **Architecture:** top-k (k = 32) during training. At export, the top-k activation is
  replaced by a JumpReLU with one global threshold per layer. The threshold is chosen so that,
  on a training batch, it keeps k·T activations in total. The decoder rows are unit norm.
- **Loss:** MSE plus AuxK (Gao et al. 2024), with k_aux = 256 and a coefficient of 1/32. A
  latent counts as dead once it has gone 500k tokens without firing.
- **Optimiser:** Adam, lr 1e-3, warmup then linear decay. Batches of 4,096 tokens, seed 0.
  Inputs and outputs are scaled to an average norm of √d during training; the scales are
  folded into the exported weights.
- **Data:**
  - [NeelNanda/pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k) at revision
    `127bfedc`.
  - 13.5M tokens, one pass, in 1,024-token pieces. Each piece's first token is skipped.
  - The 518 documents that openinterp's scan corpus draws from are excluded, so its audits run
    on text the transcoders never saw.
  - The last 100 documents are held out for validation.

## Fit

The fit is measured on openinterp's scan corpus (32 documents, 1,024-token windows, first
token excluded). It reports explained variance of the MLP output under the exported JumpReLU,
the mean number of active latents (L0), and the share of latents dead at the end of training.

| Layer | EV | L0 | Dead | Layer | EV | L0 | Dead |
|---|---|---|---|---|---|---|---|
| 0 | 0.925 | 31.8 | 0.00% | 12 | 0.589 | 32.2 | 0.18% |
| 1 | 0.750 | 31.6 | 0.01% | 13 | 0.627 | 34.2 | 0.02% |
| 2 | 0.671 | 30.9 | 0.01% | 14 | 0.612 | 33.5 | 0.01% |
| 3 | 0.689 | 32.2 | 0.03% | 15 | 0.649 | 32.6 | 0.01% |
| 4 | 0.702 | 31.2 | 0.02% | 16 | 0.756 | 31.1 | 0.04% |
| 5 | 0.663 | 26.6 | 0.59% | 17 | 0.701 | 29.0 | 0.00% |
| 6 | 0.683 | 32.4 | 0.07% | 18 | 0.640 | 29.2 | 0.07% |
| 7 | 0.671 | 29.4 | 0.51% | 19 | 0.716 | 31.1 | 0.15% |
| 8 | 0.635 | 31.9 | 0.01% | 20 | 0.723 | 33.7 | 0.13% |
| 9 | 0.622 | 31.5 | 0.12% | 21 | 0.627 | 33.4 | 0.09% |
| 10 | 0.598 | 29.6 | 0.32% | 22 | 0.659 | 31.8 | 0.07% |
| 11 | 0.643 | 32.2 | 0.20% | 23 | 0.799 | 36.9 | 0.15% |

On the 100 held-out documents, explained variance ranges from 0.61 (layer 12) to 0.93
(layer 0). Without the skip path it falls to between −0.65 and 0.34 on the scan corpus.

These are small, quickly trained dictionaries (13.5M tokens on a laptop). They serve as a
learned-basis baseline for audits, not as a state-of-the-art reconstruction.

## Use in openinterp

```
openinterp audit --model Qwen/Qwen2.5-0.5B --dictionary qwen2.5-0.5b-transcoders ...
```

They were trained with `scripts/train_transcoders.py`.
