# Dictionary presets and the Qwen2.5-0.5B transcoders

This explains what the sparse dictionaries in `openinterp/harness/dictionaries.py` are for, what the new presets add, and why Qwen2.5-0.5B needed one trained here. How a latent is audited is in docs/PROTOCOL.md §1–2. The baseline comparison is computed in `openinterp/builder/learned_baseline.py`.

## Why dictionaries

The registry rests on a premise: a model's own MLP neurons can be explained one at a time, about as well as the latents of a sparse dictionary trained on the same layers. Superposition argues against it. If neurons are polysemantic and dictionary latents are not, then a share of neurons VERIFIED means little until it is set against latents audited the same way.

The learned-basis baseline makes that comparison. Random latents of a dictionary and random neuron poles of the same model, on the same layers, corpus and scan window, go through the same scan, proposer, adversarial review and gates. The two shares VERIFIED are then compared. Feature verdicts are never pooled with native units.

Until now, only one comparison could run: GPT-2 small against Dunefsky et al.'s transcoders (`docs/analyses/transcoder-baseline.md`). There, 92% of live latents were verified against 50% of positive neuron poles. That is a single model, with GELU MLPs, read through 128-token windows, so it says little about the SwiGLU models the registry is mostly about.

## What the presets add

| Preset | Model | Dictionary | Site | Fit (variance explained) |
|---|---|---|---|---|
| `gpt2-small-transcoders` | GPT-2 small | Transcoders (Dunefsky et al.) | MLP | 45–90% (layers 0–2), through position 128 only |
| `gemma-scope-2b-transcoders` | Gemma 2 2B | Transcoders (Gemma Scope) | MLP | 32–63% |
| `gemma-scope-2b-mlp-saes` | Gemma 2 2B | MLP-output SAEs (Gemma Scope) | MLP output | 63–70% |
| `gemma-scope-2b-res-saes` | Gemma 2 2B | Residual-stream SAEs (Gemma Scope) | Residual stream | 76–93% |
| `llama-3.2-1b-transcoders` | Llama 3.2 1B | Skip transcoders (circuit-tracer) | MLP | 78% (layer 5) |
| `qwen3-0.6b-transcoders` | Qwen3 0.6B | Transcoders (circuit-tracer) | MLP | 60% (layer 5) |
| `llama-scope-8b-res-saes` | Llama 3.1 8B | Residual-stream SAEs (Llama Scope) | Residual stream | 54–86% |
| `llama-scope-8b-mlp-saes` | Llama 3.1 8B | MLP-output SAEs (Llama Scope) | MLP output | 30–84% |
| `llama-scope-8b-transcoders` | Llama 3.1 8B | Transcoders (Llama Scope) | MLP | 44–87% |
| `qwen2.5-0.5b-transcoders` | Qwen2.5-0.5B | Skip transcoders (trained here) | MLP | 59–93% |

Except where marked, fits are measured through position 1,024 on the scan corpus, leaving out each piece's first token. The comment above each preset gives how it was checked.

The table brings three gains:

- **More model families.** The baseline can now run on Gemma 2, Llama 3 and Qwen as well as GPT-2. Whether latents beat neurons by the same margin on every family is itself a result.
- **More kinds of dictionary.** A transcoder replaces the MLP, so its latents are the most direct comparison with neurons. An MLP-output SAE decomposes what the MLP writes. A residual-stream SAE decomposes the residual stream, so a latent there differs from a neuron in site as well as basis. Comparing all three on Gemma 2 or Llama 3.1 separates how much of any gap comes from the basis and how much from the site.
- **Longer contexts.** GPT-2's transcoders stop fitting past position ~128. Every new preset holds up to position 1,024, so scans of the new presets see whole paragraphs, not fragments.

Each preset's input site was checked by reconstruction, not taken from its documentation. The right site fits; the wrong ones explain less than nothing (Gemma's transcoders score −180 against the wrong output). A preset read at the wrong site would still produce latents to audit, but they would not be the dictionary's latents. The checks also turned up two conventions that are easy to get wrong:

- **Gemma 2's post-MLP norm.** Gemma 2 normalises the MLP's output before adding it to the residual stream, and Gemma Scope's dictionaries reconstruct that normalised output, not the raw MLP output.
- **Llama Scope's threshold.** Llama Scope's JumpReLU threshold has to be read as Neuronpedia reads it. With that reading, two layer-15 latents match Neuronpedia's activations to bf16 precision.

## Why train one for Qwen2.5-0.5B

Qwen2.5-0.5B is the registry's default model. It holds 913 of the 1,174 experiment records, and no public SAE or transcoder covers it. Without a dictionary, the baseline can't be run on the model where nearly all the verdicts are.

The Qwen transcoders bring several advantages over the other presets:

- **It's the model the verdicts are about.** A latent-vs-neuron comparison on Qwen2.5-0.5B sets the registry's own neuron results against a baseline directly, instead of by analogy from GPT-2.
- **CI reproduces it.** Qwen2.5-0.5B is one of the two models CI downloads and re-runs (`ALLOWED_MODELS`). Feature records on it can be reproduced by anyone, while records on Gemma or Llama must be signed by a maintainer.
- **It runs on a laptop.** A 0.5B model and 16k-latent dictionaries fit on a laptop CPU. Gemma 2 2B is slower, and Llama 3.1 8B needs bfloat16 and 16 GB just for its weights.
- **The audit text is held out.** The training data leaves out the documents the scan corpus draws from, so an audit reads text the transcoders never trained on. Public dictionaries were trained on large web corpora that may overlap with any scan.
- **Almost every latent is live.** At most 0.6% of latents per layer were dead at the end of training, and 0.06% at layer 18 on the smoke scan's 200k tokens. In the GPT-2 comparison, 21 of 61 random draws (about a third) hit dead latents. A dead draw can't be audited, which leaves the comparison open to "among the latents that fire". Here that caveat almost vanishes.
- **The weights are skip transcoders.** A dense linear skip path carries the linear part of the MLP, so the sparse latents are left with the nonlinear part. Without the skip path, the fit falls to at most 34%.
- **The pipeline is reproducible.** `scripts/train_transcoders.py`, pinned data and a fixed seed regenerate the weights. The Hub repository is pinned by commit, and its card, per-layer metadata and training log are tracked in `hf_models/`.

## Limits

- **The fit is modest.** These are quick dictionaries: one pass over 13.5M tokens on a laptop. The middle layers fit about 60% of the MLP output, with layer 12 lowest at 0.589. A latent from a dictionary that misses 40% of the variance is a weaker baseline than one from a dictionary that misses 10%. Each scan records the fit next to the latent (`dictionary_profile`), so a verdict can be read against it. A larger run would improve it.
- **The native arm needs 1,024-token windows too.** The transcoders fit contexts up to 1,024 tokens, so feature scans use `--window-tokens 1024`. The comparison pairs only units scanned the same way, while the existing Qwen neuron records read whole documents. A Qwen2.5-0.5B baseline therefore needs a fresh arm of random neuron poles scanned at `--window-tokens 1024`.
- **No verdicts yet.** Every preset's site is checked by reconstruction, and smoke audits of five of them (Gemma's MLP and residual SAEs, Llama 3.2 1B's and Llama Scope's transcoders, and Qwen2.5-0.5B's) scanned a latent end to end. No latent of the new presets has been through review yet. The shares VERIFIED, and whether the GPT-2 gap holds elsewhere, are still to be measured.

## Next steps

1. Run the Qwen2.5-0.5B baseline, about 40 random latents and 40 random neuron poles at `--window-tokens 1024`, as `docs/analyses/transcoder-baseline.md` did for GPT-2.
2. Repeat it on one Gemma 2 or Llama 3.1 dictionary of each kind (transcoder, MLP-output SAE, residual SAE), to separate the effect of the basis from that of the site.
3. Read the cross-reference (`openinterp crossref`, `docs/analyses/dictionary-alignment.jsonl`) against verdicts: whether VERIFIED poles have aligned latents more often than rejected ones, and whether aligned latents' descriptions agree with the poles' explanations. Only the GPT-2 and Qwen2.5-0.5B transcoders cover the registry's models so far; the other presets join as their models do.
