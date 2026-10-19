---
name: activation-patching-practice
description: Design and interpret causal tests of a neuron pole with activation patching, resample ablation and attribution patching, including metric choice, corruption choice, self-repair and AtP failure modes. Use when computing or questioning sufficiency/necessity scores, when choosing a patching metric, or when using attribution patching to select neurons.
---

# Activation patching in practice

## Papers

- Meng, Bau, Andonian & Belinkov, 2022, *Locating and Editing Factual Associations in GPT* (ROME), arXiv:2202.05262. Causal tracing: corrupt the input, then restore single activations.
- Zhang & Nanda, 2023, *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods*, arXiv:2309.16042. Prefer **logit difference** over probability, and **symmetric token replacement** (minimal pairs) over Gaussian-noise corruption.
- Heimersheim & Nanda, 2024, *How to use and interpret activation patching*, arXiv:2404.15255. Denoising (clean→corrupted) tests sufficiency and noising (corrupted→clean) tests necessity; they answer different questions.
- McGrath et al., 2023, *The Hydra Effect: Emergent Self-repair in Language Model Computations*, arXiv:2307.15771.
- Rushing & Nanda, 2024, *Explorations of Self-Repair in Language Models*, arXiv:2402.15390 (ICML 2024). Self-repair is partial and noisy per prompt; mechanisms include final-LayerNorm rescaling and sparse "anti-erasure" neurons.
- Syed, Rager & Conmy, 2023, *Attribution Patching Outperforms Automated Circuit Discovery*, arXiv:2310.10348.
- Kramár, Lieberum, Shah & Nanda, 2024, *AtP\*: An efficient and scalable method for localizing LLM behaviour to components*, arXiv:2403.00745. Documents attribution-patching failure modes (attention saturation, cancellation) and their fixes.

## openinterp primitives

```python
from openinterp.harness.patcher import ActivationPatcher, logit_diff_metric, token_logit_metric
from openinterp.harness.attribution import attribution_patching, top_attributed_neurons

p = ActivationPatcher(runner)
metric = logit_diff_metric(correct_id, incorrect_id)  # prefer this over a single logit
suff = p.denoise(clean, corrupted, ident, metric=metric, freeze_downstream=True).recovery
nec = p.noise(clean, corrupted, ident, metric=metric, freeze_downstream=True).recovery
```

## Choosing the metric (the current weak point)

Up to protocol 0.4 the causal metrics used the top-1 next-token logit and divided by the whole clean/corrupted gap, so they read ≈ 0 for every neuron. Since 0.5, `causal_metrics` in `agents/reviewer.py` scores claimed tokens (lookahead claims) or next-token KL, and ranks the pole against same-layer neurons whose strength changes as much between the pair (`ActivationPatcher.patch_many`). Ranking against random neurons instead reads ≈ 1, because minimal pairs are built to change the audited neuron. Candidate targets, in order of preference:
1. **Logit difference between the claimed tokens** (for lookahead neurons, the `target_upcoming_tokens` against a plausible alternative).
2. **Projection of the residual stream onto the neuron's output direction at a downstream reader** (for retrospective neurons whose effect is indirect).
3. **Output entropy or KL to the unigram distribution** (entropy and token-frequency neurons; see `neuron-family-triage`).

Gating on a new metric is a protocol change: bump `PROTOCOL_VERSION` and update `docs/PROTOCOL.md`.

## Corruption

- Use **minimal pairs** (the submission's counterfactuals) of equal token length so `positions="all"` works. When lengths differ, `positions="last"` patches only the final position, which under-states effects at earlier positions.
- Avoid Gaussian-noise corruption: it pushes activations off-distribution (Zhang & Nanda).
- For resample ablation, average over several sources (`resample_ablation`) instead of zeroing. Zero is off-distribution for neurons with a non-zero baseline.

## Self-repair

`freeze_downstream=True` pins later *MLP outputs*, but attention heads and the final norm can still compensate (Rushing & Nanda). Report both frozen and unfrozen effects. A large gap is itself evidence that something downstream reads this neuron.

## Attribution patching for selection

`attribution_patching(runner, clean, corrupted, layers, metric=..., pole=...)` gives a first-order effect for every neuron in one backward pass. Use it to *select* candidates (`provenance.selection = "attribution"`), then confirm with exact patching. Following AtP\*, distrust neurons whose attribution sign flips between pole-split and unsplit runs, and spot-check the top 10 with exact patching. `tests/test_patcher.py` verifies that the two agree exactly at the final layer.
