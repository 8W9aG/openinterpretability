---
name: neuron-family-triage
description: Classify neurons into known families (dead, unigram/token, positional, prediction, suppression, entropy, token-frequency) from weight statistics and cheap activation stats, before auditing. Use when sweeping a layer for audit candidates, when a neuron has a tiny or huge output norm, when ablation changes confidence but not the top token, or when forming a prior hypothesis.
---

# Neuron family triage

## Papers

- Gurnee et al., 2024, *Universal Neurons in GPT2 Language Models*, arXiv:2401.12181. Neurons that recur across random seeds fall into families: unigram, alphabet, position, syntax, semantic, plus functional types such as **prediction**, **suppression** and **partition** neurons, identified by the moments (kurtosis, skew) of `W_U · w_out`.
- Stolfo, Wu, Gurnee, Belinkov, Song, Sachan & Nanda, 2024, *Confidence Regulation Neurons in Language Models*, arXiv:2406.16254 (NeurIPS 2024). **Entropy neurons** have a large output norm but write into the unembedding's (near) null space, adjusting confidence through the final LayerNorm. **Token-frequency neurons** shift logits in proportion to log token frequency.
- Voita, Ferrando & Nalmpantis, 2023, *Neurons in LLMs: Dead, N-gram, Positional*, arXiv:2309.04827. They report many dead neurons in large OPT models, plus token/n-gram detectors and positional neurons.

## Per-layer sweep

```python
import torch


def weight_stats(runner, layer):
    W_U = runner.model.get_output_embeddings().weight.detach().float()  # [vocab, d_model]
    proj = runner.down_proj(layer).weight.detach().float()
    W_out = proj.T if hasattr(runner.down_proj(layer), "in_features") else proj  # [d_mlp, d_model]
    logits = W_out @ W_U.T  # [d_mlp, vocab]
    z = (logits - logits.mean(1, keepdim=True)) / logits.std(1, keepdim=True)
    # Share of each neuron's output norm that lies in the span of W_U's top singular vectors
    U, S, Vh = torch.linalg.svd(W_U, full_matrices=False)
    top = Vh[: int(0.9 * Vh.shape[0])]  # tune the rank cut-off
    in_span = (W_out @ top.T).norm(dim=1) / W_out.norm(dim=1)
    return {
        "out_norm": W_out.norm(dim=1),
        "vocab_kurtosis": (z**4).mean(1),
        "vocab_skew": (z**3).mean(1),
        "unembed_span_share": in_span,
    }
```

| Signal | Likely family | Prior hypothesis to test |
| --- | --- | --- |
| High kurtosis, positive skew | Prediction | Promotes a coherent token set (lookahead) |
| High kurtosis, negative skew | Suppression | Suppresses tokens, often the ones just seen |
| Large `out_norm`, low `unembed_span_share` | Entropy | Changes confidence, not which token wins |
| `W_U · w_out` correlates with log unigram frequency | Token frequency | Pulls the output towards or away from the unigram distribution |
| Never active on a broad corpus | Dead | Skip it, but record it for registry statistics |
| Activation depends on position, not content | Positional | Test with shuffled-content probes at a fixed position |

Get token frequencies by counting token ids over the scan corpus. Find positional neurons by correlating activation with position index across the scan. `openinterp activations --model … --layer … --neuron … --corpus corpus.jsonl --top 0 --format jsonl` gives every scanned token's id, position in its piece and strength, from the layer's cached scan when there is one, so don't write your own scan loop.

## Fit with the protocol

- Triage is a *selection* aid. Neurons picked because their statistics stood out have `provenance.selection = "manual"`, not `random`.
- Entropy and token-frequency neurons would fail a top-1-logit causal metric by design. Protocol 0.5 scores retrospective claims on next-token KL, which does see them; a test of the specific claim should measure output entropy or KL to the unigram distribution. See `activation-patching-practice`.
- Universality (Gurnee et al.) suggests checking whether a verified neuron has a counterpart in a sibling model (e.g. Qwen2.5-0.5B vs 1.5B). That is a natural registry cross-reference to add.
