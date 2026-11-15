---
name: attention-head-audit
description: Plan and run the extension of openinterp from MLP neuron poles to attention heads (and later transcoder features), using the circuits literature. Use when the user wants to expand the registry's scope beyond MLP neurons, audit an attention head, or compare native neurons against SAE/transcoder baselines.
---

# Extending the registry to attention heads and learned bases

## Papers

- Olsson et al., 2022, *In-context Learning and Induction Heads*, arXiv:2209.11895. The canonical head type: attends to the token after an earlier copy of the current token, then copies it.
- Wang et al., 2022, *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small*, arXiv:2211.00593. Name-mover, S-inhibition and duplicate-token heads, found by path patching.
- Conmy et al., 2023, *Towards Automated Circuit Discovery for Mechanistic Interpretability* (ACDC), arXiv:2304.14997.
- Syed, Rager & Conmy, 2023, arXiv:2310.10348. Edge attribution patching is a fast alternative to ACDC.
- Dunefsky, Chlenski & Nanda, 2024, *Transcoders Find Interpretable LLM Feature Circuits*, arXiv:2406.11944, and Paulo, Shabalin & Belrose, 2025, *Transcoders Beat Sparse Autoencoders for Interpretability*, arXiv:2501.18823. These learned-basis baselines are the natural comparison for the project's native-neuron thesis.

## Harness design for heads

A head's output is its slice of the `o_proj` input (Llama/Qwen) or `c_proj` input (GPT-2 attention), so the existing pre-hook approach carries over:

```python
attn = runner.layers[layer].self_attn  # Qwen2/Llama; GPT-2: .attn
head_dim = runner.model.config.hidden_size // runner.model.config.num_attention_heads


def head_slice(act, head):  # act: [b, seq, n_heads * head_dim]
    return act[..., head * head_dim : (head + 1) * head_dim]
```

How the harness handles heads (protocol 0.6, `harness/units.py` and `harness/heads.py`):

1. **Attention patterns.** Head commands load the model with `attn_implementation="eager"` (`attn_implementation(ident)`), since SDPA and flash attention don't return weights.
2. **Identifiers.** `ComponentIdentifier` is `NeuronIdentifier | HeadIdentifier`, told apart by `kind`. Submissions and evidence take either. A head submission must carry a `HeadProfile`.
3. **Strength.** `|W_O (z_t − z̄)|`, with z̄ the head's mean output over the scan, stored in `head_profile.mean_output`. The raw write norm `|W_O z|` doesn't work, because most heads write a sizeable resting vector. On GPT-2 small L5H1, the median token sits at a quarter of the maximum, so half of all text would count as "active". Against z̄, the five known GPT-2 small induction heads (L5H1, L5H5, L6H9, L7H2, L7H10) are active on every second-half token of repeated random tokens. They are active on almost none of the first half. The previous-token head L4H11 is active nearly everywhere. Ablation is mean ablation (z → z̄), so strength is exactly how far ablating the head moves the residual stream.
4. **Evidence.** The scan records the tokens each top example's peak attends to, and previous-token, duplicate-token and induction scores on random repeated sequences. There is no vocabulary lens or partner search for heads.
5. **Probe semantics.** Hard negatives for an induction head are repeated sequences with the copy broken. Holdouts are repeated random-token sequences (Olsson et al.). Probes are measured on their own, so the earlier copy must be inside the probe.

## Learned-basis baselines

To test the project's premise, compare native neurons with transcoder or SAE features on the same layer, model and corpus. Score both with this protocol: the share VERIFIED among randomly selected units, and balanced accuracy. Learned features go in as another `kind`, never mixed into native-neuron statistics. Done for transcoder latents. `FeatureIdentifier` (`kind = "dictionary_feature"`) names a latent. The presets are in `harness/dictionaries.py` (the first is `gpt2-small-transcoders`, Dunefsky et al.). The scan and patching are in `harness/features.py`. Run `open-interpretability audit --dictionary <name> --random`, with `--window-tokens` no longer than the preset's `context_tokens`. The GPT-2 transcoders fail past position ~200. The home page's learned-basis table compares the arms (`builder/learned_baseline.py`). The site and Parquet keep features apart from native units. The first full comparison, GPT-2 small with 40 latents against 40 positive poles, is in `docs/analyses/transcoder-baseline.md`: 37 against 20 VERIFIED. For a GELU model, sample the native arm with `--pole positive`.

## Suggested order

1. ~~Add the schema generalisation and a head hook to `NeuronHookRunner`, with tests on `tiny_gpt2`.~~ Done: `ComponentIdentifier` (`NeuronIdentifier` | `HeadIdentifier`) in `core/identifiers.py`. `runner.head_outputs`, `head_write_norm`, `head_output_weights`, `attention_patterns`, `hooks(head_interventions=…)` and `ablate_head` in `harness/hooks.py`. Tested on `tiny_gpt2` (`tests/test_heads.py`) and checked on Qwen2.5-0.5B (14 heads × 64, grouped-query attention).
2. ~~The audit pipeline for heads.~~ Done (protocol 0.6): `open-interpretability audit --layer L --head H` (or `--unit head --random`), then `score`, `review`, `reevaluate` and `reproduce` as for neurons. Head pages are at `L<layer>/H<idx>.html`, Parquet columns `kind`, `head_idx` and the pattern scores (`tests/test_head_audit.py`).
3. Audit known induction heads in GPT-2 small as a positive control. The protocol should give them `VERIFIED`. If it doesn't, fix the protocol before going further.
4. Only then open head audits on Qwen and Llama.
